# app/core/ephemeris_adapter.py
# -----------------------------------------------------------------------------
# Research-grade Ephemeris Adapter (Skyfield + optional SPICE)
#
# - Majors (Sun..Pluto via barycenters): Skyfield, local DE421 kernel (no net)
# - Lunar nodes: mean/true geocentric (env default; per-row override supported)
# - Small bodies (Ceres, Pallas, Juno, Vesta, Chiron): optional SPICE geocentric
# - Frames: "ecliptic-of-date" (default) or "ecliptic-j2000"
# - Observer: geocentric or WGS84 topocentric (lat/lon/elev); accepts observer={}
# - Velocities: central difference (deg/day), per-body step map for precision
#
# Uniform return shape:
#   {
#     "results": [
#       {
#         "name": "<as-requested>",
#         "longitude": <deg>,         # canonical
#         "lon": <deg>,               # alias
#         "velocity": <deg/day>?,     # canonical, if computed
#         "speed": <deg/day>?,        # alias
#         "lat": <deg>?,              # ecliptic latitude when available
#         "node_model": "true|mean"?  # on node rows only
#       }, ...
#     ],
#     "meta": {
#       "ok": true|false,
#       "warnings": [ ... ],
#       "kernel": "de421[+...]", "kernels": ["de421.bsp", ...],
#       "frame": "ecliptic-of-date|ecliptic-j2000",
#       "topocentric": true|false,          # resolved (not just requested)
#       "node_model": "true|mean",          # default (env)
#       "smalls_enabled": true|false
#     }
#   }
#
# Critical setup failures inside public APIs are captured and returned as
# meta.ok=false + warnings (to avoid unhandled 500s in routes).
# -----------------------------------------------------------------------------
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import os
import logging
import threading

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Env / constants
# ─────────────────────────────────────────────────────────────────────────────
EPHEMERIS_NAME = "de421"

# DE421 nominal span (approx; generous): 1899-12-31 .. 2053-10-09
DE421_JD_MIN = float(os.getenv("OCP_DE421_JD_MIN", "2414992.5"))
DE421_JD_MAX = float(os.getenv("OCP_DE421_JD_MAX", "2469807.5"))
ENFORCE_JD_RANGE = os.getenv("OCP_ENFORCE_JD_RANGE", "1").lower() in ("1", "true", "yes", "on")

# Velocity step (days): tuned for precision vs stability
_SPEED_STEP_MAP = {
    "Moon": 0.05,     # ±1.2 h
    "Mercury": 0.25,  # ±6 h
    "Venus": 0.33,    # ±8 h
}
_SPEED_STEP_DEFAULT = float(os.getenv("OCP_SPEED_STEP_DEFAULT", "0.5"))  # ±12 h

# Node model & small bodies toggle
_NODE_MODEL = os.getenv("OCP_NODE_MODEL", "true").strip().lower()  # {"true","mean"}
_ENABLE_SMALLS = os.getenv("OCP_ENABLE_SMALL_BODIES", "0").lower() in ("1", "true", "yes", "on")

# Policy: degrade-with-warning vs fail-fast on precision fallbacks
ALLOW_DEGRADED = os.getenv("OCP_ALLOW_DEGRADED", "1").lower() in ("1", "true", "yes", "on")

# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────
class EphemerisError(RuntimeError):
    def __init__(self, stage: str, message: str, **context: Any):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message
        self.context = context

# ─────────────────────────────────────────────────────────────────────────────
# Optional backends
# ─────────────────────────────────────────────────────────────────────────────
def _skyfield_available() -> bool:
    try:
        import skyfield  # noqa: F401
        return True
    except Exception:
        return False

try:
    import spiceypy as sp  # optional for small bodies
    _SPICE_OK = True
except Exception:
    sp = None  # type: ignore
    _SPICE_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────
_TS = None                 # Skyfield timescale
_MAIN = None               # Main DE421 kernel
_EXTRA: List[Any] = []     # Extra SPKs
_KERNEL_PATHS: List[str] = []

_SPICE_READY = False
_SPICE_KERNELS: List[str] = []

_LOCK_KERNEL = threading.Lock()
_LOCK_SPICE  = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Catalogs & canonicalization
# ─────────────────────────────────────────────────────────────────────────────
_PLANET_KEYS: Dict[str, str] = {
    "Sun": "sun",
    "Moon": "moon",
    "Mercury": "mercury",
    "Venus": "venus",
    "Mars": "mars",
    "Jupiter": "jupiter barycenter",
    "Saturn": "saturn barycenter",
    "Uranus": "uranus barycenter",
    "Neptune": "neptune barycenter",
    "Pluto": "pluto barycenter",
}
_MAJOR_CANON = {k.lower(): k for k in _PLANET_KEYS.keys()}

_NODE_ALIAS: Dict[str, Tuple[str, Optional[str]]] = {
    "north node": ("North Node", None),
    "south node": ("South Node", None),
    "true node":  ("North Node", "true"),
    "mean node":  ("North Node", "mean"),
    "true north node": ("North Node", "true"),
    "true south node": ("South Node", "true"),
    "mean north node": ("North Node", "mean"),
    "mean south node": ("South Node", "mean"),
    "rahu": ("North Node", None),
    "ketu": ("South Node", None),
}

_SMALL_CANON = {"ceres": "Ceres", "pallas": "Pallas", "juno": "Juno", "vesta": "Vesta", "chiron": "Chiron"}
_SMALLBODY_HINTS: Dict[str, List[str]] = {
    "Ceres":  ["1 Ceres", "Ceres", "00001 Ceres", "2000001", "1"],
    "Pallas": ["2 Pallas", "Pallas", "00002 Pallas", "2000002", "2"],
    "Juno":   ["3 Juno", "Juno", "00003 Juno", "2000003", "3"],
    "Vesta":  ["4 Vesta", "Vesta", "00004 Vesta", "2000004", "4"],
    "Chiron": ["2060 Chiron", "Chiron", "2002060", "2060", "95P/Chiron"],
}

def _canon_node(nm: str) -> Tuple[str, Optional[str]]:
    low = (nm or "").strip().lower()
    return _NODE_ALIAS.get(low, (nm, None))

def _canon_name(nm: str) -> Tuple[str, str, Optional[str]]:
    """Return (canonical_name, kind, node_model_override) with kind ∈ {'major','node','small','unknown'}."""
    s = (nm or "").strip()
    low = s.lower()
    if low in _MAJOR_CANON:
        return _MAJOR_CANON[low], "major", None
    if low in _SMALL_CANON:
        return _SMALL_CANON[low], "small", None
    if low in _NODE_ALIAS or low in {"north node", "south node"}:
        cn, ovr = _canon_node(low)
        return cn.title(), "node", ovr
    return s, "unknown", None

# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────
def _wrap360(x: float) -> float:
    v = float(x) % 360.0
    return 0.0 if abs(v) < 1e-12 else v

def _atan2deg(y: float, x: float) -> float:
    return _wrap360(math.degrees(math.atan2(y, x)))

def _wrap_diff_deg(a: float, b: float) -> float:
    return ((a - b + 540.0) % 360.0) - 180.0

def _speed_step_for(name: str) -> float:
    return _SPEED_STEP_MAP.get(name, _SPEED_STEP_DEFAULT)

# ─────────────────────────────────────────────────────────────────────────────
# Kernel I/O
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_kernel_path() -> Optional[str]:
    path = os.getenv("OCP_EPHEMERIS")
    if path and os.path.isfile(path):
        return path
    fallback = os.path.join(os.getcwd(), "app", "data", "de421.bsp")
    return fallback if os.path.isfile(fallback) else None

def _extra_spk_paths() -> List[str]:
    files = os.getenv("OCP_EXTRA_SPK_FILES")
    if files:
        out = [p.strip() for p in files.split(",") if p.strip()]
        return [p for p in out if os.path.isfile(p)]
    root = os.getenv("OCP_EXTRA_SPK_DIR", os.path.join(os.getcwd(), "app", "data", "spk"))
    paths: List[str] = []
    try:
        if os.path.isdir(root):
            for fn in sorted(os.listdir(root)):
                if fn.lower().endswith(".bsp"):
                    paths.append(os.path.join(root, fn))
    except Exception as e:
        log.debug("extras scan failed: %s", e)
    return paths

def _looks_like_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) <= 512:
            with open(path, "rb") as f:
                head = f.read(128)
            return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        pass
    return False

def _get_timescale():
    global _TS
    if _TS is not None:
        return _TS
    if not _skyfield_available():
        raise EphemerisError("dependency", "Skyfield not installed")
    from skyfield.api import load
    with _LOCK_KERNEL:
        if _TS is None:
            _TS = load.timescale()
    return _TS

def _load_kernel(path: str):
    from skyfield.api import load
    try:
        return load(path)
    except Exception as e:
        raise EphemerisError("kernel", f"Skyfield failed to load kernel: {path}", error=str(e))

def _get_kernels():
    """Thread-safe lazy load of main and extra kernels (fail-fast)."""
    global _MAIN, _EXTRA, _KERNEL_PATHS, EPHEMERIS_NAME
    if _MAIN is not None:
        return _MAIN, _EXTRA

    if not _skyfield_available():
        raise EphemerisError("dependency", "Skyfield not installed")

    with _LOCK_KERNEL:
        if _MAIN is not None:
            return _MAIN, _EXTRA

        path = _resolve_kernel_path()
        if not path:
            raise EphemerisError("kernel", "No local DE421 found (set OCP_EPHEMERIS or place app/data/de421.bsp)")

        if _looks_like_lfs_pointer(path):
            raise EphemerisError("kernel", f"Kernel looks like a Git LFS pointer: {path}")

        _MAIN = _load_kernel(path)
        _KERNEL_PATHS.append(path)

        _EXTRA = []
        for p in _extra_spk_paths():
            if _looks_like_lfs_pointer(p):
                log.warning("SPICE/Skyfield extra looks like a Git LFS pointer: %s", p)
            try:
                k = _load_kernel(p)
            except EphemerisError as e:
                log.warning("Extra kernel skipped: %s", e)
                continue
            _EXTRA.append(k)
            _KERNEL_PATHS.append(p)

        if _EXTRA:
            EPHEMERIS_NAME = f"de421+{len(_EXTRA)}spk"

    return _MAIN, _EXTRA

def current_kernel_name() -> str:
    if _KERNEL_PATHS:
        return ", ".join(os.path.basename(p) for p in _KERNEL_PATHS)
    return EPHEMERIS_NAME

def load_kernel(kernel_name: str = "de421"):
    k, _ = _get_kernels()
    return k, current_kernel_name()

def _to_tts(jd_tt: float):
    ts = _get_timescale()
    return ts.tt_jd(jd_tt)

# ─────────────────────────────────────────────────────────────────────────────
# SPICE helpers (optional small bodies)
# ─────────────────────────────────────────────────────────────────────────────
def _spice_bootstrap() -> bool:
    global _SPICE_READY, _SPICE_KERNELS
    if _SPICE_READY:
        return True
    if not _SPICE_OK:
        return False
    with _LOCK_SPICE:
        if _SPICE_READY:
            return True
        try:
            try:
                sp.kclear()  # type: ignore
            except Exception:
                pass
            furnished: List[str] = []

            de421_path = _resolve_kernel_path()
            if de421_path and os.path.isfile(de421_path):
                if _looks_like_lfs_pointer(de421_path):
                    raise EphemerisError("spice", f"Kernel looks like a Git LFS pointer: {de421_path}")
                sp.furnsh(de421_path)  # type: ignore
                furnished.append(de421_path)

            for p in _extra_spk_paths():
                if _looks_like_lfs_pointer(p):
                    log.warning("SPICE: %s looks like a Git LFS pointer", p)
                try:
                    sp.furnsh(p)  # type: ignore
                    furnished.append(p)
                except Exception as e:
                    log.warning("SPICE failed to furnish %s: %s", p, e)

            _SPICE_KERNELS = furnished
            _SPICE_READY = True
            return True
        except EphemerisError as e:
            log.warning("%s", e)
            _SPICE_READY = False
            _SPICE_KERNELS = []
            return False
        except Exception as e:
            log.warning("SPICE bootstrap failed: %s", e)
            _SPICE_READY = False
            _SPICE_KERNELS = []
            return False

def _et_from_jd_tt(jd_tt: float) -> float:
    return (float(jd_tt) - 2451545.0) * 86400.0

def _rotate_to_ecliptic_xyz(frame: str, jd_tt: float, x: float, y: float, z: float, warnings: List[str]) -> Tuple[float, float, float]:
    """Rotate equatorial J2000/of-date -> requested ecliptic frame."""
    try:
        import erfa
        if frame and frame.lower() in ("ecliptic-j2000", "j2000", "ecl-j2000"):
            eps = math.radians(23.439291111)  # mean obliquity at J2000
        else:
            d = math.floor(jd_tt); f = jd_tt - d
            eps0 = erfa.obl06(d, f)
            _dpsi, deps = erfa.nut06a(d, f)
            eps = float(eps0 + deps)
        ce, se = math.cos(eps), math.sin(eps)
        return x, y*ce + z*se, -y*se + z*ce
    except Exception:
        # Degraded: IAU 2006 mean obliquity polynomial (no nutation)
        T = (float(jd_tt) - 2451545.0) / 36525.0
        eps_arcsec = 84381.448 - 46.8150*T - 0.00059*(T**2) + 0.001813*(T**3)
        eps = math.radians(eps_arcsec / 3600.0)
        if not ALLOW_DEGRADED:
            raise EphemerisError("precision", "ERFA unavailable; cannot compute ecliptic-of-date precisely")
        warnings.append("precision_degraded: erfa_missing_nutation")
        ce, se = math.cos(eps), math.sin(eps)
        return x, y*ce + z*se, -y*se + z*ce

def _spice_try_id(s: str) -> Optional[int]:
    if not _SPICE_READY:
        return None
    try:
        try:
            return int(s)
        except Exception:
            pass
        return int(sp.bodn2c(s))  # type: ignore[attr-defined]
    except Exception:
        return None

def _spice_id_for_name(name: str) -> Optional[int]:
    if not _SPICE_READY:
        return None
    i = _spice_try_id(name)
    if i is not None:
        return i
    for cand in _SMALLBODY_HINTS.get(name.capitalize(), []):
        i = _spice_try_id(cand)
        if i is not None:
            return i
    fallback = {"ceres": 2000001, "pallas": 2000002, "juno": 2000003, "vesta": 2000004, "chiron": 2002060}
    return fallback.get(name.strip().lower())

def _spice_lon_lat_speed(jd_tt: float, name: str, *, frame: str, warnings: List[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Geocentric lon, lat, speed via SPICE (deg, deg, deg/day)."""
    if not (_SPICE_OK and _ENABLE_SMALLS and _SPICE_READY):
        return None, None, None
    tid = _spice_id_for_name(name)
    if tid is None:
        return None, None, None
    try:
        et = _et_from_jd_tt(jd_tt)
        pos, _lt = sp.spkpos(str(tid), et, "J2000", "NONE", "399")  # type: ignore
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        x2, y2, z2 = _rotate_to_ecliptic_xyz(frame, jd_tt, x, y, z, warnings)
        lon = _atan2deg(y2, x2)
        lat = math.degrees(math.atan2(z2, math.hypot(x2, y2)))
        # speed by central diff
        step = _speed_step_for("Moon")
        et_m = _et_from_jd_tt(jd_tt - step); et_p = _et_from_jd_tt(jd_tt + step)
        pos_m, _ = sp.spkpos(str(tid), et_m, "J2000", "NONE", "399")  # type: ignore
        pos_p, _ = sp.spkpos(str(tid), et_p, "J2000", "NONE", "399")  # type: ignore
        xm, ym, zm = float(pos_m[0]), float(pos_m[1]), float(pos_m[2])
        xp, yp, zp = float(pos_p[0]), float(pos_p[1]), float(pos_p[2])
        xm2, ym2, zm2 = _rotate_to_ecliptic_xyz(frame, jd_tt - step, xm, ym, zm, warnings)
        xp2, yp2, zp2 = _rotate_to_ecliptic_xyz(frame, jd_tt + step, xp, yp, zp, warnings)
        lon_m = _atan2deg(ym2, xm2); lon_p = _atan2deg(yp2, xp2)
        spd = _wrap_diff_deg(lon_p, lon_m) / (2.0 * step)
        return lon, lat, spd
    except Exception as e:
        warnings.append(f"spice_compute_failed:{name}:{e}")
        return None, None, None

def _spice_ecliptic_longitude_speed(jd_tt: float, name: str, *, frame: str, warnings: List[str]) -> Tuple[Optional[float], Optional[float]]:
    """Convenience: (lon, speed) with warning propagation."""
    lon, _lat, spd = _spice_lon_lat_speed(jd_tt, name, frame=frame, warnings=warnings)
    return lon, spd

# ─────────────────────────────────────────────────────────────────────────────
# Skyfield helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_ecliptic_frame(frame: str):
    """Return Skyfield ecliptic frame or raise EphemerisError."""
    if not _skyfield_available():
        raise EphemerisError("dependency", "Skyfield not installed")
    try:
        from skyfield import framelib as _fl  # type: ignore
        if frame and frame.lower() in ("ecliptic-j2000", "j2000", "ecl-j2000"):
            ef = getattr(_fl, "ecliptic_J2000_frame", None)
            if ef is None:
                raise AttributeError("ecliptic_J2000_frame missing")
            return ef
        ef = getattr(_fl, "ecliptic_frame", None)
        if ef is None:
            raise AttributeError("ecliptic_frame missing")
        return ef
    except Exception as e:
        raise EphemerisError("frame", f"Cannot construct ecliptic frame '{frame}'", error=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Ecliptic lon/lat extraction (robust) + vector math + observer resolver
# ─────────────────────────────────────────────────────────────────────────────
from typing import Tuple, Optional

def _frame_latlon(geo, ecliptic_frame) -> Tuple[float, float]:
    """
    Return (lon_deg_mod360, lat_deg) in the requested ecliptic frame.
    Some Skyfield versions hiccup on topocentric .frame_latlon(); we fall back to:
      1) geo.frame_latlon(ecliptic_frame)               (preferred)
      2) geo.frame_xyz(ecliptic_frame)  -> manual lon/lat
      3) geo.ecliptic_latlon()          -> of-date (last resort)
    """
    import math

    # 1) Preferred, fast path
    try:
        lat, lon, _ = geo.frame_latlon(ecliptic_frame)
        lon_deg = float(lon.degrees) % 360.0
        lat_deg = float(lat.degrees)
        # Validate the results are finite
        if math.isfinite(lon_deg) and math.isfinite(lat_deg):
            return lon_deg, lat_deg
    except Exception:
        pass

    # 2) Compute from Cartesian in the requested ecliptic frame
    try:
        xyz = geo.frame_xyz(ecliptic_frame)
        x, y, z = (float(xyz.au[0]), float(xyz.au[1]), float(xyz.au[2]))
        # Guard against a degenerate 0-vector (shouldn't happen, but be safe)
        rho = math.hypot(x, y)
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and (rho > 0.0 or z != 0.0):
            lon = _atan2deg(y, x)
            lat = math.degrees(math.atan2(z, rho)) if rho > 0.0 else (90.0 if z > 0.0 else -90.0)
            return lon % 360.0, float(lat)
    except Exception:
        pass

    # 3) Last resort: Skyfield helper (ecliptic-of-date)
    try:
        elat, elon, _ = geo.ecliptic_latlon()  # may differ slightly from requested frame
        lon_deg = float(elon.degrees) % 360.0
        lat_deg = float(elat.degrees)
        if math.isfinite(lon_deg) and math.isfinite(lat_deg):
            return lon_deg, lat_deg
    except Exception:
        pass

    # If all methods fail, raise an error rather than return invalid data
    raise ValueError(f"Unable to extract ecliptic coordinates from {type(geo)}")


def _cross(a, b):
    ax, ay, az = a; bx, by, bz = b
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bz)


def _observer(
    main,
    *,
    topocentric: bool,
    latitude: Optional[float],
    longitude: Optional[float],
    elevation_m: Optional[float],
    observer: Optional[Dict[str, float]],
    meta_warnings: List[str],
) -> Tuple[Optional[Any], bool]:
    """
    Build the observer. Returns (observer_object, resolved_topocentric_flag).

    Rules:
      • If observer dict is provided, it overrides top-level lat/lon/elev.
      • If topocentric requested but coords missing/invalid → warn & fall back to geocentric.
      • On any WGS84 build failure → warn & fall back to geocentric.
      • Geocentric uses main["earth"].
    """
    if main is None:
        return None, False

    # ---- Pull overrides from `observer` dict (tolerate common aliases)
    if isinstance(observer, dict):
        def _num(x):
            try:
                return float(x)
            except Exception:
                return None

        lat_o = observer.get("lat", observer.get("latitude"))
        lon_o = observer.get("lon", observer.get("lng", observer.get("longitude")))
        elev_o = observer.get("elevation_m", observer.get("elev_m", observer.get("alt_m", observer.get("altitude_m"))))

        if lat_o is not None:
            latitude = _num(lat_o)
        if lon_o is not None:
            longitude = _num(lon_o)
        if elev_o is not None:
            elevation_m = _num(elev_o)

    # ---- Normalize and validate ranges if present
    def _valid_lat(lat: Optional[float]) -> bool:
        return isinstance(lat, (int, float)) and -90.0 <= float(lat) <= 90.0

    def _normalize_lon(lon: Optional[float]) -> Optional[float]:
        if not isinstance(lon, (int, float)):
            return None
        # normalize to [-180, 180)
        x = float(lon)
        x = ((x + 180.0) % 360.0) - 180.0
        # special case for -180 -> +180 wrap consistency
        if x == -180.0:
            x = 180.0
        return x

    if isinstance(longitude, (int, float)):
        longitude = _normalize_lon(float(longitude))

    # ---- Topocentric path
    if topocentric:
        if not (_valid_lat(latitude) and isinstance(longitude, (int, float))):
            meta_warnings.append("topocentric_missing_coords: falling back to geocentric")
            try:
                return main["earth"], False
            except Exception:
                return None, False
        try:
            from skyfield.api import wgs84
            obs = wgs84.latlon(float(latitude), float(longitude), elevation_m=float(elevation_m or 0.0))
            return obs, True
        except Exception as e:
            meta_warnings.append(f"topocentric_build_failed:{e}")
            try:
                return main["earth"], False
            except Exception:
                return None, False

    # ---- Geocentric default
    try:
        return main["earth"], False
    except Exception:
        return None, False
    # ---- Geocentric default
    try:
        return main["earth"], False
    except Exception:
        return None, False
    # Geocentric default
    try:
        return main["earth"]
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Lunar nodes (geocentric)
# ─────────────────────────────────────────────────────────────────────────────
def _mean_node(jd_tt: float) -> float:
    T = (jd_tt - 2451545.0) / 36525.0
    Omega = 125.04452 - 1934.136261 * T + 0.0020708*(T**2) + (T**3)/450000.0
    return Omega % 360.0

@lru_cache(maxsize=8192)
def _true_node_geocentric_cached(jd_q: float) -> float:
    """Approx true node via angular momentum of lunar orbit (geocentric)."""
    try:
        from skyfield.framelib import ecliptic_frame  # type: ignore
    except Exception:
        return _mean_node(jd_q)

    ts = _get_timescale()
    moon = _get_body("Moon")
    main, _ = _get_kernels()
    if moon is None or main is None:
        return _mean_node(jd_q)

    try:
        earth = main["earth"]
    except Exception:
        return _mean_node(jd_q)

    step = _speed_step_for("Moon")
    t0 = ts.tt_jd(jd_q)
    tp = ts.tt_jd(jd_q + step)
    tm = ts.tt_jd(jd_q - step)
    r0 = tuple(map(float, earth.at(t0).observe(moon).apparent().frame_xyz(ecliptic_frame).au))
    rp = tuple(map(float, earth.at(tp).observe(moon).apparent().frame_xyz(ecliptic_frame).au))
    rm = tuple(map(float, earth.at(tm).observe(moon).apparent().frame_xyz(ecliptic_frame).au))
    v = tuple((rp[i] - rm[i]) / (2.0 * step) for i in range(3))
    h = _cross(r0, v)
    n = _cross((0.0, 0.0, 1.0), h)
    norm_xy = math.hypot(n[0], n[1])
    if norm_xy < 1e-12:
        return _mean_node(jd_q)
    return _atan2deg(n[1], n[0])

def _node_longitude(name: str, jd_tt: float, *, model_override: Optional[str] = None) -> Tuple[float, str]:
    model = (model_override or _NODE_MODEL).lower()
    if model == "mean":
        asc = _mean_node(jd_tt)
        return (asc if name == "North Node" else (asc + 180.0) % 360.0), "mean"
    jd_q = round(float(jd_tt), 5)
    asc = _true_node_geocentric_cached(jd_q)
    return (asc if name == "North Node" else (asc + 180.0) % 360.0), "true"

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def _check_jd_guard(jd_tt: float) -> None:
    if ENFORCE_JD_RANGE and not (DE421_JD_MIN <= float(jd_tt) <= DE421_JD_MAX):
        raise EphemerisError("validation", "Julian date outside DE421 nominal span", jd_tt=float(jd_tt))

def _meta(frame: str, topocentric_resolved: bool, warnings: List[str]) -> Dict[str, Any]:
    return {
        "ok": not any(w.startswith("error:") for w in warnings),
        "warnings": warnings,
        "kernel": current_kernel_name(),
        "kernels": [os.path.basename(p) for p in _KERNEL_PATHS] or [EPHEMERIS_NAME],
        "frame": "ecliptic-j2000" if frame.lower() in ("ecliptic-j2000", "j2000", "ecl-j2000") else "ecliptic-of-date",
        "topocentric": bool(topocentric_resolved),
        "node_model": _NODE_MODEL,
        "smalls_enabled": bool(_ENABLE_SMALLS),
    }

def _mk_row(name: str, lon: float, *, lat: Optional[float] = None, vel: Optional[float] = None, node_model: Optional[str] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": name, "longitude": float(lon), "lon": float(lon)}
    if lat is not None:
        row["lat"] = float(lat)
    if vel is not None:
        row["velocity"] = float(vel)
        row["speed"] = float(vel)
    if node_model:
        row["node_model"] = node_model
    return row

def _compute_major_row(*, body, obs, ef, jd_tt: float, name: str, ts, warnings: List[str]) -> Dict[str, Any]:
    geo = obs.at(ts.tt_jd(jd_tt)).observe(body).apparent()
    lon, lat = _frame_latlon(geo, ef)
    # central difference for velocity
    step = _speed_step_for(name)
    t0 = ts.tt_jd(jd_tt - step)
    t1 = ts.tt_jd(jd_tt + step)
    lon0, _ = _frame_latlon(obs.at(t0).observe(body).apparent(), ef)
    lon1, _ = _frame_latlon(obs.at(t1).observe(body).apparent(), ef)
    vel = _wrap_diff_deg(lon1, lon0) / (2.0 * step)
    return _mk_row(name, lon, lat=lat, vel=vel)

def _error_payload(frame: str, warnings: List[str]) -> Dict[str, Any]:
    """Return a safe, structured error response (avoids unhandled 500s)."""
    return {"results": [], "meta": _meta(frame, False, warnings)}

def ecliptic_longitudes(
    jd_tt: float,
    names: Optional[List[str]] = None,
    *,
    frame: str = "ecliptic-of-date",
    topocentric: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
    observer: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute apparent ecliptic longitudes (deg) for requested bodies/points at TT Julian Date.
    """
    warnings: List[str] = []
    try:
        _check_jd_guard(jd_tt)
        if _SPICE_OK and _ENABLE_SMALLS:
            _spice_bootstrap()

        main, _ = _get_kernels()
        ts = _get_timescale()
        ef = _get_ecliptic_frame(frame)
        obs, topo_resolved = _observer(
            main,
            topocentric=topocentric,
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m,
            observer=observer,
            meta_warnings=warnings,
        )
        if main is None or obs is None or ef is None or ts is None:
            raise EphemerisError("setup", "Ephemeris setup failed", main=bool(main), obs=bool(obs), ef=bool(ef), ts=bool(ts))

        wanted = (names[:] if names else list(_PLANET_KEYS.keys()))
        rows: List[Dict[str, Any]] = []

        for raw in wanted:
            canon, kind, node_override = _canon_name(raw)

            if kind == "node":
                lon_val, model_used = _node_longitude(canon, jd_tt, model_override=node_override)
                step = _speed_step_for("Moon")
                lon_m, _ = _node_longitude(canon, jd_tt - step, model_override=node_override)
                lon_p, _ = _node_longitude(canon, jd_tt + step, model_override=node_override)
                vel_val = _wrap_diff_deg(lon_p, lon_m) / (2.0 * step)
                rows.append(_mk_row(raw, lon_val, lat=0.0, vel=vel_val, node_model=model_used))
                continue

            if kind == "major":
                body = _get_body(canon)
                if body is None:
                    warnings.append(f"missing_body:{canon}")
                    continue
                try:
                    rows.append(_compute_major_row(body=body, obs=obs, ef=ef, jd_tt=jd_tt, name=raw, ts=ts, warnings=warnings))
                    continue
                except Exception as e:
                    warnings.append(f"error:compute_major:{canon}:{e}")
                    continue

            if kind == "small" and _SPICE_READY and _ENABLE_SMALLS:
                lon, lat, spd = _spice_lon_lat_speed(jd_tt, canon, frame=frame, warnings=warnings)
                if lon is not None and lat is not None:
                    rows.append(_mk_row(raw, lon, lat=lat, vel=spd))
                    continue

            warnings.append(f"unresolved:{raw}:{kind}")

        return {"results": rows, "meta": _meta(frame, topo_resolved, warnings)}

    except EphemerisError as e:
        warnings.append(f"error:{e.stage}:{e.message}")
        return _error_payload(frame, warnings)

def ecliptic_longitudes_and_velocities(
    jd_tt: float,
    names: List[str],
    *,
    frame: str = "ecliptic-of-date",
    topocentric: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
    observer: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Same as ecliptic_longitudes(), but emphasizes that velocities should be present
    when numerically stable. Nodes include velocity (deg/day) via finite differences.
    """
    warnings: List[str] = []
    try:
        _check_jd_guard(jd_tt)
        if _SPICE_OK and _ENABLE_SMALLS:
            _spice_bootstrap()

        main, _ = _get_kernels()
        ts = _get_timescale()
        ef = _get_ecliptic_frame(frame)
        obs, topo_resolved = _observer(
            main,
            topocentric=topocentric,
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m,
            observer=observer,
            meta_warnings=warnings,
        )
        if main is None or obs is None or ef is None or ts is None:
            raise EphemerisError("setup", "Ephemeris setup failed", main=bool(main), obs=bool(obs), ef=bool(ef), ts=bool(ts))

        rows: List[Dict[str, Any]] = []

        for raw in names:
            canon, kind, node_override = _canon_name(raw)

            if kind == "node":
                lon_val, model_used = _node_longitude(canon, jd_tt, model_override=node_override)
                step = _speed_step_for("Moon")
                lon_m, _ = _node_longitude(canon, jd_tt - step, model_override=node_override)
                lon_p, _ = _node_longitude(canon, jd_tt + step, model_override=node_override)
                vel_val = _wrap_diff_deg(lon_p, lon_m) / (2.0 * step)
                rows.append(_mk_row(raw, lon_val, vel=vel_val, node_model=model_used))
                continue

            if kind == "major":
                body = _get_body(canon)
                if body is None:
                    warnings.append(f"missing_body:{canon}")
                    continue
                try:
                    rows.append(_compute_major_row(body=body, obs=obs, ef=ef, jd_tt=jd_tt, name=raw, ts=ts, warnings=warnings))
                except Exception as e:
                    warnings.append(f"error:compute_major:{canon}:{e}")
                continue

            if kind == "small" and _SPICE_READY and _ENABLE_SMALLS:
                lon, spd = _spice_ecliptic_longitude_speed(jd_tt, canon, frame=frame, warnings=warnings)
                if lon is not None:
                    rows.append(_mk_row(raw, lon, vel=spd))
                    continue

            warnings.append(f"unresolved:{raw}:{kind}")

        return {"results": rows, "meta": _meta(frame, topo_resolved, warnings)}

    except EphemerisError as e:
        warnings.append(f"error:{e.stage}:{e.message}")
        return _error_payload(frame, warnings)

# Optional utility for callers that want {longitudes, velocities} maps.
def rows_to_maps(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    lon_map: Dict[str, float] = {}
    vel_map: Dict[str, float] = {}
    for r in rows:
        nm = r.get("name")
        if not nm:
            continue
        if "longitude" in r:
            lon_map[nm] = float(r["longitude"])
        elif "lon" in r:
            lon_map[nm] = float(r["lon"])
        if "velocity" in r:
            vel_map[nm] = float(r["velocity"])
        elif "speed" in r:
            vel_map[nm] = float(r["speed"])
    return {"longitudes": lon_map, "velocities": vel_map}

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def _kernels_iter() -> Iterable[Any]:
    main, extra = _get_kernels()
    if main:
        yield main
    for k in extra:
        yield k

@lru_cache(maxsize=2048)
def _all_kernel_labels(k) -> List[str]:
    labels: List[str] = []
    for attr in ("names", "aliases", "bodies"):
        try:
            obj = getattr(k, attr, None)
            if isinstance(obj, dict):
                labels.extend(list(obj.keys()))
        except Exception:
            pass
    for lst in _SMALLBODY_HINTS.values():
        labels.extend(lst)
    # uniq
    seen = set(); out: List[str] = []
    for s in labels:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

@lru_cache(maxsize=2048)
def _resolve_small_body(name_norm: str):
    hints = _SMALLBODY_HINTS.get(name_norm, [])
    for label in hints:
        for k in _kernels_iter():
            try:
                return k[label]
            except Exception:
                continue
    needle = name_norm.lower()
    for k in _kernels_iter():
        for lab in _all_kernel_labels(k):
            try:
                if needle in lab.lower():
                    return k[lab]
            except Exception:
                continue
    return None

@lru_cache(maxsize=2048)
def _get_body(name: str):
    if name in _PLANET_KEYS:
        key = _PLANET_KEYS[name]
        for k in _kernels_iter():
            try:
                return k[key]
            except Exception:
                continue
    if name in _SMALL_CANON.values():
        b = _resolve_small_body(name)
        if b is not None:
            return b
    return None

def ephemeris_diagnostics(requested: Optional[List[str]] = None) -> Dict[str, Any]:
    main, extras = _get_kernels()
    kernels = [p for p in _KERNEL_PATHS] or [EPHEMERIS_NAME]

    if _SPICE_OK and _spice_bootstrap():
        kernels = kernels + [f"[spice] {os.path.basename(p)}" for p in _SPICE_KERNELS]

    default_names = list(_PLANET_KEYS.keys()) + ["North Node", "South Node"]
    if _ENABLE_SMALLS:
        default_names += ["Ceres", "Pallas", "Juno", "Vesta", "Chiron"]

    wanted = requested or default_names

    resolved: Dict[str, Dict[str, str]] = {}
    missing: List[str] = []

    for raw in wanted:
        nm, kind, _override = _canon_name(raw)
        if kind == "node":
            resolved[raw] = {"type": "node", "kernel": "computed", "label": nm}
            continue

        if kind == "small" and _SPICE_OK and _spice_bootstrap() and _spice_id_for_name(nm) is not None:
            resolved[raw] = {"type": "body", "kernel": "spice", "label": str(_spice_id_for_name(nm))}
            continue

        b = _get_body(nm)
        if b is not None:
            used_label = None
            used_kernel = "main"
            for k in _kernels_iter():
                candidates = []
                if nm in _PLANET_KEYS:
                    candidates.append(_PLANET_KEYS[nm])
                candidates += _SMALLBODY_HINTS.get(nm, [])
                for lab in candidates:
                    try:
                        if k[lab] is b:
                            used_label = lab
                            used_kernel = "extra" if (k in extras) else "main"
                            break
                    except Exception:
                        pass
                if used_label:
                    break
            resolved[raw] = {"type": "body", "kernel": used_kernel, "label": used_label or "n/a"}
        else:
            missing.append(raw)

    return {
        "kernels": kernels,
        "ephemeris_name": current_kernel_name(),
        "resolved": resolved,
        "missing": missing,
        "node_model": _NODE_MODEL,
        "smalls_enabled": _ENABLE_SMALLS,
        "jd_guard": {"enforced": ENFORCE_JD_RANGE, "min": DE421_JD_MIN, "max": DE421_JD_MAX},
    }

def clear_adapter_caches() -> None:
    try:
        _resolve_small_body.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        _get_body.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        _true_node_geocentric_cached.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Back-compat aliases
# ─────────────────────────────────────────────────────────────────────────────
def get_ecliptic_longitudes(*args, **kwargs):
    return ecliptic_longitudes(*args, **kwargs)

def get_node_longitude(name: str, jd_tt: float) -> float:
    canon, kind, override = _canon_name(name)
    if kind != "node":
        raise ValueError("get_node_longitude expects a node name")
    lon, _model = _node_longitude(canon, jd_tt, model_override=override)
    return lon

__all__ = [
    "_skyfield_available",
    "ecliptic_longitudes",
    "ecliptic_longitudes_and_velocities",
    "rows_to_maps",
    "ephemeris_diagnostics",
    "load_kernel",
    "current_kernel_name",
    "clear_adapter_caches",
    "get_ecliptic_longitudes",
    "get_node_longitude",
    "EphemerisError",
]
