# app/core/ephemeris_adapter.py
# -----------------------------------------------------------------------------
# Research-grade Ephemeris Adapter (Skyfield + optional SPICE)
#
# Highlights
# • Deterministic, testable Config + Adapter class (legacy API delegates to it)
# • Precise ecliptic frame conversion (ERFA preferred; guarded degraded fallback)
# • Adaptive, Richardson-extrapolated velocities + independent XY cross-check
# • Robust true-node computation (no precision-losing rounding); explicit policy
# • Clean error taxonomy + structured meta diagnostics (for fuzzers/CI)
# • Optional SPICE for selected small bodies (Ceres, Pallas, Juno, Vesta, Chiron)
# • Thread-safe kernel bootstrap; no silent exception swallowing
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
import logging
import math
import os
import threading

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / environment (bounded; converted into Config defaults)
# ─────────────────────────────────────────────────────────────────────────────
EPHEMERIS_NAME_DEFAULT = "de421"

DE421_JD_MIN = float(os.getenv("OCP_DE421_JD_MIN", "2414992.5"))  # 1899-12-31
DE421_JD_MAX = float(os.getenv("OCP_DE421_JD_MAX", "2469807.5"))  # 2053-10-09
ENFORCE_JD_RANGE = os.getenv("OCP_ENFORCE_JD_RANGE", "1").lower() in ("1", "true", "yes", "on")

_ENABLE_SMALLS_ENV = os.getenv("OCP_ENABLE_SMALL_BODIES", "0").lower() in ("1", "true", "yes", "on")
_ALLOW_DEGRADED_ENV = os.getenv("OCP_ALLOW_DEGRADED", "1").lower() in ("1", "true", "yes", "on")
_NODE_MODEL_ENV = os.getenv("OCP_NODE_MODEL", "true").strip().lower()  # {"true","mean"}
_NODE_ON_FAIL_ENV = os.getenv("OCP_NODE_ON_FAIL", "error").strip().lower()  # {"error","mean"}

# Velocity steps (days) — initial guesses; adaptive refines
_SPEED_STEP_MAP = {
    "Moon": 0.05,     # ±1.2 h
    "Mercury": 0.25,  # ±6 h
    "Venus": 0.33,    # ±8 h
}
_SPEED_STEP_DEFAULT = float(os.getenv("OCP_SPEED_STEP_DEFAULT", "0.5"))  # ±12 h

# Diagnostics tolerances
_SPEED_TOL_ARCSEC_ENV = float(os.getenv("OCP_SPEED_TOL_ARCSEC", "0.05"))  # Richardson vs XY check (abs tol)
_SPEED_MIN_STEP_D_ENV = float(os.getenv("OCP_SPEED_MIN_STEP_D", "0.01"))  # minimum h (days) for differencing

# Node cache tick resolution in seconds (0→nanosecond ticks)
_NODE_CACHE_RES_S_ENV = float(os.getenv("OCP_NODE_CACHE_RES_S", "0.0"))

_ABS_ZERO_TOL_DEG_ENV = float(os.getenv("OCP_ABS_ZERO_TOL_DEG", "1e-13"))

# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────
class EphemerisError(RuntimeError):
    """Categorized error for adapter callers."""
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
    import spiceypy as sp  # optional
    _SPICE_OK = True
except Exception:
    sp = None  # type: ignore
    _SPICE_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Adapter configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    frame: str = "ecliptic-of-date"        # or "ecliptic-j2000"
    allow_degraded: bool = _ALLOW_DEGRADED_ENV

    # Nodes
    node_model: str = _NODE_MODEL_ENV      # "true" | "mean"
    node_on_fail: str = _NODE_ON_FAIL_ENV  # "error" | "mean"
    node_cache_res_s: float = _NODE_CACHE_RES_S_ENV

    # Velocities
    speed_tol_arcsec: float = _SPEED_TOL_ARCSEC_ENV
    speed_min_step_d: float = _SPEED_MIN_STEP_D_ENV

    # Precision / numerical
    abs_zero_tol_deg: float = _ABS_ZERO_TOL_DEG_ENV

    # Smalls/Spice
    enable_smalls: bool = _ENABLE_SMALLS_ENV

    # JD guard
    enforce_jd_range: bool = ENFORCE_JD_RANGE
    jd_min: float = DE421_JD_MIN
    jd_max: float = DE421_JD_MAX

# ─────────────────────────────────────────────────────────────────────────────
# Global singletons (kept for legacy API; the class can be used independently)
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
def _wrap360(x: float, *, abs_zero_tol_deg: float) -> float:
    v = float(x) % 360.0
    # treat near-zero as zero; avoid exact equality comparisons
    return 0.0 if math.isclose(v, 0.0, abs_tol=abs_zero_tol_deg) else v

def _atan2deg(y: float, x: float, *, abs_zero_tol_deg: float) -> float:
    return _wrap360(math.degrees(math.atan2(y, x)), abs_zero_tol_deg=abs_zero_tol_deg)

def _wrap_diff_deg(a: float, b: float) -> float:
    """Shortest signed angular difference (a-b) in degrees."""
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
    """Thread-safe lazy load of main and extra kernels."""
    global _MAIN, _EXTRA, _KERNEL_PATHS
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

    return _MAIN, _EXTRA

def current_kernel_name() -> str:
    """Human-friendly list of loaded kernel filenames (or default label)."""
    if _KERNEL_PATHS:
        return ", ".join(os.path.basename(p) for p in _KERNEL_PATHS)
    return EPHEMERIS_NAME_DEFAULT

def load_kernel(kernel_name: str = "de421"):
    """
    Load the configured Skyfield kernels (main + any extras).

    Returns:
        (kernel_object, kernel_name_string)
    """
    k, _ = _get_kernels()
    return k, current_kernel_name()

def _to_tts(jd_tt: float):
    ts = _get_timescale()
    return ts.tt_jd(jd_tt)

# ─────────────────────────────────────────────────────────────────────────────
# SPICE helpers (optional small bodies)
# ─────────────────────────────────────────────────────────────────────────────
def _spice_bootstrap(warnings: Optional[List[str]] = None) -> bool:
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
                    msg = f"Kernel looks like a Git LFS pointer: {de421_path}"
                    if warnings is not None:
                        warnings.append(f"spice:{msg}")
                    raise EphemerisError("spice", msg)
                sp.furnsh(de421_path)  # type: ignore
                furnished.append(de421_path)

            for p in _extra_spk_paths():
                if _looks_like_lfs_pointer(p):
                    log.warning("SPICE: %s looks like a Git LFS pointer", p)
                try:
                    sp.furnsh(p)  # type: ignore
                    furnished.append(p)
                except Exception as e:
                    log.warning("SPICE furnish failed %s: %s", p, e)

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

def _rotate_to_ecliptic_xyz(frame: str, jd_tt: float, x: float, y: float, z: float,
                            warnings: List[str], *, allow_degraded: bool) -> Tuple[float, float, float]:
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
        # Degraded: mean obliquity only (no nutation)
        T = (float(jd_tt) - 2451545.0) / 36525.0
        eps_arcsec = 84381.448 - 46.8150*T - 0.00059*(T**2) + 0.001813*(T**3)
        eps = math.radians(eps_arcsec / 3600.0)
        if not allow_degraded:
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

def _spice_lon_lat_speed(jd_tt: float, name: str, *, frame: str, warnings: List[str],
                         allow_degraded: bool, abs_zero_tol_deg: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Geocentric lon, lat, speed via SPICE (deg, deg, deg/day)."""
    if not (_SPICE_OK and _SPICE_READY):
        return None, None, None
    tid = _spice_id_for_name(name)
    if tid is None:
        return None, None, None
    try:
        et = _et_from_jd_tt(jd_tt)
        pos, _lt = sp.spkpos(str(tid), et, "J2000", "NONE", "399")  # type: ignore
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        x2, y2, z2 = _rotate_to_ecliptic_xyz(frame, jd_tt, x, y, z, warnings, allow_degraded=allow_degraded)
        lon = _atan2deg(y2, x2, abs_zero_tol_deg=abs_zero_tol_deg)
        lat = math.degrees(math.atan2(z2, math.hypot(x2, y2)))
        # speed by central diff on longitudes
        step = _speed_step_for("Moon")
        et_m = _et_from_jd_tt(jd_tt - step); et_p = _et_from_jd_tt(jd_tt + step)
        pos_m, _ = sp.spkpos(str(tid), et_m, "J2000", "NONE", "399")  # type: ignore
        pos_p, _ = sp.spkpos(str(tid), et_p, "J2000", "NONE", "399")  # type: ignore
        xm, ym, zm = float(pos_m[0]), float(pos_m[1]), float(pos_m[2])
        xp, yp, zp = float(pos_p[0]), float(pos_p[1]), float(pos_p[2])
        xm2, ym2, zm2 = _rotate_to_ecliptic_xyz(frame, jd_tt - step, xm, ym, zm, warnings, allow_degraded=allow_degraded)
        xp2, yp2, zp2 = _rotate_to_ecliptic_xyz(frame, jd_tt + step, xp, yp, zp, warnings, allow_degraded=allow_degraded)
        lon_m = _atan2deg(ym2, xm2, abs_zero_tol_deg=abs_zero_tol_deg)
        lon_p = _atan2deg(yp2, xp2, abs_zero_tol_deg=abs_zero_tol_deg)
        spd = _wrap_diff_deg(lon_p, lon_m) / (2.0 * step)
        return lon, lat, spd
    except Exception as e:
        warnings.append(f"spice_compute_failed:{name}:{type(e).__name__}")
        log.debug("SPICE compute failed %s: %s", name, e)
        return None, None, None

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
# Ecliptic lon/lat extraction (robust) + observer resolver
# ─────────────────────────────────────────────────────────────────────────────
def _frame_latlon(geo, ecliptic_frame, *, abs_zero_tol_deg: float) -> Tuple[float, float]:
    """
    Return (lon_deg_mod360, lat_deg) in the requested ecliptic frame.

    Fallbacks:
      1) geo.frame_latlon(ecliptic_frame)
      2) geo.frame_xyz(ecliptic_frame)  -> manual lon/lat
      3) geo.ecliptic_latlon()          -> of-date (last resort)
    """
    # 1) Preferred
    try:
        lat, lon, _ = geo.frame_latlon(ecliptic_frame)
        lon_deg = float(lon.degrees) % 360.0
        lat_deg = float(lat.degrees)
        if math.isfinite(lon_deg) and math.isfinite(lat_deg):
            return lon_deg, lat_deg
    except Exception:
        pass

    # 2) From Cartesian
    try:
        xyz = geo.frame_xyz(ecliptic_frame)
        x, y, z = (float(xyz.au[0]), float(xyz.au[1]), float(xyz.au[2]))
        rho = math.hypot(x, y)
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and (rho > 0.0 or z != 0.0):
            lon = _atan2deg(y, x, abs_zero_tol_deg=abs_zero_tol_deg)
            lat = math.degrees(math.atan2(z, rho)) if rho > 0.0 else (90.0 if z > 0.0 else -90.0)
            return lon % 360.0, float(lat)
    except Exception:
        pass

    # 3) Last resort (ecliptic-of-date)
    try:
        elat, elon, _ = geo.ecliptic_latlon()
        lon_deg = float(elon.degrees) % 360.0
        lat_deg = float(elat.degrees)
        if math.isfinite(lon_deg) and math.isfinite(lat_deg):
            return lon_deg, lat_deg
    except Exception:
        pass

    raise EphemerisError("compute", f"Unable to extract ecliptic coordinates from {type(geo)}")

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
    """
    if main is None:
        return None, False

    try:
        earth = main["earth"]
    except Exception:
        return None, False

    # Dict overrides
    if isinstance(observer, dict):
        def _num(x):
            try:
                return float(x)
            except Exception:
                return None
        lat_o  = observer.get("lat", observer.get("latitude"))
        lon_o  = observer.get("lon", observer.get("lng", observer.get("longitude")))
        elev_o = observer.get("elevation_m", observer.get("elev_m", observer.get("alt_m", observer.get("altitude_m"))))
        if lat_o  is not None: latitude    = _num(lat_o)
        if lon_o  is not None: longitude   = _num(lon_o)
        if elev_o is not None: elevation_m = _num(elev_o)

    def _valid_lat(lat: Optional[float]) -> bool:
        return isinstance(lat, (int, float)) and -90.0 <= float(lat) <= 90.0

    def _normalize_lon(lon: Optional[float]) -> Optional[float]:
        if not isinstance(lon, (int, float)):
            return None
        x = ((float(lon) + 180.0) % 360.0) - 180.0  # [-180,180)
        return 180.0 if x == -180.0 else x

    if isinstance(longitude, (int, float)):
        longitude = _normalize_lon(float(longitude))

    if topocentric:
        if not (_valid_lat(latitude) and isinstance(longitude, (int, float))):
            meta_warnings.append("topocentric_missing_coords: falling back to geocentric")
            return earth, False
        try:
            from skyfield.api import wgs84
            topo = wgs84.latlon(float(latitude), float(longitude), elevation_m=float(elevation_m or 0.0))
            return earth + topo, True
        except Exception as e:
            meta_warnings.append(f"topocentric_build_failed:{type(e).__name__}")
            return earth, False

    return earth, False

# ─────────────────────────────────────────────────────────────────────────────
# Lunar nodes (geocentric)
# ─────────────────────────────────────────────────────────────────────────────
def _mean_node(jd_tt: float) -> float:
    # Meeus/IAU 2006 style polynomial for mean longitude of ascending node
    T = (jd_tt - 2451545.0) / 36525.0
    Omega = 125.04452 - 1934.136261 * T + 0.0020708*(T**2) + (T**3)/450000.0
    return Omega % 360.0

def _node_tick(jd_tt: float, res_s: float) -> int:
    # Quantize JD to integer “ticks” to avoid float-key precision loss in caching.
    # res_s=0 → 1 ns ticks.
    day_s = 86400.0
    q = max(res_s, 1e-9)
    return int(round(jd_tt * (day_s / q)))

@lru_cache(maxsize=8192)
def _true_node_geocentric_tick(cache_key: Tuple[int, float, float]) -> float:
    """Compute true node (North) longitude in ecliptic-of-date using lunar angular-momentum vector."""
    tick, jd_tt, step = cache_key  # tick is used only to make the key hashable & robust
    try:
        from skyfield.framelib import ecliptic_frame  # type: ignore
    except Exception:
        return float("nan")

    ts = _get_timescale()
    main, _ = _get_kernels()
    try:
        earth = main["earth"]
        moon = main["moon"]
    except Exception:
        return float("nan")

    try:
        t0 = ts.tt_jd(jd_tt)
        tp = ts.tt_jd(jd_tt + step)
        tm = ts.tt_jd(jd_tt - step)
        r0 = tuple(map(float, earth.at(t0).observe(moon).apparent().frame_xyz(ecliptic_frame).au))
        rp = tuple(map(float, earth.at(tp).observe(moon).apparent().frame_xyz(ecliptic_frame).au))
        rm = tuple(map(float, earth.at(tm).observe(moon).apparent().frame_xyz(ecliptic_frame).au))
        v = tuple((rp[i] - rm[i]) / (2.0 * step) for i in range(3))

        def _cross(a, b):
            ax, ay, az = a; bx, by, bz = b
            return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bz)

        h = _cross(r0, v)
        n = _cross((0.0, 0.0, 1.0), h)  # Ecliptic north is Ẑ in ecliptic frame by construction
        nx, ny = n[0], n[1]
        norm_xy = math.hypot(nx, ny)
        if not math.isfinite(norm_xy) or norm_xy < 1e-18:
            return float("nan")
        return (math.degrees(math.atan2(ny, nx)) % 360.0)
    except Exception as e:
        log.debug("true-node compute failed: %s", e)
        return float("nan")

def _node_longitude(
    name: str, jd_tt: float, *, cfg: Config, warnings: List[str], model_override: Optional[str] = None
) -> Tuple[float, str, Optional[str]]:
    """Return (longitude, model_used, fallback_flag)."""
    model = (model_override or cfg.node_model).lower()
    if model == "mean":
        asc = _mean_node(jd_tt)
        lon = asc if name == "North Node" else (asc + 180.0) % 360.0
        return lon, "mean", None

    # true model
    step = _speed_step_for("Moon")
    key = (_node_tick(jd_tt, cfg.node_cache_res_s), float(jd_tt), float(step))
    asc_true = _true_node_geocentric_tick(key)
    if math.isfinite(asc_true):
        lon = asc_true if name == "North Node" else (asc_true + 180.0) % 360.0
        return lon, "true", None

    # failed
    if cfg.node_on_fail == "mean":
        warnings.append("true_node_fallback_to_mean")
        asc = _mean_node(jd_tt)
        lon = asc if name == "North Node" else (asc + 180.0) % 360.0
        return lon, "mean", "fallback"
    raise EphemerisError("node", "true_node_failed")

# ─────────────────────────────────────────────────────────────────────────────
# Velocity helpers (adaptive Richardson + XY cross-check)
# ─────────────────────────────────────────────────────────────────────────────
def _richardson_velocity(
    f_lon: Callable[[float], float],
    t: float,
    h0: float,
    *,
    tol_deg_per_day: float,
    h_min: float,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Central difference on wrapped longitudes with Richardson extrapolation.
    Returns (vel, h_used, debug_info).
    """
    def D(h: float) -> float:
        lp = f_lon(t + h)
        lm = f_lon(t - h)
        return _wrap_diff_deg(lp, lm) / (2.0*h)

    info: Dict[str, Any] = {"stencil": []}
    h = max(h0, h_min)
    d1 = D(h)
    info["stencil"].append((h, d1))
    # refine up to 5 levels
    for _ in range(5):
        h2 = max(h/2.0, h_min)
        d2 = D(h2)
        info["stencil"].append((h2, d2))
        # Richardson (error ~ h^2)
        r2 = d2 + (d2 - d1) / 3.0
        if math.isfinite(r2) and math.isfinite(d2) and abs(r2 - d2) <= tol_deg_per_day:
            return r2, h2, info
        h, d1 = h2, d2
    return d1, h, info  # best we have

def _ang_speed_via_xy(
    xyz_at: Callable[[float], Tuple[float, float]],
    t: float,
    h: float
) -> float:
    """Independent estimator: (x y) central diff, lamdot = (x ydot - y xdot)/(x^2+y^2)."""
    x0, y0 = xyz_at(t - h)
    x1, y1 = xyz_at(t + h)
    xdot = (x1 - x0) / (2.0*h)
    ydot = (y1 - y0) / (2.0*h)
    x, y = xyz_at(t)
    denom = x*x + y*y
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    lamdot_rad = (x*ydot - y*xdot) / denom
    return math.degrees(lamdot_rad)

# ─────────────────────────────────────────────────────────────────────────────
# Adapter class
# ─────────────────────────────────────────────────────────────────────────────
class EphemerisAdapter:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()

    # ---- body resolution -----------------------------------------------------
    @lru_cache(maxsize=2048)
    def _all_kernel_labels(self, k) -> List[str]:
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

    def _kernels_iter(self) -> Iterable[Any]:
        main, extra = _get_kernels()
        if main:
            yield main
        for k in extra:
            yield k

    @lru_cache(maxsize=2048)
    def _resolve_small_body(self, name_norm: str):
        hints = _SMALLBODY_HINTS.get(name_norm, [])
        for label in hints:
            for k in self._kernels_iter():
                try:
                    return k[label]
                except Exception:
                    continue
        needle = name_norm.lower()
        for k in self._kernels_iter():
            for lab in self._all_kernel_labels(k):
                try:
                    if needle in lab.lower():
                        return k[lab]
                except Exception:
                    continue
        return None

    @lru_cache(maxsize=2048)
    def _get_body(self, name: str):
        if name in _PLANET_KEYS:
            key = _PLANET_KEYS[name]
            for k in self._kernels_iter():
                try:
                    return k[key]
                except Exception:
                    continue
        if name in _SMALL_CANON.values():
            b = self._resolve_small_body(name)
            if b is not None:
                return b
        return None

    # ---- meta ---------------------------------------------------------------
    def _meta(self, frame: str, topocentric_resolved: bool, warnings: List[str], diags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = {
            "ok": not any(w.startswith("error:") for w in warnings),
            "warnings": warnings,
            "kernel": current_kernel_name(),
            "kernels": [os.path.basename(p) for p in _KERNEL_PATHS] or [EPHEMERIS_NAME_DEFAULT],
            "frame": "ecliptic-j2000" if frame.lower() in ("ecliptic-j2000", "j2000", "ecl-j2000") else "ecliptic-of-date",
            "topocentric": bool(topocentric_resolved),
            "node_model": self.cfg.node_model,
            "smalls_enabled": bool(self.cfg.enable_smalls and _SPICE_OK),
        }
        if diags:
            base["diagnostics"] = diags
        return base

    # ---- rows ---------------------------------------------------------------
    @staticmethod
    def _mk_row(name: str, lon: float, *, lat: Optional[float] = None,
                vel: Optional[float] = None, node_model: Optional[str] = None,
                node_fallback: Optional[str] = None) -> Dict[str, Any]:
        row: Dict[str, Any] = {"name": name, "longitude": float(lon), "lon": float(lon)}
        if lat is not None:
            row["lat"] = float(lat)
        if vel is not None and math.isfinite(vel):
            row["velocity"] = float(vel)
            row["speed"] = float(vel)
        if node_model:
            row["node_model"] = node_model
        if node_fallback:
            row["node_fallback"] = node_fallback
        return row

    # ---- validation ---------------------------------------------------------
    def _check_jd_guard(self, jd_tt: float) -> None:
        if self.cfg.enforce_jd_range and not (self.cfg.jd_min <= float(jd_tt) <= self.cfg.jd_max):
            raise EphemerisError("validation", "Julian date outside DE421 nominal span", jd_tt=float(jd_tt))

    # ---- public computations ------------------------------------------------
    def _compute_major_row(self, *, body, obs, ef, jd_tt: float, name: str, ts, warnings: List[str], diags: Dict[str, Any]) -> Dict[str, Any]:
        # lon/lat now
        geo_now = obs.at(ts.tt_jd(jd_tt)).observe(body).apparent()
        lon_now, lat_now = _frame_latlon(geo_now, ef, abs_zero_tol_deg=self.cfg.abs_zero_tol_deg)

        # longitude sampling function for Richardson
        def _lon_at(tjd: float) -> float:
            geo = obs.at(ts.tt_jd(tjd)).observe(body).apparent()
            lon, _ = _frame_latlon(geo, ef, abs_zero_tol_deg=self.cfg.abs_zero_tol_deg)
            return lon

        # XY estimator (independent)
        def _xy_at(tjd: float) -> Tuple[float, float]:
            xyz = obs.at(ts.tt_jd(tjd)).observe(body).apparent().frame_xyz(ef)
            return float(xyz.au[0]), float(xyz.au[1])

        h0 = _speed_step_for(name)
        vel_R, h_used, Rinfo = _richardson_velocity(
            _lon_at, jd_tt, h0,
            tol_deg_per_day=self.cfg.speed_tol_arcsec / 3600.0,
            h_min=self.cfg.speed_min_step_d
        )
        vel_XY = _ang_speed_via_xy(_xy_at, jd_tt, max(h_used, self.cfg.speed_min_step_d))

        diags.setdefault("velocity_method", "richardson+xy")
        diags["velocity_step_days"] = h_used
        diags["velocity_richardson"] = vel_R
        if math.isfinite(vel_XY):
            diags["velocity_xy"] = vel_XY
            absdiff = abs(vel_R - vel_XY)
            diags["velocity_abs_diff_deg_per_day"] = absdiff
            if absdiff > (self.cfg.speed_tol_arcsec / 3600.0):
                warnings.append(f"velocity_disagree:{name}:|R-XY|={absdiff:.6f}°/d>tol")
                diags["velocity_check"] = "mismatch"
                vel = vel_XY  # prefer XY near wraps
            else:
                diags["velocity_check"] = "ok"
                vel = vel_R
        else:
            diags["velocity_check"] = "xy_nan"
            vel = vel_R

        return self._mk_row(name, lon_now, lat=lat_now, vel=vel)

    def ecliptic_longitudes(
        self,
        jd_tt: float,
        names: Optional[List[str]] = None,
        *,
        frame: Optional[str] = None,
        topocentric: bool = False,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        elevation_m: Optional[float] = None,
        observer: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        diags: Dict[str, Any] = {}
        used_frame = (frame or self.cfg.frame)

        try:
            self._check_jd_guard(jd_tt)

            if self.cfg.enable_smalls and _SPICE_OK:
                _spice_bootstrap(warnings)

            main, _ = _get_kernels()
            ts = _get_timescale()
            ef = _get_ecliptic_frame(used_frame)

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
                    # Node longitude + velocity via finite difference of node function
                    lon_val, model_used, fb = _node_longitude(canon, jd_tt, cfg=self.cfg, warnings=warnings, model_override=node_override)
                    step = _speed_step_for("Moon")
                    lon_m, _m, _ = _node_longitude(canon, jd_tt - step, cfg=self.cfg, warnings=warnings, model_override=node_override)
                    lon_p, _p, _ = _node_longitude(canon, jd_tt + step, cfg=self.cfg, warnings=warnings, model_override=node_override)
                    vel_val = _wrap_diff_deg(lon_p, lon_m) / (2.0 * step)
                    rows.append(self._mk_row(raw, lon_val, lat=0.0, vel=vel_val, node_model=model_used, node_fallback=fb))
                    continue

                if kind == "major":
                    body = self._get_body(canon)
                    if body is None:
                        warnings.append(f"missing_body:{canon}")
                        continue
                    try:
                        rows.append(self._compute_major_row(body=body, obs=obs, ef=ef, jd_tt=jd_tt, name=raw, ts=ts, warnings=warnings, diags=diags))
                        continue
                    except EphemerisError as e:
                        warnings.append(f"error:compute_major:{canon}:{e.stage}")
                        continue
                    except Exception as e:
                        warnings.append(f"error:compute_major:{canon}:{type(e).__name__}")
                        log.debug("compute_major failed %s: %s", canon, e)
                        continue

                if kind == "small" and self.cfg.enable_smalls and _SPICE_READY:
                    lon, lat, spd = _spice_lon_lat_speed(jd_tt, canon, frame=used_frame, warnings=warnings,
                                                         allow_degraded=self.cfg.allow_degraded, abs_zero_tol_deg=self.cfg.abs_zero_tol_deg)
                    if lon is not None and lat is not None:
                        rows.append(self._mk_row(raw, lon, lat=lat, vel=spd))
                        continue

                warnings.append(f"unresolved:{raw}:{kind}")

            return {"results": rows, "meta": self._meta(used_frame, topo_resolved, warnings, diags)}

        except EphemerisError as e:
            warnings.append(f"error:{e.stage}:{e.message}")
            return {"results": [], "meta": self._meta(used_frame or self.cfg.frame, False, warnings, diags)}

    def ecliptic_longitudes_and_velocities(
        self,
        jd_tt: float,
        names: List[str],
        *,
        frame: Optional[str] = None,
        topocentric: bool = False,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        elevation_m: Optional[float] = None,
        observer: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        # identical to ecliptic_longitudes, but emphasizes velocity presence
        return self.ecliptic_longitudes(
            jd_tt,
            names,
            frame=frame,
            topocentric=topocentric,
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m,
            observer=observer,
        )

    def ephemeris_diagnostics(self, requested: Optional[List[str]] = None) -> Dict[str, Any]:
        main, extras = _get_kernels()
        kernels = [p for p in _KERNEL_PATHS] or [EPHEMERIS_NAME_DEFAULT]

        if self.cfg.enable_smalls and _SPICE_OK and _spice_bootstrap():
            kernels = kernels + [f"[spice] {os.path.basename(p)}" for p in _SPICE_KERNELS]

        default_names = list(_PLANET_KEYS.keys()) + ["North Node", "South Node"]
        if self.cfg.enable_smalls:
            default_names += ["Ceres", "Pallas", "Juno", "Vesta", "Chiron"]

        wanted = requested or default_names

        resolved: Dict[str, Dict[str, str]] = {}
        missing: List[str] = []

        for raw in wanted:
            nm, kind, _override = _canon_name(raw)
            if kind == "node":
                resolved[raw] = {"type": "node", "kernel": "computed", "label": nm}
                continue

            if kind == "small" and _SPICE_OK and _spice_bootstrap():
                tid = _spice_id_for_name(nm) if _SPICE_READY else None
                if tid is not None:
                    resolved[raw] = {"type": "body", "kernel": "spice", "label": str(tid)}
                    continue

            b = self._get_body(nm)
            if b is not None:
                used_label = None
                used_kernel = "main"
                for k in self._kernels_iter():
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

        # cache stats for node true-value
        try:
            info = _true_node_geocentric_tick.cache_info()  # type: ignore[attr-defined]
            node_cache = {"hits": info.hits, "misses": info.misses, "maxsize": info.maxsize, "currsize": info.currsize}
        except Exception:
            node_cache = {}

        return {
            "kernels": kernels,
            "ephemeris_name": current_kernel_name(),
            "resolved": resolved,
            "missing": missing,
            "node_model": self.cfg.node_model,
            "node_on_fail": self.cfg.node_on_fail,
            "node_cache": node_cache,
            "smalls_enabled": self.cfg.enable_smalls and _SPICE_OK,
            "jd_guard": {"enforced": self.cfg.enforce_jd_range, "min": self.cfg.jd_min, "max": self.cfg.jd_max},
        }

    def clear_caches(self) -> None:
        try:
            self._resolve_small_body.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self._get_body.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            _true_node_geocentric_tick.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Public module-level API (legacy; delegates to a process-global adapter)
# ─────────────────────────────────────────────────────────────────────────────
_default_adapter: Optional[EphemerisAdapter] = None

def _config_from_env() -> Config:
    return Config(
        frame=os.getenv("OCP_DEFAULT_FRAME", "ecliptic-of-date"),
        allow_degraded=_ALLOW_DEGRADED_ENV,
        node_model=_NODE_MODEL_ENV,
        node_on_fail=_NODE_ON_FAIL_ENV,
        node_cache_res_s=_NODE_CACHE_RES_S_ENV,
        speed_tol_arcsec=_SPEED_TOL_ARCSEC_ENV,
        speed_min_step_d=_SPEED_MIN_STEP_D_ENV,
        abs_zero_tol_deg=_ABS_ZERO_TOL_DEG_ENV,
        enable_smalls=_ENABLE_SMALLS_ENV,
        enforce_jd_range=ENFORCE_JD_RANGE,
        jd_min=DE421_JD_MIN,
        jd_max=DE421_JD_MAX,
    )

def _get_default_adapter() -> EphemerisAdapter:
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = EphemerisAdapter(_config_from_env())
    return _default_adapter

def ecliptic_longitudes(*args, **kwargs):
    return _get_default_adapter().ecliptic_longitudes(*args, **kwargs)

def ecliptic_longitudes_and_velocities(*args, **kwargs):
    return _get_default_adapter().ecliptic_longitudes_and_velocities(*args, **kwargs)

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

def ephemeris_diagnostics(*args, **kwargs) -> Dict[str, Any]:
    return _get_default_adapter().ephemeris_diagnostics(*args, **kwargs)

def clear_adapter_caches() -> None:
    _get_default_adapter().clear_caches()

# Back-compat aliases
def get_ecliptic_longitudes(*args, **kwargs):
    return ecliptic_longitudes(*args, **kwargs)

def get_node_longitude(name: str, jd_tt: float) -> float:
    canon, kind, override = _canon_name(name)
    if kind != "node":
        raise ValueError("get_node_longitude expects a node name")
    warnings: List[str] = []
    cfg = _get_default_adapter().cfg
    lon, _model, _fb = _node_longitude(canon, jd_tt, cfg=cfg, warnings=warnings, model_override=override)
    return lon

__all__ = [
    "Config",
    "EphemerisAdapter",
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
