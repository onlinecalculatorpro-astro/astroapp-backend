# app/core/ephemeris_adapter.py
# -----------------------------------------------------------------------------
# Ephemeris adapter (no house-engine coupling).
# - Skyfield + local DE421 for majors (Sun..Pluto, barycenters as needed)
# - Optional SPICE for selected small bodies (geocentric only)
# - Lunar nodes: mean/true (geocentric)
# - Frames: "ecliptic-of-date" (default) or "ecliptic-J2000"
# - Topocentric supported via WGS84 (lat/lon/elevation)
#
# IMPORTANT: ecliptic_longitudes(...) ALWAYS RETURNS A LIST OF ROWS:
#   [{"name": <as requested>, "lon": <deg>, "lat": <deg>, "speed": <deg/day>?}, ...]
# This matches harness expectations and avoids "adapter_return_invalid".
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
from functools import lru_cache
import os
import math
import logging

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config (env)
# ─────────────────────────────────────────────────────────────────────────────
EPHEMERIS_NAME = "de421"

_SPEED_STEP_DEFAULT = float(os.getenv("OCP_SPEED_STEP_DEFAULT", "0.5"))   # ±12h
_SPEED_STEP_FAST    = float(os.getenv("OCP_SPEED_STEP_FAST", "0.05"))     # ±1.2h (Moon)
_NODE_MODEL         = os.getenv("OCP_NODE_MODEL", "true").strip().lower() # {"true","mean"}
_ENABLE_SMALLS      = os.getenv("OCP_ENABLE_SMALL_BODIES", "0").lower() in ("1","true","yes","on")

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
# Runtime singletons / state
# ─────────────────────────────────────────────────────────────────────────────
_TS = None                 # Skyfield timescale
_MAIN = None               # Main DE421 kernel (Skyfield)
_EXTRA: List[Any] = []     # Extra SPKs Skyfield can read (often none)
_KERNEL_PATHS: List[str] = []

_SPICE_READY = False
_SPICE_KERNELS: List[str] = []

# ─────────────────────────────────────────────────────────────────────────────
# Body catalogs & canonicalization
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

_NODE_CANON  = {"north node": "North Node", "south node": "South Node"}
_NODE_NAMES = set(_NODE_CANON.values())

_SMALL_CANON = {"ceres": "Ceres", "pallas": "Pallas", "juno": "Juno", "vesta": "Vesta", "chiron": "Chiron"}
_SMALLBODY_HINTS: Dict[str, List[str]] = {
    "Ceres":  ["1 Ceres", "Ceres", "00001 Ceres", "2000001", "1"],
    "Pallas": ["2 Pallas", "Pallas", "00002 Pallas", "2000002", "2"],
    "Juno":   ["3 Juno", "Juno", "00003 Juno", "2000003", "3"],
    "Vesta":  ["4 Vesta", "Vesta", "00004 Vesta", "2000004", "4"],
    "Chiron": ["2060 Chiron", "Chiron", "2002060", "2060", "95P/Chiron"],
}

def _canon_name(nm: str) -> Tuple[str, str]:
    """Return (canonical_name, kind) where kind in {'major','node','small','unknown'}."""
    s = (nm or "").strip()
    low = s.lower()
    if low in _MAJOR_CANON:
        return _MAJOR_CANON[low], "major"
    if low in _NODE_CANON:
        return _NODE_CANON[low], "node"
    if low in _SMALL_CANON:
        return _SMALL_CANON[low], "small"
    return s, "unknown"

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
    return _SPEED_STEP_FAST if name == "Moon" else _SPEED_STEP_DEFAULT

# ─────────────────────────────────────────────────────────────────────────────
# Kernel I/O (no network)
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

def _get_timescale():
    global _TS
    if _TS is not None:
        return _TS
    if not _skyfield_available():
        return None
    try:
        from skyfield.api import load
        _TS = load.timescale()
        return _TS
    except Exception as e:
        log.warning("Skyfield timescale unavailable: %s", e)
        return None

def _load_kernel(path: str):
    try:
        from skyfield.api import load
        return load(path)
    except Exception as e:
        log.info("Skyfield failed to load %s (ok for small-body SPKs): %s", path, e)
        return None

def _get_kernels():
    global _MAIN, _EXTRA, _KERNEL_PATHS, EPHEMERIS_NAME
    if _MAIN is not None:
        return _MAIN, _EXTRA
    if not _skyfield_available():
        log.warning("Skyfield not installed; disabling Skyfield path")
        return None, []

    path = _resolve_kernel_path()
    if path:
        _MAIN = _load_kernel(path)
        if _MAIN:
            _KERNEL_PATHS.append(path)
    else:
        log.warning("No local DE421 found (OCP_EPHEMERIS/app/data/de421.bsp)")
        _MAIN = None

    _EXTRA = []
    for p in _extra_spk_paths():
        k = _load_kernel(p)
        if k:
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
    if ts is None:
        return None
    try:
        return ts.tt_jd(jd_tt)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SPICE helpers (optional small bodies)
# ─────────────────────────────────────────────────────────────────────────────
def _looks_like_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) <= 512:
            with open(path, "rb") as f:
                head = f.read(128)
            return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        pass
    return False

def _spice_bootstrap() -> bool:
    global _SPICE_READY, _SPICE_KERNELS
    if _SPICE_READY:
        return True
    if not _SPICE_OK:
        return False
    try:
        try:
            sp.kclear()
        except Exception:
            pass
        furnished: List[str] = []

        de421_path = _resolve_kernel_path()
        if de421_path and os.path.isfile(de421_path):
            if _looks_like_lfs_pointer(de421_path):
                log.warning("SPICE: %s looks like a Git LFS pointer", de421_path)
            try:
                sp.furnsh(de421_path)
                furnished.append(de421_path)
            except Exception as e:
                log.warning("SPICE failed to furnish %s: %s", de421_path, e)

        for p in _extra_spk_paths():
            if _looks_like_lfs_pointer(p):
                log.warning("SPICE: %s looks like a Git LFS pointer", p)
            try:
                sp.furnsh(p)
                furnished.append(p)
            except Exception as e:
                log.warning("SPICE failed to furnish %s: %s", p, e)

        _SPICE_KERNELS = furnished
        _SPICE_READY = True
        return True
    except Exception as e:
        log.warning("SPICE bootstrap failed: %s", e)
        _SPICE_READY = False
        _SPICE_KERNELS = []
        return False

def _et_from_jd_tt(jd_tt: float) -> float:
    return (float(jd_tt) - 2451545.0) * 86400.0

def _rotate_to_ecliptic_xyz(frame: str, jd_tt: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
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
        # Simple IAU 2006 polynomial fallback
        T = (float(jd_tt) - 2451545.0) / 36525.0
        eps_arcsec = 84381.448 - 46.8150*T - 0.00059*(T**2) + 0.001813*(T**3)
        eps = math.radians(eps_arcsec / 3600.0)
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

def _spice_lon_lat_speed(jd_tt: float, name: str, *, frame: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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
        x2, y2, z2 = _rotate_to_ecliptic_xyz(frame, jd_tt, x, y, z)
        lon = _atan2deg(y2, x2)
        lat = math.degrees(math.atan2(z2, math.hypot(x2, y2)))
        # speed by central diff
        step = _speed_step_for(name)
        et_m = _et_from_jd_tt(jd_tt - step); et_p = _et_from_jd_tt(jd_tt + step)
        pos_m, _ = sp.spkpos(str(tid), et_m, "J2000", "NONE", "399")  # type: ignore
        pos_p, _ = sp.spkpos(str(tid), et_p, "J2000", "NONE", "399")  # type: ignore
        xm, ym, zm = float(pos_m[0]), float(pos_m[1]), float(pos_m[2])
        xp, yp, zp = float(pos_p[0]), float(pos_p[1]), float(pos_p[2])
        xm2, ym2, zm2 = _rotate_to_ecliptic_xyz(frame, jd_tt - step, xm, ym, zm)
        xp2, yp2, zp2 = _rotate_to_ecliptic_xyz(frame, jd_tt + step, xp, yp, zp)
        lon_m = _atan2deg(ym2, xm2); lon_p = _atan2deg(yp2, xp2)
        spd = _wrap_diff_deg(lon_p, lon_m) / (2.0 * step)
        return lon, lat, spd
    except Exception as e:
        log.debug("SPICE position failed for %s (ID %s): %s", name, tid, e)
        return None, None, None

# ─────────────────────────────────────────────────────────────────────────────
# Skyfield helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_ecliptic_frame(frame: str):
    try:
        from skyfield import framelib as _fl
        if frame and frame.lower() in ("ecliptic-j2000", "j2000", "ecl-j2000"):
            return getattr(_fl, "ecliptic_J2000_frame", None)
        return getattr(_fl, "ecliptic_frame", None)
    except Exception:
        return None

def _frame_latlon(geo, ecliptic_frame) -> Tuple[float, float]:
    lat, lon, _ = geo.frame_latlon(ecliptic_frame)
    return float(lon.degrees) % 360.0, float(lat.degrees)

def _cross(a, b):
    ax, ay, az = a; bx, by, bz = b
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bz)

def _observer(main, *, topocentric: bool, latitude: Optional[float], longitude: Optional[float], elevation_m: Optional[float]):
    if main is None:
        return None
    if topocentric and (latitude is None or longitude is None):
        log.debug("topocentric requested but latitude/longitude missing — using geocentric observer")
    if topocentric and latitude is not None and longitude is not None:
        try:
            from skyfield.api import wgs84
            return wgs84.latlon(float(latitude), float(longitude), elevation_m=float(elevation_m or 0.0))
        except Exception as e:
            log.debug("Topocentric build failed: %s", e)
    try:
        return main["earth"]
    except Exception:
        return None

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
    log.debug("Body not resolved in Skyfield kernels: %s", name)
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
    try:
        from skyfield.framelib import ecliptic_frame
    except Exception:
        return _mean_node(jd_q)

    ts = _get_timescale()
    moon = _get_body("Moon")
    main, _ = _get_kernels()
    if ts is None or moon is None or main is None:
        return _mean_node(jd_q)

    try:
        earth = main["earth"]
    except Exception:
        return _mean_node(jd_q)

    step = _speed_step_for("Moon")
    try:
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
        val = _atan2deg(n[1], n[0])
        return val
    except Exception as e:
        log.debug("True node calc failed at JD %.6f: %s", jd_q, e)
        return _mean_node(jd_q)

def _node_longitude(name: str, jd_tt: float) -> float:
    if _NODE_MODEL == "mean":
        asc = _mean_node(jd_tt)
    else:
        jd_q = round(float(jd_tt), 5)  # cache-friendly quantization
        asc = _true_node_geocentric_cached(jd_q)
    return asc if name == "North Node" else (asc + 180.0) % 360.0

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def ecliptic_longitudes(
    jd_tt: float,
    names: Optional[List[str]] = None,
    *,
    frame: str = "ecliptic-of-date",
    topocentric: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    RETURN TYPE (always): list of rows
      [{"name": <as requested>, "lon": <deg>, "lat": <deg>, "speed": <deg/day>?}, ...]
    """
    # Prepare SPICE (no-op if unavailable); only used for small bodies/fallbacks
    if _SPICE_OK and _ENABLE_SMALLS:
        _spice_bootstrap()

    # Resolve kernels & frames
    main, _ = _get_kernels()
    t = _to_tts(jd_tt) if main is not None else None
    ef = _get_ecliptic_frame(frame) if main is not None else None
    obs = _observer(main, topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m) if main is not None else None
    ts = _get_timescale()

    wanted = names[:] if names else list(_PLANET_KEYS.keys())
    rows: List[Dict[str, float]] = []

    for raw in wanted:
        canon, kind = _canon_name(raw)

        # Lunar nodes (geocentric)
        if kind == "node":
            try:
                lon = _node_longitude(canon, jd_tt)
                # optional speed by finite difference (use same step as Moon)
                step = _speed_step_for("Moon")
                lon_m = _node_longitude(canon, jd_tt - step)
                lon_p = _node_longitude(canon, jd_tt + step)
                spd = _wrap_diff_deg(lon_p, lon_m) / (2.0 * step)
                rows.append({"name": raw, "lon": lon, "lat": 0.0, "speed": spd})
            except Exception:
                # still push a minimal row if anything fails
                rows.append({"name": raw, "lon": 0.0, "lat": 0.0})
            continue

        # Majors via Skyfield
        if kind == "major" and main is not None and t is not None and ef is not None and obs is not None:
            body = _get_body(canon)
            if body is not None:
                try:
                    geo = obs.at(t).observe(body).apparent()
                    lon, lat = _frame_latlon(geo, ef)
                    row = {"name": raw, "lon": lon, "lat": lat}
                    if ts is not None:
                        step = _speed_step_for(canon)
                        t0 = ts.tt_jd(jd_tt - step); t1 = ts.tt_jd(jd_tt + step)
                        lon0, _ = _frame_latlon(obs.at(t0).observe(body).apparent(), ef)
                        lon1, _ = _frame_latlon(obs.at(t1).observe(body).apparent(), ef)
                        row["speed"] = _wrap_diff_deg(lon1, lon0) / (2.0 * step)
                    rows.append(row)
                    continue
                except Exception as e:
                    log.debug("Skyfield failed for %s: %s", canon, e)

        # Small bodies via SPICE (geo)
        if kind == "small" and _SPICE_READY and _ENABLE_SMALLS:
            lon, lat, spd = _spice_lon_lat_speed(jd_tt, canon, frame=frame)
            if lon is not None and lat is not None:
                row = {"name": raw, "lon": lon, "lat": lat}
                if spd is not None:
                    row["speed"] = spd
                rows.append(row)
                continue

        # If nothing worked, omit or push a placeholder (omit is cleaner)
        log.debug("Ephemeris: unresolved body %s (kind=%s)", raw, kind)

    return rows

def ecliptic_longitudes_and_velocities(
    jd_tt: float,
    names: List[str],
    *,
    frame: str = "ecliptic-of-date",
    topocentric: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Return rows (NOT maps) to satisfy the astronomy harness:
      [{"name": <as requested>, "lon": <deg>, "lat": <deg?>, "speed": <deg/day?>}, ...]
    Notes:
      • Nodes: geocentric, longitude only (no speed). We omit 'lat'.
      • Small bodies (if enabled): provide lon (+ speed if available).
    """
    # Prepare SPICE (for small bodies only; harmless if unavailable)
    if _SPICE_OK and _ENABLE_SMALLS:
        _spice_bootstrap()

    rows: List[Dict[str, float]] = []

    main, _ = _get_kernels()
    if main is None:
        return rows

    t  = _to_tts(jd_tt)
    ef = _get_ecliptic_frame(frame)
    obs = _observer(
        main,
        topocentric=topocentric,
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
    )

    if t is None or ef is None or obs is None:
        return rows

    ts = _get_timescale()

    for raw in names:
        canon, kind = _canon_name(raw)

        # Nodes (geocentric only; no speed)
        if kind == "node":
            try:
                lon = _node_longitude(canon, jd_tt)
                rows.append({"name": raw, "lon": float(lon)})
            except Exception as e:
                log.debug("Node computation failed for %s: %s", canon, e)
            continue

        skyfield_success = False
        if kind == "major":
            body = _get_body(canon)
            if body is not None:
                try:
                    geo = obs.at(t).observe(body).apparent()
                    lon, lat = _frame_latlon(geo, ef)
                    row: Dict[str, float] = {"name": raw, "lon": float(lon), "lat": float(lat)}
                    if ts is not None:
                        step = _speed_step_for(canon)
                        t0 = ts.tt_jd(jd_tt - step)
                        t1 = ts.tt_jd(jd_tt + step)
                        lon0, _ = _frame_latlon(obs.at(t0).observe(body).apparent(), ef)
                        lon1, _ = _frame_latlon(obs.at(t1).observe(body).apparent(), ef)
                        row["speed"] = _wrap_diff_deg(lon1, lon0) / (2.0 * step)
                    rows.append(row)
                    skyfield_success = True
                except Exception as e:
                    log.debug("Skyfield failed for %s: %s", canon, e)

        # Optional small-body path (SPICE); lat not computed here
        if (not skyfield_success) and _SPICE_READY and _ENABLE_SMALLS:
            lon, spd = _spice_ecliptic_longitude_speed(jd_tt, canon, frame=frame)
            if lon is not None:
                row: Dict[str, float] = {"name": raw, "lon": float(lon)}
                if spd is not None:
                    row["speed"] = float(spd)
                rows.append(row)

    return rows


# Optional utility for any caller that still wants {longitudes, velocities} maps.
def rows_to_maps(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    lon_map: Dict[str, float] = {}
    vel_map: Dict[str, float] = {}
    for r in rows:
        nm = r.get("name")
        if not nm:
            continue
        if "lon" in r:
            lon_map[nm] = float(r["lon"])
        if "speed" in r:
            vel_map[nm] = float(r["speed"])
    return {"longitudes": lon_map, "velocities": vel_map}

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
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
        nm, kind = _canon_name(raw)
        if kind == "node":
            resolved[raw] = {"type": "node", "kernel": "computed", "label": nm}
            continue

        if kind == "small" and _SPICE_READY and _ENABLE_SMALLS and _spice_id_for_name(nm) is not None:
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

__all__ = [
    "ecliptic_longitudes",
    "ecliptic_longitudes_and_velocities",
    "get_ecliptic_longitudes",
    "get_node_longitude",
    "ephemeris_diagnostics",
    "load_kernel",
    "current_kernel_name",
    "clear_adapter_caches",
]
