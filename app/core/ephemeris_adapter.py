# app/core/ephemeris_adapter.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Iterable
import os
import math
import logging
from functools import lru_cache

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Human label for the primary kernel
# ──────────────────────────────────────────────────────────────────────────────
EPHEMERIS_NAME = "de421"

# ──────────────────────────────────────────────────────────────────────────────
# Optional backends
# ──────────────────────────────────────────────────────────────────────────────
def _skyfield_available() -> bool:
    try:
        import skyfield  # noqa: F401
        return True
    except Exception:
        return False

try:
    import spiceypy as sp  # optional, used for small bodies
    _SPICE_OK = True
except Exception:
    sp = None  # type: ignore
    _SPICE_OK = False

# ──────────────────────────────────────────────────────────────────────────────
# Lazy singletons / state
# ──────────────────────────────────────────────────────────────────────────────
_TS = None                 # Skyfield timescale
_MAIN = None               # Main ephemeris (DE421 by default) for Skyfield
_EXTRA: List[Any] = []     # Extra SPKs (if Skyfield can load any)
_KERNEL_PATHS: List[str] = []  # For reporting/diagnostics

_SPICE_READY = False
_SPICE_KERNELS: List[str] = []  # all kernels furnished to SPICE (de421 + extras)

# ──────────────────────────────────────────────────────────────────────────────
# Canonical bodies (DE421 uses barycenters for outer planets)
# ──────────────────────────────────────────────────────────────────────────────
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

# Small bodies we want to support out-of-the-box (SPICE path)
_SMALLBODY_HINTS: Dict[str, List[str]] = {
    # include NAIF-style numeric IDs and common labels
    "Ceres":  ["1 Ceres", "Ceres", "00001 Ceres", "2000001", "1"],
    "Pallas": ["2 Pallas", "Pallas", "00002 Pallas", "2000002", "2"],
    "Juno":   ["3 Juno", "Juno", "00003 Juno", "2000003", "3"],
    "Vesta":  ["4 Vesta", "Vesta", "00004 Vesta", "2000004", "4"],
    "Chiron": ["2060 Chiron", "Chiron", "20002060", "2060", "95P/Chiron"],
}
_NODE_NAMES = {"North Node", "South Node"}

# ──────────────────────────────────────────────────────────────────────────────
# Tunables (via env)
# ──────────────────────────────────────────────────────────────────────────────
_SPEED_STEP_DEFAULT = float(os.getenv("OCP_SPEED_STEP_DEFAULT", "0.5"))   # ±12h
_SPEED_STEP_FAST    = float(os.getenv("OCP_SPEED_STEP_FAST", "0.05"))     # ±1.2h (Moon)
_NODE_MODEL         = os.getenv("OCP_NODE_MODEL", "true").strip().lower()  # "true" | "mean"
_NODE_ALERT_DEG     = float(os.getenv("OCP_NODE_ALERT_DEG", "10.0"))
_NODE_CACHE_DECIMALS = int(os.getenv("OCP_NODE_CACHE_DECIMALS", "5"))

# Kernel resolution rules (NO network):
# OCP_EPHEMERIS (absolute path) else app/data/de421.bsp
# Extras: OCP_EXTRA_SPK_FILES (comma list) or OCP_EXTRA_SPK_DIR (default app/data/spk)

# ──────────────────────────────────────────────────────────────────────────────
# Kernel resolution & loading (Skyfield)
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_kernel_path() -> Optional[str]:
    """
    Resolve local DE421 only. No network.
    Priority:
      1) OCP_EPHEMERIS (absolute path)
      2) app/data/de421.bsp
    """
    p = os.getenv("OCP_EPHEMERIS")
    if p and os.path.isfile(p):
        return p
    p2 = os.path.join(os.getcwd(), "app", "data", "de421.bsp")
    if os.path.isfile(p2):
        return p2
    return None

def _extra_spk_paths() -> List[str]:
    files = os.getenv("OCP_EXTRA_SPK_FILES")
    if files:
        out = [f.strip() for f in files.split(",") if f.strip()]
        return [p for p in out if os.path.isfile(p)]
    root = os.getenv("OCP_EXTRA_SPK_DIR", os.path.join(os.getcwd(), "app", "data", "spk"))
    out: List[str] = []
    try:
        if os.path.isdir(root):
            for fn in sorted(os.listdir(root)):
                if fn.lower().endswith(".bsp"):
                    out.append(os.path.join(root, fn))
    except Exception as e:
        log.debug("Scanning extras failed: %s", e)
    return out

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
        log.warning("Timescale unavailable: %s", e)
        return None

def _load_kernel(path: str):
    try:
        from skyfield.api import load
        return load(path)
    except Exception as e:
        # Skyfield can't read many SB SPKs (Type 13/21) — that's OK, SPICE will.
        log.info("Skyfield failed to load kernel %s (ok for small-bodies): %s", path, e)
        return None

def _get_kernels():
    """
    Returns (main_kernel, [extra_kernels]) for Skyfield use.
    Small-body SPKs may fail to load here; SPICE will handle them.
    No network fetches are attempted.
    """
    global _MAIN, _EXTRA, _KERNEL_PATHS, EPHEMERIS_NAME
    if _MAIN is not None:
        return _MAIN, _EXTRA
    if not _skyfield_available():
        log.warning("Skyfield not installed; ephemeris unavailable for Skyfield path.")
        return None, []

    # Main kernel (no remote fallback)
    path = _resolve_kernel_path()
    if path:
        _MAIN = _load_kernel(path)
        if _MAIN:
            _KERNEL_PATHS.append(path)
    else:
        log.warning("No local DE421 found (OCP_EPHEMERIS/app/data/de421.bsp). Skyfield path disabled.")
        _MAIN = None

    # Extra SPKs (often unsupported by Skyfield; harmless if None)
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

# ──────────────────────────────────────────────────────────────────────────────
# SPICE bootstrap (for small-bodies)
# ──────────────────────────────────────────────────────────────────────────────
def _spice_bootstrap() -> bool:
    """Furnish de421 + all .bsp in app/data/spk (or env overrides)."""
    global _SPICE_READY, _SPICE_KERNELS
    if _SPICE_READY:
        return True
    if not _SPICE_OK:
        return False
    try:
        # Clear any previous set and furnish afresh
        try:
            sp.kclear()
        except Exception:
            pass

        furnished: List[str] = []
        # Furnish DE421 if present
        de421_path = _resolve_kernel_path()
        if de421_path and os.path.isfile(de421_path):
            sp.furnsh(de421_path)
            furnished.append(de421_path)

        # Furnish extras (these are the small-body SPKs that Skyfield cannot read)
        for p in _extra_spk_paths():
            try:
                sp.furnsh(p)
                furnished.append(p)
            except Exception as e:
                log.warning("SPICE failed to furnish %s: %s", p, e)

        if not furnished:
            log.info("SPICE furnished no kernels (none found).")
        _SPICE_KERNELS = furnished
        _SPICE_READY = True
        return True
    except Exception as e:
        log.warning("SPICE bootstrap failed: %s", e)
        _SPICE_READY = False
        _SPICE_KERNELS = []
        return False

def _et_from_jd_tt(jd_tt: float) -> float:
    # SPICE ET seconds from J2000 (using TT~TDB is acceptable for our usage)
    return (float(jd_tt) - 2451545.0) * 86400.0

def _rotate_to_ecliptic_xyz(frame: str, jd_tt: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Equatorial J2000 -> ecliptic ({of-date}|J2000)"""
    try:
        import erfa
        if frame and frame.lower() in ("ecliptic-j2000", "j2000", "ecl-j2000"):
            # Mean obliquity at J2000
            eps = 23.439291111  # degrees
            eps = math.radians(eps)
        else:
            d = math.floor(jd_tt); f = jd_tt - d
            eps0 = erfa.obl06(d, f)
            _dpsi, deps = erfa.nut06a(d, f)
            eps = float(eps0 + deps)
        ce, se = math.cos(eps), math.sin(eps)
        x2 = x
        y2 =  y * ce + z * se
        z2 = -y * se + z * ce
        return x2, y2, z2
    except Exception:
        # Very close fallback (mean obliquity polynomial)
        T = (float(jd_tt) - 2451545.0) / 36525.0
        eps_arcsec = 84381.448 - 46.8150*T - 0.00059*(T**2) + 0.001813*(T**3)
        eps = math.radians(eps_arcsec / 3600.0)
        ce, se = math.cos(eps), math.sin(eps)
        x2 = x
        y2 =  y * ce + z * se
        z2 = -y * se + z * ce
        return x2, y2, z2

def _atan2deg(y, x) -> float:
    return (math.degrees(math.atan2(y, x)) % 360.0)

def _wrap_diff_deg(a: float, b: float) -> float:
    return ((a - b + 540.0) % 360.0) - 180.0

def _speed_step_for(name: str) -> float:
    return _SPEED_STEP_FAST if name == "Moon" else _SPEED_STEP_DEFAULT

def _spice_try_id(s: str) -> Optional[int]:
    """Try to convert 'Ceres' or '2000001' into a NAIF id using SPICE tables."""
    if not _SPICE_READY:
        return None
    try:
        # If it's an int string, accept directly
        try:
            return int(s)
        except Exception:
            pass
        # Resolve via SPICE name -> id
        return int(sp.bodn2c(s))  # type: ignore[attr-defined]
    except Exception:
        return None

def _spice_id_for_name(name: str) -> Optional[int]:
    # Try SPICE name tables then our hints (including numeric forms)
    if not _SPICE_READY:
        return None
    # 1) direct
    i = _spice_try_id(name)
    if i is not None:
        return i
    # 2) try hints
    for cand in _SMALLBODY_HINTS.get(name.capitalize(), []):
        i = _spice_try_id(cand)
        if i is not None:
            return i
    # 3) last-ditch hard maps for the five we care about
    fallback = {
        "ceres": 2000001, "pallas": 2000002, "juno": 2000003, "vesta": 2000004,
        "chiron": 20002060,
    }
    return fallback.get(name.strip().lower())

def _spice_ecliptic_longitude(
    jd_tt: float, name: str, *, frame: str = "ecliptic-of-date"
) -> Optional[float]:
    """
    Geocentric ecliptic longitude via SPICE (J2000 state rotated to ecliptic).
    Note: For small-bodies we ignore topocentric (geocentric is fine).
    """
    if not _SPICE_READY:
        return None
    tid = _spice_id_for_name(name)
    if tid is None:
        return None
    try:
        et = _et_from_jd_tt(jd_tt)
        # Earth center id = 399; J2000 inertial frame
        # Use spkpos to avoid need for leapsecond kernels (we provide ET directly).
        pos, _lt = sp.spkpos(str(tid), et, "J2000", "NONE", "399")  # type: ignore
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        x2, y2, z2 = _rotate_to_ecliptic_xyz(frame, jd_tt, x, y, z)
        return _atan2deg(y2, x2)
    except Exception as e:
        log.debug("SPICE position failed for %s (ID %s): %s", name, tid, e)
        return None

def _spice_ecliptic_longitude_speed(
    jd_tt: float, name: str, *, frame: str
) -> Tuple[Optional[float], Optional[float]]:
    lon0 = _spice_ecliptic_longitude(jd_tt, name, frame=frame)
    if lon0 is None:
        return None, None
    step = _speed_step_for(name)
    lon_m = _spice_ecliptic_longitude(jd_tt - step, name, frame=frame)
    lon_p = _spice_ecliptic_longitude(jd_tt + step, name, frame=frame)
    spd = None
    if lon_m is not None and lon_p is not None:
        spd = _wrap_diff_deg(lon_p, lon_m) / (2.0 * step)
    return lon0, spd

# ──────────────────────────────────────────────────────────────────────────────
# Skyfield math helpers
# ──────────────────────────────────────────────────────────────────────────────
def _frame_latlon(geo, ecliptic_frame) -> Tuple[float, float]:
    lat, lon, _ = geo.frame_latlon(ecliptic_frame)
    return float(lon.degrees) % 360.0, float(lat.degrees)

def _cross(a, b):
    ax, ay, az = a; bx, by, bz = b
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bz)

# ──────────────────────────────────────────────────────────────────────────────
# Lunar nodes (geocentric)
# ──────────────────────────────────────────────────────────────────────────────
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
        mean_val = _mean_node(jd_q)
        diff = abs(_wrap_diff_deg(val, mean_val))
        if diff > _NODE_ALERT_DEG:
            log.warning("True vs Mean node differ %.2f° at JD %.6f (threshold=%.1f°)",
                        diff, jd_q, _NODE_ALERT_DEG)
        return val
    except Exception as e:
        log.debug("True node calc failed at JD %.6f: %s", jd_q, e)
        return _mean_node(jd_q)

def _node_longitude(name: str, jd_tt: float) -> float:
    if _NODE_MODEL == "mean":
        asc = _mean_node(jd_tt)
    else:
        jd_q = round(float(jd_tt), _NODE_CACHE_DECIMALS)
        asc = _true_node_geocentric_cached(jd_q)
    return asc if name == "North Node" else (asc + 180.0) % 360.0

# ──────────────────────────────────────────────────────────────────────────────
# Observer (Skyfield path)
# ──────────────────────────────────────────────────────────────────────────────
def _get_ecliptic_frame(frame: str):
    try:
        from skyfield import framelib as _fl
        if frame and frame.lower() in ("ecliptic-j2000", "j2000", "ecl-j2000"):
            f = getattr(_fl, "ecliptic_J2000_frame", None)
            if f is not None:
                return f
        return getattr(_fl, "ecliptic_frame", None)
    except Exception:
        return None

def _observer(main, *, topocentric: bool,
              latitude: Optional[float],
              longitude: Optional[float],
              elevation_m: Optional[float]):
    if main is None:
        return None
    if topocentric and latitude is not None and longitude is not None:
        try:
            from skyfield.api import wgs84
            return wgs84.latlon(float(latitude), float(longitude),
                                elevation_m=float(elevation_m or 0.0))
        except Exception as e:
            log.debug("Topocentric build failed: %s", e)
    try:
        return main["earth"]
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Body resolution (Skyfield)
# ──────────────────────────────────────────────────────────────────────────────
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
    # include our small-body hints for a loose match pass
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
    if name in _SMALLBODY_HINTS or name.lower() in ("ceres", "pallas", "juno", "vesta", "chiron"):
        b = _resolve_small_body(name.capitalize())
        if b is not None:
            return b
    log.debug("Body not resolved in Skyfield kernels: %s", name)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Public APIs (SIGNATURES MUST NOT CHANGE)
# ──────────────────────────────────────────────────────────────────────────────
def ecliptic_longitudes(
    jd_tt: float,
    names: Optional[List[str]] = None,
    *,
    frame: str = "ecliptic-of-date",
    topocentric: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
) -> Any:
    """
    If `names` is provided: returns {name: longitude_deg} only for those names.
    If `names` is None: returns a legacy list for the 10 planets:
        [{ "name": <str>, "lon": <deg>, "lat": <deg>, "speed": <deg/day> }, ...]
    Nodes are geocentric by definition.

    Strategy:
      • Use Skyfield for anything it can resolve (incl. topocentric) — primarily majors.
      • If longitude is not actually computed via Skyfield (including small bodies or any failure),
        fall through to SPICE (geocentric). DO NOT “continue” on Skyfield exceptions.
    """
    # Ensure SPICE is ready if available (no-op if already done)
    if _SPICE_OK:
        _spice_bootstrap()

    main, _ = _get_kernels()
    t = _to_tts(jd_tt) if main is not None else None
    ef = _get_ecliptic_frame(frame) if main is not None else None
    obs = _observer(main, topocentric=topocentric, latitude=latitude,
                    longitude=longitude, elevation_m=elevation_m) if main is not None else None

    # Map mode
    if names:
        out: Dict[str, float] = {}
        for nm in names:
            # Nodes (geocentric)
            if nm in _NODE_NAMES:
                try:
                    out[nm] = _node_longitude(nm, jd_tt)
                except Exception:
                    pass
                continue

            skyfield_success = False

            # Skyfield for classical bodies only (and only if we can produce a longitude)
            if nm in _PLANET_KEYS and main is not None and t is not None and ef is not None and obs is not None:
                body = _get_body(nm)
                if body is not None:
                    try:
                        geo = obs.at(t).observe(body).apparent()
                        lon, _ = _frame_latlon(geo, ef)
                        out[nm] = lon
                        skyfield_success = True
                    except Exception as e:
                        log.debug("Skyfield failed for %s: %s", nm, e)

            # If Skyfield didn't actually yield a longitude, try SPICE
            if not skyfield_success and _SPICE_READY:
                lon = _spice_ecliptic_longitude(jd_tt, nm, frame=frame)
                if lon is not None:
                    out[nm] = lon

        return out

    # Legacy rows (10 planets) — Skyfield path only (topocentric if requested)
    rows: List[Dict[str, Any]] = []
    if main is None or t is None or ef is None or obs is None:
        return rows
    for nm, key in _PLANET_KEYS.items():
        body = _get_body(nm)
        if body is None:
            continue
        try:
            geo = obs.at(t).observe(body).apparent()
            lon, lat = _frame_latlon(geo, ef)
            # speed via central difference in the same requested frame
            step = _speed_step_for(nm)
            ts = _get_timescale()
            t0 = ts.tt_jd(jd_tt - step); t1 = ts.tt_jd(jd_tt + step)
            lon0, _ = _frame_latlon(obs.at(t0).observe(body).apparent(), ef)
            lon1, _ = _frame_latlon(obs.at(t1).observe(body).apparent(), ef)
            spd = _wrap_diff_deg(lon1, lon0) / (2.0 * step)
            rows.append({"name": nm, "lon": lon, "lat": lat, "speed": spd})
        except Exception:
            continue
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
) -> Dict[str, Dict[str, float]]:
    """
    Returns {"longitudes": {name: deg}, "velocities": {name: deg/day}}.
    Nodes receive longitudes; velocities are not defined for nodes (omitted).
    """
    lon_map: Dict[str, float] = {}
    vel_map: Dict[str, float] = {}

    # Ensure SPICE furnished
    if _SPICE_OK:
        _spice_bootstrap()

    main, _ = _get_kernels()
    t = _to_tts(jd_tt) if main is not None else None
    ef = _get_ecliptic_frame(frame) if main is not None else None
    obs = _observer(main, topocentric=topocentric, latitude=latitude,
                    longitude=longitude, elevation_m=elevation_m) if main is not None else None

    for nm in names:
        if nm in _NODE_NAMES:
            try:
                lon_map[nm] = _node_longitude(nm, jd_tt)
            except Exception:
                pass
            continue

        skyfield_success = False

        # 1) Try Skyfield (majors only) and compute velocity in same requested frame
        if nm in _PLANET_KEYS and main is not None and t is not None and ef is not None and obs is not None:
            body = _get_body(nm)
            if body is not None:
                try:
                    geo = obs.at(t).observe(body).apparent()
                    lon, _ = _frame_latlon(geo, ef)
                    lon_map[nm] = lon
                    # speed
                    step = _speed_step_for(nm)
                    ts = _get_timescale()
                    t0 = ts.tt_jd(jd_tt - step); t1 = ts.tt_jd(jd_tt + step)
                    lon0, _ = _frame_latlon(obs.at(t0).observe(body).apparent(), ef)
                    lon1, _ = _frame_latlon(obs.at(t1).observe(body).apparent(), ef)
                    vel_map[nm] = _wrap_diff_deg(lon1, lon0) / (2.0 * step)
                    skyfield_success = True
                except Exception as e:
                    log.debug("Skyfield failed for %s: %s", nm, e)

        # 2) SPICE fallback (small-bodies and failed majors)
        if not skyfield_success and _SPICE_READY:
            lon, spd = _spice_ecliptic_longitude_speed(jd_tt, nm, frame=frame)
            if lon is not None:
                lon_map[nm] = lon
            if spd is not None:
                vel_map[nm] = spd

    return {"longitudes": lon_map, "velocities": vel_map}

# Convenience alias
def get_ecliptic_longitudes(
    jd_tt: float,
    names: List[str],
    *,
    frame: str = "ecliptic-of-date",
    topocentric: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
) -> Dict[str, float]:
    return ecliptic_longitudes(
        jd_tt,
        names=names,
        frame=frame,
        topocentric=topocentric,
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def ephemeris_diagnostics(requested: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Report kernels, resolved labels, and misses for reproducible tests.
    Default set: 10 majors + 5 small bodies + 2 nodes.
    """
    main, extras = _get_kernels()
    kernels = [p for p in _KERNEL_PATHS] or [EPHEMERIS_NAME]

    if _SPICE_OK and _spice_bootstrap():
        kernels = kernels + [f"[spice] {os.path.basename(p)}" for p in _SPICE_KERNELS]

    default_names = list(_PLANET_KEYS.keys()) + \
                    ["Ceres", "Pallas", "Juno", "Vesta", "Chiron",
                     "North Node", "South Node"]
    wanted = requested or default_names

    resolved: Dict[str, Dict[str, str]] = {}
    missing: List[str] = []
    for nm in wanted:
        if nm in _NODE_NAMES:
            resolved[nm] = {"type": "node", "kernel": "computed", "label": nm}
            continue

        # Prefer SPICE for small bodies
        if _SPICE_READY and _spice_id_for_name(nm) is not None:
            resolved[nm] = {"type": "body", "kernel": "spice", "label": str(_spice_id_for_name(nm))}
            continue

        # Fall back to Skyfield kernel resolution
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
            resolved[nm] = {"type": "body", "kernel": used_kernel, "label": used_label or "n/a"}
        else:
            missing.append(nm)

    return {
        "kernels": kernels,
        "ephemeris_name": current_kernel_name(),
        "resolved": resolved,
        "missing": missing,
        "node_model": _NODE_MODEL,
    }

# Optional helper for tests
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
