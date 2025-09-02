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
# Capability check
# ──────────────────────────────────────────────────────────────────────────────
def _skyfield_available() -> bool:
    try:
        import skyfield  # noqa: F401
        return True
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Lazy singletons
# ──────────────────────────────────────────────────────────────────────────────
_TS = None                 # Skyfield timescale
_MAIN = None               # Main ephemeris (DE421 by default)
_EXTRA: List[Any] = []     # Extra SPKs (small bodies)
_KERNEL_PATHS: List[str] = []  # For reporting/diagnostics

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

# Small bodies we want to support out-of-the-box
_SMALLBODY_HINTS: Dict[str, List[str]] = {
    # Include a mix of common labels and NAIF-like numeric IDs that kernels expose
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
_NODE_ALERT_DEG     = float(os.getenv("OCP_NODE_ALERT_DEG", "10.0"))       # warn if |true-mean| >
_NODE_CACHE_DECIMALS = int(os.getenv("OCP_NODE_CACHE_DECIMALS", "5"))      # JD rounding for cache (~0.864 s)
# Allow explicit list of SPK files (comma separated) or fallback to directory scan
# OCP_EXTRA_SPK_FILES=/abs/path/asteroids.bsp,/abs/path/chiron.bsp
# OCP_EXTRA_SPK_DIR=app/data/spk
# OCP_EPHEMERIS=/abs/path/de421.bsp

# ──────────────────────────────────────────────────────────────────────────────
# Kernel resolution & loading
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_kernel_path() -> Optional[str]:
    """
    Resolve a local DE421 first; else allow Skyfield to load by name.
    Priority:
      1) OCP_EPHEMERIS (absolute path)
      2) app/data/de421.bsp
      3) data/de421.bsp
      4) None -> load("de421.bsp")
    """
    p = os.getenv("OCP_EPHEMERIS")
    if p and os.path.isfile(p):
        return p
    for c in (
        os.path.join(os.getcwd(), "app", "data", "de421.bsp"),
        os.path.join(os.getcwd(), "data", "de421.bsp"),
    ):
        if os.path.isfile(c):
            return c
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
            # deterministic order for fuzzers
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
        log.warning("Failed to load kernel %s: %s", path, e)
        return None

def _get_kernels():
    """
    Returns (main_kernel, [extra_kernels]).
    Soft-fails to (None, []) if Skyfield or kernels not available.
    """
    global _MAIN, _EXTRA, _KERNEL_PATHS, EPHEMERIS_NAME
    if _MAIN is not None:
        return _MAIN, _EXTRA
    if not _skyfield_available():
        log.warning("Skyfield not installed; ephemeris unavailable.")
        return None, []

    # Main kernel
    path = _resolve_kernel_path()
    if path:
        _MAIN = _load_kernel(path)
        if _MAIN:
            _KERNEL_PATHS.append(path)
    else:
        try:
            from skyfield.api import load
            _MAIN = load("de421.bsp")
            _KERNEL_PATHS.append("de421 (cache/name)")
        except Exception as e:
            log.warning("Failed to load main kernel: %s", e)
            _MAIN = None

    # Extra SPKs (small bodies)
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
    """Human-friendly kernel list for meta/debug."""
    if _KERNEL_PATHS:
        return ", ".join(os.path.basename(p) for p in _KERNEL_PATHS)
    return EPHEMERIS_NAME

def load_kernel(kernel_name: str = "de421"):
    """Legacy helper used by some callers (kept for compatibility)."""
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
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────
def _frame_latlon(geo, ecliptic_frame) -> Tuple[float, float]:
    # returns (lon_deg [0..360), lat_deg)
    lat, lon, _ = geo.frame_latlon(ecliptic_frame)
    return float(lon.degrees) % 360.0, float(lat.degrees)

def _cross(a, b):
    ax, ay, az = a; bx, by, bz = b
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)

def _atan2deg(y, x):
    return (math.degrees(math.atan2(y, x)) % 360.0)

def _wrap_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference a-b in degrees in (-180, 180]."""
    return ((a - b + 540.0) % 360.0) - 180.0

def _central_diff_speed(jd_tt: float, body, observer, ecliptic_frame, delta_days: float) -> float:
    ts = _get_timescale()
    if ts is None:
        return 0.0
    try:
        t0 = ts.tt_jd(jd_tt - delta_days)
        t1 = ts.tt_jd(jd_tt + delta_days)
        lon0, _ = _frame_latlon(observer.at(t0).observe(body).apparent(), ecliptic_frame)
        lon1, _ = _frame_latlon(observer.at(t1).observe(body).apparent(), ecliptic_frame)
        return _wrap_diff_deg(lon1, lon0) / (2.0 * delta_days)
    except Exception:
        return 0.0

def _speed_step_for(name: str) -> float:
    # Tighter step for Moon; default for all others
    return _SPEED_STEP_FAST if name == "Moon" else _SPEED_STEP_DEFAULT

# ──────────────────────────────────────────────────────────────────────────────
# Frames
# ──────────────────────────────────────────────────────────────────────────────
def _get_ecliptic_frame(frame: str):
    """
    Returns a Skyfield frame object.
    Supported: "ecliptic-of-date" (default), "ecliptic-J2000".
    """
    try:
        from skyfield import framelib as _fl
        # Prefer explicit J2000 frame when requested
        if frame and frame.lower() in ("ecliptic-j2000", "j2000", "ecl-j2000"):
            f = getattr(_fl, "ecliptic_J2000_frame", None)
            if f is not None:
                return f
        # Fallback/default: of-date
        return getattr(_fl, "ecliptic_frame", None)
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Kernel iteration
# ──────────────────────────────────────────────────────────────────────────────
def _kernels_iter() -> Iterable[Any]:
    main, extra = _get_kernels()
    if main: 
        yield main
    for k in extra: 
        yield k

# ──────────────────────────────────────────────────────────────────────────────
# Small-body & body resolution (cached, case-insensitive, fuzzy)
# ──────────────────────────────────────────────────────────────────────────────
def _all_kernel_labels(k) -> List[str]:
    """Enumerate likely labels from a kernel for fuzzy matching."""
    labels: List[str] = []
    for attr in ("names", "aliases", "bodies"):
        try:
            obj = getattr(k, attr, None)
            if isinstance(obj, dict):
                labels.extend(list(obj.keys()))
        except Exception:
            pass
    # Seed with our known hints to increase hit rate
    for lst in _SMALLBODY_HINTS.values():
        labels.extend(lst)
    # unique preserving order
    seen = set(); out: List[str] = []
    for s in labels:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

@lru_cache(maxsize=2048)
def _resolve_small_body(name_norm: str):
    """
    Resolve small body by:
      1) Trying known hint labels exactly (across kernels),
      2) Fuzzy case-insensitive substring search in kernel labels.
    name_norm is canonical capitalization (e.g., "Ceres").
    """
    hints = _SMALLBODY_HINTS.get(name_norm, [])
    # 1) Try hints directly
    for label in hints:
        for k in _kernels_iter():
            try:
                return k[label]
            except Exception:
                continue
    # 2) Fuzzy (substring, case-insensitive)
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
    # First try main mapping (10 bodies)
    if name in _PLANET_KEYS:
        key = _PLANET_KEYS[name]
        for k in _kernels_iter():
            try:
                return k[key]
            except Exception:
                continue
    # Then small-bodies
    if name in _SMALLBODY_HINTS or name.lower() in ("ceres", "pallas", "juno", "vesta", "chiron"):
        b = _resolve_small_body(name.capitalize())
        if b is not None:
            return b
    log.debug("Body not resolved: %s", name)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Lunar nodes (geocentric)
# ──────────────────────────────────────────────────────────────────────────────
def _mean_node(jd_tt: float) -> float:
    """
    Mean node longitude (deg) in ecliptic-of-date, geocentric.
    Simplified classical expression sufficient for sanity check & fallback.
    """
    T = (jd_tt - 2451545.0) / 36525.0
    Omega = 125.04452 - 1934.136261 * T + 0.0020708*(T**2) + (T**3)/450000.0
    return Omega % 360.0

@lru_cache(maxsize=8192)
def _true_node_geocentric_cached(jd_q: float) -> float:
    """
    Cached true (osculating) ascending node longitude (deg), geocentric.
    Uses central difference of lunar state around jd_q in ecliptic-of-date coords.
    """
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

        # Lunar position vectors in ecliptic-of-date
        r0 = tuple(map(float, earth.at(t0).observe(moon).apparent().frame_xyz(ecliptic_frame).au))
        rp = tuple(map(float, earth.at(tp).observe(moon).apparent().frame_xyz(ecliptic_frame).au))
        rm = tuple(map(float, earth.at(tm).observe(moon).apparent().frame_xyz(ecliptic_frame).au))

        # Velocity via central difference
        v = tuple((rp[i] - rm[i]) / (2.0 * step) for i in range(3))

        # Orbit plane normal h = r × v ; ecliptic normal k = (0,0,1)
        h = _cross(r0, v)
        n = _cross((0.0, 0.0, 1.0), h)  # node line in ecliptic plane

        # Degeneracy guard (robust norm check)
        norm_xy = math.hypot(n[0], n[1])
        if norm_xy < 1e-12:
            return _mean_node(jd_q)

        val = _atan2deg(n[1], n[0])

        # Sanity check vs mean node
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
    """
    Public resolver for node longitude (deg), geocentric.
    Honors OCP_NODE_MODEL = "true" | "mean". Default "true".
    """
    if _NODE_MODEL == "mean":
        asc = _mean_node(jd_tt)
    else:
        # Quantize JD for cache hit (e.g., 1e-5 d ≈ 0.864 s)
        jd_q = round(float(jd_tt), _NODE_CACHE_DECIMALS)
        asc = _true_node_geocentric_cached(jd_q)
    return asc if name == "North Node" else (asc + 180.0) % 360.0

# ──────────────────────────────────────────────────────────────────────────────
# Observer (for planets/small-bodies)
# ──────────────────────────────────────────────────────────────────────────────
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
# Public APIs
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
    If `names` is None: returns a legacy list of dict rows for the original 10 planets:
        [{ "name": <str>, "lon": <deg>, "lat": <deg>, "speed": <deg/day> }, ...]
    Topocentric supported for bodies; nodes are always geocentric by definition.
    Soft-fails to {} or [] if kernels/Skyfield unavailable.
    """
    main, _ = _get_kernels()
    if main is None:
        return {} if names else []

    t = _to_tts(jd_tt)
    if t is None:
        return {} if names else []

    ef = _get_ecliptic_frame(frame)
    if ef is None:
        return {} if names else []

    obs = _observer(main, topocentric=topocentric, latitude=latitude,
                    longitude=longitude, elevation_m=elevation_m)
    if obs is None:
        return {} if names else []

    # Map mode
    if names:
        out: Dict[str, float] = {}
        for nm in names:
            if nm in _NODE_NAMES:
                try:
                    out[nm] = _node_longitude(nm, jd_tt)
                except Exception:
                    pass
                continue
            body = _get_body(nm)
            if body is None:
                continue
            try:
                geo = obs.at(t).observe(body).apparent()
                lon, _ = _frame_latlon(geo, ef)
                out[nm] = lon
            except Exception:
                continue
        return out

    # Legacy rows for the 10 canonical bodies
    rows: List[Dict[str, Any]] = []
    for nm, key in _PLANET_KEYS.items():
        body = _get_body(nm)
        if body is None:
            continue
        try:
            geo = obs.at(t).observe(body).apparent()
            lon, lat = _frame_latlon(geo, ef)
            spd = _central_diff_speed(jd_tt, body, obs, ef, delta_days=_speed_step_for(nm))
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

    main, _ = _get_kernels()
    if main is None:
        return {"longitudes": lon_map, "velocities": vel_map}

    t = _to_tts(jd_tt)
    if t is None:
        return {"longitudes": lon_map, "velocities": vel_map}

    ef = _get_ecliptic_frame(frame)
    if ef is None:
        return {"longitudes": lon_map, "velocities": vel_map}

    obs = _observer(main, topocentric=topocentric, latitude=latitude,
                    longitude=longitude, elevation_m=elevation_m)
    if obs is None:
        return {"longitudes": lon_map, "velocities": vel_map}

    for nm in names:
        if nm in _NODE_NAMES:
            try:
                lon_map[nm] = _node_longitude(nm, jd_tt)
            except Exception:
                pass
            continue

        body = _get_body(nm)
        if body is None:
            continue
        try:
            geo = obs.at(t).observe(body).apparent()
            lon, _ = _frame_latlon(geo, ef)
            lon_map[nm] = lon
            vel_map[nm] = _central_diff_speed(
                jd_tt, body, obs, ef, delta_days=_speed_step_for(nm)
            )
        except Exception:
            continue

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
# Diagnostics (for harness/fuzzer parity)
# ──────────────────────────────────────────────────────────────────────────────
def ephemeris_diagnostics(requested: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Report kernels, resolved labels, and misses for reproducible tests.
    Default set includes 10 majors + 5 small bodies + 2 nodes = 17 bodies.
    """
    main, extras = _get_kernels()
    kernels = [p for p in _KERNEL_PATHS] or [EPHEMERIS_NAME]

    default_names = list(_PLANET_KEYS.keys()) + \
                    ["Ceres", "Pallas", "Juno", "Vesta", "Chiron",
                     "North Node", "South Node"]
    wanted = requested or default_names

    resolved = {}
    missing: List[str] = []
    for nm in wanted:
        if nm in _NODE_NAMES:
            resolved[nm] = {"type": "node", "kernel": "computed", "label": nm}
            continue
        b = _get_body(nm)
        if b is None:
            missing.append(nm)
            continue
        # Try to identify which kernel/label matched
        used_label = None
        used_kernel = "main"
        for k in _kernels_iter():
            # Probe the most likely labels
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

    return {
        "kernels": kernels,
        "ephemeris_name": current_kernel_name(),
        "resolved": resolved,
        "missing": missing,
        "node_model": _NODE_MODEL,
    }

# Optional helper for tests
def clear_adapter_caches() -> None:
    """Clear internal caches (useful in tests between kernel configurations)."""
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
