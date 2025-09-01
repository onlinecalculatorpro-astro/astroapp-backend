# app/core/ephemeris_adapter.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import math
import logging

log = logging.getLogger(__name__)

# Public marker used by callers for metadata
EPHEMERIS_NAME = "de421"

# ---------------------------
# Public capability check
# ---------------------------
def _skyfield_available() -> bool:
    try:
        import skyfield  # noqa: F401
        return True
    except Exception:
        return False

# ---------------------------
# Internal lazy singletons
# ---------------------------
_TS = None           # Timescale
_EPH = None          # Ephemeris kernel (BSP)
_EPH_PATH = None     # Resolved kernel path (for logging)

# Preferred body keys in JPL kernels (DE421/DE440 etc.)
# Jupiter..Pluto often exist as barycenters in DE421 — acceptable for Phase-1.
_PLANET_KEYS = {
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
    # Nodes are omitted in Phase-1 (engine tolerates absence)
}

# ---------------------------
# Kernel resolution strategy
# ---------------------------
def _resolve_kernel_path() -> Optional[str]:
    """
    Decide which BSP to use without hitting the network unless absolutely needed.
    Priority (first that exists):
    1) env OCP_EPHEMERIS
    2) ./app/data/de421.bsp (repo-bundled)
    3) ./data/de421.bsp
    4) Let Skyfield cache/download 'de421.bsp' (last resort; may be blocked)
    """
    # 1) explicit env var
    p = os.getenv("OCP_EPHEMERIS")
    if p and os.path.isfile(p):
        return p

    # 2) repo-bundled locations
    candidates = [
        os.path.join(os.getcwd(), "app", "data", "de421.bsp"),
        os.path.join(os.getcwd(), "data", "de421.bsp"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # 3) None => ask Skyfield to load by name (may download)
    return None


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


def _get_ephemeris():
    """
    Lazy-load the ephemeris. Never raise on import; log and return None on failure.
    """
    global _EPH, _EPH_PATH
    if _EPH is not None:
        return _EPH

    if not _skyfield_available():
        log.warning("Skyfield not installed; ephemeris unavailable.")
        return None

    try:
        from skyfield.api import load
        path = _resolve_kernel_path()
        if path:
            _EPH = load(path)
            _EPH_PATH = path
            log.info("Loaded ephemeris from local file: %s", path)
        else:
            # May download to Skyfield cache (only if outbound net allowed)
            _EPH = load("de421.bsp")
            _EPH_PATH = "de421 (cached/downloaded by Skyfield)"
            log.info("Loaded ephemeris by name: %s", _EPH_PATH)
        return _EPH
    except Exception as e:
        log.warning("Failed to load ephemeris kernel: %s", e)
        _EPH = None
        return None

# ---------------------------
# Public helpers / metadata
# ---------------------------
def load_kernel(kernel_name: str = "de421"):
    """
    Backwards-compatible entry. Returns (eph, info_str) or (None, None).
    Kept for older callers; new code should rely on ecliptic_longitudes directly.
    """
    eph = _get_ephemeris()
    return eph, _EPH_PATH

def current_kernel_name() -> str:
    """Human-friendly kernel identifier for meta/debug."""
    return _EPH_PATH or EPHEMERIS_NAME

def _to_tts(jd_tt: float):
    ts = _get_timescale()
    if ts is None:
        return None
    try:
        return ts.tt_jd(jd_tt)
    except Exception as e:
        log.warning("Failed to create Skyfield time from JD_TT %.6f: %s", jd_tt, e)
        return None

def _frame_latlon(geo, ecliptic_frame):
    # Return (lon_deg, lat_deg)
    lat, lon, _ = geo.frame_latlon(ecliptic_frame)
    # Skyfield returns angles; convert to degrees
    return float(lon.degrees) % 360.0, float(lat.degrees)

def _deg_speed(
    jd_tt: float,
    body,
    observer,
    ecliptic_frame,
    delta_days: float = 1.0,
) -> float:
    """
    Approx angular speed (deg/day) using central difference for the SAME observer
    (geocentric or topocentric).
    """
    ts = _get_timescale()
    if ts is None:
        return 0.0
    try:
        t0 = ts.tt_jd(jd_tt - delta_days)
        t1 = ts.tt_jd(jd_tt + delta_days)
        lon0, _ = _frame_latlon(observer.at(t0).observe(body).apparent(), ecliptic_frame)
        lon1, _ = _frame_latlon(observer.at(t1).observe(body).apparent(), ecliptic_frame)
        d = (lon1 - lon0 + 540.0) % 360.0 - 180.0  # wrap to [-180,180]
        return d / (2.0 * delta_days)
    except Exception:
        return 0.0

# ---------------------------
# Main APIs
# ---------------------------
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
    If `names` is provided: returns {name: longitude_deg} for those names.
    If `names` is None: returns list of dicts:
       [{ "name": <str>, "lon": <deg>, "lat": <deg>, "speed": <deg/day> }, ...]
    Supports topocentric when topocentric=True and lat/lon are provided.

    If ephemeris is unavailable, returns {} or [] accordingly (never raises).
    """
    # Load lazily
    eph = _get_ephemeris()
    if eph is None:
        return {} if names else []

    # Build Skyfield time
    t = _to_tts(jd_tt)
    if t is None:
        return {} if names else []

    # Import frames lazily to avoid import-time failures if Skyfield absent
    try:
        from skyfield.framelib import ecliptic_frame
        from skyfield.api import wgs84
    except Exception as e:
        log.warning("Ecliptic frame unavailable: %s", e)
        return {} if names else []

    # Build observer: geocentric or topocentric
    try:
        if topocentric and latitude is not None and longitude is not None:
            elev = float(elevation_m or 0.0)
            observer = wgs84.latlon(float(latitude), float(longitude), elevation_m=elev)
        else:
            observer = eph["earth"]
    except Exception as e:
        log.debug("Observer build failed (%s); falling back to geocentric.", e)
        observer = eph["earth"]

    if names:
        # Dict form for requested names
        out: Dict[str, float] = {}
        # Build a reverse map once to avoid lookups on each name
        key_by_name = _PLANET_KEYS
        for public_name in names:
            key = key_by_name.get(str(public_name))
            if not key:
                continue
            try:
                body = eph[key]
                geo = observer.at(t).observe(body).apparent()
                lon_deg, _lat_deg = _frame_latlon(geo, ecliptic_frame)
                out[str(public_name)] = float(lon_deg)
            except Exception as be:
                log.debug("Failed computing %s: %s", public_name, be)
        return out

    # Legacy list-of-dicts form (Sun..Pluto)
    rows: List[Dict[str, Any]] = []
    try:
        for public_name, key in _PLANET_KEYS.items():
            try:
                body = eph[key]
                geo = observer.at(t).observe(body).apparent()
                lon_deg, lat_deg = _frame_latlon(geo, ecliptic_frame)
                spd = _deg_speed(jd_tt, body, observer, ecliptic_frame, delta_days=1.0)
                rows.append({
                    "name": public_name,
                    "lon": lon_deg,
                    "lat": lat_deg,
                    "speed": spd,
                })
            except KeyError:
                log.debug("Body %s (%s) not in kernel; skipping.", public_name, key)
            except Exception as be:
                log.debug("Failed computing %s: %s", public_name, be)
        return rows
    except Exception as e:
        log.warning("Ephemeris computation failed: %s", e)
        return []

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
    Return {"longitudes": {name: deg}, "velocities": {name: deg/day}}
    for the requested names. Uses the same observer for speed FD.
    """
    eph = _get_ephemeris()
    if eph is None:
        return {"longitudes": {}, "velocities": {}}

    t = _to_tts(jd_tt)
    if t is None:
        return {"longitudes": {}, "velocities": {}}

    try:
        from skyfield.framelib import ecliptic_frame
        from skyfield.api import wgs84
    except Exception as e:
        log.warning("Ecliptic frame unavailable: %s", e)
        return {"longitudes": {}, "velocities": {}}

    try:
        if topocentric and latitude is not None and longitude is not None:
            elev = float(elevation_m or 0.0)
            observer = wgs84.latlon(float(latitude), float(longitude), elevation_m=elev)
        else:
            observer = eph["earth"]
    except Exception as e:
        log.debug("Observer build failed (%s); falling back to geocentric.", e)
        observer = eph["earth"]

    lon_map: Dict[str, float] = {}
    vel_map: Dict[str, float] = {}

    for nm in names:
        key = _PLANET_KEYS.get(str(nm))
        if not key:
            continue
        try:
            body = eph[key]
            geo = observer.at(t).observe(body).apparent()
            lon_deg, _lat_deg = _frame_latlon(geo, ecliptic_frame)
            lon_map[str(nm)] = float(lon_deg)
            vel_map[str(nm)] = float(_deg_speed(jd_tt, body, observer, ecliptic_frame, delta_days=1.0))
        except Exception as be:
            log.debug("Failed computing %s: %s", nm, be)

    return {"longitudes": lon_map, "velocities": vel_map}

# Convenience alias for engines that expect just a {name: lon} mapping
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

# ---------------------------
# Utility (optional)
# ---------------------------
def _angular_sep(a: float, b: float) -> float:
    """Smallest angular separation in degrees [0..180]."""
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d
