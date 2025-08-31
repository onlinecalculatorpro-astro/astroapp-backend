# app/core/ephemeris_adapter.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import math
import logging

log = logging.getLogger(__name__)

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
# Public helpers
# ---------------------------
def load_kernel(kernel_name: str = "de421"):
    """
    Backwards-compatible entry. Returns (eph, info_str) or (None, None).
    Kept for older callers; new code should rely on ecliptic_longitudes directly.
    """
    eph = _get_ephemeris()
    return eph, _EPH_PATH

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

def _deg_speed(jd_tt: float, body, ecliptic_frame, delta_days: float = 1.0) -> float:
    """
    Approx angular speed (deg/day) using central difference.
    """
    ts = _get_timescale()
    if ts is None:
        return 0.0
    try:
        t0 = ts.tt_jd(jd_tt - delta_days)
        t1 = ts.tt_jd(jd_tt + delta_days)
        earth = _EPH["earth"]
        lon0, _ = _frame_latlon(earth.at(t0).observe(body).apparent(), ecliptic_frame)
        lon1, _ = _frame_latlon(earth.at(t1).observe(body).apparent(), ecliptic_frame)
        d = (lon1 - lon0 + 540.0) % 360.0 - 180.0  # wrap to [-180,180]
        return d / (2.0 * delta_days)
    except Exception:
        return 0.0

# ---------------------------
# Main API
# ---------------------------
def ecliptic_longitudes(jd_tt: float, lat: float = 0.0, lon: float = 0.0) -> List[Dict[str, Any]]:
    """
    Return list of dicts:
      { "name": <str>, "lon": <deg>, "lat": <deg>, "speed": <deg/day> }
    for Sun..Pluto (barycenters where applicable). Ignores observer lat/lon for Phase-1.

    If ephemeris is unavailable, returns [] (never raises).
    """
    # Load lazily
    eph = _get_ephemeris()
    if eph is None:
        return []

    # Build Skyfield time
    t = _to_tts(jd_tt)
    if t is None:
        return []

    # Import frame lazily to avoid import-time failures if Skyfield absent
    try:
        from skyfield.framelib import ecliptic_frame
    except Exception as e:
        log.warning("Ecliptic frame unavailable: %s", e)
        return []

    out: List[Dict[str, Any]] = []

    try:
        earth = eph["earth"]
        for public_name, key in _PLANET_KEYS.items():
            try:
                body = eph[key]
                geo = earth.at(t).observe(body).apparent()
                lon_deg, lat_deg = _frame_latlon(geo, ecliptic_frame)

                # Estimate speed (deg/day) via +/- 1 day central difference
                spd = _deg_speed(jd_tt, body, ecliptic_frame, delta_days=1.0)

                out.append({
                    "name": public_name,
                    "lon": lon_deg,
                    "lat": lat_deg,
                    "speed": spd,
                })
            except KeyError:
                # Body key not present in this kernel
                log.debug("Body %s (%s) not in kernel; skipping.", public_name, key)
            except Exception as be:
                log.debug("Failed computing %s: %s", public_name, be)

        return out
    except Exception as e:
        log.warning("Ephemeris computation failed: %s", e)
        return []

# ---------------------------
# Utility (optional)
# ---------------------------
def _angular_sep(a: float, b: float) -> float:
    """Smallest angular separation in degrees [0..180]."""
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d
