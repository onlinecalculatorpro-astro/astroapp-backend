# app/core/ephemeris_adapter.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import math
import logging

log = logging.getLogger(__name__)

# Public marker used by callers for metadata
EPHEMERIS_NAME = "de421+extras"

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
_EPH = None          # Main planetary kernel (BSP)
_EPH_PATH = None     # Resolved kernel path (for logging)
_EXTRAS: Dict[str, Any] = {}  # name → SPK for asteroids (Ceres, …)
_EXTRAS_PATHS: Dict[str, str] = {}

# Preferred body keys in JPL kernels (DE421/DE44x etc.)
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
}

# Public extra bodies we want to support
_ASTEROID_PUBLIC = ["Ceres", "Pallas", "Juno", "Vesta", "Chiron"]
_NODES_PUBLIC = ["North Node", "South Node"]  # mean nodes (lat=0)

# ---------------------------
# Kernel resolution strategy
# ---------------------------
def _resolve_kernel_path() -> Optional[str]:
    """
    Priority (first that exists):
    1) env OCP_EPHEMERIS
    2) ./app/data/de421.bsp
    3) ./data/de421.bsp
    4) Let Skyfield cache/download 'de421.bsp' (last resort; may be blocked)
    """
    p = os.getenv("OCP_EPHEMERIS")
    if p and os.path.isfile(p):
        return p
    candidates = [
        os.path.join(os.getcwd(), "app", "data", "de421.bsp"),
        os.path.join(os.getcwd(), "data", "de421.bsp"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None  # let Skyfield resolve by name

def _extras_candidates_for(name: str) -> List[str]:
    """
    Candidate local filenames for each asteroid. We try local files first,
    then (optionally) let Skyfield fetch by URL/name if permitted.
    """
    # per-body env override, e.g. OCP_SPK_CERES=/path/to/ceres.bsp
    env_key = f"OCP_SPK_{name.replace(' ', '').upper()}"
    env_path = os.getenv(env_key)
    if env_path:
        return [env_path]

    base = os.path.join(os.getcwd(), "app", "data", "spk")
    base2 = os.path.join(os.getcwd(), "data", "spk")
    filename = f"{name.lower().replace(' ', '_')}.bsp"  # e.g. ceres.bsp
    return [
        os.path.join(base, filename),
        os.path.join(base2, filename),
        # As a last resort, try common alt names (if placed manually)
        os.path.join(base, f"{name.lower()}.bsp"),
        os.path.join(base2, f"{name.lower()}.bsp"),
    ]

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
    Lazy-load the main ephemeris. Never raise on import; log and return None on failure.
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
            _EPH = load("de421.bsp")
            _EPH_PATH = "de421 (cached/downloaded by Skyfield)"
            log.info("Loaded ephemeris by name: %s", _EPH_PATH)
        return _EPH
    except Exception as e:
        log.warning("Failed to load ephemeris kernel: %s", e)
        _EPH = None
        return None

def _load_extras_if_any():
    """
    Try to load extra SPK files for our asteroid list into _EXTRAS.
    This is idempotent and cheap on subsequent calls.
    """
    global _EXTRAS, _EXTRAS_PATHS
    if _EXTRAS:
        return

    if not _skyfield_available():
        return

    try:
        from skyfield.api import load
    except Exception as e:
        log.warning("Skyfield load() unavailable: %s", e)
        return

    for nm in _ASTEROID_PUBLIC:
        loaded = False
        for cand in _extras_candidates_for(nm):
            try:
                if os.path.isfile(cand):
                    _EXTRAS[nm] = load(cand)
                    _EXTRAS_PATHS[nm] = cand
                    log.info("Loaded SPK for %s: %s", nm, cand)
                    loaded = True
                    break
            except Exception as e:
                log.debug("Failed loading %s from %s: %s", nm, cand, e)
        if not loaded:
            # Optional: allow skyfield to fetch if given a URL in env (no default URL)
            url_key = f"OCP_SPK_URL_{nm.replace(' ', '').upper()}"
            url_val = os.getenv(url_key)
            if url_val:
                try:
                    _EXTRAS[nm] = load(url_val)
                    _EXTRAS_PATHS[nm] = url_val
                    log.info("Loaded SPK for %s from URL: %s", nm, url_val)
                except Exception as e:
                    log.warning("Could not download SPK for %s from %s: %s", nm, url_val, e)

# ---------------------------
# Public helpers / metadata
# ---------------------------
def load_kernel(kernel_name: str = "de421"):
    """
    Backwards-compatible entry. Returns (eph, info_str) or (None, None).
    """
    eph = _get_ephemeris()
    _load_extras_if_any()
    return eph, _EPH_PATH

def current_kernel_name() -> str:
    """Human-friendly kernel identifier for meta/debug."""
    tag = _EPH_PATH or "de421"
    if _EXTRAS_PATHS:
        extras = ",".join(f"{k}:{os.path.basename(v)}" for k, v in _EXTRAS_PATHS.items())
        return f"{tag} + [{extras}]"
    return tag

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
    return float(lon.degrees) % 360.0, float(lat.degrees)

def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0

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
        return _wrap180(lon1 - lon0) / (2.0 * delta_days)
    except Exception:
        return 0.0

# ---------------------------
# Node support (mean nodes)
# ---------------------------
def _centuries_since_j2000(jd_tt: float) -> float:
    return (jd_tt - 2451545.0) / 36525.0

def _mean_ascending_node_lon_deg(jd_tt: float) -> float:
    """
    Mean longitude of Moon's ascending node Ω (degrees, IAU/Meeus-style polynomial).
    Adequate for astrological use; lat=0 by definition; distance undefined.
    """
    T = _centuries_since_j2000(jd_tt)
    # Meeus (1998) Chap. 21/22 variant:
    Ω = (
        125.04455501
        - 1934.13618503 * T
        + 0.0020761667 * T * T
        + (T ** 3) / 467410.0
        - (T ** 4) / 606160.0
    )
    return Ω % 360.0

def _node_longitudes(jd_tt: float) -> Tuple[float, float]:
    """Return (North Node λ, South Node λ). South Node = North Node + 180°."""
    nn = _mean_ascending_node_lon_deg(jd_tt)
    sn = (nn + 180.0) % 360.0
    return nn, sn

# ---------------------------
# Target resolution
# ---------------------------
def _resolve_target_from_ephem(eph, public_name: str):
    """
    Given a public name, return a Skyfield target from either:
      - main planetary ephemeris (DE421),
      - or one of the loaded asteroid SPK files (for Ceres, Pallas, Juno, Vesta, Chiron).
    If not found, return None and let caller decide (e.g., nodes).
    """
    # 1) Main planets
    key = _PLANET_KEYS.get(public_name)
    if key:
        try:
            return eph[key]
        except Exception:
            pass

    # 2) Extras: try friendly matches against target names inside each SPK
    if _EXTRAS:
        needle = public_name.lower()
        for nm, spk in _EXTRAS.items():
            if nm.lower() != needle:
                continue
            try:
                # Try a few common target labels
                for target_key in (
                    public_name,                     # "Ceres"
                    f"{public_name} barycenter",     # rare variant
                    f"{public_name} Barycenter",
                    f"{public_name} (1)" if public_name == "Ceres" else None,
                    f"{public_name} (2)" if public_name == "Pallas" else None,
                    f"{public_name} (3)" if public_name == "Juno" else None,
                    f"{public_name} (4)" if public_name == "Vesta" else None,
                    "1 Ceres" if public_name == "Ceres" else None,
                    "2 Pallas" if public_name == "Pallas" else None,
                    "3 Juno" if public_name == "Juno" else None,
                    "4 Vesta" if public_name == "Vesta" else None,
                    "Chiron" if public_name == "Chiron" else None,
                    "2060 Chiron" if public_name == "Chiron" else None,
                    "95P/Chiron" if public_name == "Chiron" else None,
                ):
                    if not target_key:
                        continue
                    try:
                        return spk[target_key]
                    except Exception:
                        continue

                # Fallback: brute force name search inside SPK keys
                for k in getattr(spk, "names", spk.keys()):
                    if needle in str(k).lower():
                        try:
                            return spk[k]
                        except Exception:
                            continue
            except Exception as e:
                log.debug("Extras lookup failed for %s: %s", public_name, e)
    return None

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
    _load_extras_if_any()
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
        if topocentric and (latitude is not None) and (longitude is not None):
            elev = float(elevation_m or 0.0)
            observer = wgs84.latlon(float(latitude), float(longitude), elevation_m=elev)
        else:
            observer = eph["earth"]
    except Exception as e:
        log.debug("Observer build failed (%s); falling back to geocentric.", e)
        observer = eph["earth"]

    # Helper to compute lon/lat/speed for a resolved target
    def _compute_for_target(target) -> Tuple[float, float, float]:
        geo = observer.at(t).observe(target).apparent()
        lon_deg, lat_deg = _frame_latlon(geo, ecliptic_frame)
        spd = _deg_speed(jd_tt, target, observer, ecliptic_frame, delta_days=1.0)
        return lon_deg, lat_deg, spd

    # Dict form when names are provided
    if names:
        out: Dict[str, float] = {}
        for public_name in names:
            pn = str(public_name)
            # First: classical planets
            target = _resolve_target_from_ephem(eph, pn)
            if target is not None:
                try:
                    lon_deg, _lat, _spd = _compute_for_target(target)
                    out[pn] = float(lon_deg)
                    continue
                except Exception as be:
                    log.debug("Failed computing %s from ephemeris: %s", pn, be)
                    continue

            # Nodes (mean)
            if pn in _NODES_PUBLIC:
                try:
                    nn, sn = _node_longitudes(jd_tt)
                    out["North Node"] = nn
                    out["South Node"] = sn
                except Exception as ne:
                    log.debug("Failed computing nodes: %s", ne)
                continue

            # Not found — silently skip (caller may warn)
        return out

    # Legacy list-of-dicts form (Sun..Pluto + extras if available + nodes)
    rows: List[Dict[str, Any]] = []
    try:
        # Classical planets
        for public_name, key in _PLANET_KEYS.items():
            try:
                body = eph[key]
                lon_deg, lat_deg, spd = _compute_for_target(body)
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

        # Extras (asteroids) — only those we managed to resolve
        for nm in _ASTEROID_PUBLIC:
            target = _resolve_target_from_ephem(eph, nm)
            if target is None:
                continue
            try:
                lon_deg, lat_deg, spd = _compute_for_target(target)
                rows.append({
                    "name": nm,
                    "lon": lon_deg,
                    "lat": lat_deg,
                    "speed": spd,
                })
            except Exception as be:
                log.debug("Failed computing %s: %s", nm, be)

        # Mean nodes (lat=0, no distance)
        try:
            nn, sn = _node_longitudes(jd_tt)
            rows.append({"name": "North Node", "lon": nn, "lat": 0.0, "speed": 0.0})
            rows.append({"name": "South Node", "lon": sn, "lat": 0.0, "speed": 0.0})
        except Exception as ne:
            log.debug("Failed computing nodes: %s", ne)

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
    Includes extras and nodes when requested by name.
    """
    eph = _get_ephemeris()
    _load_extras_if_any()
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
        if topocentric and (latitude is not None) and (longitude is not None):
            elev = float(elevation_m or 0.0)
            observer = wgs84.latlon(float(latitude), float(longitude), elevation_m=elev)
        else:
            observer = eph["earth"]
    except Exception as e:
        log.debug("Observer build failed (%s); falling back to geocentric.", e)
        observer = eph["earth"]

    def _compute_lon_spd(target) -> Tuple[float, float]:
        geo = observer.at(t).observe(target).apparent()
        lon_deg, _lat = _frame_latlon(geo, ecliptic_frame)
        spd = _deg_speed(jd_tt, target, observer, ecliptic_frame, delta_days=1.0)
        return lon_deg, spd

    lon_map: Dict[str, float] = {}
    vel_map: Dict[str, float] = {}

    for nm in names:
        nm = str(nm)
        target = _resolve_target_from_ephem(eph, nm)
        if target is not None:
            try:
                lon, spd = _compute_lon_spd(target)
                lon_map[nm] = float(lon)
                vel_map[nm] = float(spd)
                continue
            except Exception as be:
                log.debug("Failed computing %s: %s", nm, be)
                continue

        # Nodes (mean)
        if nm in _NODES_PUBLIC:
            try:
                nn, sn = _node_longitudes(jd_tt)
                lon_map["North Node"] = nn
                lon_map["South Node"] = sn
                # Provide rough finite-difference speed (deg/day) for nodes too
                # (not critical, but keeps downstream API shape stable)
                dt = 1.0 / 24.0
                nn2, sn2 = _node_longitudes(jd_tt + dt)
                vel_map["North Node"] = _wrap180(nn2 - nn) / dt
                vel_map["South Node"] = _wrap180(sn2 - sn) / dt
            except Exception as ne:
                log.debug("Failed computing nodes: %s", ne)

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
