# app/core/astronomy.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import math
import os
import inspect
from functools import lru_cache

__all__ = ["compute_chart"]

# ───────────────────────────── Exceptions ─────────────────────────────
class AstronomyError(ValueError):
    """User-friendly exception with a stable error code."""
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")

# ───────────────────────────── Resilient imports ─────────────────────────────
try:
    from app.core import ephemeris_adapter as eph
except Exception as e:
    eph = None
    _EPH_IMPORT_ERROR = e

try:
    from app.core import time_kernel as _tk  # single SoT for JD conversions
except Exception:
    _tk = None

# ───────────────────────────── Config & constants ─────────────────────────────
# Allowed bodies (extendable later)
ALLOWED_BODIES = {
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"
    # Optional later: "True Node", "Mean Node", "Chiron"
}
_DEF_BODIES: Tuple[str, ...] = tuple(ALLOWED_BODIES)

# Caching quantization (for stable LRU keys)
_JD_QUANT       = float(os.getenv("OCP_ASTRO_JD_QUANT", "1e-7"))   # ~0.009 s
_LATLON_QUANT   = float(os.getenv("OCP_ASTRO_LL_QUANT", "1e-6"))   # ~0.11 m
_ELEV_QUANT     = float(os.getenv("OCP_ASTRO_ELEV_QUANT", "0.1"))  # 10 cm

# Speed finite-difference step (days) when adapter lacks velocities
_SPEED_STEP     = float(os.getenv("OCP_SPEED_FD_STEP_DAYS", "0.5"))  # ±12h

# Default ayanamsa name when sidereal and none supplied
_DEF_AYANAMSA   = os.getenv("OCP_AYANAMSA_DEFAULT", "lahiri").strip().lower()

# Geographic safety / messages
_GEO_SOFT_LAT       = float(os.getenv("OCP_GEO_SOFT_LAT", "89.5"))   # deg; nudge if |lat| ≥ this
_GEO_HARD_LAT       = float(os.getenv("OCP_GEO_HARD_LAT", "89.9"))   # deg; disable topo past this
_ELEV_MIN           = float(os.getenv("OCP_GEO_ELEV_MIN", "-500.0")) # Dead Sea ~ -430 m
_ELEV_MAX           = float(os.getenv("OCP_GEO_ELEV_MAX", "10000.0"))
_ELEV_WARN          = float(os.getenv("OCP_GEO_ELEV_WARN", "3000.0"))
_ANTIMER_WARN_LON   = float(os.getenv("OCP_GEO_ANTI_WARN", "179.9"))

# ───────────────────────────── Helpers ─────────────────────────────
def _norm360(x: float) -> float:
    r = math.fmod(x, 360.0)
    if r < 0.0:
        r += 360.0
    return r

def _shortest_signed_delta_deg(a2: float, a1: float) -> float:
    """Signed delta (a2 - a1) in (-180, +180]."""
    d = (a2 - a1 + 540.0) % 360.0 - 180.0
    return -180.0 if d == 180.0 else d

def _coerce_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
    return default

def _q(x: Optional[float], q: float) -> Optional[float]:
    if x is None:
        return None
    return round(float(x) / q) * q

def _is_finite(x: Any) -> bool:
    try:
        xf = float(x)
        return math.isfinite(xf)
    except Exception:
        return False

def _normalize_lon180(lon: float) -> float:
    """Normalize longitude to [-180, 180)."""
    return ((float(lon) + 180.0) % 360.0) - 180.0

# ───────────────────────────── Validation ─────────────────────────────
def _validate_payload(payload: Dict[str, Any]) -> Tuple[str, List[str]]:
    mode = str(payload.get("mode", "tropical")).strip().lower()
    if mode not in ("tropical", "sidereal"):
        raise AstronomyError("invalid_input", "mode must be 'tropical' or 'sidereal'")

    bodies = payload.get("bodies")
    if bodies is None:
        bodies_list = list(_DEF_BODIES)
    else:
        try:
            bodies_list = [str(b) for b in bodies]
        except Exception:
            raise AstronomyError("invalid_input", "bodies must be a list of names")
        unknown = [b for b in bodies_list if b not in ALLOWED_BODIES]
        if unknown:
            allowed = ", ".join(sorted(ALLOWED_BODIES))
            raise AstronomyError("unsupported_body", f"'{unknown[0]}' not supported (allowed: {allowed})")
    return mode, bodies_list

def _ensure_timescales(payload: Dict[str, Any]) -> Tuple[float, float, float]:
    jd_ut  = payload.get("jd_ut")
    jd_tt  = payload.get("jd_tt")
    jd_ut1 = payload.get("jd_ut1")

    if all(isinstance(x, (int, float)) for x in (jd_ut, jd_tt, jd_ut1)):
        return float(jd_ut), float(jd_tt), float(jd_ut1)

    if _tk is not None:
        for fname in ("timescales_from_civil", "compute_timescales", "build_timescales",
                      "to_timescales", "from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(
                        payload.get("date"),
                        payload.get("time"),
                        payload.get("place_tz"),
                        payload.get("latitude"),
                        payload.get("longitude"),
                    )
                    return float(out["jd_ut"]), float(out["jd_tt"]), float(out["jd_ut1"])
                except Exception:
                    continue

    missing = [k for k, v in (("jd_ut", jd_ut), ("jd_tt", jd_tt), ("jd_ut1", jd_ut1)) if not isinstance(v, (int, float))]
    raise AstronomyError("timescales_missing", f"Supply {', '.join(missing)} or configure time_kernel")

# ───────────────────────────── Geographic sanity for topocentric ─────────────────────────────
def _validate_and_normalize_geo_for_topo(
    lat: Any, lon: Any, elev: Any
) -> Tuple[float, float, Optional[float], List[str], bool]:
    """
    Returns (lat_deg, lon_deg, elev_m, warnings[], downgraded_to_geocentric)
    - Rejects NaN/inf, clamps latitude, normalizes longitude
    - Nudge near poles (soft), disable topocentric at extreme poles (hard)
    - Clamps elevation into sane range (with warnings)
    """
    warnings: List[str] = []
    downgraded = False

    if not _is_finite(lat) or not _is_finite(lon):
        raise AstronomyError("invalid_input", "latitude/longitude must be finite numbers")

    latf = float(lat)
    lonf = float(lon)

    if latf < -90.0 or latf > 90.0:
        warnings.append("latitude_clamped_to_range")
        latf = max(-90.0, min(90.0, latf))
    lonf = _normalize_lon180(lonf)

    abslat = abs(latf)
    if abslat >= _GEO_HARD_LAT:
        warnings.append("topocentric_disabled_near_pole(hard)")
        downgraded = True
    elif abslat >= _GEO_SOFT_LAT:
        # Nudge slightly away from the pole to avoid singularities
        nudge_target = math.copysign((_GEO_HARD_LAT - 0.05), latf)
        latf = nudge_target
        warnings.append("latitude_soft_nudged_from_pole")

    if abs(lonf) >= _ANTIMER_WARN_LON:
        warnings.append("near_antimeridian_longitude")

    # Elevation
    if elev is None or (isinstance(elev, str) and elev.strip() == ""):
        elev_m: Optional[float] = None
    else:
        if not _is_finite(elev):
            raise AstronomyError("invalid_input", "elevation_m must be a finite number in meters")
        elev_m = float(elev)
        if elev_m < _ELEV_MIN:
            warnings.append("elevation_clamped_min")
            elev_m = _ELEV_MIN
        elif elev_m > _ELEV_MAX:
            warnings.append("elevation_clamped_max")
            elev_m = _ELEV_MAX
        elif abs(elev_m) >= _ELEV_WARN:
            warnings.append("very_high_elevation_site")

    return latf, lonf, elev_m, warnings, downgraded

# ───────────────────────────── Ayanāṁśa resolver ─────────────────────────────
@lru_cache(maxsize=4096)
def _ayanamsa_deg_cached(jd_tt_q: float, ay_key: str) -> Tuple[float, str]:
    """Resolve ayanamsa degrees for quantized JD(TT) + normalized name key."""
    # Try project-level resolvers first
    for modpath, fn in (("app.core.ayanamsa", "get_ayanamsa_deg"),
                        ("app.core.astro_extras", "get_ayanamsa_deg")):
        try:
            mod = __import__(modpath, fromlist=[fn])
            fnobj = getattr(mod, fn, None)
            if callable(fnobj):
                try:
                    return float(fnobj(jd_tt_q, ay_key)), ay_key
                except TypeError:
                    return float(fnobj(ay_key, jd_tt_q)), ay_key
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback: high-quality linearized model around J2000 for common variants
    name = (ay_key or _DEF_AYANAMSA or "lahiri").lower()
    AY_J2000_DEG       = (23 + 51/60 + 26.26/3600)      # ≈ 23.857294444°
    RATE_AS_PER_YEAR   = 50.290966                      # arcsec/year
    Tcent              = (jd_tt_q - 2451545.0) / 36525.0
    years              = Tcent * 100.0
    base               = AY_J2000_DEG + (RATE_AS_PER_YEAR * years) / 3600.0

    if name in ("lahiri", "chitrapaksha", "default", "sidereal"):
        return base, "lahiri(fallback)"
    if name in ("fagan", "fagan_bradley", "fagan/bradley"):
        return base + (0.83 / 60.0), "fagan_bradley(fallback)"
    if name in ("krishnamurti", "kp"):
        return base - (20.0 / 3600.0), "krishnamurti(fallback)"
    return base, f"ayanamsa_fallback_to_lahiri({name})"

def _resolve_ayanamsa(jd_tt: float, ayanamsa: Any) -> Tuple[Optional[float], Optional[str]]:
    """Accept explicit degrees or named model; return (deg, note)."""
    if ayanamsa is None or (isinstance(ayanamsa, str) and not ayanamsa.strip()):
        key = _DEF_AYANAMSA
    elif isinstance(ayanamsa, (int, float)):
        return float(ayanamsa), "explicit"
    else:
        key = str(ayanamsa).strip().lower()

    jd_q = _q(jd_tt, _JD_QUANT)
    ay, note = _ayanamsa_deg_cached(jd_q, key)
    return float(ay), note

# ───────────────────────────── Ephemeris adapter calls (cached) ─────────────────────────────
def _adapter_source_tag() -> str:
    return getattr(eph, "current_kernel_name", None) or getattr(eph, "EPHEMERIS_NAME", None) or "adapter"

def _adapter_callable(*names: str):
    for n in names:
        fn = getattr(eph, n, None)
        if callable(fn):
            return fn
    return None

def _frame_kw(sig: inspect.Signature) -> Dict[str, Any]:
    return {"frame": "ecliptic-of-date"} if "frame" in sig.parameters else {}

@lru_cache(maxsize=8192)
def _cached_longitudes(
    jd_tt_q: float,
    names_key: Tuple[str, ...],
    topocentric: bool,
    lat_q: Optional[float],
    lon_q: Optional[float],
    elev_q: Optional[float],
) -> Tuple[Dict[str, float], str]:
    """LRU-cached ecliptic longitudes call to adapter (TT, ecliptic-of-date)."""
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")

    fn = _adapter_callable("ecliptic_longitudes", "planetary_longitudes", "get_ecliptic_longitudes")
    if fn is None:
        raise AstronomyError("adapter_api_mismatch",
                             "ephemeris_adapter missing any of: ecliptic_longitudes/planetary_longitudes/get_ecliptic_longitudes")

    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    if "jd_tt" in sig.parameters: kwargs["jd_tt"] = jd_tt_q
    elif "jd"   in sig.parameters: kwargs["jd"]   = jd_tt_q
    if "names" in sig.parameters:  kwargs["names"] = list(names_key)
    kwargs.update(_frame_kw(sig))

    if "topocentric" in sig.parameters:
        kwargs["topocentric"] = topocentric
    if topocentric:
        if "latitude" in sig.parameters and lat_q is not None:    kwargs["latitude"]   = float(lat_q)
        if "longitude" in sig.parameters and lon_q is not None:   kwargs["longitude"]  = float(lon_q)
        if "elevation_m" in sig.parameters and elev_q is not None:kwargs["elevation_m"]= float(elev_q)

    res = fn(**kwargs)
    if isinstance(res, dict):
        out = {str(k): float(v) for k, v in res.items() if str(k) in names_key}
    elif isinstance(res, (list, tuple)):
        out = {names_key[i]: float(res[i]) for i in range(min(len(names_key), len(res)))}
    else:
        raise AstronomyError("adapter_return_invalid", f"{fn.__name__} returned unsupported type: {type(res)}")
    return out, _adapter_source_tag()

def _longitudes_with_velocity_if_available(
    jd_tt_q: float,
    names_key: Tuple[str, ...],
    topocentric: bool,
    lat_q: Optional[float],
    lon_q: Optional[float],
    elev_q: Optional[float],
) -> Optional[Tuple[Dict[str, Tuple[float, float]], str]]:
    """Try adapter API that returns (lon, speed); return None if not available."""
    if eph is None:
        return None

    fn = _adapter_callable(
        "ecliptic_longitudes_and_velocities",
        "ecliptic_longitudes_with_speed",
        "get_ecliptic_longitudes_and_velocities",
        "planetary_longitudes_and_velocities",
    )
    if fn is None:
        return None

    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    if "jd_tt" in sig.parameters: kwargs["jd_tt"] = jd_tt_q
    elif "jd"   in sig.parameters: kwargs["jd"]   = jd_tt_q
    if "names" in sig.parameters:  kwargs["names"] = list(names_key)
    kwargs.update(_frame_kw(sig))
    if "topocentric" in sig.parameters: kwargs["topocentric"] = topocentric
    if topocentric:
        if "latitude" in sig.parameters and lat_q is not None:    kwargs["latitude"]   = float(lat_q)
        if "longitude" in sig.parameters and lon_q is not None:   kwargs["longitude"]  = float(lon_q)
        if "elevation_m" in sig.parameters and elev_q is not None:kwargs["elevation_m"]= float(elev_q)

    res = fn(**kwargs)
    out: Dict[str, Tuple[float, float]] = {}
    if isinstance(res, dict):
        # {name: (lon, speed)} OR {"longitudes":{...}, "velocities":{...}}
        sample = next(iter(res.values()))
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            for k, v in res.items():
                if str(k) in names_key:
                    out[str(k)] = (float(v[0]), float(v[1]))
        elif "longitudes" in res and "velocities" in res:
            for nm in names_key:
                out[nm] = (float(res["longitudes"][nm]), float(res["velocities"][nm]))
        else:
            return None
    elif isinstance(res, (list, tuple)):
        for i, nm in enumerate(names_key[:len(res)]):
            pair = res[i]
            if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
                return None
            out[nm] = (float(pair[0]), float(pair[1]))
    else:
        return None

    return out, _adapter_source_tag()

def _longitudes_and_speeds(
    jd_tt: float,
    names: List[str],
    *,
    topocentric: bool,
    latitude: Optional[float],
    longitude: Optional[float],
    elevation_m: Optional[float],
    speed_step_days: float,
) -> Tuple[Dict[str, Tuple[float, float]], str]:
    """Get (lon, speed) per body. Prefer adapter velocities; else FD using cached longitudes."""
    names_key = tuple(names)
    jd_tt_q = _q(jd_tt, _JD_QUANT)
    lat_q = _q(latitude, _LATLON_QUANT) if topocentric else None
    lon_q = _q(longitude, _LATLON_QUANT) if topocentric else None
    elev_q = _q(elevation_m, _ELEV_QUANT) if topocentric else None

    # 1) Try velocities directly (fastest & most precise)
    vres = _longitudes_with_velocity_if_available(jd_tt_q, names_key, topocentric, lat_q, lon_q, elev_q)
    if vres is not None:
        return vres

    # 2) FD path (3 cached calls)
    now_map, source = _cached_longitudes(jd_tt_q, names_key, topocentric, lat_q, lon_q, elev_q)
    if speed_step_days and speed_step_days > 0:
        jm = _q(jd_tt - speed_step_days, _JD_QUANT)
        jp = _q(jd_tt + speed_step_days, _JD_QUANT)
        minus_map, _ = _cached_longitudes(jm, names_key, topocentric, lat_q, lon_q, elev_q)
        plus_map, _  = _cached_longitudes(jp, names_key, topocentric, lat_q, lon_q, elev_q)
        dt = (speed_step_days * 2.0)
    else:
        minus_map = plus_map = now_map
        dt = 1.0

    out: Dict[str, Tuple[float, float]] = {}
    for nm in names_key:
        l0 = _norm360(float(now_map[nm]))
        l_m = _norm360(float(minus_map[nm]))
        l_p = _norm360(float(plus_map[nm]))
        speed = _shortest_signed_delta_deg(l_p, l_m) / dt
        out[nm] = (l0, speed)
    return out, source

# ───────────────────────────── Main API ─────────────────────────────
def compute_chart(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gold-standard planetary chart computation.
    - Planets at JD_TT (ecliptic-of-date).
    - Sidereal = tropical − ayanamsa (pluggable; explicit degrees or named model).
    - Speeds from adapter if available; else cached finite difference.
    - Strict validation, robust geo handling for topocentric, friendly errors.

    Inputs (dict):
      date, time, place_tz,
      latitude, longitude, elevation_m?,
      mode: 'tropical'|'sidereal',
      ayanamsa: str|float?,
      jd_ut, jd_tt, jd_ut1 (preferred) or resolvable via time_kernel,
      bodies?: [names], topocentric?: bool

    Output (dict):
      {
        'jd_ut': float, 'jd_tt': float,
        'meta': { 'mode', 'ayanamsa_deg'|None, 'frame', 'observer', 'source', 'warnings'?: [...] },
        'bodies': [ { 'name', 'longitude_deg', 'speed_deg_per_day' }, ... ]
      }

    Raises AstronomyError for predictable error codes.
    """
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")

    # Validate high-level inputs & get bodies
    mode, bodies = _validate_payload(payload)

    # Timescales (strict: TT for planets)
    jd_ut, jd_tt, jd_ut1 = _ensure_timescales(payload)  # noqa: F841 (jd_ut1 kept for parity/logs)

    # Geometry (topocentric handling)
    topocentric = _coerce_bool(payload.get("topocentric"), False)
    warnings: List[str] = []
    if topocentric:
        lat, lon, elev, w, downgraded = _validate_and_normalize_geo_for_topo(
            payload.get("latitude"), payload.get("longitude"), payload.get("elevation_m")
        )
        warnings.extend(w)
        if downgraded:
            topocentric = False
            # fall back to geocentric silently; keep warning in meta
            lat = lon = elev = None
    else:
        lat = lon = elev = None  # ensure geo doesn't leak into cache keys

    # Ephemeris (longitudes + speeds)
    results, source_tag = _longitudes_and_speeds(
        jd_tt, bodies,
        topocentric=topocentric,
        latitude=lat, longitude=lon, elevation_m=elev,
        speed_step_days=_SPEED_STEP,
    )

    # Sidereal offset (if requested)
    ay_deg: Optional[float] = None
    ay_note: Optional[str] = None
    if mode == "sidereal":
        ay_deg, ay_note = _resolve_ayanamsa(jd_tt, payload.get("ayanamsa"))

    # Build bodies output
    out_bodies: List[Dict[str, Any]] = []
    for nm in bodies:
        lon_deg, speed = results[nm]
        if mode == "sidereal":
            lon_deg = _norm360(lon_deg - float(ay_deg))  # apply ayanamsa
        out_bodies.append({
            "name": nm,
            "longitude_deg": float(_norm360(lon_deg)),
            "speed_deg_per_day": float(speed),
        })

    meta: Dict[str, Any] = {
        "mode": mode,
        "ayanamsa_deg": float(ay_deg) if ay_deg is not None else None,
        "frame": "ecliptic-of-date",
        "observer": "topocentric" if topocentric else "geocentric",
        "source": str(source_tag),
    }
    if warnings:
        meta["warnings"] = warnings
    if ay_note and "fallback" in ay_note:
        meta.setdefault("warnings", []).append(ay_note)

    return {
        "jd_ut": float(jd_ut),
        "jd_tt": float(jd_tt),
        "meta": meta,
        "bodies": out_bodies,
    }
