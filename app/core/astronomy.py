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


# ───────────────────────────── Resilient imports ─────────────────────
try:
    from app.core import ephemeris_adapter as eph
except Exception as e:
    eph = None
    _EPH_IMPORT_ERROR = e

try:
    # single SoT for JD conversions (only used when payload doesn't pass jd_* directly)
    from app.core import time_kernel as _tk  # type: ignore
except Exception:
    _tk = None


# ───────────────────────────── Config & constants ─────────────────────
ALLOWED_BODIES = {
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
}
_DEF_BODIES: Tuple[str, ...] = tuple(ALLOWED_BODIES)

_JD_QUANT       = float(os.getenv("OCP_ASTRO_JD_QUANT", "1e-7"))     # ~0.009 s
_LATLON_QUANT   = float(os.getenv("OCP_ASTRO_LL_QUANT", "1e-6"))     # ~0.11 m
_ELEV_QUANT     = float(os.getenv("OCP_ASTRO_ELEV_QUANT", "0.1"))    # 10 cm
_SPEED_STEP     = float(os.getenv("OCP_SPEED_FD_STEP_DAYS", "0.5"))  # ±12h
_DEF_AYANAMSA   = os.getenv("OCP_AYANAMSA_DEFAULT", "lahiri").strip().lower()

_GEO_SOFT_LAT     = float(os.getenv("OCP_GEO_SOFT_LAT", "89.5"))
_GEO_HARD_LAT     = float(os.getenv("OCP_GEO_HARD_LAT", "89.9"))
_ELEV_MIN         = float(os.getenv("OCP_GEO_ELEV_MIN", "-500.0"))
_ELEV_MAX         = float(os.getenv("OCP_GEO_ELEV_MAX", "10000.0"))
_ELEV_WARN        = float(os.getenv("OCP_GEO_ELEV_WARN", "3000.0"))
_ANTIMER_WARN_LON = float(os.getenv("OCP_GEO_ANTI_WARN", "179.9"))


# ───────────────────────────── Helpers ─────────────────────────────
def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    if r < 0.0:
        r += 360.0
    return r

def _shortest_signed_delta_deg(a2: float, a1: float) -> float:
    d = (a2 - a1 + 540.0) % 360.0 - 180.0
    return -180.0 if d == 180.0 else d

def _coerce_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"): return True
        if s in ("0", "false", "f", "no", "n", "off"): return False
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


# ───────────────────────────── Geographic sanity for topocentric ─────
def _validate_and_normalize_geo_for_topo(
    lat: Any, lon: Any, elev: Any
) -> Tuple[float, float, Optional[float], List[str], bool]:
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
        latf = math.copysign((_GEO_HARD_LAT - 0.05), latf)
        warnings.append("latitude_soft_nudged_from_pole")

    if abs(lonf) >= _ANTIMER_WARN_LON:
        warnings.append("near_antimeridian_longitude")

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


# ───────────────────────────── Ayanāṁśa resolver ─────────────────────
@lru_cache(maxsize=4096)
def _ayanamsa_deg_cached(jd_tt_q: float, ay_key: str) -> Tuple[float, str]:
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

    # Fallback: linearized model near J2000 for common variants
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
    if ayanamsa is None or (isinstance(ayanamsa, str) and not ayanamsa.strip()):
        key = _DEF_AYANAMSA
    elif isinstance(ayanamsa, (int, float)):
        return float(ayanamsa), "explicit"
    else:
        key = str(ayanamsa).strip().lower()
    jd_q = _q(jd_tt, _JD_QUANT)
    ay, note = _ayanamsa_deg_cached(jd_q, key)
    return float(ay), note


# ───────────────────────────── Ephemeris adapter calls (cached) ───────
def _adapter_source_tag() -> str:
    return getattr(eph, "current_kernel_name", None) or getattr(eph, "EPHEMERIS_NAME", None) or "adapter"

def _adapter_callable(*names: str):
    for n in names:
        fn = getattr(eph, n, None)
        if callable(fn):
            return fn
    return None

def _geo_kwargs_for_sig(sig: inspect.Signature, *, topocentric: bool, lat_q, lon_q, elev_q) -> Dict[str, Any]:
    """Map our geo/topo to whatever parameter names the adapter exposes."""
    kw: Dict[str, Any] = {}
    if "topocentric" in sig.parameters:
        kw["topocentric"] = topocentric
    if topocentric:
        if "latitude" in sig.parameters and lat_q is not None: kw["latitude"] = float(lat_q)
        if "longitude" in sig.parameters and lon_q is not None: kw["longitude"] = float(lon_q)
        if "elevation_m" in sig.parameters and elev_q is not None: kw["elevation_m"] = float(elev_q)
        if "lat" in sig.parameters and lat_q is not None: kw["lat"] = float(lat_q)
        if "lon" in sig.parameters and lon_q is not None: kw["lon"] = float(lon_q)
        if "phi" in sig.parameters and lat_q is not None: kw["phi"] = float(lat_q)
        if "lam" in sig.parameters and lon_q is not None: kw["lam"] = float(lon_q)
        if "lambda" in sig.parameters and lon_q is not None: kw["lambda"] = float(lon_q)
    return kw


def _normalize_adapter_output_to_maps(
    res: Any, names_key: Tuple[str, ...]
) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
    """
    Normalize various adapter return styles into:
      (longitudes_map {name: deg}, speeds_map {name: deg/day or None})
    """
    longitudes: Dict[str, float] = {}
    speeds: Dict[str, Optional[float]] = {}

    # Case A: dict formats
    if isinstance(res, dict):
        # A1: {name: deg}
        if all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in res.items()):
            for nm in names_key:
                if nm in res:
                    longitudes[nm] = float(res[nm])
                    speeds[nm] = None
            return longitudes, speeds
        # A2: {"longitudes": {...}, "velocities": {...}}
        if "longitudes" in res and isinstance(res["longitudes"], dict):
            for nm in names_key:
                if nm in res["longitudes"]:
                    longitudes[nm] = float(res["longitudes"][nm])
                    speeds[nm] = None
            if "velocities" in res and isinstance(res["velocities"], dict):
                for nm in names_key:
                    if nm in res["velocities"]:
                        speeds[nm] = float(res["velocities"][nm])
            return longitudes, speeds

    # Case B: list/tuple formats
    if isinstance(res, (list, tuple)):
        # B1: list of dicts with "name","lon","speed"
        if len(res) and isinstance(res[0], dict):
            tmp_lon: Dict[str, float] = {}
            tmp_spd: Dict[str, Optional[float]] = {}
            for row in res:
                try:
                    nm = str(row.get("name"))
                    if not nm:
                        continue
                    if row.get("lon") is not None:
                        lonv = float(row["lon"])
                    elif row.get("longitude") is not None:
                        lonv = float(row["longitude"])
                    elif row.get("longitude_deg") is not None:
                        lonv = float(row["longitude_deg"])
                    else:
                        continue
                    tmp_lon[nm] = lonv
                    sp = row.get("speed")
                    if sp is None: sp = row.get("speed_deg_per_day")
                    tmp_spd[nm] = float(sp) if sp is not None else None
                except Exception:
                    continue
            for nm in names_key:
                if nm in tmp_lon:
                    longitudes[nm] = float(tmp_lon[nm])
                    speeds[nm] = tmp_spd.get(nm, None)
            return longitudes, speeds
        # B2: aligned numeric list with names
        for i, nm in enumerate(names_key[:len(res)]):
            longitudes[nm] = float(res[i])
            speeds[nm] = None
        return longitudes, speeds

    # Anything else → unsupported
    raise AstronomyError("adapter_return_invalid", f"Unsupported adapter return type: {type(res)}")


@lru_cache(maxsize=8192)
def _cached_positions(
    jd_tt_q: float,
    names_key: Tuple[str, ...],
    topocentric: bool,
    lat_q: Optional[float],
    lon_q: Optional[float],
    elev_q: Optional[float],
) -> Tuple[Dict[str, float], Dict[str, Optional[float]], str]:
    """
    LRU-cached call to adapter.
    Returns (longitudes_map, speeds_map, source_tag).
    """
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")

    fn = _adapter_callable(
        "get_ecliptic_longitudes",
        "ecliptic_longitudes",
        "planetary_longitudes",
    )
    if fn is None:
        raise AstronomyError("adapter_api_mismatch",
                             "ephemeris_adapter missing any of: get_ecliptic_longitudes/ecliptic_longitudes/planetary_longitudes")

    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}

    # Time
    if "jd_tt" in sig.parameters: kwargs["jd_tt"] = jd_tt_q
    elif "jd" in sig.parameters:  kwargs["jd"] = jd_tt_q

    # Names if supported
    if "names" in sig.parameters:
        kwargs["names"] = list(names_key)

    # Frame if adapter exposes
    if "frame" in sig.parameters:
        kwargs["frame"] = "ecliptic-of-date"

    kwargs.update(_geo_kwargs_for_sig(sig,
        topocentric=topocentric, lat_q=lat_q, lon_q=lon_q, elev_q=elev_q
    ))

    try:
        res = fn(**kwargs) if kwargs else fn(jd_tt_q)
    except TypeError:
        res = fn(jd_tt_q)

    lon_map, spd_map = _normalize_adapter_output_to_maps(res, names_key)
    return lon_map, spd_map, _adapter_source_tag()


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
    """
    Get (lon, speed) per body.
    Prefer adapter speeds when available; otherwise compute via FD using cached positions at jd±step.
    """
    names_key = tuple(names)
    jd_tt_q = _q(jd_tt, _JD_QUANT)
    lat_q = _q(latitude, _LATLON_QUANT) if topocentric else None
    lon_q = _q(longitude, _LATLON_QUANT) if topocentric else None
    elev_q = _q(elevation_m, _ELEV_QUANT) if topocentric else None

    now_lon, now_spd, source = _cached_positions(jd_tt_q, names_key, topocentric, lat_q, lon_q, elev_q)

    if all(now_spd.get(nm) is not None for nm in names_key):
        return {nm: (_norm360(float(now_lon[nm])), float(now_spd[nm])) for nm in names_key}, source

    if speed_step_days and speed_step_days > 0:
        jm = _q(jd_tt - speed_step_days, _JD_QUANT)
        jp = _q(jd_tt + speed_step_days, _JD_QUANT)
        minus_lon, _, _ = _cached_positions(jm, names_key, topocentric, lat_q, lon_q, elev_q)
        plus_lon,  _, _ = _cached_positions(jp, names_key, topocentric, lat_q, lon_q, elev_q)
        dt = (speed_step_days * 2.0)
    else:
        minus_lon = plus_lon = now_lon
        dt = 1.0

    out: Dict[str, Tuple[float, float]] = {}
    for nm in names_key:
        l0 = _norm360(float(now_lon[nm]))
        l_m = _norm360(float(minus_lon[nm]))
        l_p = _norm360(float(plus_lon[nm]))
        spd = _shortest_signed_delta_deg(l_p, l_m) / dt
        out[nm] = (l0, spd)
    return out, source


# ───────────────────────────── Angles (Asc/MC) ────────────────────────
def _mean_obliquity_deg(jd_tt: float) -> float:
    """IAU 2006/2000A mean obliquity (arcsec poly trimmed; accurate to ~mas scale)."""
    T = (float(jd_tt) - 2451545.0) / 36525.0
    # Meeus 2000, Eq. 22.3 (sufficient here)
    eps_arcsec = 84381.448 - 46.8150*T - 0.00059*(T**2) + 0.001813*(T**3)
    return eps_arcsec / 3600.0

def _gmst_deg_from_jd_ut1(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time in degrees (IAU 1982-ish, good to <0.1s for our use)."""
    T = (float(jd_ut1) - 2451545.0) / 36525.0
    theta = 280.46061837 + 360.98564736629 * (float(jd_ut1) - 2451545.0) \
            + 0.000387933*(T**2) - (T**3)/38710000.0
    return _norm360(theta)

def _asc_mc_from_sidereal(lst_deg: float, lat_deg: float, eps_deg: float) -> Tuple[float, float]:
    """
    Compute (Asc, MC) ecliptic longitudes from local sidereal angle, latitude and obliquity.
    Formulas follow standard spherical astronomy identities.
    """
    # convert to radians
    th  = math.radians(_norm360(lst_deg))
    phi = math.radians(max(-90.0, min(90.0, float(lat_deg))))
    eps = math.radians(float(eps_deg))

    # Midheaven: λ_MC = atan2(sin θ, cos θ * cos ε)
    num_mc = math.sin(th)
    den_mc = math.cos(th) * math.cos(eps)
    lam_mc = math.degrees(math.atan2(num_mc, den_mc))
    lam_mc = _norm360(lam_mc)

    # Ascendant: λ_ASC = atan2(-cos θ, sin θ * cos ε + tan φ * sin ε)
    num_asc = -math.cos(th)
    den_asc = (math.sin(th) * math.cos(eps)) + (math.tan(phi) * math.sin(eps))
    lam_asc = math.degrees(math.atan2(num_asc, den_asc))
    lam_asc = _norm360(lam_asc)

    return lam_asc, lam_mc

def _compute_angles_best_effort(
    jd_ut1: float,
    jd_tt: float,
    latitude: Optional[float],
    longitude: Optional[float],
    *,
    mode: str,
    ayanamsa_deg: Optional[float],
) -> Optional[Tuple[float, float]]:
    """
    Best-effort Asc/MC computation.
    1) Pure-Python fallback (GMST+ε) → preferred (no extra deps).
    2) As a last resort, attempt houses engine to read asc/mc (kept lazy to avoid cycles).
    Returns angles in the chart's zodiac (tropical or sidereal), or None on failure.
    """
    try:
        if latitude is None or longitude is None:
            return None

        # Local sidereal angle (degrees)
        gmst = _gmst_deg_from_jd_ut1(jd_ut1)     # Greenwich
        lst  = _norm360(gmst + float(longitude)) # Local (east positive)

        eps  = _mean_obliquity_deg(jd_tt)
        asc, mc = _asc_mc_from_sidereal(lst, float(latitude), eps)

        # Rotate for sidereal charts
        if mode == "sidereal" and ayanamsa_deg is not None:
            asc = _norm360(asc - float(ayanamsa_deg))
            mc  = _norm360(mc  - float(ayanamsa_deg))
        return float(asc), float(mc)
    except Exception:
        pass

    # Last resort: borrow angles from houses engine (only for numbers, not ideal for parity)
    try:
        # local import to avoid import cycles on module load
        try:
            from app.core.house import compute_houses_with_policy as _houses_fn  # type: ignore
        except Exception:
            from app.core.houses_advanced import compute_house_system as _houses_fn  # type: ignore

        kwargs: Dict[str, Any] = {}
        # use whatever args the function accepts
        pars = inspect.signature(_houses_fn).parameters
        if "lat" in pars: kwargs["lat"] = float(latitude)
        if "lon" in pars: kwargs["lon"] = float(longitude)
        if "latitude" in pars: kwargs["latitude"] = float(latitude)
        if "longitude" in pars: kwargs["longitude"] = float(longitude)
        if "jd_tt" in pars: kwargs["jd_tt"] = float(jd_tt)
        if "jd_ut1" in pars: kwargs["jd_ut1"] = float(jd_ut1)
        if "jd_ut" in pars and "jd_tt" not in pars and "jd_ut1" not in pars:
            kwargs["jd_ut"] = float(jd_ut1)  # close enough for engines expecting UT as UT1
        if "mode" in pars: kwargs["mode"] = mode

        H = _houses_fn(**kwargs)
        if isinstance(H, dict):
            asc = H.get("asc_deg", H.get("asc"))
            mc  = H.get("mc_deg", H.get("mc"))
            if isinstance(asc, (int,float)) and isinstance(mc, (int,float)):
                return float(asc), float(mc)
    except Exception:
        pass

    return None


# ───────────────────────────── Main API ───────────────────────────────
def compute_chart(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gold-standard planetary chart computation.
    - Planets at JD_TT (ecliptic-of-date).
    - Sidereal = tropical − ayanamsa (pluggable; explicit degrees or named model).
    - Adapter speeds preferred; FD speeds as fallback.
    - Emits angles (Asc/MC) with a pure-python fallback.
    - Strict validation, robust geo handling for topocentric, friendly errors.
    """
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")

    # Validate high-level inputs & get bodies
    mode, bodies = _validate_payload(payload)

    # Timescales (TT for planets, UT1 for angles/LST)
    jd_ut, jd_tt, jd_ut1 = _ensure_timescales(payload)  # noqa: F841 (jd_ut kept for completeness)

    # Geometry (topocentric handling for planets; angles need lat/lon regardless)
    topocentric = _coerce_bool(payload.get("topocentric"), False)
    warnings: List[str] = []

    lat: Optional[float]
    lon: Optional[float]
    elev: Optional[float]

    if topocentric:
        lat, lon, elev, w, downgraded = _validate_and_normalize_geo_for_topo(
            payload.get("latitude"), payload.get("longitude"), payload.get("elevation_m")
        )
        warnings.extend(w)
        if downgraded:
            topocentric = False
            lat = lon = elev = None
    else:
        # still keep raw lat/lon for angles (not quantized), if present
        lat = float(payload.get("latitude")) if _is_finite(payload.get("latitude")) else None
        lon = float(payload.get("longitude")) if _is_finite(payload.get("longitude")) else None
        elev = None  # not used for planetary calc when geocentric

    # Ephemeris (longitudes + speeds)
    results, source_tag = _longitudes_and_speeds(
        jd_tt, bodies,
        topocentric=topocentric,
        latitude=lat, longitude=lon, elevation_m=elev,
        speed_step_days=_SPEED_STEP,
    )

    # Ayanāṁśa (if sidereal)
    ay_deg: Optional[float] = None
    ay_note: Optional[str] = None
    if mode == "sidereal":
        ay_deg, ay_note = _resolve_ayanamsa(jd_tt, payload.get("ayanamsa"))

    # Build bodies output (apply ayanāṁśa if needed)
    out_bodies: List[Dict[str, Any]] = []
    for nm in bodies:
        lon_deg, speed = results[nm]
        if mode == "sidereal" and ay_deg is not None:
            lon_deg = _norm360(lon_deg - float(ay_deg))
        out_bodies.append({
            "name": nm,
            "longitude_deg": float(_norm360(lon_deg)),
            "speed_deg_per_day": float(speed),
        })

    # Angles (Asc/MC) — try best-effort computation
    asc_mc = _compute_angles_best_effort(
        jd_ut1=jd_ut1, jd_tt=jd_tt, latitude=lat, longitude=lon,
        mode=mode, ayanamsa_deg=ay_deg
    )
    angles: Optional[Dict[str, float]] = None
    if asc_mc is not None:
        angles = {"asc_deg": float(asc_mc[0]), "mc_deg": float(asc_mc[1])}

    # Meta
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

    out: Dict[str, Any] = {
        "jd_ut": float(jd_ut),
        "jd_tt": float(jd_tt),
        "meta": meta,
        "bodies": out_bodies,
    }
    if angles:
        out["angles"] = angles
        out["asc_deg"] = angles["asc_deg"]
        out["mc_deg"]  = angles["mc_deg"]

    return out
