# -*- coding: utf-8 -*-
"""
High-precision planetary chart computation (ecliptic-of-date).

New in this version
-------------------
- Accepts optional `points` (e.g., ["North Node","South Node"]) and returns
  BOTH majors (in `bodies`) and nodes (in `points`) in the same response.
- If client puts node names in `bodies`, we move them to `points` internally
  and still return majors (default 10 if the majors subset ends up empty).
- Points are tagged with `is_point: True` and `speed/speed_deg_per_day: None`.

Key points
----------
- Timescales: prefer caller-supplied jd_ut/jd_tt/jd_ut1; else derive from
  app.core.time_kernel → app.core.timescales → stdlib fallback.
- Ephemeris: delegate positions to app.core.ephemeris_adapter (Skyfield DE421).
- Angles: ERFA (IAU 2006/2000A) when available; Meeus-grade fallback otherwise.
- Sidereal: rotate all longitudes (planets + angles + points) by ayanāṁśa.
- Topocentric: supported for bodies via adapter; points (nodes) are geocentric.
- Schema: predictable output; legacy keys (lon/speed) retained for compatibility.

Public API
----------
compute_chart(payload: dict) -> dict
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Callable
import math
import os
import inspect
from functools import lru_cache

__all__ = ["compute_chart"]


# ───────────────────────────── Exceptions ─────────────────────────────
class AstronomyError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


# ───────────────────────────── Resilient imports ─────────────────────
try:
    from app.core import ephemeris_adapter as eph  # primary source of positions
except Exception as e:  # pragma: no cover
    eph = None
    _EPH_IMPORT_ERROR = e

try:
    from app.core import time_kernel as _tk  # single source of truth if present
except Exception:  # pragma: no cover
    _tk = None

try:
    from app.core import timescales as _ts  # civil→JD helpers
except Exception:  # pragma: no cover
    _ts = None

try:
    import erfa  # PyERFA (SOFA)
except Exception as e:  # pragma: no cover
    erfa = None
    _ERFA_IMPORT_ERROR = e


# ───────────────────────────── Config & constants ─────────────────────
_CLASSIC_10 = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
)
# Present in adapter but not used by app right now; kept for forward-compat.
_EXTRA_ALLOWED = (
    "Ceres", "Pallas", "Juno", "Vesta", "Chiron",
    "North Node", "South Node",
)

ALLOWED_BODIES = set(_CLASSIC_10) | set(_EXTRA_ALLOWED)
_DEF_BODIES: Tuple[str, ...] = _CLASSIC_10

# Node canonical names (case-insensitive matching)
_NODE_CANON = {
    "north node": "North Node",
    "south node": "South Node",
}
_NODE_SET_LC = set(_NODE_CANON.keys())

_JD_QUANT       = float(os.getenv("OCP_ASTRO_JD_QUANT", "1e-7"))      # ~0.009 s
_LATLON_QUANT   = float(os.getenv("OCP_ASTRO_LL_QUANT", "1e-6"))      # ~0.11 m
_ELEV_QUANT     = float(os.getenv("OCP_GEO_ELEV_QUANT", "0.1"))       # 10 cm
_SPEED_STEP_DEF = float(os.getenv("OCP_SPEED_FD_STEP_DAYS", "0.25"))  # ±6 h
_DEF_AYANAMSA   = os.getenv("OCP_AYANAMSA_DEFAULT", "lahiri").strip().lower()

_GEO_SOFT_LAT     = float(os.getenv("OCP_GEO_SOFT_LAT", "89.5"))
_GEO_HARD_LAT     = float(os.getenv("OCP_GEO_HARD_LAT", "89.9"))
_ELEV_MIN         = float(os.getenv("OCP_GEO_ELEV_MIN", "-500.0"))
_ELEV_MAX         = float(os.getenv("OCP_GEO_ELEV_MAX", "10000.0"))
_ELEV_WARN        = float(os.getenv("OCP_GEO_ELEV_WARN", "3000.0"))
_ANTIMER_WARN_LON = float(os.getenv("OCP_GEO_ANTI_WARN", "179.9"))

_DUT1_SECONDS = float(
    os.getenv("ASTRO_DUT1_BROADCAST", os.getenv("OCP_DUT1_SECONDS", "0.0"))
)

_PROJECT_SOURCE_TAG = "astronomy(core)"


# ───────────────────────────── Small helpers ──────────────────────────
def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    if r < 0.0:
        r += 360.0
    return 0.0 if abs(r) < 1e-12 else r


def _wrap180(x: float) -> float:
    return ((float(x) + 180.0) % 360.0) - 180.0


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


# ───────────────────────────── Validation & parsing ───────────────────
def _validate_mode(payload: Dict[str, Any]) -> str:
    mode = str(payload.get("mode", "tropical")).strip().lower()
    if mode not in ("tropical", "sidereal"):
        raise AstronomyError("invalid_input", "mode must be 'tropical' or 'sidereal'")
    return mode


def _canon_node_name(s: str) -> Optional[str]:
    key = str(s).strip().lower()
    return _NODE_CANON.get(key)


def _split_bodies_points(
    payload: Dict[str, Any]
) -> Tuple[List[str], List[str], bool, List[str]]:
    """
    Returns: (majors, points, bodies_were_omitted, warnings)
    - Move any node names from `bodies` → `points`.
    - If bodies explicitly provided and become empty after moving nodes,
      we fall back to default 10 majors.
    """
    warnings: List[str] = []

    # Parse requested bodies
    bodies_raw = payload.get("bodies", None)
    bodies_were_omitted = bodies_raw is None
    if bodies_raw is None:
        majors: List[str] = list(_DEF_BODIES)
    else:
        try:
            majors = [str(b) for b in bodies_raw]
        except Exception:
            raise AstronomyError("invalid_input", "bodies must be a list of names")

    # Parse requested points
    points_raw = payload.get("points", [])
    points: List[str] = []
    if points_raw is None:
        points_raw = []
    if isinstance(points_raw, (list, tuple)):
        for p in points_raw:
            nm = _canon_node_name(p) or None
            if nm:
                points.append(nm)
            else:
                # Only nodes supported as "points" in this version; ignore others silently.
                pass
    else:
        warnings.append("points_ignored_non_list")

    # Move any node names present in `majors` to `points`
    majors_out: List[str] = []
    for b in majors:
        if b not in ALLOWED_BODIES:
            allowed = ", ".join(sorted(ALLOWED_BODIES))
            raise AstronomyError("unsupported_body", f"'{b}' not supported (allowed: {allowed})")
        b_lc = str(b).strip().lower()
        if b_lc in _NODE_SET_LC:
            canon = _canon_node_name(b_lc)
            if canon and canon not in points:
                points.append(canon)
            # Do not keep in majors
        else:
            majors_out.append(b)

    # If the client explicitly set bodies but after moving nodes none remain,
    # return the default 10 majors (acceptance test D).
    if not bodies_were_omitted and len(majors_out) == 0:
        majors_out = list(_DEF_BODIES)

    # Deduplicate stable order for points
    points_seen = set()
    points_final: List[str] = []
    for p in points:
        if p not in points_seen:
            points_seen.add(p)
            points_final.append(p)

    return majors_out, points_final, bodies_were_omitted, warnings


# ───────────────────────────── Timescales ─────────────────────────────
def _ensure_timescales(payload: Dict[str, Any]) -> Tuple[float, float, float, List[str]]:
    """Return (jd_ut, jd_tt, jd_ut1, warnings)."""
    warnings: List[str] = []

    jd_ut  = payload.get("jd_ut")
    jd_tt  = payload.get("jd_tt")
    jd_ut1 = payload.get("jd_ut1")

    if all(isinstance(x, (int, float)) for x in (jd_ut, jd_tt, jd_ut1)):
        return float(jd_ut), float(jd_tt), float(jd_ut1), warnings

    # 1) project time_kernel
    if _tk is not None:
        for fname in ("timescales_from_civil", "compute_timescales", "build_timescales",
                      "to_timescales", "from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                out = fn(
                    payload.get("date"),
                    payload.get("time"),
                    payload.get("place_tz") or payload.get("tz") or "UTC",
                    payload.get("latitude"),
                    payload.get("longitude"),
                )
                return float(out["jd_ut"]), float(out["jd_tt"]), float(out["jd_ut1"]), warnings

    # 2) app.core.timescales or stdlib fallback
    date_str = payload.get("date"); time_str = payload.get("time")
    tz_str   = payload.get("place_tz") or payload.get("tz") or "UTC"
    if not isinstance(date_str, str) or not isinstance(time_str, str):
        missing = [k for k, v in (("jd_ut", jd_ut), ("jd_tt", jd_tt), ("jd_ut1", jd_ut1)) if not isinstance(v, (int, float))]
        raise AstronomyError("timescales_missing", f"Supply {', '.join(missing)} or provide date/time/tz")

    def _jd_utc_via_ts(d: str, t: str, z: str) -> float:
        if _ts is None:
            raise RuntimeError("timescales module not available")
        return float(_ts.julian_day_utc(d, t, z))

    def _jd_utc_via_stdlib(d: str, t: str, z: str) -> float:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        parts = t.split(":")
        timestr = t if len(parts) >= 3 else (t + ":00")
        dt_local = datetime.fromisoformat(f"{d}T{timestr}").replace(tzinfo=ZoneInfo(z))
        dt_utc = dt_local.astimezone(timezone.utc)
        # Meeus JD (UTC)
        Y = dt_utc.year; M = dt_utc.month; D = dt_utc.day
        h = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600 + dt_utc.microsecond/3.6e9
        if M <= 2:
            Y -= 1; M += 12
        A = Y // 100
        B = 2 - A + A // 4
        JD0 = int(365.25*(Y + 4716)) + int(30.6001*(M + 1)) + D + B - 1524.5
        return JD0 + h/24.0

    used_stdlib = False
    try:
        jd_utc = _jd_utc_via_ts(date_str, time_str, tz_str)
    except Exception:
        jd_utc = _jd_utc_via_stdlib(date_str, time_str, tz_str)
        used_stdlib = True
        warnings.append("timescales_fallback_local_utc_jd")

    try:
        y, m = map(int, str(date_str).split("-")[:2])
    except Exception:
        y, m = 2000, 1

    if _ts is not None:
        try:
            jd_tt_calc = float(_ts.jd_tt_from_utc_jd(jd_utc, y, m))
        except Exception:
            jd_tt_calc = jd_utc + 69.0/86400.0
            warnings.append("deltaT_fallback_69s")
    else:
        jd_tt_calc = jd_utc + 69.0/86400.0
        if not used_stdlib:
            warnings.append("deltaT_fallback_69s")

    jd_ut_calc  = jd_utc
    jd_ut1_calc = jd_utc + (_DUT1_SECONDS / 86400.0)

    return jd_ut_calc, jd_tt_calc, jd_ut1_calc, warnings


# ───────────────────────────── Geo / Topocentric ──────────────────────
def _normalize_lon180(lon: float) -> float:
    return _wrap180(lon)


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

    if elev is None or (isinstance(elev, str) and str(elev).strip() == ""):
        elev_m: Optional[float] = None
    else:
        if not _is_finite(elev):
            raise AstronomyError("invalid_input", "elevation_m must be a finite number in meters")
        elev_m = float(elev)
        if elev_m < _ELEV_MIN:
            warnings.append("elevation_clamped_min"); elev_m = _ELEV_MIN
        elif elev_m > _ELEV_MAX:
            warnings.append("elevation_clamped_max"); elev_m = _ELEV_MAX
        elif abs(elev_m) >= _ELEV_WARN:
            warnings.append("very_high_elevation_site")

    return latf, lonf, elev_m, warnings, downgraded


# ───────────────────────────── Ayanāṁśa ───────────────────────────────
@lru_cache(maxsize=4096)
def _ayanamsa_deg_cached(jd_tt_q: float, ay_key: str) -> Tuple[float, str]:
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

    # Linearized fallback around J2000
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


def _resolve_ayanamsa(jd_tt: float, ayanamsa: Any, warnings: List[str]) -> Tuple[Optional[float], Optional[str]]:
    if ayanamsa is None or (isinstance(ayanamsa, str) and not str(ayanamsa).strip()):
        key = _DEF_AYANAMSA
    elif isinstance(ayanamsa, (int, float)):
        return float(ayanamsa), "explicit"
    else:
        key = str(ayanamsa).strip().lower()
    jd_q = _q(jd_tt, _JD_QUANT) or jd_tt
    ay, note = _ayanamsa_deg_cached(jd_q, key)
    if note and "fallback" in note:
        warnings.append(note)
    return float(ay), note


# ───────────────────────────── Adapter I/O ────────────────────────────
def _adapter_source_tag() -> str:
    tag = getattr(eph, "current_kernel_name", None) \
          or getattr(eph, "EPHEMERIS_NAME", None) or "adapter"
    return str(tag)


def _adapter_callable(*names: str) -> Optional[Callable[..., Any]]:
    for n in names:
        fn = getattr(eph, n, None)
        if callable(fn):
            return fn
    return None


def _geo_kwargs_for_sig(sig: inspect.Signature, *, topocentric: bool, lat_q, lon_q, elev_q) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if "topocentric" in sig.parameters:
        kw["topocentric"] = topocentric
    if topocentric:
        if "latitude" in sig.parameters and lat_q is not None: kw["latitude"] = float(lat_q)
        if "longitude" in sig.parameters and lon_q is not None: kw["longitude"] = float(lon_q)
        if "elevation_m" in sig.parameters and elev_q is not None: kw["elevation_m"] = float(elev_q)
        if "lat" in sig.parameters and lat_q is not None: kw["lat"] = float(lat_q)
        if "lon" in sig.parameters and lon_q is not None: kw["lon"] = float(lon_q)
    return kw


def _normalize_adapter_output_to_maps(
    res: Any, names_key: Tuple[str, ...]
) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
    """
    Normalize various adapter return shapes to:
      longitudes: {Name: deg}
      speeds:     {Name: deg/day or None}
    Accepts dict, list-of-dicts, list-of-tuples, or positional lists.
    """
    want = list(names_key)
    want_lc = [w.lower() for w in want]
    longitudes: Dict[str, float] = {}
    speeds: Dict[str, Optional[float]] = {}

    def _merge_numeric_map(src: Dict[Any, Any], *, into: str) -> None:
        src_lc = {str(k).lower(): v for k, v in src.items()}
        for nm, nm_lc in zip(want, want_lc):
            if nm_lc in src_lc and isinstance(src_lc[nm_lc], (int, float)):
                if into == "lon":
                    longitudes[nm] = float(src_lc[nm_lc]); speeds.setdefault(nm, None)
                else:
                    speeds[nm] = float(src_lc[nm_lc])

    # 1) Dict-like containers
    if isinstance(res, dict):
        # peel common wrappers
        for k in ("rows", "result", "data", "payload"):
            if k in res and isinstance(res[k], (list, tuple)):
                res = res[k]
                break
        else:
            # map {longitudes:{}, velocities:{}}
            for lon_key in ("longitudes", "longitude", "lon"):
                if lon_key in res and isinstance(res[lon_key], dict):
                    _merge_numeric_map(res[lon_key], into="lon")
                    break
            for spd_key in ("velocities", "velocity", "speeds", "speed"):
                if spd_key in res and isinstance(res[spd_key], dict):
                    _merge_numeric_map(res[spd_key], into="spd")
                    break
            if longitudes:
                return longitudes, speeds
            # flat {name:deg}
            if all(isinstance(k, (str, int)) and isinstance(v, (int, float)) for k, v in res.items()):
                _merge_numeric_map(res, into="lon")
                return longitudes, speeds
            # if nothing matched but dict contains a single list, fall through
            for v in res.values():
                if isinstance(v, (list, tuple)):
                    res = v
                    break

    # 2) List/tuple forms
    if isinstance(res, (list, tuple)):
        if not res:
            return longitudes, speeds

        # list of dict rows
        if isinstance(res[0], dict):
            tmp_lon_lc: Dict[str, float] = {}
            tmp_spd_lc: Dict[str, Optional[float]] = {}
            for row in res:
                try:
                    # name can be in 'name' or implied by 'body'
                    nm = row.get("name") or row.get("body") or ""
                    if not nm:
                        continue
                    nm_lc = str(nm).lower()

                    # longitude variants
                    if row.get("lon") is not None:
                        lonv = float(row["lon"])
                    elif row.get("longitude") is not None:
                        lonv = float(row["longitude"])
                    elif row.get("longitude_deg") is not None:
                        lonv = float(row["longitude_deg"])
                    elif row.get("lambda") is not None:
                        lonv = float(row["lambda"])
                    else:
                        continue

                    tmp_lon_lc[nm_lc] = lonv

                    # speed variants
                    sp = (
                        row.get("speed")
                        or row.get("velocity")
                        or row.get("speed_deg_per_day")
                        or row.get("lambda_dot")
                    )
                    tmp_spd_lc[nm_lc] = float(sp) if isinstance(sp, (int, float)) else None
                except Exception:
                    continue

            for nm, nm_lc in zip(want, want_lc):
                if nm_lc in tmp_lon_lc:
                    longitudes[nm] = float(tmp_lon_lc[nm_lc])
                    speeds[nm] = tmp_spd_lc.get(nm_lc, None)
            return longitudes, speeds

        # list of tuples: (name, lon[, speed])
        if isinstance(res[0], (list, tuple)) and len(res[0]) >= 2:
            tmp_lon: Dict[str, float] = {}
            tmp_spd: Dict[str, Optional[float]] = {}
            for item in res:
                try:
                    nm = str(item[0]); lonv = float(item[1])
                    tmp_lon[nm] = lonv
                    tmp_spd[nm] = float(item[2]) if len(item) >= 3 and isinstance(item[2], (int, float)) else None
                except Exception:
                    continue
            for nm in want:
                if nm in tmp_lon:
                    longitudes[nm] = float(tmp_lon[nm])
                    speeds[nm] = tmp_spd.get(nm, None)
            return longitudes, speeds

        # positional list of numbers → map by order
        if all(isinstance(x, (int, float)) for x in res):
            for i, nm in enumerate(want[:len(res)]):
                longitudes[nm] = float(res[i]); speeds[nm] = None
            return longitudes, speeds

        # list of objects with attributes
        first = res[0]
        if hasattr(first, "name"):
            tmp_lon: Dict[str, float] = {}
            tmp_spd: Dict[str, Optional[float]] = {}
            for obj in res:
                try:
                    nm = str(getattr(obj, "name"))
                    lonv = getattr(obj, "lon", None)
                    if lonv is None:
                        lonv = getattr(obj, "longitude", None)
                    if lonv is None:
                        lonv = getattr(obj, "longitude_deg", None)
                    if lonv is None:
                        continue
                    tmp_lon[nm] = float(lonv)
                    sp = getattr(obj, "speed", None) or getattr(obj, "velocity", None)
                    tmp_spd[nm] = float(sp) if isinstance(sp, (int, float)) else None
                except Exception:
                    continue
            for nm in want:
                if nm in tmp_lon:
                    longitudes[nm] = float(tmp_lon[nm])
                    speeds[nm] = tmp_spd.get(nm, None)
            return longitudes, speeds

    # 3) Fallback: attributes on object with .longitudes/.velocities
    longs = getattr(res, "longitudes", None)
    if isinstance(longs, dict):
        vels = getattr(res, "velocities", {}) or {}
        _merge_numeric_map(longs, into="lon")
        if isinstance(vels, dict):
            _merge_numeric_map(vels, into="spd")
        return longitudes, speeds

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
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")

    fn = _adapter_callable(
        "ecliptic_longitudes_and_velocities",
        "get_ecliptic_longitudes_and_velocities",
        "ecliptic_longitudes",
        "get_ecliptic_longitudes",
    )
    if fn is None:
        raise AstronomyError("adapter_api_mismatch", "ephemeris_adapter missing longitudes API")

    sig = inspect.signature(fn)

    base_kwargs: Dict[str, Any] = {}
    if "jd_tt" in sig.parameters:
        base_kwargs["jd_tt"] = jd_tt_q
    elif "jd" in sig.parameters:
        base_kwargs["jd"] = jd_tt_q
    if "frame" in sig.parameters:
        base_kwargs["frame"] = "ecliptic-of-date"

    base_kwargs.update(
        _geo_kwargs_for_sig(
            sig,
            topocentric=topocentric,
            lat_q=lat_q,
            lon_q=lon_q,
            elev_q=elev_q,
        )
    )

    attempts: List[Optional[List[str]]] = [None]
    if "names" in sig.parameters:
        attempts = [list(names_key), [n.lower() for n in names_key]]

    last_err: Optional[Exception] = None
    res: Any = None
    for name_list in attempts:
        kwargs = dict(base_kwargs)
        if name_list is not None:
            kwargs["names"] = name_list
        try:
            try:
                res = fn(**kwargs)
            except TypeError:
                res = fn(jd_tt_q)  # positional jd_tt only
            break
        except Exception as e:
            last_err = e
            continue

    if res is None:
        raise AstronomyError(
            "adapter_failed",
            f"ephemeris adapter failed for names {list(names_key)}; last error: {last_err!r}",
        )

    lon_map, spd_map = _normalize_adapter_output_to_maps(res, names_key)
    return lon_map, spd_map, _adapter_source_tag()


def _adaptive_speed_step(name: str, default_step: float) -> float:
    return 0.125 if name == "Moon" and default_step > 0.125 else default_step


def _longitudes_and_speeds(
    jd_tt: float,
    names: List[str],
    *,
    topocentric: bool,
    latitude: Optional[float],
    longitude: Optional[float],
    elevation_m: Optional[float],
    speed_step_days: float,
) -> Tuple[Dict[str, Tuple[float, Optional[float]]], str]:
    names_key = tuple(names)
    jd_tt_q = _q(jd_tt, _JD_QUANT) or jd_tt
    lat_q = _q(latitude, _LATLON_QUANT) if topocentric else None
    lon_q = _q(longitude, _LATLON_QUANT) if topocentric else None
    elev_q = _q(elevation_m, _ELEV_QUANT) if topocentric else None

    now_lon, now_spd, source = _cached_positions(jd_tt_q, names_key, topocentric, lat_q, lon_q, elev_q)

    out: Dict[str, Tuple[float, Optional[float]]] = {}

    if all(nm in now_lon for nm in names_key) and all(now_spd.get(nm) is not None for nm in names_key):
        for nm in names_key:
            out[nm] = (_norm360(float(now_lon[nm])), float(now_spd[nm]))
        return out, source

    minus_lon_cache: Dict[float, Dict[str, float]] = {}
    plus_lon_cache:  Dict[float, Dict[str, float]] = {}

    def _get_cached(jd_q: float) -> Dict[str, float]:
        if jd_q in minus_lon_cache:
            return minus_lon_cache[jd_q]
        if jd_q in plus_lon_cache:
            return plus_lon_cache[jd_q]
        lon_m, _spd_m, _ = _cached_positions(jd_q, names_key, topocentric, lat_q, lon_q, elev_q)
        minus_lon_cache[jd_q] = lon_m; plus_lon_cache[jd_q] = lon_m
        return lon_m

    for nm in names_key:
        if nm not in now_lon:
            continue
        l0 = _norm360(float(now_lon[nm]))
        spd: Optional[float] = None
        if now_spd.get(nm) is not None:
            spd = float(now_spd[nm])
        else:
            step = _adaptive_speed_step(nm, speed_step_days)
            if step and step > 0:
                jm = _q(jd_tt - step, _JD_QUANT) or (jd_tt - step)
                jp = _q(jd_tt + step, _JD_QUANT) or (jd_tt + step)
                l_m_map = _get_cached(jm)
                l_p_map = _get_cached(jp)
                if nm in l_m_map and nm in l_p_map:
                    l_m = _norm360(float(l_m_map[nm]))
                    l_p = _norm360(float(l_p_map[nm]))
                    spd = _shortest_signed_delta_deg(l_p, l_m) / (2.0 * step)
        out[nm] = (l0, spd)

    return out, source


def _longitudes_only_geocentric(
    jd_tt: float,
    names: List[str],
) -> Tuple[Dict[str, float], str]:
    """
    Fetch only longitudes for points (nodes) as GEOCENTRIC regardless of topo setting.
    We call the same adapter machinery but with topocentric=False to ensure geo.
    """
    names_key = tuple(names)
    jd_tt_q = _q(jd_tt, _JD_QUANT) or jd_tt
    lon_map, _spd_map, source = _cached_positions(jd_tt_q, names_key, False, None, None, None)
    return {k: _norm360(float(v)) for k, v in lon_map.items()}, source


# ───────────────────────────── Angles (Asc/MC) ────────────────────────
_def_warn_angles = "angles_fallback_Meeus"


def _split_jd(jd: float) -> Tuple[float, float]:
    d = math.floor(jd)
    return d, jd - d


def _atan2d(y: float, x: float) -> float:
    if x == 0.0 and y == 0.0:
        return 0.0
    return _norm360(math.degrees(math.atan2(y, x)))


def _sind(a: float) -> float: return math.sin(math.radians(a))
def _cosd(a: float) -> float: return math.cos(math.radians(a))
def _tand(a: float) -> float: return math.tan(math.radians(a))


def _acotd(x: float) -> float:
    return _norm360(math.degrees(math.atan2(1.0, x)))


def _gast_deg(jd_ut1: float, jd_tt: float, warnings: List[str]) -> float:
    # ERFA first
    if erfa is not None:
        try:
            d1u, d2u = _split_jd(jd_ut1)
            d1t, d2t = _split_jd(jd_tt)
            gst_rad = erfa.gst06a(d1u, d2u, d1t, d2t)
            return _norm360(math.degrees(gst_rad))
        except Exception:
            pass
    # Meeus GMST-like fallback
    T = (float(jd_ut1) - 2451545.0) / 36525.0
    theta = (
        280.46061837
        + 360.98564736629 * (float(jd_ut1) - 2451545.0)
        + 0.000387933 * (T**2)
        - (T**3) / 38710000.0
    )
    warnings.append(_def_warn_angles)
    return _norm360(theta)


def _true_obliquity_deg(jd_tt: float, warnings: List[str]) -> float:
    if erfa is not None:
        try:
            d1, d2 = _split_jd(jd_tt)
            eps0 = erfa.obl06(d1, d2)
            _dpsi, deps = erfa.nut06a(d1, d2)
            return math.degrees(eps0 + deps)
        except Exception:
            pass
    # Meeus mean obliquity polynomial
    T = (float(jd_tt) - 2451545.0) / 36525.0
    eps_arcsec = 84381.448 - 46.8150*T - 0.00059*(T**2) + 0.001813*(T**3)
    warnings.append(_def_warn_angles)
    return eps_arcsec / 3600.0


def _compute_angles(
    jd_ut1: float,
    jd_tt: float,
    latitude: Optional[float],
    longitude: Optional[float],
    *,
    mode: str,
    ayanamsa_deg: Optional[float],
    warnings: List[str],
) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    if latitude is None or longitude is None:
        warnings.append("angles_missing_geography")
        return None, None, {}

    eps = _true_obliquity_deg(jd_tt, warnings)
    # GAST and RAMC
    T = (float(jd_ut1) - 2451545.0) / 36525.0  # keep for dbg even if ERFA path used
    gast = _gast_deg(jd_ut1, jd_tt, warnings)
    ramc = _norm360(gast + float(longitude))

    # MC (ecliptic-of-date)
    mc = _atan2d(_sind(ramc) * _cosd(eps), _cosd(ramc))

    # ASC (ecliptic-of-date)
    # formula: acot( - (tanφ sinε + sinRAMC cosε) / cosRAMC )
    def _acotd_safe(num: float, den: float) -> float:
        den = den if abs(den) > 1e-15 else math.copysign(1e-15, den if den != 0 else 1.0)
        return _acotd(num / den)

    asc = _acotd_safe(-((_tand(float(latitude)) * _sind(eps)) + (_sind(ramc) * _cosd(eps))), _cosd(ramc))

    if mode == "sidereal" and ayanamsa_deg is not None:
        asc = _norm360(asc - float(ayanamsa_deg))
        mc  = _norm360(mc  - float(ayanamsa_deg))

    dbg = {"eps_true_deg": float(eps), "gast_deg": float(gast), "ramc_deg": float(ramc)}
    return float(asc), float(mc), dbg


# ───────────────────────────── Main API ───────────────────────────────
def compute_chart(payload: Dict[str, Any]) -> Dict[str, Any]:
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")

    mode = _validate_mode(payload)

    # Bodies & points (nodes)
    majors_req, points_req, bodies_were_omitted, move_warnings = _split_bodies_points(payload)

    # Timescales
    jd_ut, jd_tt, jd_ut1, ts_warnings = _ensure_timescales(payload)

    # Observer settings
    topocentric = _coerce_bool(payload.get("topocentric"), False)
    warnings: List[str] = list(ts_warnings) + list(move_warnings)

    # Geo for bodies (topo optional) and for angles (required → can be None)
    if topocentric:
        lat, lon, elev, w, downgraded = _validate_and_normalize_geo_for_topo(
            payload.get("latitude"), payload.get("longitude"), payload.get("elevation_m")
        )
        warnings.extend(w)
        if downgraded:
            topocentric = False
            lat = lon = elev = None
    else:
        lat = float(payload.get("latitude")) if _is_finite(payload.get("latitude")) else None
        lon = float(payload.get("longitude")) if _is_finite(payload.get("longitude")) else None
        elev = None

    # Ephemeris (longitudes + speeds) for majors
    results, source_tag = _longitudes_and_speeds(
        jd_tt, majors_req,
        topocentric=topocentric,
        latitude=lat, longitude=lon, elevation_m=elev,
        speed_step_days=_SPEED_STEP_DEF,
    )

    # Sidereal ayanāṁśa
    ay_deg: Optional[float] = None
    ay_note: Optional[str] = None
    if mode == "sidereal":
        ay_deg, ay_note = _resolve_ayanamsa(jd_tt, payload.get("ayanamsa"), warnings)

    # Build majors (bodies) list
    out_bodies: List[Dict[str, Any]] = []
    missing_bodies: List[str] = []
    for nm in majors_req:
        tup = results.get(nm)
        if not tup:
            missing_bodies.append(nm)
            continue
        lon_deg, speed = tup
        if mode == "sidereal" and ay_deg is not None:
            lon_deg = _norm360(lon_deg - float(ay_deg))
        row: Dict[str, Any] = {
            "name": nm,
            "lon": float(_norm360(lon_deg)),
            "longitude_deg": float(_norm360(lon_deg)),
            "speed": (float(speed) if speed is not None else None),
            "speed_deg_per_day": (float(speed) if speed is not None else None),
            "lat": None,
        }
        out_bodies.append(row)

    if missing_bodies:
        warnings.append(f"adapter_missing_bodies({', '.join(missing_bodies)})")

    # Points (nodes): always geocentric; speed intentionally None
    out_points: List[Dict[str, Any]] = []
    if points_req:
        # Try to fetch both nodes via adapter in one shot.
        # If one is missing, compute it from the other (+180).
        lon_map_nodes, source_nodes = _longitudes_only_geocentric(jd_tt, points_req)

        # Fill any missing node via 180° complement if its counterpart exists.
        need_north = ("North Node" in points_req) and ("North Node" not in lon_map_nodes)
        need_south = ("South Node" in points_req) and ("South Node" not in lon_map_nodes)

        if need_north or need_south:
            # If we have one node, derive the other
            if "North Node" in lon_map_nodes and need_south:
                lon_map_nodes["South Node"] = _norm360(lon_map_nodes["North Node"] + 180.0)
            elif "South Node" in lon_map_nodes and need_north:
                lon_map_nodes["North Node"] = _norm360(lon_map_nodes["South Node"] + 180.0)
            else:
                # As a last resort, try fetching the counterpart explicitly
                counterpart = []
                if need_north:
                    counterpart.append("North Node")
                if need_south:
                    counterpart.append("South Node")
                extra_map, _src2 = _longitudes_only_geocentric(jd_tt, counterpart)
                lon_map_nodes.update(extra_map)

        for nm in points_req:
            if nm not in lon_map_nodes:
                warnings.append(f"adapter_missing_points({nm})")
                continue
            lon_deg = float(lon_map_nodes[nm])
            if mode == "sidereal" and ay_deg is not None:
                lon_deg = _norm360(lon_deg - float(ay_deg))
            out_points.append({
                "name": nm,
                "is_point": True,
                "lon": float(_norm360(lon_deg)),
                "longitude_deg": float(_norm360(lon_deg)),
                "speed": None,
                "speed_deg_per_day": None,
                "lat": None,
            })

        # Override source tag if adapter used a different kernel path for nodes.
        if source_nodes and source_nodes != source_tag:
            # keep the first (majors) source, but note mismatch
            warnings.append(f"points_source({source_nodes})")

    # Angles
    asc_deg, mc_deg, dbg = _compute_angles(
        jd_ut1=jd_ut1, jd_tt=jd_tt, latitude=lat, longitude=lon,
        mode=mode, ayanamsa_deg=ay_deg, warnings=warnings,
    )

    meta: Dict[str, Any] = {
        "mode": mode,
        "ayanamsa_deg": float(ay_deg) if ay_deg is not None else None,
        "frame": "ecliptic-of-date",
        "observer": "topocentric" if topocentric else "geocentric",
        "source": str(source_tag),
        "angles_engine": ("ERFA gst06a + true_obliquity" if erfa is not None else "Meeus fallback"),
        "module": _PROJECT_SOURCE_TAG,
        **dbg,
    }
    if warnings:
        meta["warnings"] = warnings
    if ay_note and "fallback" in (ay_note or ""):
        meta.setdefault("warnings", []).append(ay_note)

    out: Dict[str, Any] = {
        "mode": mode,
        "ayanamsa_deg": float(ay_deg) if ay_deg is not None else None,  # top-level mirror
        "jd_ut": float(jd_ut),
        "jd_tt": float(jd_tt),
        "jd_ut1": float(jd_ut1),
        "bodies": out_bodies,
        "points": out_points,  # <── new top-level key
        "angles": {
            "asc_deg": (float(asc_deg) if asc_deg is not None else None),
            "mc_deg": (float(mc_deg) if mc_deg is not None else None),
        },
        "asc_deg": (float(asc_deg) if asc_deg is not None else None),
        "mc_deg": (float(mc_deg) if mc_deg is not None else None),
        "meta": meta,
        "warnings": warnings,  # also mirror warnings at top-level for convenience
    }
    return out
