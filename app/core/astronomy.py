# -*- coding: utf-8 -*-
"""
High-precision planetary chart computation.

Public API:
    compute_chart(payload: dict) -> dict
Optional debug:
    clear_ephemeris_cache() -> None
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import inspect
import time
import traceback
import warnings  # narrow: silence ERFA "dubious year" warnings only

warnings.filterwarnings(
    "ignore",
    message=r"ERFA function .*dubious year",
    category=UserWarning,
    module=r"erfa",
)

# --- add 'resolve_place' to the public API ---
__all__ = ["compute_chart", "clear_ephemeris_cache", "resolve_place"]

def resolve_place(
    q: str | None = None, *,
    place_city: str | None = None,
    place_state: str | None = None,
    place_country: str | None = None,
) -> Dict[str, Any]:
    """
    Resolve a place string or parts into:
      { "lat": float|None, "lon": float|None, "tz": str, "elevation_m": float|None }

    - Accepts either a single freeform `q` ("City, State, Country") or parts.
    - Tries project resolvers if present.
    - Never raises; will fall back to tz="UTC" when nothing better is available.
    """
    # Build a nice query string if parts were provided
    if not q:
        parts = [str(x).strip() for x in (place_city, place_state, place_country) if x]
        q = ", ".join(parts)
    q = (q or "").strip()
    if not q:
        raise AstronomyError("place_missing", "place string or city/country required")

    # Try any project-provided resolvers
    providers: list[Callable[[str], Any]] = []
    for modname, fname in (
        ("app.core.geo", "resolve_place"),
        ("app.core.place", "resolve_place"),
        ("app.core.place_resolver", "resolve_place"),
    ):
        try:
            m = __import__(modname, fromlist=[fname])
            fn = getattr(m, fname, None)
            if callable(fn):
                providers.append(fn)
        except Exception:
            pass

    for fn in providers:
        try:
            out = fn(q)
            if isinstance(out, dict) and ("lat" in out) and ("lon" in out):
                lat = float(out["lat"]) if out["lat"] is not None else None
                lon = float(out["lon"]) if out["lon"] is not None else None
                tz = str(out.get("tz") or out.get("timezone") or "UTC")
                elev = out.get("elevation_m", out.get("elevation"))
                elev_m = float(elev) if isinstance(elev, (int, float, str)) and str(elev).strip() != "" else None
                return {"lat": lat, "lon": lon, "tz": tz, "elevation_m": elev_m}
        except Exception:
            continue

    # Tiny built-in map (extend as you like)
    _HARDCODED = {
        "patna, bihar, india": (25.5941, 85.1376, "Asia/Kolkata", 53.0),
        "new delhi, india": (28.6139, 77.2090, "Asia/Kolkata", 216.0),
        "mumbai, maharashtra, india": (19.0760, 72.8777, "Asia/Kolkata", 14.0),
        "london, united kingdom": (51.5074, -0.1278, "Europe/London", 24.0),
    }
    key = q.lower().strip()
    if key in _HARDCODED:
        lat, lon, tz, elev = _HARDCODED[key]
        return {"lat": float(lat), "lon": float(lon), "tz": tz, "elevation_m": float(elev)}

    # Last resort: unknown coords, safe tz
    return {"lat": None, "lon": None, "tz": "UTC", "elevation_m": None}


# ───────────────────────────── Exceptions ─────────────────────────────
class AstronomyError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


# ───────────────────────────── Resilient imports ─────────────────────
try:
    from app.core import ephemeris_adapter as eph  # primary ephemeris adapter
except Exception as e:  # pragma: no cover
    eph = None
    _EPH_IMPORT_ERROR = e

try:
    from app.core import time_kernel as _tk  # preferred timescales (optional)
except Exception:  # pragma: no cover
    _tk = None

try:
    from app.core import timescales as _ts  # civil→JD helpers (optional)
except Exception:  # pragma: no cover
    _ts = None

try:
    import erfa  # PyERFA (SOFA)
except Exception as e:  # pragma: no cover
    erfa = None
    _ERFA_IMPORT_ERROR = e


# ───────────────────────────── Config ────────────────────────────────
def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name, "")
    if v == "" or v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _float_env(*names: str, default: float) -> float:
    for n in names:
        if n and os.getenv(n) not in (None, ""):
            try:
                return float(os.getenv(n, ""))
            except Exception:
                pass
    return float(default)


@dataclass(frozen=True)
class _AstroCfg:
    ayanamsa_default: str
    jd_quant: float
    ll_quant: float
    elev_quant: float
    speed_fd_step_days: float
    geo_soft_lat: float
    geo_hard_lat: float
    elev_min: float
    elev_max: float
    elev_warn: float
    antimer_warn_lon: float
    dut1_seconds: float
    always_include_majors_with_points: bool
    cache_ttl_sec: float
    debug_adapter: bool


CFG = _AstroCfg(
    ayanamsa_default=(os.getenv("OCP_AYANAMSA_DEFAULT", "lahiri").strip().lower() or "lahiri"),
    jd_quant=float(os.getenv("OCP_ASTRO_JD_QUANT", "1e-7")),       # ~0.009 s
    ll_quant=float(os.getenv("OCP_ASTRO_LL_QUANT", "1e-6")),       # ~0.11 m
    elev_quant=float(os.getenv("OCP_GEO_ELEV_QUANT", "0.1")),      # 10 cm
    speed_fd_step_days=float(os.getenv("OCP_SPEED_FD_STEP_DAYS", "0.25")),  # ±6 h
    geo_soft_lat=_float_env("ASTRO_POLAR_SOFT_LAT", "OCP_GEO_SOFT_LAT", default=89.5),
    geo_hard_lat=_float_env("ASTRO_POLAR_HARD_LAT", "OCP_GEO_HARD_LAT", default=89.9),
    elev_min=float(os.getenv("OCP_GEO_ELEV_MIN", "-500.0")),
    elev_max=float(os.getenv("OCP_GEO_ELEV_MAX", "10000.0")),
    elev_warn=_float_env("ASTRO_ELEV_WARN_M", "OCP_GEO_ELEV_WARN", default=4500.0),
    antimer_warn_lon=float(os.getenv("OCP_GEO_ANTI_WARN", "179.9")),
    dut1_seconds=_float_env("ASTRO_DUT1_BROADCAST", "ASTRO_DUT1", "OCP_DUT1_SECONDS", default=0.0),
    always_include_majors_with_points=_bool_env("OCP_ALWAYS_INCLUDE_MAJORS_WITH_POINTS", True),
    cache_ttl_sec=float(os.getenv("OCP_ASTRO_CACHE_TTL_SEC", "600")),  # 10 min
    debug_adapter=_bool_env("ASTRO_DEBUG_ADAPTER", False),
)


# ───────────────────────────── Constants ──────────────────────────────
_CLASSIC_10 = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
)
_EXTRA_ALLOWED = ("Ceres", "Pallas", "Juno", "Vesta", "Chiron", "North Node", "South Node")
ALLOWED_BODIES = set(_CLASSIC_10) | set(_EXTRA_ALLOWED)
_DEF_BODIES: Tuple[str, ...] = _CLASSIC_10

_NODE_CANON = {
    "north node": "North Node",
    "south node": "South Node",
    "node": "North Node",
    "true node": "North Node",
    "mean node": "North Node",
    "truenode": "North Node",
    "meannode": "North Node",
    "ascending node": "North Node",
    "descending node": "South Node",
    "northnode": "North Node",
    "southnode": "South Node",
    "rahu": "North Node",
    "ketu": "South Node",
    "lunar node": "North Node",
    "moon node": "North Node",
    "☊": "North Node",
    "☋": "South Node",
}
_NODE_SET_LC = set(_NODE_CANON.keys())

_BODY_SYNONYMS = {
    "sun": "Sun", "moon": "Moon", "mercury": "Mercury", "venus": "Venus",
    "mars": "Mars", "jupiter": "Jupiter", "saturn": "Saturn", "uranus": "Uranus",
    "neptune": "Neptune", "pluto": "Pluto", "ceres": "Ceres", "pallas": "Pallas",
    "juno": "Juno", "vesta": "Vesta", "chiron": "Chiron",
    "sol": "Sun", "luna": "Moon", "earth": "Earth",
    "♀": "Venus", "♂": "Mars", "♃": "Jupiter", "♄": "Saturn", "♅": "Uranus", "♆": "Neptune", "♇": "Pluto",
}

_PROJECT_SOURCE_TAG = "astronomy(core)"


# ───────────────────────────── Warnings ───────────────────────────────
class _W:
    AYA_FALLBACK = "ayanamsa_fallback"
    TIME_STD_FALLBACK = "timescales_fallback_local_utc_jd"
    DELTAT_CONST = "deltaT_fallback_69s"
    ANGLES_MEEUS = "angles_fallback_meeus"
    ANGLES_MISSING_GEO = "angles_missing_geography"
    LAT_CLAMP = "latitude_clamped_to_range"
    LAT_SOFT_NUDGE = "latitude_soft_nudged_from_pole"
    TOPO_DISABLED_NEAR_POLE = "topocentric_disabled_near_pole(hard)"
    TOPO_MISSING_COORDS = "topocentric_missing_coords_fallback_geocentric"
    ANTIM_MERIDIAN = "near_antimeridian_longitude"
    ELEV_CLAMP_MIN = "elevation_clamped_min"
    ELEV_CLAMP_MAX = "elevation_clamped_max"
    ELEV_HIGH = "very_high_elevation_site"
    PTS_NON_LIST = "points_ignored_non_list"
    ADAPTER_MISS_BODIES = "adapter_missing_bodies"
    ADAPTER_MISS_POINTS = "adapter_missing_points"
    PTS_SOURCE_MISMATCH = "points_source"
    LEAP_SECOND = "leap_second"
    TOPO_FALLBACK_GEO = "topo_position_fallback_geocentric"
    ADAPTER_NO_TOPO = "adapter_no_topocentric_support"
    ADAPTER_ERROR = "adapter_error"
    ADAPTER_PARSE_ERROR = "adapter_response_parse_error"
    BODY_NAME_FUZZY_MATCH = "body_name_fuzzy_matched"
    DUT1_CLAMPED = "dut1_clamped"


def _warn_add(store: List[str], seen: set[str], code: str, detail: Optional[str] = None) -> None:
    s = code if not detail else f"{code}({detail})"
    if s not in seen:
        seen.add(s)
        store.append(s)


def _debug_log(message: str) -> None:
    if CFG.debug_adapter:
        print(f"DEBUG ASTRO: {message}")


# ───────────────────────────── Tiny math helpers ──────────────────────
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


# ───────────────────────────── Mode / names parsing ───────────────────
def _validate_mode(payload: Dict[str, Any]) -> str:
    mode = str(payload.get("mode", "tropical")).strip().lower()
    if mode not in ("tropical", "sidereal"):
        raise AstronomyError("invalid_input", "mode must be 'tropical' or 'sidereal'")
    return mode


def _canon_node_name(s: str) -> Optional[str]:
    return _NODE_CANON.get(str(s).strip().lower())


def _normalize_body_name(name: str) -> str:
    normalized = str(name).strip()
    if normalized in ALLOWED_BODIES:
        return normalized
    lower_name = normalized.lower()
    if lower_name in _BODY_SYNONYMS:
        return _BODY_SYNONYMS[lower_name]
    for allowed in ALLOWED_BODIES:
        if allowed.lower() == lower_name:
            return allowed
    return normalized


def _split_bodies_points(payload: Dict[str, Any], warnings: List[str], seen: set[str]) -> Tuple[List[str], List[str]]:
    bodies_raw = payload.get("bodies", None)
    if bodies_raw is None:
        majors: List[str] = list(_DEF_BODIES)
        bodies_were_omitted = True
    else:
        bodies_were_omitted = False
        try:
            majors = [str(b) for b in bodies_raw]
        except Exception:
            raise AstronomyError("invalid_input", "bodies must be a list of names")

    points_raw = payload.get("points", [])
    points: List[str] = []
    if points_raw is None:
        points_raw = []
    if isinstance(points_raw, (list, tuple)):
        for p in points_raw:
            nm = _canon_node_name(p)
            if nm:
                points.append(nm)
    else:
        _warn_add(warnings, seen, _W.PTS_NON_LIST)

    majors_out: List[str] = []
    for b in majors:
        normalized_body = _normalize_body_name(b)
        if normalized_body not in ALLOWED_BODIES and str(b).strip().lower() not in _NODE_SET_LC:
            allowed = ", ".join(sorted(ALLOWED_BODIES))
            raise AstronomyError("unsupported_body", f"'{b}' not supported (allowed: {allowed})")
        if str(b).strip().lower() in _NODE_SET_LC or normalized_body in ("North Node", "South Node"):
            canon = _canon_node_name(b) or (normalized_body if normalized_body in ("North Node", "South Node") else None)
            if canon and canon not in points:
                points.append(canon)
        else:
            majors_out.append(normalized_body)

    if len(majors_out) == 0:
        if bodies_were_omitted or CFG.always_include_majors_with_points:
            majors_out = list(_DEF_BODIES)

    pts_seen: set[str] = set()
    pts_final: List[str] = []
    for p in points:
        if p not in pts_seen:
            pts_seen.add(p)
            pts_final.append(p)

    return majors_out, pts_final


# ───────────────────────────── Timescales ─────────────────────────────
def _detect_leap_second(time_str: Optional[str]) -> bool:
    if not isinstance(time_str, str):
        return False
    try:
        _hh, _mm, ss = time_str.split(":")
        return int(ss.split(".")[0]) == 60
    except Exception:
        return False


def _normalize_time_for_leap_second(time_str: str) -> str:
    try:
        hh, mm, ss = time_str.split(":")
        if int(ss.split(".")[0]) != 60:
            return time_str
        return f"{hh}:{mm}:59.999999"
    except Exception:
        return time_str


def _ensure_timescales(payload: Dict[str, Any], warnings: List[str], seen: set[str]) -> Tuple[float, float, float, float]:
    """
    Return (jd_ut, jd_tt, jd_ut1, dut1_used_seconds).

    Notes
    -----
    - jd_ut here means *UTC JD* (historical naming); ERFA is *never* given UTC.
      ERFA routines later are called with jd_ut1 (UT1) and jd_tt (TT).
    - UT1 is ALWAYS derived as: jd_ut + dut1_used_seconds / 86400.0
      (deterministic: ignores any UT1 a forwarder may provide).
    - Prefer 'jd_utc' when present; fall back to 'jd_ut' (both represent UTC JD).
    """
    # caller DUT1 (with clamp) — accept numeric strings; use CFG default
    dut1_raw = payload.get("dut1", payload.get("dut1_seconds", None))
    if dut1_raw is None or (isinstance(dut1_raw, str) and not dut1_raw.strip()):
        dut1_raw = CFG.dut1_seconds
    try:
        dut1_used = float(dut1_raw)
    except Exception:
        dut1_used = float(CFG.dut1_seconds)

    if abs(dut1_used) > 0.9:
        _warn_add(warnings, seen, _W.DUT1_CLAMPED, f"{dut1_used}")
        dut1_used = max(-0.9, min(0.9, dut1_used))

    # accept provided JDs (UTC + TT)
    jd_utc_in = payload.get("jd_utc")
    jd_ut_in  = payload.get("jd_ut")
    jd_tt_in  = payload.get("jd_tt")

    ju = jd_utc_in if isinstance(jd_utc_in, (int, float)) else jd_ut_in
    if all(isinstance(x, (int, float)) for x in (ju, jd_tt_in)):
        ju = float(ju)
        jt = float(jd_tt_in)
        j1 = ju + (dut1_used / 86400.0)  # deterministic
        return ju, jt, j1, dut1_used

    # civil path
    d = payload.get("date")
    t = payload.get("time")
    tz = payload.get("place_tz") or payload.get("tz") or "UTC"

    if _detect_leap_second(t):
        _warn_add(warnings, seen, _W.LEAP_SECOND)
        t = _normalize_time_for_leap_second(str(t))

    # preferred: time_kernel
    if _tk is not None:
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
            fn = getattr(_tk, fname, None)
            if not callable(fn):
                continue
            try:
                params = inspect.signature(fn).parameters
            except Exception:
                params = {}

            kwargs: Dict[str, Any] = {}
            if "date" in params:     kwargs["date"] = d
            if "time" in params:     kwargs["time"] = t
            if "tz" in params:       kwargs["tz"] = tz
            if "place_tz" in params: kwargs["place_tz"] = tz
            if "dut1" in params:           kwargs["dut1"] = float(dut1_used)
            if "dut1_seconds" in params:   kwargs["dut1_seconds"] = float(dut1_used)

            out = None
            try:
                out = fn(**kwargs)
            except TypeError:
                try:
                    if ("dut1" in params) or ("dut1_seconds" in params):
                        out = fn(d, t, tz, float(dut1_used))
                    else:
                        out = fn(d, t, tz)
                except Exception:
                    out = None
            except Exception:
                out = None
            if out is None:
                continue

            # forwarder warnings
            try:
                for w in (out.get("warnings") or []):
                    if isinstance(w, str) and w:
                        _warn_add(warnings, seen, w)
            except Exception:
                pass

            # prefer jd_utc; fallback jd_ut
            if isinstance(out, dict):
                ju = out.get("jd_utc", out.get("jd_ut"))
                jt = out.get("jd_tt")
                if all(isinstance(x, (int, float)) for x in (ju, jt)):
                    ju = float(ju); jt = float(jt)
                    # If caller didn't set DUT1, adopt forwarder dut1 (clamped)
                    if not isinstance(payload.get("dut1"), (int, float)) and not isinstance(payload.get("dut1_seconds"), (int, float)):
                        if isinstance(out.get("dut1"), (int, float)):
                            dut1_used = float(out["dut1"])
                            if abs(dut1_used) > 0.9:
                                _warn_add(warnings, seen, _W.DUT1_CLAMPED, f"{dut1_used}")
                                dut1_used = max(-0.9, min(0.9, dut1_used))
                    j1 = ju + (dut1_used / 86400.0)  # deterministic
                    return ju, jt, j1, dut1_used

            if isinstance(out, (list, tuple)) and len(out) >= 2:
                ju, jt = map(float, out[:2])
                j1 = ju + (dut1_used / 86400.0)
                return float(ju), float(jt), float(j1), dut1_used

    # fallbacks to compute UTC JD, then TT & UT1
    def _jd_utc_via_ts(d_: str, t_: str, z_: str) -> float:
        if _ts is None:
            raise AstronomyError("timescales_missing", "timescales module not available")
        return float(_ts.julian_day_utc(d_, t_, z_))

    def _jd_utc_via_stdlib(d_: str, t_: str, z_: str) -> float:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        parts = t_.split(":")
        timestr = t_ if len(parts) >= 3 else t_ + ":00"
        try:
            dt_local = datetime.fromisoformat(f"{d_}T{timestr}").replace(tzinfo=ZoneInfo(z_))
        except Exception as e:
            raise AstronomyError("timescales_missing", f"Invalid tz '{z_}': {e}")
        dt_utc = dt_local.astimezone(timezone.utc)
        Y, M, D = dt_utc.year, dt_utc.month, dt_utc.day
        h = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600 + dt_utc.microsecond/3.6e9
        if M <= 2:
            Y -= 1; M += 12
        A = Y // 100
        B = 2 - A + A // 4
        JD0 = int(365.25*(Y + 4716)) + int(30.6001*(M + 1)) + D + B - 1524.5
        return JD0 + h/24.0

    if not (isinstance(d, str) and isinstance(t, str)):
        missing = [k for k, v in (("jd_ut/jd_utc", ju), ("jd_tt", jd_tt_in)) if not isinstance(v, (int, float))]
        raise AstronomyError("timescales_missing", f"Supply {', '.join(missing)} or provide date/time/tz")

    used_stdlib = False
    try:
        ju = _jd_utc_via_ts(d, t, tz)
    except Exception:
        try:
            ju = _jd_utc_via_stdlib(d, t, tz)
            used_stdlib = True
            _warn_add(warnings, seen, _W.TIME_STD_FALLBACK)
        except AstronomyError:
            raise
        except Exception as e:
            raise AstronomyError("timescales_missing", f"Failed to compute JD from {d} {t} {tz}: {e}")

    try:
        y, m = map(int, str(d).split("-")[:2])
    except Exception:
        y, m = 2000, 1

    if _ts is not None:
        try:
            jt = float(_ts.jd_tt_from_utc_jd(ju, y, m))
        except Exception:
            jt = ju + 69.0/86400.0
            _warn_add(warnings, seen, _W.DELTAT_CONST)
    else:
        jt = ju + 69.0/86400.0
        if not used_stdlib:
            _warn_add(warnings, seen, _W.DELTAT_CONST)

    j1 = float(ju) + (dut1_used / 86400.0)
    return float(ju), float(jt), float(j1), float(dut1_used)


# ───────────────────────────── Geo / Topocentric ──────────────────────
def _normalize_lon180(lon: float) -> float:
    return _wrap180(lon)


def _validate_and_normalize_geo_for_topo(
    lat: Any, lon: Any, elev: Any, warnings: List[str], seen: set[str]
) -> Tuple[float, float, Optional[float], bool]:
    downgraded = False
    if not _is_finite(lat) or not _is_finite(lon):
        raise AstronomyError("invalid_input", "latitude/longitude must be finite numbers")

    latf = float(lat)
    lonf = _normalize_lon180(float(lon))

    if latf < -90.0 or latf > 90.0:
        _warn_add(warnings, seen, _W.LAT_CLAMP)
        latf = max(-90.0, min(90.0, latf))

    abslat = abs(latf)
    if abslat >= CFG.geo_hard_lat:
        _warn_add(warnings, seen, _W.TOPO_DISABLED_NEAR_POLE)
        downgraded = True
    elif abslat >= CFG.geo_soft_lat:
        latf = math.copysign((CFG.geo_hard_lat - 0.05), latf)
        _warn_add(warnings, seen, _W.LAT_SOFT_NUDGE)

    if abs(lonf) >= CFG.antimer_warn_lon:
        _warn_add(warnings, seen, _W.ANTIM_MERIDIAN)

    if elev is None or (isinstance(elev, str) and str(elev).strip() == ""):
        elev_m: Optional[float] = None
    else:
        if not _is_finite(elev):
            raise AstronomyError("invalid_input", "elevation_m must be a finite number in meters")
        elev_m = float(elev)
        if elev_m < CFG.elev_min:
            _warn_add(warnings, seen, _W.ELEV_CLAMP_MIN)
            elev_m = CFG.elev_min
        elif elev_m > CFG.elev_max:
            _warn_add(warnings, seen, _W.ELEV_CLAMP_MAX)
            elev_m = CFG.elev_max
        elif abs(elev_m) >= CFG.elev_warn:
            _warn_add(warnings, seen, _W.ELEV_HIGH)

    return latf, lonf, elev_m, downgraded


# ───────────────────────────── Ayanāṁśa (delegated) ───────────────────
@lru_cache(maxsize=4096)
def _ayanamsa_deg_cached(jd_tt_q: float, ay_key: str) -> Tuple[float, str]:
    """
    Resolve ayanāṁśa using app.core.ayanamsa, with robust shape handling.
    """
    try:
        mod = __import__("app.core.ayanamsa", fromlist=[
            "get_ayanamsa_with_resolution", "get_ayanamsa_deg"
        ])

        # 1) Preferred helper with resolution flags
        helper = getattr(mod, "get_ayanamsa_with_resolution", None)
        if callable(helper):
            try:
                val, canonical, is_alias, is_unknown = helper(jd_tt_q, ay_key)
            except TypeError:
                val, canonical, is_alias, is_unknown = helper(ay_key, jd_tt_q)
            note_bits = [str(canonical)]
            if is_alias:
                note_bits.append("alias")
            if is_unknown:
                note_bits.append("fallback")
            return float(val), ",".join(note_bits)

        # 2) Legacy function
        fn = getattr(mod, "get_ayanamsa_deg", None)
        if not callable(fn):
            raise AttributeError("get_ayanamsa_deg not found")

        try:
            res = fn(jd_tt_q, ay_key)
        except TypeError:
            res = fn(ay_key, jd_tt_q)

        if isinstance(res, (tuple, list)) and len(res) >= 1:
            val = float(res[0])
            note = str(res[1]) if len(res) >= 2 else str(ay_key)
            return val, note

        if isinstance(res, dict):
            for k in ("ayanamsa_deg", "deg", "value"):
                if k in res and isinstance(res[k], (int, float)):
                    val = float(res[k])
                    note = str(res.get("note") or res.get("name") or res.get("key") or ay_key)
                    return val, note
            raise ValueError("unexpected dict shape from get_ayanamsa_deg")

        return float(res), str(ay_key)

    except Exception as e:
        raise AstronomyError("ayanamsa_unavailable", f"failed to resolve ayanamsa '{ay_key}': {e}")


def _resolve_ayanamsa(
    jd_tt: float, ayanamsa: Any, warnings: List[str], seen: set[str]
) -> Tuple[Optional[float], Optional[str]]:
    """
    Resolve ayanāṁśa from payload value or default and return (deg, note).
    """
    if ayanamsa is None or (isinstance(ayanamsa, str) and not str(ayanamsa).strip()):
        key = CFG.ayanamsa_default
    elif isinstance(ayanamsa, (int, float)):
        return float(ayanamsa), "explicit"
    else:
        key = str(ayanamsa).strip().lower()

    jd_q = _q(jd_tt, CFG.jd_quant) or jd_tt
    ay, note = _ayanamsa_deg_cached(jd_q, key)

    if isinstance(note, str) and "fallback" in note.lower():
        _warn_add(warnings, seen, _W.AYA_FALLBACK, note)

    return float(ay), note


# ─────────────────────── Adapter I/O & normalization ──────────────────
def _adapter_source_tag() -> str:
    tag = getattr(eph, "current_kernel_name", None) or getattr(eph, "EPHEMERIS_NAME", None) or "adapter"
    try:
        return str(tag())
    except Exception:
        return str(tag)


def _adapter_kernel_info():
    path = None
    coverage = None
    try:
        get_path = getattr(eph, "current_kernel_path", None)
        path = get_path() if callable(get_path) else getattr(eph, "EPHEMERIS_PATH", None)
    except Exception:
        pass
    try:
        coverage = getattr(eph, "KERNEL_COVERAGE_JD", None)
    except Exception:
        pass
    return path, coverage


def _adapter_callable(*names: str) -> Optional[Callable[..., Any]]:
    for n in names:
        fn = getattr(eph, n, None)
        if callable(fn):
            return fn
    return None


def _build_rich_geo_kwargs(topocentric: bool, lat_q, lon_q, elev_q, force_center: Optional[str]) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if force_center is not None:
        kw["center"] = force_center
        kw["topocentric"] = (force_center == "topocentric")
    else:
        kw["topocentric"] = bool(topocentric)
        kw["center"] = "topocentric" if topocentric else "geocentric"

    if topocentric or (force_center == "topocentric"):
        if lat_q is not None:   kw["latitude"] = float(lat_q); kw["lat"] = float(lat_q)
        if lon_q is not None:   kw["longitude"] = float(lon_q); kw["lon"] = float(lon_q)
        if elev_q is not None:  kw["elevation_m"] = float(elev_q); kw["elev_m"] = float(elev_q); kw["elevation"] = float(elev_q)
        if (lat_q is not None) and (lon_q is not None):
            obs = {"latitude": float(lat_q), "longitude": float(lon_q)}
            if elev_q is not None:
                obs["elevation_m"] = float(elev_q)
            kw["observer"] = obs
    return kw


def _unwrap_adapter_result(res: Any) -> Tuple[str, Any]:
    if res is None:
        return "empty", []
    if isinstance(res, dict):
        for k in ("rows", "result", "data", "payload"):
            v = res.get(k)
            if isinstance(v, (list, tuple)):
                return "rows", v
        if any(k in res for k in ("longitudes", "longitude", "lon", "velocities", "velocity", "speed", "speeds", "names", "bodies", "planets", "ids", "labels")):
            return "maps", res
        for v in res.values():
            if isinstance(v, (list, tuple)):
                return "rows", v
        return "flat", res

    if isinstance(res, (list, tuple)):
        if not res:
            return "empty", []
        first = res[0]
        if isinstance(first, dict):
            return "rowdicts", res
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return "tuples", res
        if all(isinstance(x, (int, float)) for x in res):
            return "positional", res
        if hasattr(first, "name"):
            return "objects", res
        return "rows", res

    if hasattr(res, "longitudes"):
        return "maps", {"longitudes": getattr(res, "longitudes", None), "velocities": getattr(res, "velocities", None)}

    return "unknown", res


def _extract_rows(kind: str, payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if kind == "empty":
        return rows

    if kind in ("maps", "flat"):
        data = payload or {}
        longmaps = None
        for lk in ("longitudes", "longitude", "lon"):
            if isinstance(data.get(lk), dict):
                longmaps = data[lk]
                break
        velmaps = None
        for sk in ("velocities", "velocity", "speed", "speeds"):
            if isinstance(data.get(sk), dict):
                velmaps = data[sk]
                break

        list_longs = None
        for lk in ("longitudes", "longitude", "lon"):
            if isinstance(data.get(lk), (list, tuple)):
                list_longs = list(data[lk])
                break
        list_vels = None
        for sk in ("velocities", "velocity", "speed", "speeds"):
            if isinstance(data.get(sk), (list, tuple)):
                list_vels = list(data[sk])
                break
        names_list = None
        for nk in ("names", "bodies", "planets", "ids", "labels"):
            if isinstance(data.get(nk), (list, tuple)) and all(isinstance(x, (str, int)) for x in data[nk]):
                names_list = [str(x) for x in data[nk]]
                break

        if isinstance(longmaps, dict):
            for name, lon in longmaps.items():
                if isinstance(lon, (int, float)):
                    sp = None
                    if isinstance(velmaps, dict):
                        v = velmaps.get(name)
                        sp = float(v) if isinstance(v, (int, float)) else None
                    rows.append({"name": str(name), "lon": float(lon), "speed": sp})
            return rows

        if isinstance(list_longs, list) and all(isinstance(v, (int, float)) for v in list_longs):
            n = len(list_longs)
            if isinstance(list_vels, list) and len(list_vels) != n:
                list_vels = None
            for i, lon in enumerate(list_longs):
                sp = float(list_vels[i]) if (isinstance(list_vels, list) and isinstance(list_vels[i], (int, float))) else None
                if isinstance(names_list, list) and len(names_list) == n:
                    rows.append({"name": names_list[i], "lon": float(lon), "speed": sp})
                else:
                    rows.append({"_pos": i, "lon": float(lon)})
            return rows

        if kind == "flat" and isinstance(data, dict) and all(isinstance(k, (str, int)) and isinstance(v, (int, float)) for k, v in data.items()):
            for name, lon in data.items():
                rows.append({"name": str(name), "lon": float(lon), "speed": None})
            return rows

    if kind in ("rows", "rowdicts"):
        for r in payload:
            if not isinstance(r, dict):
                continue
            if len(r) == 1:
                k, v = next(iter(r.items()))
                if isinstance(k, (str, int)) and isinstance(v, (int, float)):
                    rows.append({"name": str(k), "lon": float(v), "speed": None})
                    continue

            nm = r.get("name") or r.get("body") or r.get("planet") or r.get("id") or r.get("label")
            if not nm:
                continue
            lon = (
                r.get("lon", None) if r.get("lon", None) is not None else
                r.get("longitude", None) if r.get("longitude", None) is not None else
                r.get("longitude_deg", None) if r.get("longitude_deg", None) is not None else
                r.get("lambda", None) if r.get("lambda", None) is not None else
                r.get("ecliptic_longitude", None)
            )
            if lon is None or not isinstance(lon, (int, float)):
                continue
            sp = (
                r.get("speed", None) if isinstance(r.get("speed", None), (int, float)) else
                r.get("velocity", None) if isinstance(r.get("velocity", None), (int, float)) else
                r.get("speed_deg_per_day", None) if isinstance(r.get("speed_deg_per_day", None), (int, float)) else
                r.get("lambda_dot", None) if isinstance(r.get("lambda_dot", None), (int, float)) else
                r.get("deg_per_day", None) if isinstance(r.get("deg_per_day", None), (int, float)) else
                None
            )
            rows.append({"name": str(nm), "lon": float(lon), "speed": (float(sp) if sp is not None else None)})
        return rows

    if kind == "tuples":
        for item in payload:
            try:
                nm = str(item[0])
                lon = float(item[1])
                sp = float(item[2]) if len(item) >= 3 and isinstance(item[2], (int, float)) else None
                rows.append({"name": nm, "lon": lon, "speed": sp})
            except Exception:
                continue
        return rows

    if kind == "positional":
        return [{"_pos": i, "lon": float(v)} for i, v in enumerate(payload)]

    if kind == "objects":
        for obj in payload:
            try:
                nm = str(getattr(obj, "name"))
                lon = getattr(obj, "lon", None) or getattr(obj, "longitude", None) or getattr(obj, "longitude_deg", None)
                if lon is None:
                    continue
                sp = getattr(obj, "speed", None) or getattr(obj, "velocity", None)
                rows.append({"name": nm, "lon": float(lon), "speed": (float(sp) if isinstance(sp, (int, float)) else None)})
            except Exception:
                continue
        return rows

    return rows


def _map_rows_to_requested(
    rows: List[Dict[str, Any]], requested: Tuple[str, ...], warnings: List[str], seen: set[str]
) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
    want = list(requested)
    want_lc = [w.lower() for w in want]
    lon_map: Dict[str, float] = {}
    spd_map: Dict[str, Optional[float]] = {}

    by_name: Dict[str, Dict[str, Any]] = {}
    by_name_lower: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if "name" in r:
            name = str(r["name"])
            by_name[name] = r
            by_name_lower[name.lower()] = r

    for nm, key in zip(want, want_lc):
        r = by_name.get(nm) or by_name_lower.get(key)
        if r and isinstance(r.get("lon"), (int, float)):
            lon_map[nm] = float(r["lon"])
            spd_map[nm] = float(r["speed"]) if isinstance(r.get("speed"), (int, float)) else None

    for nm in want:
        if nm in lon_map:
            continue
        normalized = _normalize_body_name(nm)
        if normalized != nm:
            for adapter_name, r in by_name.items():
                if _normalize_body_name(adapter_name) == normalized and isinstance(r.get("lon"), (int, float)):
                    lon_map[nm] = float(r["lon"])
                    spd_map[nm] = float(r["speed"]) if isinstance(r.get("speed"), (int, float)) else None
                    _warn_add(warnings, seen, _W.BODY_NAME_FUZZY_MATCH, f"{nm}->{adapter_name}")
                    break

    for r in rows:
        if "_pos" in r and isinstance(r.get("lon"), (int, float)):
            i = int(r["_pos"])
            if 0 <= i < len(want) and want[i] not in lon_map:
                lon_map[want[i]] = float(r["lon"])
                spd_map[want[i]] = None

    return lon_map, spd_map



# ─────────────────────────── TTL cache around adapter ─────────────────
_PosCache: Dict[Tuple[Any, ...], Tuple[float, Dict[str, float], Dict[str, Optional[float]], str]] = {}


def _ttl_get_or_compute(
    key: Tuple[Any, ...],
    ttl: float,
    compute: Callable[[], Tuple[Dict[str, float], Dict[str, Optional[float]], str]],
) -> Tuple[Dict[str, float], Dict[str, Optional[float]], str]:
    now = time.time()
    item = _PosCache.get(key)
    if item is not None:
        t0, lon_map, spd_map, src = item
        if (now - t0) <= ttl:
            return lon_map, spd_map, src
    lon_map, spd_map, src = compute()
    _PosCache[key] = (now, lon_map, spd_map, src)
    return lon_map, spd_map, src


def clear_ephemeris_cache() -> None:
    _PosCache.clear()
    try:
        _ayanamsa_deg_cached.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def _cached_positions(
    jd_tt_q: float,
    names_key: Tuple[str, ...],
    topocentric: bool,
    lat_q: Optional[float],
    lon_q: Optional[float],
    elev_q: Optional[float],
    warnings: Optional[List[str]] = None,
    seen: Optional[set[str]] = None,
    frame: str = "ecliptic-of-date",
) -> Tuple[Dict[str, float], Dict[str, Optional[float]], str]:
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")

    if warnings is None:
        warnings = []
    if seen is None:
        seen = set()

    kernel_tag = _adapter_source_tag()
    cache_key = (kernel_tag, jd_tt_q, names_key, bool(topocentric), lat_q, lon_q, elev_q, frame)

    def _compute() -> Tuple[Dict[str, float], Dict[str, Optional[float]], str]:
        fn = _adapter_callable(
            "ecliptic_longitudes_and_velocities",
            "get_ecliptic_longitudes_and_velocities",
            "ecliptic_longitudes",
            "get_ecliptic_longitudes",
        )
        if fn is None:
            raise AstronomyError("adapter_api_mismatch", "ephemeris_adapter missing longitudes API")

        try:
            sig = inspect.signature(fn)
            params = sig.parameters
        except Exception:
            sig = None
            params = {}

        def _base_kwargs() -> Dict[str, Any]:
            base = {}
            if "jd_tt" in params:
                base["jd_tt"] = jd_tt_q
            elif "jd" in params:
                base["jd"] = jd_tt_q
            if "frame" in params:
                base["frame"] = frame
            return base

        def _geo_variants(force_center: Optional[str]) -> List[Dict[str, Any]]:
            rich = _build_rich_geo_kwargs(topocentric, lat_q, lon_q, elev_q, force_center)
            rich = {k: v for k, v in rich.items() if v is not None}
            obs_only = {k: v for k, v in rich.items() if k in ("observer", "center", "topocentric")}
            latlon_ll = {k: v for k, v in rich.items() if k in ("latitude", "longitude", "elevation_m", "center", "topocentric")}
            latlon_short = {k: v for k, v in rich.items() if k in ("lat", "lon", "elev_m", "center", "topocentric")}
            center_only = {k: v for k, v in rich.items() if k in ("center", "topocentric")}
            none_geo: Dict[str, Any] = {}
            return [rich, obs_only, latlon_ll, latlon_short, center_only, none_geo]

        names_to_try: List[List[str]] = [
            list(names_key),
            [n.lower() for n in names_key],
            [_normalize_body_name(n) for n in names_key],
        ]
        name_keys_order = ["names", "bodies", "planets", "ids", "labels"]

        def _try_call(names_list: List[str], geo_kw: Dict[str, Any]) -> Any:
            bk = _base_kwargs()
            detected = [k for k in name_keys_order if k in params]
            for nk in (detected or name_keys_order):
                kw = dict(bk); kw.update(geo_kw); kw[nk] = names_list
                try:
                    return fn(**kw)
                except TypeError:
                    continue
                except Exception:
                    continue
            try:
                extra = dict(geo_kw)
                if "jd_tt" in bk:
                    return fn(bk["jd_tt"], names_list, **{k: v for k, v in extra.items() if k != "jd_tt"})
                if "jd" in bk:
                    return fn(bk["jd"], names_list, **{k: v for k, v in extra items() if k != "jd"})
            except Exception:
                pass
            try:
                return fn(jd_tt_q, names_list)
            except Exception:
                pass
            try:
                return fn(jd_tt_q)
            except Exception:
                pass
            return None

        res = None
        adapter_error = None
        for names_variant in names_to_try:
            for geo_kw in _geo_variants(force_center="topocentric"):
                try:
                    res = _try_call(names_variant, geo_kw)
                    if res is not None:
                        raise StopIteration
                except StopIteration:
                    break
                except Exception as e:
                    adapter_error = e
                    continue
            if res is not None:
                break

        if res is None and topocentric:
            _warn_add(warnings, seen, _W.ADAPTER_NO_TOPO, str(kernel_tag))
            for names_variant in names_to_try:
                for geo_kw in _geo_variants(force_center="geocentric"):
                    try:
                        res = _try_call(names_variant, geo_kw)
                        if res is not None:
                            raise StopIteration
                    except StopIteration:
                        break
                    except Exception as e:
                        adapter_error = e
                        continue
                if res is not None:
                    break

        if res is None:
            _warn_add(warnings, seen, _W.ADAPTER_ERROR, f"kernel={kernel_tag}, last_error={adapter_error}")
            return {}, {}, kernel_tag

        try:
            kind, payload = _unwrap_adapter_result(res)
            rows = _extract_rows(kind, payload)
            lon_map, spd_map = _map_rows_to_requested(rows, names_key, warnings, seen)
            return lon_map, spd_map, kernel_tag
        except Exception as e:
            _warn_add(warnings, seen, _W.ADAPTER_PARSE_ERROR, f"{type(e).__name__}: {e}")
            traceback.print_exc()
            return {}, {}, kernel_tag

    return _ttl_get_or_compute(cache_key, CFG.cache_ttl_sec, _compute)


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
    warnings: List[str],
    seen: set[str],
    frame: str,
) -> Tuple[Dict[str, Tuple[float, Optional[float]]], str]:
    names_key = tuple(names)
    jd_tt_q = _q(jd_tt, CFG.jd_quant) or jd_tt
    lat_q = _q(latitude, CFG.ll_quant) if topocentric else None
    lon_q = _q(longitude, CFG.ll_quant) if topocentric else None
    elev_q = _q(elevation_m, CFG.elev_quant) if topocentric else None

    now_lon, now_spd, source = _cached_positions(
        jd_tt_q, names_key, topocentric, lat_q, lon_q, elev_q, warnings=warnings, seen=seen, frame=frame
    )

    out: Dict[str, Tuple[float, Optional[float]]] = {}
    if all(nm in now_lon for nm in names_key) and all(now_spd.get(nm) is not None for nm in names_key):
        for nm in names_key:
            out[nm] = (_norm360(float(now_lon[nm])), float(now_spd[nm]))  # type: ignore[arg-type]
        return out, source

    minus_lon_cache: Dict[float, Dict[str, float]] = {}
    plus_lon_cache: Dict[float, Dict[str, float]] = {}

    def _get_cached(jd_q: float) -> Dict[str, float]:
        if jd_q in minus_lon_cache:
            return minus_lon_cache[jd_q]
        if jd_q in plus_lon_cache:
            return plus_lon_cache[jd_q]
        lon_m, _spd_m, _ = _cached_positions(
            jd_q, names_key, topocentric, lat_q, lon_q, elev_q, warnings=warnings, seen=seen, frame=frame
        )
        minus_lon_cache[jd_q] = lon_m
        plus_lon_cache[jd_q] = lon_m
        return lon_m

    for nm in names_key:
        l0 = _norm360(float(now_lon[nm])) if nm in now_lon else None
        spd: Optional[float] = float(now_spd[nm]) if nm in now_spd and now_spd[nm] is not None else None
        if l0 is None:
            continue
        if spd is None:
            step = _adaptive_speed_step(nm, speed_step_days)
            if step and step > 0:
                jm = _q(jd_tt - step, CFG.jd_quant) or (jd_tt - step)
                jp = _q(jd_tt + step, CFG.jd_quant) or (jd_tt + step)
                l_m_map = _get_cached(jm)
                l_p_map = _get_cached(jp)
                if nm in l_m_map and nm in l_p_map:
                    l_m = _norm360(float(l_m_map[nm]))
                    l_p = _norm360(float(l_p_map[nm]))
                    spd = _shortest_signed_delta_deg(l_p, l_m) / (2.0 * step)
        out[nm] = (l0, spd)

    return out, source


def _longitudes_only_geocentric(jd_tt: float, names: List[str], frame: str) -> Tuple[Dict[str, float], str]:
    names_key = tuple(names)
    jd_tt_q = _q(jd_tt, CFG.jd_quant) or jd_tt
    lon_map, _spd_map, source = _cached_positions(jd_tt_q, names_key, False, None, None, None, frame=frame)
    return {k: _norm360(float(v)) for k, v in lon_map.items()}, source


# ───────────────────────────── Angles (Asc/MC) ────────────────────────
def _split_jd(jd: float) -> Tuple[float, float]:
    d = math.floor(jd)
    return d, jd - d


def _atan2d(y: float, x: float) -> float:
    if x == 0.0 and y == 0.0:
        return 0.0
    return _norm360(math.degrees(math.atan2(y, x)))


def _sind(a: float) -> float:
    return math.sin(math.radians(a))


def _cosd(a: float) -> float:
    return math.cos(math.radians(a))


def _tand(a: float) -> float:
    return math.tan(math.radians(a))


def _acotd(x: float) -> float:
    return _norm360(math.degrees(math.atan2(1.0, x)))


def _gast_deg(jd_ut1: float, jd_tt: float, warnings: List[str], seen: set[str]) -> float:
    if erfa is not None:
        try:
            d1u, d2u = _split_jd(jd_ut1)
            d1t, d2t = _split_jd(jd_tt)
            gst_rad = erfa.gst06a(d1u, d2u, d1t, d2t)
            return _norm360(math.degrees(gst_rad))
        except Exception:
            pass
    T = (float(jd_ut1) - 2451545.0) / 36525.0
    theta = 280.46061837 + 360.98564736629 * (float(jd_ut1) - 2451545.0) + 0.000387933 * (T**2) - (T**3) / 38710000.0
    _warn_add(warnings, seen, _W.ANGLES_MEEUS)
    return _norm360(theta)


def _true_obliquity_deg(jd_tt: float, warnings: List[str], seen: set[str]) -> float:
    if erfa is not None:
        try:
            d1, d2 = _split_jd(jd_tt)
            eps0 = erfa.obl06(d1, d2)
            _dpsi, deps = erfa.nut06a(d1, d2)
            return math.degrees(eps0 + deps)
        except Exception:
            pass
    T = (float(jd_tt) - 2451545.0) / 36525.0
    eps_arcsec = 84381.448 - 46.8150 * T - 0.00059 * (T**2) + 0.001813 * (T**3)
    _warn_add(warnings, seen, _W.ANGLES_MEEUS)
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
    seen: set[str],
) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    """
    Compute Ascendant and MC (ecliptic longitudes, true-of-date).

    EASTERN-ASC RULE (robust):
      Let ASC_raw be the initial intersection. Define hour-angle:
          H = wrap[-180,+180) of (RAMC − ASC_raw).
      If H < 0 → ASC_raw is WEST → flip ASC = ASC_raw + 180°.
      (Apply sidereal shift to ASC_raw & MC before the rule when in sidereal mode.)
    """
    if latitude is None or longitude is None:
        _warn_add(warnings, seen, _W.ANGLES_MISSING_GEO)
        return None, None, {}

    eps = _true_obliquity_deg(jd_tt, warnings, seen)
    gast = _gast_deg(jd_ut1, jd_tt, warnings, seen)
    ramc = _norm360(gast + float(longitude))

    # MC (true-of-date, ecliptic)
    mc = _atan2d(_sind(ramc) * _cosd(eps), _cosd(ramc))

    # ASC raw (true-of-date, ecliptic) — Meeus-derivative form
    def _acotd_safe(num: float, den: float) -> float:
        den = den if abs(den) > 1e-15 else math.copysign(1e-15, den if den != 0 else 1.0)
        return _acotd(num / den)

    asc_raw = _acotd_safe(-((_tand(float(latitude)) * _sind(eps)) + (_sind(ramc) * _cosd(eps))), _cosd(ramc))

    # Apply sidereal shift equally to both angles, if needed
    if mode == "sidereal" and ayanamsa_deg is not None:
        asc_raw = _norm360(asc_raw - float(ayanamsa_deg))
        mc      = _norm360(mc      - float(ayanamsa_deg))

    # Hour-angle test: ensure ASC is on the east
    H = ((float(ramc) - float(asc_raw) + 540.0) % 360.0) - 180.0  # ∈ (-180,+180]
    asc = asc_raw if H >= 0.0 else _norm360(float(asc_raw) + 180.0)

    # Diagnostics (post-fix forward separation MC→ASC)
    d_fwd = (float(asc) - float(mc) + 360.0) % 360.0

    dbg = {
        "eps_true_deg": float(eps),
        "gast_deg": float(gast),
        "ramc_deg": float(ramc),
        "H_deg": float(H),  # < 0 means west; we flipped
        "d_MC_to_ASC_forward_deg": float(d_fwd),
    }
    return float(asc), float(mc), dbg


# ───────────────────────────── Elevation helper ───────────────────────
def _pick_elev(p: Dict[str, Any]) -> Optional[float]:
    """
    Robustly read elevation value, preserving 0.0.
    Accept any of: elev_m, elevation_m, elevation.
    Returns float or None (if absent / blank / non-numeric).
    """
    for k in ("elev_m", "elevation_m", "elevation"):
        if k in p:
            v = p[k]
            if v is None:
                continue
            try:
                return float(str(v).strip()) if isinstance(v, str) else float(v)
            except Exception:
                continue
    return None


# ───────────────────────────── Main API ───────────────────────────────
def compute_chart(payload: Dict[str, Any]) -> Dict[str, Any]:
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")

    warnings_list: List[str] = []
    _seen: set[str] = set()

    mode = _validate_mode(payload)
    frame_raw = payload.get("frame")
    frame = (str(frame_raw).strip() if isinstance(frame_raw, str) and frame_raw.strip() else "ecliptic-of-date")

    majors_req, points_req = _split_bodies_points(payload, warnings_list, _seen)

    jd_ut, jd_tt, jd_ut1, dut1_used = _ensure_timescales(payload, warnings_list, _seen)

    topocentric = _coerce_bool(payload.get("topocentric"), False)
    if topocentric:
        lat_in = payload.get("latitude")
        lon_in = payload.get("longitude")
        elev_in = _pick_elev(payload)  # ← preserves 0.0
        if not (_is_finite(lat_in) and _is_finite(lon_in)):
            _warn_add(warnings_list, _seen, _W.TOPO_MISSING_COORDS)
            topocentric = False
            lat = lon = elev = None
        else:
            lat, lon, elev, downgraded = _validate_and_normalize_geo_for_topo(lat_in, lon_in, elev_in, warnings_list, _seen)
            if downgraded:
                topocentric = False
                lat = lon = elev = None
    else:
        lat = float(payload.get("latitude")) if _is_finite(payload.get("latitude")) else None
        lon = float(payload.get("longitude")) if _is_finite(payload.get("longitude")) else None
        elev = None

    results, source_tag = _longitudes_and_speeds(
        jd_tt,
        majors_req,
        topocentric=topocentric,
        latitude=lat,
        longitude=lon,
        elevation_m=elev,
        speed_step_days=CFG.speed_fd_step_days,
        warnings=warnings_list,
        seen=_seen,
        frame=frame,
    )

    # ── Ayanāṁśa resolution (with meta details in sidereal mode) ────────────
    ay_deg: Optional[float] = None
    aya_meta: Optional[Dict[str, Any]] = None
    if mode == "sidereal":
        ay_deg, _ = _resolve_ayanamsa(jd_tt, payload.get("ayanamsa"), warnings_list, _seen)
        try:
            from app.core.ayanamsa import get_ayanamsa_with_resolution, resolve_ayanamsa_scheme  # type: ignore
            val, canonical, is_alias, is_unknown = get_ayanamsa_with_resolution(jd_tt, payload.get("ayanamsa"))
            if ay_deg is None:
                ay_deg = float(val)
            aya_meta = {
                "requested": (None if payload.get("ayanamsa") is None else str(payload.get("ayanamsa"))),
                "canonical": str(canonical),
                "is_alias": bool(is_alias),
                "is_unknown": bool(is_unknown),
            }
        except Exception:
            try:
                from app.core.ayanamsa import resolve_ayanamsa_scheme  # type: ignore
                canonical, is_alias, is_unknown = resolve_ayanamsa_scheme(payload.get("ayanamsa"))
                aya_meta = {
                    "requested": (None if payload.get("ayanamsa") is None else str(payload.get("ayanamsa"))),
                    "canonical": str(canonical),
                    "is_alias": bool(is_alias),
                    "is_unknown": bool(is_unknown),
                }
            except Exception:
                aya_meta = None

    out_bodies: List[Dict[str, Any]] = []
    missing_bodies: List[str] = []

    def _is_num(x: Any) -> bool:
        try:
            return isinstance(x, (int, float)) and math.isfinite(float(x))
        except Exception:
            return False

    for nm in majors_req:
        tup = results.get(nm)
        lon_deg: Optional[float] = None
        speed: Optional[float] = None
        if tup:
            lon0, sp0 = tup
            lon_deg = float(lon0) if _is_num(lon0) else None
            speed = float(sp0) if _is_num(sp0) else None
        if lon_deg is None:
            geo_map, _src = _longitudes_only_geocentric(jd_tt, [nm], frame=frame)
            if nm in geo_map and _is_num(geo_map[nm]):
                lon_deg = float(geo_map[nm])
                _warn_add(warnings_list, _seen, _W.TOPO_FALLBACK_GEO, nm)
        if lon_deg is None:
            missing_bodies.append(nm)
            continue
        if mode == "sidereal" and ay_deg is not None:
            lon_deg = _norm360(lon_deg - float(ay_deg))
        out_bodies.append(
            {
                "name": nm,
                "lon": float(_norm360(lon_deg)),
                "longitude_deg": float(_norm360(lon_deg)),
                "speed": (float(speed) if _is_num(speed) else None),
                "speed_deg_per_day": (float(speed) if _is_num(speed) else None),
                "lat": None,
            }
        )

    if missing_bodies:
        _warn_add(warnings_list, _seen, _W.ADAPTER_MISS_BODIES, ", ".join(missing_bodies))

    out_points: List[Dict[str, Any]] = []
    if points_req:
        lon_map_nodes, source_nodes = _longitudes_only_geocentric(jd_tt, points_req, frame=frame)
        need_n = ("North Node" in points_req) and ("North Node" not in lon_map_nodes)
        need_s = ("South Node" in points_req) and ("South Node" not in lon_map_nodes)
        if need_n or need_s:
            if "North Node" in lon_map_nodes and need_s:
                lon_map_nodes["South Node"] = _norm360(lon_map_nodes["North Node"] + 180.0)
            elif "South Node" in lon_map_nodes and need_n:
                lon_map_nodes["North Node"] = _norm360(lon_map_nodes["South Node"] + 180.0)
            else:
                missing_pts: List[str] = []
                if need_n: missing_pts.append("North Node")
                if need_s: missing_pts.append("South Node")
                extra_map, _ = _longitudes_only_geocentric(jd_tt, missing_pts, frame=frame)
                lon_map_nodes.update(extra_map)

        for nm in points_req:
            if nm not in lon_map_nodes:
                _warn_add(warnings_list, _seen, _W.ADAPTER_MISS_POINTS, nm)
                out_points.append(
                    {
                        "name": nm,
                        "is_point": True,
                        "lon": None,
                        "longitude_deg": None,
                        "speed": None,
                        "speed_deg_per_day": None,
                        "lat": None,
                    }
                )
                continue
            lon_deg = float(lon_map_nodes[nm])
            if mode == "sidereal" and ay_deg is not None:
                lon_deg = _norm360(lon_deg - float(ay_deg))
            out_points.append(
                {
                    "name": nm,
                    "is_point": True,
                    "lon": float(_norm360(lon_deg)),
                    "longitude_deg": float(_norm360(lon_deg)),
                    "speed": None,
                    "speed_deg_per_day": None,
                    "lat": None,
                }
            )

        if source_nodes and source_nodes != source_tag:
            _warn_add(warnings_list, _seen, _W.PTS_SOURCE_MISMATCH, source_nodes)

    asc_deg, mc_deg, dbg = _compute_angles(
        jd_ut1=jd_ut1,
        jd_tt=jd_tt,
        latitude=lat,
        longitude=lon,
        mode=mode,
        ayanamsa_deg=ay_deg,
        warnings=warnings_list,
        seen=_seen,
    )

    center = "topocentric" if topocentric else "geocentric"
    meta: Dict[str, Any] = {
        "mode": mode,
        "ayanamsa_deg": float(ay_deg) if ay_deg is not None else None,
        "frame": frame,
        "center": center,
        "topocentric": bool(topocentric),
        "angles_engine": ("ERFA gst06a + true_obliquity" if erfa is not None else "Meeus fallback"),
        "angles_frame": "true-of-date",
        "source": str(source_tag),
        "module": _PROJECT_SOURCE_TAG,
        **dbg,
        "angles_east_fix": True,                     # sentinel → confirms this build
        "angles_east_rule": "hour-angle(H=RAMC-ASC_raw)≥0 ⇒ east",
        "timescales_locked": False,                  # allow DUT1-shift test to run
        "timescales": {
            "jd_utc": float(jd_ut),
            "jd_ut": float(jd_ut),   # echo for convenience
            "jd_ut1": float(jd_ut1),
            "jd_tt": float(jd_tt),
            "dut1": float(dut1_used),
        },
    }
    if aya_meta:
        meta["ayanamsa"] = aya_meta

    # Always echo observer when topocentric and lat/lon are finite; include elevation when numeric
    if topocentric and _is_finite(lat) and _is_finite(lon):
        meta["observer"] = {"latitude": float(lat), "longitude": float(lon)}
        if isinstance(elev, (int, float)):
            meta["observer"]["elevation_m"] = float(elev)

    _kpath, _kcov = _adapter_kernel_info()
    if _kpath:
        meta["ephemeris_path"] = str(_kpath)
    if _kcov and isinstance(_kcov, (tuple, list)) and len(_kcov) == 2:
        try:
            meta["ephemeris_coverage_jd"] = {"start": float(_kcov[0]), "end": float(_kcov[1])}
        except Exception:
            pass

    out: Dict[str, Any] = {
        "mode": mode,
        "ayanamsa_deg": float(ay_deg) if ay_deg is not None else None,
        "jd_ut": float(jd_ut),
        "jd_tt": float(jd_tt),
        "jd_ut1": float(jd_ut1),
        "bodies": out_bodies,
        "points": out_points,
        "angles": {
            "asc_deg": (float(asc_deg) if asc_deg is not None else None),
            "mc_deg": (float(mc_deg) if mc_deg is not None else None),
        },
        "asc_deg": (float(asc_deg) if asc_deg is not None else None),
        "mc_deg": (float(mc_deg) if mc_deg is not None else None),
        "meta": meta,
        "warnings": list(warnings_list),
    }
    return out
