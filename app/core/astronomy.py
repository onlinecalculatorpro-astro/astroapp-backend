# -*- coding: utf-8 -*-
"""
High-precision planetary chart computation (ecliptic-of-date).

What's new in this rewrite (single-file; no extra modules):
- Smaller, focused helpers; thin `compute_chart`.
- Robust adapter normalization that understands many result shapes.
- Structured warnings (internally) with dedup, serialized to strings at the edge.
- Typed-ish config read once, with sane defaults and env overrides.
- Tiny TTL cache for adapter calls, keyed by (kernel tag, jd, topo tuple, names).
- Always include classic 10 majors even if the client only asks for nodes
  (can be turned off via env OCP_ALWAYS_INCLUDE_MAJORS_WITH_POINTS=0).

Public API:
    compute_chart(payload: dict) -> dict
Optional debug:
    clear_ephemeris_cache() -> None
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from functools import lru_cache
import math, os, inspect, time

__all__ = ["compute_chart", "clear_ephemeris_cache"]


# ───────────────────────────── Exceptions ─────────────────────────────
class AstronomyError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


# ───────────────────────────── Resilient imports ─────────────────────
try:
    from app.core import ephemeris_adapter as eph  # primary ephemeris
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


# ───────────────────────────── Config (single source) ─────────────────
def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name, "")
    if v == "" or v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


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


CFG = _AstroCfg(
    ayanamsa_default=(os.getenv("OCP_AYANAMSA_DEFAULT", "lahiri").strip().lower() or "lahiri"),
    jd_quant=float(os.getenv("OCP_ASTRO_JD_QUANT", "1e-7")),       # ~0.009 s
    ll_quant=float(os.getenv("OCP_ASTRO_LL_QUANT", "1e-6")),       # ~0.11 m
    elev_quant=float(os.getenv("OCP_GEO_ELEV_QUANT", "0.1")),      # 10 cm
    speed_fd_step_days=float(os.getenv("OCP_SPEED_FD_STEP_DAYS", "0.25")),  # ±6 h
    geo_soft_lat=float(os.getenv("OCP_GEO_SOFT_LAT", "89.5")),
    geo_hard_lat=float(os.getenv("OCP_GEO_HARD_LAT", "89.9")),
    elev_min=float(os.getenv("OCP_GEO_ELEV_MIN", "-500.0")),
    elev_max=float(os.getenv("OCP_GEO_ELEV_MAX", "10000.0")),
    elev_warn=float(os.getenv("OCP_GEO_ELEV_WARN", "3000.0")),
    antimer_warn_lon=float(os.getenv("OCP_GEO_ANTI_WARN", "179.9")),
    dut1_seconds=float(os.getenv("ASTRO_DUT1_BROADCAST", os.getenv("OCP_DUT1_SECONDS", "0.0"))),
    always_include_majors_with_points=_bool_env("OCP_ALWAYS_INCLUDE_MAJORS_WITH_POINTS", True),
    cache_ttl_sec=float(os.getenv("OCP_ASTRO_CACHE_TTL_SEC", "600")),  # 10 min
)


# ───────────────────────────── Constants ──────────────────────────────
_CLASSIC_10 = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
)
_EXTRA_ALLOWED = ("Ceres", "Pallas", "Juno", "Vesta", "Chiron", "North Node", "South Node")
ALLOWED_BODIES = set(_CLASSIC_10) | set(_EXTRA_ALLOWED)
_DEF_BODIES: Tuple[str, ...] = _CLASSIC_10

_NODE_CANON = {"north node": "North Node", "south node": "South Node"}
_NODE_SET_LC = set(_NODE_CANON.keys())

_PROJECT_SOURCE_TAG = "astronomy(core)"

# ───────────────────────────── Structured warnings (internal) ────────
class _W:
    AYA_FALLBACK = "ayanamsa_fallback"
    TIME_STD_FALLBACK = "timescales_fallback_local_utc_jd"
    DELTAT_CONST = "deltaT_fallback_69s"
    ANGLES_MEEUS = "angles_fallback_meeus"
    ANGLES_MISSING_GEO = "angles_missing_geography"
    LAT_CLAMP = "latitude_clamped_to_range"
    LAT_SOFT_NUDGE = "latitude_soft_nudged_from_pole"
    TOPO_DISABLED_NEAR_POLE = "topocentric_disabled_near_pole(hard)"
    ANTIM_MERIDIAN = "near_antimeridian_longitude"
    ELEV_CLAMP_MIN = "elevation_clamped_min"
    ELEV_CLAMP_MAX = "elevation_clamped_max"
    ELEV_HIGH = "very_high_elevation_site"
    PTS_NON_LIST = "points_ignored_non_list"
    ADAPTER_MISS_BODIES = "adapter_missing_bodies"
    ADAPTER_MISS_POINTS = "adapter_missing_points"
    PTS_SOURCE_MISMATCH = "points_source"
    LEAP_SECOND = "leap_second"  # explicit token


def _warn_add(store: List[str], seen: set[str], code: str, detail: Optional[str] = None) -> None:
    s = code if not detail else f"{code}({detail})"
    if s not in seen:
        seen.add(s)
        store.append(s)


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
    if isinstance(val, bool): return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"): return True
        if s in ("0", "false", "f", "no", "n", "off"): return False
    return default


def _q(x: Optional[float], q: float) -> Optional[float]:
    if x is None: return None
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
    key = str(s).strip().lower()
    return _NODE_CANON.get(key)


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
        if b not in ALLOWED_BODIES:
            allowed = ", ".join(sorted(ALLOWED_BODIES))
            raise AstronomyError("unsupported_body", f"'{b}' not supported (allowed: {allowed})")
        if str(b).strip().lower() in _NODE_SET_LC:
            canon = _canon_node_name(b)
            if canon and canon not in points:
                points.append(canon)
        else:
            majors_out.append(b)

    # If the client explicitly gave bodies but after moving nodes nothing remains,
    # optionally re-add the classic 10.
    if (not bodies_were_omitted) and (len(majors_out) == 0) and CFG.always_include_majors_with_points:
        majors_out = list(_DEF_BODIES)

    # Dedup points, keep order
    pts_seen = set(); pts_final: List[str] = []
    for p in points:
        if p not in pts_seen:
            pts_seen.add(p); pts_final.append(p)

    return majors_out, pts_final


# ───────────────────────────── Timescales ─────────────────────────────
def _detect_leap_second(time_str: Optional[str]) -> bool:
    """Return True if 'hh:mm:60' style is detected."""
    if not isinstance(time_str, str):
        return False
    try:
        hh, mm, ss = time_str.split(":")
        return int(ss.split(".")[0]) == 60
    except Exception:
        return False


def _normalize_time_for_leap_second(time_str: str) -> str:
    """
    Convert 'hh:mm:60[.fff]' → 'hh:mm:59.999999' for parsers that reject :60.
    We only nudge the seconds; ERFA will still handle the leap via warnings if in the chain.
    """
    try:
        hh, mm, ss = time_str.split(":")
        frac = ""
        if "." in ss:
            s, frac = ss.split(".", 1)
        else:
            s = ss
        if int(s) != 60:
            return time_str
        return f"{hh}:{mm}:59.999999"
    except Exception:
        return time_str


def _ensure_timescales(payload: Dict[str, Any], warnings: List[str], seen: set[str]) -> Tuple[float, float, float]:
    jd_ut  = payload.get("jd_ut") or payload.get("jd_utc")
    jd_tt  = payload.get("jd_tt")
    jd_ut1 = payload.get("jd_ut1")

    if all(isinstance(x, (int, float)) for x in (jd_ut, jd_tt, jd_ut1)):
        return float(jd_ut), float(jd_tt), float(jd_ut1)

    # 1) time_kernel first (various entrypoints)
    if _tk is not None:
        for fname in ("timescales_from_civil", "compute_timescales", "build_timescales",
                      "to_timescales", "from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                # Leap second: nudge input string + add warning token
                d = payload.get("date")
                t = payload.get("time")
                tz = payload.get("place_tz") or payload.get("tz") or "UTC"
                if _detect_leap_second(t):
                    _warn_add(warnings, seen, _W.LEAP_SECOND)
                    t = _normalize_time_for_leap_second(str(t))
                out = fn(d, t, tz, payload.get("latitude"), payload.get("longitude"))
                # Expect keys jd_ut, jd_tt, jd_ut1 (or similar)
                return float(out.get("jd_ut") or out.get("jd_utc")), float(out["jd_tt"]), float(out["jd_ut1"])

    # 2) app.core.timescales or stdlib fallback
    date_str = payload.get("date")
    time_str = payload.get("time")
    tz_str   = payload.get("place_tz") or payload.get("tz") or "UTC"
    if not isinstance(date_str, str) or not isinstance(time_str, str):
        missing = [k for k, v in (("jd_ut", jd_ut), ("jd_tt", jd_tt), ("jd_ut1", jd_ut1)) if not isinstance(v, (int, float))]
        raise AstronomyError("timescales_missing", f"Supply {', '.join(missing)} or provide date/time/tz")

    # Leap-second nudging + explicit token
    if _detect_leap_second(time_str):
        _warn_add(warnings, seen, _W.LEAP_SECOND)
        time_str = _normalize_time_for_leap_second(time_str)

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
        Y, M, D = dt_utc.year, dt_utc.month, dt_utc.day
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
        _warn_add(warnings, seen, _W.TIME_STD_FALLBACK)

    try:
        y, m = map(int, str(date_str).split("-")[:2])
    except Exception:
        y, m = 2000, 1

    if _ts is not None:
        try:
            jd_tt_calc = float(_ts.jd_tt_from_utc_jd(jd_utc, y, m))
        except Exception:
            jd_tt_calc = jd_utc + 69.0/86400.0
            _warn_add(warnings, seen, _W.DELTAT_CONST)
    else:
        jd_tt_calc = jd_utc + 69.0/86400.0
        if not used_stdlib:
            _warn_add(warnings, seen, _W.DELTAT_CONST)

    jd_ut_calc  = jd_utc
    jd_ut1_calc = jd_utc + (CFG.dut1_seconds / 86400.0)
    return jd_ut_calc, jd_tt_calc, jd_ut1_calc


# ───────────────────────────── Geo / Topocentric ──────────────────────
def _normalize_lon180(lon: float) -> float: return _wrap180(lon)

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
            _warn_add(warnings, seen, _W.ELEV_CLAMP_MIN); elev_m = CFG.elev_min
        elif elev_m > CFG.elev_max:
            _warn_add(warnings, seen, _W.ELEV_CLAMP_MAX); elev_m = CFG.elev_max
        elif abs(elev_m) >= CFG.elev_warn:
            _warn_add(warnings, seen, _W.ELEV_HIGH)

    return latf, lonf, elev_m, downgraded


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

    # Fallback: linearized around J2000 (document + warn)
    AY_J2000_DEG     = (23 + 51/60 + 26.26/3600)      # ≈ 23.857294444°
    RATE_AS_PER_YR   = 50.290966                      # arcsec/year
    Tcent            = (jd_tt_q - 2451545.0) / 36525.0
    years            = Tcent * 100.0
    base             = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
    name = (ay_key or CFG.ayanamsa_default or "lahiri").lower()
    if name in ("lahiri", "chitrapaksha", "default", "sidereal"):
        return base, "lahiri(fallback)"
    if name in ("fagan", "fagan_bradley", "fagan/bradley"):
        return base + (0.83 / 60.0), "fagan_bradley(fallback)"
    if name in ("krishnamurti", "kp"):
        return base - (20.0 / 3600.0), "krishnamurti(fallback)"
    return base, f"ayanamsa_fallback_to_lahiri({name})"


def _resolve_ayanamsa(jd_tt: float, ayanamsa: Any, warnings: List[str], seen: set[str]) -> Tuple[Optional[float], Optional[str]]:
    if ayanamsa is None or (isinstance(ayanamsa, str) and not str(ayanamsa).strip()):
        key = CFG.ayanamsa_default
    elif isinstance(ayanamsa, (int, float)):
        return float(ayanamsa), "explicit"
    else:
        key = str(ayanamsa).strip().lower()
    jd_q = _q(jd_tt, CFG.jd_quant) or jd_tt
    ay, note = _ayanamsa_deg_cached(jd_q, key)
    if note and "fallback" in note:
        _warn_add(warnings, seen, _W.AYA_FALLBACK, note)
    return float(ay), note


# ───────────────────────────── Adapter I/O & normalization ────────────
def _adapter_source_tag() -> str:
    tag = getattr(eph, "current_kernel_name", None) \
          or getattr(eph, "EPHEMERIS_NAME", None) or "adapter"
    try:
        return str(tag())
    except Exception:
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


def _unwrap_adapter_result(res: Any) -> Tuple[str, Any]:
    """Return (kind, payload) where kind in: maps|rows|rowdicts|tuples|positional|objects|empty|flat."""
    if res is None:
        return "empty", []
    if isinstance(res, dict):
        for k in ("rows", "result", "data", "payload"):
            v = res.get(k)
            if isinstance(v, (list, tuple)):
                return "rows", v
        if any(k in res for k in ("longitudes", "longitude", "lon", "velocities", "velocity", "speed", "speeds")):
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
    """Transform many shapes into a list of {name, lon, speed?} rows."""
    rows: List[Dict[str, Any]] = []
    if kind == "empty":
        return rows

    if kind in ("maps", "flat"):
        data = payload or {}
        longmaps = None
        for lk in ("longitudes", "longitude", "lon"):
            if isinstance(data.get(lk), dict):
                longmaps = data[lk]; break
        velmaps = None
        for sk in ("velocities", "velocity", "speed", "speeds"):
            if isinstance(data.get(sk), dict):
                velmaps = data[sk]; break

        if longmaps is None and kind == "flat" and all(isinstance(k, (str, int)) and isinstance(v, (int, float)) for k, v in data.items()):
            longmaps = data

        if isinstance(longmaps, dict):
            for name, lon in longmaps.items():
                if isinstance(lon, (int, float)):
                    sp = None
                    if isinstance(velmaps, dict):
                        v = velmaps.get(name)
                        sp = float(v) if isinstance(v, (int, float)) else None
                    rows.append({"name": str(name), "lon": float(lon), "speed": sp})
            return rows

    if kind in ("rows", "rowdicts"):
        for row in payload:
            if not isinstance(row, dict):
                continue
            nm = row.get("name") or row.get("body")
            if not nm:
                continue
            lon = (row.get("lon", None) if row.get("lon", None) is not None else
                   row.get("longitude", None) if row.get("longitude", None) is not None else
                   row.get("longitude_deg", None) if row.get("longitude_deg", None) is not None else
                   row.get("lambda", None))
            if lon is None or not isinstance(lon, (int, float)):
                continue
            sp = (row.get("speed", None) if isinstance(row.get("speed", None), (int, float)) else
                  row.get("velocity", None) if isinstance(row.get("velocity", None), (int, float)) else
                  row.get("speed_deg_per_day", None) if isinstance(row.get("speed_deg_per_day", None), (int, float)) else
                  row.get("lambda_dot", None) if isinstance(row.get("lambda_dot", None), (int, float)) else None)
            rows.append({"name": str(nm), "lon": float(lon), "speed": (float(sp) if sp is not None else None)})
        return rows

    if kind == "tuples":
        for item in payload:
            try:
                nm = str(item[0]); lon = float(item[1])
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
                lon = (getattr(obj, "lon", None) or getattr(obj, "longitude", None) or getattr(obj, "longitude_deg", None))
                if lon is None: continue
                sp = getattr(obj, "speed", None) or getattr(obj, "velocity", None)
                rows.append({"name": nm, "lon": float(lon), "speed": (float(sp) if isinstance(sp, (int, float)) else None)})
            except Exception:
                continue
        return rows

    return rows


def _map_rows_to_requested(rows: List[Dict[str, Any]], requested: Tuple[str, ...]) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
    want = list(requested)
    want_lc = [w.lower() for w in want]
    lon_map: Dict[str, float] = {}
    spd_map: Dict[str, Optional[float]] = {}

    by_name = {str(r.get("name", "")).lower(): r for r in rows if "name" in r}
    for nm, key in zip(want, want_lc):
        r = by_name.get(key)
        if r and isinstance(r.get("lon"), (int, float)):
            lon_map[nm] = float(r["lon"])
            spd_map[nm] = float(r["speed"]) if isinstance(r.get("speed"), (int, float)) else None

    pos_rows = [r for r in rows if "_pos" in r and isinstance(r.get("lon"), (int, float))]
    if pos_rows:
        for pr in pos_rows:
            i = int(pr["_pos"])
            if 0 <= i < len(want) and want[i] not in lon_map:
                lon_map[want[i]] = float(pr["lon"])
                spd_map[want[i]] = None

    return lon_map, spd_map


# ───────────────────────────── TTL cache around adapter ───────────────
_PosCache: Dict[Tuple[Any, ...], Tuple[float, Dict[str, float], Dict[str, Optional[float]], str]] = {}

def _ttl_get_or_compute(
    key: Tuple[Any, ...],
    ttl: float,
    compute: Callable[[], Tuple[Dict[str, float], Dict[str, Optional[float]], str]]
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
) -> Tuple[Dict[str, float], Dict[str, Optional[float]], str]:
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")

    kernel_tag = _adapter_source_tag()
    cache_key = (kernel_tag, jd_tt_q, names_key, bool(topocentric), lat_q, lon_q, elev_q)

    def _compute() -> Tuple[Dict[str, float], Dict[str, Optional[float]], str]:
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
            _geo_kwargs_for_sig(sig, topocentric=topocentric, lat_q=lat_q, lon_q=lon_q, elev_q=elev_q)
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
                    res = fn(jd_tt_q)  # positional-only jd_tt
                break
            except Exception as e:
                last_err = e
                continue

        if res is None:
            raise AstronomyError("adapter_failed", f"ephemeris adapter failed; last error: {last_err!r}")

        kind, payload = _unwrap_adapter_result(res)
        rows = _extract_rows(kind, payload)
        lon_map, spd_map = _map_rows_to_requested(rows, names_key)
        return lon_map, spd_map, kernel_tag

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
) -> Tuple[Dict[str, Tuple[float, Optional[float]]], str]:
    names_key = tuple(names)
    jd_tt_q = _q(jd_tt, CFG.jd_quant) or jd_tt
    lat_q = _q(latitude, CFG.ll_quant) if topocentric else None
    lon_q = _q(longitude, CFG.ll_quant) if topocentric else None
    elev_q = _q(elevation_m, CFG.elev_quant) if topocentric else None

    now_lon, now_spd, source = _cached_positions(jd_tt_q, names_key, topocentric, lat_q, lon_q, elev_q)

    out: Dict[str, Tuple[float, Optional[float]]] = {}
    # Fast path: all speeds present
    if all(nm in now_lon for nm in names_key) and all(now_spd.get(nm) is not None for nm in names_key):
        for nm in names_key:
            out[nm] = (_norm360(float(now_lon[nm])), float(now_spd[nm]))
        return out, source

    # Compute missing speeds by central difference reusing cache
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


def _longitudes_only_geocentric(jd_tt: float, names: List[str]) -> Tuple[Dict[str, float], str]:
    """Nodes/points: force geocentric regardless of topo request."""
    names_key = tuple(names)
    jd_tt_q = _q(jd_tt, CFG.jd_quant) or jd_tt
    lon_map, _spd_map, source = _cached_positions(jd_tt_q, names_key, False, None, None, None)
    return {k: _norm360(float(v)) for k, v in lon_map.items()}, source


# ───────────────────────────── Angles (Asc/MC) ────────────────────────
def _split_jd(jd: float) -> Tuple[float, float]:
    d = math.floor(jd); return d, jd - d

def _atan2d(y: float, x: float) -> float:
    if x == 0.0 and y == 0.0: return 0.0
    return _norm360(math.degrees(math.atan2(y, x)))

def _sind(a: float) -> float: return math.sin(math.radians(a))
def _cosd(a: float) -> float: return math.cos(math.radians(a))
def _tand(a: float) -> float: return math.tan(math.radians(a))
def _acotd(x: float) -> float: return _norm360(math.degrees(math.atan2(1.0, x)))

def _gast_deg(jd_ut1: float, jd_tt: float, warnings: List[str], seen: set[str]) -> float:
    if erfa is not None:
        try:
            d1u, d2u = _split_jd(jd_ut1)
            d1t, d2t = _split_jd(jd_tt)
            gst_rad = erfa.gst06a(d1u, d2u, d1t, d2t)
            return _norm360(math.degrees(gst_rad))
        except Exception:
            pass
    # Meeus-like fallback
    T = (float(jd_ut1) - 2451545.0) / 36525.0
    theta = 280.46061837 + 360.98564736629 * (float(jd_ut1) - 2451545.0) + 0.000387933*(T**2) - (T**3)/38710000.0
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
    # Meeus polynomial
    T = (float(jd_tt) - 2451545.0) / 36525.0
    eps_arcsec = 84381.448 - 46.8150*T - 0.00059*(T**2) + 0.001813*(T**3)
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
    if latitude is None or longitude is None:
        _warn_add(warnings, seen, _W.ANGLES_MISSING_GEO)
        return None, None, {}

    eps = _true_obliquity_deg(jd_tt, warnings, seen)
    gast = _gast_deg(jd_ut1, jd_tt, warnings, seen)
    ramc = _norm360(gast + float(longitude))

    # MC
    mc = _atan2d(_sind(ramc) * _cosd(eps), _cosd(ramc))

    # ASC: acot( - (tanφ sinε + sinRAMC cosε) / cosRAMC )
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

    warnings: List[str] = []
    _seen: set[str] = set()

    mode = _validate_mode(payload)

    # Names
    majors_req, points_req = _split_bodies_points(payload, warnings, _seen)

    # Timescales (handles jd_* direct OR civil; adds "leap_second" token if hh:mm:60 seen)
    jd_ut, jd_tt, jd_ut1 = _ensure_timescales(payload, warnings, _seen)

    # Observer
    topocentric = _coerce_bool(payload.get("topocentric"), False)
    if topocentric:
        lat, lon, elev, downgraded = _validate_and_normalize_geo_for_topo(
            payload.get("latitude"), payload.get("longitude"), payload.get("elevation_m"), warnings, _seen
        )
        if downgraded:
            topocentric = False
            lat = lon = elev = None
    else:
        lat = float(payload.get("latitude")) if _is_finite(payload.get("latitude")) else None
        lon = float(payload.get("longitude")) if _is_finite(payload.get("longitude")) else None
        elev = None

    # Majors (longitudes + speeds)
    results, source_tag = _longitudes_and_speeds(
        jd_tt, majors_req,
        topocentric=topocentric,
        latitude=lat, longitude=lon, elevation_m=elev,
        speed_step_days=CFG.speed_fd_step_days,
    )

    # Sidereal ayanāṁśa
    ay_deg: Optional[float] = None
    ay_note: Optional[str] = None
    if mode == "sidereal":
        ay_deg, ay_note = _resolve_ayanamsa(jd_tt, payload.get("ayanamsa"), warnings, _seen)

    # Build bodies list
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
        out_bodies.append({
            "name": nm,
            "lon": float(_norm360(lon_deg)),
            "longitude_deg": float(_norm360(lon_deg)),
            "speed": (float(speed) if speed is not None else None),
            "speed_deg_per_day": (float(speed) if speed is not None else None),
            "lat": None,
        })

    if missing_bodies:
        _warn_add(warnings, _seen, _W.ADAPTER_MISS_BODIES, ", ".join(missing_bodies))

    # Points (nodes): always geocentric; speed = None
    out_points: List[Dict[str, Any]] = []
    if points_req:
        lon_map_nodes, source_nodes = _longitudes_only_geocentric(jd_tt, points_req)

        # Fill counterpart by 180° complement if only one is present
        need_n = ("North Node" in points_req) and ("North Node" not in lon_map_nodes)
        need_s = ("South Node" in points_req) and ("South Node" not in lon_map_nodes)
        if need_n or need_s:
            if "North Node" in lon_map_nodes and need_s:
                lon_map_nodes["South Node"] = _norm360(lon_map_nodes["North Node"] + 180.0)
            elif "South Node" in lon_map_nodes and need_n:
                lon_map_nodes["North Node"] = _norm360(lon_map_nodes["South Node"] + 180.0)
            else:
                missing = []
                if need_n: missing.append("North Node")
                if need_s: missing.append("South Node")
                extra_map, _ = _longitudes_only_geocentric(jd_tt, missing)
                lon_map_nodes.update(extra_map)

        for nm in points_req:
            if nm not in lon_map_nodes:
                _warn_add(warnings, _seen, _W.ADAPTER_MISS_POINTS, nm)
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

        if source_nodes and source_nodes != source_tag:
            _warn_add(warnings, _seen, _W.PTS_SOURCE_MISMATCH, source_nodes)

    # Angles
    asc_deg, mc_deg, dbg = _compute_angles(
        jd_ut1=jd_ut1, jd_tt=jd_tt, latitude=lat, longitude=lon,
        mode=mode, ayanamsa_deg=ay_deg, warnings=warnings, seen=_seen,
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
        # legacy mirrors for clients
        "asc_deg": (float(asc_deg) if asc_deg is not None else None),
        "mc_deg": (float(mc_deg) if mc_deg is not None else None),
        "meta": meta,
        "warnings": warnings,  # top-level mirror of the (deduped) warning strings
    }
    return out
