# app/core/validators.py
from __future__ import annotations

import re
from datetime import datetime, date, timedelta
from typing import Any, Dict, Tuple, List, Optional, Union, TypedDict, Literal
from zoneinfo import ZoneInfo

# ───────────────────────── errors ─────────────────────────

class ValidationError(ValueError):
    """Structured validator error compatible with routes.py (has .errors())."""
    def __init__(self, details: Union[str, Dict[str, Any], List[Dict[str, Any]]]):
        if isinstance(details, str):
            self._details = [{"loc": [], "msg": details, "type": "value_error"}]
            super().__init__(details)
        elif isinstance(details, dict):
            self._details = [details]
            super().__init__(details.get("msg", "validation_error"))
        elif isinstance(details, list):
            self._details = details
            super().__init__(self._details[0]["msg"] if self._details else "validation_error")
        else:
            self._details = [{"loc": [], "msg": "validation_error", "type": "value_error"}]
            super().__init__("validation_error")

    def errors(self) -> List[Dict[str, Any]]:
        return list(self._details)


# ───────────────────────── helpers ─────────────────────────

def _err(loc: List[str] | str, msg: str, typ: str = "value_error") -> Dict[str, Any]:
    return {"loc": [loc] if isinstance(loc, str) else loc, "msg": msg, "type": typ}

def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except Exception:
        return None

def _truthy(val: Any) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None

def _validate_iana_tz(tz: str, loc: Optional[List[str]] = None) -> str:
    # Only validate if provided; routes will default to UTC otherwise.
    try:
        _ = ZoneInfo(tz)
    except Exception:
        raise ValidationError([{
            "loc": loc or ["tz"],
            "msg": "must be a valid IANA zone like 'Asia/Kolkata'",
            "type": "value_error",
        }])
    return tz


# ───────────────────────── atomic parsers ─────────────────────────

_TIME_RE = re.compile(r"^\s*(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2})(?:\.(?P<f>\d+))?)?\s*$")

def _normalize_time_hms(s: str) -> str:
    """
    Accept 'HH:MM', 'HH:MM:SS', or 'HH:MM:SS.frac'. Allow leap-second (SS==60).
    Disallow 24:00 except exactly '24:00:00'. Return canonical 'HH:MM:SS[.frac]'.
    """
    m = _TIME_RE.match(s or "")
    if not m:
        raise ValidationError(_err("time", "time must be 'HH:MM' or 'HH:MM:SS[.frac]'", "value_error.time"))
    hh = int(m.group("h")); mm = int(m.group("m"))
    ss = int(m.group("s") or 0); frac = (m.group("f") or "")
    if not (0 <= hh <= 24 and 0 <= mm <= 59 and 0 <= ss <= 60):
        raise ValidationError(_err("time", "time fields out of range", "value_error.time"))
    if hh == 24:
        if not (mm == 0 and ss == 0 and frac == ""):
            raise ValidationError(_err("time", "24:00:00 is only allowed exactly", "value_error.time"))
        return "24:00:00"
    frac = "".join(ch for ch in frac if ch.isdigit())
    if m.group("s") is None:
        return f"{hh:02d}:{mm:02d}:00"
    return f"{hh:02d}:{mm:02d}:{ss:02d}" + (f".{frac}" if frac else "")

def parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise ValidationError(_err("date", "date must be 'YYYY-MM-DD'", "value_error.date"))

def parse_time_str(s: str) -> str:
    return _normalize_time_hms(s)

def parse_latlon(lat: Any, lon: Any, lat_key="latitude", lon_key="longitude") -> Tuple[float, float]:
    lat_f = _as_float(lat); lon_f = _as_float(lon)
    if lat_f is None or lon_f is None:
        raise ValidationError(_err([lat_key, lon_key], "latitude/longitude must be finite numbers", "type_error.float"))
    if not (-90.0 <= lat_f <= 90.0):
        raise ValidationError(_err(lat_key, "latitude must be between -90 and 90"))
    if not (-180.0 <= lon_f <= 180.0):
        raise ValidationError(_err(lon_key, "longitude must be between -180 and 180"))
    return float(lat_f), float(lon_f)

def parse_mode(mode: Any | None) -> Literal["sidereal", "tropical"]:
    m = str(mode or "tropical").strip().lower()
    if m not in ("sidereal", "tropical"):
        raise ValidationError(_err("mode", "mode must be 'tropical' or 'sidereal'", "value_error.mode"))
    return m  # type: ignore

def parse_house_system(val: Any | None) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    if s in ("whole-sign", "whole_sign", "wholesign", "whole sign"):
        return "whole"
    return s

def parse_frame(val: Any | None) -> Literal["ecliptic-of-date", "ecliptic-j2000"]:
    s = str(val or "ecliptic-of-date").strip().lower()
    aliases = {
        "ecliptic-of-date": "ecliptic-of-date",
        "ecliptic_date": "ecliptic-of-date",
        "eod": "ecliptic-of-date",
        "ecliptic-j2000": "ecliptic-j2000",
        "j2000": "ecliptic-j2000",
        "ecliptic_j2000": "ecliptic-j2000",
        "ecl-j2000": "eclipctic-j2000",  # typo-safe; corrected below
    }
    out = aliases.get(s) or s
    if out == "eclipctic-j2000":  # fix typo alias
        out = "ecliptic-j2000"
    if out not in ("ecliptic-of-date", "ecliptic-j2000"):
        raise ValidationError(_err("frame", "frame must be 'ecliptic-of-date' or 'ecliptic-j2000'"))
    return out  # type: ignore


# ───────────────────────── chart / predictions ─────────────────────────

class ChartPayload(TypedDict, total=False):
    date: str
    time: str           # canonical 'HH:MM:SS[.frac]'
    place_tz: Optional[str]
    timezone: Optional[str]
    latitude: float | None
    longitude: float | None
    elev_m: Optional[float]
    mode: Literal["sidereal", "tropical"]
    house_system: Optional[str]
    topocentric: bool
    ayanamsa: float | str | None
    dut1: float | None  # optional, numeric if present

def parse_chart_payload(body: Dict[str, Any]) -> ChartPayload:
    """
    Normalize chart inputs for /api/calculate, /api/report, /api/predictions.

    Important:
    - Require 'date' and 'time'. Do NOT require tz/lat/lon here.
      routes.py computes timescales and emits specific 422s (e.g. houses coords).
    - Validate tz only if provided (IANA). routes defaults to 'UTC' if absent.
    - Accept 'topocentric' flag without forcing coords here.
    - Pass through ayanamsa (string or number).
    - Do not clamp elevation; astronomy.py handles warnings (e.g. very_high_elevation_site).
    """
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # Required
    date_s = body.get("date")
    time_s = body.get("time")
    if not isinstance(date_s, str) or not date_s.strip():
        raise ValidationError(_err("date", "required string", "value_error"))
    if not isinstance(time_s, str) or not time_s.strip():
        raise ValidationError(_err("time", "required string", "value_error"))

    d = parse_date(date_s)
    t_str = parse_time_str(time_s)

    # Optional tz
    tz = body.get("place_tz") or body.get("tz") or body.get("timezone")
    if tz is not None:
        if not isinstance(tz, str) or not tz.strip():
            raise ValidationError(_err("place_tz", "must be a string (IANA)", "value_error"))
        _validate_iana_tz(tz.strip(), ["place_tz"])
        tz_out: Optional[str] = tz.strip()
    else:
        tz_out = None  # let routes default to UTC

    # Optional coords (not required here)
    lat = _as_float(body.get("latitude") if "latitude" in body else body.get("lat"))
    lon = _as_float(body.get("longitude") if "longitude" in body else body.get("lon"))
    elev = _as_float(body.get("elevation_m") if "elevation_m" in body else body.get("elev_m"))

    # Mode
    mode = parse_mode(body.get("mode"))

    # House system
    house_system = parse_house_system(body.get("house_system") or body.get("system"))

    # Ayanamsa
    aya = body.get("ayanamsa")
    if aya is not None and not isinstance(aya, (int, float, str)):
        raise ValidationError(_err("ayanamsa", "must be string or number", "type_error"))

    # Topocentric boolean (do not enforce coords here)
    topo = _truthy(body.get("topocentric"))
    topo = False if topo is None else topo

    # DUT1 (optional numeric; precedence handled in routes)
    dut1 = body.get("dut1")
    if dut1 is not None and _as_float(dut1) is None:
        raise ValidationError(_err("dut1", "must be a number (seconds)", "type_error.float"))

    out: ChartPayload = {
        "date": d.strftime("%Y-%m-%d"),
        "time": t_str,
        "place_tz": tz_out,
        "timezone": tz_out,
        "latitude": float(lat) if lat is not None else None,
        "longitude": float(lon) if lon is not None else None,
        "elev_m": float(elev) if elev is not None else None,
        "mode": mode,
        "house_system": house_system,
        "topocentric": bool(topo),
        "ayanamsa": aya if aya is None or isinstance(aya, (int, float)) else str(aya).strip().lower(),
        "dut1": float(dut1) if _as_float(dut1) is not None else None,
    }
    return out

def parse_prediction_payload(body: Dict[str, Any]) -> Tuple[ChartPayload, Any]:
    chart = parse_chart_payload(body)
    # horizon is routed through to predictions; validate lightly here
    horizon = body.get("horizon") or {}
    if isinstance(horizon, dict):
        if "days" in horizon and _as_float(horizon["days"]) is None:
            raise ValidationError(_err(["horizon", "days"], "must be number (days)", "value_error"))
    elif horizon is not None and not isinstance(horizon, (int, float, str)):
        raise ValidationError(_err("horizon", "must be an object, number, or string", "value_error"))
    return chart, horizon

def parse_rectification_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    # kept for completeness parity with routes
    return parse_chart_payload(body)


# ───────────────────────── ephemeris payload ─────────────────────────

class EphemerisPayload(TypedDict, total=False):
    jd_tt: float
    frame: Literal["ecliptic-of-date", "eclipctic-j2000"]  # keep original spelling in type to match parse_frame map
    bodies: List[str]
    names: List[str]

def _names_from(body: Dict[str, Any]) -> Optional[List[str]]:
    # accept bodies / names / planets; strings, numbers -> strings
    for key in ("names", "bodies", "planets"):
        val = body.get(key)
        if val is None:
            continue
        if not isinstance(val, (list, tuple)):
            raise ValidationError(_err(key, "must be an array", "type_error.list"))
        out: List[str] = []
        for i, v in enumerate(val):
            if isinstance(v, (str, int, float)):
                s = str(v).strip()
                if s:
                    out.append(s)
            elif isinstance(v, dict):
                nm = v.get("name") or v.get("body") or v.get("planet") or v.get("id") or v.get("label")
                if isinstance(nm, (str, int, float)):
                    s = str(nm).strip()
                    if s:
                        out.append(s)
                else:
                    raise ValidationError(_err([key, i], "unsupported element shape", "type_error"))
            else:
                raise ValidationError(_err([key, i], "must be string or object", "type_error"))
        return out
    return None

def parse_ephemeris_payload(body: Dict[str, Any], require_bodies: bool = False) -> EphemerisPayload:
    """
    Minimal, route-friendly parser:
      • Require jd_tt (or timescales.jd_tt if provided)
      • Normalize frame
      • Extract names/bodies list (required if require_bodies=True)
    The /api/ephemeris/longitudes route itself resolves topocentric logic & 422s.
    """
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    jd_tt = _as_float(body.get("jd_tt"))
    if jd_tt is None and isinstance(body.get("timescales"), dict):
        jd_tt = _as_float(body["timescales"].get("jd_tt"))
    if jd_tt is None:
        raise ValidationError([{"loc": ["jd_tt"], "msg": "required (or provide timescales.jd_tt)", "type": "value_error"}])

    frame = parse_frame(body.get("frame"))

    names = _names_from(body)
    if require_bodies and (not names or len(names) == 0):
        raise ValidationError(_err("bodies", "at least one body is required", "value_error"))

    # Preserve caller-provided order for both 'bodies' and 'names'
    bodies = list(names or [])
    canon = [str(n).strip() for n in bodies]

    return {"jd_tt": float(jd_tt), "frame": frame, "bodies": bodies, "names": canon}


# ───────────────────────── timescale resolver (for predictive.py) ─────────────
def resolve_timescales_from_civil_erfa(
    d: date,
    time_hh_mm_ss: str,
    place_tz: str,
) -> Dict[str, float]:
    """
    Convert a local civil (date, time, tz) into time scales used by the ephemeris.
    Returns: {"jd_tt", "jd_ut1", "jd_utc", "tz"}.

    Notes:
    • Accepts leap second (SS==60) and the special "24:00:00" end-of-day.
    • Uses Skyfield's ΔT (TT−UT1) to derive UT1; if unavailable, falls back to UT1≈UTC.
    """
    # Validate/normalize inputs
    tz = _validate_iana_tz(str(place_tz).strip())
    t_norm = parse_time_str(time_hh_mm_ss)

    # Handle 24:00:00 → next day 00:00:00
    if t_norm == "24:00:00":
        d = d + timedelta(days=1)
        hh = mm = ss = 0
        frac = "0"
    else:
        m = _TIME_RE.match(t_norm)
        assert m is not None  # guaranteed by parse_time_str
        hh = int(m.group("h"))
        mm = int(m.group("m"))
        ss = int(m.group("s") or 0)
        frac = (m.group("f") or "0")

    # Leap second 60 → build 59 and add one second later
    add_one_sec = False
    if ss == 60:
        ss = 59
        add_one_sec = True

    # Microseconds from fractional seconds
    us = int((frac + "000000")[:6])

    # Localize and convert to UTC
    dt_local = datetime(d.year, d.month, d.day, hh, mm, ss, us, tzinfo=ZoneInfo(tz))
    if add_one_sec:
        dt_local += timedelta(seconds=1)
    dt_utc = dt_local.astimezone(ZoneInfo("UTC"))

    # Skyfield times
    try:
        from skyfield.api import load
    except Exception as e:
        raise RuntimeError(f"Skyfield not installed: {e}") from e

    ts = load.timescale()
    t = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond/1e6)

    jd_tt = float(t.tt)
    jd_utc = float(t.utc_jd)

    try:
        delta_t_sec = float(t.delta_t)  # TT − UT1 (seconds)
    except Exception:
        delta_t_sec = 0.0  # fallback: UT1≈UTC if ΔT unavailable

    jd_ut1 = jd_tt - (delta_t_sec / 86400.0)

    return {"jd_tt": jd_tt, "jd_ut1": jd_ut1, "jd_utc": jd_utc, "tz": tz}


__all__ = [
    "ValidationError",
    "parse_chart_payload",
    "parse_prediction_payload",
    "parse_rectification_payload",
    "parse_ephemeris_payload",
    "parse_frame",
    "resolve_timescales_from_civil_erfa",
]
