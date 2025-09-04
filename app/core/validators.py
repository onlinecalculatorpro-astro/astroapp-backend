# app/core/validators.py
from __future__ import annotations

import os
import re
from datetime import datetime, date
from typing import Any, Dict, Tuple, List, Optional, Union, TypedDict, Literal
from zoneinfo import ZoneInfo

# ───────────────────────────── errors ─────────────────────────────

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
        return self._details


# ───────────────────────────── helpers ─────────────────────────────

def _err(loc: List[str] | str, msg: str, typ: str = "value_error") -> Dict[str, Any]:
    return {"loc": [loc] if isinstance(loc, str) else loc, "msg": msg, "type": typ}

def _require(data: Dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in data or data[k] is None]
    if missing:
        raise ValidationError([_err(k, "field required", "missing") for k in missing])

def _coalesce(data: Dict[str, Any], *candidates: str) -> Optional[Any]:
    for k in candidates:
        if k in data and data[k] is not None:
            return data[k]
    return None

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _to_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"): return True
        if s in ("0", "false", "f", "no", "n", "off"): return False
    return default


# ───────────────────────────── atomic parsers ─────────────────────────────

_TIME_RE = re.compile(r"^\s*(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2})(?:\.(?P<f>\d+))?)?\s*$")

def _normalize_time_hms(s: str) -> str:
    """
    Accept 'HH:MM', 'HH:MM:SS', or 'HH:MM:SS.frac'.
    Allow leap second (SS == 60).
    Disallow 24:00 except exactly '24:00:00'.
    Return canonical 'HH:MM:SS[.frac]'.
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

def parse_tz(tz: str, loc_key: str = "place_tz") -> ZoneInfo:
    if tz.upper() != "UTC" and "/" not in tz:
        raise ValidationError(_err(loc_key, "must be a valid IANA timezone (e.g., 'Asia/Kolkata')", "value_error.timezone"))
    try:
        return ZoneInfo(tz)
    except Exception:
        raise ValidationError(_err(loc_key, "must be a valid IANA timezone (e.g., 'Asia/Kolkata')", "value_error.timezone"))

def parse_latlon(lat: Any, lon: Any, lat_key="latitude", lon_key="longitude") -> Tuple[float, float]:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except Exception:
        raise ValidationError(_err([lat_key, lon_key], "latitude/longitude must be numbers", "type_error.float"))
    if not (-90.0 <= lat_f <= 90.0):
        raise ValidationError(_err(lat_key, "latitude must be between -90 and 90"))
    if not (-180.0 <= lon_f <= 180.0):
        raise ValidationError(_err(lon_key, "longitude must be between -180 and 180"))
    return lat_f, lon_f

def parse_elev(elev: Any | None, key: str = "elev_m") -> float:
    if elev is None or (isinstance(elev, str) and elev.strip() == ""):
        return 0.0
    try:
        e = float(elev)
    except Exception:
        raise ValidationError(_err(key, "elev_m must be a number (meters)", "type_error.float"))
    if not (-12000.0 <= e <= 10000.0):
        raise ValidationError(_err(key, "elev_m out of plausible range (-12000 .. 10000 m)"))
    return e

def parse_mode(mode: Any | None) -> Literal["sidereal", "tropical"]:
    m = (mode or os.environ.get("ASTRO_MODE") or "tropical")
    m = str(m).strip().lower()
    if m not in ("sidereal", "tropical"):
        raise ValidationError(_err("mode", "mode must be 'sidereal' or 'tropical'", "value_error.mode"))
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
    }
    out = aliases.get(s)
    if not out:
        raise ValidationError(_err("frame", "frame must be 'ecliptic-of-date' or 'ecliptic-j2000'"))
    return out  # type: ignore

def parse_center(val: Any | None) -> Literal["geocentric", "topocentric"]:
    s = str(val or "geocentric").strip().lower()
    aliases = {"geo": "geocentric", "geocentric": "geocentric", "topo": "topocentric", "topocentric": "topocentric"}
    out = aliases.get(s)
    if not out:
        raise ValidationError(_err("center", "center must be 'geocentric' or 'topocentric'"))
    return out  # type: ignore

def parse_node_model(val: Any | None) -> Literal["true", "mean"]:
    s = str(val or os.environ.get("OCP_NODE_MODEL") or "true").strip().lower()
    if s not in ("true", "mean"):
        raise ValidationError(_err("node_model", "node_model must be 'true' or 'mean'"))
    return s  # type: ignore

def parse_horizon(value: Any | None) -> Dict[str, int] | str:
    if value is None:
        return {"days": 30}
    if isinstance(value, dict) and "days" in value:
        try:
            days = int(value["days"])
        except Exception:
            raise ValidationError(_err(["horizon", "days"], "days must be an integer", "type_error.integer"))
        if not (1 <= days <= 3650):
            raise ValidationError(_err(["horizon", "days"], "days must be between 1 and 3650"))
        return {"days": days}
    if isinstance(value, (int, float)) and int(value) == value:
        days = int(value)
        if 1 <= days <= 3650:
            return {"days": days}
    s = str(value).strip().lower()
    if s in ("short", "medium", "long"):
        return s
    m = re.fullmatch(r"(?:p)?(\d{1,4})d", s)  # "30d" or "P30D"
    if m:
        days = int(m.group(1))
        if 1 <= days <= 3650:
            return {"days": days}
    raise ValidationError(_err("horizon", "horizon must be {'days': N}, '30d', 'P30D', integer days, or 'short|medium|long'", "value_error.horizon"))


# ───────────────────────────── timescale helpers ─────────────────────────────

def _env_dut1() -> Optional[float]:
    s = os.getenv("ASTRO_DUT1_BROADCAST", os.getenv("ASTRO_DUT1"))
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None

def resolve_timescales_from_civil_erfa(d: date, time_str: str, tzinfo: ZoneInfo) -> Dict[str, float]:
    """
    Use app.core.timescales.build_timescales(date, time, tz, dut1)
    Return: {'jd_tt', 'jd_utc', 'jd_ut'} (UT mirrors UTC if UT1 not provided).
    """
    try:
        from app.core.timescales import build_timescales  # lazy import
    except Exception as e:
        raise ValidationError(_err([], f"timescales core unavailable: {e}", "runtime_error"))

    dut1_val = _env_dut1()
    if dut1_val is None:
        # Builder needs a number; 0.0 keeps behavior deterministic when env absent.
        dut1_val = 0.0

    tz_name = getattr(tzinfo, "key", None) or "UTC"
    ts = build_timescales(d.strftime("%Y-%m-%d"), time_str, str(tz_name), float(dut1_val))

    out = {"jd_tt": float(getattr(ts, "jd_tt"))}
    if getattr(ts, "jd_utc", None) is not None:
        out["jd_utc"] = float(ts.jd_utc)
        out["jd_ut"] = float(ts.jd_utc)
    return out


# ───────────────────────────── payload parsers ─────────────────────────────

class ChartPayload(TypedDict, total=False):
    date: str
    time: str           # canonical 'HH:MM:SS[.frac]'
    place_tz: str
    timezone: str
    latitude: float | None
    longitude: float | None
    elev_m: float
    mode: Literal["sidereal", "tropical"]
    house_system: Optional[str]
    topocentric: bool
    dut1: float         # optional, validated if provided

def parse_chart_payload(data: Dict[str, Any]) -> ChartPayload:
    """
    Normalize chart inputs for /api/calculate and friends.
    - require date/time and place_tz
    - require lat/lon only when topocentric=true (or center=topocentric alias)
    - allow leap-second ':60' in time
    - dut1 optional, validated if present
    """
    place_tz_val = _coalesce(data, "place_tz", "tz", "timezone")
    _require({"place_tz": place_tz_val}, "place_tz")
    _require(data, "date", "time")

    d = parse_date(str(data["date"]))
    t_str = parse_time_str(str(data["time"]))
    tzinfo = parse_tz(str(place_tz_val), loc_key="place_tz")

    # center/topocentric flag (accept both styles)
    center = parse_center(data.get("center"))
    topocentric = _to_bool(data.get("topocentric"), center == "topocentric")

    lat_val = _coalesce(data, "latitude", "lat")
    lon_val = _coalesce(data, "longitude", "lon")
    elev_val = _coalesce(data, "elev_m", "elevation_m", "elevation")

    lat: float | None = None
    lon: float | None = None
    if topocentric:
        _require({"latitude": lat_val, "longitude": lon_val}, "latitude", "longitude")
        lat, lon = parse_latlon(lat_val, lon_val)
    else:
        if lat_val is not None and lon_val is not None:
            lat, lon = parse_latlon(lat_val, lon_val)

    elev = parse_elev(elev_val)
    mode = parse_mode(data.get("mode"))
    house_system = parse_house_system(data.get("house_system") or data.get("system"))

    out: ChartPayload = {
        "date": d.strftime("%Y-%m-%d"),
        "time": t_str,
        "place_tz": str(place_tz_val),
        "timezone": str(place_tz_val),
        "latitude": lat,
        "longitude": lon,
        "elev_m": elev,
        "mode": mode,
        "topocentric": bool(topocentric),
    }

    # dut1 (optional)
    if "dut1" in data and data["dut1"] is not None:
        try:
            dut1 = float(data["dut1"])
        except Exception:
            raise ValidationError(_err("dut1", "must be a number (seconds)", "type_error.float"))
        if abs(dut1) > 0.9:
            raise ValidationError(_err("dut1", "DUT1 magnitude must be ≤ 0.9 s", "value_error"))
        out["dut1"] = dut1

    if house_system:
        out["house_system"] = house_system

    return out

def parse_prediction_payload(data: Dict[str, Any]) -> Tuple[ChartPayload, Any]:
    chart = parse_chart_payload(data)
    horizon = parse_horizon(data.get("horizon"))
    return chart, horizon

def parse_rectification_payload(data: Dict[str, Any]) -> Tuple[ChartPayload, int]:
    chart = parse_chart_payload(data)
    wm = data.get("window_minutes", 120)
    try:
        wm = int(wm)
    except Exception:
        raise ValidationError(_err("window_minutes", "window_minutes must be an integer", "type_error.integer"))
    if not (5 <= wm <= 7 * 24 * 60):
        raise ValidationError(_err("window_minutes", "window_minutes must be between 5 and 10080"))
    return chart, wm


# ─────────────────────── ephemeris-specific payloads ───────────────────────

class EphemerisPayload(TypedDict, total=False):
    jd_tt: float
    jd_utc: float
    jd_ut: float
    frame: Literal["ecliptic-of-date", "ecliptic-j2000"]
    center: Literal["geocentric", "topocentric"]
    latitude: float
    longitude: float
    elev_m: float
    node_model: Literal["true", "mean"]
    bodies: List[str]
    names: List[str]

def normalize_bodies_and_names(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Make 'names' optional; validate length if provided."""
    bodies = data.get("bodies")
    if not isinstance(bodies, list) or not all(isinstance(b, str) and b.strip() for b in bodies):
        raise ValidationError(_err("bodies", "field required (non-empty list of strings)", "missing"))
    bodies = [b.strip() for b in bodies]
    names = data.get("names")
    if names is None:
        names = [b.replace("_", " ").title() for b in bodies]
    elif not (isinstance(names, list) and all(isinstance(n, str) for n in names)):
        raise ValidationError(_err("names", "must be a list of strings", "type_error.list"))
    elif len(names) != len(bodies):
        raise ValidationError(_err("names", "length mismatch with bodies", "value_error"))
    return bodies, names  # type: ignore

def parse_ephemeris_payload(data: Dict[str, Any], *, require_bodies: bool = False) -> EphemerisPayload:
    """
    - If jd_tt present → use it (number).
    - Else resolve from civil: require date/time and place_tz (tz alias ok).
      lat/lon required only if center/topocentric.
    - frame, center, node_model normalized.
    - if require_bodies=True, bodies must be non-empty; names optional/derived.
    """
    out: EphemerisPayload = {}
    out["frame"] = parse_frame(data.get("frame"))
    out["center"] = parse_center(data.get("center"))
    out["node_model"] = parse_node_model(data.get("node_model"))

    # timescales
    jd_tt = data.get("jd_tt")
    if jd_tt is not None:
        if not _is_number(jd_tt):
            raise ValidationError(_err("jd_tt", "must be a number (Julian Day TT)", "type_error.float"))
        out["jd_tt"] = float(jd_tt)
    else:
        place_tz_val = _coalesce(data, "place_tz", "tz", "timezone")
        _require({"place_tz": place_tz_val}, "place_tz")
        _require(data, "date", "time")
        d = parse_date(str(data["date"]))
        t_str = parse_time_str(str(data["time"]))
        tzinfo = parse_tz(str(place_tz_val), loc_key="place_tz")
        out.update(resolve_timescales_from_civil_erfa(d, t_str, tzinfo))

    # coordinates
    lat_val = _coalesce(data, "latitude", "lat")
    lon_val = _coalesce(data, "longitude", "lon")
    elev_val = _coalesce(data, "elev_m", "elevation_m", "elevation")
    if out["center"] == "topocentric":
        _require({"latitude": lat_val, "longitude": lon_val}, "latitude", "longitude")
        lat, lon = parse_latlon(lat_val, lon_val)
        out["latitude"], out["longitude"] = lat, lon
        out["elev_m"] = parse_elev(elev_val)
    else:
        if lat_val is not None and lon_val is not None:
            lat, lon = parse_latlon(lat_val, lon_val)
            out["latitude"], out["longitude"] = lat, lon
        if elev_val is not None:
            out["elev_m"] = parse_elev(elev_val)

    # bodies
    if require_bodies:
        bodies, names = normalize_bodies_and_names(data)
        out["bodies"], out["names"] = bodies, names

    return out


__all__ = [
    "ValidationError",
    "parse_chart_payload",
    "parse_prediction_payload",
    "parse_rectification_payload",
    "parse_ephemeris_payload",
    "normalize_bodies_and_names",
    "parse_frame",
    "parse_center",
    "parse_node_model",
]
