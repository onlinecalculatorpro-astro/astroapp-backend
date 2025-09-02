# app/core/validators.py
from __future__ import annotations

import os
import re
from datetime import datetime, date, time, timezone
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
    """Return the first present key's value (or None)."""
    for k in candidates:
        if k in data and data[k] is not None:
            return data[k]
    return None

def _fmt_hms(t: time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ───────────────────────────── atomic parsers ─────────────────────────────

def parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise ValidationError(_err("date", "date must be 'YYYY-MM-DD'", "value_error.date"))

def parse_time(s: str) -> time:
    """Accept 'HH:MM' or 'HH:MM:SS' (24h)."""
    try:
        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise ValueError
        hh, mm = int(parts[0]), int(parts[1])
        ss = int(parts[2]) if len(parts) == 3 else 0
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
            raise ValueError
        return time(hh, mm, ss)
    except Exception:
        raise ValidationError(_err("time", "time must be 'HH:MM' or 'HH:MM:SS' (24-hour)", "value_error.time"))

def parse_tz(tz: str, loc_key: str = "place_tz") -> ZoneInfo:
    """Only IANA zones or 'UTC' are accepted."""
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
    if elev is None:
        return 0.0
    try:
        e = float(elev)
    except Exception:
        raise ValidationError(_err(key, "elev_m must be a number (meters)", "type_error.float"))
    # Keep it sane: Mariana trench to Everest-ish range
    if not (-12000.0 <= e <= 9000.0):
        raise ValidationError(_err(key, "elev_m out of plausible range (-12000 .. 9000 m)"))
    return e

def parse_mode(mode: Any | None) -> Literal["sidereal", "tropical"]:
    m = (mode or os.environ.get("ASTRO_MODE") or "sidereal")
    m = str(m).strip().lower()
    if m not in ("sidereal", "tropical"):
        raise ValidationError(_err("mode", "mode must be 'sidereal' or 'tropical'", "value_error.mode"))
    return m  # type: ignore

def parse_house_system(val: Any | None) -> Optional[str]:
    """Normalize common aliases for house systems (leave validation of supported list to caller)."""
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
    aliases = {
        "geo": "geocentric",
        "geocentric": "geocentric",
        "topo": "topocentric",
        "topocentric": "topocentric",
    }
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
    """
    Acceptable forms:
      - {"days": N}, int N, "30d"/"P30D", "short"|"medium"|"long"
    """
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

def resolve_timescales_from_civil(d: date, t: time, tzinfo: ZoneInfo) -> Dict[str, float]:
    """
    Convert aware local datetime -> UTC -> JD scales using app.core.time_kernel.
    Tries multiple method names for compatibility:
      - to_jd_tt(dt_utc) OR to_jd_tt_from_utc(dt_utc)
      - to_jd_utc(dt_utc) + utc_to_tt(jd_utc)
      - to_jd(dt_utc, scale="utc"/"tt")
    """
    try:
        from app.core import time_kernel as tk  # lazy import
    except Exception as e:
        raise ValidationError(_err([], f"time kernel unavailable: {e}", "runtime_error"))

    dt_local = datetime.combine(d, t).replace(tzinfo=tzinfo)
    dt_utc = dt_local.astimezone(timezone.utc).replace(tzinfo=None)

    jd_utc: Optional[float] = None
    jd_tt: Optional[float] = None
    jd_ut: Optional[float] = None

    # Try direct JD(TT)
    for name in ("to_jd_tt", "to_jd_tt_from_utc"):
        fn = getattr(tk, name, None)
        if callable(fn):
            try:
                jd_tt = float(fn(dt_utc))
                break
            except Exception:
                pass

    # Try JD(UTC) then convert
    if jd_tt is None:
        for name in ("to_jd_utc", "to_jd_from_utc", "to_jd"):
            fn = getattr(tk, name, None)
            if callable(fn):
                try:
                    # Some to_jd need a scale
                    try:
                        jd_utc = float(fn(dt_utc, scale="utc"))  # type: ignore
                    except TypeError:
                        jd_utc = float(fn(dt_utc))  # type: ignore
                    break
                except Exception:
                    pass
        if jd_utc is not None:
            conv = getattr(tk, "utc_to_tt", None)
            if callable(conv):
                try:
                    jd_tt = float(conv(jd_utc))
                except Exception:
                    pass

    # Optional UT (not always needed)
    if jd_utc is not None:
        jd_ut = jd_utc  # if tk distinguishes, caller can override

    if jd_tt is None:
        raise ValidationError(_err([], "could not resolve jd_tt from civil time with time_kernel", "value_error.timescale"))

    out = {"jd_tt": jd_tt}
    if jd_utc is not None:
        out["jd_utc"] = jd_utc
    if jd_ut is not None:
        out["jd_ut"] = jd_ut
    return out


# ───────────────────────────── payload parsers ─────────────────────────────

class ChartPayload(TypedDict, total=False):
    date: str
    time: str
    place_tz: str
    timezone: str
    latitude: float
    longitude: float
    elev_m: float
    mode: Literal["sidereal", "tropical"]
    house_system: Optional[str]
    dt: datetime

def parse_chart_payload(data: Dict[str, Any]) -> ChartPayload:
    """Normalize input for chart/predictions/report/rectification endpoints."""
    place_tz_val = _coalesce(data, "place_tz", "tz", "timezone")
    lat_val = _coalesce(data, "latitude", "lat")
    lon_val = _coalesce(data, "longitude", "lon")
    elev_val = _coalesce(data, "elev_m", "elevation_m", "elevation")

    _require({"place_tz": place_tz_val, "latitude": lat_val, "longitude": lon_val}, "place_tz", "latitude", "longitude")
    _require(data, "date", "time")

    d = parse_date(str(data["date"]))
    t = parse_time(str(data["time"]))
    tzinfo = parse_tz(str(place_tz_val), loc_key="place_tz")
    lat, lon = parse_latlon(lat_val, lon_val)
    elev = parse_elev(elev_val)
    mode = parse_mode(data.get("mode"))
    house_system = parse_house_system(data.get("house_system") or data.get("system"))

    out: ChartPayload = {
        "date": d.strftime("%Y-%m-%d"),
        "time": _fmt_hms(t),
        "place_tz": str(place_tz_val),
        "timezone": str(place_tz_val),
        "latitude": lat,
        "longitude": lon,
        "elev_m": elev,
        "mode": mode,
        "dt": datetime.combine(d, t).replace(tzinfo=tzinfo),
    }
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
    Parse common ephemeris inputs. Behavior:
      - If jd_tt is present (number) → use it.
      - Else, if civil fields present (date, time, place_tz|tz and lat/lon when center=topocentric)
        → resolve jd_tt via time_kernel.
      - Validates/normalizes frame, center, node_model, elev_m.
      - If require_bodies, validates and (optionally) derives names.
    """
    out: EphemerisPayload = {}

    # Scales
    jd_tt = data.get("jd_tt")
    if jd_tt is not None:
        if not _is_number(jd_tt):
            raise ValidationError(_err("jd_tt", "must be a number (Julian Day TT)", "type_error.float"))
        out["jd_tt"] = float(jd_tt)
    else:
        # Try resolve from civil
        place_tz_val = _coalesce(data, "place_tz", "tz", "timezone")
        if all(k in data for k in ("date", "time")) and place_tz_val:
            d = parse_date(str(data["date"]))
            t = parse_time(str(data["time"]))
            tzinfo = parse_tz(str(place_tz_val), loc_key="place_tz")
            ts = resolve_timescales_from_civil(d, t, tzinfo)
            out.update(ts)  # includes jd_tt (+ jd_utc/jd_ut if available)
        else:
            raise ValidationError(_err("jd_tt", "missing field: jd_tt (or provide civil fields: date/time/(place_tz|tz))", "missing"))

    # Frame / Center / Node model
    out["frame"] = parse_frame(data.get("frame"))
    out["center"] = parse_center(data.get("center"))
    out["node_model"] = parse_node_model(data.get("node_model"))

    # Geo/topo coordinates
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

    # Bodies / Names
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
