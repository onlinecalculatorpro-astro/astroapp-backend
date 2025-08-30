# app/core/validators.py
from __future__ import annotations

from datetime import datetime, date, time
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo


# -----------------------------------------------------------------------------
# Error type compatible with `e.errors()` that routes expect to jsonify
# -----------------------------------------------------------------------------
class ValidationError(ValueError):
    def __init__(self, errors: List[Dict[str, Any]] | None = None):
        super().__init__("validation_error")
        self._errors = errors or []

    def add(self, field: str, msg: str, err_type: str = "value_error") -> None:
        self._errors.append({"loc": [field], "msg": msg, "type": err_type})

    def extend(self, items: List[Dict[str, Any]]) -> None:
        self._errors.extend(items)

    def errors(self) -> List[Dict[str, Any]]:
        return list(self._errors)


# -----------------------------------------------------------------------------
# Primitive validators
# -----------------------------------------------------------------------------
def _require(data: Dict[str, Any], *keys: str) -> List[Dict[str, Any]]:
    missing = [k for k in keys if k not in data]
    return [{"loc": [k], "msg": "field required", "type": "missing"} for k in missing]


def parse_date(s: Any) -> date:
    if not isinstance(s, str):
        raise ValidationError([{"loc": ["date"], "msg": "must be a string 'YYYY-MM-DD'", "type": "type_error"}])
    try:
        # Strict YYYY-MM-DD
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise ValidationError([{"loc": ["date"], "msg": "must be 'YYYY-MM-DD'", "type": "value_error.date"}])


def parse_time(s: Any) -> time:
    if not isinstance(s, str):
        raise ValidationError([{"loc": ["time"], "msg": "must be a string 'HH:MM' (24-hour)", "type": "type_error"}])
    try:
        hh, mm = s.split(":")
        hh_i, mm_i = int(hh), int(mm)
        if not (0 <= hh_i <= 23):
            raise ValueError
        if not (0 <= mm_i <= 59):
            raise ValueError
        return time(hh_i, mm_i)
    except Exception:
        raise ValidationError([{"loc": ["time"], "msg": "must be 'HH:MM' in 24-hour time", "type": "value_error.time"}])


def validate_iana_timezone(tz_str: Any) -> str:
    """
    Accept only:
      • 'UTC' exactly, or
      • IANA names with a slash (e.g., 'America/New_York', 'Europe/London') that resolve in tzdb.
    Reject:
      • Abbreviations (EST, PST, IST, etc.)
      • Offset formats (+05:30, -0700)
      • Empty / non-string
    """
    if not isinstance(tz_str, str) or not tz_str:
        raise ValidationError([{"loc": ["place_tz"], "msg": "must be a non-empty string", "type": "type_error"}])

    if tz_str == "UTC":
        return tz_str

    if "/" not in tz_str:
        # Force IANA style only
        raise ValidationError([{
            "loc": ["place_tz"],
            "msg": "must be an IANA timezone like 'Area/City' (e.g., 'America/New_York')",
            "type": "value_error.timezone"
        }])

    try:
        ZoneInfo(tz_str)  # ensure it exists
    except Exception:
        raise ValidationError([{"loc": ["place_tz"], "msg": "unknown timezone", "type": "value_error.timezone"}])

    return tz_str


def parse_latlon(lat: Any, lon: Any) -> Tuple[float, float]:
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except Exception:
        raise ValidationError([{
            "loc": ["latitude", "longitude"],
            "msg": "must be numbers",
            "type": "type_error.float"
        }])

    errs: List[Dict[str, Any]] = []
    if not (-90.0 <= lat_f <= 90.0):
        errs.append({"loc": ["latitude"], "msg": "must be between -90 and 90", "type": "value_error.number.not_in_range"})
    if not (-180.0 <= lon_f <= 180.0):
        errs.append({"loc": ["longitude"], "msg": "must be between -180 and 180", "type": "value_error.number.not_in_range"})
    if errs:
        raise ValidationError(errs)

    return lat_f, lon_f


def parse_mode(mode: Any | None) -> str:
    """
    Normalize mode; default 'sidereal'. Accept only 'sidereal' or 'tropical'.
    """
    if mode is None:
        return "sidereal"
    if not isinstance(mode, str):
        raise ValidationError([{"loc": ["mode"], "msg": "must be 'sidereal' or 'tropical'", "type": "type_error"}])
    m = mode.strip().lower()
    if m not in ("sidereal", "tropical"):
        raise ValidationError([{"loc": ["mode"], "msg": "must be 'sidereal' or 'tropical'", "type": "value_error"}])
    return m


# -----------------------------------------------------------------------------
# High-level payload parser used by endpoints
# -----------------------------------------------------------------------------
def parse_chart_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uniform parser for chart-like payloads used by /api/calculate, /predictions,
    /rectification/quick, and /api/report.

    Returns normalized dict:
        {
          "date": "YYYY-MM-DD",
          "time": "HH:MM",
          "place_tz": "Area/City" | "UTC",
          "latitude": float,
          "longitude": float,
          "mode": "sidereal" | "tropical",
          "dt": timezone-aware datetime with tzinfo
        }
    Raises ValidationError(errors=[...]) with .errors() details for 400s.
    """
    errors = _require(data, "date", "time", "place_tz", "latitude", "longitude")
    if errors:
        raise ValidationError(errors)

    # Field-by-field validation (collect as many as possible before raising)
    d_val = t_val = tz_val = None
    lat_val = lon_val = None
    mode_val = None

    field_errors: List[Dict[str, Any]] = []

    # date
    try:
        d_val = parse_date(data["date"])
    except ValidationError as e:
        field_errors.extend(e.errors())

    # time
    try:
        t_val = parse_time(data["time"])
    except ValidationError as e:
        field_errors.extend(e.errors())

    # tz
    try:
        tz_name = validate_iana_timezone(data["place_tz"])
        tz_val = ZoneInfo(tz_name if tz_name != "UTC" else "UTC")
    except ValidationError as e:
        field_errors.extend(e.errors())

    # lat/lon
    try:
        lat_val, lon_val = parse_latlon(data["latitude"], data["longitude"])
    except ValidationError as e:
        field_errors.extend(e.errors())

    # mode (optional)
    try:
        mode_val = parse_mode(data.get("mode"))
    except ValidationError as e:
        field_errors.extend(e.errors())

    if field_errors:
        raise ValidationError(field_errors)

    # Construct aware local datetime
    dt_local = datetime.combine(d_val, t_val).replace(tzinfo=tz_val)

    return {
        "date": data["date"],
        "time": data["time"],
        "place_tz": "UTC" if tz_val.key == "UTC" else tz_val.key,  # normalized
        "latitude": lat_val,
        "longitude": lon_val,
        "mode": mode_val,
        "dt": dt_local,
    }
