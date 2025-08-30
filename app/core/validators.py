# app/core/validators.py
from __future__ import annotations

from datetime import datetime, date, time
from typing import Any, Dict, Tuple, List
from zoneinfo import ZoneInfo
import os


class ValidationError(ValueError):
    """Structured validator error compatible with routes.py (has .errors())."""

    def __init__(self, details: str | Dict[str, Any] | List[Dict[str, Any]]):
        if isinstance(details, str):
            self._details = [{"loc": [], "msg": details, "type": "value_error"}]
            super().__init__(details)
        elif isinstance(details, dict):
            self._details = [details]
            super().__init__(details.get("msg", "validation_error"))
        else:
            self._details = details
            super().__init__(self._details[0]["msg"] if self._details else "validation_error")

    def errors(self) -> List[Dict[str, Any]]:
        return self._details


def _require(data: Dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValidationError(
            [{"loc": [k], "msg": "field required", "type": "missing"} for k in missing]
        )


def parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise ValidationError({"loc": ["date"], "msg": "date must be 'YYYY-MM-DD'", "type": "value_error"})


def parse_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        hh, mm = int(hh), int(mm)
        if not (0 <= hh <= 23) or not (0 <= mm <= 59):
            raise ValueError
        return time(hh, mm)
    except Exception:
        raise ValidationError({"loc": ["time"], "msg": "time must be 'HH:MM' 24-hour", "type": "value_error"})


def parse_tz(tz: str) -> ZoneInfo:
    # Reject abbreviations like "EST", "PST", "IST" etc.; allow "UTC"
    if tz.upper() != "UTC" and "/" not in tz:
        raise ValidationError({
            "loc": ["place_tz"],
            "msg": "place_tz must be a valid IANA timezone (e.g., 'Asia/Kolkata')",
            "type": "value_error.timezone"
        })
    try:
        return ZoneInfo(tz)
    except Exception:
        raise ValidationError({
            "loc": ["place_tz"],
            "msg": "place_tz must be a valid IANA timezone (e.g., 'Asia/Kolkata')",
            "type": "value_error.timezone"
        })


def parse_latlon(lat: Any, lon: Any) -> Tuple[float, float]:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except Exception:
        raise ValidationError({
            "loc": ["latitude", "longitude"],
            "msg": "latitude/longitude must be numbers",
            "type": "type_error.float"
        })
    if not (-90.0 <= lat_f <= 90.0):
        raise ValidationError({"loc": ["latitude"], "msg": "latitude must be between -90 and 90", "type": "value_error"})
    if not (-180.0 <= lon_f <= 180.0):
        raise ValidationError({"loc": ["longitude"], "msg": "longitude must be between -180 and 180", "type": "value_error"})
    return lat_f, lon_f


def parse_mode(mode: Any | None) -> str:
    m = (mode or os.environ.get("ASTRO_MODE") or "sidereal").strip().lower()
    if m not in ("sidereal", "tropical"):
        raise ValidationError({"loc": ["mode"], "msg": "mode must be 'sidereal' or 'tropical'", "type": "value_error"})
    return m


def parse_horizon(value: Any | None) -> str:
    h = (value or "short").strip().lower()
    if h not in ("short", "medium", "long"):
        raise ValidationError({"loc": ["horizon"], "msg": "horizon must be 'short', 'medium', or 'long'", "type": "value_error"})
    return h


def parse_chart_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes input for chart/predictions/report/rectification endpoints.
    Returns:
      {
        "date": "YYYY-MM-DD",
        "time": "HH:MM",
        "place_tz": "<IANA tz>",
        "latitude": float,
        "longitude": float,
        "mode": "sidereal|tropical",
        "dt": aware datetime in local tz
      }
    """
    _require(data, "date", "time", "place_tz", "latitude", "longitude")

    d = parse_date(str(data["date"]))
    t = parse_time(str(data["time"]))
    tzinfo = parse_tz(str(data["place_tz"]))
    lat, lon = parse_latlon(data["latitude"], data["longitude"])
    mode = parse_mode(data.get("mode"))

    return {
        "date": d.strftime("%Y-%m-%d"),
        "time": f"{t.hour:02d}:{t.minute:02d}",
        "place_tz": str(data["place_tz"]),
        "latitude": lat,
        "longitude": lon,
        "mode": mode,
        "dt": datetime.combine(d, t).replace(tzinfo=tzinfo),
    }


def parse_prediction_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    chart = parse_chart_payload(data)
    horizon = parse_horizon(data.get("horizon"))
    return chart, horizon


def parse_rectification_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    chart = parse_chart_payload(data)
    # window_minutes optional, default sane; keep strict integer and range to avoid abuse
    wm = data.get("window_minutes", 120)
    try:
        wm = int(wm)
    except Exception:
        raise ValidationError({"loc": ["window_minutes"], "msg": "window_minutes must be an integer", "type": "type_error.integer"})
    if not (5 <= wm <= 7 * 24 * 60):
        raise ValidationError({"loc": ["window_minutes"], "msg": "window_minutes must be between 5 and 10080", "type": "value_error"})
    return chart, wm


__all__ = [
    "ValidationError",
    "parse_chart_payload",
    "parse_prediction_payload",
    "parse_rectification_payload",
]
