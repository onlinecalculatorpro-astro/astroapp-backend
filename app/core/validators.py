# app/core/validators.py
from __future__ import annotations

from datetime import datetime, date, time
from typing import Any, Dict, Tuple, List
from zoneinfo import ZoneInfo
import os


class ValidationError(ValueError):
    """Structured error compatible with e.errors() used by routes.py."""
    def __init__(self, details: str | List[Dict[str, Any]]):
        if isinstance(details, str):
            self._details = [{"loc": [], "msg": details, "type": "value_error"}]
            super().__init__(details)
        else:
            # details is a list of {"loc": [...], "msg": str, "type": str}
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
        # Strict YYYY-MM-DD
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise ValidationError({"loc": ["date"], "msg": "date must be 'YYYY-MM-DD'", "type": "value_error"})


def parse_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        hh, mm = int(hh), int(mm)
        if not (0 <= hh <= 23):
            raise ValueError
        if not (0 <= mm <= 59):
            raise ValueError
        return time(hh, mm)
    except Exception:
        raise ValidationError({"loc": ["time"], "msg": "time must be 'HH:MM' 24-hour", "type": "value_error"})


def parse_tz(tz: str) -> ZoneInfo:
    # Explicitly reject abbreviations like "EST", "PST", "IST" etc.
    # Require IANA form with "/" (e.g., "America/New_York"), except allow literal "UTC".
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


def parse_chart_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uniform parser used by calculate/predictions/rectification/report.
    Returns normalized fields expected by routes/_call_compute_chart.
    """
    _require(data, "date", "time", "place_tz", "latitude", "longitude")

    d = parse_date(str(data["date"]))
    t = parse_time(str(data["time"]))
    tzinfo = parse_tz(str(data["place_tz"]))
    lat, lon = parse_latlon(data["latitude"], data["longitude"])
    mode = parse_mode(data.get("mode"))

    # Keep originals (strings) that compute_chart expects, but also give a ready datetime if needed.
    return {
        "date": d.strftime("%Y-%m-%d"),
        "time": f"{t.hour:02d}:{t.minute:02d}",
        "place_tz": str(data["place_tz"]),
        "latitude": lat,
        "longitude": lon,
        "mode": mode,
        "dt": datetime.combine(d, t).replace(tzinfo=tzinfo),
    }
