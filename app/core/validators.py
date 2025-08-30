# app/core/validators.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date, time
from typing import Any, Dict, Tuple
from zoneinfo import ZoneInfo

class ValidationError(ValueError):
    pass

def _require(data: Dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValidationError(f"Missing required field(s): {', '.join(missing)}")

def parse_date(s: str) -> date:
    try:
        # Strict YYYY-MM-DD
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise ValidationError("date must be 'YYYY-MM-DD'")

def parse_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        hh, mm = int(hh), int(mm)
        if not (0 <= hh <= 23): raise ValueError
        if not (0 <= mm <= 59): raise ValueError
        return time(hh, mm)
    except Exception:
        raise ValidationError("time must be 'HH:MM' 24-hour")

def parse_tz(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except Exception:
        raise ValidationError("place_tz must be a valid IANA timezone (e.g., 'Asia/Kolkata')")

def parse_latlon(lat: Any, lon: Any) -> Tuple[float, float]:
    try:
        lat, lon = float(lat), float(lon)
    except Exception:
        raise ValidationError("latitude/longitude must be numbers")
    if not (-90.0 <= lat <= 90.0):
        raise ValidationError("latitude must be between -90 and 90")
    if not (-180.0 <= lon <= 180.0):
        raise ValidationError("longitude must be between -180 and 180")
    return lat, lon

def parse_chart_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Uniform parser used by calculate/predictions/rectification/report."""
    _require(data, "date", "time", "place_tz", "latitude", "longitude")
    d = parse_date(data["date"])
    t = parse_time(data["time"])
    tz = parse_tz(data["place_tz"])
    lat, lon = parse_latlon(data["latitude"], data["longitude"])
    # Return normalized payload (keep original strings too)
    return {
        "date": data["date"],
        "time": data["time"],
        "place_tz": data["place_tz"],
        "latitude": lat,
        "longitude": lon,
        "dt": datetime.combine(d, t).replace(tzinfo=tz),
    }
