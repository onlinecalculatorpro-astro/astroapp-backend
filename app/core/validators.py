# app/core/validators.py
from __future__ import annotations

from datetime import datetime, date, time
from typing import Any, Dict, Tuple, List, Optional
from zoneinfo import ZoneInfo
import os
import re


# ───────────────────────────── errors ─────────────────────────────
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


# ───────────────────────────── helpers ─────────────────────────────
def _require(data: Dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValidationError(
            [{"loc": [k], "msg": "field required", "type": "missing"} for k in missing]
        )


def _coalesce(data: Dict[str, Any], *candidates: str) -> Optional[Any]:
    """Return the first present key's value (or None)."""
    for k in candidates:
        if k in data and data[k] is not None:
            return data[k]
    return None


# ───────────────────────────── atomic parsers ─────────────────────────────
def parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise ValidationError({"loc": ["date"], "msg": "date must be 'YYYY-MM-DD'", "type": "value_error"})


def parse_time(s: str) -> time:
    """
    Accept 'HH:MM' or 'HH:MM:SS' (24h). Returns a time object with seconds if given.
    """
    try:
        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise ValueError
        hh = int(parts[0]); mm = int(parts[1]); ss = int(parts[2]) if len(parts) == 3 else 0
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
            raise ValueError
        return time(hh, mm, ss)
    except Exception:
        raise ValidationError({
            "loc": ["time"],
            "msg": "time must be 'HH:MM' or 'HH:MM:SS' (24-hour)",
            "type": "value_error"
        })


def parse_tz(tz: str) -> ZoneInfo:
    # Reject ambiguous abbreviations; allow 'UTC'
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


# House system is optional; keep permissive to match backends.
# We only normalize a couple of common aliases.
def parse_house_system(val: Any | None) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    # normalize a few aliases
    if s in ("whole-sign", "whole_sign", "wholesign", "whole sign"):
        s = "whole"
    return s


# Horizon: accept string ('short'|'medium'|'long' or '30d'/'P30D'), int days, or {"days": <int>}
def parse_horizon(value: Any | None) -> Any:
    if value is None:
        return {"days": 30}  # sensible default
    # dict {"days": N}
    if isinstance(value, dict) and "days" in value:
        try:
            days = int(value["days"])
        except Exception:
            raise ValidationError({"loc": ["horizon", "days"], "msg": "days must be an integer", "type": "type_error.integer"})
        if not (1 <= days <= 3650):
            raise ValidationError({"loc": ["horizon", "days"], "msg": "days must be between 1 and 3650", "type": "value_error"})
        return {"days": days}
    # integer-like -> days
    try:
        if isinstance(value, (int, float)) and int(value) == value:
            days = int(value)
            if 1 <= days <= 3650:
                return {"days": days}
    except Exception:
        pass
    # string forms
    s = str(value).strip().lower()
    if s in ("short", "medium", "long"):
        return s
    m = re.fullmatch(r"(?:p)?(\d{1,4})d", s)  # "30d" or "P30D"
    if m:
        days = int(m.group(1))
        if 1 <= days <= 3650:
            return {"days": days}
    raise ValidationError({"loc": ["horizon"], "msg": "horizon must be {'days': N}, '30d', 'P30D', integer days, or 'short|medium|long'", "type": "value_error"})


# ───────────────────────────── payload parsers ─────────────────────────────
def parse_chart_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes input for chart/predictions/report/rectification endpoints.

    Accepts aliases:
      - timezone ↔ place_tz
      - lat/lon ↔ latitude/longitude
      - time: 'HH:MM' or 'HH:MM:SS'
      - optional house_system (passed through if present)

    Returns:
      {
        "date": "YYYY-MM-DD",
        "time": "HH:MM[:SS]",
        "place_tz": "<IANA tz>",
        "timezone": "<same as place_tz>",
        "latitude": float,
        "longitude": float,
        "mode": "sidereal|tropical",
        "house_system": Optional[str],
        "dt": aware datetime in local tz
      }
    """
    # coalesced field lookups (keep errors mapped to canonical keys)
    place_tz_val = _coalesce(data, "place_tz", "timezone")
    lat_val = _coalesce(data, "latitude", "lat")
    lon_val = _coalesce(data, "longitude", "lon")

    # still require presence (so errors point to canonical names)
    _require({"place_tz": place_tz_val, "latitude": lat_val, "longitude": lon_val}, "place_tz", "latitude", "longitude")
    _require(data, "date", "time")

    d = parse_date(str(data["date"]))
    t = parse_time(str(data["time"]))
    tzinfo = parse_tz(str(place_tz_val))
    lat, lon = parse_latlon(lat_val, lon_val)
    mode = parse_mode(data.get("mode"))
    house_system = parse_house_system(data.get("house_system") or data.get("system"))

    # preserve seconds only if provided
    time_str = f"{t.hour:02d}:{t.minute:02d}" if t.second == 0 else f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"

    out: Dict[str, Any] = {
        "date": d.strftime("%Y-%m-%d"),
        "time": time_str,
        "place_tz": str(place_tz_val),
        "timezone": str(place_tz_val),
        "latitude": lat,
        "longitude": lon,
        "mode": mode,
        "dt": datetime.combine(d, t).replace(tzinfo=tzinfo),
    }
    if house_system:
        out["house_system"] = house_system
    return out


def parse_prediction_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    chart = parse_chart_payload(data)
    horizon = parse_horizon(data.get("horizon"))
    return chart, horizon


def parse_rectification_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    chart = parse_chart_payload(data)
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
