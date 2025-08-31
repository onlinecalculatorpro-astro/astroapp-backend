from __future__ import annotations
from dataclasses import dataclass
from datetime import date, time, datetime, timedelta
from typing import Any, Dict, List, Literal, Tuple
from zoneinfo import ZoneInfo

class ValidationError(ValueError):
    def __init__(self, details: str | Dict[str, Any] | List[Dict[str, Any]]):
        if isinstance(details, str):
            self._details: List[Dict[str, Any]] = [{"loc": [], "msg": details, "type": "value_error"}]
            super().__init__(details)
        elif isinstance(details, dict):
            self._details = [details]
            super().__init__(details.get("msg", "validation_error"))
        else:
            self._details = details
            super().__init__(self._details[0]["msg"] if self._details else "validation_error")
    def errors(self) -> List[Dict[str, Any]]:
        return list(self._details)

Zodiac = Literal["sidereal", "tropical"]
HouseSystem = Literal["placidus", "equal", "whole"]
Ayanamsa = Literal["lahiri", "krishnamurti", "raman", "fagan_bradley"]

@dataclass(frozen=True)
class BirthInput:
    date: date
    time: time
    tz: ZoneInfo
    lat: float
    lon: float
    def as_aware_datetime(self) -> datetime:
        return datetime.combine(self.date, self.time).replace(tzinfo=self.tz)

@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

@dataclass(frozen=True)
class Preferences:
    zodiac: Zodiac = "sidereal"
    ayanamsa: Ayanamsa = "lahiri"
    house_system: HouseSystem = "placidus"

@dataclass(frozen=True)
class PredictionRequest:
    birth: BirthInput
    window: Window
    topics: Tuple[str, ...]
    preferences: Preferences
    max_events: int = 20
    version: str = "predictions_v1"

def _require(obj: Dict[str, Any], key: str, ctx: List[str]) -> Any:
    if key not in obj:
        raise ValidationError({"loc": ctx + [key], "msg": "field required", "type": "value_error.missing"})
    return obj[key]

def _parse_date(s: str, loc: List[str]) -> date:
    try:
        return date.fromisoformat(s)
    except Exception:
        raise ValidationError({"loc": loc, "msg": "must be 'YYYY-MM-DD'", "type": "value_error.date"})

def _parse_time(s: str | None, loc: List[str]) -> time:
    if not s:
        return time(12, 0, 0)
    try:
        parts = s.split(":")
        if len(parts) == 2:
            hh, mm = int(parts[0]), int(parts[1])
            return time(hh, mm, 0)
        hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
        return time(hh, mm, ss)
    except Exception:
        raise ValidationError({"loc": loc, "msg": "must be 'HH:MM' or 'HH:MM:SS'", "type": "value_error.time"})

def _parse_tz(s: str, loc: List[str]) -> ZoneInfo:
    try:
        return ZoneInfo(s)
    except Exception:
        raise ValidationError({"loc": loc, "msg": "unknown IANA tz", "type": "value_error.timezone"})

def _parse_lat(v: Any, loc: List[str]) -> float:
    try:
        f = float(v)
    except Exception:
        raise ValidationError({"loc": loc, "msg": "must be a number", "type": "value_error.float"})
    if not -90.0 <= f <= 90.0:
        raise ValidationError({"loc": loc, "msg": "must be between -90 and 90", "type": "value_error.range"})
    return f

def _parse_lon(v: Any, loc: List[str]) -> float:
    try:
        f = float(v)
    except Exception:
        raise ValidationError({"loc": loc, "msg": "must be a number", "type": "value_error.float"})
    if not -180.0 <= f <= 180.0:
        raise ValidationError({"loc": loc, "msg": "must be between -180 and 180", "type": "value_error.range"})
    return f

MAX_WINDOW_DAYS = 365

def validate_prediction_payload(data: Dict[str, Any]) -> PredictionRequest:
    if not isinstance(data, dict):
        raise ValidationError("JSON body must be an object")

    birth_raw = _require(data, "birth", ["birth"])
    if not isinstance(birth_raw, dict):
        raise ValidationError({"loc": ["birth"], "msg": "must be an object", "type": "type_error.dict"})
    b_date = _parse_date(_require(birth_raw, "date", ["birth"]), ["birth", "date"])
    b_time = _parse_time(birth_raw.get("time"), ["birth", "time"])
    b_tz   = _parse_tz(_require(birth_raw, "tz", ["birth"]), ["birth", "tz"])
    b_lat  = _parse_lat(_require(birth_raw, "lat", ["birth"]), ["birth", "lat"])
    b_lon  = _parse_lon(_require(birth_raw, "lon", ["birth"]), ["birth", "lon"])
    birth  = BirthInput(date=b_date, time=b_time, tz=b_tz, lat=b_lat, lon=b_lon)

    window_raw = data.get("window", {}) or {}
    if not isinstance(window_raw, dict):
        raise ValidationError({"loc": ["window"], "msg": "must be an object", "type": "type_error.dict"})
    if "start" in window_raw or "end" in window_raw:
        start_s = _require(window_raw, "start", ["window"])
        end_s   = _require(window_raw, "end", ["window"])
        start_local = datetime.combine(_parse_date(start_s, ["window", "start"]), time(0,0,0)).replace(tzinfo=birth.tz)
        end_local   = datetime.combine(_parse_date(end_s,   ["window", "end"]),   time(23,59,59)).replace(tzinfo=birth.tz)
    else:
        days = int(window_raw.get("days_ahead", 90) or 90)
        if days < 1:
            raise ValidationError({"loc": ["window", "days_ahead"], "msg": "must be >= 1", "type": "value_error"})
        start_local = datetime.now(birth.tz).replace(microsecond=0)
        end_local   = start_local + timedelta(days=days)

    start = start_local.astimezone(ZoneInfo("UTC"))
    end   = end_local.astimezone(ZoneInfo("UTC"))
    if end <= start:
        raise ValidationError({"loc": ["window"], "msg": "end must be after start", "type": "value_error"})
    if (end - start).days > MAX_WINDOW_DAYS:
        raise ValidationError({"loc": ["window"], "msg": f"window too large; max {MAX_WINDOW_DAYS} days", "type": "value_error"})
    window = Window(start=start, end=end)

    pref_raw = data.get("preferences", {}) or {}
    if not isinstance(pref_raw, dict):
        raise ValidationError({"loc": ["preferences"], "msg": "must be an object", "type": "type_error.dict"})
    zodiac = pref_raw.get("zodiac", "sidereal")
    if zodiac not in ("sidereal", "tropical"):
        raise ValidationError({"loc": ["preferences", "zodiac"], "msg": "must be 'sidereal' or 'tropical'", "type": "value_error"})
    ayanamsa = pref_raw.get("ayanamsa", "lahiri")
    if ayanamsa not in ("lahiri", "krishnamurti", "raman", "fagan_bradley"):
        raise ValidationError({"loc": ["preferences", "ayanamsa"], "msg": "unsupported ayanamsa", "type": "value_error"})
    house_system = pref_raw.get("house_system", "placidus")
    if house_system not in ("placidus", "equal", "whole"):
        raise ValidationError({"loc": ["preferences", "house_system"], "msg": "unsupported house system", "type": "value_error"})
    preferences = Preferences(zodiac=zodiac, ayanamsa=ayanamsa, house_system=house_system)

    topics_raw = data.get("topics", [])
    if topics_raw is None:
        topics_raw = []
    if not isinstance(topics_raw, list) or not all(isinstance(t, str) for t in topics_raw):
        raise ValidationError({"loc": ["topics"], "msg": "must be an array of strings", "type": "type_error.list"})
    normalized = tuple(sorted({t.strip().lower() for t in topics_raw if t and isinstance(t, str)}))
    topics = normalized or ("career", "relationships", "health", "finance", "family", "travel", "education", "spirituality")

    max_events = int(data.get("max_events", 20) or 20)
    if max_events < 1 or max_events > 100:
        raise ValidationError({"loc": ["max_events"], "msg": "must be between 1 and 100", "type": "value_error.range"})

    return PredictionRequest(
        birth=birth,
        window=window,
        topics=topics,
        preferences=preferences,
        max_events=max_events,
    )
