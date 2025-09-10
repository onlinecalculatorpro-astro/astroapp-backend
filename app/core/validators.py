# app/core/validators.py
from __future__ import annotations

import math
import re
from datetime import datetime, date, timedelta
from typing import Any, Dict, Tuple, List, Optional, Union, TypedDict, Literal, Sequence
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


def _err(loc: Sequence[str] | str, msg: str, typ: str = "value_error") -> Dict[str, Any]:
    return {"loc": [loc] if isinstance(loc, str) else list(loc), "msg": msg, "type": typ}


def _as_float(v: Any) -> Optional[float]:
    """Parse value as finite float. Returns None for None, NaN, inf, or invalid."""
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            # Do not treat booleans as numbers by default here.
            return float(int(v))
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _as_int(v: Any) -> Optional[int]:
    """Parse value as integer (accepts numeric strings/floats). Returns None if invalid."""
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if s == "":
            return None
        # Allow floats that are whole numbers or numeric strings (e.g. "12e3")
        f = float(s)
        if not math.isfinite(f):
            return None
        return int(f)
    except Exception:
        return None


def _truthy(val: Any) -> Optional[bool]:
    """Parse common truthy/falsey values. Returns None if unknown."""
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return bool(int(val != 0))
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _validate_iana_tz(tz: str, loc: Optional[List[str]] = None) -> str:
    """Validate IANA timezone string, raising ValidationError with location context."""
    try:
        _ = ZoneInfo(tz)
    except Exception:
        raise ValidationError(
            [
                {
                    "loc": loc or ["tz"],
                    "msg": "must be a valid IANA zone like 'Asia/Kolkata'",
                    "type": "value_error",
                }
            ]
        )
    return tz


# ───────────────────────── atomic parsers ─────────────────────────

_TIME_RE = re.compile(
    r"^\s*(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2})(?:\.(?P<f>\d+))?)?\s*$"
)


def _normalize_time_hms(s: str) -> str:
    """
    Accept 'HH:MM', 'HH:MM:SS', or 'HH:MM:SS.frac'. Allow leap second (SS==60).
    Disallow 24:00 except exactly '24:00:00'. Return canonical 'HH:MM:SS[.frac]'.
    """
    m = _TIME_RE.match(s or "")
    if not m:
        raise ValidationError(
            _err("time", "time must be 'HH:MM' or 'HH:MM:SS[.frac]'", "value_error.time")
        )
    hh = int(m.group("h"))
    mm = int(m.group("m"))
    ss = int(m.group("s") or 0)
    frac = (m.group("f") or "")

    if not (0 <= hh <= 24 and 0 <= mm <= 59 and 0 <= ss <= 60):
        raise ValidationError(_err("time", "time fields out of range", "value_error.time"))
    if hh == 24:
        if not (mm == 0 and ss == 0 and frac == ""):
            raise ValidationError(
                _err("time", "24:00:00 is only allowed exactly", "value_error.time")
            )
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


def parse_latlon(
    lat: Any, lon: Any, lat_key: str = "latitude", lon_key: str = "longitude"
) -> Tuple[float, float]:
    lat_f = _as_float(lat)
    lon_f = _as_float(lon)
    if lat_f is None or lon_f is None:
        raise ValidationError(
            _err([lat_key, lon_key], "latitude/longitude must be finite numbers", "type_error.float")
        )
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
    if s in {"whole-sign", "whole_sign", "wholesign", "whole sign"}:
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
        "ecl-j2000": "ecliptic-j2000",  # accept typo-safe alias
        "eclipctic-j2000": "ecliptic-j2000",  # common misspelling
    }
    out = aliases.get(s) or s
    if out not in ("ecliptic-of-date", "ecliptic-j2000"):
        raise ValidationError(
            _err("frame", "frame must be 'ecliptic-of-date' or 'ecliptic-j2000'")
        )
    return out  # type: ignore


def parse_earth_model(val: Any | None) -> Literal["spherical", "wgs84"]:
    """Parse earth model with validation."""
    s = str(val or "spherical").strip().lower()
    if s not in ("spherical", "wgs84"):
        raise ValidationError(_err("earth_model", "earth_model must be 'spherical' or 'wgs84'"))
    return s  # type: ignore


def parse_bodies_list(val: Any) -> List[str]:
    """Parse and validate list of celestial bodies."""
    if val is None:
        # Default major bodies for parans
        return [
            "Sun",
            "Moon",
            "Mercury",
            "Venus",
            "Mars",
            "Jupiter",
            "Saturn",
            "Uranus",
            "Neptune",
            "Pluto",
        ]

    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValidationError(_err("bodies", "body name cannot be empty", "value_error"))
        return [s]

    if not isinstance(val, (list, tuple)):
        raise ValidationError(_err("bodies", "must be array of strings or single string", "type_error.list"))

    bodies: List[str] = []
    for i, body in enumerate(val):
        if not isinstance(body, str):
            raise ValidationError(_err(["bodies", i], "must be string", "type_error.string"))
        body_name = body.strip()
        if not body_name:
            raise ValidationError(_err(["bodies", i], "body name cannot be empty", "value_error"))
        bodies.append(body_name)

    if not bodies:
        raise ValidationError(_err("bodies", "at least one body is required", "value_error"))

    return bodies


# ───────────────────────── chart / predictions (v1) ─────────────────────────
# NOTE: Kept for backward compatibility with any legacy routes that still use V1.
#       The V2 prediction engine payload lives later as parse_prediction_payload_v2.


class ChartPayload(TypedDict, total=False):
    date: str
    time: str  # canonical 'HH:MM:SS[.frac]'
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
    Normalize chart inputs for /api/calculate, /api/report, /api/predictions (v1).

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
    """
    Legacy V1 helper: returns (chart_payload, horizon).
    Kept for backward compatibility with old prediction routes only.
    """
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
    frame: Literal["ecliptic-of-date", "ecliptic-j2000"]
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
        raise ValidationError(
            [{"loc": ["jd_tt"], "msg": "required (or provide timescales.jd_tt)", "type": "value_error"}]
        )

    frame = parse_frame(body.get("frame"))

    names = _names_from(body)
    if require_bodies and (not names or len(names) == 0):
        raise ValidationError(_err("bodies", "at least one body is required", "value_error"))

    # Preserve caller-provided order for both 'bodies' and 'names'
    bodies = list(names or [])
    canon = [str(n).strip() for n in bodies]

    return {"jd_tt": float(jd_tt), "frame": frame, "bodies": bodies, "names": canon}


# ───────────────────────── progressions payload (v1) ─────────────────────────

ProgressionsMethod = Literal["secondary", "minor", "tertiary"]
TertiaryMode = Literal["day-for-month", "lunar-day-for-year"]
ZodiacMode = Literal["tropical", "sidereal"]
LunarMonthKind = Literal["synodic", "sidereal"]


class NatalProgressions(TypedDict, total=False):
    date: str
    time: str
    place_tz: str
    latitude: float
    longitude: float
    elev_m: float


class TargetProgressions(TypedDict, total=False):
    date: str
    time: str
    place_tz: str


class PlaceOverride(TypedDict, total=False):
    latitude: float
    longitude: float
    elev_m: float


class ProgressionsPayload(TypedDict, total=False):
    natal: NatalProgressions
    method: ProgressionsMethod
    target: TargetProgressions
    years_after: float
    jd_tt_natal: float
    jd_ut1_natal: float
    place: PlaceOverride
    frame: Literal["ecliptic-of-date", "ecliptic-j2000"]
    house_system: str
    zodiac_mode: ZodiacMode
    ayanamsa_deg: float
    lunar_month: LunarMonthKind
    tertiary_mode: TertiaryMode
    aspects_to_natal: bool
    orbs: Dict[str, float]
    parallels: bool
    antiscia: bool
    profile: bool
    validation: Literal["none", "basic"]


def _parse_progressions_natal(natal: Any) -> NatalProgressions:
    if not isinstance(natal, dict):
        raise ValidationError(_err("natal", "required object", "type_error.dict"))

    out: NatalProgressions = {}
    # natal date/time/tz are required IF strict timescales are not supplied at top-level
    if "date" in natal:
        out["date"] = parse_date(str(natal["date"])).strftime("%Y-%m-%d")
    if "time" in natal:
        out["time"] = parse_time_str(str(natal["time"]))
    if "place_tz" in natal:
        out["place_tz"] = _validate_iana_tz(str(natal["place_tz"]).strip(), ["natal", "place_tz"])

    # optional coords
    lat = natal.get("latitude")
    lon = natal.get("longitude")
    elev = natal.get("elev_m")
    if lat is not None or lon is not None:
        la, lo = parse_latlon(lat, lon, "natal.latitude", "natal.longitude")
        out["latitude"] = la
        out["longitude"] = lo
    if elev is not None and _as_float(elev) is None:
        raise ValidationError(_err(["natal", "elev_m"], "must be number", "type_error.float"))
    if elev is not None:
        out["elev_m"] = float(elev)

    return out


def _parse_progressions_target(target: Any) -> TargetProgressions:
    if not isinstance(target, dict):
        raise ValidationError(_err("target", "must be object", "type_error.dict"))
    if "date" not in target or "time" not in target or ("place_tz" not in target and "timezone" not in target):
        raise ValidationError(_err("target", "requires date, time, place_tz", "value_error"))
    tz_val = target.get("place_tz") or target.get("timezone")
    return {
        "date": parse_date(str(target["date"])).strftime("%Y-%m-%d"),
        "time": parse_time_str(str(target["time"])),
        "place_tz": _validate_iana_tz(str(tz_val).strip(), ["target", "place_tz"]),
    }


def _parse_place_override(place: Any) -> PlaceOverride:
    if not isinstance(place, dict):
        raise ValidationError(_err("place", "must be object", "type_error.dict"))
    la, lo = parse_latlon(place.get("latitude"), place.get("longitude"), "place.latitude", "place.longitude")
    out: PlaceOverride = {"latitude": la, "longitude": lo}
    if place.get("elev_m") is not None:
        elev = _as_float(place.get("elev_m"))
        if elev is None:
            raise ValidationError(_err(["place", "elev_m"], "must be number", "type_error.float"))
        out["elev_m"] = float(elev)
    return out


def parse_progressions_payload(body: Dict[str, Any]) -> ProgressionsPayload:
    """
    Validate + normalize payload for /api/progressions (v1).
    """
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # natal (always present, but date/time/tz are optional if strict timescales provided)
    natal_raw = body.get("natal")
    if natal_raw is None:
        raise ValidationError(_err("natal", "required object", "value_error"))
    natal = _parse_progressions_natal(natal_raw)

    # strict natal timescales (optional, must be both if provided)
    jd_tt_natal = _as_float(body.get("jd_tt_natal"))
    jd_ut1_natal = _as_float(body.get("jd_ut1_natal"))
    if (jd_tt_natal is None) ^ (jd_ut1_natal is None):
        raise ValidationError(_err(["jd_tt_natal", "jd_ut1_natal"], "supply both or neither", "value_error"))
    have_strict_ts = jd_tt_natal is not None and jd_ut1_natal is not None

    # if no strict timescales, ensure we have natal date/time/place_tz
    if not have_strict_ts:
        for key in ("date", "time", "place_tz"):
            if key not in natal:
                raise ValidationError(
                    _err(["natal", key], "required when strict timescales are not supplied", "value_error")
                )

    # method
    method_raw = (body.get("method") or "secondary").strip().lower()
    if method_raw not in ("secondary", "minor", "tertiary"):
        raise ValidationError(_err("method", "must be 'secondary', 'minor', or 'tertiary'", "value_error"))
    method: ProgressionsMethod = method_raw  # type: ignore

    # target vs years_after (one required)
    target_raw = body.get("target")
    years_after = _as_float(body.get("years_after"))
    target: Optional[TargetProgressions] = None
    if target_raw is not None:
        target = _parse_progressions_target(target_raw)
    if target is None and years_after is None:
        raise ValidationError(_err(["target", "years_after"], "provide either target or years_after", "value_error"))

    # place override (optional)
    place_override: Optional[PlaceOverride] = None
    if body.get("place") is not None:
        place_override = _parse_place_override(body["place"])

    # frame / house system / zodiac mode
    frame = parse_frame(body.get("frame"))
    house_system = parse_house_system(body.get("house_system")) or "placidus"
    zodiac_mode = parse_mode(body.get("zodiac_mode") or body.get("mode"))

    # ayanamsa_deg (float, default 0.0)
    aya = body.get("ayanamsa_deg", 0.0)
    aya_f = _as_float(aya)
    if aya_f is None:
        raise ValidationError(_err("ayanamsa_deg", "must be a number", "type_error.float"))
    ayanamsa_deg = float(aya_f)

    # lunar_month (for minor)
    lunar_month_raw = (body.get("lunar_month") or "synodic").strip().lower()
    if lunar_month_raw not in ("synodic", "sidereal"):
        raise ValidationError(_err("lunar_month", "must be 'synodic' or 'sidereal'", "value_error"))
    lunar_month: LunarMonthKind = lunar_month_raw  # type: ignore

    # tertiary_mode
    tertiary_mode_raw = (body.get("tertiary_mode") or "day-for-month").strip().lower()
    if tertiary_mode_raw not in ("day-for-month", "lunar-day-for-year"):
        raise ValidationError(_err("tertiary_mode", "must be 'day-for-month' or 'lunar-day-for-year'", "value_error"))
    tertiary_mode: TertiaryMode = tertiary_mode_raw  # type: ignore

    # booleans
    aspects_to_natal = _truthy(body.get("aspects_to_natal"))
    aspects_to_natal = True if aspects_to_natal is None else aspects_to_natal
    parallels = bool(_truthy(body.get("parallels")) or False)
    antiscia = bool(_truthy(body.get("antiscia")) or False)
    profile = bool(_truthy(body.get("profile")) or False)

    # validation
    validation = str(body.get("validation") or "basic").strip().lower()
    if validation not in ("none", "basic"):
        raise ValidationError(_err("validation", "must be 'none' or 'basic'", "value_error"))

    # orbs (optional dict[str,float])
    orbs_obj = body.get("orbs")
    if orbs_obj is not None:
        if not isinstance(orbs_obj, dict):
            raise ValidationError(_err("orbs", "must be object", "type_error.dict"))
        new_orbs: Dict[str, float] = {}
        for k, v in orbs_obj.items():
            f = _as_float(v)
            if f is not None:
                new_orbs[str(k)] = float(f)
        orbs = new_orbs
    else:
        orbs = None

    out: ProgressionsPayload = {
        "natal": natal,
        "method": method,
        "target": target if target is not None else {},  # keep key presence predictable
        "years_after": float(years_after) if years_after is not None else None,  # type: ignore
        "jd_tt_natal": float(jd_tt_natal) if jd_tt_natal is not None else None,  # type: ignore
        "jd_ut1_natal": float(jd_ut1_natal) if jd_ut1_natal is not None else None,  # type: ignore
        "place": place_override or {},  # empty if absent
        "frame": frame,
        "house_system": house_system,
        "zodiac_mode": zodiac_mode,  # "tropical" | "sidereal"
        "ayanamsa_deg": ayanamsa_deg,
        "lunar_month": lunar_month,
        "tertiary_mode": tertiary_mode,
        "aspects_to_natal": bool(aspects_to_natal),
        "orbs": orbs or {},  # empty if absent
        "parallels": bool(parallels),
        "antiscia": bool(antiscia),
        "profile": bool(profile),
        "validation": validation,  # "none" | "basic"
    }
    return out


# ───────────────────────── returns (v1) ─────────────────────────

ReturnKind = Literal["solar", "lunar"]


class NatalReturns(TypedDict, total=False):
    date: str
    time: str
    place_tz: str
    latitude: float
    longitude: float
    elev_m: float


class ReturnsPayload(TypedDict, total=False):
    natal: NatalReturns
    kind: ReturnKind
    jd_tt_natal: float
    jd_ut1_natal: float
    place: PlaceOverride
    frame: Literal["ecliptic-of-date", "ecliptic-j2000"]
    house_system: str
    zodiac_mode: ZodiacMode
    ayanamsa_deg: float
    lunar_month: LunarMonthKind  # used for lunar seed length (sidereal|synodic); default "sidereal"
    guess_years_offset: int | None
    around_jd_tt: float | None
    tol_arcmin: float
    max_iters: int
    estimate_uncertainty: bool
    fd_step_minutes: float
    profile: bool
    validation: Literal["none", "basic", "extended"]
    validation_residual_arcmin: float


def _parse_returns_natal(natal: Any) -> NatalReturns:
    """Parse natal chart data for returns, allowing flexible field requirements"""
    if not isinstance(natal, dict):
        raise ValidationError(_err("natal", "required object", "type_error.dict"))

    out: NatalReturns = {}

    # Only validate fields that are actually present
    if "date" in natal:
        date_val = natal["date"]
        if not isinstance(date_val, str) or not date_val.strip():
            raise ValidationError(_err(["natal", "date"], "must be non-empty string", "value_error"))
        out["date"] = parse_date(str(date_val)).strftime("%Y-%m-%d")

    if "time" in natal:
        time_val = natal["time"]
        if not isinstance(time_val, str) or not time_val.strip():
            raise ValidationError(_err(["natal", "time"], "must be non-empty string", "value_error"))
        out["time"] = parse_time_str(str(time_val))

    if "place_tz" in natal:
        tz_val = natal["place_tz"]
        if not isinstance(tz_val, str) or not tz_val.strip():
            raise ValidationError(_err(["natal", "place_tz"], "must be non-empty string", "value_error"))
        out["place_tz"] = _validate_iana_tz(str(tz_val).strip(), ["natal", "place_tz"])

    # Optional coordinates - only validate if present
    lat = natal.get("latitude")
    lon = natal.get("longitude")
    if lat is not None or lon is not None:
        try:
            la, lo = parse_latlon(lat, lon, "natal.latitude", "natal.longitude")
            out["latitude"] = la
            out["longitude"] = lo
        except ValidationError as e:
            # Re-raise with proper location context
            details = e.errors()
            for detail in details:
                if detail["loc"]:
                    detail["loc"] = ["natal"] + detail["loc"]
            raise ValidationError(details)

    if "elev_m" in natal:
        elev = _as_float(natal.get("elev_m"))
        if elev is None:
            raise ValidationError(_err(["natal", "elev_m"], "must be number", "type_error.float"))
        out["elev_m"] = float(elev)

    return out


def parse_returns_payload(body: Dict[str, Any]) -> ReturnsPayload:
    """
    Validate + normalize payload for /api/return (solar/lunar) (v1).
    """
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # Parse natal chart data
    natal_raw = body.get("natal")
    if natal_raw is None:
        raise ValidationError(_err("natal", "required object", "value_error"))

    natal = _parse_returns_natal(natal_raw)

    # Strict timescales (both or none)
    jd_tt_natal = _as_float(body.get("jd_tt_natal"))
    jd_ut1_natal = _as_float(body.get("jd_ut1_natal"))

    if (jd_tt_natal is None) ^ (jd_ut1_natal is None):
        raise ValidationError(_err(["jd_tt_natal", "jd_ut1_natal"], "supply both or neither", "value_error"))

    have_strict_timescales = jd_tt_natal is not None and jd_ut1_natal is not None

    # If no strict timescales, we need natal date/time/place_tz
    if not have_strict_timescales:
        missing_fields = []
        for required_field in ["date", "time", "place_tz"]:
            if required_field not in natal:
                missing_fields.append(f"natal.{required_field}")

        if missing_fields:
            raise ValidationError(_err(missing_fields, "required strings", "value_error"))

    # Return kind validation
    raw_kind = (body.get("kind") or body.get("type") or "solar").strip().lower()
    if raw_kind not in ("solar", "lunar"):
        raise ValidationError(_err("kind", "must be 'solar' or 'lunar'", "value_error"))
    kind: ReturnKind = raw_kind  # type: ignore

    # Optional place override
    place_override: Optional[PlaceOverride] = None
    if body.get("place") is not None:
        place_override = _parse_place_override(body["place"])

    # Frame parsing
    frame_val = body.get("frame")
    if frame_val is not None:
        frame = parse_frame(frame_val)
    else:
        frame = "ecliptic-of-date"  # Default value

    # House system and zodiac mode
    house_system = parse_house_system(body.get("house_system")) or "placidus"
    zodiac_mode = parse_mode(body.get("zodiac_mode") or body.get("mode"))

    # Ayanamsa handling
    aya = _as_float(body.get("ayanamsa_deg", 0.0))
    if aya is None:
        raise ValidationError(_err("ayanamsa_deg", "must be a number", "type_error.float"))
    ayanamsa_deg = float(aya)

    # Lunar month for lunar returns
    lm = (body.get("lunar_month") or "sidereal").strip().lower()
    if lm not in ("sidereal", "synodic"):
        raise ValidationError(_err("lunar_month", "must be 'sidereal' or 'synodic'", "value_error"))
    lunar_month: LunarMonthKind = lm  # type: ignore

    # Seed values for return calculation
    guess_years_offset = _as_int(body.get("guess_years_offset"))
    around_jd_tt = _as_float(body.get("around_jd_tt"))

    # Numerical parameters with validation
    tol_arcmin = _as_float(body.get("tol_arcmin", 1.0))
    if tol_arcmin is None or tol_arcmin <= 0:
        raise ValidationError(_err("tol_arcmin", "must be > 0", "value_error"))

    max_iters = _as_int(body.get("max_iters", 12))
    if max_iters is None or max_iters < 1:
        raise ValidationError(_err("max_iters", "must be >= 1", "value_error"))

    fd_step_minutes = _as_float(body.get("fd_step_minutes", 2.0))
    if fd_step_minutes is None or fd_step_minutes <= 0:
        raise ValidationError(_err("fd_step_minutes", "must be > 0", "value_error"))

    # Boolean flags
    est_unc = _truthy(body.get("estimate_uncertainty"))
    estimate_uncertainty = True if est_unc is None else bool(est_unc)
    profile = bool(_truthy(body.get("profile")) or False)

    # Validation level
    validation = str(body.get("validation") or "basic").strip().lower()
    if validation not in ("none", "basic", "extended"):
        raise ValidationError(_err("validation", "must be 'none', 'basic', or 'extended'", "value_error"))

    v_resid = _as_float(body.get("validation_residual_arcmin", 1.0))
    if v_resid is None or v_resid <= 0:
        raise ValidationError(_err("validation_residual_arcmin", "must be > 0", "value_error"))

    # Build output payload
    out: ReturnsPayload = {
        "natal": natal,
        "kind": kind,
        "jd_tt_natal": float(jd_tt_natal) if jd_tt_natal is not None else None,  # type: ignore
        "jd_ut1_natal": float(jd_ut1_natal) if jd_ut1_natal is not None else None,  # type: ignore
        "place": place_override or {},
        "frame": frame,
        "house_system": house_system,
        "zodiac_mode": zodiac_mode,
        "ayanamsa_deg": ayanamsa_deg,
        "lunar_month": lunar_month,
        "guess_years_offset": guess_years_offset,
        "around_jd_tt": float(around_jd_tt) if around_jd_tt is not None else None,  # type: ignore
        "tol_arcmin": float(tol_arcmin),
        "max_iters": int(max_iters),
        "estimate_uncertainty": bool(estimate_uncertainty),
        "fd_step_minutes": float(fd_step_minutes),
        "profile": bool(profile),
        "validation": validation,
        "validation_residual_arcmin": float(v_resid),
    }
    return out


# ───────────────────────── parans (v1) ─────────────────────────


class SubjectParans(TypedDict, total=False):
    """Subject data for timescale resolution in parans."""
    date: str
    time: str
    place_tz: str


class PlaceParans(TypedDict, total=False):
    """Place coordinates for paran calculations."""
    latitude: float
    longitude: float
    elev_m: float


class ParansPayload(TypedDict, total=False):
    """Complete payload structure for /api/parans endpoint."""
    subject: SubjectParans
    place: PlaceParans
    jd_tt_ref: float
    jd_ut1_ref: float
    frame: Literal["ecliptic-of-date", "ecliptic-j2000"]
    zodiac_mode: ZodiacMode
    ayanamsa_deg: float
    bodies: List[str]
    tolerance_minutes: float
    search_window_days: float
    max_iters: int
    fd_step_minutes: float
    earth_model: Literal["spherical", "wgs84"]
    apply_refraction: bool
    pressure_hPa: float
    temperature_C: float
    profile: bool
    validation: Literal["none", "basic"]


def _parse_parans_subject(subject: Any) -> SubjectParans:
    """Parse subject data for parans timescale resolution."""
    if not isinstance(subject, dict):
        raise ValidationError(_err("subject", "required object", "type_error.dict"))

    out: SubjectParans = {}

    # Only validate fields that are present
    if "date" in subject:
        date_val = subject["date"]
        if not isinstance(date_val, str) or not date_val.strip():
            raise ValidationError(_err(["subject", "date"], "must be non-empty string", "value_error"))
        out["date"] = parse_date(str(date_val)).strftime("%Y-%m-%d")

    if "time" in subject:
        time_val = subject["time"]
        if not isinstance(time_val, str) or not time_val.strip():
            raise ValidationError(_err(["subject", "time"], "must be non-empty string", "value_error"))
        out["time"] = parse_time_str(str(time_val))

    if "place_tz" in subject:
        tz_val = subject["place_tz"]
        if not isinstance(tz_val, str) or not tz_val.strip():
            raise ValidationError(_err(["subject", "place_tz"], "must be non-empty string", "value_error"))
        out["place_tz"] = _validate_iana_tz(str(tz_val).strip(), ["subject", "place_tz"])

    return out


def _parse_parans_place(place: Any) -> PlaceParans:
    """Parse place coordinates for paran calculations."""
    if not isinstance(place, dict):
        raise ValidationError(_err("place", "required object", "type_error.dict"))

    # Latitude and longitude are required
    lat = place.get("latitude")
    lon = place.get("longitude")
    if lat is None or lon is None:
        raise ValidationError(_err("place", "latitude and longitude are required", "value_error"))

    try:
        la, lo = parse_latlon(lat, lon, "place.latitude", "place.longitude")
    except ValidationError as e:
        # Re-raise with proper location context
        details = e.errors()
        for detail in details:
            if detail["loc"] and detail["loc"][0] not in ["place.latitude", "place.longitude"]:
                detail["loc"] = ["place"] + detail["loc"]
        raise ValidationError(details)

    out: PlaceParans = {"latitude": la, "longitude": lo}

    # Optional elevation
    if "elev_m" in place:
        elev = _as_float(place.get("elev_m"))
        if elev is None:
            raise ValidationError(_err(["place", "elev_m"], "must be number", "type_error.float"))
        out["elev_m"] = float(elev)

    return out


def parse_parans_payload(body: Dict[str, Any]) -> ParansPayload:
    """
    Validate and normalize payload for /api/parans endpoint.
    """
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # Parse place data - required for horizon calculations
    place_raw = body.get("place")
    if place_raw is None:
        raise ValidationError(_err("place", "required object with latitude/longitude", "value_error"))
    place = _parse_parans_place(place_raw)

    # Strict timescales (both or neither)
    jd_tt_ref = _as_float(body.get("jd_tt_ref"))
    jd_ut1_ref = _as_float(body.get("jd_ut1_ref"))

    if (jd_tt_ref is None) ^ (jd_ut1_ref is None):
        raise ValidationError(_err(["jd_tt_ref", "jd_ut1_ref"], "supply both or neither", "value_error"))

    have_strict_timescales = jd_tt_ref is not None and jd_ut1_ref is not None

    # Subject data - only required if strict timescales not provided
    subject_raw = body.get("subject")
    subject: SubjectParans | Dict[str, Any] = {}

    if have_strict_timescales:
        if subject_raw is not None:
            subject = _parse_parans_subject(subject_raw)
    else:
        if subject_raw is None:
            raise ValidationError(_err("subject", "required when strict timescales not provided", "value_error"))
        subject = _parse_parans_subject(subject_raw)
        missing_fields = []
        for required_field in ["date", "time", "place_tz"]:
            if required_field not in subject:
                missing_fields.append(f"subject.{required_field}")
        if missing_fields:
            raise ValidationError(_err(missing_fields, "required when strict timescales not provided", "value_error"))

    # Frame and coordinate system parameters
    frame = parse_frame(body.get("frame"))
    zodiac_mode = parse_mode(body.get("zodiac_mode") or body.get("mode"))

    # Ayanamsa
    aya = _as_float(body.get("ayanamsa_deg", 0.0))
    if aya is None:
        raise ValidationError(_err("ayanamsa_deg", "must be a number", "type_error.float"))
    ayanamsa_deg = float(aya)

    # Bodies list
    bodies = parse_bodies_list(body.get("bodies"))

    # Tolerance and search parameters
    tolerance_min = _as_float(body.get("tolerance_minutes", 4.0))
    if tolerance_min is None or tolerance_min <= 0:
        raise ValidationError(_err("tolerance_minutes", "must be > 0", "value_error"))

    window_days = _as_float(body.get("search_window_days", 1.0))
    if window_days is None or window_days <= 0:
        raise ValidationError(_err("search_window_days", "must be > 0", "value_error"))

    # Iteration parameters
    max_iters = _as_int(body.get("max_iters", 10))
    if max_iters is None or max_iters < 1:
        raise ValidationError(_err("max_iters", "must be >= 1", "value_error"))

    fd_step_min = _as_float(body.get("fd_step_minutes", 2.0))
    if fd_step_min is None or fd_step_min <= 0:
        raise ValidationError(_err("fd_step_minutes", "must be > 0", "value_error"))

    # Earth model and atmospheric parameters
    earth_model = parse_earth_model(body.get("earth_model"))
    apply_refraction = bool(_truthy(body.get("apply_refraction")) or False)

    pressure = _as_float(body.get("pressure_hPa", 1010.0))
    if pressure is None or pressure <= 0:
        raise ValidationError(_err("pressure_hPa", "must be > 0", "value_error"))

    temperature = _as_float(body.get("temperature_C", 10.0))
    if temperature is None:
        raise ValidationError(_err("temperature_C", "must be a number", "type_error.float"))

    # Diagnostic flags
    profile = bool(_truthy(body.get("profile")) or False)
    validation = str(body.get("validation") or "basic").strip().lower()
    if validation not in ("none", "basic"):
        raise ValidationError(_err("validation", "must be 'none' or 'basic'", "value_error"))

    # Build output payload
    out: ParansPayload = {
        "subject": subject,
        "place": place,
        "jd_tt_ref": float(jd_tt_ref) if jd_tt_ref is not None else None,  # type: ignore
        "jd_ut1_ref": float(jd_ut1_ref) if jd_ut1_ref is not None else None,  # type: ignore
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa_deg": ayanamsa_deg,
        "bodies": bodies,
        "tolerance_minutes": float(tolerance_min),
        "search_window_days": float(window_days),
        "max_iters": int(max_iters),
        "fd_step_minutes": float(fd_step_min),
        "earth_model": earth_model,
        "apply_refraction": apply_refraction,
        "pressure_hPa": float(pressure),
        "temperature_C": float(temperature),
        "profile": profile,
        "validation": validation,
    }

    return out


# ───────────────────────── synastry / composite / synastry-report ─────────────


class NatalSynastry(TypedDict, total=False):
    """Minimal natal data for synastry calculations."""
    date: str
    time: str
    place_tz: str
    latitude: float
    longitude: float
    elev_m: float
    mode: str


class PlaceSynastry(TypedDict, total=False):
    """Place override for synastry calculations."""
    latitude: float
    longitude: float
    elev_m: float


class SynastryPayload(TypedDict, total=False):
    """Payload for /api/synastry endpoint."""
    natal_a: NatalSynastry
    natal_b: NatalSynastry
    jd_tt_a: float
    jd_ut1_a: float
    jd_tt_b: float
    jd_ut1_b: float
    place_a: PlaceSynastry
    place_b: PlaceSynastry
    frame: str
    ayanamsa_deg: float
    zodiac_mode: str
    house_system: str
    orbs: Dict[str, float]
    parallels: bool
    antiscia: bool


class CompositePayload(TypedDict, total=False):
    """Payload for /api/composite endpoint."""
    natal_a: NatalSynastry
    natal_b: NatalSynastry
    method: str
    jd_tt_ref: float
    jd_ut1_ref: float
    place_ref: PlaceSynastry
    frame: str
    house_system: str
    ayanamsa_deg: float
    zodiac_mode: str


class SynastryReportPayload(TypedDict, total=False):
    """Payload for /api/synastry/report endpoint."""
    natal_a: NatalSynastry
    natal_b: NatalSynastry
    jd_tt_a: float
    jd_ut1_a: float
    jd_tt_b: float
    jd_ut1_b: float
    place_a: PlaceSynastry
    place_b: PlaceSynastry
    frame: str
    ayanamsa_deg: float
    zodiac_mode: str
    house_system: str
    orbs: Dict[str, float]
    parallels: bool
    antiscia: bool
    composite_method: str
    composite_place_ref: PlaceSynastry


def _parse_natal_synastry(natal_raw: Any) -> NatalSynastry:
    """Parse and validate natal data for synastry."""
    if not isinstance(natal_raw, dict):
        raise ValidationError(_err("natal", "must be an object", "value_error"))

    natal: NatalSynastry = {}

    if "date" in natal_raw:
        natal["date"] = str(natal_raw["date"]).strip()
        if not natal["date"]:
            raise ValidationError(_err("natal.date", "cannot be empty", "value_error"))

    if "time" in natal_raw:
        natal["time"] = str(natal_raw["time"]).strip()
        if not natal["time"]:
            raise ValidationError(_err("natal.time", "cannot be empty", "value_error"))

    if "place_tz" in natal_raw:
        natal["place_tz"] = str(natal_raw["place_tz"]).strip()
        if not natal["place_tz"]:
            raise ValidationError(_err("natal.place_tz", "cannot be empty", "value_error"))

    # Optional coordinates
    if "latitude" in natal_raw:
        lat = _as_float(natal_raw["latitude"])
        if lat is None:
            raise ValidationError(_err("natal.latitude", "must be a number", "type_error.float"))
        if not (-90.0 <= lat <= 90.0):
            raise ValidationError(_err("natal.latitude", "must be between -90 and 90 degrees", "value_error"))
        natal["latitude"] = lat

    if "longitude" in natal_raw:
        lon = _as_float(natal_raw["longitude"])
        if lon is None:
            raise ValidationError(_err("natal.longitude", "must be a number", "type_error.float"))
        if not (-180.0 <= lon <= 180.0):
            raise ValidationError(_err("natal.longitude", "must be between -180 and 180 degrees", "value_error"))
        natal["longitude"] = lon

    if "elev_m" in natal_raw:
        elev = _as_float(natal_raw["elev_m"])
        if elev is None:
            raise ValidationError(_err("natal.elev_m", "must be a number", "type_error.float"))
        natal["elev_m"] = elev

    if "mode" in natal_raw:
        mode = str(natal_raw["mode"]).strip().lower()
        if mode not in ("tropical", "sidereal"):
            raise ValidationError(_err("natal.mode", "must be 'tropical' or 'sidereal'", "value_error"))
        natal["mode"] = mode

    return natal


def _parse_place_synastry(place_raw: Any) -> PlaceSynastry:
    """Parse and validate place override for synastry."""
    if not isinstance(place_raw, dict):
        raise ValidationError(_err("place", "must be an object", "value_error"))

    place: PlaceSynastry = {}

    if "latitude" in place_raw:
        lat = _as_float(place_raw["latitude"])
        if lat is None:
            raise ValidationError(_err("place.latitude", "must be a number", "type_error.float"))
        if not (-90.0 <= lat <= 90.0):
            raise ValidationError(_err("place.latitude", "must be between -90 and 90 degrees", "value_error"))
        place["latitude"] = lat

    if "longitude" in place_raw:
        lon = _as_float(place_raw["longitude"])
        if lon is None:
            raise ValidationError(_err("place.longitude", "must be a number", "type_error.float"))
        if not (-180.0 <= lon <= 180.0):
            raise ValidationError(_err("place.longitude", "must be between -180 and 180 degrees", "value_error"))
        place["longitude"] = lon

    if "elev_m" in place_raw:
        elev = _as_float(place_raw["elev_m"])
        if elev is None:
            raise ValidationError(_err("place.elev_m", "must be a number", "type_error.float"))
        place["elev_m"] = elev

    return place


def _parse_orbs_dict(orbs_raw: Any) -> Dict[str, float]:
    """Parse and validate orbs dictionary."""
    if not isinstance(orbs_raw, dict):
        raise ValidationError(_err("orbs", "must be an object", "value_error"))

    orbs: Dict[str, float] = {}
    valid_aspects = {
        "conjunction",
        "opposition",
        "trine",
        "square",
        "sextile",
        "quincunx",
        "parallel",           # include for parallel logic
        "parallel_arcmin",    # backward compatibility
        "antiscia",
    }

    for aspect, orb_raw in orbs_raw.items():
        if not isinstance(aspect, str):
            continue

        aspect_clean = aspect.strip().lower()
        if aspect_clean not in valid_aspects:
            raise ValidationError(_err(f"orbs.{aspect}", "unknown aspect type", "value_error"))

        orb = _as_float(orb_raw)
        if orb is None:
            raise ValidationError(_err(f"orbs.{aspect}", "must be a number", "type_error.float"))
        if orb < 0:
            raise ValidationError(_err(f"orbs.{aspect}", "must be non-negative", "value_error"))

        orbs[aspect_clean] = orb

    return orbs


def parse_synastry_payload(body: Dict[str, Any]) -> SynastryPayload:
    """Validate and normalize payload for /api/synastry endpoint."""
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # Parse natal charts (required)
    natal_a_raw = body.get("natal_a")
    if natal_a_raw is None:
        raise ValidationError(_err("natal_a", "required object", "value_error"))
    natal_a = _parse_natal_synastry(natal_a_raw)

    natal_b_raw = body.get("natal_b")
    if natal_b_raw is None:
        raise ValidationError(_err("natal_b", "required object", "value_error"))
    natal_b = _parse_natal_synastry(natal_b_raw)

    # Strict timescales (optional)
    jd_tt_a = _as_float(body.get("jd_tt_a"))
    jd_ut1_a = _as_float(body.get("jd_ut1_a"))
    jd_tt_b = _as_float(body.get("jd_tt_b"))
    jd_ut1_b = _as_float(body.get("jd_ut1_b"))

    # Place overrides (optional)
    place_a = None
    if body.get("place_a") is not None:
        place_a = _parse_place_synastry(body["place_a"])

    place_b = None
    if body.get("place_b") is not None:
        place_b = _parse_place_synastry(body["place_b"])

    # Frame and coordinate system
    frame = parse_frame(body.get("frame"))
    ayanamsa_deg = _as_float(body.get("ayanamsa_deg", 0.0)) or 0.0
    zodiac_mode = parse_mode(body.get("zodiac_mode") or body.get("mode"))
    house_system = str(body.get("house_system", "placidus")).strip().lower()

    # Orbs (optional)
    orbs = None
    if body.get("orbs") is not None:
        orbs = _parse_orbs_dict(body["orbs"])

    # Flags
    parallels = bool(_truthy(body.get("parallels", True)))
    antiscia = bool(_truthy(body.get("antiscia", True)))

    # Build output
    out: SynastryPayload = {
        "natal_a": natal_a,
        "natal_b": natal_b,
        "frame": frame,
        "ayanamsa_deg": ayanamsa_deg,
        "zodiac_mode": zodiac_mode,
        "house_system": house_system,
        "parallels": parallels,
        "antiscia": antiscia,
    }

    # Add optional fields
    if jd_tt_a is not None:
        out["jd_tt_a"] = jd_tt_a
    if jd_ut1_a is not None:
        out["jd_ut1_a"] = jd_ut1_a
    if jd_tt_b is not None:
        out["jd_tt_b"] = jd_tt_b
    if jd_ut1_b is not None:
        out["jd_ut1_b"] = jd_ut1_b
    if place_a is not None:
        out["place_a"] = place_a
    if place_b is not None:
        out["place_b"] = place_b
    if orbs is not None:
        out["orbs"] = orbs

    return out


def parse_composite_payload(body: Dict[str, Any]) -> CompositePayload:
    """Validate and normalize payload for /api/composite endpoint."""
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # Parse natal charts (required)
    natal_a = _parse_natal_synastry(body.get("natal_a"))
    natal_b = _parse_natal_synastry(body.get("natal_b"))

    # Composite method
    method = str(body.get("method", "midpoint")).strip().lower()
    if method not in ("midpoint", "davison"):
        raise ValidationError(_err("method", "must be 'midpoint' or 'davison'", "composite_value_error"))

    # Reference timescales and place (optional)
    jd_tt_ref = _as_float(body.get("jd_tt_ref"))
    jd_ut1_ref = _as_float(body.get("jd_ut1_ref"))

    place_ref = None
    if body.get("place_ref") is not None:
        place_ref = _parse_place_synastry(body["place_ref"])

    # Coordinate system parameters
    frame = parse_frame(body.get("frame"))
    house_system = str(body.get("house_system", "placidus")).strip().lower()
    ayanamsa_deg = _as_float(body.get("ayanamsa_deg", 0.0)) or 0.0
    zodiac_mode = parse_mode(body.get("zodiac_mode") or body.get("mode"))

    # Build output
    out: CompositePayload = {
        "natal_a": natal_a,
        "natal_b": natal_b,
        "method": method,
        "frame": frame,
        "house_system": house_system,
        "ayanamsa_deg": ayanamsa_deg,
        "zodiac_mode": zodiac_mode,
    }

    # Add optional fields
    if jd_tt_ref is not None:
        out["jd_tt_ref"] = jd_tt_ref
    if jd_ut1_ref is not None:
        out["jd_ut1_ref"] = jd_ut1_ref
    if place_ref is not None:
        out["place_ref"] = place_ref

    return out


def parse_synastry_report_payload(body: Dict[str, Any]) -> SynastryReportPayload:
    """Validate and normalize payload for /api/synastry/report endpoint."""
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # Start with synastry validation
    synastry_payload = parse_synasry_payload_safe(body)
    # (helper below ensures we don't re-raise partially; keeps compatibility)
    # Add composite-specific fields
    composite_method = str(body.get("composite_method", "midpoint")).strip().lower()
    if composite_method not in ("midpoint", "davison"):
        raise ValidationError(_err("composite_method", "must be 'midpoint' or 'davison'", "value_error"))

    composite_place_ref = None
    if body.get("composite_place_ref") is not None:
        composite_place_ref = _parse_place_synastry(body["composite_place_ref"])

    # Build output by extending synastry payload
    out: SynastryReportPayload = {
        **synastry_payload,  # type: ignore
        "composite_method": composite_method,
    }

    if composite_place_ref is not None:
        out["composite_place_ref"] = composite_place_ref

    return out


def parse_synasry_payload_safe(body: Dict[str, Any]) -> SynastryPayload:
    """Internal helper to call parse_synastry_payload with a consistent name."""
    return parse_synastry_payload(body)


# ───────────────────────── relocation / astrocartography ─────────────────────────

class RelocationPayload(TypedDict, total=False):
    """Payload for /api/relocation endpoint."""
    natal: Dict[str, Any]
    place_new: Dict[str, Any]
    jd_tt_natal: float
    jd_ut1_natal: float
    frame: str
    house_system: str
    zodiac_mode: str
    ayanamsa_deg: float
    topocentric_positions: bool

class AstrocartographyPayload(TypedDict, total=False):
    """Payload for /api/astrocartography endpoint."""
    natal: Dict[str, Any]
    jd_tt: float
    jd_ut1: float
    frame: str
    zodiac_mode: str
    ayanamsa_deg: float
    bodies: List[str]
    lon_step_deg: float
    lat_clip_deg: float
    earth_model: str
    apply_refraction: bool
    pressure_hPa: float
    temperature_C: float
    default_elev_m: float

def _parse_relocation_place(place_raw: Any) -> Dict[str, Any]:
    """Parse and validate place data for relocation."""
    if not isinstance(place_raw, dict):
        raise ValidationError(_err("place", "must be an object", "value_error"))

    place = {}

    # Required coordinates
    lat = _as_float(place_raw.get("latitude"))
    if lat is None:
        raise ValidationError(_err("place.latitude", "required number", "value_error"))
    if not (-90.0 <= lat <= 90.0):
        raise ValidationError(_err("place.latitude", "must be between -90 and 90 degrees", "value_error"))
    place["latitude"] = lat

    lon = _as_float(place_raw.get("longitude"))
    if lon is None:
        raise ValidationError(_err("place.longitude", "required number", "value_error"))
    if not (-180.0 <= lon <= 180.0):
        raise ValidationError(_err("place.longitude", "must be between -180 and 180 degrees", "value_error"))
    place["longitude"] = lon

    # Optional elevation
    if "elev_m" in place_raw:
        elev = _as_float(place_raw["elev_m"])
        if elev is None:
            raise ValidationError(_err("place.elev_m", "must be a number", "type_error.float"))
        place["elev_m"] = elev

    return place

def _parse_relocation_natal(natal_raw: Any) -> Dict[str, Any]:
    """Parse and validate natal data for relocation."""
    if not isinstance(natal_raw, dict):
        raise ValidationError(_err("natal", "must be an object", "value_error"))

    natal = {}

    # Required fields for timescale resolution
    required_fields = ["date", "time", "place_tz"]
    for field in required_fields:
        if field not in natal_raw:
            raise ValidationError(_err(f"natal.{field}", "required string", "value_error"))

        value = str(natal_raw[field]).strip()
        if not value:
            raise ValidationError(_err(f"natal.{field}", "cannot be empty", "value_error"))
        natal[field] = value

    # Optional coordinates (for original chart reference)
    for coord_field in ["latitude", "longitude", "elev_m"]:
        if coord_field in natal_raw:
            coord_val = _as_float(natal_raw[coord_field])
            if coord_val is not None:
                if coord_field == "latitude" and not (-90.0 <= coord_val <= 90.0):
                    raise ValidationError(_err(f"natal.{coord_field}", "must be between -90 and 90 degrees", "value_error"))
                elif coord_field == "longitude" and not (-180.0 <= coord_val <= 180.0):
                    raise ValidationError(_err(f"natal.{coord_field}", "must be between -180.0 and 180.0 degrees", "value_error"))
                natal[coord_field] = coord_val

    # Optional mode
    if "mode" in natal_raw:
        mode = str(natal_raw["mode"]).strip().lower()
        if mode not in ("tropical", "sidereal"):
            raise ValidationError(_err("natal.mode", "must be 'tropical' or 'sidereal'", "value_error"))
        natal["mode"] = mode

    return natal

def _parse_bodies_list_relocation(bodies_raw: Any) -> List[str]:
    """Parse and validate bodies list for astrocartography."""
    if bodies_raw is None:
        # Default major bodies
        return ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]

    if not isinstance(bodies_raw, list):
        raise ValidationError(_err("bodies", "must be a list", "type_error.list"))

    if not bodies_raw:
        raise ValidationError(_err("bodies", "cannot be empty", "value_error"))

    bodies = []
    valid_bodies = {
        "sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn",
        "uranus", "neptune", "pluto", "chiron", "ceres", "pallas", "juno", "vesta"
    }

    for i, body in enumerate(bodies_raw):
        if not isinstance(body, str):
            raise ValidationError(_err(f"bodies[{i}]", "must be a string", "type_error.str"))

        body_clean = body.strip()
        if not body_clean:
            raise ValidationError(_err(f"bodies[{i}]", "cannot be empty", "value_error"))

        body_lower = body_clean.lower()
        if body_lower not in valid_bodies:
            raise ValidationError(_err(f"bodies[{i}]", f"unknown body '{body_clean}'", "value_error"))

        bodies.append(body_clean.capitalize())

    return bodies

def parse_relocation_payload(body: Dict[str, Any]) -> RelocationPayload:
    """Validate and normalize payload for /api/relocation endpoint."""
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # Parse natal chart (required)
    natal_raw = body.get("natal")
    if natal_raw is None:
        raise ValidationError(_err("natal", "required object", "value_error"))
    natal = _parse_relocation_natal(natal_raw)

    # Parse new place (required)
    place_new_raw = body.get("place_new")
    if place_new_raw is None:
        raise ValidationError(_err("place_new", "required object", "value_error"))
    place_new = _parse_relocation_place(place_new_raw)

    # Optional strict timescales
    jd_tt_natal = _as_float(body.get("jd_tt_natal"))
    jd_ut1_natal = _as_float(body.get("jd_ut1_natal"))

    # Frame and coordinate system
    frame = parse_frame(body.get("frame"))
    house_system = str(body.get("house_system", "placidus")).strip().lower()
    zodiac_mode = parse_mode(body.get("zodiac_mode") or body.get("mode"))

    # Ayanamsa
    ayanamsa_deg = _as_float(body.get("ayanamsa_deg", 0.0))
    if ayanamsa_deg is None:
        raise ValidationError(_err("ayanamsa_deg", "must be a number", "type_error.float"))

    # Topocentric flag
    topocentric_positions = bool(_truthy(body.get("topocentric_positions", False)))

    # Build output
    out: RelocationPayload = {
        "natal": natal,
        "place_new": place_new,
        "frame": frame,
        "house_system": house_system,
        "zodiac_mode": zodiac_mode,
        "ayanamsa_deg": float(ayanamsa_deg),
        "topocentric_positions": topocentric_positions,
    }

    # Add optional strict timescales
    if jd_tt_natal is not None:
        out["jd_tt_natal"] = float(jd_tt_natal)
    if jd_ut1_natal is not None:
        out["jd_ut1_natal"] = float(jd_ut1_natal)

    return out

def parse_astrocartography_payload(body: Dict[str, Any]) -> AstrocartographyPayload:
    """Validate and normalize payload for /api/astrocartography endpoint."""
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # Parse natal chart (required)
    natal_raw = body.get("natal")
    if natal_raw is None:
        raise ValidationError(_err("natal", "required object", "value_error"))
    natal = _parse_relocation_natal(natal_raw)

    # Optional strict timescales
    jd_tt = _as_float(body.get("jd_tt"))
    jd_ut1 = _as_float(body.get("jd_ut1"))

    # Frame and coordinate system
    frame = parse_frame(body.get("frame"))
    zodiac_mode = parse_mode(body.get("zodiac_mode") or body.get("mode"))

    # Ayanamsa
    ayanamsa_deg = _as_float(body.get("ayanamsa_deg", 0.0))
    if ayanamsa_deg is None:
        raise ValidationError(_err("ayanamsa_deg", "must be a number", "type_error.float"))

    # Bodies list
    bodies = _parse_bodies_list_relocation(body.get("bodies"))

    # Sampling parameters
    lon_step_deg = _as_float(body.get("lon_step_deg", 1.0))
    if lon_step_deg is None or lon_step_deg <= 0 or lon_step_deg > 45.0:
        raise ValidationError(_err("lon_step_deg", "must be between 0 and 45 degrees", "value_error"))

    lat_clip_deg = _as_float(body.get("lat_clip_deg", 89.5))
    if lat_clip_deg is None or lat_clip_deg <= 0 or lat_clip_deg > 90.0:
        raise ValidationError(_err("lat_clip_deg", "must be between 0 and 90 degrees", "value_error"))

    # Earth model
    earth_model = str(body.get("earth_model", "spherical")).strip().lower()
    if earth_model not in ("spherical", "wgs84"):
        raise ValidationError(_err("earth_model", "must be 'spherical' or 'wgs84'", "value_error"))

    # Atmospheric refraction
    apply_refraction = bool(_truthy(body.get("apply_refraction", False)))

    pressure_hPa = _as_float(body.get("pressure_hPa", 1010.0))
    if pressure_hPa is None or pressure_hPa <= 0:
        raise ValidationError(_err("pressure_hPa", "must be positive", "value_error"))

    temperature_C = _as_float(body.get("temperature_C", 10.0))
    if temperature_C is None:
        raise ValidationError(_err("temperature_C", "must be a number", "type_error.float"))

    # Default elevation
    default_elev_m = _as_float(body.get("default_elev_m", 0.0))
    if default_elev_m is None:
        raise ValidationError(_err("default_elev_m", "must be a number", "type_error.float"))

    # Build output
    out: AstrocartographyPayload = {
        "natal": natal,
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa_deg": float(ayanamsa_deg),
        "bodies": bodies,
        "lon_step_deg": float(lon_step_deg),
        "lat_clip_deg": float(lat_clip_deg),
        "earth_model": earth_model,
        "apply_refraction": apply_refraction,
        "pressure_hPa": float(pressure_hPa),
        "temperature_C": float(temperature_C),
        "default_elev_m": float(default_elev_m),
    }

    # Add optional strict timescales
    if jd_tt is not None:
        out["jd_tt"] = float(jd_tt)
    if jd_ut1 is not None:
        out["jd_ut1"] = float(jd_ut1)

    return out


# ───────────────────────── directions ─────────────────────────

class DirectionsPayload(TypedDict, total=False):
    """Payload for /api/directions endpoint."""
    natal: Dict[str, Any]
    method: str
    rate: str
    target: Dict[str, Any]
    years_after: float
    jd_tt_natal: float
    jd_ut1_natal: float
    place: Dict[str, Any]
    frame: str
    house_system: str
    zodiac_mode: str
    ayanamsa_deg: float
    arcs: str
    orbs: Dict[str, float]
    include_hits_to: List[str]
    parallels: bool
    antiscia: bool
    profile: bool
    validation: str

def _parse_directions_natal(natal_raw: Any) -> Dict[str, Any]:
    """Parse and validate natal data for directions."""
    if not isinstance(natal_raw, dict):
        raise ValidationError(_err("natal", "must be an object", "value_error"))

    natal = {}

    # Required fields for timescale resolution (if strict JD not provided)
    required_fields = ["date", "time", "place_tz"]
    for field in required_fields:
        if field in natal_raw:
            value = str(natal_raw[field]).strip()
            if not value:
                raise ValidationError(_err(f"natal.{field}", "cannot be empty", "value_error"))
            natal[field] = value

    # Optional coordinates
    for coord_field in ["latitude", "longitude", "elev_m"]:
        if coord_field in natal_raw:
            coord_val = _as_float(natal_raw[coord_field])
            if coord_val is not None:
                if coord_field == "latitude" and not (-90.0 <= coord_val <= 90.0):
                    raise ValidationError(_err(f"natal.{coord_field}", "must be between -90 and 90 degrees", "value_error"))
                elif coord_field == "longitude" and not (-180.0 <= coord_val <= 180.0):
                    raise ValidationError(_err(f"natal.{coord_field}", "must be between -180.0 and 180.0 degrees", "value_error"))
                natal[coord_field] = coord_val

    # Optional mode
    if "mode" in natal_raw:
        mode = str(natal_raw["mode"]).strip().lower()
        if mode not in ("tropical", "sidereal"):
            raise ValidationError(_err("natal.mode", "must be 'tropical' or 'sidereal'", "value_error"))
        natal["mode"] = mode

    return natal

def _parse_directions_target(target_raw: Any) -> Dict[str, Any]:
    """Parse and validate target data for directions."""
    if not isinstance(target_raw, dict):
        raise ValidationError(_err("target", "must be an object", "value_error"))

    target = {}
    required_fields = ["date", "time", "place_tz"]
    for field in required_fields:
        if field not in target_raw:
            raise ValidationError(_err(f"target.{field}", "required string", "value_error"))

        value = str(target_raw[field]).strip()
        if not value:
            raise ValidationError(_err(f"target.{field}", "cannot be empty", "value_error"))
        target[field] = value

    return target

def _parse_directions_place(place_raw: Any) -> Dict[str, Any]:
    """Parse and validate place override for directions."""
    if not isinstance(place_raw, dict):
        raise ValidationError(_err("place", "must be an object", "value_error"))

    place = {}

    # Required coordinates for place override
    lat = _as_float(place_raw.get("latitude"))
    if lat is None:
        raise ValidationError(_err("place.latitude", "required number", "value_error"))
    if not (-90.0 <= lat <= 90.0):
        raise ValidationError(_err("place.latitude", "must be between -90 and 90 degrees", "value_error"))
    place["latitude"] = lat

    lon = _as_float(place_raw.get("longitude"))
    if lon is None:
        raise ValidationError(_err("place.longitude", "required number", "value_error"))
    if not (-180.0 <= lon <= 180.0):
        raise ValidationError(_err("place.longitude", "must be between -180 and 180 degrees", "value_error"))
    place["longitude"] = lon

    # Optional elevation
    if "elev_m" in place_raw:
        elev = _as_float(place_raw.get("elev_m"))
        if elev is None:
            raise ValidationError(_err("place.elev_m", "must be a number", "type_error.float"))
        place["elev_m"] = elev

    return place

def _parse_directions_orbs(orbs_raw: Any) -> Dict[str, float]:
    """Parse and validate orbs for directions."""
    if not isinstance(orbs_raw, dict):
        raise ValidationError(_err("orbs", "must be an object", "value_error"))

    orbs = {}
    valid_aspects = {"conjunction", "opposition", "trine", "square", "sextile", "quincunx"}

    for aspect, orb_val in orbs_raw.items():
        if not isinstance(aspect, str):
            continue

        aspect_clean = aspect.strip().lower()
        if aspect_clean not in valid_aspects:
            raise ValidationError(_err(f"orbs.{aspect}", "unknown aspect type", "value_error"))

        orb = _as_float(orb_val)
        if orb is None:
            raise ValidationError(_err(f"orbs.{aspect}", "must be a number", "type_error.float"))
        if orb < 0:
            raise ValidationError(_err(f"orbs.{aspect}", "must be non-negative", "value_error"))

        orbs[aspect_clean] = orb

    return orbs

def parse_directions_payload(body: Dict[str, Any]) -> DirectionsPayload:
    """Validate and normalize payload for /api/directions endpoint (v1)."""
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    # Parse natal chart (required)
    natal_raw = body.get("natal")
    if natal_raw is None:
        raise ValidationError(_err("natal", "required object", "value_error"))
    natal = _parse_directions_natal(natal_raw)

    # Method validation
    method = str(body.get("method", "solar_arc")).strip().lower()
    if method not in ("solar_arc",):
        raise ValidationError(_err("method", "must be 'solar_arc'", "value_error"))

    # Rate validation
    rate = str(body.get("rate", "naibod")).strip().lower()
    if rate not in ("naibod", "true_sun"):
        raise ValidationError(_err("rate", "must be 'naibod' or 'true_sun'", "value_error"))

    # Target vs years_after (one required)
    target_raw = body.get("target")
    years_after = _as_float(body.get("years_after"))

    target = None
    if target_raw is not None:
        target = _parse_directions_target(target_raw)

    if target is None and years_after is None:
        raise ValidationError(_err(["target", "years_after"], "provide either target or years_after", "value_error"))

    if target is not None and years_after is not None:
        raise ValidationError(_err(["target", "years_after"], "provide only one of target or years_after", "value_error"))

    # Optional strict timescales
    jd_tt_natal = _as_float(body.get("jd_tt_natal"))
    jd_ut1_natal = _as_float(body.get("jd_ut1_natal"))

    # Optional place override
    place = None
    if body.get("place") is not None:
        place = _parse_directions_place(body["place"])

    # Frame and coordinate system
    frame = parse_frame(body.get("frame"))
    house_system = str(body.get("house_system", "placidus")).strip().lower()
    zodiac_mode = parse_mode(body.get("zodiac_mode") or body.get("mode"))

    # Ayanamsa
    ayanamsa_deg = _as_float(body.get("ayanamsa_deg", 0.0))
    if ayanamsa_deg is None:
        raise ValidationError(_err("ayanamsa_deg", "must be a number", "type_error.float"))

    # Arcs type
    arcs = str(body.get("arcs", "direct")).strip().lower()
    if arcs not in ("direct", "converse", "both"):
        raise ValidationError(_err("arcs", "must be 'direct', 'converse', or 'both'", "value_error"))

    # Hit targets
    include_hits_to = body.get("include_hits_to", ["planets", "angles", "cusps"])
    if not isinstance(include_hits_to, list):
        raise ValidationError(_err("include_hits_to", "must be a list", "type_error.list"))

    valid_targets = {"planets", "angles", "cusps"}
    for i, target_type in enumerate(include_hits_to):
        if target_type not in valid_targets:
            raise ValidationError(_err(f"include_hits_to[{i}]", f"must be one of {valid_targets}", "value_error"))

    # Optional orbs
    orbs = None
    if body.get("orbs") is not None:
        orbs = _parse_directions_orbs(body["orbs"])

    # Boolean flags
    parallels = bool(_truthy(body.get("parallels", False)))
    antiscia = bool(_truthy(body.get("antiscia", False)))
    profile = bool(_truthy(body.get("profile", False)))

    # Validation level
    validation = str(body.get("validation", "basic")).strip().lower()
    if validation not in ("none", "basic"):
        raise ValidationError(_err("validation", "must be 'none' or 'basic'", "value_error"))

    # Build output
    out: DirectionsPayload = {
        "natal": natal,
        "method": method,
        "rate": rate,
        "frame": frame,
        "house_system": house_system,
        "zodiac_mode": zodiac_mode,
        "ayanamsa_deg": float(ayanamsa_deg),
        "arcs": arcs,
        "include_hits_to": include_hits_to,
        "parallels": parallels,
        "antiscia": antiscia,
        "profile": profile,
        "validation": validation,
    }

    # Add optional fields
    if target is not None:
        out["target"] = target
    if years_after is not None:
        out["years_after"] = float(years_after)
    if jd_tt_natal is not None:
        out["jd_tt_natal"] = float(jd_tt_natal)
    if jd_ut1_natal is not None:
        out["jd_ut1_natal"] = float(jd_ut1_natal)
    if place is not None:
        out["place"] = place
    if orbs is not None:
        out["orbs"] = orbs

    return out


# ───────────────────────── PREDICTION ENGINE PARSER (V2) ─────────────────────────

def parse_prediction_payload_v2(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate V2 prediction engine request payload.

    This is intentionally lightweight because each route enforces the fields
    it needs (e.g., time_range vs target_date). We verify shapes and types here
    and preserve the original body so routes can pass-through technique-specific
    kwargs (transit_*, progression_*, return_*, vedic_*).
    """
    if not isinstance(body, dict):
        raise ValidationError("payload must be an object")

    errors: List[Dict[str, Any]] = []

    # natal_chart is required by all V2 routes
    natal_chart = body.get("natal_chart")
    if not isinstance(natal_chart, dict):
        errors.append({"loc": ["natal_chart"], "msg": "required object", "type": "value_error"})

    # If provided, time_range must be a 2-element array-like
    if "time_range" in body:
        time_range = body.get("time_range")
        if not (isinstance(time_range, (list, tuple)) and len(time_range) == 2):
            errors.append({"loc": ["time_range"], "msg": "must be [start_date, end_date] array", "type": "value_error"})

    # If provided, target_date must be a string (ISO), int/float (JD), or datetime
    if "target_date" in body:
        target_date = body.get("target_date")
        if not isinstance(target_date, (str, int, float, datetime)):
            errors.append({"loc": ["target_date"], "msg": "must be ISO string, JD float, or datetime", "type": "value_error"})

    if errors:
        raise ValidationError(errors)

    return body


# ───────────────────────── timescale resolver (for predictive.py) ─────────────

class TimescalesOut(TypedDict):
    jd_tt: float
    jd_ut1: float
    jd_utc: float
    tz: str

def resolve_timescales_from_civil_erfa(
    d: date,
    time_hh_mm_ss: str,
    place_tz: str,
) -> TimescalesOut:
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
    add_one_sec = (ss == 60)
    if add_one_sec:
        ss = 59

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
    # Skyfield can take the aware UTC datetime directly, or components
    t = ts.utc(
        dt_utc.year, dt_utc.month, dt_utc.day,
        dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond / 1e6
    )

    jd_tt = float(t.tt)

    # ΔT = TT − UT1 (seconds) → convert to days to get JD(UT1)
    try:
        delta_t_sec = float(t.delta_t)
    except Exception:
        delta_t_sec = 0.0  # fallback: UT1≈UTC if ΔT unavailable
    jd_ut1 = jd_tt - (delta_t_sec / 86400.0)

    # Compute JD(UTC) from the UTC datetime (Skyfield has no `utc_jd`)
    J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    jd_utc = 2451545.0 + (dt_utc - J2000_UTC).total_seconds() / 86400.0

    return {"jd_tt": jd_tt, "jd_ut1": jd_ut1, "jd_utc": jd_utc, "tz": tz}


__all__ = [
    # errors & helpers
    "ValidationError",
    "_err",
    "_as_float",
    "_as_int",
    "_truthy",
    "parse_date",
    "parse_time_str",
    "parse_latlon",
    "parse_mode",
    "parse_house_system",
    "parse_frame",
    "parse_earth_model",
    "parse_bodies_list",

    # v1 charts/predictions (kept)
    "parse_chart_payload",
    "parse_prediction_payload",
    "parse_rectification_payload",

    # ephemeris
    "parse_ephemeris_payload",

    # progressions / returns / parans (v1)
    "parse_progressions_payload",
    "parse_returns_payload",
    "parse_parans_payload",

    # synastry / composite / synastry report
    "parse_synastry_payload",
    "parse_composite_payload",
    "parse_synastry_report_payload",

    # relocation / astrocartography
    "parse_relocation_payload",
    "parse_astrocartography_payload",

    # directions
    "parse_directions_payload",

    # v2 prediction engine (new API layer)
    "parse_prediction_payload_v2",

    # utility
    "TimescalesOut",
    "resolve_timescales_from_civil_erfa",
]
