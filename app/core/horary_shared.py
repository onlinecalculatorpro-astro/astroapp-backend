# -*- coding: utf-8 -*-
"""
horary_shared.py
----------------
Shared utilities and dataclasses for Horary / Prashna analysis.

What this module provides
- Input dataclasses (HoraryInput, QuerentBirthData, HybridPrasnaInput) + QuestionType enum
- Question→house mappings (ENHANCED_QUESTION_HOUSES)
- Zodiac helpers: deg_wrap, sign_name_from_deg, lord_of_sign, etc.
- Dignity & basic aspects
- KP star/sub computations
- Chart & Houses helpers:
    - ensure_coords_and_tz(...)  -> (date, time, tz, lat, lon)
    - build_chart(...)           -> compute_chart wrapper
    - compute_houses_from_chart(..., house_system, zodiac_mode, ayanamsa_deg)
      (tries app.core.houses_advanced.compute_house_system; falls back to Equal from ASC)
- Radicality flags (ASC lord vs day/hour lord)

No system-specific judgement is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone
from enum import Enum
import math

# Core engines — provided by your project
from app.core.astronomy import compute_chart, resolve_place
from app.core.houses_advanced import compute_house_system  # requires JD(TT)/UT/UT1

# =============================================================================
# Constants & basic helpers
# =============================================================================

SIGN_NAMES = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

# Traditional 7 planets commonly used in horary
TRAD_PLANETS = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"]

# Sign lords by sign index (0=Aries..11=Pisces)
SIGN_LORDS = {
    0: "Mars", 1: "Venus", 2: "Mercury", 3: "Moon", 4: "Sun", 5: "Mercury",
    6: "Venus", 7: "Mars", 8: "Jupiter", 9: "Saturn", 10: "Saturn", 11: "Jupiter"
}

def deg_wrap(x: float) -> float:
    return (x % 360.0 + 360.0) % 360.0

def sign_index(deg: float) -> int:
    return int(deg_wrap(deg) // 30)

def sign_name_from_deg(deg: float) -> str:
    return SIGN_NAMES[sign_index(deg)]

def lord_of_sign(deg: float) -> str:
    return SIGN_LORDS[sign_index(deg)]

def angular_sep(a: float, b: float) -> float:
    """Smallest absolute angular separation in degrees [0..180]."""
    d = abs(deg_wrap(a) - deg_wrap(b))
    return d if d <= 180.0 else 360.0 - d

def shift_sidereal(values: List[float], ay_deg: float) -> List[float]:
    """Shift ecliptic longitudes by -ayanamsa (tropical→sidereal)."""
    return [deg_wrap(v - ay_deg) for v in values]

def house_of(long_deg: float, cusps_deg: List[float]) -> int:
    """
    Return house number 1..12 for a longitude given cusp longitudes (H1..H12).
    Wrap-safe across H12→H1.
    """
    if not cusps_deg or len(cusps_deg) != 12:
        return 1
    x = deg_wrap(long_deg)
    c = [deg_wrap(d) for d in cusps_deg]
    for i in range(12):
        a = c[i]
        b = c[(i + 1) % 12]
        if a <= b:
            inside = (a <= x) and (x < b)
        else:
            inside = (x >= a) or (x < b)   # wrap interval
        if inside:
            return i + 1
    return 1

# =============================================================================
# Aspects & Dignity
# =============================================================================

def calculate_planetary_dignity(longitude: float, planet: str) -> float:
    """
    Simple dignity score (-2..+2).
    +2: exalted; +1: own sign; -2: fall; 0: neutral (elsewhere).
    """
    s = sign_index(longitude)
    exalt = {
        "Sun": 0, "Moon": 1, "Mars": 9, "Mercury": 5,
        "Jupiter": 3, "Venus": 11, "Saturn": 6
    }
    own = {
        "Sun": [4], "Moon": [3], "Mars": [0,7], "Mercury": [2,5],
        "Jupiter": [8,11], "Venus": [1,6], "Saturn": [9,10]
    }
    if planet in exalt and s == exalt[planet]:
        return 2.0
    if planet in own and s in own[planet]:
        return 1.0
    if planet in exalt and s == (exalt[planet] + 6) % 12:
        return -2.0
    return 0.0

def calculate_aspects(long1: float, long2: float) -> Tuple[float, Optional[str]]:
    """Loose Ptolemaic aspects; returns (sep_degrees, aspect_name|None)."""
    diff = abs(deg_wrap(long1 - long2))
    if diff > 180: diff = 360 - diff
    for deg, orb, name in [
        (0,8,"conjunction"), (60,6,"sextile"),
        (90,8,"square"),     (120,8,"trine"),
        (180,8,"opposition")
    ]:
        if abs(diff - deg) <= orb:
            return diff, name
    return diff, None

# =============================================================================
# KP star/sub engine (minimal)
# =============================================================================

KP_STAR_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
KP_DASHA_YEARS = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
STAR_LEN_DEG = 360.0 / 27.0  # 13°20′

def kp_star_and_sublord(ecl_deg_sidereal: float) -> Tuple[str, str, float, float]:
    """
    Args: ecl_deg_sidereal — position in sidereal zodiac.
    Returns: (star_lord, sub_lord, star_span_deg, pos_within_star_deg).
    Sub-lords proportioned by Vimshottari years, starting at the star-lord.
    """
    pos = deg_wrap(ecl_deg_sidereal)
    star_idx = int(pos // STAR_LEN_DEG)  # 0..26
    star_lord = KP_STAR_ORDER[star_idx % 9]
    inside = pos - STAR_LEN_DEG * star_idx

    total_years = 120.0
    acc = 0.0
    sub_lord = KP_STAR_ORDER[-1]
    sub_start = 0.0

    # Start sub-cycle at the star_lord
    start_i = KP_STAR_ORDER.index(star_lord)
    cycle = KP_STAR_ORDER[start_i:] + KP_STAR_ORDER[:start_i]
    for lord in cycle:
        portion = STAR_LEN_DEG * (KP_DASHA_YEARS[lord] / total_years)
        if inside < acc + portion:
            sub_lord = lord
            sub_start = acc
            break
        acc += portion

    return (star_lord, sub_lord, STAR_LEN_DEG, inside - sub_start)

# =============================================================================
# Dataclasses / Inputs
# =============================================================================

class QuestionType(Enum):
    JOB = "job"
    MARRIAGE = "marriage"
    LITIGATION = "litigation"
    HEALTH = "health"
    LOST_ITEM = "lost_item"
    PROPERTY = "property"
    FOREIGN = "foreign"
    EDUCATION = "education"
    CHILDREN = "children"
    BUSINESS = "business"

# Richer house mapping for Parāśarī & KP decisions
ENHANCED_QUESTION_HOUSES: Dict[QuestionType, Dict[str, List[int]]] = {
    QuestionType.JOB:        {"primary":[10],      "secondary":[6,2],  "supportive":[9,11], "obstructive":[12,8]},
    QuestionType.MARRIAGE:   {"primary":[7],       "secondary":[2,11], "supportive":[1,5,9], "obstructive":[6,8,12]},
    QuestionType.LITIGATION: {"primary":[6],       "secondary":[8,12], "supportive":[1,9,10], "obstructive":[7,11]},
    QuestionType.HEALTH:     {"primary":[1],       "secondary":[5,9],  "supportive":[10,11], "obstructive":[6,8,12]},
    QuestionType.LOST_ITEM:  {"primary":[2],       "secondary":[4,11], "supportive":[1,5,9], "obstructive":[8,12]},
    QuestionType.EDUCATION:  {"primary":[4,5],     "secondary":[2,9],  "supportive":[1,10,11], "obstructive":[6,8,12]},
    QuestionType.CHILDREN:   {"primary":[5],       "secondary":[1,9],  "supportive":[2,7,11], "obstructive":[6,8,12]},
    QuestionType.PROPERTY:   {"primary":[4],       "secondary":[2,11], "supportive":[1,5,9], "obstructive":[6,8,12]},
    QuestionType.FOREIGN:    {"primary":[12],      "secondary":[7,9],  "supportive":[3,11],   "obstructive":[2,4]},
    QuestionType.BUSINESS:   {"primary":[7],       "secondary":[3,10], "supportive":[2,11],   "obstructive":[6,8,12]},
}

@dataclass
class HoraryInput:
    # Moment of question
    date: Optional[str] = None     # "YYYY-MM-DD"
    time: Optional[str] = None     # "HH:MM:SS"
    tz_name: Optional[str] = None

    # Location
    place: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Astro settings (question chart)
    zodiac_mode: str = "sidereal"
    ayanamsa: str | float = "lahiri"
    ayanamsa_deg: Optional[float] = None  # explicit ayanamsa in degrees (optional)
    house_system: str = "sripati"

    # KP options (for KP method)
    kp_house_system: str = "placidus"
    kp_ayanamsa: str | float = "krishnamurti"
    kp_number: Optional[int] = None
    kp_number_mode: str = "anchor_asc"  # "anchor_asc" | "advisory"

    # Question semantics
    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None
    querent_house: int = 1
    quesited_house: Optional[int] = None

@dataclass
class QuerentBirthData:
    date: str
    time: str
    tz_name: str
    place: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    zodiac_mode: str = "sidereal"
    ayanamsa: str | float = "lahiri"

@dataclass
class HybridPrasnaInput:
    # Question moment
    question_date: Optional[str] = None
    question_time: Optional[str] = None
    question_tz: Optional[str] = None
    question_place: Optional[str] = None
    question_latitude: Optional[float] = None
    question_longitude: Optional[float] = None

    # Birth
    querent_birth: Optional[QuerentBirthData] = None

    # Meta
    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None

    # Astro settings
    zodiac_mode: str = "sidereal"
    ayanamsa: str | float = "lahiri"
    house_system: str = "sripati"

# =============================================================================
# Chart & Houses helpers
# =============================================================================

def ensure_coords_and_tz(
    date: Optional[str],
    time_: Optional[str],
    tz: Optional[str],
    place: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> Tuple[str, str, str, float, float]:
    """
    Fill missing lat/lon/tz from resolve_place; default to now/UTC if needed.
    Returns (date, time, tz, lat, lon) with concrete values.
    """
    if not (date and time_):
        now = datetime.now(timezone.utc)
        date = now.date().isoformat()
        time_ = now.time().replace(microsecond=0).isoformat()
        tz = tz or "UTC"

    if (lat is None or lon is None) and place:
        rp = resolve_place(place)
        lat = lat if lat is not None else rp.get("lat")
        lon = lon if lon is not None else rp.get("lon")
        tz = tz or rp.get("tz") or "UTC"

    tz = tz or "UTC"
    if lat is None or lon is None:
        lat, lon = 0.0, 0.0

    return str(date), str(time_), str(tz), float(lat), float(lon)

def build_chart(
    *,
    date: Optional[str],
    time: Optional[str],
    tz: Optional[str],
    place: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    zodiac_mode: str,
    ayanamsa: str | float,
    topocentric: bool = True,
) -> Dict[str, Any]:
    """
    Wrapper over compute_chart with safe defaults for missing dt/place/coords.
    """
    d, t, tzr, la, lo = ensure_coords_and_tz(date, time, tz, place, latitude, longitude)
    return compute_chart({
        "date": d, "time": t, "tz": tzr,
        "place": place, "latitude": la, "longitude": lo,
        "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa,
        "topocentric": bool(topocentric),
    })

def _pick_ts(ts: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = ts.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None

def safe_get_asc(chart: Dict[str, Any]) -> Optional[float]:
    ang = chart.get("angles") or {}
    return ang.get("asc_deg", chart.get("asc_deg"))

def compute_houses_from_chart(
    chart: Dict[str, Any],
    *,
    latitude: float,
    longitude: float,
    house_system: str,
    zodiac_mode: str,
    ayanamsa_deg: Optional[float],
) -> Dict[str, Any]:
    """
    Calls app.core.houses_advanced.compute_house_system with whatever JD inputs
    are available from compute_chart(...).meta.timescales. If that fails,
    returns Equal Houses from ASC. Applies sidereal shift to cusps if requested.
    """
    ts = (chart.get("meta", {}) or {}).get("timescales", {}) or {}
    jd_ut  = _pick_ts(ts, "jd_ut", "jd_utc")
    jd_tt  = _pick_ts(ts, "jd_tt", "tt_jd", "jd_tdb")
    jd_ut1 = _pick_ts(ts, "jd_ut1", "ut1_jd")

    # Try multiple calling signatures for robustness across versions
    trials = [
        dict(latitude=latitude, longitude=longitude, house_system=house_system,
             jd_ut=jd_ut, jd_tt=jd_tt, jd_ut1=jd_ut1),
        dict(latitude=latitude, longitude=longitude, house_system=house_system,
             jd_tt=jd_tt, jd_ut1=jd_ut1),
        dict(latitude=latitude, longitude=longitude, house_system=house_system,
             jd_ut=jd_ut, jd_ut1=jd_ut1),
        dict(latitude=latitude, longitude=longitude, house_system=house_system,
             jd_tt=jd_tt),
    ]

    payload = None
    for kwargs in trials:
        # Require supplied JD values
        if "jd_tt" in kwargs and kwargs["jd_tt"] is None: 
            continue
        if "jd_ut1" in kwargs and kwargs["jd_ut1"] is None:
            continue
        if "jd_ut" in kwargs and kwargs["jd_ut"] is None:
            continue
        try:
            payload = compute_house_system(**kwargs)  # type: ignore[arg-type]
            break
        except TypeError:
            continue
        except Exception:
            continue

    # Fallback: Equal from ASC
    if not payload:
        asc_any = safe_get_asc(chart)
        if asc_any is None:
            raise ValueError("houses_fallback_failed:no_asc_in_chart")
        asc_deg = float(asc_any)
        cusps = [deg_wrap(asc_deg + i * 30.0) for i in range(12)]
        ang = chart.get("angles") or {}
        mc_guess = (ang.get("mc_deg") if ang.get("mc_deg") is not None
                    else deg_wrap(asc_deg + 90.0))

        if zodiac_mode.lower() == "sidereal" and isinstance(ayanamsa_deg, (int, float)):
            cusps = shift_sidereal(cusps, float(ayanamsa_deg))
            asc_out = deg_wrap(asc_deg - float(ayanamsa_deg))
            mc_out  = deg_wrap(float(mc_guess) - float(ayanamsa_deg))
        else:
            asc_out = asc_deg
            mc_out  = float(mc_guess)

        return {
            "house_system": f"{house_system} (fallback=Equal from ASC)",
            "cusps_deg": cusps,
            "asc_deg": asc_out,
            "mc_deg": mc_out,
            "vertex": None,
            "eastpoint": None,
            "warnings": ["houses_engine_unavailable_fallback_equal"],
        }

    # Advanced payload → apply sidereal shift if requested
    cusps = list(payload["cusps_deg"])
    asc_deg_h = float(payload["asc_deg"])
    mc_deg_h  = float(payload["mc_deg"])

    if zodiac_mode.lower() == "sidereal" and isinstance(ayanamsa_deg, (int, float)):
        cusps     = shift_sidereal(cusps, float(ayanamsa_deg))
        asc_deg_h = deg_wrap(asc_deg_h - float(ayanamsa_deg))
        mc_deg_h  = deg_wrap(mc_deg_h  - float(ayanamsa_deg))

    return {
        "house_system": payload.get("house_system", house_system),
        "cusps_deg": cusps,
        "asc_deg": asc_deg_h,
        "mc_deg": mc_deg_h,
        "vertex": payload.get("vertex"),
        "eastpoint": payload.get("eastpoint"),
        "warnings": payload.get("warnings", []),
    }

# =============================================================================
# Radicality (light)
# =============================================================================

def radicality_flags(chart: Dict[str, Any], tz_name: str, date: str, time_: str) -> Dict[str, Any]:
    """
    Lightweight 'radicality' check: ASC lord vs day/hour lord.
    Hour lord is approximated as the day lord (placeholder).
    """
    asc_deg = safe_get_asc(chart)
    asc_lord = lord_of_sign(float(asc_deg or 0.0))

    try:
        from zoneinfo import ZoneInfo
        dt_local = datetime.fromisoformat(f"{date}T{time_}").replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        dt_local = datetime.fromisoformat(f"{date}T{time_}").replace(tzinfo=timezone.utc)

    # Monday..Sunday → Moon..Sun
    WEEK_TO_LORD = ["Moon","Mars","Mercury","Jupiter","Venus","Saturn","Sun"]
    daylord = WEEK_TO_LORD[dt_local.weekday()]
    hour_lord = daylord
    fits = (asc_lord == daylord) or (asc_lord == hour_lord)

    return {
        "asc_lord": asc_lord,
        "day_lord": daylord,
        "hour_lord": hour_lord,
        "fits": fits,
    }

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Dataclasses & enums
    "HoraryInput", "QuerentBirthData", "HybridPrasnaInput", "QuestionType",
    "ENHANCED_QUESTION_HOUSES",

    # Constants
    "TRAD_PLANETS", "SIGN_NAMES", "SIGN_LORDS",

    # Helpers
    "deg_wrap", "sign_index", "sign_name_from_deg", "lord_of_sign",
    "angular_sep", "shift_sidereal", "house_of",
    "calculate_planetary_dignity", "calculate_aspects",

    # KP
    "KP_STAR_ORDER", "KP_DASHA_YEARS", "STAR_LEN_DEG", "kp_star_and_sublord",

    # Chart/Houses
    "ensure_coords_and_tz", "build_chart", "compute_houses_from_chart", "safe_get_asc",

    # Radicality
    "radicality_flags",
]
