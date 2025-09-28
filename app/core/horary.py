# -*- coding: utf-8 -*-
"""
Horary / Prashna systems (Parāśari, KP, Hybrid) — single-file module.

Key points
----------
- No dependency on app.core.houses.compute_houses (which was missing).
  We call app.core.houses_advanced.compute_house_system(...) directly.
- Timescales come from compute_chart(...).meta.timescales (jd_tt, jd_ut1, jd_utc).
- If zodiac_mode == "sidereal", house cusps are shifted by ayanamsa so
  house tests (occupancy, sign lords, KP sub-lords) are internally consistent.
- Public entrypoint: analyze_prasna_enhanced(method=...), with helpers per system.

Exported API
------------
- HoraryInput, QuestionType
- QuerentBirthData, HybridPrasnaInput
- analyze_prasna_enhanced (for routes), analyze_prasna
- analyze_parashari, analyze_kp, analyze_hybrid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone
from enum import Enum
import math

# --- Core engines (already in your project) ---
from app.core.astronomy import compute_chart, resolve_place
from app.core.houses_advanced import compute_house_system  # strict JD(TT)+UT1

# =============================================================================
# Basic helpers
# =============================================================================

DEG = math.degrees
RAD = math.radians

SIGN_NAMES = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

# Use the traditional 7 for horary scoring; we’ll also accept any present bodies.
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

def house_of(long_deg: float, cusps_deg: List[float]) -> int:
    """
    Return house number 1..12 for a longitude given cusp longitudes (H1..H12).
    Correctly handles circular wrap between H12→H1.
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

def shift_sidereal(values: List[float], ay_deg: float) -> List[float]:
    """Shift a list of ecliptic longitudes by -ayanamsa (tropical→sidereal)."""
    return [deg_wrap(v - ay_deg) for v in values]

# =============================================================================
# KP partitions (nakshatra/sub) — minimal engine for cusp sub-lord & planet labels
# =============================================================================

KP_STAR_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
KP_DASHA_YEARS = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
STAR_LEN_DEG = 360.0 / 27.0  # 13°20′

def kp_star_and_sublord(ecl_deg_sidereal: float) -> Tuple[str, str, float, float]:
    """
    Return (star_lord, sub_lord, star_span_deg, pos_within_star_deg).
    Sub-lords proportioned by Vimshottari years (starting at the star-lord).
    """
    pos = deg_wrap(ecl_deg_sidereal)
    star_idx = int(pos // STAR_LEN_DEG)  # 0..26
    star_lord = KP_STAR_ORDER[star_idx % 9]
    inside = pos - STAR_LEN_DEG * star_idx

    total_years = 120.0
    acc = 0.0
    sub_lord = KP_STAR_ORDER[-1]
    sub_start = 0.0

    # start at star_lord in the Vim cycle
    cycle = KP_STAR_ORDER[(KP_STAR_ORDER.index(star_lord)):] + KP_STAR_ORDER[:KP_STAR_ORDER.index(star_lord)]
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

# Richer house mapping for Parāśari & KP decisions
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
    # moment
    date: Optional[str] = None     # "YYYY-MM-DD"
    time: Optional[str] = None     # "HH:MM:SS"
    tz_name: Optional[str] = None
    # location
    place: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # astro mode
    zodiac_mode: str = "sidereal"
    ayanamsa: str | float = "lahiri"
    house_system: str = "sripati"
    # KP options
    kp_house_system: str = "placidus"
    kp_ayanamsa: str | float = "krishnamurti"
    kp_number: Optional[int] = None
    kp_number_mode: str = "anchor_asc"  # "anchor_asc" | "advisory"
    # question
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
    # meta
    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None
    # astro settings
    zodiac_mode: str = "sidereal"
    ayanamsa: str | float = "lahiri"
    house_system: str = "sripati"

# =============================================================================
# Shared: dignity, aspects, etc.
# =============================================================================

def calculate_planetary_dignity(longitude: float, planet: str) -> float:
    """
    Simple dignity score (-2..+2). Tunable if you want finer granularity.
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
    """Loose Ptolemaic aspects; returns (sep, name|None)."""
    diff = abs(deg_wrap(long1 - long2))
    if diff > 180: diff = 360 - diff
    for deg, orb, name in [(0,8,"conjunction"),(60,6,"sextile"),
                           (90,8,"square"),(120,8,"trine"),(180,8,"opposition")]:
        if abs(diff - deg) <= orb:
            return diff, name
    return diff, None

# =============================================================================
# Low-level: get charts + houses in a consistent way (with sidereal shift for cusps)
# =============================================================================

def _ensure_coords_and_tz(date: Optional[str], time_: Optional[str], tz: Optional[str],
                          place: Optional[str], lat: Optional[float], lon: Optional[float]) -> Tuple[str,str,str,float,float]:
    """Fill missing lat/lon/tz from resolve_place; default to now/UTC if needed."""
    if not (date and time_):
        now = datetime.now(timezone.utc)
        date = now.date().isoformat()
        time_ = now.time().replace(microsecond=0).isoformat()
        tz = tz or "UTC"
    # resolve place if needed
    if (lat is None or lon is None) and place:
        rp = resolve_place(place)
        lat = lat if lat is not None else rp.get("lat")
        lon = lon if lon is not None else rp.get("lon")
        tz = tz or rp.get("tz") or "UTC"
    tz = tz or "UTC"
    if lat is None or lon is None:
        # last resort: (0,0)
        lat, lon = 0.0, 0.0
    return str(date), str(time_), str(tz), float(lat), float(lon)

def _chart(date: Optional[str], time_: Optional[str], tz: Optional[str],
           place: Optional[str], lat: Optional[float], lon: Optional[float],
           *, zodiac_mode: str, ayanamsa: str|float, topocentric: bool=True) -> Dict[str, Any]:
    d, t, tzr, la, lo = _ensure_coords_and_tz(date, time_, tz, place, lat, lon)
    return compute_chart({
        "date": d, "time": t, "tz": tzr,
        "place": place, "latitude": la, "longitude": lo,
        "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa,
        "topocentric": bool(topocentric)
    })

def _houses_from_chart(
    chart: Dict[str, Any],
    *,
    latitude: float,
    longitude: float,
    house_system: str,
    zodiac_mode: str,
    ayanamsa_deg: Optional[float]
) -> Dict[str, Any]:
    """
    Use compute_house_system with strict timescales coming from chart.meta.timescales,
    then optionally shift cusps by ayanamsa for sidereal mode.
    """
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

    ts = chart.get("meta", {}).get("timescales", {}) or {}

    # Be tolerant to different field names from the astronomy layer
    jd_ut  = _pick_ts(ts, "jd_ut", "jd_utc")         # UT (some builds call it jd_ut)
    jd_tt  = _pick_ts(ts, "jd_tt", "tt_jd", "jd_tdb")# TT (some builds expose tt_jd/jd_tdb)
    jd_ut1 = _pick_ts(ts, "jd_ut1", "ut1_jd")        # UT1

    if jd_ut is None or jd_tt is None or jd_ut1 is None:
        have = { "jd_ut": jd_ut, "jd_tt": jd_tt, "jd_ut1": jd_ut1, "raw_keys": list(ts.keys()) }
        raise ValueError(f"Timescales incomplete for houses: {have}")

    payload = compute_house_system(
        latitude=latitude,
        longitude=longitude,
        house_system=house_system,
        jd_ut=jd_ut,
        jd_tt=jd_tt,
        jd_ut1=jd_ut1,
    )

    cusps = list(payload["cusps_deg"])
    asc_deg_h = float(payload["asc_deg"])
    mc_deg_h  = float(payload["mc_deg"])

    if zodiac_mode.lower() == "sidereal" and isinstance(ayanamsa_deg, (int, float)):
        cusps = shift_sidereal(cusps, float(ayanamsa_deg))
        asc_deg_h = deg_wrap(asc_deg_h - float(ayanamsa_deg))
        mc_deg_h  = deg_wrap(mc_deg_h  - float(ayanamsa_deg))

    return {
        "house_system": payload["house_system"],
        "cusps_deg": cusps,
        "asc_deg": asc_deg_h,
        "mc_deg": mc_deg_h,
        "vertex": payload.get("vertex"),
        "eastpoint": payload.get("eastpoint"),
        "warnings": payload.get("warnings", []),
    }

# =============================================================================
# Parāśari system
# =============================================================================

# Richer mapping reused here
_ENHANCED_QH = ENHANCED_QUESTION_HOUSES

def _calculate_house_strength(house_num: int, question_type: Optional[QuestionType]) -> float:
    if not question_type or question_type not in _ENHANCED_QH:
        return 0.5
    m = _ENHANCED_QH[question_type]
    if house_num in m.get("primary", []):    return 1.0
    if house_num in m.get("secondary", []):  return 0.8
    if house_num in m.get("supportive", []): return 0.6
    if house_num in m.get("obstructive", []):return 0.2
    return 0.4

def _parashari_strengths(chart: Dict[str, Any], cusps: List[float],
                         question_type: Optional[QuestionType]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    byname = {b["name"]: b for b in chart.get("bodies", [])}
    for nm, b in byname.items():
        if nm not in TRAD_PLANETS:
            continue
        lon = float(b["longitude_deg"])
        h = house_of(lon, cusps)
        dig = calculate_planetary_dignity(lon, nm)
        hstr = _calculate_house_strength(h, question_type)
        is_retro = bool((b.get("speed_deg_per_day") or b.get("speed") or 0) < 0)
        overall = (dig + hstr + 2.0) / 4.0  # normalize approx to 0..1
        out.append({
            "planet": nm, "longitude_deg": lon, "sign": sign_name_from_deg(lon),
            "house": h, "is_retrograde": is_retro,
            "dignity_score": dig, "house_strength": hstr,
            "overall_strength": round(overall, 4)
        })
    return out

def _radicality_flags(chart: Dict[str, Any], tz_name: str, date: str, time_: str) -> Dict[str, Any]:
    """Lightweight: ASC lord vs day/hour lord. Hour lord ~ day lord placeholder."""
    asc_deg = chart.get("angles", {}).get("asc_deg", chart.get("asc_deg"))
    asc_lord = lord_of_sign(float(asc_deg or 0.0))
    # day lord from local date (we assume the given tz_name)
    try:
        from zoneinfo import ZoneInfo
        dt_local = datetime.fromisoformat(f"{date}T{time_}").replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        dt_local = datetime.fromisoformat(f"{date}T{time_}").replace(tzinfo=timezone.utc)
    WEEK_TO_LORD = ["Moon","Mars","Mercury","Jupiter","Venus","Saturn","Sun"]  # Mon..Sun
    daylord = WEEK_TO_LORD[dt_local.weekday()]
    hour_lord = daylord  # placeholder
    fits = (asc_lord == daylord) or (asc_lord == hour_lord)
    return {"asc_lord": asc_lord, "day_lord": daylord, "hour_lord": hour_lord, "fits": fits}

def analyze_parashari(inp: HoraryInput) -> Dict[str, Any]:
    # Chart (topocentric) — compute first so we get timescales + ayanamsa
    chart = _chart(inp.date, inp.time, inp.tz_name, inp.place, inp.latitude, inp.longitude,
                   zodiac_mode=inp.zodiac_mode, ayanamsa=inp.ayanamsa, topocentric=True)

    meta = chart.get("meta", {})
    ay_deg = meta.get("ayanamsa_deg")
    observer = meta.get("observer") or {}
    la = float(observer.get("latitude", inp.latitude or 0.0))
    lo = float(observer.get("longitude", inp.longitude or 0.0))

    # Houses (strict) shifted to sidereal if needed
    houses = _houses_from_chart(chart, latitude=la, longitude=lo,
                                house_system=inp.house_system,
                                zodiac_mode=inp.zodiac_mode,
                                ayanamsa_deg=ay_deg)

    cusps = houses.get("cusps_deg", []) or []
    asc_deg = chart.get("angles", {}).get("asc_deg", chart.get("asc_deg"))
    moon_lon = next((b["longitude_deg"] for b in chart.get("bodies", []) if b["name"]=="Moon"), None)

    qtype = inp.question_type or QuestionType.JOB
    strengths = _parashari_strengths(chart, cusps, qtype)

    # House lords (by sign on cusps)
    house_lords: Dict[int, str] = {i+1: lord_of_sign(c) for i, c in enumerate(cusps)}

    # Scoring
    mapping = _ENHANCED_QH.get(qtype, {})
    primary, secondary = mapping.get("primary", []), mapping.get("secondary", [])
    supportive, obstructive = mapping.get("supportive", []), mapping.get("obstructive", [])

    primary_score = secondary_score = obstruction_score = 0.0
    for s in strengths:
        if s["house"] in primary:      primary_score   += s["overall_strength"] * 3.0
        elif s["house"] in secondary:  secondary_score += s["overall_strength"] * 2.0
        elif s["house"] in supportive: secondary_score += s["overall_strength"] * 1.0
        elif s["house"] in obstructive:obstruction_score += s["overall_strength"] * 2.0

    # Moon proximity bonus to lords of target houses
    prox_bonus = 0.0
    if moon_lon is not None:
        target_lords = {house_lords.get(h) for h in (primary + secondary)}
        bodies = {b["name"]: b for b in chart.get("bodies", [])}
        for pl in filter(None, target_lords):
            if pl in bodies:
                if angular_sep(float(moon_lon), float(bodies[pl]["longitude_deg"])) < 12.0:
                    prox_bonus += 1.0

    # Radicality multiplier
    date = inp.date or datetime.now(timezone.utc).date().isoformat()
    time_ = inp.time or datetime.now(timezone.utc).time().replace(microsecond=0).isoformat()
    rad = _radicality_flags(chart, inp.tz_name or "UTC", date, time_)
    rad_mult = 1.10 if rad["fits"] else 1.0

    total_positive = (primary_score + secondary_score + prox_bonus) * rad_mult
    total_negative = obstruction_score
    net_score = total_positive - total_negative

    if net_score > 0.75:
        answer, conf = "yes", min(0.95, 0.72 + 0.06 * net_score)
    elif net_score < -0.75:
        answer, conf = "no",  min(0.95, 0.72 + 0.06 * abs(net_score))
    else:
        answer, conf = "uncertain", 0.60

    return {
        "ok": True,
        "system": "parashari",
        "meta": {
            "question_type": qtype.value,
            "zodiac_mode": inp.zodiac_mode,
            "ayanamsa": inp.ayanamsa,
            "house_system": inp.house_system,
            "analysis_time": datetime.now(timezone.utc).isoformat()
        },
        "chart_data": {
            "ASC_deg": asc_deg,
            "ASC_sign": sign_name_from_deg(float(asc_deg or 0.0)),
            "Moon_deg": moon_lon,
            "Moon_sign": (sign_name_from_deg(moon_lon) if moon_lon is not None else None),
            "cusps_deg": cusps,
            "house_lords": house_lords
        },
        "planetary_analysis": strengths,
        "scoring_breakdown": {
            "primary_score": round(primary_score, 3),
            "secondary_score": round(secondary_score, 3),
            "obstruction_score": round(obstruction_score, 3),
            "moon_proximity_bonus": round(prox_bonus, 3),
            "radicality_multiplier": rad_mult,
            "net_score": round(net_score, 3)
        },
        "judgement": {"answer": answer, "confidence": round(conf, 3)}
    }

# =============================================================================
# KP system
# =============================================================================

def kp_signified_houses(planet_name: str, chart: Dict[str, Any], cusps: List[float]) -> Set[int]:
    """
    Minimal KP signification: sub-lord's own occupancy + lordship of cusp signs,
    and star-lord's occupancy + lordship.
    """
    houses: Set[int] = set()
    bodies = {b["name"]: b for b in chart.get("bodies", [])}
    p = bodies.get(planet_name)
    if not p:
        return houses

    # Occupancy of planet
    houses.add(house_of(p["longitude_deg"], cusps))

    # Lordship of cusp signs
    for i, cusp in enumerate(cusps, 1):
        if lord_of_sign(cusp) == planet_name:
            houses.add(i)

    # Star-lord chain
    star_lord, _, _, _ = kp_star_and_sublord(p["longitude_deg"])
    s = bodies.get(star_lord)
    if s:
        houses.add(house_of(s["longitude_deg"], cusps))
        for i, cusp in enumerate(cusps, 1):
            if lord_of_sign(cusp) == star_lord:
                houses.add(i)
    return houses

def kp_cusp_sub_lord_ok(cusp_house: int, chart: Dict[str, Any], cusps: List[float],
                        positive: Set[int], negative: Set[int]) -> Tuple[bool, Set[int], str]:
    cusp_deg = cusps[cusp_house-1]
    _, sub_lord, _, _ = kp_star_and_sublord(cusp_deg)
    sig = kp_signified_houses(sub_lord, chart, cusps)
    good = (len(sig & positive) > 0) and (len(sig & negative) == 0)
    reason = f"sub={sub_lord}, signified={sorted(sig)}"
    return good, sig, reason

def analyze_kp(inp: HoraryInput) -> Dict[str, Any]:
    kp_aya = inp.kp_ayanamsa if inp.kp_ayanamsa is not None else "krishnamurti"

    # Chart in SIDEREAL with KP ayanamsa
    chart = _chart(inp.date, inp.time, inp.tz_name, inp.place, inp.latitude, inp.longitude,
                   zodiac_mode="sidereal", ayanamsa=kp_aya, topocentric=True)
    meta = chart.get("meta", {})
    ay_deg = meta.get("ayanamsa_deg")
    observer = meta.get("observer") or {}
    la = float(observer.get("latitude", inp.latitude or 0.0))
    lo = float(observer.get("longitude", inp.longitude or 0.0))

    # Houses (KP-selected system), shift already handled via sidereal branch
    houses = _houses_from_chart(chart, latitude=la, longitude=lo,
                                house_system=inp.kp_house_system,
                                zodiac_mode="sidereal", ayanamsa_deg=ay_deg)
    cusps = houses.get("cusps_deg", []) or []

    # (Optional) KP number anchoring of ASC (advisory/override)
    asc_deg_sid = chart.get("angles", {}).get("asc_deg", chart.get("asc_deg"))
    if inp.kp_number and (inp.kp_number_mode or "anchor_asc").lower() == "anchor_asc":
        # 1..249 mapped evenly on zodiac (anchor only, we do not rotate houses)
        asc_deg_sid = (max(1, min(249, int(inp.kp_number))) - 1) * (360.0 / 249.0)

    # Decision via cusp sub-lords on primary houses
    qtype = inp.question_type or QuestionType.JOB
    mapping = _ENHANCED_QH.get(qtype, {})
    pos_h = set(mapping.get("primary", []) + mapping.get("secondary", []) + mapping.get("supportive", []))
    neg_h = set(mapping.get("obstructive", []))

    yes_hits = no_hits = 0
    evidence = []
    for h in mapping.get("primary", []):
        ok, sig, reason = kp_cusp_sub_lord_ok(h, chart, cusps, pos_h, neg_h)
        evidence.append({"cusp": h, "ok": ok, "reason": reason})
        yes_hits += int(ok)
        no_hits += int(not ok)

    if yes_hits > no_hits:
        answer, conf = "yes", round(0.68 + 0.06*(yes_hits - no_hits), 3)
    elif no_hits > yes_hits:
        answer, conf = "no",  round(0.68 + 0.06*(no_hits - yes_hits), 3)
    else:
        answer, conf = "uncertain", 0.62

    # Cusp table with star/sub
    detailed_cusps = []
    for i, cusp_deg in enumerate(cusps, 1):
        star_lord, sub_lord, star_deg, sub_pos = kp_star_and_sublord(cusp_deg)
        detailed_cusps.append({
            "house": i,
            "degree": round(cusp_deg, 4),
            "sign": sign_name_from_deg(cusp_deg),
            "star_lord": star_lord,
            "sub_lord": sub_lord,
            "star_span_deg": round(star_deg, 4),
            "pos_within_star_deg": round(sub_pos, 4)
        })

    # Planetary star/sub snapshot
    planetary_kp = []
    for b in chart.get("bodies", []):
        nm = b["name"]
        if nm not in TRAD_PLANETS:
            continue
        lon = float(b["longitude_deg"])
        star_lord, sub_lord, star_deg, sub_pos = kp_star_and_sublord(lon)
        planetary_kp.append({
            "planet": nm,
            "longitude_deg": round(lon, 4),
            "sign": sign_name_from_deg(lon),
            "star_lord": star_lord,
            "sub_lord": sub_lord,
            "star_span_deg": round(star_deg, 4),
            "pos_within_star_deg": round(sub_pos, 4)
        })

    return {
        "ok": True,
        "system": "kp",
        "meta": {
            "question_type": qtype.value,
            "ayanamsa": kp_aya,
            "house_system": inp.kp_house_system,
            "kp_number": inp.kp_number,
            "kp_number_mode": inp.kp_number_mode,
            "analysis_time": datetime.now(timezone.utc).isoformat()
        },
        "kp_core": {
            "ASC_deg_sid": asc_deg_sid,
            "ASC_sign": sign_name_from_deg(float(asc_deg_sid or 0.0)),
        },
        "cusp_analysis": detailed_cusps,
        "planetary_kp": planetary_kp,
        "signification_evidence": evidence,
        "judgement": {"answer": answer, "confidence": conf, "method": "KP cusp sub-lords"}
    }

# =============================================================================
# Hybrid system (Birth chart + Question moment)
# =============================================================================

def _find_planet_house(longitude: float, cusps: List[float]) -> int:
    """House membership using forward intervals (wrap-safe)."""
    if not cusps or len(cusps) != 12:
        return 1
    for i in range(12):
        a = deg_wrap(cusps[i])
        b = deg_wrap(cusps[(i + 1) % 12])
        x = deg_wrap(longitude)
        if a <= b:
            if a <= x < b: return i + 1
        else:
            if x >= a or x < b: return i + 1
    return 1

def _compare_birth_question(birth_chart: Dict, question_chart: Dict,
                            birth_cusps: List[float], question_cusps: List[float]) -> List[Dict[str, Any]]:
    comps = []
    birth_planets = {p["name"]: p for p in birth_chart.get("bodies", []) if p["name"] in TRAD_PLANETS}
    question_planets = {p["name"]: p for p in question_chart.get("bodies", []) if p["name"] in TRAD_PLANETS}
    for nm in TRAD_PLANETS:
        if nm in birth_planets and nm in question_planets:
            b = birth_planets[nm]; q = question_planets[nm]
            bL, qL = float(b["longitude_deg"]), float(q["longitude_deg"])
            bH = _find_planet_house(bL, birth_cusps)
            qH = _find_planet_house(qL, question_cusps)
            sep, asp = calculate_aspects(bL, qL)
            dig_b = calculate_planetary_dignity(bL, nm)
            dig_q = calculate_planetary_dignity(qL, nm)
            comps.append({
                "planet": nm,
                "birth_longitude": bL,
                "birth_sign": sign_name_from_deg(bL),
                "birth_house": bH,
                "question_longitude": qL,
                "question_sign": sign_name_from_deg(qL),
                "question_house": qH,
                "angular_distance": sep,
                "aspect_type": asp,
                "strength_change": dig_q - dig_b
            })
    return comps

def analyze_hybrid(inp: HybridPrasnaInput) -> Dict[str, Any]:
    if not inp.querent_birth:
        return {"ok": False, "system": "hybrid", "error": "Querent birth data required"}

    # Question chart/houses
    q_chart = _chart(inp.question_date, inp.question_time, inp.question_tz,
                     inp.question_place, inp.question_latitude, inp.question_longitude,
                     zodiac_mode=inp.zodiac_mode, ayanamsa=inp.ayanamsa, topocentric=True)
    q_meta = q_chart.get("meta", {})
    q_aya = q_meta.get("ayanamsa_deg")
    q_obs = q_meta.get("observer") or {}
    q_la = float(q_obs.get("latitude", inp.question_latitude or 0.0))
    q_lo = float(q_obs.get("longitude", inp.question_longitude or 0.0))
    q_houses = _houses_from_chart(q_chart, latitude=q_la, longitude=q_lo,
                                  house_system=inp.house_system,
                                  zodiac_mode=inp.zodiac_mode, ayanamsa_deg=q_aya)

    # Birth chart/houses
    b = inp.querent_birth
    b_chart = _chart(b.date, b.time, b.tz_name, b.place, b.latitude, b.longitude,
                     zodiac_mode=b.zodiac_mode, ayanamsa=b.ayanamsa, topocentric=True)
    b_meta = b_chart.get("meta", {})
    b_aya = b_meta.get("ayanamsa_deg")
    b_obs = b_meta.get("observer") or {}
    b_la = float(b_obs.get("latitude", b.latitude or 0.0))
    b_lo = float(b_obs.get("longitude", b.longitude or 0.0))
    b_houses = _houses_from_chart(b_chart, latitude=b_la, longitude=b_lo,
                                  house_system=inp.house_system,
                                  zodiac_mode=b.zodiac_mode, ayanamsa_deg=b_aya)

    # Comparisons
    comps = _compare_birth_question(b_chart, q_chart,
                                    b_houses.get("cusps_deg", []),
                                    q_houses.get("cusps_deg", []))

    # Evaluate components
    qtype = inp.question_type or QuestionType.JOB
    target = _ENHANCED_QH.get(qtype, {})
    birth_relevance = question_strength = 0.0

    for p in b_chart.get("bodies", []):
        if p["name"] not in TRAD_PLANETS: continue
        h = _find_planet_house(p["longitude_deg"], b_houses.get("cusps_deg", []))
        dig = calculate_planetary_dignity(p["longitude_deg"], p["name"])
        if h in target.get("primary", []):   birth_relevance += (dig + 2) * 0.3
        elif h in target.get("secondary", []): birth_relevance += (dig + 2) * 0.2
    birth_relevance = max(0.0, min(1.0, birth_relevance))

    for p in q_chart.get("bodies", []):
        if p["name"] not in TRAD_PLANETS: continue
        h = _find_planet_house(p["longitude_deg"], q_houses.get("cusps_deg", []))
        dig = calculate_planetary_dignity(p["longitude_deg"], p["name"])
        if h in target.get("primary", []):   question_strength += (dig + 2) * 0.3
        elif h in target.get("secondary", []): question_strength += (dig + 2) * 0.2
    question_strength = max(0.0, min(1.0, question_strength))

    # Placeholder dasha support (tunable / plug real Vimshottari)
    moon_long = next((p["longitude_deg"] for p in b_chart.get("bodies", []) if p["name"]=="Moon"), 0.0)
    moon_lord = lord_of_sign(moon_long)
    dasha_support = 0.5 + (0.2 if moon_lord in ("Jupiter","Venus","Moon") else (-0.1 if moon_lord in ("Saturn","Mars") else 0.0))
    dasha_support = max(0.0, min(1.0, dasha_support))

    # Transit support proxy (constant for now; slot for your transit engine)
    transit_support = 0.6

    # Harmony (beneficial aspect fraction among TRAD planets)
    bene, total = 0, 0
    q_map = {p["name"]: p for p in q_chart.get("bodies", []) if p["name"] in TRAD_PLANETS}
    b_map = {p["name"]: p for p in b_chart.get("bodies", []) if p["name"] in TRAD_PLANETS}
    for nm in TRAD_PLANETS:
        if nm in q_map and nm in b_map:
            _, asp = calculate_aspects(q_map[nm]["longitude_deg"], b_map[nm]["longitude_deg"])
            if asp in ("trine","sextile","conjunction"): bene += 1
            total += 1
    harmony = (bene / total) if total else 0.0

    # Composite
    W = {"birth":0.25, "question":0.30, "dasha":0.20, "transit":0.15, "harm":0.10}
    composite = (birth_relevance*W["birth"] + question_strength*W["question"] +
                 dasha_support*W["dasha"] + transit_support*W["transit"] +
                 harmony*W["harm"])

    if composite > 0.65:
        answer = "yes"; confidence = 0.70 + (composite - 0.65) * 0.6
    elif composite < 0.35:
        answer = "no";  confidence = 0.70 + (0.35 - composite) * 0.6
    else:
        answer = "uncertain"; confidence = 0.55 + abs(composite - 0.5) * 0.2

    return {
        "ok": True,
        "system": "hybrid",
        "meta": {
            "question_type": qtype.value,
            "analysis_time": datetime.now(timezone.utc).isoformat(),
            "zodiac_mode": inp.zodiac_mode,
            "ayanamsa": inp.ayanamsa
        },
        "charts": {
            "birth_asc": f"{b_chart.get('angles',{}).get('asc_deg', b_chart.get('asc_deg')):.2f}° {sign_name_from_deg(b_chart.get('angles',{}).get('asc_deg', b_chart.get('asc_deg')))}",
            "question_asc": f"{q_chart.get('angles',{}).get('asc_deg', q_chart.get('asc_deg')):.2f}° {sign_name_from_deg(q_chart.get('angles',{}).get('asc_deg', q_chart.get('asc_deg')))}",
        },
        "analysis_components": {
            "birth_chart_relevance": round(birth_relevance, 3),
            "question_chart_strength": round(question_strength, 3),
            "dasha_timing_support": round(dasha_support, 3),
            "transit_support": round(transit_support, 3),
            "chart_harmony": round(harmony, 3),
            "composite_score": round(composite, 3)
        },
        "chart_comparisons": comps,
        "judgement": {
            "answer": answer,
            "confidence": round(min(0.95, confidence), 3),
            "reasoning": f"Hybrid synthesis of birth potential and question-moment factors ({qtype.value})."
        }
    }

# =============================================================================
# Public dispatcher
# =============================================================================

def analyze_prasna(inp: HoraryInput | HybridPrasnaInput,
                   method: str = "parashari") -> Dict[str, Any]:
    """
    method in {"parashari","kp","hybrid"}
    - For "hybrid", pass HybridPrasnaInput (with querent_birth).
    - For "parashari"/"kp", pass HoraryInput.
    """
    try:
        m = (method or "parashari").strip().lower()
        if m == "kp":
            assert isinstance(inp, HoraryInput)
            return analyze_kp(inp)
        if m == "hybrid":
            assert isinstance(inp, HybridPrasnaInput)
            return analyze_hybrid(inp)
        assert isinstance(inp, HoraryInput)
        return analyze_parashari(inp)
    except Exception as e:
        return {"ok": False, "system": method, "error": str(e), "error_type": type(e).__name__}

# --- Adapter for vedic_routes compatibility ----------------------------------

def analyze_prasna_enhanced(inp, method: str = "parashari"):
    """
    Thin wrapper so routes can import analyze_prasna_enhanced().
    Delegates to analyze_prasna() and adds a 'method' key for convenience.
    """
    res = analyze_prasna(inp, method=method)
    if isinstance(res, dict):
        res.setdefault("method", (method or "").lower())
    return res

__all__ = [
    "HoraryInput", "QuestionType",
    "QuerentBirthData", "HybridPrasnaInput",
    "analyze_prasna_enhanced",
    "analyze_prasna", "analyze_parashari", "analyze_kp", "analyze_hybrid",
]
