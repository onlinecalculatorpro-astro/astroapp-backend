# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone
from enum import Enum
import math

# --- Import project cores ---
from app.core.timescales import build_timescales
from app.core.astronomy import compute_chart
from app.core.houses import compute_houses
# from app.core.validators import coerce_place_payload  # optional if you use it

# ============================== Constants & Helpers ==============================

DEG = math.degrees
RAD = math.radians

SIGN_NAMES = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]
SIGN_ABBR = ["Ar","Ta","Ge","Cn","Le","Vi","Li","Sc","Sg","Cp","Aq","Pi"]

PLANET_NAMES = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Rahu","Ketu"]

# Traditional sign lords by sign index (0=Aries..11=Pisces)
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

def sign_abbr_from_deg(deg: float) -> str:
    return SIGN_ABBR[sign_index(deg)]

def lord_of_sign(deg: float) -> str:
    return SIGN_LORDS[sign_index(deg)]

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

def angular_sep(a: float, b: float) -> float:
    """Smallest absolute angular separation in degrees [0..180]."""
    d = abs(deg_wrap(a) - deg_wrap(b))
    return d if d <= 180.0 else 360.0 - d

# ============================== KP definitions ==============================

KP_STAR_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
KP_DASHA_YEARS = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
STAR_LEN_DEG = 13.333333333333334  # 13°20′

def kp_star_and_sublord(ecl_deg_sidereal: float) -> Tuple[str, str, float, float]:
    """
    Return (star_lord, sub_lord, star_span_deg, pos_within_star_deg).
    Minimal KP partition: sub-lords in proportion to Vimshottari years.
    """
    pos = deg_wrap(ecl_deg_sidereal)
    star_idx = int(pos // STAR_LEN_DEG)  # 0..26
    star_lord = KP_STAR_ORDER[star_idx % 9]
    inside = pos - STAR_LEN_DEG * star_idx

    total_years = sum(KP_DASHA_YEARS[p] for p in KP_STAR_ORDER)  # 120
    acc = 0.0
    sub_lord = KP_STAR_ORDER[-1]
    sub_start = 0.0
    for lord in KP_STAR_ORDER:
        portion = STAR_LEN_DEG * (KP_DASHA_YEARS[lord] / total_years)
        if inside < acc + portion:
            sub_lord = lord
            sub_start = acc
            break
        acc += portion

    return (star_lord, sub_lord, STAR_LEN_DEG, inside - sub_start)

def kp_number_to_asc_deg(n: int) -> float:
    """Map KP horary number 1..249 to a zodiac degree (anchor method)."""
    n = max(1, min(249, int(n)))
    return (n - 1) * (360.0 / 249.0)

# ============================== Question taxonomy ==============================

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

# Richer house mapping
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

# ============================== Data structures ==============================

@dataclass
class PrasnaStrength:
    planet: str
    longitude: float
    sign: str
    house: int
    is_retrograde: bool
    dignity_score: float   # -2..+2
    house_strength: float  # 0..1
    overall_strength: float
    aspects_received: List[str] = field(default_factory=list)  # placeholder

@dataclass
class HoraryInput:
    date: Optional[str] = None     # "YYYY-MM-DD"
    time: Optional[str] = None     # "HH:MM:SS"
    tz_name: Optional[str] = None
    place: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    zodiac_mode: str = "sidereal"
    ayanamsa: str | float = "lahiri"
    ayanamsa_deg: Optional[float] = None
    house_system: str = "sripati"

    # KP options
    kp_house_system: str = "placidus"
    kp_ayanamsa: str | float = "krishnamurti"
    kp_number: Optional[int] = None
    kp_number_mode: str = "anchor_asc"  # "anchor_asc" | "advisory"

    # Question
    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None
    querent_house: int = 1
    quesited_house: Optional[int] = None

# ============================== Strength & dignity ==============================

def calculate_planetary_dignity(longitude: float, planet: str) -> float:
    """
    Simple dignity score (-2..+2). You can refine with ranges/linear falloff.
    """
    s = sign_index(longitude)
    exalt = {
        "Sun": (0, 10), "Moon": (1, 3), "Mars": (9, 28), "Mercury": (5, 15),
        "Jupiter": (3, 5), "Venus": (11, 27), "Saturn": (6, 20)
    }
    own = {
        "Sun": [4], "Moon": [3], "Mars": [0,7], "Mercury": [2,5],
        "Jupiter": [8,11], "Venus": [1,6], "Saturn": [9,10]
    }
    deb = {
        "Sun": (6, 10), "Moon": (7, 3), "Mars": (3, 28), "Mercury": (11, 15),
        "Jupiter": (9, 5), "Venus": (5, 27), "Saturn": (0, 20)
    }

    if planet in exalt and s == exalt[planet][0]:
        return 2.0
    if planet in own and s in own[planet]:
        return 1.0
    if planet in deb and s == deb[planet][0]:
        return -2.0
    return 0.0

def calculate_house_strength(house_num: int, question_type: Optional[QuestionType]) -> float:
    if not question_type or question_type not in ENHANCED_QUESTION_HOUSES:
        return 0.5
    m = ENHANCED_QUESTION_HOUSES[question_type]
    if house_num in m.get("primary", []): return 1.0
    if house_num in m.get("secondary", []): return 0.8
    if house_num in m.get("supportive", []): return 0.6
    if house_num in m.get("obstructive", []): return 0.2
    return 0.4

def analyze_planetary_strengths(chart: Dict[str, Any], houses: Dict[str, Any],
                                question_type: Optional[QuestionType]) -> List[PrasnaStrength]:
    out: List[PrasnaStrength] = []
    cusps = houses.get("cusps_deg", []) or []
    byname = {b["name"]: b for b in chart.get("bodies", [])}
    for nm, b in byname.items():
        if nm not in PLANET_NAMES:
            continue
        lon = float(b["longitude_deg"])
        h = house_of(lon, cusps)
        dig = calculate_planetary_dignity(lon, nm)
        hstr = calculate_house_strength(h, question_type)
        is_retro = bool(b.get("speed_deg_per_day", 0) < 0)
        overall = (dig + hstr + 2.0) / 4.0
        out.append(PrasnaStrength(
            planet=nm, longitude=lon, sign=sign_name_from_deg(lon), house=h,
            is_retrograde=is_retro, dignity_score=dig, house_strength=hstr,
            overall_strength=overall
        ))
    return out

# ============================== Radicality (Parāśari) ==============================

WEEK_LORDS = ["Monday:Moon","Tuesday:Mars","Wednesday:Mercury","Thursday:Jupiter","Friday:Venus","Saturday:Saturn","Sunday:Sun"]
WEEK_TO_LORD = ["Moon","Mars","Mercury","Jupiter","Venus","Saturn","Sun"]  # Python weekday(): Mon=0..Sun=6

def radicality_flags(chart: Dict[str, Any], local_dt: datetime) -> Dict[str, Any]:
    """Minimal radicality: alignment among ASC lord, day lord, hour lord (placeholder = day lord)."""
    asc_deg = chart["angles"]["ASC"]["deg"]
    asc_lord = lord_of_sign(asc_deg)
    daylord = WEEK_TO_LORD[local_dt.weekday()]
    hour_lord = daylord  # TODO: replace with true planetary hour if available
    fits = (asc_lord == daylord) or (asc_lord == hour_lord)
    return {"asc_lord": asc_lord, "day_lord": daylord, "hour_lord": hour_lord, "fits": fits}

# ============================== KP Signification Engine ==============================

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

# ============================== Parāśari Analysis ==============================

def enhanced_parashari_analysis(inp: HoraryInput) -> Dict[str, Any]:
    # Timescales (ensures tz coherence)
    if inp.date and inp.time and inp.tz_name:
        ts = build_timescales(inp.date, inp.time, inp.tz_name, dut1_seconds=0.0)
        local_dt = datetime.fromisoformat(f"{inp.date}T{inp.time}")
    else:
        now = datetime.now(timezone.utc)
        d = now.date().isoformat()
        t = now.time().replace(microsecond=0).isoformat()
        tz = inp.tz_name or "UTC"
        ts = build_timescales(d, t, tz, dut1_seconds=0.0)
        local_dt = now.astimezone(timezone.utc)

    # Houses / Chart
    houses = compute_houses(
        date=inp.date, time=inp.time, place=inp.place,
        latitude=inp.latitude, longitude=inp.longitude,
        house_system=inp.house_system, zodiac_mode=inp.zodiac_mode,
        ayanamsa=inp.ayanamsa
    )
    chart = compute_chart(
        date=inp.date, time=inp.time, place=inp.place,
        latitude=inp.latitude, longitude=inp.longitude,
        zodiac_mode=inp.zodiac_mode, ayanamsa=inp.ayanamsa,
        topocentric=True
    )

    cusps = houses.get("cusps_deg", []) or []
    asc_deg = chart["angles"]["ASC"]["deg"]
    moon_lon = next((b["longitude_deg"] for b in chart["bodies"] if b["name"]=="Moon"), None)

    qtype = inp.question_type or QuestionType.JOB
    strengths = analyze_planetary_strengths(chart, houses, qtype)
    moon_strength = next((s for s in strengths if s.planet == "Moon"), None)

    # House lords (by sign on cusps)
    house_lords: Dict[int, str] = {}
    for i, cdeg in enumerate(cusps, 1):
        house_lords[i] = lord_of_sign(cdeg)

    # Scoring
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    primary = mapping.get("primary", [])
    secondary = mapping.get("secondary", [])
    supportive = mapping.get("supportive", [])
    obstructive = mapping.get("obstructive", [])

    primary_score = secondary_score = obstruction_score = 0.0
    for s in strengths:
        if s.house in primary:    primary_score   += s.overall_strength * 3.0
        elif s.house in secondary: secondary_score += s.overall_strength * 2.0
        elif s.house in supportive: secondary_score += s.overall_strength * 1.0
        elif s.house in obstructive: obstruction_score += s.overall_strength * 2.0

    # Moon proximity bonus: closeness to lords of target houses
    prox_bonus = 0.0
    if moon_lon is not None:
        target_lords = {house_lords.get(h) for h in (primary + secondary)}
        bodies = {b["name"]: b for b in chart.get("bodies", [])}
        for ln in filter(None, [bodies.get(pl) and bodies[pl]["longitude_deg"] for pl in target_lords if pl]):
            if angular_sep(moon_lon, ln) < 12.0:
                prox_bonus += 1.0

    # Radicality multiplier (light weight)
    rad = radicality_flags(chart, local_dt)
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
        "meta": {
            "mode": "enhanced_parashari",
            "question_type": qtype.value,
            "zodiac_mode": inp.zodiac_mode,
            "ayanamsa": inp.ayanamsa,
            "house_system": inp.house_system,
            "analysis_time": datetime.now(timezone.utc).isoformat()
        },
        "prasna_meta": [
            {"Key":"ASC Lord", "Value": lord_of_sign(asc_deg)},
            {"Key":"Day Lord", "Value": rad["day_lord"]},
            {"Key":"Hour Lord (approx)", "Value": rad["hour_lord"]},
            {"Key":"Radicality Fits", "Value": rad["fits"]},
        ],
        "chart_data": {
            "ASC_deg": asc_deg,
            "ASC_sign": sign_name_from_deg(asc_deg),
            "Moon_deg": moon_lon,
            "Moon_sign": sign_name_from_deg(moon_lon) if moon_lon is not None else None,
            "cusps_deg": cusps,
            "house_lords": house_lords
        },
        "planetary_analysis": [
            {
                "planet": s.planet,
                "longitude_deg": round(s.longitude, 4),
                "sign": s.sign,
                "house": s.house,
                "is_retrograde": s.is_retrograde,
                "dignity_score": s.dignity_score,
                "house_strength": s.house_strength,
                "overall_strength": round(s.overall_strength, 4)
            } for s in strengths
        ],
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

# ============================== KP Analysis ==============================

def enhanced_kp_analysis(inp: HoraryInput) -> Dict[str, Any]:
    kp_aya = inp.kp_ayanamsa if inp.kp_ayanamsa is not None else "krishnamurti"

    # Timescales
    if inp.date and inp.time and inp.tz_name:
        ts = build_timescales(inp.date, inp.time, inp.tz_name, dut1_seconds=0.0)
    else:
        now = datetime.now(timezone.utc)
        d = now.date().isoformat()
        t = now.time().replace(microsecond=0).isoformat()
        tz = inp.tz_name or "UTC"
        ts = build_timescales(d, t, tz, dut1_seconds=0.0)

    # Houses / Chart (sidereal, KP ayanamsa)
    houses = compute_houses(
        date=inp.date, time=inp.time, place=inp.place,
        latitude=inp.latitude, longitude=inp.longitude,
        house_system=inp.kp_house_system, zodiac_mode="sidereal",
        ayanamsa=kp_aya
    )
    chart = compute_chart(
        date=inp.date, time=inp.time, place=inp.place,
        latitude=inp.latitude, longitude=inp.longitude,
        zodiac_mode="sidereal", ayanamsa=kp_aya,
        topocentric=True
    )

    cusps = houses.get("cusps_deg", []) or []
    asc_deg_sid = chart["angles"]["ASC"]["deg"]
    if inp.kp_number and inp.kp_number_mode == "anchor_asc":
        asc_deg_sid = kp_number_to_asc_deg(inp.kp_number)

    # Cusp sub-lords table
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

    # Planetary KP table (planet → star/sub)
    planetary_kp = []
    for b in chart.get("bodies", []):
        nm = b["name"]
        if nm not in PLANET_NAMES:
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

    # Decision via sub-lord signification on primary houses
    qtype = inp.question_type or QuestionType.JOB
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
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

    return {
        "ok": True,
        "meta": {
            "mode": "enhanced_kp",
            "question_type": qtype.value,
            "ayanamsa": kp_aya,
            "house_system": inp.kp_house_system,
            "kp_number": inp.kp_number,
            "kp_number_mode": inp.kp_number_mode,
            "analysis_time": datetime.now(timezone.utc).isoformat()
        },
        "kp_core": {
            "ASC_deg_sid": asc_deg_sid,
            "ASC_sign": sign_name_from_deg(asc_deg_sid),
        },
        "cusp_analysis": detailed_cusps,
        "planetary_kp": planetary_kp,
        "signification_evidence": evidence,
        "judgement": {"answer": answer, "confidence": conf, "method": "KP sub-lord signification"}
    }

# ============================== Public interface ==============================

def analyze_prasna_enhanced(inp: HoraryInput, method: str = "parashari") -> Dict[str, Any]:
    """
    Entrypoint:
      method="parashari" → enhanced_parashari_analysis
      method="kp"        → enhanced_kp_analysis
    """
    try:
        if method.lower() == "kp":
            return enhanced_kp_analysis(inp)
        return enhanced_parashari_analysis(inp)
    except Exception as e:
        return {"ok": False, "error": str(e), "meta": {"mode": method, "error_type": type(e).__name__}}

__all__ = [
    "HoraryInput", "QuestionType",
    "analyze_prasna_enhanced",
    "enhanced_parashari_analysis", "enhanced_kp_analysis",
]
