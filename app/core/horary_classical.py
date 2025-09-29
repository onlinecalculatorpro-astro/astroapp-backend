# -*- coding: utf-8 -*-
"""
horary_classical.py
-------------------
Classical Horary / Prashna systems: Parāśarī and KP (no Hybrid here).

Key points
----------
- Uses app.core.astronomy.compute_chart(...) for planets/angles + timescales.
- Uses app.core.houses_advanced.compute_house_system(...) (strict JD inputs).
- If zodiac_mode == "sidereal", house cusps are shifted by ayanāṃśa so tests
  (occupancy, sign lords, KP sub-lords) stay internally consistent.
- Public entrypoints:
    - analyze_parashari(inp: HoraryInput) -> dict
    - analyze_kp(inp: HoraryInput) -> dict
    - analyze_prasna_enhanced(inp, method="parashari") -> dict  # route wrapper
    - analyze_prasna(inp, method="parashari"|"kp") -> dict

Exports
-------
- HoraryInput, QuestionType, ENHANCED_QUESTION_HOUSES
- analyze_prasna_enhanced, analyze_prasna
- analyze_parashari, analyze_kp
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
    ayanamsa_deg: Optional[float] = None   # accept explicit ayanamsa degrees if provided
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
# Low-level: charts + houses (with robust fallback)
# =============================================================================

def _ensure_coords_and_tz(
    date: Optional[str],
    time_: Optional[str],
    tz: Optional[str],
    place: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> Tuple[str, str, str, float, float]:
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

def _chart(
    date: Optional[str],
    time_: Optional[str],
    tz: Optional[str],
    place: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    *,
    zodiac_mode: str,
    ayanamsa: str | float,
    topocentric: bool = True
) -> Dict[str, Any]:
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
    ayanamsa_deg: Optional[float],
) -> Dict[str, Any]:
    """
    Try advanced house engine with multiple signatures; if it fails,
    fall back to Equal Houses from ASC (so analysis still works).
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

    ts = (chart.get("meta", {}) or {}).get("timescales", {}) or {}
    jd_ut  = _pick_ts(ts, "jd_ut", "jd_utc")
    jd_tt  = _pick_ts(ts, "jd_tt", "tt_jd", "jd_tdb")
    jd_ut1 = _pick_ts(ts, "jd_ut1", "ut1_jd")

    def _try_engine() -> Optional[Dict[str, Any]]:
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
        for kwargs in trials:
            if "jd_tt" in kwargs and kwargs["jd_tt"] is None:
                continue
            if "jd_ut1" in kwargs and kwargs["jd_ut1"] is None:
                continue
            if "jd_ut" in kwargs and kwargs["jd_ut"] is None:
                continue
            try:
                return compute_house_system(**kwargs)  # type: ignore[arg-type]
            except TypeError:
                continue
            except Exception:
                continue
        return None

    payload = _try_engine()

    # Fallback: Equal Houses from ASC (never crash analysis)
    if not payload:
        ang = chart.get("angles") or {}
        asc_any = ang.get("asc_deg", chart.get("asc_deg"))
        if asc_any is None:
            raise ValueError("houses_fallback_failed:no_asc_in_chart")

        asc_deg = float(asc_any)
        cusps = [deg_wrap(asc_deg + i * 30.0) for i in range(12)]
        mc_guess = (ang.get("mc_deg") if ang.get("mc_deg") is not None else deg_wrap(asc_deg + 90.0))

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

    # Advanced payload OK — apply sidereal shift if needed
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
# Parāśarī system
# =============================================================================

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
    ay_from_chart = meta.get("ayanamsa_deg")
    # Allow explicit override if client provided ayanamsa_deg
    ay_deg = float(inp.ayanamsa_deg) if isinstance(inp.ayanamsa_deg, (int, float)) else ay_from_chart

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
            "ayanamsa_deg_used": ay_deg,
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
# Public dispatcher (classical only)
# =============================================================================

def analyze_prasna(inp: HoraryInput, method: str = "parashari") -> Dict[str, Any]:
    """
    method in {"parashari","kp"}.
    (Hybrid is intentionally NOT implemented in this classical module.)
    """
    try:
        m = (method or "parashari").strip().lower()
        if m == "kp":
            return analyze_kp(inp)
        # default: parashari
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
    "HoraryInput", "QuestionType", "ENHANCED_QUESTION_HOUSES",
    "analyze_prasna_enhanced",
    "analyze_prasna", "analyze_parashari", "analyze_kp",
]
