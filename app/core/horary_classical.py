# -*- coding: utf-8 -*-
"""
horary_classical.py — REWRITE (2025-09-29)
-----------------------------------------
Classical Horary / Prashna systems: Parāśarī and KP (no Hybrid here).

Highlights
----------
- Sidereal-correct houses: If zodiac_mode == "sidereal", cusps/angles shift by
  ayanāṃśa so sign lords, KP partitions, and house assignments are consistent.
- Whole-Sign support: `house_system="whole_sign"` yields house = sign from Lagna.
- Robust houses: advanced `houses_advanced.compute_house_system(...)` tried with
  multiple JD signatures; fallback to Equal-from-ASC (never crash analysis).
- Parāśarī upgrades:
    • Rich dignity (own/exalt/debil + mūlatrikoṇa, friend/enemy, combustion,
      retrograde, sandhi) → normalized score.
    • Vedic graha-dṛṣṭi (special aspects of Mars/Jupiter/Saturn, everyone 7th)
      contributes malefic/benefic pressure on target houses.
    • Moon proximity bonus to lords of target houses; “radicality” (day-lord).
    • Optional pañcāṅga snapshot (tithi index & Moon nakṣatra star).
- KP upgrades:
    • Nakṣatra star → sub → sub-sub (SSL) partition.
    • Significator chain: planet ⇒ star-lord ⇒ sign-lord (+ nodes as agents).
    • Cusp evaluation by sub-lord (primary houses) with SSL tie-break heuristic.
    • KP number anchoring *rotates cusps* (not just ASC label).
    • Ruling Planets (RP) used as a tie-breaker to bias close calls.

Public entrypoints (unchanged)
------------------------------
- analyze_parashari(inp: HoraryInput) -> dict
- analyze_kp(inp: HoraryInput) -> dict
- analyze_prasna_enhanced(inp, method="parashari") -> dict
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

# --- Core engines (in your project) ---
from app.core.astronomy import compute_chart, resolve_place
from app.core.houses_advanced import compute_house_system  # strict JD(TT)+UT1

# =============================================================================
# Constants & helpers
# =============================================================================

SIGN_NAMES = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]
PARASHARI_PLANETS = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"]
PLANETS_WITH_NODES = PARASHARI_PLANETS + ["Rahu","Ketu"]

# Sign lords by sign index (0=Aries..11=Pisces)
SIGN_LORDS = {
    0: "Mars", 1: "Venus", 2: "Mercury", 3: "Moon", 4: "Sun", 5: "Mercury",
    6: "Venus", 7: "Mars", 8: "Jupiter", 9: "Saturn", 10: "Saturn", 11: "Jupiter"
}

# Mūlatrikoṇa signs (sign-level bonus)
MOOLATRIKONA = {
    "Sun": 4,       # Leo (we use sign-level bonus)
    "Moon": None,   # Variable/rarely used sign-level bonus in horary
    "Mars": 0,      # Aries
    "Mercury": 5,   # Virgo
    "Jupiter": 8,   # Sagittarius
    "Venus": 1,     # Taurus
    "Saturn": 10,   # Aquarius
}

# Permanent friendship (simplified standard)
FRIENDS = {
    "Sun": {"Moon","Mars","Jupiter"},
    "Moon": {"Sun","Mercury"},
    "Mars": {"Sun","Moon","Jupiter"},
    "Mercury": {"Sun","Venus"},
    "Jupiter": {"Sun","Moon","Mars"},
    "Venus": {"Mercury","Saturn"},
    "Saturn": {"Mercury","Venus"},
}
ENEMIES = {
    "Sun": {"Venus","Saturn"},
    "Moon": set(),
    "Mars": {"Mercury"},
    "Mercury": {"Moon"},
    "Jupiter": {"Venus","Mercury"},
    "Venus": {"Sun","Moon"},
    "Saturn": {"Sun","Moon"},
}

# Combustion thresholds (deg from Sun) — conservative defaults
COMBUST_DEG = {"Moon": 12.0,"Mercury": 12.0,"Venus": 10.0,"Mars": 17.0,"Jupiter": 11.0,"Saturn": 15.0}

STAR_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
KP_DASHA_YEARS = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
STAR_LEN_DEG = 360.0 / 27.0
TOTAL_DASHA = 120.0

BENEFICS = {"Jupiter","Venus","Moon"}   # simple model; Mercury often treated neutral/benefic
MALEFICS = {"Saturn","Mars","Sun","Rahu","Ketu"}  # Sun mild malefic; nodes malefic in horary

# Panchanga labels (simple tithi indexing)
TITHI_NAMES = [
    "Pratipada","Dvitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Purnima/Amavasya",
]*2  # 30 entries

# =============================================================================
# Math & sign helpers
# =============================================================================

def deg_wrap(x: float) -> float:
    return (x % 360.0 + 360.0) % 360.0

def sign_index(deg: float) -> int:
    return int(deg_wrap(deg) // 30)

def sign_name_from_deg(deg: float | None) -> Optional[str]:
    if deg is None:
        return None
    return SIGN_NAMES[sign_index(float(deg))]

def lord_of_sign(deg: float) -> str:
    return SIGN_LORDS[sign_index(deg)]

def angular_sep(a: float, b: float) -> float:
    d = abs(deg_wrap(a) - deg_wrap(b))
    return d if d <= 180.0 else 360.0 - d

def is_sandhi(deg: float, window: float = 1.0) -> bool:
    """Near sign boundary within `window` degrees."""
    d = deg_wrap(deg) % 30.0
    return (d < window) or (30.0 - d < window)

# =============================================================================
# Nakṣatra star/sub/subsub partition — KP
# =============================================================================

def kp_star_sub_sub(ecl_deg_sidereal: float) -> Tuple[str, str, str, float, float, float, float]:
    """
    Return (star_lord, sub_lord, sub_sub_lord, star_span_deg, pos_in_star_deg,
            sub_span_deg, pos_in_sub_deg). Proportions follow Vimśottari years.
    """
    pos = deg_wrap(ecl_deg_sidereal)
    star_idx = int(pos // STAR_LEN_DEG)  # 0..26
    star_lord = STAR_ORDER[star_idx % 9]
    pos_in_star = pos - STAR_LEN_DEG * star_idx

    # Sub level
    cycle = STAR_ORDER[STAR_ORDER.index(star_lord):] + STAR_ORDER[:STAR_ORDER.index(star_lord)]
    acc = 0.0
    sub_lord = STAR_ORDER[-1]
    sub_start = 0.0
    sub_span = STAR_LEN_DEG  # default
    for lord in cycle:
        portion = STAR_LEN_DEG * (KP_DASHA_YEARS[lord] / TOTAL_DASHA)
        if pos_in_star < acc + portion:
            sub_lord = lord
            sub_start = acc
            sub_span = portion
            break
        acc += portion
    pos_in_sub = pos_in_star - sub_start

    # Sub-sub level: cycle from sub_lord
    cycle2 = STAR_ORDER[STAR_ORDER.index(sub_lord):] + STAR_ORDER[:STAR_ORDER.index(sub_lord)]
    acc2 = 0.0
    ssl = STAR_ORDER[-1]
    for lord in cycle2:
        portion2 = sub_span * (KP_DASHA_YEARS[lord] / TOTAL_DASHA)
        if pos_in_sub < acc2 + portion2:
            ssl = lord
            break
        acc2 += portion2

    return (star_lord, sub_lord, ssl, STAR_LEN_DEG, pos_in_star, sub_span, pos_in_sub)

# =============================================================================
# House utilities
# =============================================================================

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
        inside = ((a <= x) and (x < b)) if a <= b else ((x >= a) or (x < b))
        if inside:
            return i + 1
    return 1

def shift_sidereal(values: List[float], ay_deg: float) -> List[float]:
    return [deg_wrap(v - ay_deg) for v in values]

def whole_sign_cusps_from_asc(asc_deg: float) -> List[float]:
    """House 1 cusp = 0° of ascendant's sign, then every 30°."""
    first = sign_index(asc_deg) * 30.0
    return [deg_wrap(first + 30.0 * i) for i in range(12)]

# =============================================================================
# Vedic drishti (sign-based)
# =============================================================================

def graha_drishti_offsets(planet: str, include_nodes_special: bool = True) -> Set[int]:
    """
    Returns house offsets (1..12 relative to planet's house) that receive aspect.
    Everyone: 7th; Mars: 4th & 8th; Jupiter: 5th & 9th; Saturn: 3rd & 10th;
    Nodes optional: 5th & 9th in some traditions.
    """
    base = {7}
    if planet == "Mars":
        base |= {4, 8}
    elif planet == "Jupiter":
        base |= {5, 9}
    elif planet == "Saturn":
        base |= {3, 10}
    elif include_nodes_special and planet in {"Rahu","Ketu"}:
        base |= {5, 9}
    return base

def houses_aspected_by(planet_name: str, cusps: List[float], bodies: Dict[str, Any]) -> Set[int]:
    """
    Compute sign-based drishti targets for a planet using cusps (house = sign zone).
    """
    p = bodies.get(planet_name)
    if not p:
        return set()
    lon = float(p.get("longitude_deg"))
    h = house_of(lon, cusps)
    targets: Set[int] = set()
    for off in graha_drishti_offsets(planet_name):
        t = ((h - 1 + (off - 1)) % 12) + 1
        targets.add(t)
    return targets

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
    ayanamsa_deg: Optional[float] = None   # explicit override (deg)
    house_system: str = "sripati"          # accepts "whole_sign"
    # KP options
    kp_house_system: str = "placidus"
    kp_ayanamsa: str | float = "krishnamurti"
    kp_number: Optional[int] = None
    kp_number_mode: str = "anchor_asc"     # "anchor_asc" | "advisory"
    # question
    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None
    querent_house: int = 1
    quesited_house: Optional[int] = None

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
    fall back to Equal Houses from ASC. If house_system == "whole_sign",
    build whole-sign cusps from ASC.
    """
    ang = chart.get("angles") or {}
    asc_any = ang.get("asc_deg", chart.get("asc_deg"))
    mc_any  = ang.get("mc_deg", chart.get("mc_deg"))
    ts = (chart.get("meta", {}) or {}).get("timescales", {}) or {}

    jd_ut  = _pick_ts(ts, "jd_ut", "jd_utc")
    jd_tt  = _pick_ts(ts, "jd_tt", "tt_jd", "jd_tdb")
    jd_ut1 = _pick_ts(ts, "jd_ut1", "ut1_jd")

    # Whole-Sign shortcut (ASC required)
    if (house_system or "").lower() == "whole_sign" and asc_any is not None:
        asc_deg = float(asc_any)
        cusps = whole_sign_cusps_from_asc(asc_deg)
        if zodiac_mode.lower() == "sidereal" and isinstance(ayanamsa_deg, (int, float)):
            cusps = shift_sidereal(cusps, float(ayanamsa_deg))
            asc_out = deg_wrap(asc_deg - float(ayanamsa_deg))
            mc_out  = deg_wrap(float(mc_any) - float(ayanamsa_deg)) if mc_any is not None else deg_wrap(asc_out + 90.0)
        else:
            asc_out = asc_deg
            mc_out  = float(mc_any) if mc_any is not None else deg_wrap(asc_deg + 90.0)
        return {
            "house_system": "whole_sign",
            "cusps_deg": cusps,
            "asc_deg": asc_out,
            "mc_deg": mc_out,
            "vertex": None,
            "eastpoint": None,
            "warnings": [],
        }

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
            # Skip invalid None combos
            if "jd_tt" in kwargs and kwargs["jd_tt"] is None: continue
            if "jd_ut1" in kwargs and kwargs["jd_ut1"] is None: continue
            if "jd_ut" in kwargs and kwargs["jd_ut"] is None: continue
            try:
                return compute_house_system(**kwargs)  # type: ignore[arg-type]
            except TypeError:
                continue
            except Exception:
                continue
        return None

    payload = _try_engine()

    # Fallback: Equal Houses from ASC
    if not payload:
        if asc_any is None:
            raise ValueError("houses_fallback_failed:no_asc_in_chart")
        asc_deg = float(asc_any)
        cusps = [deg_wrap(asc_deg + i * 30.0) for i in range(12)]
        mc_guess = float(mc_any) if mc_any is not None else deg_wrap(asc_deg + 90.0)
        if zodiac_mode.lower() == "sidereal" and isinstance(ayanamsa_deg, (int, float)):
            cusps = shift_sidereal(cusps, float(ayanamsa_deg))
            asc_out = deg_wrap(asc_deg - float(ayanamsa_deg))
            mc_out  = deg_wrap(mc_guess - float(ayanamsa_deg))
        else:
            asc_out, mc_out = asc_deg, mc_guess
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
# Panchanga mini: tithi & Moon nakshatra
# =============================================================================

def _tithi_index(moon_deg: Optional[float], sun_deg: Optional[float]) -> Optional[int]:
    if moon_deg is None or sun_deg is None:
        return None
    el = deg_wrap(moon_deg - sun_deg)  # Moon - Sun elongation
    return int(el // 12.0)  # 0..29

def _moon_star(chart: Dict[str, Any]) -> Optional[str]:
    moon = next((b for b in chart.get("bodies", []) if b.get("name") == "Moon"), None)
    if not moon: return None
    star, *_ = kp_star_sub_sub(float(moon["longitude_deg"]))
    return star

# =============================================================================
# Parāśarī system (enhanced)
# =============================================================================

def _calculate_planetary_dignity(long_deg: float, planet: str) -> float:
    """
    Rich dignity score. Rough scale ~ [-2.5 .. +2.5] (before normalization):
      +2.0 exaltation (sign-level)
      +1.0 own sign
      +0.75 moolatrikona (sign-level)
      +0.5 friend sign / -0.5 enemy sign
      -2.0 debilitation
      -0.3 combustion penalty (if combust)
      -0.2 sandhi penalty (near sign boundary)
      +/-0.25 retrograde (malefics +0.25, benefics -0.25)
    """
    sidx = sign_index(long_deg)
    dign = 0.0

    # Exaltation/debilitation (sign-level, simplified canonical pairs)
    exalt_sign = {
        "Sun": 0, "Moon": 2, "Mars": 9, "Mercury": 5,
        "Jupiter": 3, "Venus": 11, "Saturn": 6
    }
    if planet in exalt_sign and sidx == exalt_sign[planet]:
        dign += 2.0
    elif planet in exalt_sign and sidx == (exalt_sign[planet] + 6) % 12:
        dign -= 2.0

    # Own signs
    own = {"Sun":[4], "Moon":[3], "Mars":[0,7], "Mercury":[2,5],
           "Jupiter":[8,11], "Venus":[1,6], "Saturn":[9,10]}
    if planet in own and sidx in own[planet]:
        dign += 1.0

    # Moolatrikona
    if MOOLATRIKONA.get(planet) is not None and sidx == MOOLATRIKONA[planet]:
        dign += 0.75

    # Friend/enemy sign
    lord = SIGN_LORDS[sidx]
    if planet in FRIENDS and lord in FRIENDS[planet]:
        dign += 0.5
    if planet in ENEMIES and lord in ENEMIES[planet]:
        dign -= 0.5

    # Sandhi penalty
    if is_sandhi(long_deg, 1.0):
        dign -= 0.2

    # Combustion & retro handled outside (need Sun distance & speed)
    return dign

def _parashari_strengths(chart: Dict[str, Any], cusps: List[float],
                         question_type: Optional[QuestionType]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    bodies = {b["name"]: b for b in chart.get("bodies", [])}
    # include nodes as malefics for visibility in table
    planet_order = [p for p in PLANETS_WITH_NODES if p in bodies]

    # Sun needed for combustion distances
    sun_lon: Optional[float] = None
    if "Sun" in bodies:
        sun_lon = float(bodies["Sun"].get("longitude_deg"))

    def _house_relevance(h: int) -> float:
        if not question_type or question_type not in ENHANCED_QUESTION_HOUSES:
            return 0.5
        m = ENHANCED_QUESTION_HOUSES[question_type]
        if h in m.get("primary", []):    return 1.0
        if h in m.get("secondary", []):  return 0.8
        if h in m.get("supportive", []): return 0.6
        if h in m.get("obstructive", []):return 0.2
        return 0.4

    for nm in planet_order:
        b = bodies[nm]
        lon = float(b["longitude_deg"])
        spd = float(b.get("speed_deg_per_day") or b.get("speed") or 0.0)
        h = house_of(lon, cusps)

        # Base dignity
        dig = _calculate_planetary_dignity(lon, nm)

        # Combustion penalty (skip for Sun)
        if nm != "Sun" and sun_lon is not None and nm in COMBUST_DEG:
            if angular_sep(lon, sun_lon) <= COMBUST_DEG[nm]:
                dig -= 0.3

        # Retrograde adjustment
        if spd < 0:
            if nm in MALEFICS:
                dig += 0.25
            elif nm in BENEFICS:
                dig -= 0.25

        hstr = _house_relevance(h)
        is_retro = spd < 0
        overall = (dig + hstr + 2.5) / 5.0  # normalize rough range to ~0..1

        out.append({
            "planet": nm,
            "longitude_deg": round(lon, 4),
            "sign": sign_name_from_deg(lon),
            "house": h,
            "is_retrograde": is_retro,
            "dignity_score": round(dig, 3),
            "house_strength": round(hstr, 3),
            "overall_strength": round(max(0.0, min(1.0, overall)), 4)
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
    sun_lon  = next((b["longitude_deg"] for b in chart.get("bodies", []) if b["name"]=="Sun"), None)

    qtype = inp.question_type or QuestionType.JOB
    strengths = _parashari_strengths(chart, cusps, qtype)

    # House lords by cusp sign
    house_lords: Dict[int, str] = {i+1: lord_of_sign(c) for i, c in enumerate(cusps)}

    # Drishti pressure on target houses
    bodies_by_name = {b["name"]: b for b in chart.get("bodies", [])}
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    target_pos = set(mapping.get("primary", []) + mapping.get("secondary", []) + mapping.get("supportive", []))
    target_neg = set(mapping.get("obstructive", []))

    drishti_bonus = drishti_malus = 0.0
    for nm in bodies_by_name:
        if nm not in PLANETS_WITH_NODES:  # use 7 + nodes
            continue
        tgt_houses = houses_aspected_by(nm, cusps, bodies_by_name)
        hits_pos = len(tgt_houses & target_pos)
        hits_neg = len(tgt_houses & target_neg)
        if nm in BENEFICS:
            drishti_bonus += 0.15 * hits_pos
            drishti_malus += 0.10 * hits_neg  # benefic hitting obstructive may amplify those topics mildly
        else:
            drishti_malus += 0.20 * hits_pos
            drishti_bonus += 0.05 * hits_neg  # malefic hitting obstructive can cancel it slightly

    # Scoring buckets from planetary strengths
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
        for pl in filter(None, target_lords):
            if pl in bodies_by_name:
                if angular_sep(float(moon_lon), float(bodies_by_name[pl]["longitude_deg"])) < 12.0:
                    prox_bonus += 1.0

    # Radicality multiplier
    date = inp.date or datetime.now(timezone.utc).date().isoformat()
    time_ = inp.time or datetime.now(timezone.utc).time().replace(microsecond=0).isoformat()
    rad = _radicality_flags(chart, inp.tz_name or "UTC", date, time_)
    rad_mult = 1.10 if rad["fits"] else 1.0

    # Panchanga snapshot
    tithi_idx = _tithi_index(moon_lon, sun_lon)
    moon_star = _moon_star(chart)

    # Final scores
    total_positive = (primary_score + secondary_score + prox_bonus + drishti_bonus) * rad_mult
    total_negative = obstruction_score + drishti_malus
    net_score = total_positive - total_negative

    if net_score > 1.25:
        answer, conf = "yes", min(0.96, 0.74 + 0.06 * net_score)
    elif net_score < -1.25:
        answer, conf = "no",  min(0.96, 0.74 + 0.06 * abs(net_score))
    else:
        answer, conf = "uncertain", 0.62

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
        "panchanga": {
            "tithi_index": tithi_idx,
            "tithi_label": (TITHI_NAMES[tithi_idx] if tithi_idx is not None else None),
            "moon_star": moon_star
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
            "drishti_bonus": round(drishti_bonus, 3),
            "drishti_malus": round(drishti_malus, 3),
            "radicality_multiplier": rad_mult,
            "net_score": round(net_score, 3)
        },
        "judgement": {"answer": answer, "confidence": round(conf, 3)}
    }

# =============================================================================
# KP system (enhanced)
# =============================================================================

def _kp_signified_houses_base(planet_name: str, chart: Dict[str, Any], cusps: List[float]) -> Set[int]:
    """Occupancy + lordship of cusp signs for the given planet."""
    houses: Set[int] = set()
    bodies = {b["name"]: b for b in chart.get("bodies", [])}
    p = bodies.get(planet_name)
    if not p:
        return houses

    # Occupancy of planet
    houses.add(house_of(float(p["longitude_deg"]), cusps))

    # Lordship of cusp signs
    for i, cusp in enumerate(cusps, 1):
        if lord_of_sign(cusp) == planet_name:
            houses.add(i)
    return houses

def _kp_signified_houses_chain(planet_name: str, chart: Dict[str, Any], cusps: List[float]) -> Set[int]:
    """
    KP chain: planet ⇒ star-lord ⇒ sign-lord.
    Nodes act as agents of their star-/sign-lords and the planets they conjoin (±3°) or oppose (7th).
    """
    bodies = {b["name"]: b for b in chart.get("bodies", [])}
    p = bodies.get(planet_name)
    if not p:
        return set()

    lon = float(p["longitude_deg"])
    star_lord, sub_lord, ssl, *_ = kp_star_sub_sub(lon)
    signlord = lord_of_sign(lon)

    sig: Set[int] = set()
    for who in {planet_name, star_lord, signlord}:
        sig |= _kp_signified_houses_base(who, chart, cusps)

    # Node agent rule: if planet itself is Rahu/Ketu, expand to its star-lord/sign-lord
    if planet_name in {"Rahu","Ketu"}:
        sig |= _kp_signified_houses_base(star_lord, chart, cusps)
        sig |= _kp_signified_houses_base(signlord, chart, cusps)

        # Conjunction (±3°) & opposition (7th sign)
        for other in bodies:
            if other == planet_name: continue
            olon = float(bodies[other]["longitude_deg"])
            if angular_sep(lon, olon) <= 3.0 or abs(sign_index(lon) - sign_index(olon)) in {6, 6 % 12}:
                sig |= _kp_signified_houses_base(other, chart, cusps)

    # SSL tie-break: mark presence (used later, not added here to avoid overreach)
    return sig

def _kp_cusp_sub_lord_eval(cusp_house: int, chart: Dict[str, Any], cusps: List[float],
                           positive: Set[int], negative: Set[int]) -> Tuple[bool, Set[int], str, Optional[str]]:
    """
    Evaluate a cusp by its sub-lord; if borderline, peek at SSL for tie-break.
    """
    cusp_deg = cusps[cusp_house-1]
    star_lord, sub_lord, ssl, star_deg, sub_pos, sub_span, pos_in_sub = kp_star_sub_sub(cusp_deg)
    sig = _kp_signified_houses_chain(sub_lord, chart, cusps)
    good = (len(sig & positive) > 0) and (len(sig & negative) == 0)

    # SSL tie-breaker: if not good, but SSL chain is clean positive, flip weakly
    ssl_sig = _kp_signified_houses_chain(ssl, chart, cusps)
    ssl_good = (len(ssl_sig & positive) > 0) and (len(ssl_sig & negative) == 0)
    reason = f"sub={sub_lord}, ssl={ssl}, signified={sorted(sig)}, ssl_signified={sorted(ssl_sig)}"
    chosen_ssl = ssl if (not good and ssl_good) else None
    return good or ssl_good, (sig if good else (ssl_sig if ssl_good else sig)), reason, chosen_ssl

def _ruling_planets(chart: Dict[str, Any], cusps: List[float], tz_name: str, date: str, time_: str) -> Set[str]:
    """KP Ruling Planets: day-lord, Moon sign-lord, Moon star-lord, ASC sign-lord, ASC star-lord."""
    bodies = {b["name"]: b for b in chart.get("bodies", [])}
    # Day-lord
    try:
        from zoneinfo import ZoneInfo
        dt_local = datetime.fromisoformat(f"{date}T{time_}").replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        dt_local = datetime.fromisoformat(f"{date}T{time_}").replace(tzinfo=timezone.utc)
    WEEK_TO_LORD = ["Moon","Mars","Mercury","Jupiter","Venus","Saturn","Sun"]
    daylord = WEEK_TO_LORD[dt_local.weekday()]

    # Moon sign/star
    moon = bodies.get("Moon")
    moon_lord = moon_star = None
    if moon:
        md = float(moon["longitude_deg"])
        moon_lord = lord_of_sign(md)
        moon_star, *_ = kp_star_sub_sub(md)

    # ASC sign/star
    asc_deg = chart.get("angles", {}).get("asc_deg", chart.get("asc_deg")) or 0.0
    asc_lord = lord_of_sign(float(asc_deg))
    asc_star, *_ = kp_star_sub_sub(float(asc_deg))

    return {p for p in [daylord, moon_lord, moon_star, asc_lord, asc_star] if p}

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
    cusps = list(houses.get("cusps_deg", []) or [])

    # KP number anchoring: rotate cusps so ASC equals number-derived degree
    asc_deg_sid = float(houses.get("asc_deg"))
    if inp.kp_number and (inp.kp_number_mode or "anchor_asc").lower() == "anchor_asc":
        target = (max(1, min(249, int(inp.kp_number))) - 1) * (360.0 / 249.0)
        delta = deg_wrap(target - asc_deg_sid)
        cusps = [deg_wrap(c + delta) for c in cusps]
        asc_deg_sid = target

    # Decision via cusp sub-lords on primary houses
    qtype = inp.question_type or QuestionType.JOB
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    pos_h = set(mapping.get("primary", []) + mapping.get("secondary", []) + mapping.get("supportive", []))
    neg_h = set(mapping.get("obstructive", []))

    yes_hits = no_hits = 0
    evidence = []
    for h in mapping.get("primary", []):
        ok, sig, reason, ssl_used = _kp_cusp_sub_lord_eval(h, chart, cusps, pos_h, neg_h)
        evidence.append({"cusp": h, "ok": ok, "reason": reason, "ssl_used": ssl_used})
        yes_hits += int(ok)
        no_hits += int(not ok)

    # Ruling Planets bias
    date = inp.date or datetime.now(timezone.utc).date().isoformat()
    time_ = inp.time or datetime.now(timezone.utc).time().replace(microsecond=0).isoformat()
    rp = _ruling_planets(chart, cusps, inp.tz_name or "UTC", date, time_)
    rp_support = 0
    # Count RP that also signify positive houses via chains
    for p in rp:
        sig = _kp_signified_houses_chain(p, chart, cusps)
        if len(sig & pos_h) > 0 and len(sig & neg_h) == 0:
            rp_support += 1

    # Verdict
    if yes_hits > no_hits:
        base_conf = 0.70 + 0.05*(yes_hits - no_hits)
        base_ans = "yes"
    elif no_hits > yes_hits:
        base_conf = 0.70 + 0.05*(no_hits - yes_hits)
        base_ans = "no"
    else:
        base_conf = 0.62
        base_ans = "uncertain"

    # RP bias: nudge confidence up to +0.06
    conf = min(0.97, base_conf + min(0.06, 0.02 * rp_support))

    # Cusp table with star/sub/ssl
    detailed_cusps = []
    for i, cusp_deg in enumerate(cusps, 1):
        star_lord, sub_lord, ssl, star_deg, pos_in_star, sub_span, pos_in_sub = kp_star_sub_sub(cusp_deg)
        detailed_cusps.append({
            "house": i,
            "degree": round(cusp_deg, 4),
            "sign": sign_name_from_deg(cusp_deg),
            "star_lord": star_lord,
            "sub_lord": sub_lord,
            "sub_sub_lord": ssl,
            "star_span_deg": round(star_deg, 4),
            "pos_within_star_deg": round(pos_in_star, 4),
            "sub_span_deg": round(sub_span, 4),
            "pos_within_sub_deg": round(pos_in_sub, 4),
        })

    # Planetary star/sub snapshot (trad planets only for KP horary table)
    planetary_kp = []
    for b in chart.get("bodies", []):
        nm = b["name"]
        if nm not in PARASHARI_PLANETS:
            continue
        lon = float(b["longitude_deg"])
        star_lord, sub_lord, ssl, star_deg, pos_in_star, sub_span, pos_in_sub = kp_star_sub_sub(lon)
        planetary_kp.append({
            "planet": nm,
            "longitude_deg": round(lon, 4),
            "sign": sign_name_from_deg(lon),
            "star_lord": star_lord,
            "sub_lord": sub_lord,
            "sub_sub_lord": ssl,
            "star_span_deg": round(star_deg, 4),
            "pos_within_star_deg": round(pos_in_star, 4),
            "sub_span_deg": round(sub_span, 4),
            "pos_within_sub_deg": round(pos_in_sub, 4),
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
            "ruling_planets": sorted(list(rp)),
        },
        "cusp_analysis": detailed_cusps,
        "planetary_kp": planetary_kp,
        "signification_evidence": evidence,
        "judgement": {"answer": base_ans, "confidence": round(conf, 3), "method": "KP cusp sub-lords (+SSL & RP bias)"}
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
