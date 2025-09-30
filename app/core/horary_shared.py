# -*- coding: utf-8 -*-
"""
horary_shared.py — unified Vedic/Prashna helpers (2025-09-29, patched)

What this module provides
-------------------------
• Dataclasses & enums:
    - HoraryInput, QuerentBirthData, HybridPrasnaInput
    - QuestionType (Sanskrit, with legacy-English aliasing)
• Traditional constants:
    - SIGN_NAMES (Sanskrit) + SIGN_NAMES_EN (English)
    - TRAD_PLANETS, RAHU_KETU, ALL_GRAHAS
    - SIGN_LORDS, EXALTATION_SIGNS, MOOLATRIKONA_SIGNS
    - NAKSHATRAS (27 + Abhijit), NAKSHATRA_LORDS
    - VIMSHOTTARI_ORDER, VIMSHOTTARI_YEARS
    - VEDIC_ASPECTS (graha dṛṣṭi), GANDANTA ranges (helpers provided)
    - BENEFICS/MALEFICS, COMBUST_DEG
• Zodiac math & dignity:
    - deg_wrap, sign_index, sign_name_from_deg(lang="en"/"sa")
    - lord_of_sign, angular_sep, is_sandhi, is_gandanta
    - calculate_aspects (Ptolemaic), calc_dignity_simple / calc_dignity_rich
• KP-like star/sub/ssub helpers:
    - kp_star_sub_sub, kp_star_and_sublord (27-equal scheme; Abhijit is informational only)
• Houses & charts:
    - ensure_coords_and_tz (uses local 'now' in resolved tz)
    - build_chart (wraps compute_chart)
    - compute_houses_from_chart (tries strict JD → engine; fallback Equal; **no extra sidereal shift for Whole-Sign/Equal**)
    - whole_sign_cusps_from_asc, rotate_cusps_to_target_asc, house_of
    - safe_get_asc, safe_get_mc
• Vedic dṛṣṭi utilities:
    - graha_drishti_offsets, houses_aspected_by
• Pañcāṅga mini:
    - tithi_index, moon_star
• Radicality (light):
    - radicality_flags
• Question mapping (classical + karakas):
    - ENHANCED_QUESTION_HOUSES
    - normalize_question_type / normalize_question_type_str
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
# Enhanced Constants & Traditional Data
# =============================================================================

SIGN_NAMES = [
    "Mesha","Vrishabha","Mithuna","Karkataka","Simha","Kanya",
    "Tula","Vrishchika","Dhanus","Makara","Kumbha","Meena"
]

SIGN_NAMES_EN = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

TRAD_PLANETS = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
RAHU_KETU = ["Rahu","Ketu"]
ALL_GRAHAS = TRAD_PLANETS + RAHU_KETU

# Traditional dignities
SIGN_LORDS: Dict[int, str] = {
    0: "Mars", 1: "Venus", 2: "Mercury", 3: "Moon", 4: "Sun", 5: "Mercury",
    6: "Venus", 7: "Mars", 8: "Jupiter", 9: "Saturn", 10: "Saturn", 11: "Jupiter"
}

EXALTATION_SIGNS: Dict[str, int] = {
    "Sun": 0, "Moon": 1, "Mars": 9, "Mercury": 5,
    "Jupiter": 3, "Venus": 11, "Saturn": 6
}

MOOLATRIKONA_SIGNS: Dict[str, int] = {
    "Sun": 4, "Moon": 1, "Mars": 0, "Mercury": 5,
    "Jupiter": 8, "Venus": 6, "Saturn": 10
}

# Combustion thresholds (deg from Sun) — conservative defaults
COMBUST_DEG: Dict[str, float] = {
    "Moon": 12.0, "Mercury": 12.0, "Venus": 10.0, "Mars": 17.0, "Jupiter": 11.0, "Saturn": 15.0
}

# Simple benefic/malefic sets
BENEFICS: Set[str] = {"Jupiter","Venus","Moon"}
MALEFICS: Set[str] = {"Saturn","Mars","Sun","Rahu","Ketu"}  # Sun often treated mild malefic in horary

# Nakshatra system (27 regular + Abhijit informational; KP math uses 27 equal)
NAKSHATRAS = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashirsha","Ardra","Punarvasu",
    "Pushya","Ashlesha","Magha","Purva Phalguni","Uttara Phalguni","Hasta",
    "Chitra","Swati","Vishakha","Anuradha","Jyeshtha","Mula","Purva Ashadha",
    "Uttara Ashadha","Shravana","Dhanishta","Shatabhisha","Purva Bhadrapada",
    "Uttara Bhadrapada","Revati","Abhijit"
]

NAKSHATRA_LORDS = [
    "Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter",
    "Saturn","Mercury","Ketu","Venus","Sun","Moon",
    "Mars","Rahu","Jupiter","Saturn","Mercury","Ketu",
    "Venus","Sun","Moon","Mars","Rahu","Jupiter",
    "Saturn","Mercury","Sun"  # Abhijit (traditional lore; not part of 27-equal KP)
]

# Gandanta (last/first 3.6° of Cancer↔Leo, Scorpio↔Sagittarius, Pisces↔Aries)
GANDANTA_WATER_SIGNS = {3, 7, 11}  # Cancer, Scorpio, Pisces
GANDANTA_FIRE_SIGNS  = {4, 8, 0}   # Leo, Sagittarius, Aries
GANDANTA_WINDOW_DEG = 3.6

# Traditional Vedic aspects (graha dṛṣṭi) — sign-based offsets from planet's house
VEDIC_ASPECTS: Dict[str, List[int]] = {
    "Sun": [7], "Moon": [7], "Mercury": [7], "Venus": [7],
    "Mars": [4,7,8], "Jupiter": [5,7,9], "Saturn": [3,7,10],
    "Rahu": [5,7,9], "Ketu": [5,7,9]
}

# Vimshottari dasha sequence and years
VIMSHOTTARI_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
VIMSHOTTARI_YEARS = {"Ketu":7, "Venus":20, "Sun":6, "Moon":10, "Mars":7, "Rahu":18, "Jupiter":16, "Saturn":19, "Mercury":17}
STAR_LEN_DEG: float = 360.0 / 27.0
TOTAL_DASHA: float = 120.0

# =============================================================================
# Question Types (Classical + Legacy Aliases)
# =============================================================================

class QuestionType(Enum):
    # Classical Sanskrit taxonomy
    DHANA   = "dhana"        # wealth/finances (2)
    SAHAJA  = "sahaja"       # siblings/effort (3)
    GRIHA   = "griha"        # home/property (4)
    SANTANA = "santana"      # children (5)
    ROGA    = "roga"         # health (1/6)
    KALATRA = "kalatra"      # marriage/spouse (7)
    MRITYU  = "mrityu"       # longevity/transformations (8)
    VIDYA   = "vidya"        # education (4/5)
    KARMA   = "karma"        # profession (10)
    LABHA   = "labha"        # gains (11)
    VYAYA   = "vyaya"        # losses (12)
    PRAVASA = "pravasa"      # travel/foreign (12/9)
    YUDDHA  = "yuddha"       # conflict/litigation (6)
    NASHTA  = "nashta"       # lost objects (2)

    # Legacy English (kept for compatibility with older callers)
    JOB        = "job"
    MARRIAGE   = "marriage"
    LITIGATION = "litigation"
    HEALTH     = "health"
    LOST_ITEM  = "lost_item"
    PROPERTY   = "property"
    FOREIGN    = "foreign"
    EDUCATION  = "education"
    CHILDREN   = "children"
    BUSINESS   = "business"

# Map legacy English → canonical Sanskrit
_QUESTION_ALIASES: Dict[QuestionType, QuestionType] = {
    QuestionType.JOB:        QuestionType.KARMA,
    QuestionType.MARRIAGE:   QuestionType.KALATRA,
    QuestionType.LITIGATION: QuestionType.YUDDHA,
    QuestionType.HEALTH:     QuestionType.ROGA,
    QuestionType.LOST_ITEM:  QuestionType.NASHTA,
    QuestionType.PROPERTY:   QuestionType.GRIHA,
    QuestionType.FOREIGN:    QuestionType.PRAVASA,
    QuestionType.EDUCATION:  QuestionType.VIDYA,
    QuestionType.CHILDREN:   QuestionType.SANTANA,
    QuestionType.BUSINESS:   QuestionType.KARMA,  # many schools also use 7th; tune if needed
}

def normalize_question_type(q: QuestionType) -> QuestionType:
    return _QUESTION_ALIASES.get(q, q)

def normalize_question_type_str(s: str) -> QuestionType:
    """Accept either Sanskrit/English strings and return enum."""
    key = str(s or "").strip().lower()
    # First try exact enum values
    for qt in QuestionType:
        if qt.value == key:
            return normalize_question_type(qt)
    # Heuristics
    table = {
        "wealth":"dhana", "money":"dhana", "finance":"dhana",
        "siblings":"sahaja", "effort":"sahaja", "property":"griha", "home":"griha",
        "children":"santana", "progeny":"santana",
        "health":"roga", "disease":"roga",
        "marriage":"kalatra", "spouse":"kalatra", "relationship":"kalatra",
        "longevity":"mrityu", "death":"mrityu",
        "education":"vidya", "study":"vidya", "studies":"vidya",
        "career":"karma", "profession":"karma", "job":"karma", "business":"karma",
        "gains":"labha", "income":"labha",
        "loss":"vyaya", "expenses":"vyaya",
        "foreign":"pravasa", "travel":"pravasa",
        "conflict":"yuddha", "litigation":"yuddha", "lawsuit":"yuddha",
        "lost":"nashta", "lost_item":"nashta"
    }
    mapped = table.get(key, key)
    for qt in QuestionType:
        if qt.value == mapped:
            return normalize_question_type(qt)
    # Default
    return QuestionType.KARMA

# Enhanced question-house mappings (classical + karaka list)
ENHANCED_QUESTION_HOUSES: Dict[QuestionType, Dict[str, List[int]]] = {
    QuestionType.DHANA:     {"primary":[2],    "secondary":[11,9],  "supportive":[1,5,10], "obstructive":[6,8,12], "karaka":["Jupiter","Mercury"]},
    QuestionType.SAHAJA:    {"primary":[3],    "secondary":[1,11],  "supportive":[5,9],    "obstructive":[6,8,12], "karaka":["Mars"]},
    QuestionType.GRIHA:     {"primary":[4],    "secondary":[2,11],  "supportive":[1,9,10], "obstructive":[6,8,12], "karaka":["Moon","Mars"]},
    QuestionType.SANTANA:   {"primary":[5],    "secondary":[1,9],   "supportive":[2,7,11], "obstructive":[6,8,12], "karaka":["Jupiter"]},
    QuestionType.ROGA:      {"primary":[1,6],  "secondary":[8],     "supportive":[9,10],   "obstructive":[2,12],   "karaka":["Sun","Mars"]},
    QuestionType.KALATRA:   {"primary":[7],    "secondary":[2,8],   "supportive":[1,5,11], "obstructive":[6,12],   "karaka":["Venus"]},
    QuestionType.MRITYU:    {"primary":[8],    "secondary":[3,12],  "supportive":[1,9],    "obstructive":[2,6],    "karaka":["Saturn"]},
    QuestionType.VIDYA:     {"primary":[4,5],  "secondary":[2,9],   "supportive":[1,10],   "obstructive":[6,8,12], "karaka":["Mercury","Jupiter"]},
    QuestionType.KARMA:     {"primary":[10],   "secondary":[6,2],   "supportive":[1,9,11], "obstructive":[8,12],   "karaka":["Sun","Mercury","Saturn"]},
    QuestionType.LABHA:     {"primary":[11],   "secondary":[2,9],   "supportive":[1,5,10], "obstructive":[6,8,12], "karaka":["Jupiter"]},
    QuestionType.VYAYA:     {"primary":[12],   "secondary":[8,9],   "supportive":[1,4],    "obstructive":[2,6],    "karaka":["Saturn"]},
    QuestionType.PRAVASA:   {"primary":[12,9], "secondary":[3,7],   "supportive":[1,11],   "obstructive":[2,4,8],  "karaka":["Moon","Rahu"]},
    QuestionType.YUDDHA:    {"primary":[6],    "secondary":[8,3],   "supportive":[1,10],   "obstructive":[7,12],   "karaka":["Mars"]},
    QuestionType.NASHTA:    {"primary":[2],    "secondary":[4,8],   "supportive":[1,11],   "obstructive":[6,12],   "karaka":["Mercury"]},
}

def get_question_profile(q: QuestionType) -> Dict[str, List[int]]:
    return ENHANCED_QUESTION_HOUSES[normalize_question_type(q)]

# =============================================================================
# Math & Zodiac helpers
# =============================================================================

def deg_wrap(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else (0.0 if abs(r) < 1e-12 else r)

def sign_index(deg: float) -> int:
    return int(deg_wrap(deg) // 30) % 12

def sign_name_from_deg(deg: Optional[float], *, lang: str = "en") -> Optional[str]:
    if deg is None:
        return None
    idx = sign_index(float(deg))
    return (SIGN_NAMES_EN if lang.lower().startswith("en") else SIGN_NAMES)[idx]

def lord_of_sign(deg: float) -> str:
    return SIGN_LORDS[sign_index(deg)]

def angular_sep(a: float, b: float) -> float:
    d = abs(deg_wrap(a) - deg_wrap(b))
    return d if d <= 180.0 else 360.0 - d

def is_sandhi(deg: float, window: float = 1.0) -> bool:
    """Within `window` degrees of any sign boundary."""
    d = deg_wrap(deg) % 30.0
    return (d < window) or (30.0 - d < window)

def is_gandanta(deg: float, window: float = GANDANTA_WINDOW_DEG) -> bool:
    """True if in last/first `window` degrees of Cancer↔Leo, Scorpio↔Sagittarius, Pisces↔Aries."""
    idx = sign_index(deg)
    pos = deg_wrap(deg) % 30.0
    if idx in GANDANTA_WATER_SIGNS:
        return pos >= (30.0 - window)
    if idx in GANDANTA_FIRE_SIGNS:
        return pos <= window
    return False

# =============================================================================
# Aspects (Western/Ptolemaic — general use)
# =============================================================================

def calculate_aspects(long1: float, long2: float) -> Tuple[float, Optional[str]]:
    """Loose Ptolemaic aspects; returns (sep_degrees, aspect_name|None)."""
    diff = angular_sep(long1, long2)
    for deg0, orb, name in [(0,8,"conjunction"), (60,6,"sextile"), (90,8,"square"), (120,8,"trine"), (180,8,"opposition")]:
        if abs(diff - deg0) <= orb:
            return diff, name
    return diff, None

# Simple dignity — own/exalt/fall
def calc_dignity_simple(longitude: float, planet: str) -> float:
    s = sign_index(longitude)
    exalt = EXALTATION_SIGNS
    own   = {"Sun":[4], "Moon":[3], "Mars":[0,7], "Mercury":[2,5], "Jupiter":[8,11], "Venus":[1,6], "Saturn":[9,10]}
    if planet in exalt and s == exalt[planet]: return 2.0
    if planet in own and s in own[planet]:     return 1.0
    if planet in exalt and s == (exalt[planet] + 6) % 12: return -2.0
    return 0.0

# Richer dignity — adds moolatrikona, friend/enemy, sandhi/gandanta penalties
_FRIENDS = {
    "Sun":{"Moon","Mars","Jupiter"},
    "Moon":{"Sun","Mercury"},
    "Mars":{"Sun","Moon","Jupiter"},
    "Mercury":{"Sun","Venus"},
    "Jupiter":{"Sun","Moon","Mars"},
    "Venus":{"Mercury","Saturn"},
    "Saturn":{"Mercury","Venus"},
}
_ENEMIES = {
    "Sun":{"Venus","Saturn"},
    "Moon":set(),
    "Mars":{"Mercury"},
    "Mercury":{"Moon"},
    "Jupiter":{"Venus","Mercury"},
    "Venus":{"Sun","Moon"},
    "Saturn":{"Sun","Moon"},
}

def calc_dignity_rich(longitude: float, planet: str) -> float:
    sidx = sign_index(longitude)
    dign = 0.0
    if planet in EXALTATION_SIGNS:
        if sidx == EXALTATION_SIGNS[planet]: dign += 2.0
        elif sidx == (EXALTATION_SIGNS[planet] + 6) % 12: dign -= 2.0
    own = {"Sun":[4], "Moon":[3], "Mars":[0,7], "Mercury":[2,5], "Jupiter":[8,11], "Venus":[1,6], "Saturn":[9,10]}
    if planet in own and sidx in own[planet]: dign += 1.0
    if planet in MOOLATRIKONA_SIGNS and sidx == MOOLATRIKONA_SIGNS[planet]: dign += 0.75
    lord = SIGN_LORDS[sidx]
    if planet in _FRIENDS and lord in _FRIENDS[planet]: dign += 0.5
    if planet in _ENEMIES and lord in _ENEMIES[planet]: dign -= 0.5
    if is_sandhi(longitude, 1.0): dign -= 0.2
    if is_gandanta(longitude): dign -= 0.4
    return dign

# =============================================================================
# KP / Nakshatra star/sub/sub-sub (27-equal only)
# =============================================================================

def kp_star_sub_sub(ecl_deg_sidereal: float) -> Tuple[str, str, str, float, float, float, float]:
    """
    Returns: (star_lord, sub_lord, sub_sub_lord, star_span_deg, pos_in_star_deg, sub_span_deg, pos_in_sub_deg)
    """
    pos = deg_wrap(ecl_deg_sidereal)
    star_idx = int(pos // STAR_LEN_DEG)  # 0..26
    star_lord = VIMSHOTTARI_ORDER[star_idx % 9]
    pos_in_star = pos - STAR_LEN_DEG * star_idx

    # Sub level
    start_i = VIMSHOTTARI_ORDER.index(star_lord)
    cycle = VIMSHOTTARI_ORDER[start_i:] + VIMSHOTTARI_ORDER[:start_i]
    acc = 0.0
    sub_lord = cycle[-1]
    sub_start = 0.0
    sub_span = STAR_LEN_DEG
    for lord in cycle:
        portion = STAR_LEN_DEG * (VIMSHOTTARI_YEARS[lord] / TOTAL_DASHA)
        if pos_in_star < acc + portion:
            sub_lord = lord
            sub_start = acc
            sub_span = portion
            break
        acc += portion
    pos_in_sub = pos_in_star - sub_start

    # Sub-sub level
    cycle2 = VIMSHOTTARI_ORDER[VIMSHOTTARI_ORDER.index(sub_lord):] + VIMSHOTTARI_ORDER[:VIMSHOTTARI_ORDER.index(sub_lord)]
    acc2 = 0.0
    ssl = cycle2[-1]
    for lord in cycle2:
        portion2 = sub_span * (VIMSHOTTARI_YEARS[lord] / TOTAL_DASHA)
        if pos_in_sub < acc2 + portion2:
            ssl = lord
            break
        acc2 += portion2

    return (star_lord, sub_lord, ssl, STAR_LEN_DEG, pos_in_star, sub_span, pos_in_sub)

def kp_star_and_sublord(ecl_deg_sidereal: float) -> Tuple[str, str, float, float]:
    star, sub, _ssl, star_span, pos_in_star, _sub_span, _pos_in_sub = kp_star_sub_sub(ecl_deg_sidereal)
    return (star, sub, star_span, pos_in_star)

# =============================================================================
# Houses & zodiac conversions
# =============================================================================

def shift_sidereal(values: List[float], ay_deg: float) -> List[float]:
    """Shift tropical ecliptic longitudes by -ayanamsa → sidereal."""
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
        a, b = c[i], c[(i + 1) % 12]
        inside = ((a <= x) and (x < b)) if a <= b else ((x >= a) or (x < b))
        if inside:
            return i + 1
    return 1

def whole_sign_cusps_from_asc(asc_deg: float) -> List[float]:
    """Whole-Sign cusps: house 1 cusp = 0° of the ascendant's sign, then every 30°."""
    first = sign_index(asc_deg) * 30.0
    return [deg_wrap(first + 30.0 * i) for i in range(12)]

def rotate_cusps_to_target_asc(cusps: List[float], current_asc: float, target_asc: float) -> List[float]:
    """Rotate all cusps so that ASC becomes `target_asc` (for KP 'number' anchoring)."""
    delta = deg_wrap(target_asc - current_asc)
    return [deg_wrap(c + delta) for c in cusps]

# =============================================================================
# Vedic graha-dṛṣṭi helpers (sign-based)
# =============================================================================

def graha_drishti_offsets(planet: str, include_nodes_special: bool = True) -> Set[int]:
    """Returns house offsets (1..12) receiving aspect."""
    if planet in VEDIC_ASPECTS:
        offs = set(VEDIC_ASPECTS[planet])
        if not include_nodes_special and planet in {"Rahu","Ketu"}:
            offs = {7}
        return offs
    return {7}

def houses_aspected_by(planet_name: str, cusps: List[float], bodies: Dict[str, Any],
                       include_nodes_special: bool = True) -> Set[int]:
    """
    Compute sign-based dṛṣṭi targets for a planet using cusps (house zones).
    `bodies` should map planet name -> payload with 'longitude_deg'.
    """
    p = bodies.get(planet_name)
    if not p:
        return set()
    try:
        lon = float(p.get("longitude_deg"))
    except Exception:
        return set()
    h = house_of(lon, cusps)
    targets: Set[int] = set()
    for off in graha_drishti_offsets(planet_name, include_nodes_special=include_nodes_special):
        t = ((h - 1 + (off - 1)) % 12) + 1
        targets.add(t)
    return targets

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
    Resolve tz from args/place first, then generate local 'now' if date/time missing.
    Fills lat/lon from place when absent; defaults to (0,0) and 'UTC' only if unresolved.
    """
    tz_guess = tz

    # Pull from place if helpful
    if (lat is None or lon is None or tz_guess is None) and place:
        rp = resolve_place(place) or {}
        lat = lat if lat is not None else rp.get("lat")
        lon = lon if lon is not None else rp.get("lon")
        tz_guess = tz_guess or rp.get("tz")

    tz_guess = tz_guess or "UTC"

    # If date/time missing, use 'now' in the resolved tz
    if not (date and time_):
        try:
            from zoneinfo import ZoneInfo
            now_local = datetime.now(ZoneInfo(tz_guess))
        except Exception:
            now_local = datetime.now(timezone.utc)
            tz_guess = "UTC"
        date  = now_local.date().isoformat()
        time_ = now_local.time().replace(microsecond=0).isoformat()

    if lat is None or lon is None:
        lat, lon = 0.0, 0.0

    return str(date), str(time_), str(tz_guess), float(lat), float(lon)

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
    NOTE: compute_chart expects 'mode' (not 'zodiac_mode').
    """
    d, t, tzr, la, lo = ensure_coords_and_tz(date, time, tz, place, latitude, longitude)
    return compute_chart({
        "date": d, "time": t, "tz": tzr,
        "place": place, "latitude": la, "longitude": lo,
        "mode": zodiac_mode,                 # ← REQUIRED key for astronomy.compute_chart
        "ayanamsa": ayanamsa,
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

def safe_get_mc(chart: Dict[str, Any]) -> Optional[float]:
    ang = chart.get("angles") or {}
    return ang.get("mc_deg", chart.get("mc_deg"))

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
    returns Equal Houses from ASC. Applies sidereal shift **only** to the advanced
    engine output when requested. Whole-Sign and Equal fallback use chart's zodiac
    as-is (no extra shift).
    """
    asc_any = safe_get_asc(chart)
    mc_any  = safe_get_mc(chart)

    # Whole-Sign branch (ASC required) — use chart's zodiac as-is (no extra shift).
    if (house_system or "").lower() == "whole_sign" and asc_any is not None:
        asc_deg = float(asc_any)
        cusps = whole_sign_cusps_from_asc(asc_deg)
        mc_out  = float(mc_any) if mc_any is not None else deg_wrap(asc_deg + 90.0)
        return {
            "house_system": "whole_sign",
            "cusps_deg": cusps,
            "asc_deg": asc_deg,
            "mc_deg": mc_out,
            "vertex": None,
            "eastpoint": None,
            "warnings": [],
        }

    # Advanced engine path
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
            # NEW: minimal jd_ut-only attempt
            dict(latitude=latitude, longitude=longitude, house_system=house_system,
                 jd_ut=jd_ut),
        ]
        for kwargs in trials:
            # Skip trials where any provided JD is None
            if any(k in kwargs and kwargs[k] is None for k in ("jd_tt","jd_ut","jd_ut1")):
                continue
            try:
                return compute_house_system(**kwargs)  # type: ignore[arg-type]
            except TypeError:
                continue
            except Exception:
                continue
        return None

    payload = _try_engine()

    # Fallback: Equal from ASC — use chart's zodiac as-is (no extra shift).
    if not payload:
        if asc_any is None:
            raise ValueError("houses_fallback_failed:no_asc_in_chart")
        asc_deg = float(asc_any)
        cusps = [deg_wrap(asc_deg + i * 30.0) for i in range(12)]
        mc_guess = float(mc_any) if mc_any is not None else deg_wrap(asc_deg + 90.0)

        return {
            "house_system": f"{house_system} (fallback=Equal from ASC)",
            "cusps_deg": cusps,
            "asc_deg": asc_deg,
            "mc_deg": mc_guess,
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
# Pañcāṅga mini
# =============================================================================

def tithi_index(moon_deg: Optional[float], sun_deg: Optional[float]) -> Optional[int]:
    """Moon–Sun elongation // 12 → 0..29. Returns None if data missing."""
    if moon_deg is None or sun_deg is None:
        return None
    el = deg_wrap(float(moon_deg) - float(sun_deg))
    return int(el // 12.0)

def moon_star(chart: Dict[str, Any]) -> Optional[str]:
    """Return the nakshatra lord (star-lord) of the Moon, if available (sidereal)."""
    moon = next((b for b in chart.get("bodies", []) if b.get("name") == "Moon"), None)
    if not moon:
        return None
    # If chart is tropical, this yields tropical star — caller should pass sidereal Moon.
    star, *_ = kp_star_sub_sub(float(moon["longitude_deg"]))
    return star

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
# Dataclasses / Inputs
# =============================================================================

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
    house_system: str = "sripati"         # accepts "whole_sign"

    # KP options (for KP method)
    kp_house_system: str = "placidus"
    kp_ayanamsa: str | float = "krishnamurti"
    kp_number: Optional[int] = None
    kp_number_mode: str = "anchor_asc"    # "anchor_asc" | "advisory"

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
# Exports
# =============================================================================

__all__ = [
    # Dataclasses & enums
    "HoraryInput","QuerentBirthData","HybridPrasnaInput","QuestionType",
    "ENHANCED_QUESTION_HOUSES","normalize_question_type","normalize_question_type_str","get_question_profile",

    # Planet sets & constants
    "TRAD_PLANETS","RAHU_KETU","ALL_GRAHAS",
    "SIGN_NAMES","SIGN_NAMES_EN","SIGN_LORDS","EXALTATION_SIGNS","MOOLATRIKONA_SIGNS",
    "COMBUST_DEG","BENEFICS","MALEFICS",
    "NAKSHATRAS","NAKSHATRA_LORDS","VIMSHOTTARI_ORDER","VIMSHOTTARI_YEARS","STAR_LEN_DEG","TOTAL_DASHA",
    "VEDIC_ASPECTS","GANDANTA_WINDOW_DEG","is_gandanta",

    # Helpers
    "deg_wrap","sign_index","sign_name_from_deg","lord_of_sign",
    "angular_sep","is_sandhi",
    "calculate_aspects","calc_dignity_simple","calc_dignity_rich",

    # KP
    "kp_star_sub_sub","kp_star_and_sublord",

    # Houses & zodiac
    "shift_sidereal","house_of","whole_sign_cusps_from_asc","rotate_cusps_to_target_asc",

    # Vedic drishti
    "graha_drishti_offsets","houses_aspected_by",

    # Chart/Houses
    "ensure_coords_and_tz","build_chart","compute_houses_from_chart","safe_get_asc","safe_get_mc",

    # Pañcāṅга
    "tithi_index","moon_star",

    # Radicality
    "radicality_flags",
]
