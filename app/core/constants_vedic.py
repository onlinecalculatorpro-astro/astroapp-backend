# app/core/constants_vedic.py
# -*- coding: utf-8 -*-
"""
Canonical Vedic constants and helpers (gold-standard, deterministic).

This module centralizes:
- Zodiac & rulers
- Exaltation / Debilitation points
- Mūlatrikoṇa spans
- Natural benefic/malefic sets
- Natural friendship matrix
- Combustion orbs (with retro allowances)
- Graha dṛṣṭi (aspect) schema
- Nakṣatra metadata + Vimśottarī scaffolding

All angles are in degrees. Longitudes are ecliptic, 0..360. Sign indices: 0..11
(0=Aries, 1=Taurus, …, 11=Pisces). Śaṣṭiāṁśa mappings are left to callers.

Public helpers (stable API)
---------------------------
sign_index(lon) -> int  # 0..11
sign_name(idx) -> str
sign_lord(idx) -> str

exaltation_point(planet) -> (sign_idx, deg_in_sign)
debilitation_point(planet) -> (sign_idx, deg_in_sign)
moolatrikona_span(planet) -> (sign_idx, start_deg_in_sign, end_deg_in_sign) | None

is_natural_benefic(planet) -> bool
is_natural_malefic(planet) -> bool
natural_friendships(planet) -> {"friends":[...], "enemies":[...], "neutrals":[...]}

combustion_orb_deg(planet, *, retro: bool=False) -> float
is_combust(sun_lon, planet_lon, planet, *, retro: bool=False) -> bool

graha_drishti_schema(planet) -> {7:1.0, ...}  # house offsets and weights (1.0 = full)
drishti_strength_factor(planet, sep_deg) -> float  # simple falloff helper (optional)

nakshatra_index(lon) -> 1..27
nakshatra_name(idx) -> str
nakshatra_lord(idx) -> str

vimshottari_order() -> [9 lords starting from Ketu]
vimshottari_years() -> {lord: years}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

# ───────────────────────── Basic zodiac scaffold ─────────────────────────

SIGNS: List[str] = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

SIGN_RULERS: List[str] = [
    "Mars","Venus","Mercury","Moon","Sun","Mercury",
    "Venus","Mars","Jupiter","Saturn","Saturn","Jupiter"
]

PLANETS_SEVEN: Tuple[str, ...] = ("Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn")
PLANETS_NINE: Tuple[str, ...] = ("Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Rahu","Ketu")

def _norm(x: float) -> float:
    r = float(x) % 360.0
    return r if r >= 0.0 else r + 360.0

def sign_index(lon: float) -> int:
    return int(math.floor(_norm(lon) / 30.0)) % 12

def sign_name(idx: int) -> str:
    return SIGNS[idx % 12]

def sign_lord(idx: int) -> str:
    return SIGN_RULERS[idx % 12]

# ───────────────── Exaltation / Debilitation / Mūlatrikoṇa ─────────────

# Degrees listed as degrees within the sign (0..30)
_EXALTATION: Dict[str, Tuple[int, float]] = {
    "Sun":      (0, 10.0),   # Aries 10°
    "Moon":     (1, 3.0),    # Taurus 3°
    "Mars":     (9, 28.0),   # Capricorn 28°
    "Mercury":  (5, 15.0),   # Virgo 15°
    "Jupiter":  (3, 5.0),    # Cancer 5°
    "Venus":    (11, 27.0),  # Pisces 27°
    "Saturn":   (6, 20.0),   # Libra 20°
    # Nodes are lineage-dependent; omitted here by design.
}

# Debilitation is opposite the exaltation point
def _opposite(sign_idx: int, deg_in_sign: float) -> Tuple[int, float]:
    return ((sign_idx + 6) % 12, deg_in_sign)

_MOOLATRIKONA: Dict[str, Tuple[int, float, float]] = {
    # (sign_idx, start_deg, end_deg) within that sign
    "Sun":     (4, 0.0, 20.0),   # Leo 0°–20°
    "Moon":    (1, 4.0, 30.0),   # Taurus 4°–30°
    "Mars":    (0, 0.0, 12.0),   # Aries 0°–12°
    "Mercury": (5, 16.0, 20.0),  # Virgo 16°–20°
    "Jupiter": (8, 0.0, 10.0),   # Sagittarius 0°–10°
    "Venus":   (6, 0.0, 15.0),   # Libra 0°–15°
    "Saturn":  (10, 0.0, 20.0),  # Aquarius 0°–20°
}

def exaltation_point(planet: str) -> Optional[Tuple[int, float]]:
    return _EXALTATION.get(planet)

def debilitation_point(planet: str) -> Optional[Tuple[int, float]]:
    ex = _EXALTATION.get(planet)
    return _opposite(*ex) if ex else None

def moolatrikona_span(planet: str) -> Optional[Tuple[int, float, float]]:
    return _MOOLATRIKONA.get(planet)

# ───────────────────────── Benefics / Malefics ─────────────────────────

NATURAL_BENEFICS_BASE = {"Jupiter","Venus","Mercury"}  # Mercury benefic when unafflicted
NATURAL_MALEFICS_BASE = {"Saturn","Mars","Sun","Rahu","Ketu"}

def is_natural_benefic(planet: str) -> bool:
    return planet in NATURAL_BENEFICS_BASE or planet == "Moon"  # waxing/waning handled upstream

def is_natural_malefic(planet: str) -> bool:
    return planet in NATURAL_MALEFICS_BASE or planet == "Moon"  # waning handled upstream

# ───────────────────────── Natural friendships ─────────────────────────
# Classical matrix (BPHS lineage). Neutrals fill the remainder of the seven.

_FRIENDS = {
    "Sun":     {"Moon","Mars","Jupiter"},
    "Moon":    {"Sun","Mercury"},
    "Mars":    {"Sun","Moon","Jupiter"},
    "Mercury": {"Sun","Venus"},
    "Jupiter": {"Sun","Moon","Mars"},
    "Venus":   {"Mercury","Saturn"},
    "Saturn":  {"Mercury","Venus"},
}
_ENEMIES = {
    "Sun":     {"Venus","Saturn"},
    "Moon":    set(),  # some lineages: none
    "Mars":    {"Mercury"},
    "Mercury": {"Moon"},
    "Jupiter": {"Venus","Mercury"},
    "Venus":   {"Sun","Moon"},
    "Saturn":  {"Sun","Moon"},
}

def natural_friendships(planet: str) -> Dict[str, List[str]]:
    if planet not in PLANETS_SEVEN:
        return {"friends": [], "enemies": [], "neutrals": []}
    fr = set(_FRIENDS.get(planet, set()))
    en = set(_ENEMIES.get(planet, set()))
    all7 = set(PLANETS_SEVEN) - {planet}
    ne = all7 - fr - en
    return {
        "friends": sorted(fr),
        "enemies": sorted(en),
        "neutrals": sorted(ne),
    }

# ───────────────────────── Combustion orbs (deg) ──────────────────────
# Orbs vary by authority. These are conservative classical values.
# Mercury has a wider orb when retrograde.

_COMBUST_ORBS = {
    "Moon":    12.0,   # used for amavasya style proximity; many do not treat "combust" Moon
    "Mercury": 12.0,
    "Venus":   10.0,
    "Mars":    17.0,
    "Jupiter": 11.0,
    "Saturn":  15.0,
}

def combustion_orb_deg(planet: str, *, retro: bool = False) -> float:
    if planet == "Mercury" and retro:
        return 15.0
    return _COMBUST_ORBS.get(planet, 0.0)

def is_combust(sun_lon: float, planet_lon: float, planet: str, *, retro: bool=False) -> bool:
    if planet == "Sun":
        return False
    sep = abs((_norm(planet_lon) - _norm(sun_lon) + 180.0) % 360.0 - 180.0)
    return sep <= combustion_orb_deg(planet, retro=retro)

# ───────────────────────── Graha dṛṣṭi schema ─────────────────────────
# House-offset style (Vedic): all planets fully aspect the 7th.
# Mars fully aspects 4th & 8th; Jupiter 5th & 9th; Saturn 3rd & 10th.
# Weights are multipliers (1.0=full). You can extend with falloff models in code.

_GRAHA_DRISHTI = {
    "Sun":     {7: 1.0},
    "Moon":    {7: 1.0},
    "Mars":    {4: 1.0, 7: 1.0, 8: 1.0},
    "Mercury": {7: 1.0},
    "Jupiter": {5: 1.0, 7: 1.0, 9: 1.0},
    "Venus":   {7: 1.0},
    "Saturn":  {3: 1.0, 7: 1.0, 10: 1.0},
    # Nodes handled by lineage; commonly mirror Saturn or no classical drishti.
}

def graha_drishti_schema(planet: str) -> Dict[int, float]:
    return dict(_GRAHA_DRISHTI.get(planet, {7: 1.0}))

def drishti_strength_factor(planet: str, sep_deg: float) -> float:
    """
    Optional smooth falloff: map separation from the exact drishti axis to [0..1].
    For the 7th, axis is 180°. For 'k-th' house drishti, axis is k*30°.
    Uses a triangular kernel within ±12° around axis as a conservative orb.
    """
    schema = graha_drishti_schema(planet)
    best = 0.0
    for k, w in schema.items():
        axis = (k * 30.0) % 360.0
        # sep from axis
        d = abs((_norm(sep_deg) - axis + 180.0) % 360.0 - 180.0)
        if d <= 12.0:
            best = max(best, (1.0 - d / 12.0) * w)
    return best

# ───────────────────────── Nakṣatras & Vimśottarī ─────────────────────

NAKSHATRAS_27: List[str] = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra","Punarvasu",
    "Pushya","Ashlesha","Magha","Purva Phalguni","Uttara Phalguni","Hasta",
    "Chitra","Swati","Vishakha","Anuradha","Jyeshtha","Mula","Purva Ashadha",
    "Uttara Ashadha","Shravana","Dhanishta","Shatabhisha","Purva Bhadrapada",
    "Uttara Bhadrapada","Revati"
]

# Vimśottarī order (Ashwini starts with Ketu)
_VIM_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
_VIM_YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7, "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17}

def nakshatra_index(lon: float) -> int:
    """Return 1..27 index (Ashwini=1). Uses 13°20′ = 13.333…° bins."""
    width = 360.0 / 27.0
    return int(math.floor(_norm(lon) / width)) + 1

def nakshatra_name(idx: int) -> str:
    return NAKSHATRAS_27[(idx - 1) % 27]

def nakshatra_lord(idx: int) -> str:
    return _VIM_ORDER[(idx - 1) % 9]

def vimshottari_order() -> List[str]:
    return list(_VIM_ORDER)

def vimshottari_years() -> Dict[str, int]:
    return dict(_VIM_YEARS)

# ───────────────────────── Convenience / dignity helpers ──────────────

def degrees_in_sign(lon: float) -> float:
    """0..30 degrees inside its sign."""
    return _norm(lon) % 30.0

def is_in_own_sign(planet: str, lon: float) -> bool:
    return sign_lord(sign_index(lon)) == planet

def is_exalted(planet: str, lon: float, tol_deg: float = 1e-6) -> bool:
    ep = exaltation_point(planet)
    if not ep: return False
    s, d = ep
    return sign_index(lon) == s and abs(degrees_in_sign(lon) - d) <= tol_deg

def is_debilitated(planet: str, lon: float, tol_deg: float = 1e-6) -> bool:
    dp = debilitation_point(planet)
    if not dp: return False
    s, d = dp
    return sign_index(lon) == s and abs(degrees_in_sign(lon) - d) <= tol_deg

def in_moolatrikona(planet: str, lon: float) -> bool:
    span = moolatrikona_span(planet)
    if not span: return False
    s, a, b = span
    if sign_index(lon) != s: return False
    x = degrees_in_sign(lon)
    return (a <= x <= b)

# ───────────────────────── Public export list ─────────────────────────

__all__ = [
    "SIGNS","SIGN_RULERS","PLANETS_SEVEN","PLANETS_NINE",
    "sign_index","sign_name","sign_lord",
    "exaltation_point","debilitation_point","moolatrikona_span",
    "is_natural_benefic","is_natural_malefic","natural_friendships",
    "combustion_orb_deg","is_combust",
    "graha_drishti_schema","drishti_strength_factor",
    "NAKSHATRAS_27","nakshatra_index","nakshatra_name","nakshatra_lord",
    "vimshottari_order","vimshottari_years",
    "degrees_in_sign","is_in_own_sign","is_exalted","is_debilitated","in_moolatrikona",
]
