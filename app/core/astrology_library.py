# western_vedic_astrology_library.py
# -*- coding: utf-8 -*-
"""
Comprehensive Western/Vedic Astrology Integration Library

This library provides authentic integration of Western (Tropical) and Vedic (Sidereal) 
astrological systems, using traditional knowledge from both cultures while referencing
Arabic, Persian, and Hellenistic sources for enhanced understanding.

Key Features:
- Dual zodiac support (Tropical/Sidereal with accurate ayanamsa)
- Complete planetary dignity systems for both traditions
- Western aspects + Vedic Drishti (aspects) integration  
- House systems: Western (Placidus, Equal) + Vedic (Whole Sign/Bhava)
- Traditional timing techniques from both systems
- Comprehensive orb systems and strength calculations
- Cultural authenticity with practical modern application

Sources Integrated:
Western: Ptolemy, Lilly, modern evolutionary astrology
Vedic: Brihat Parashara Hora Shastra, Jaimini, classical texts
Reference: Arabic (Al-Biruni, Abu Ma'shar), Hellenistic (Paulus Alexandrinus)
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# Core System Configuration
# -----------------------------------------------------------------------------

class AstroSystem(Enum):
    WESTERN_TROPICAL = "western_tropical"
    VEDIC_SIDEREAL = "vedic_sidereal"
    DUAL_MODE = "dual_mode"  # Calculate both systems

class ZodiacType(Enum):
    TROPICAL = "tropical"      # Fixed to seasons (Western)
    SIDEREAL = "sidereal"     # Fixed to stars (Vedic)

class HouseSystem(Enum):
    # Western Systems
    PLACIDUS = "placidus"
    EQUAL = "equal"
    WHOLE_SIGN = "whole_sign"
    PORPHYRY = "porphyry"
    KOCH = "koch"
    
    # Vedic Systems  
    VEDIC_WHOLE_SIGN = "vedic_whole_sign"  # Rashi chart
    BHAVA = "bhava"                        # Cusp-based houses
    
class AspectSystem(Enum):
    WESTERN_ASPECTS = "western_aspects"    # Degree-based with orbs
    VEDIC_DRISHTI = "vedic_drishti"       # House-based, fixed
    COMBINED = "combined"                  # Both systems

# Ayanamsa (Precession correction) - Using Lahiri
AYANAMSA_EPOCHS = {
    2000.0: 23.85,  # Lahiri ayanamsa for epoch 2000.0
    2025.0: 24.18   # Current approximate value
}

def calculate_ayanamsa(year: float) -> float:
    """Calculate Lahiri ayanamsa for given year."""
    # Linear interpolation for intermediate years
    base_year = 2000.0
    base_ayanamsa = 23.85
    annual_increase = 0.0139  # degrees per year (approximate)
    
    return base_ayanamsa + (year - base_year) * annual_increase

def tropical_to_sidereal(tropical_longitude: float, year: float = 2025.0) -> float:
    """Convert tropical longitude to sidereal."""
    ayanamsa = calculate_ayanamsa(year)
    sidereal = tropical_longitude - ayanamsa
    return sidereal % 360.0

def sidereal_to_tropical(sidereal_longitude: float, year: float = 2025.0) -> float:
    """Convert sidereal longitude to tropical."""
    ayanamsa = calculate_ayanamsa(year)
    tropical = sidereal_longitude + ayanamsa
    return tropical % 360.0

# -----------------------------------------------------------------------------
# Enhanced Signs with Dual System Support
# -----------------------------------------------------------------------------

SIGNS: List[str] = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
]

VEDIC_SIGN_NAMES: List[str] = [
    "mesha", "vrishabha", "mithuna", "karka", "simha", "kanya",
    "tula", "vrischika", "dhanus", "makara", "kumbha", "meena"
]

@dataclass
class SignData:
    """Comprehensive sign data for both Western and Vedic systems."""
    name: str
    vedic_name: str
    element: str
    modality: str           # Western: cardinal, fixed, mutable
    vedic_nature: str       # Vedic: movable, fixed, dual
    polarity: str          # positive/masculine, negative/feminine
    season: str            # Western seasonal association
    body_parts: List[str]
    
    # Rulership - Western
    western_ruler: str
    western_exaltation: Optional[str]
    western_exaltation_degree: Optional[int]
    western_detriment: List[str]
    western_fall: Optional[str]
    western_fall_degree: Optional[int]
    
    # Rulership - Vedic (same as Western in most cases)
    vedic_ruler: str
    vedic_exaltation: Optional[str] 
    vedic_exaltation_degree: Optional[int]
    vedic_debilitation: Optional[str]
    vedic_debilitation_degree: Optional[int]
    
    # Qualities
    temperament: str       # Hot/Cold, Dry/Moist combinations
    keywords: List[str]

SIGN_DATABASE: Dict[str, SignData] = {
    "aries": SignData(
        name="Aries", vedic_name="Mesha", element="fire", 
        modality="cardinal", vedic_nature="movable", polarity="positive", season="spring",
        body_parts=["head", "brain", "face", "eyes"],
        western_ruler="mars", western_exaltation="sun", western_exaltation_degree=19,
        western_detriment=["venus"], western_fall="saturn", western_fall_degree=21,
        vedic_ruler="mars", vedic_exaltation="sun", vedic_exaltation_degree=10,
        vedic_debilitation="saturn", vedic_debilitation_degree=20,
        temperament="hot_dry", 
        keywords=["initiative", "leadership", "courage", "impulsiveness", "pioneering"]
    ),
    
    "taurus": SignData(
        name="Taurus", vedic_name="Vrishabha", element="earth",
        modality="fixed", vedic_nature="fixed", polarity="negative", season="spring", 
        body_parts=["neck", "throat", "thyroid", "voice"],
        western_ruler="venus", western_exaltation="moon", western_exaltation_degree=3,
        western_detriment=["mars"], western_fall=None, western_fall_degree=None,
        vedic_ruler="venus", vedic_exaltation="moon", vedic_exaltation_degree=3,
        vedic_debilitation=None, vedic_debilitation_degree=None,
        temperament="cold_dry",
        keywords=["stability", "material", "sensuality", "persistence", "values"]
    ),
    
    "gemini": SignData(
        name="Gemini", vedic_name="Mithuna", element="air",
        modality="mutable", vedic_nature="dual", polarity="positive", season="spring",
        body_parts=["arms", "hands", "shoulders", "lungs"],
        western_ruler="mercury", western_exaltation=None, western_exaltation_degree=None,
        western_detriment=["jupiter"], western_fall=None, western_fall_degree=None,
        vedic_ruler="mercury", vedic_exaltation=None, vedic_exaltation_degree=None,
        vedic_debilitation=None, vedic_debilitation_degree=None,
        temperament="hot_moist",
        keywords=["communication", "versatility", "learning", "curiosity", "duality"]
    ),
    
    "cancer": SignData(
        name="Cancer", vedic_name="Karka", element="water",
        modality="cardinal", vedic_nature="movable", polarity="negative", season="summer",
        body_parts=["chest", "stomach", "breasts", "ribcage"],
        western_ruler="moon", western_exaltation="jupiter", western_exaltation_degree=15,
        western_detriment=["saturn"], western_fall="mars", western_fall_degree=28,
        vedic_ruler="moon", vedic_exaltation="jupiter", vedic_exaltation_degree=5,
        vedic_debilitation="mars", vedic_debilitation_degree=28,
        temperament="cold_moist",
        keywords=["nurturing", "emotional", "protective", "intuitive", "home"]
    ),
    
    "leo": SignData(
        name="Leo", vedic_name="Simha", element="fire",
        modality="fixed", vedic_nature="fixed", polarity="positive", season="summer",
        body_parts=["heart", "spine", "back", "upper back"],
        western_ruler="sun", western_exaltation=None, western_exaltation_degree=None,
        western_detriment=["saturn"], western_fall=None, western_fall_degree=None,
        vedic_ruler="sun", vedic_exaltation=None, vedic_exaltation_degree=None,
        vedic_debilitation=None, vedic_debilitation_degree=None,
        temperament="hot_dry",
        keywords=["creativity", "leadership", "drama", "confidence", "authority"]
    ),
    
    "virgo": SignData(
        name="Virgo", vedic_name="Kanya", element="earth", 
        modality="mutable", vedic_nature="dual", polarity="negative", season="late_summer",
        body_parts=["digestive_system", "intestines", "abdomen"],
        western_ruler="mercury", western_exaltation="mercury", western_exaltation_degree=15,
        western_detriment=["jupiter"], western_fall="venus", western_fall_degree=27,
        vedic_ruler="mercury", vedic_exaltation="mercury", vedic_exaltation_degree=15,
        vedic_debilitation="venus", vedic_debilitation_degree=27,
        temperament="cold_dry",
        keywords=["analysis", "service", "perfection", "health", "discrimination"]
    ),
    
    "libra": SignData(
        name="Libra", vedic_name="Tula", element="air",
        modality="cardinal", vedic_nature="movable", polarity="positive", season="autumn",
        body_parts=["kidneys", "lower_back", "adrenals"],
        western_ruler="venus", western_exaltation="saturn", western_exaltation_degree=21,
        western_detriment=["mars"], western_fall="sun", western_fall_degree=19,
        vedic_ruler="venus", vedic_exaltation="saturn", vedic_exaltation_degree=20,
        vedic_debilitation="sun", vedic_debilitation_degree=10,
        temperament="hot_moist",
        keywords=["balance", "harmony", "relationships", "justice", "diplomacy"]
    ),
    
    "scorpio": SignData(
        name="Scorpio", vedic_name="Vrischika", element="water",
        modality="fixed", vedic_nature="fixed", polarity="negative", season="autumn", 
        body_parts=["reproductive_organs", "bladder", "rectum"],
        western_ruler="mars", western_exaltation=None, western_exaltation_degree=None,
        western_detriment=["venus"], western_fall="moon", western_fall_degree=3,
        vedic_ruler="mars", vedic_exaltation=None, vedic_exaltation_degree=None,
        vedic_debilitation="moon", vedic_debilitation_degree=3,
        temperament="cold_moist",
        keywords=["transformation", "intensity", "secrets", "power", "depth"]
    ),
    
    "sagittarius": SignData(
        name="Sagittarius", vedic_name="Dhanus", element="fire",
        modality="mutable", vedic_nature="dual", polarity="positive", season="autumn",
        body_parts=["hips", "thighs", "liver", "sciatic_nerve"],
        western_ruler="jupiter", western_exaltation=None, western_exaltation_degree=None,
        western_detriment=["mercury"], western_fall=None, western_fall_degree=None,
        vedic_ruler="jupiter", vedic_exaltation=None, vedic_exaltation_degree=None,
        vedic_debilitation=None, vedic_debilitation_degree=None,
        temperament="hot_dry",
        keywords=["philosophy", "expansion", "teaching", "truth", "adventure"]
    ),
    
    "capricorn": SignData(
        name="Capricorn", vedic_name="Makara", element="earth",
        modality="cardinal", vedic_nature="movable", polarity="negative", season="winter",
        body_parts=["bones", "knees", "skin", "teeth"],
        western_ruler="saturn", western_exaltation="mars", western_exaltation_degree=28,
        western_detriment=["moon"], western_fall="jupiter", western_fall_degree=15,
        vedic_ruler="saturn", vedic_exaltation="mars", vedic_exaltation_degree=28,
        vedic_debilitation="jupiter", vedic_debilitation_degree=5,
        temperament="cold_dry",
        keywords=["structure", "ambition", "discipline", "authority", "achievement"]
    ),
    
    "aquarius": SignData(
        name="Aquarius", vedic_name="Kumbha", element="air",
        modality="fixed", vedic_nature="fixed", polarity="positive", season="winter",
        body_parts=["ankles", "calves", "circulatory_system"],
        western_ruler="saturn", western_exaltation=None, western_exaltation_degree=None,
        western_detriment=["sun"], western_fall=None, western_fall_degree=None,
        vedic_ruler="saturn", vedic_exaltation=None, vedic_exaltation_degree=None,
        vedic_debilitation=None, vedic_debilitation_degree=None,
        temperament="hot_moist",
        keywords=["innovation", "freedom", "groups", "idealism", "humanitarian"]
    ),
    
    "pisces": SignData(
        name="Pisces", vedic_name="Meena", element="water",
        modality="mutable", vedic_nature="dual", polarity="negative", season="winter",
        body_parts=["feet", "lymphatic_system", "immune_system"],
        western_ruler="jupiter", western_exaltation="venus", western_exaltation_degree=27,
        western_detriment=["mercury"], western_fall="mercury", western_fall_degree=15,
        vedic_ruler="jupiter", vedic_exaltation="venus", vedic_exaltation_degree=27,
        vedic_debilitation="mercury", vedic_debilitation_degree=15,
        temperament="cold_moist",
        keywords=["spirituality", "compassion", "dreams", "intuition", "dissolution"]
    )
}

# -----------------------------------------------------------------------------
# Comprehensive House Systems (Western + Vedic)
# -----------------------------------------------------------------------------

@dataclass
class HouseData:
    """House data supporting both Western and Vedic interpretations."""
    number: int
    name: str
    vedic_name: str
    western_title: str
    vedic_title: str
    
    # House classifications
    western_type: str        # angular, succedent, cadent
    vedic_type: List[str]    # kendra, trikona, upachaya, dusthana, etc.
    
    # Traditional associations
    natural_sign: str        # Traditional sign association
    natural_ruler: str       # Traditional planetary ruler
    planetary_joy: Optional[str]  # Planet that has joy in this house
    
    # Life themes
    western_themes: List[str]
    vedic_themes: List[str]
    shared_themes: List[str]
    
    # Significators
    natural_karaka: str      # Vedic natural significator
    body_parts: List[str]
    
    # Strength factors
    life_areas: Dict[str, float]

HOUSE_DATABASE: Dict[int, HouseData] = {
    1: HouseData(
        number=1, name="First House", vedic_name="Tanu Bhava",
        western_title="Self/Personality", vedic_title="Self/Body/Appearance",
        western_type="angular", vedic_type=["kendra", "trikona", "dharma"],
        natural_sign="aries", natural_ruler="mars", planetary_joy="mercury",
        western_themes=["identity", "appearance", "first impressions", "vitality"],
        vedic_themes=["self", "body", "health", "personality", "general_life_force"],
        shared_themes=["self", "vitality", "appearance", "life_approach"],
        natural_karaka="sun", body_parts=["head", "brain", "face"],
        life_areas={"self": 1.0, "health": 0.8, "career": 0.3, "relationships": 0.2, "spiritual": 0.3}
    ),
    
    2: HouseData(
        number=2, name="Second House", vedic_name="Dhana Bhava", 
        western_title="Money/Values", vedic_title="Wealth/Family/Speech",
        western_type="succedent", vedic_type=["artha"],
        natural_sign="taurus", natural_ruler="venus", planetary_joy=None,
        western_themes=["possessions", "values", "self_worth", "resources"],
        vedic_themes=["wealth", "family", "speech", "food", "early_education"],
        shared_themes=["resources", "values", "material_security"],
        natural_karaka="jupiter", body_parts=["face", "neck", "throat"],
        life_areas={"career": 0.8, "self": 0.5, "relationships": 0.4, "health": 0.3, "spiritual": 0.2}
    ),
    
    3: HouseData(
        number=3, name="Third House", vedic_name="Sahaja Bhava",
        western_title="Communication/Siblings", vedic_title="Courage/Siblings/Arts",
        western_type="cadent", vedic_type=["upachaya", "kama"],
        natural_sign="gemini", natural_ruler="mercury", planetary_joy="moon",
        western_themes=["communication", "learning", "short_trips", "neighbors"],
        vedic_themes=["courage", "siblings", "arts", "skills", "short_journeys"],
        shared_themes=["communication", "siblings", "skills", "short_travel"],
        natural_karaka="mars", body_parts=["shoulders", "arms", "hands"],
        life_areas={"self": 0.6, "career": 0.6, "relationships": 0.6, "health": 0.3, "spiritual": 0.2}
    ),
    
    4: HouseData(
        number=4, name="Fourth House", vedic_name="Sukha Bhava",
        western_title="Home/Family", vedic_title="Home/Mother/Happiness",
        western_type="angular", vedic_type=["kendra", "moksha"],
        natural_sign="cancer", natural_ruler="moon", planetary_joy=None,
        western_themes=["home", "family", "roots", "real_estate", "endings"],
        vedic_themes=["mother", "home", "property", "vehicles", "happiness", "education"],
        shared_themes=["home", "family", "emotional_security", "property"],
        natural_karaka="moon", body_parts=["chest", "heart", "lungs"],
        life_areas={"self": 0.6, "relationships": 0.7, "career": 0.4, "health": 0.4, "spiritual": 0.4}
    ),
    
    5: HouseData(
        number=5, name="Fifth House", vedic_name="Putra Bhava",
        western_title="Creativity/Children", vedic_title="Children/Intelligence/Dharma",
        western_type="succedent", vedic_type=["trikona", "dharma"], 
        natural_sign="leo", natural_ruler="sun", planetary_joy="venus",
        western_themes=["children", "creativity", "romance", "speculation", "fun"],
        vedic_themes=["children", "intelligence", "creativity", "dharma", "mantra", "speculation"],
        shared_themes=["children", "creativity", "intelligence", "romance"],
        natural_karaka="jupiter", body_parts=["stomach", "upper_abdomen"],
        life_areas={"relationships": 0.8, "self": 0.6, "spiritual": 0.7, "career": 0.4, "health": 0.3}
    ),
    
    6: HouseData(
        number=6, name="Sixth House", vedic_name="Ripu Bhava",
        western_title="Health/Service", vedic_title="Enemies/Disease/Service", 
        western_type="cadent", vedic_type=["dusthana", "upachaya", "artha"],
        natural_sign="virgo", natural_ruler="mercury", planetary_joy="mars",
        western_themes=["health", "daily_work", "service", "habits", "pets"],
        vedic_themes=["enemies", "disease", "debts", "service", "daily_work", "obstacles"],
        shared_themes=["health", "service", "daily_routine", "obstacles"],
        natural_karaka="mars", body_parts=["intestines", "lower_abdomen"],
        life_areas={"health": 1.0, "career": 0.7, "self": 0.4, "relationships": 0.3, "spiritual": 0.2}
    ),
    
    7: HouseData(
        number=7, name="Seventh House", vedic_name="Kalatra Bhava",
        western_title="Partnerships/Marriage", vedic_title="Marriage/Business/Travel",
        western_type="angular", vedic_type=["kendra", "kama"],
        natural_sign="libra", natural_ruler="venus", planetary_joy=None,
        western_themes=["marriage", "partnerships", "open_enemies", "contracts"],
        vedic_themes=["spouse", "marriage", "business_partners", "travel", "death"],
        shared_themes=["marriage", "partnerships", "contracts", "others"],
        natural_karaka="venus", body_parts=["kidneys", "lower_back"],
        life_areas={"relationships": 1.0, "self": 0.4, "career": 0.5, "health": 0.3, "spiritual": 0.2}
    ),
    
    8: HouseData(
        number=8, name="Eighth House", vedic_name="Ayu Bhava",
        western_title="Transformation/Death", vedic_title="Longevity/Transformation/Occult",
        western_type="succedent", vedic_type=["dusthana", "moksha"],
        natural_sign="scorpio", natural_ruler="mars", planetary_joy="saturn",
        western_themes=["death", "transformation", "others_money", "occult", "crisis"],
        vedic_themes=["longevity", "transformation", "occult", "inheritance", "research", "secrets"],
        shared_themes=["transformation", "occult", "inheritance", "crisis"],
        natural_karaka="saturn", body_parts=["reproductive_organs", "rectum"],
        life_areas={"spiritual": 0.8, "self": 0.7, "relationships": 0.6, "health": 0.5, "career": 0.3}
    ),
    
    9: HouseData(
        number=9, name="Ninth House", vedic_name="Dharma Bhava",
        western_title="Philosophy/Higher Learning", vedic_title="Father/Dharma/Fortune",
        western_type="cadent", vedic_type=["trikona", "dharma"],
        natural_sign="sagittarius", natural_ruler="jupiter", planetary_joy="sun",
        western_themes=["philosophy", "higher_education", "long_travel", "publishing"],
        vedic_themes=["father", "dharma", "guru", "fortune", "higher_learning", "pilgrimage"],
        shared_themes=["philosophy", "higher_learning", "spirituality", "long_travel"],
        natural_karaka="jupiter", body_parts=["hips", "thighs"],
        life_areas={"spiritual": 0.9, "career": 0.6, "self": 0.5, "relationships": 0.3, "health": 0.2}
    ),
    
    10: HouseData(
        number=10, name="Tenth House", vedic_name="Karma Bhava",
        western_title="Career/Reputation", vedic_title="Career/Status/Father",
        western_type="angular", vedic_type=["kendra", "upachaya", "artha"],
        natural_sign="capricorn", natural_ruler="saturn", planetary_joy="jupiter",
        western_themes=["career", "reputation", "status", "authority", "public_image"],
        vedic_themes=["career", "status", "government", "authority", "fame", "father"],
        shared_themes=["career", "status", "authority", "reputation"],
        natural_karaka="sun", body_parts=["knees", "bones"],
        life_areas={"career": 1.0, "self": 0.6, "relationships": 0.3, "health": 0.2, "spiritual": 0.3}
    ),
    
    11: HouseData(
        number=11, name="Eleventh House", vedic_name="Labha Bhava",
        western_title="Friends/Hopes", vedic_title="Gains/Friends/Elder Siblings",
        western_type="succedent", vedic_type=["upachaya", "kama"],
        natural_sign="aquarius", natural_ruler="saturn", planetary_joy=None,
        western_themes=["friends", "groups", "hopes", "social_causes", "income"],
        vedic_themes=["gains", "friends", "elder_siblings", "income", "desires", "networks"],
        shared_themes=["friends", "gains", "income", "social_networks"],
        natural_karaka="jupiter", body_parts=["ankles", "calves"],
        life_areas={"relationships": 0.8, "career": 0.7, "self": 0.4, "health": 0.2, "spiritual": 0.3}
    ),
    
    12: HouseData(
        number=12, name="Twelfth House", vedic_name="Vyaya Bhava",
        western_title="Subconscious/Hidden", vedic_title="Losses/Liberation/Foreign",
        western_type="cadent", vedic_type=["dusthana", "moksha"],
        natural_sign="pisces", natural_ruler="jupiter", planetary_joy=None,
        western_themes=["subconscious", "hidden_enemies", "institutions", "sacrifice"],
        vedic_themes=["losses", "expenses", "liberation", "foreign_lands", "bed_pleasures", "moksha"],
        shared_themes=["spirituality", "hidden", "sacrifice", "liberation"],
        natural_karaka="saturn", body_parts=["feet", "left_eye"],
        life_areas={"spiritual": 1.0, "health": 0.6, "self": 0.4, "relationships": 0.3, "career": 0.2}
    )
}

# Vedic House Classifications
VEDIC_HOUSE_GROUPS: Dict[str, List[int]] = {
    "kendra": [1, 4, 7, 10],        # Angular - Most powerful
    "trikona": [1, 5, 9],           # Trinal - Most auspicious
    "upachaya": [3, 6, 10, 11],     # Growing - Improve with time
    "dusthana": [6, 8, 12],         # Difficult - 6=enemies, 8=death, 12=loss
    "dharma": [1, 5, 9],            # Life purpose/righteousness
    "artha": [2, 6, 10],            # Material/wealth
    "kama": [3, 7, 11],             # Desires/relationships
    "moksha": [4, 8, 12]            # Liberation/spirituality
}

# -----------------------------------------------------------------------------
# Enhanced Planetary System (Western + Vedic Integration)
# -----------------------------------------------------------------------------

@dataclass
class PlanetData:
    """Complete planetary data for both Western and Vedic systems."""
    name: str
    vedic_name: str
    symbol: str
    
    # Classification
    class_type: str          # luminary, personal, social, outer, node
    vedic_class: str         # graha, chaya_graha (shadow planet)
    gender: str              # masculine, feminine, neuter
    nature: str              # benefic, malefic, neutral
    vedic_nature: str        # saumya (benefic), krura (malefic)
    
    # Physical properties
    element: str             # hot, cold, dry, moist combinations
    body_parts: List[str]
    
    # Domains & keywords
    western_domains: List[str]
    vedic_domains: List[str]
    shared_keywords: List[str]
    
    # Dignities - Western
    western_domicile: List[str]
    western_exaltation: Optional[str]
    western_exaltation_degree: Optional[int]
    western_detriment: List[str] 
    western_fall: Optional[str]
    western_fall_degree: Optional[int]
    
    # Dignities - Vedic
    vedic_own_signs: List[str]
    vedic_exaltation: Optional[str]
    vedic_exaltation_degree: Optional[int]
    vedic_debilitation: Optional[str]
    vedic_debilitation_degree: Optional[int]
    vedic_moolatrikona: Optional[str]  # Special dignity in Vedic
    
    # Vedic relationships
    vedic_friends: List[str]
    vedic_enemies: List[str]
    vedic_neutrals: List[str]
    
    # Motion & behavior
    average_daily_motion: float
    retrograde_frequency: str
    combustion_distance: float
    
    # Aspects - Vedic special aspects (beyond standard 7th house)
    vedic_special_aspects: List[int]
    
    # Maturation age in Vedic astrology
    vedic_maturation_age: int
    
    # Life areas influence
    life_areas: Dict[str, float]

PLANET_DATABASE: Dict[str, PlanetData] = {
    "sun": PlanetData(
        name="Sun", vedic_name="Surya", symbol="☉",
        class_type="luminary", vedic_class="graha", gender="masculine", 
        nature="benefic", vedic_nature="krura_but_gentle",
        element="hot_dry",
        body_parts=["heart", "spine", "right_eye_male", "circulation"],
        western_domains=["identity", "ego", "vitality", "authority", "creativity"],
        vedic_domains=["soul", "father", "government", "authority", "health", "fame"],
        shared_keywords=["self", "leadership", "vitality", "authority", "consciousness"],
        western_domicile=["leo"], western_exaltation="aries", western_exaltation_degree=19,
        western_detriment=["aquarius"], western_fall="libra", western_fall_degree=19,
        vedic_own_signs=["leo"], vedic_exaltation="aries", vedic_exaltation_degree=10,
        vedic_debilitation="libra", vedic_debilitation_degree=10, vedic_moolatrikona="leo",
        vedic_friends=["moon", "mars", "jupiter"], vedic_enemies=["venus", "saturn"],
        vedic_neutrals=["mercury"],
        average_daily_motion=1.0, retrograde_frequency="never", combustion_distance=0.0,
        vedic_special_aspects=[], vedic_maturation_age=22,
        life_areas={"self": 0.9, "career": 0.8, "relationships": 0.3, "health": 0.6, "spiritual": 0.5}
    ),
    
    "moon": PlanetData(
        name="Moon", vedic_name="Chandra", symbol="☽",
        class_type="luminary", vedic_class="graha", gender="feminine",
        nature="benefic", vedic_nature="saumya", 
        element="cold_moist",
        body_parts=["mind", "breasts", "stomach", "left_eye_male", "blood"],
        western_domains=["emotions", "subconscious", "habits", "mother", "public"],
        vedic_domains=["mind", "mother", "emotions", "travel", "water", "popularity"],
        shared_keywords=["emotions", "mind", "mother", "nurturing", "change"],
        western_domicile=["cancer"], western_exaltation="taurus", western_exaltation_degree=3,
        western_detriment=["capricorn"], western_fall="scorpio", western_fall_degree=3,
        vedic_own_signs=["cancer"], vedic_exaltation="taurus", vedic_exaltation_degree=3,
        vedic_debilitation="scorpio", vedic_debilitation_degree=3, vedic_moolatrikona="taurus",
        vedic_friends=["sun", "mercury"], vedic_enemies=[], vedic_neutrals=["mars", "jupiter", "venus", "saturn"],
        average_daily_motion=13.2, retrograde_frequency="never", combustion_distance=12.0,
        vedic_special_aspects=[], vedic_maturation_age=24,
        life_areas={"self": 0.6, "health": 0.7, "relationships": 0.7, "spiritual": 0.4, "career": 0.4}
    ),
    
    "mercury": PlanetData(
        name="Mercury", vedic_name="Budha", symbol="☿",
        class_type="personal", vedic_class="graha", gender="neuter",
        nature="neutral", vedic_nature="saumya",
        element="variable",
        body_parts=["nervous_system", "skin", "hands", "speech"],
        western_domains=["communication", "intellect", "travel", "commerce", "adaptability"],
        vedic_domains=["intelligence", "speech", "learning", "business", "mathematics"],
        shared_keywords=["communication", "intelligence", "learning", "adaptability"],
        western_domicile=["gemini", "virgo"], western_exaltation="virgo", western_exaltation_degree=15,
        western_detriment=["sagittarius", "pisces"], western_fall="pisces", western_fall_degree=15,
        vedic_own_signs=["gemini", "virgo"], vedic_exaltation="virgo", vedic_exaltation_degree=15,
        vedic_debilitation="pisces", vedic_debilitation_degree=15, vedic_moolatrikona="virgo",
        vedic_friends=["sun", "venus"], vedic_enemies=["moon"], vedic_neutrals=["mars", "jupiter", "saturn"],
        average_daily_motion=1.4, retrograde_frequency="periodic", combustion_distance=14.0,
        vedic_special_aspects=[], vedic_maturation_age=32,
        life_areas={"self": 0.5, "career": 0.7, "relationships": 0.5, "health": 0.3, "spiritual": 0.3}
    ),
    
    "venus": PlanetData(
        name="Venus", vedic_name="Shukra", symbol="♀",
        class_type="personal", vedic_class="graha", gender="feminine",
        nature="benefic", vedic_nature="saumya",
        element="cold_moist",
        body_parts=["reproductive_system", "kidneys", "throat", "face"],
        western_domains=["love", "beauty", "values", "relationships", "arts", "pleasure"],
        vedic_domains=["spouse", "luxury", "vehicles", "arts", "beauty", "wealth"],
        shared_keywords=["love", "beauty", "relationships", "harmony", "values"],
        western_domicile=["taurus", "libra"], western_exaltation="pisces", western_exaltation_degree=27,
        western_detriment=["scorpio", "aries"], western_fall="virgo", western_fall_degree=27,
        vedic_own_signs=["taurus", "libra"], vedic_exaltation="pisces", vedic_exaltation_degree=27,
        vedic_debilitation="virgo", vedic_debilitation_degree=27, vedic_moolatrikona="libra",
        vedic_friends=["mercury", "saturn"], vedic_enemies=["sun", "moon"], vedic_neutrals=["mars", "jupiter"],
        average_daily_motion=1.2, retrograde_frequency="periodic", combustion_distance=10.0,
        vedic_special_aspects=[], vedic_maturation_age=25,
        life_areas={"relationships": 0.9, "self": 0.5, "career": 0.5, "health": 0.3, "spiritual": 0.2}
    ),
    
    "mars": PlanetData(
        name="Mars", vedic_name="Mangala", symbol="♂",
        class_type="personal", vedic_class="graha", gender="masculine",
        nature="malefic", vedic_nature="krura",
        element="hot_dry",
        body_parts=["muscles", "blood", "bone_marrow", "genitals"],
        western_domains=["energy", "action", "desire", "conflict", "courage", "sexuality"],
        vedic_domains=["strength", "courage", "siblings", "property", "accidents", "surgery"],
        shared_keywords=["energy", "courage", "action", "conflict", "strength"],
        western_domicile=["aries", "scorpio"], western_exaltation="capricorn", western_exaltation_degree=28,
        western_detriment=["libra", "taurus"], western_fall="cancer", western_fall_degree=28,
        vedic_own_signs=["aries", "scorpio"], vedic_exaltation="capricorn", vedic_exaltation_degree=28,
        vedic_debilitation="cancer", vedic_debilitation_degree=28, vedic_moolatrikona="aries",
        vedic_friends=["sun", "moon", "jupiter"], vedic_enemies=["mercury"], vedic_neutrals=["venus", "saturn"],
        average_daily_motion=0.7, retrograde_frequency="periodic", combustion_distance=17.0,
        vedic_special_aspects=[4, 8], vedic_maturation_age=28,  # Aspects 4th and 8th from itself
        life_areas={"self": 0.7, "relationships": 0.5, "career": 0.7, "health": 0.6, "spiritual": 0.2}
    ),
    
    "jupiter": PlanetData(
        name="Jupiter", vedic_name="Guru", symbol="♃",
        class_type="social", vedic_class="graha", gender="masculine",
        nature="benefic", vedic_nature="saumya",
        element="hot_moist",
        body_parts=["liver", "fat", "hips", "thighs", "pancreas"],
        western_domains=["expansion", "wisdom", "teaching", "philosophy", "growth", "optimism"],
        vedic_domains=["wisdom", "guru", "children", "wealth", "dharma", "husband"],
        shared_keywords=["wisdom", "expansion", "teaching", "growth", "spirituality"],
        western_domicile=["sagittarius", "pisces"], western_exaltation="cancer", western_exaltation_degree=5,
        western_detriment=["gemini", "virgo"], western_fall="capricorn", western_fall_degree=5,
        vedic_own_signs=["sagittarius", "pisces"], vedic_exaltation="cancer", vedic_exaltation_degree=5,
        vedic_debilitation="capricorn", vedic_debilitation_degree=5, vedic_moolatrikona="sagittarius",
        vedic_friends=["sun", "moon", "mars"], vedic_enemies=["mercury", "venus"], vedic_neutrals=["saturn"],
        average_daily_motion=0.08, retrograde_frequency="periodic", combustion_distance=11.0,
        vedic_special_aspects=[5, 9], vedic_maturation_age=16,  # Aspects 5th and 9th from itself
        life_areas={"spiritual": 0.9, "career": 0.7, "self": 0.6, "relationships": 0.5, "health": 0.4}
    ),
    
    "saturn": PlanetData(
        name="Saturn", vedic_name="Shani", symbol="♄",
        class_type="social", vedic_class="graha", gender="neuter",
        nature="malefic", vedic_nature="krura",
        element="cold_dry",
        body_parts=["bones", "teeth", "hair", "nails", "joints"],
        western_domains=["limitation", "discipline", "time", "karma", "responsibility", "structure"],
        vedic_domains=["longevity", "obstacles", "delays", "discipline", "servants", "old_age"],
        shared_keywords=["discipline", "limitation", "time", "karma", "responsibility"],
        western_domicile=["capricorn", "aquarius"], western_exaltation="libra", western_exaltation_degree=20,
        western_detriment=["cancer", "leo"], western_fall="aries", western_fall_degree=20,
        vedic_own_signs=["capricorn", "aquarius"], vedic_exaltation="libra", vedic_exaltation_degree=20,
        vedic_debilitation="aries", vedic_debilitation_degree=20, vedic_moolatrikona="aquarius",
        vedic_friends=["mercury", "venus"], vedic_enemies=["sun", "moon", "mars"], vedic_neutrals=["jupiter"],
        average_daily_motion=0.03, retrograde_frequency="periodic", combustion_distance=15.0,
        vedic_special_aspects=[3, 10], vedic_maturation_age=36,  # Aspects 3rd and 10th from itself
        life_areas={"career": 0.8, "self": 0.7, "relationships": 0.4, "health": 0.5, "spiritual": 0.6}
    ),
    
    "north_node": PlanetData(
        name="North Node", vedic_name="Rahu", symbol="☊",
        class_type="node", vedic_class="chaya_graha", gender="neuter",
        nature="malefic", vedic_nature="krura",
        element="hot_dry",
        body_parts=["nervous_system", "head_region"],
        western_domains=["future_karma", "soul_growth", "life_purpose", "what_to_develop"],
        vedic_domains=["materialism", "illusion", "foreign", "sudden_events", "obsession"],
        shared_keywords=["growth", "ambition", "future", "material_desire"],
        western_domicile=[], western_exaltation="taurus", western_exaltation_degree=None,
        western_detriment=[], western_fall="scorpio", western_fall_degree=None,
        vedic_own_signs=[], vedic_exaltation="taurus", vedic_exaltation_degree=None,
        vedic_debilitation="scorpio", vedic_debilitation_degree=None, vedic_moolatrikona=None,
        vedic_friends=["venus", "saturn"], vedic_enemies=["sun", "moon", "mars"], 
        vedic_neutrals=["mercury", "jupiter"],
        average_daily_motion=-0.05, retrograde_frequency="always", combustion_distance=0.0,
        vedic_special_aspects=[5, 9], vedic_maturation_age=48,  # Some traditions give special aspects
        life_areas={"self": 0.6, "career": 0.7, "spiritual": 0.5, "relationships": 0.4, "health": 0.3}
    ),
    
    "south_node": PlanetData(
        name="South Node", vedic_name="Ketu", symbol="☋",
        class_type="node", vedic_class="chaya_graha", gender="neuter", 
        nature="malefic", vedic_nature="krura_but_spiritual",
        element="hot_dry",
        body_parts=["lower_extremities", "elimination"],
        western_domains=["past_karma", "talents", "what_to_release", "spiritual_gifts"],
        vedic_domains=["spirituality", "moksha", "detachment", "research", "occult"],
        shared_keywords=["spirituality", "detachment", "past", "release"],
        western_domicile=[], western_exaltation="scorpio", western_exaltation_degree=None,
        western_detriment=[], western_fall="taurus", western_fall_degree=None,
        vedic_own_signs=[], vedic_exaltation="scorpio", vedic_exaltation_degree=None,
        vedic_debilitation="taurus", vedic_debilitation_degree=None, vedic_moolatrikona=None,
        vedic_friends=["mars", "jupiter"], vedic_enemies=["sun", "moon"],
        vedic_neutrals=["mercury", "venus", "saturn"],
        average_daily_motion=-0.05, retrograde_frequency="always", combustion_distance=0.0,
        vedic_special_aspects=[5, 9], vedic_maturation_age=48,  # Some traditions give special aspects
        life_areas={"spiritual": 0.8, "self": 0.5, "health": 0.4, "relationships": 0.3, "career": 0.2}
    )
    
    # Note: Outer planets (Uranus, Neptune, Pluto) are primarily Western
    # but can be included for modern Western astrology integration
}

# -----------------------------------------------------------------------------
# Aspect Systems (Western + Vedic Integration)
# -----------------------------------------------------------------------------

# Western Aspects (Degree-based with orbs)
WESTERN_ASPECTS: Dict[str, Dict[str, Any]] = {
    "conjunction": {
        "angle": 0, "polarity": 0.0, "strength": "major", "nature": "neutral",
        "description": "Unity and fusion of planetary energies",
        "keywords": ["fusion", "emphasis", "blend", "focus"]
    },
    "opposition": {
        "angle": 180, "polarity": -0.6, "strength": "major", "nature": "dynamic",
        "description": "Awareness through separation and projection",
        "keywords": ["separation", "awareness", "projection", "balance"]
    },
    "square": {
        "angle": 90, "polarity": -0.7, "strength": "major", "nature": "challenging",
        "description": "Dynamic tension requiring action and growth",
        "keywords": ["challenge", "action", "crisis", "growth"]
    },
    "trine": {
        "angle": 120, "polarity": 0.7, "strength": "major", "nature": "harmonious",
        "description": "Natural flow and ease of expression",
        "keywords": ["harmony", "ease", "talent", "support"]
    },
    "sextile": {
        "angle": 60, "polarity": 0.4, "strength": "major", "nature": "supportive",
        "description": "Opportunities requiring conscious activation",
        "keywords": ["opportunity", "cooperation", "potential", "skill"]
    },
    "quincunx": {
        "angle": 150, "polarity": -0.2, "strength": "minor", "nature": "adjusting",
        "description": "Adjustment and redirection needed",
        "keywords": ["adjustment", "redirect", "health", "service"]
    },
    "semi_square": {
        "angle": 45, "polarity": -0.3, "strength": "minor", "nature": "irritating", 
        "description": "Minor friction and restlessness",
        "keywords": ["irritation", "restless", "minor_crisis"]
    },
    "sesquiquadrate": {
        "angle": 135, "polarity": -0.4, "strength": "minor", "nature": "building",
        "description": "Building pressure seeking release",
        "keywords": ["pressure", "release", "culmination"]
    },
    "semi_sextile": {
        "angle": 30, "polarity": 0.1, "strength": "minor", "nature": "connecting",
        "description": "Subtle connection and growth link",
        "keywords": ["connection", "growth", "development"]
    }
}

# Vedic Aspects (Drishti) - House-based, no orbs
VEDIC_DRISHTI: Dict[str, List[int]] = {
    "sun": [7],           # All planets aspect 7th house
    "moon": [7],
    "mercury": [7],
    "venus": [7],
    "mars": [4, 7, 8],    # Mars has special aspects to 4th, 7th, 8th
    "jupiter": [5, 7, 9], # Jupiter aspects 5th, 7th, 9th
    "saturn": [3, 7, 10], # Saturn aspects 3rd, 7th, 10th
    "north_node": [5, 7, 9],  # Rahu (some traditions)
    "south_node": [5, 7, 9]   # Ketu (some traditions)
}

# Orb Systems
WESTERN_ORBS: Dict[str, Dict[str, Dict[str, float]]] = {
    "natal": {
        "luminaries": {  # Sun & Moon get wider orbs
            "conjunction": 10.0, "opposition": 10.0, "square": 10.0,
            "trine": 10.0, "sextile": 6.0, "quincunx": 3.0,
            "semi_square": 3.0, "sesquiquadrate": 3.0, "semi_sextile": 3.0
        },
        "personal": {  # Mercury, Venus, Mars
            "conjunction": 8.0, "opposition": 8.0, "square": 8.0,
            "trine": 8.0, "sextile": 6.0, "quincunx": 3.0,
            "semi_square": 2.0, "sesquiquadrate": 2.0, "semi_sextile": 2.0
        },
        "social": {  # Jupiter, Saturn
            "conjunction": 8.0, "opposition": 8.0, "square": 8.0,
            "trine": 8.0, "sextile": 6.0, "quincunx": 3.0,
            "semi_square": 2.0, "sesquiquadrate": 2.0, "semi_sextile": 2.0
        }
    },
    "transit": {
        "luminaries": {
            "conjunction": 8.0, "opposition": 8.0, "square": 8.0,
            "trine": 8.0, "sextile": 4.0, "quincunx": 2.0
        },
        "personal": {
            "conjunction": 6.0, "opposition": 6.0, "square": 6.0,
            "trine": 6.0, "sextile": 4.0, "quincunx": 2.0
        },
        "social": {
            "conjunction": 6.0, "opposition": 6.0, "square": 6.0,
            "trine": 6.0, "sextile": 4.0, "quincunx": 2.0
        }
    },
    "progression": {
        "all": {
            "conjunction": 1.0, "opposition": 1.0, "square": 1.0,
            "trine": 1.0, "sextile": 1.0, "quincunx": 0.5
        }
    }
}

# Vedic orbs (mainly for conjunctions within signs)
VEDIC_ORBS: Dict[str, float] = {
    "conjunction": 15.0,  # Same sign conjunction
    "close_conjunction": 5.0,  # Very tight conjunction
    "exact_conjunction": 1.0   # Near-exact conjunction
}

# -----------------------------------------------------------------------------
# Dignity and Strength Assessment
# -----------------------------------------------------------------------------

class DignityLevel(Enum):
    # Essential Dignities (sign-based)
    OWN_SIGN = "own_sign"           # Swakshetra
    EXALTATION = "exaltation"       # Uttcha
    MOOLATRIKONA = "moolatrikona"   # Special Vedic dignity
    FRIENDLY = "friendly"           # Friend's sign
    NEUTRAL = "neutral"             # Neutral sign
    ENEMY = "enemy"                 # Enemy's sign
    DEBILITATION = "debilitation"   # Neecha
    
    # Accidental Dignities (placement-based)
    ANGULAR = "angular"             # Kendra houses
    SUCCEDENT = "succedent"         # 2,5,8,11 houses
    CADENT = "cadent"              # 3,6,9,12 houses
    
    # Special conditions
    COMBUST = "combust"            # Too close to Sun
    RETROGRADE = "retrograde"      # Backward motion
    
class StrengthAssessment(Enum):
    EXALTED = "exalted"           # Very strong
    STRONG = "strong"             # Well-placed 
    MODERATE = "moderate"         # Average condition
    WEAK = "weak"                # Poorly placed
    DEBILITATED = "debilitated"   # Very weak

@dataclass
class PlanetaryStrength:
    """Comprehensive planetary strength assessment for both systems."""
    planet: str
    longitude_tropical: float
    longitude_sidereal: float
    
    # Sign positions
    tropical_sign: str
    sidereal_sign: str
    house: int
    
    # Western strength factors
    western_dignities: List[DignityLevel]
    western_strength_score: float
    
    # Vedic strength factors  
    vedic_dignities: List[DignityLevel]
    vedic_strength_score: float
    
    # Combined assessment
    overall_assessment: StrengthAssessment
    
    # Special conditions
    is_combust: bool
    is_retrograde: bool
    vedic_friendship_with_house_lord: str

def assess_planetary_strength(
    planet: str,
    tropical_longitude: float,
    sidereal_longitude: float, 
    house: int,
    house_lord: str,
    sun_longitude: float,
    system: AstroSystem = AstroSystem.DUAL_MODE
) -> PlanetaryStrength:
    """Comprehensive strength assessment for both Western and Vedic systems."""
    
    planet_data = PLANET_DATABASE.get(planet)
    if not planet_data:
        return None
    
    tropical_sign = get_sign_from_longitude(tropical_longitude)
    sidereal_sign = get_sign_from_longitude(sidereal_longitude)
    
    western_dignities = []
    vedic_dignities = []
    western_score = 0.0
    vedic_score = 0.0
    
    # Western dignity assessment
    if system in [AstroSystem.WESTERN_TROPICAL, AstroSystem.DUAL_MODE]:
        if tropical_sign in planet_data.western_domicile:
            western_dignities.append(DignityLevel.OWN_SIGN)
            western_score += 5.0
            
        if tropical_sign == planet_data.western_exaltation:
            western_dignities.append(DignityLevel.EXALTATION)
            western_score += 4.0
            
        if tropical_sign in planet_data.western_detriment:
            western_dignities.append(DignityLevel.ENEMY)
            western_score -= 5.0
            
        if tropical_sign == planet_data.western_fall:
            western_dignities.append(DignityLevel.DEBILITATION)
            western_score -= 4.0
    
    # Vedic dignity assessment
    if system in [AstroSystem.VEDIC_SIDEREAL, AstroSystem.DUAL_MODE]:
        if sidereal_sign in planet_data.vedic_own_signs:
            vedic_dignities.append(DignityLevel.OWN_SIGN)
            vedic_score += 5.0
            
        if sidereal_sign == planet_data.vedic_exaltation:
            vedic_dignities.append(DignityLevel.EXALTATION)
            vedic_score += 4.0
            
        if sidereal_sign == planet_data.vedic_moolatrikona:
            vedic_dignities.append(DignityLevel.MOOLATRIKONA)
            vedic_score += 3.0
            
        if sidereal_sign == planet_data.vedic_debilitation:
            vedic_dignities.append(DignityLevel.DEBILITATION)
            vedic_score -= 4.0
            
        # Vedic friendship assessment
        house_lord_data = PLANET_DATABASE.get(house_lord)
        friendship = "neutral"
        if house_lord in planet_data.vedic_friends:
            friendship = "friend"
            vedic_score += 1.0
        elif house_lord in planet_data.vedic_enemies:
            friendship = "enemy" 
            vedic_score -= 1.0
    
    # House strength (both systems)
    house_data = HOUSE_DATABASE.get(house)
    if house_data:
        if house_data.western_type == "angular":
            western_dignities.append(DignityLevel.ANGULAR)
            western_score += 2.0
            
        if house in VEDIC_HOUSE_GROUPS["kendra"]:
            vedic_dignities.append(DignityLevel.ANGULAR)
            vedic_score += 2.0
    
    # Combustion check
    sun_distance = abs(tropical_longitude - sun_longitude)
    is_combust = sun_distance <= planet_data.combustion_distance
    if is_combust:
        western_score -= 5.0
        vedic_score -= 5.0
    
    # Overall assessment
    avg_score = (western_score + vedic_score) / 2
    if avg_score >= 8:
        assessment = StrengthAssessment.EXALTED
    elif avg_score >= 4:
        assessment = StrengthAssessment.STRONG
    elif avg_score >= -2:
        assessment = StrengthAssessment.MODERATE
    elif avg_score >= -6:
        assessment = StrengthAssessment.WEAK
    else:
        assessment = StrengthAssessment.DEBILITATED
    
    return PlanetaryStrength(
        planet=planet,
        longitude_tropical=tropical_longitude,
        longitude_sidereal=sidereal_longitude,
        tropical_sign=tropical_sign,
        sidereal_sign=sidereal_sign,
        house=house,
        western_dignities=western_dignities,
        western_strength_score=western_score,
        vedic_dignities=vedic_dignities,
        vedic_strength_score=vedic_score,
        overall_assessment=assessment,
        is_combust=is_combust,
        is_retrograde=False,  # Would need motion data
        vedic_friendship_with_house_lord=friendship if system != AstroSystem.WESTERN_TROPICAL else "n/a"
    )

# -----------------------------------------------------------------------------
# Traditional Timing Techniques
# -----------------------------------------------------------------------------

TIMING_TECHNIQUES: Dict[str, Dict[str, Any]] = {
    # Western Techniques
    "transits": {
        "system": "western",
        "description": "Current planetary positions activating natal chart",
        "duration": "days to years depending on planet",
        "focus": "external events and triggers",
        "orbs": "standard western orbs"
    },
    "progressions": {
        "system": "western", 
        "description": "Symbolic advancement of natal planets (1 day = 1 year)",
        "duration": "months to years",
        "focus": "internal psychological development",
        "orbs": "tight (1 degree or less)"
    },
    "solar_returns": {
        "system": "western",
        "description": "Annual chart when Sun returns to natal position", 
        "duration": "one year",
        "focus": "yearly themes and emphasis",
        "notes": "relocate to current residence"
    },
    "lunar_returns": {
        "system": "western",
        "description": "Monthly chart when Moon returns to natal position",
        "duration": "approximately 28 days", 
        "focus": "monthly emotional themes",
        "notes": "good for short-term planning"
    },
    
    # Vedic Techniques  
    "dasha": {
        "system": "vedic",
        "description": "Planetary period system showing life phases",
        "duration": "varies by dasha system (Vimshottari: 120 years total)",
        "focus": "karmic unfoldment and major life themes",
        "types": ["Vimshottari", "Ashtottari", "Yogini", "Chara"]
    },
    "transits_vedic": {
        "system": "vedic",
        "description": "Current sidereal positions with house-based aspects",
        "duration": "days to years",
        "focus": "triggering dasha results and natal promises", 
        "orbs": "house-based aspects, some conjunction orbs"
    },
    "varshphal": {
        "system": "vedic",
        "description": "Vedic annual chart (similar to solar return)",
        "duration": "one year",
        "focus": "yearly predictions and themes",
        "notes": "uses sidereal zodiac and special calculation methods"
    },
    "prashna": {
        "system": "vedic", 
        "description": "Horary/question chart using Vedic principles",
        "duration": "specific to question",
        "focus": "answering specific queries",
        "notes": "strict traditional rules and dignity assessment"
    },
    
    # Shared/Integrated
    "eclipses": {
        "system": "both",
        "description": "Lunar and solar eclipses activating sensitive points",
        "duration": "6 months activation period",
        "focus": "major life changes and shifts",
        "notes": "especially significant if hitting natal planets/angles"
    },
    "planetary_returns": {
        "system": "both",
        "description": "When planets return to natal positions",
        "duration": "varies by planet (Mercury: 88 days to Saturn: 29 years)",
        "focus": "cyclical themes and renewals",
        "notes": "Jupiter return (12 years) and Saturn return (29 years) most significant"
    }
}

# Vimshottari Dasha Periods (120-year cycle)
VIMSHOTTARI_DASHA: Dict[str, int] = {
    "ketu": 7,
    "venus": 20,
    "sun": 6,
    "moon": 10,
    "mars": 7,
    "north_node": 18,  # Rahu
    "jupiter": 16,
    "saturn": 19,
    "mercury": 17
}

def calculate_dasha_sequence(birth_moon_longitude: float) -> List[Dict[str, Any]]:
    """Calculate Vimshottari Dasha sequence from birth Moon position."""
    # Simplified calculation - full implementation would need nakshatra calculation
    
    # 27 Nakshatras, each 13°20' (800')
    nakshatra_index = int(birth_moon_longitude * 60 / 800) % 27
    
    # Dasha lords by nakshatra (simplified mapping)
    nakshatra_lords = [
        "ketu", "venus", "sun", "moon", "mars", "north_node", "jupiter", "saturn", "mercury"
    ] * 3  # Repeat pattern for all 27 nakshatras
    
    starting_lord = nakshatra_lords[nakshatra_index]
    
    # Create sequence starting from birth dasha lord
    planets = list(VIMSHOTTARI_DASHA.keys())
    start_index = planets.index(starting_lord)
    
    sequence = []
    age = 0
    
    for i in range(9):  # Complete 120-year cycle
        planet_index = (start_index + i) % 9
        planet = planets[planet_index]
        duration = VIMSHOTTARI_DASHA[planet]
        
        sequence.append({
            "planet": planet,
            "start_age": age,
            "end_age": age + duration,
            "duration_years": duration
        })
        age += duration
    
    return sequence

# -----------------------------------------------------------------------------
# Calculation Utilities  
# -----------------------------------------------------------------------------

def get_sign_from_longitude(longitude: float) -> str:
    """Get zodiac sign from longitude."""
    sign_index = int((longitude % 360) / 30)
    return SIGNS[sign_index]

def calculate_house_cusps(ascendant: float, latitude: float, 
                         system: HouseSystem = HouseSystem.WHOLE_SIGN) -> Dict[int, float]:
    """Calculate house cusps for different house systems."""
    
    cusps = {}
    
    if system in [HouseSystem.WHOLE_SIGN, HouseSystem.VEDIC_WHOLE_SIGN]:
        # Whole sign houses - each house is exactly 30 degrees
        for house in range(1, 13):
            cusps[house] = (ascendant + (house - 1) * 30) % 360
            
    elif system == HouseSystem.EQUAL:
        # Equal houses - divide circle equally from ASC
        for house in range(1, 13):
            cusps[house] = (ascendant + (house - 1) * 30) % 360
            
    else:
        # For Placidus, Koch, etc. - simplified calculation
        # Full implementation would require complex trigonometry
        for house in range(1, 13):
            cusps[house] = (ascendant + (house - 1) * 30) % 360
    
    return cusps

def find_aspects_western(planet1_long: float, planet2_long: float,
                        planet1_type: str, planet2_type: str,
                        technique: str = "natal") -> List[Dict[str, Any]]:
    """Find Western aspects between two planets."""
    
    separation = abs(planet1_long - planet2_long)
    if separation > 180:
        separation = 360 - separation
    
    aspects_found = []
    
    # Get appropriate orbs
    orb_table = WESTERN_ORBS.get(technique, WESTERN_ORBS["natal"])
    planet_orbs = orb_table.get(planet1_type, orb_table.get("personal", {}))
    
    for aspect_name, aspect_data in WESTERN_ASPECTS.items():
        aspect_angle = aspect_data["angle"]
        max_orb = planet_orbs.get(aspect_name, 0)
        
        if abs(separation - aspect_angle) <= max_orb:
            orb = abs(separation - aspect_angle)
            aspects_found.append({
                "aspect": aspect_name,
                "orb": orb,
                "separating_angle": separation,
                "exact_angle": aspect_angle,
                "strength": aspect_data["strength"],
                "nature": aspect_data["nature"],
                "applying": separation < aspect_angle  # Simplified
            })
    
    return aspects_found

def find_aspects_vedic(planet1_house: int, planet2_house: int,
                      planet1_name: str) -> List[Dict[str, Any]]:
    """Find Vedic aspects (Drishti) from planet1 to planet2."""
    
    aspects_found = []
    
    # Get special aspects for this planet
    special_aspects = VEDIC_DRISHTI.get(planet1_name, [7])  # Default 7th house aspect
    
    for aspect_house in special_aspects:
        target_house = (planet1_house + aspect_house - 1) % 12
        if target_house == 0:
            target_house = 12
            
        if target_house == planet2_house:
            aspects_found.append({
                "aspect": f"{aspect_house}th_house_aspect",
                "aspect_house": aspect_house,
                "strength": "full" if aspect_house == 7 else "special",
                "nature": "influence"
            })
    
    return aspects_found

# -----------------------------------------------------------------------------
# High-Level Chart Analysis
# -----------------------------------------------------------------------------

@dataclass
class ChartAnalysis:
    """Comprehensive chart analysis combining Western and Vedic systems."""
    
    # Chart data
    system: AstroSystem
    tropical_planets: Dict[str, float]
    sidereal_planets: Dict[str, float]
    house_cusps: Dict[int, float]
    house_system: HouseSystem
    
    # Planetary strengths
    planetary_strengths: Dict[str, PlanetaryStrength]
    
    # Aspects
    western_aspects: List[Dict[str, Any]]
    vedic_aspects: List[Dict[str, Any]]
    
    # Chart patterns and configurations
    chart_patterns: List[str]
    dominant_elements: Dict[str, int]
    dominant_modalities: Dict[str, int]
    
    # Overall assessment
    chart_strength: str
    primary_focus_areas: List[str]
    
    # Timing
    current_dasha: Optional[Dict[str, Any]]
    significant_transits: List[Dict[str, Any]]

def create_integrated_chart_analysis(
    tropical_planets: Dict[str, float],
    ascendant_tropical: float,
    birth_datetime: datetime,
    latitude: float = 0.0,
    longitude: float = 0.0,
    system: AstroSystem = AstroSystem.DUAL_MODE,
    house_system: HouseSystem = HouseSystem.WHOLE_SIGN
) -> ChartAnalysis:
    """Create comprehensive chart analysis integrating Western and Vedic systems."""
    
    # Convert to sidereal
    birth_year = birth_datetime.year + birth_datetime.timetuple().tm_yday / 365.25
    sidereal_planets = {}
    for planet, trop_long in tropical_planets.items():
        sidereal_planets[planet] = tropical_to_sidereal(trop_long, birth_year)
    
    ascendant_sidereal = tropical_to_sidereal(ascendant_tropical, birth_year)
    
    # Calculate house cusps
    house_cusps = calculate_house_cusps(
        ascendant_tropical if system == AstroSystem.WESTERN_TROPICAL else ascendant_sidereal,
        latitude, house_system
    )
    
    # Assess planetary strengths
    planetary_strengths = {}
    sun_tropical = tropical_planets.get("sun", 0)
    
    for planet, trop_long in tropical_planets.items():
        house = find_planet_house(trop_long, house_cusps)
        house_lord = find_house_lord(house, house_cusps, tropical_planets)
        
        strength = assess_planetary_strength(
            planet, trop_long, sidereal_planets[planet], 
            house, house_lord, sun_tropical, system
        )
        if strength:
            planetary_strengths[planet] = strength
    
    # Find aspects
    western_aspects = []
    vedic_aspects = []
    
    planet_list = list(tropical_planets.keys())
    for i, planet1 in enumerate(planet_list):
        for planet2 in planet_list[i+1:]:
            
            # Western aspects
            if system in [AstroSystem.WESTERN_TROPICAL, AstroSystem.DUAL_MODE]:
                p1_data = PLANET_DATABASE.get(planet1)
                p2_data = PLANET_DATABASE.get(planet2)
                if p1_data and p2_data:
                    aspects = find_aspects_western(
                        tropical_planets[planet1], tropical_planets[planet2],
                        p1_data.class_type, p2_data.class_type
                    )
                    for aspect in aspects:
                        aspect.update({"planet1": planet1, "planet2": planet2})
                        western_aspects.append(aspect)
            
            # Vedic aspects
            if system in [AstroSystem.VEDIC_SIDEREAL, AstroSystem.DUAL_MODE]:
                p1_house = find_planet_house(sidereal_planets[planet1], house_cusps)
                p2_house = find_planet_house(sidereal_planets[planet2], house_cusps)
                
                # Check both directions
                aspects1 = find_aspects_vedic(p1_house, p2_house, planet1)
                aspects2 = find_aspects_vedic(p2_house, p1_house, planet2)
                
                for aspect in aspects1:
                    aspect.update({"from_planet": planet1, "to_planet": planet2})
                    vedic_aspects.append(aspect)
                    
                for aspect in aspects2:
                    aspect.update({"from_planet": planet2, "to_planet": planet1})
                    vedic_aspects.append(aspect)
    
    # Analyze chart patterns and dominance
    chart_patterns = analyze_chart_patterns(tropical_planets, western_aspects)
    dominant_elements = calculate_elemental_dominance(tropical_planets)
    dominant_modalities = calculate_modal_dominance(tropical_planets)
    
    # Overall chart strength
    strength_scores = [ps.western_strength_score + ps.vedic_strength_score 
                      for ps in planetary_strengths.values()]
    avg_strength = sum(strength_scores) / len(strength_scores) if strength_scores else 0
    
    if avg_strength >= 6:
        chart_strength = "strong"
    elif avg_strength >= 0:
        chart_strength = "moderate"
    else:
        chart_strength = "challenged"
    
    # Primary focus areas (based on strongest houses and planets)
    focus_areas = determine_focus_areas(planetary_strengths, house_cusps)
    
    # Current dasha (if Vedic system)
    current_dasha = None
    if system in [AstroSystem.VEDIC_SIDEREAL, AstroSystem.DUAL_MODE]:
        moon_longitude = sidereal_planets.get("moon", 0)
        dasha_sequence = calculate_dasha_sequence(moon_longitude)
        # Would need current age to determine current dasha
    
    return ChartAnalysis(
        system=system,
        tropical_planets=tropical_planets,
        sidereal_planets=sidereal_planets,
        house_cusps=house_cusps,
        house_system=house_system,
        planetary_strengths=planetary_strengths,
        western_aspects=western_aspects,
        vedic_aspects=vedic_aspects,
        chart_patterns=chart_patterns,
        dominant_elements=dominant_elements,
        dominant_modalities=dominant_modalities,
        chart_strength=chart_strength,
        primary_focus_areas=focus_areas,
        current_dasha=current_dasha,
        significant_transits=[]
    )

def find_planet_house(planet_longitude: float, house_cusps: Dict[int, float]) -> int:
    """Find which house a planet occupies."""
    # Simplified whole sign approach
    for house in range(1, 13):
        cusp = house_cusps[house]
        next_cusp = house_cusps.get(house + 1, house_cusps[1])
        if house == 12:
            next_cusp = house_cusps[1] + 360
            
        if cusp <= planet_longitude < next_cusp or (house == 12 and planet_longitude >= cusp):
            return house
    return 1

def find_house_lord(house: int, house_cusps: Dict[int, float], 
                   planets: Dict[str, float]) -> str:
    """Find the ruling planet of a house."""
    house_cusp = house_cusps[house]
    house_sign = get_sign_from_longitude(house_cusp)
    sign_data = SIGN_DATABASE.get(house_sign)
    return sign_data.western_ruler if sign_data else "sun"

def analyze_chart_patterns(planets: Dict[str, float], aspects: List[Dict[str, Any]]) -> List[str]:
    """Analyze major chart patterns like stelliums, grand trines, etc."""
    patterns = []
    
    # Look for stelliums (3+ planets in same sign)
    sign_counts = {}
    for planet, longitude in planets.items():
        sign = get_sign_from_longitude(longitude)
        sign_counts[sign] = sign_counts.get(sign, 0) + 1
    
    for sign, count in sign_counts.items():
        if count >= 3:
            patterns.append(f"stellium_in_{sign}")
    
    # Look for grand trines, T-squares, etc. (simplified)
    trine_aspects = [a for a in aspects if a.get("aspect") == "trine"]
    square_aspects = [a for a in aspects if a.get("aspect") == "square"]
    
    if len(trine_aspects) >= 3:
        patterns.append("grand_trine_potential")
    
    if len(square_aspects) >= 2:
        patterns.append("t_square_potential")
    
    return patterns

def calculate_elemental_dominance(planets: Dict[str, float]) -> Dict[str, int]:
    """Calculate elemental emphasis in chart."""
    elements = {"fire": 0, "earth": 0, "air": 0, "water": 0}
    
    for planet, longitude in planets.items():
        sign = get_sign_from_longitude(longitude)
        sign_data = SIGN_DATABASE.get(sign)
        if sign_data:
            elements[sign_data.element] += 1
    
    return elements

def calculate_modal_dominance(planets: Dict[str, float]) -> Dict[str, int]:
    """Calculate modal emphasis in chart."""
    modalities = {"cardinal": 0, "fixed": 0, "mutable": 0}
    
    for planet, longitude in planets.items():
        sign = get_sign_from_longitude(longitude)
        sign_data = SIGN_DATABASE.get(sign)
        if sign_data:
            modalities[sign_data.modality] += 1
    
    return modalities

def determine_focus_areas(strengths: Dict[str, PlanetaryStrength], 
                         house_cusps: Dict[int, float]) -> List[str]:
    """Determine primary life focus areas based on planetary strengths."""
    
    focus_areas = []
    
    # Check for strong planets in angular houses
    for planet, strength in strengths.items():
        if strength.overall_assessment in [StrengthAssessment.EXALTED, StrengthAssessment.STRONG]:
            house_data = HOUSE_DATABASE.get(strength.house)
            if house_data and house_data.western_type == "angular":
                focus_areas.extend(house_data.shared_themes)
    
    # Remove duplicates and return top themes
    return list(set(focus_areas))[:5]

# -----------------------------------------------------------------------------
# Self-Test & Demo
# -----------------------------------------------------------------------------

def _self_test() -> None:
    """Test the integrated library."""
    
    # Test coordinate conversion
    tropical_long = 150.0  # 0° Virgo
    sidereal_long = tropical_to_sidereal(tropical_long, 2025.0)
    assert 125.0 < sidereal_long < 127.0  # Should be around 126° (Leo)
    
    # Test planetary strength assessment  
    strength = assess_planetary_strength(
        "jupiter", 95.0, 71.0, 5, "sun", 120.0, AstroSystem.DUAL_MODE
    )
    assert strength.overall_assessment in [StrengthAssessment.EXALTED, StrengthAssessment.STRONG]
    
    # Test aspect finding
    aspects = find_aspects_western(0.0, 120.0, "luminary", "social", "natal")
    assert len(aspects) > 0
    assert aspects[0]["aspect"] == "trine"
    
    print("Western/Vedic integrated library self-test passed!")

if __name__ == "__main__":
    _self_test()
    
    # Demo integrated analysis
    sample_planets = {
        "sun": 150.0, "moon": 45.0, "mercury": 140.0,
        "venus": 160.0, "mars": 200.0, "jupiter": 95.0, 
        "saturn": 300.0, "north_node": 120.0, "south_node": 300.0
    }
    
    sample_datetime = datetime(1990, 6, 15, 14, 30, tzinfo=timezone.utc)
    
    analysis = create_integrated_chart_analysis(
        sample_planets, 0.0, sample_datetime, 40.0, -74.0,
        AstroSystem.DUAL_MODE, HouseSystem.WHOLE_SIGN
    )
    
    print(f"Chart Analysis Complete:")
    print(f"System: {analysis.system.value}")
    print(f"Chart Strength: {analysis.chart_strength}")
    print(f"Focus Areas: {analysis.primary_focus_areas}")
    print(f"Western Aspects: {len(analysis.western_aspects)}")
    print(f"Vedic Aspects: {len(analysis.vedic_aspects)}")
    print(f"Dominant Elements: {analysis.dominant_elements}")
