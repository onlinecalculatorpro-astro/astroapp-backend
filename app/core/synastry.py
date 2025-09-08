# app/core/synastry.py
# -*- coding: utf-8 -*-
"""
Research-grade Synastry & Composite module (v12)

Simplified, robust integration patterns aligned with working paran module.
Focuses on core functionality with defensive programming.

Public APIs
-----------
compute_synastry(natal_a, natal_b, **kwargs) -> dict
compute_composite(natal_a, natal_b, **kwargs) -> dict  
synastry_report(natal_a, natal_b, **kwargs) -> dict

Key simplifications:
- Streamlined ephemeris calls using proven patterns
- Robust error handling with graceful degradation
- Simplified dependency resolution
- Clear separation of concerns
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime

# ═══════════════════════════════ RESILIENT IMPORTS ═══════════════════════════════

# Ephemeris adapter (required)
try:
    from app.core.ephemeris_adapter import EphemerisAdapter
    _EPHEMERIS_AVAILABLE = True
except Exception as e:
    EphemerisAdapter = None
    _EPHEMERIS_AVAILABLE = False
    _EPHEMERIS_ERROR = e

# Timescales builder (required)
try:
    from app.core.timescales import build_timescales
    _TIMESCALES_AVAILABLE = True
except Exception as e:
    build_timescales = None
    _TIMESCALES_AVAILABLE = False
    _TIMESCALES_ERROR = e

# Houses computation (optional for overlays)
try:
    from app.core.houses import compute_houses_with_policy
    _HOUSES_AVAILABLE = True
except Exception as e:
    compute_houses_with_policy = None
    _HOUSES_AVAILABLE = False
    _HOUSES_ERROR = e

# ═══════════════════════════════ CONSTANTS ═══════════════════════════════════════

MAJOR_BODIES = ("Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")

DEFAULT_ORBS = {
    "conjunction": 8.0,
    "opposition": 6.0,
    "trine": 6.0,
    "square": 5.0,
    "sextile": 3.0,
    "quincunx": 2.0,
    "parallel": 1.0,  # degrees
    "antiscia": 2.0,
}

ASPECT_ANGLES = {
    "conjunction": 0.0,
    "sextile": 60.0,
    "square": 90.0,
    "trine": 120.0,
    "opposition": 180.0,
    "quincunx": 150.0,
}

ASPECT_WEIGHTS = {
    "conjunction": 5.0,
    "trine": 4.0,
    "sextile": 2.5,
    "square": -3.5,
    "opposition": -4.0,
    "quincunx": -1.0,
    "parallel": 2.0,
    "antiscia": 1.5,
}

# ═══════════════════════════════ UTILITIES ═══════════════════════════════════════

def _normalize_angle(degrees: float) -> float:
    """Normalize angle to [0, 360) range."""
    return degrees % 360.0

def _angular_separation(a: float, b: float) -> float:
    """Calculate shortest angular separation between two angles."""
    diff = abs(_normalize_angle(b) - _normalize_angle(a))
    return min(diff, 360.0 - diff)

def _circular_midpoint(a: float, b: float) -> float:
    """Calculate circular midpoint between two angles."""
    a_norm = _normalize_angle(a)
    b_norm = _normalize_angle(b)
    
    # Choose the direction with smaller arc
    diff = _normalize_angle(b_norm - a_norm)
    if diff <= 180.0:
        return _normalize_angle(a_norm + diff * 0.5)
    else:
        return _normalize_angle(a_norm - (360.0 - diff) * 0.5)

def _antiscia_point(longitude: float) -> float:
    """Calculate antiscia point (reflection across 0° Cancer)."""
    return _normalize_angle(180.0 - longitude)

def _mean_obliquity(jd_tt: float) -> float:
    """Calculate mean obliquity of the ecliptic (IAU 2006 approximation)."""
    T = (jd_tt - 2451545.0) / 36525.0
    epsilon_0 = 84381.406 - 46.836769*T - 0.0001831*T*T + 0.00200340*T*T*T
    return epsilon_0 / 3600.0  # Convert arcseconds to degrees

def _ecliptic_to_declination(longitude: float, latitude: float, jd_tt: float) -> float:
    """Convert ecliptic coordinates to declination."""
    epsilon = math.radians(_mean_obliquity(jd_tt))
    lambda_rad = math.radians(longitude)
    beta_rad = math.radians(latitude)
    
    sin_dec = math.sin(beta_rad) * math.cos(epsilon) + math.cos(beta_rad) * math.sin(epsilon) * math.sin(lambda_rad)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_dec))))

# ═══════════════════════════════ CORE FUNCTIONS ══════════════════════════════════

def _resolve_timescales(natal: Dict[str, Any], jd_tt: Optional[float], jd_ut1: Optional[float]) -> Tuple[float, float, List[str]]:
    """
    Resolve timescales for a natal chart.
    Returns (jd_tt, jd_ut1, warnings)
    """
    warnings = []
    
    # Use provided strict timescales if available
    if jd_tt is not None and jd_ut1 is not None:
        return float(jd_tt), float(jd_ut1), warnings
    
    # Build from civil time
    if not _TIMESCALES_AVAILABLE:
        raise RuntimeError(f"Timescales builder unavailable: {_TIMESCALES_ERROR}")
    
    try:
        date_str = natal.get("date")
        time_str = natal.get("time") 
        tz_name = natal.get("place_tz")
        
        if not all([date_str, time_str, tz_name]):
            raise ValueError("Missing date, time, or place_tz in natal data")
        
        ts = build_timescales(date_str, time_str, tz_name, 0.0)  # DUT1=0.0s assumption
        warnings.append("timescales_computed_with_dut1_0")
        
        return float(ts.jd_tt), float(ts.jd_ut1), warnings
        
    except Exception as e:
        raise RuntimeError(f"Failed to resolve timescales: {e}")

def _get_planet_positions(jd_tt: float, place: Optional[Dict[str, Any]], frame: str, bodies: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Get planetary positions using ephemeris adapter.
    Returns (positions, warnings)
    """
    warnings = []
    
    if not _EPHEMERIS_AVAILABLE:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPHEMERIS_ERROR}")
    
    try:
        adapter = EphemerisAdapter(frame=frame)
        
        # Determine center and coordinates
        if place and all(k in place for k in ["latitude", "longitude"]):
            center = "topocentric"
            kwargs = {
                "jd_tt": jd_tt,
                "bodies": bodies,
                "center": center,
                "latitude": float(place["latitude"]),
                "longitude": float(place["longitude"]),
                "elevation_m": float(place.get("elev_m", 0.0)),
            }
        else:
            center = "geocentric"
            kwargs = {
                "jd_tt": jd_tt,
                "bodies": bodies,
                "center": center,
            }
            if place is None:
                warnings.append("geocentric_no_coordinates")
        
        # Try different adapter methods
        result = None
        
        # Method 1: Modern comprehensive method
        if hasattr(adapter, 'ecliptic_longitudes_and_velocities'):
            try:
                result = adapter.ecliptic_longitudes_and_velocities(**kwargs)
            except Exception as e:
                warnings.append(f"longitudes_and_velocities_failed_{type(e).__name__}")
        
        # Method 2: Longitudes only
        if result is None and hasattr(adapter, 'ecliptic_longitudes'):
            try:
                result = adapter.ecliptic_longitudes(**kwargs)
            except Exception as e:
                warnings.append(f"longitudes_failed_{type(e).__name__}")
        
        if result is None:
            raise RuntimeError("No working ephemeris method found")
        
        # Normalize result to standard format
        positions = []
        
        if isinstance(result, dict):
            # Handle different result formats
            if "results" in result and isinstance(result["results"], list):
                # Standard format
                for item in result["results"]:
                    if isinstance(item, dict) and "name" in item:
                        pos = {
                            "name": item["name"],
                            "longitude": float(item.get("longitude", item.get("lon", 0.0))),
                            "latitude": float(item.get("latitude", item.get("lat", 0.0))),
                        }
                        if "speed" in item or "velocity" in item:
                            pos["speed"] = float(item.get("speed", item.get("velocity", 0.0)))
                        positions.append(pos)
            else:
                # Simple dict format {body: longitude}
                for name, value in result.items():
                    if isinstance(value, (int, float)):
                        positions.append({
                            "name": name,
                            "longitude": float(value),
                            "latitude": 0.0,
                        })
        
        return positions, warnings
        
    except Exception as e:
        raise RuntimeError(f"Ephemeris query failed: {e}")

def _compute_houses(jd_tt: float, jd_ut1: float, place: Optional[Dict[str, Any]], house_system: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Compute houses if coordinates are available.
    Returns (houses_dict, warnings)
    """
    warnings = []
    
    if not place or not all(k in place for k in ["latitude", "longitude"]):
        warnings.append("houses_no_coordinates")
        return None, warnings
    
    if not _HOUSES_AVAILABLE:
        warnings.append("houses_computation_unavailable")
        return None, warnings
    
    try:
        result = compute_houses_with_policy(
            jd_tt=jd_tt,
            jd_ut1=jd_ut1,
            latitude=float(place["latitude"]),
            longitude=float(place["longitude"]),
            elevation_m=float(place.get("elev_m", 0.0)),
            system=house_system,
        )
        return result, warnings
        
    except Exception as e:
        warnings.append(f"houses_computation_failed_{type(e).__name__}")
        return None, warnings

def _find_aspects(positions_a: List[Dict[str, Any]], positions_b: List[Dict[str, Any]], orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    """Find zodiacal aspects between two sets of positions."""
    aspects = []
    
    for pos_a in positions_a:
        for pos_b in positions_b:
            lon_a = pos_a["longitude"]
            lon_b = pos_b["longitude"]
            separation = _angular_separation(lon_a, lon_b)
            
            for aspect_name, aspect_angle in ASPECT_ANGLES.items():
                orb = orbs.get(aspect_name, DEFAULT_ORBS.get(aspect_name, 0.0))
                if orb <= 0:
                    continue
                
                deviation = abs(separation - aspect_angle)
                if deviation <= orb:
                    aspects.append({
                        "planet_a": pos_a["name"],
                        "planet_b": pos_b["name"],
                        "aspect": aspect_name,
                        "angle": aspect_angle,
                        "separation": separation,
                        "orb": deviation,
                        "applying": separation < aspect_angle,  # Simplified
                    })
    
    return aspects

def _find_antiscia_aspects(positions_a: List[Dict[str, Any]], positions_b: List[Dict[str, Any]], orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    """Find antiscia aspects between two sets of positions."""
    aspects = []
    orb = orbs.get("antiscia", DEFAULT_ORBS["antiscia"])
    
    if orb <= 0:
        return aspects
    
    for pos_a in positions_a:
        antiscia_a = _antiscia_point(pos_a["longitude"])
        
        for pos_b in positions_b:
            # Direct antiscia (conjunction-like)
            sep = _angular_separation(antiscia_a, pos_b["longitude"])
            if sep <= orb:
                aspects.append({
                    "planet_a": pos_a["name"],
                    "planet_b": pos_b["name"],
                    "aspect": "antiscia",
                    "angle": 0.0,
                    "separation": sep,
                    "orb": sep,
                    "applying": False,
                })
            
            # Contra-antiscia (opposition-like)
            contra_sep = _angular_separation(antiscia_a, pos_b["longitude"] + 180.0)
            if contra_sep <= orb:
                aspects.append({
                    "planet_a": pos_a["name"],
                    "planet_b": pos_b["name"],
                    "aspect": "contra-antiscia",
                    "angle": 180.0,
                    "separation": contra_sep,
                    "orb": contra_sep,
                    "applying": False,
                })
    
    return aspects

def _find_parallel_aspects(positions_a: List[Dict[str, Any]], positions_b: List[Dict[str, Any]], jd_tt: float, orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    """Find declination parallel aspects between two sets of positions."""
    aspects = []
    orb = orbs.get("parallel", DEFAULT_ORBS["parallel"])
    
    if orb <= 0:
        return aspects
    
    # Calculate declinations
    for pos_a in positions_a:
        dec_a = _ecliptic_to_declination(pos_a["longitude"], pos_a.get("latitude", 0.0), jd_tt)
        
        for pos_b in positions_b:
            dec_b = _ecliptic_to_declination(pos_b["longitude"], pos_b.get("latitude", 0.0), jd_tt)
            
            # Parallel (same declination)
            if abs(dec_a - dec_b) <= orb:
                aspects.append({
                    "planet_a": pos_a["name"],
                    "planet_b": pos_b["name"],
                    "aspect": "parallel",
                    "angle": 0.0,
                    "separation": abs(dec_a - dec_b),
                    "orb": abs(dec_a - dec_b),
                    "applying": False,
                })
            
            # Contra-parallel (opposite declination)
            if abs(dec_a + dec_b) <= orb:
                aspects.append({
                    "planet_a": pos_a["name"],
                    "planet_b": pos_b["name"],
                    "aspect": "contra-parallel",
                    "angle": 180.0,
                    "separation": abs(dec_a + dec_b),
                    "orb": abs(dec_a + dec_b),
                    "applying": False,
                })
    
    return aspects

def _calculate_house_overlays(positions: List[Dict[str, Any]], houses: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate which houses planets fall into."""
    overlays = []
    
    if not houses or "cusps_deg" not in houses:
        return overlays
    
    cusps = houses["cusps_deg"]
    if len(cusps) != 12:
        return overlays
    
    for pos in positions:
        house_num = _find_house_number(pos["longitude"], cusps)
        overlays.append({
            "planet": pos["name"],
            "longitude": pos["longitude"],
            "house": house_num,
        })
    
    return overlays

def _find_house_number(longitude: float, cusps: List[float]) -> int:
    """Find which house a longitude falls into (1-12)."""
    lon = _normalize_angle(longitude)
    
    for i in range(12):
        cusp_current = _normalize_angle(cusps[i])
        cusp_next = _normalize_angle(cusps[(i + 1) % 12])
        
        if cusp_current <= cusp_next:
            # Normal case
            if cusp_current <= lon < cusp_next:
                return i + 1
        else:
            # Wraps around 0°
            if lon >= cusp_current or lon < cusp_next:
                return i + 1
    
    return 1  # Fallback

def _calculate_midpoints(positions_a: List[Dict[str, Any]], positions_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate planetary midpoints between two charts."""
    midpoints = []
    
    # Create lookup for positions_b
    b_positions = {pos["name"]: pos for pos in positions_b}
    
    for pos_a in positions_a:
        if pos_a["name"] in b_positions:
            pos_b = b_positions[pos_a["name"]]
            midpoint_lon = _circular_midpoint(pos_a["longitude"], pos_b["longitude"])
            
            midpoints.append({
                "planet": pos_a["name"],
                "longitude": midpoint_lon,
            })
    
    return midpoints

def _score_aspects(aspects: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """Calculate heuristic compatibility score from aspects."""
    total_score = 0.0
    aspect_scores = {}
    
    for aspect in aspects:
        aspect_type = aspect["aspect"]
        weight = ASPECT_WEIGHTS.get(aspect_type, 0.0)
        
        # Apply tightness factor (closer to exact = higher weight)
        orb = aspect["orb"]
        max_orb = DEFAULT_ORBS.get(aspect_type, 1.0)
        tightness = max(0.0, 1.0 - (orb / max_orb))
        
        score = weight * tightness
        total_score += score
        
        if aspect_type not in aspect_scores:
            aspect_scores[aspect_type] = 0.0
        aspect_scores[aspect_type] += score
    
    return total_score, aspect_scores

# ═══════════════════════════════ PUBLIC API ══════════════════════════════════════

def compute_synastry(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    *,
    jd_tt_a: Optional[float] = None,
    jd_ut1_a: Optional[float] = None,
    jd_tt_b: Optional[float] = None,
    jd_ut1_b: Optional[float] = None,
    place_a: Optional[Dict[str, Any]] = None,
    place_b: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    ayanamsa_deg: float = 0.0,
    zodiac_mode: str = "tropical",
    house_system: str = "placidus",
    orbs: Optional[Dict[str, float]] = None,
    parallels: bool = True,
    antiscia: bool = True,
) -> Dict[str, Any]:
    """
    Compute synastry between two natal charts.
    
    Returns a dictionary with aspects, overlays, midpoints, and compatibility scores.
    """
    all_warnings = []
    
    try:
        # Resolve orbs
        effective_orbs = {**DEFAULT_ORBS, **(orbs or {})}
        
        # Resolve timescales
        jd_tt_a, jd_ut1_a, warnings_a = _resolve_timescales(natal_a, jd_tt_a, jd_ut1_a)
        jd_tt_b, jd_ut1_b, warnings_b = _resolve_timescales(natal_b, jd_tt_b, jd_ut1_b)
        all_warnings.extend(warnings_a)
        all_warnings.extend(warnings_b)
        
        # Resolve places
        place_a_final = place_a if place_a else natal_a
        place_b_final = place_b if place_b else natal_b
        
        # Get planetary positions
        positions_a, warnings_pa = _get_planet_positions(jd_tt_a, place_a_final, frame, list(MAJOR_BODIES))
        positions_b, warnings_pb = _get_planet_positions(jd_tt_b, place_b_final, frame, list(MAJOR_BODIES))
        all_warnings.extend(warnings_pa)
        all_warnings.extend(warnings_pb)
        
        # Apply ayanamsa for sidereal mode
        if zodiac_mode.lower() == "sidereal" and abs(ayanamsa_deg) > 1e-9:
            for pos in positions_a + positions_b:
                pos["longitude"] = _normalize_angle(pos["longitude"] - ayanamsa_deg)
        
        # Compute houses
        houses_a, warnings_ha = _compute_houses(jd_tt_a, jd_ut1_a, place_a_final, house_system)
        houses_b, warnings_hb = _compute_houses(jd_tt_b, jd_ut1_b, place_b_final, house_system)
        all_warnings.extend(warnings_ha)
        all_warnings.extend(warnings_hb)
        
        # Find aspects
        aspects_a_to_b = _find_aspects(positions_a, positions_b, effective_orbs)
        aspects_b_to_a = _find_aspects(positions_b, positions_a, effective_orbs)
        intra_aspects_a = _find_aspects(positions_a, positions_a, effective_orbs)
        intra_aspects_b = _find_aspects(positions_b, positions_b, effective_orbs)
        
        # Add antiscia aspects if requested
        antiscia_aspects = []
        if antiscia:
            antiscia_a_to_b = _find_antiscia_aspects(positions_a, positions_b, effective_orbs)
            antiscia_b_to_a = _find_antiscia_aspects(positions_b, positions_a, effective_orbs)
            antiscia_aspects = antiscia_a_to_b + antiscia_b_to_a
            aspects_a_to_b.extend(antiscia_a_to_b)
            aspects_b_to_a.extend(antiscia_b_to_a)
        
        # Add parallel aspects if requested
        parallel_aspects = []
        if parallels:
            parallel_a_to_b = _find_parallel_aspects(positions_a, positions_b, jd_tt_a, effective_orbs)
            parallel_b_to_a = _find_parallel_aspects(positions_b, positions_a, jd_tt_b, effective_orbs)
            parallel_aspects = parallel_a_to_b + parallel_b_to_a
            aspects_a_to_b.extend(parallel_a_to_b)
            aspects_b_to_a.extend(parallel_b_to_a)
        
        # Calculate overlays
        overlays_a_in_b = _calculate_house_overlays(positions_a, houses_b)
        overlays_b_in_a = _calculate_house_overlays(positions_b, houses_a)
        
        # Calculate midpoints
        midpoints = _calculate_midpoints(positions_a, positions_b)
        
        # Calculate compatibility scores
        all_aspects = aspects_a_to_b + aspects_b_to_a
        total_score, aspect_breakdown = _score_aspects(all_aspects)
        
        return {
            "ok": True,
            "meta": {
                "frame": frame,
                "zodiac_mode": zodiac_mode,
                "ayanamsa_deg": ayanamsa_deg,
                "house_system": house_system,
                "orbs_used": effective_orbs,
                "warnings": all_warnings,
                "timescales": {
                    "chart_a": {"jd_tt": jd_tt_a, "jd_ut1": jd_ut1_a},
                    "chart_b": {"jd_tt": jd_tt_b, "jd_ut1": jd_ut1_b},
                },
            },
            "aspects": {
                "A_to_B": aspects_a_to_b,
                "B_to_A": aspects_b_to_a,
                "intra_A": intra_aspects_a,
                "intra_B": intra_aspects_b,
            },
            "overlays": {
                "A_in_B": overlays_a_in_b,
                "B_in_A": overlays_b_in_a,
            },
            "midpoints": midpoints,
            "scores": {
                "total": total_score,
                "by_aspect": aspect_breakdown,
                "heuristic_warning": "Scores are heuristic only, not predictive",
            },
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": "synastry_computation_failed",
            "details": str(e),
            "warnings": all_warnings,
        }

def compute_composite(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    *,
    method: str = "midpoint",
    jd_tt_ref: Optional[float] = None,
    jd_ut1_ref: Optional[float] = None,
    place_ref: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    ayanamsa_deg: float = 0.0,
    zodiac_mode: str = "tropical",
) -> Dict[str, Any]:
    """
    Compute composite chart using midpoint or Davison method.
    """
    all_warnings = []
    
    try:
        if method not in ["midpoint", "davison"]:
            raise ValueError(f"Unknown composite method: {method}")
        
        # Resolve timescales for both charts
        jd_tt_a, jd_ut1_a, warnings_a = _resolve_timescales(natal_a, None, None)
        jd_tt_b, jd_ut1_b, warnings_b = _resolve_timescales(natal_b, None, None)
        all_warnings.extend(warnings_a)
        all_warnings.extend(warnings_b)
        
        if method == "midpoint":
            # Get positions for both charts
            positions_a, warnings_pa = _get_planet_positions(jd_tt_a, natal_a, frame, list(MAJOR_BODIES))
            positions_b, warnings_pb = _get_planet_positions(jd_tt_b, natal_b, frame, list(MAJOR_BODIES))
            all_warnings.extend(warnings_pa)
            all_warnings.extend(warnings_pb)
            
            # Apply ayanamsa for sidereal mode
            if zodiac_mode.lower() == "sidereal" and abs(ayanamsa_deg) > 1e-9:
                for pos in positions_a + positions_b:
                    pos["longitude"] = _normalize_angle(pos["longitude"] - ayanamsa_deg)
            
            # Calculate midpoint positions
            composite_positions = {}
            for pos_a in positions_a:
                for pos_b in positions_b:
                    if pos_a["name"] == pos_b["name"]:
                        composite_positions[pos_a["name"]] = _circular_midpoint(
                            pos_a["longitude"], pos_b["longitude"]
                        )
                        break
            
            # Calculate midpoint houses if both charts have coordinates
            houses_a, warnings_ha = _compute_houses(jd_tt_a, jd_ut1_a, natal_a, house_system)
            houses_b, warnings_hb = _compute_houses(jd_tt_b, jd_ut1_b, natal_b, house_system)
            all_warnings.extend(warnings_ha)
            all_warnings.extend(warnings_hb)
            
            asc = None
            mc = None
            cusps = None
            
            if houses_a and houses_b:
                if "asc_deg" in houses_a and "asc_deg" in houses_b:
                    asc = _circular_midpoint(houses_a["asc_deg"], houses_b["asc_deg"])
                if "mc_deg" in houses_a and "mc_deg" in houses_b:
                    mc = _circular_midpoint(houses_a["mc_deg"], houses_b["mc_deg"])
                if "cusps_deg" in houses_a and "cusps_deg" in houses_b and len(houses_a["cusps_deg"]) == 12 and len(houses_b["cusps_deg"]) == 12:
                    cusps = [_circular_midpoint(a, b) for a, b in zip(houses_a["cusps_deg"], houses_b["cusps_deg"])]
            
        elif method == "davison":
            # Calculate time and space midpoints
            jd_tt_mid = (jd_tt_a + jd_tt_b) / 2.0
            jd_ut1_mid = (jd_ut1_a + jd_ut1_b) / 2.0
            
            # Calculate place midpoint if both charts have coordinates
            place_mid = None
            if (all(k in natal_a for k in ["latitude", "longitude"]) and 
                all(k in natal_b for k in ["latitude", "longitude"])):
                place_mid = {
                    "latitude": (float(natal_a["latitude"]) + float(natal_b["latitude"])) / 2.0,
                    "longitude": (float(natal_a["longitude"]) + float(natal_b["longitude"])) / 2.0,
                    "elev_m": (float(natal_a.get("elev_m", 0.0)) + float(natal_b.get("elev_m", 0.0))) / 2.0,
                }
            elif place_ref:
                place_mid = place_ref
            else:
                all_warnings.append("davison_no_coordinates")
            
            # Get positions at midpoint time and place
            positions, warnings_p = _get_planet_positions(jd_tt_mid, place_mid, frame, list(MAJOR_BODIES))
            all_warnings.extend(warnings_p)
            
            # Apply ayanamsa for sidereal mode
            if zodiac_mode.lower() == "sidereal" and abs(ayanamsa_deg) > 1e-9:
                for pos in positions:
                    pos["longitude"] = _normalize_angle(pos["longitude"] - ayanamsa_deg)
            
            composite_positions = {pos["name"]: pos["longitude"] for pos in positions}
            
            # Calculate houses at midpoint time and place
            houses, warnings_h = _compute_houses(jd_tt_mid, jd_ut1_mid, place_mid, house_system)
            all_warnings.extend(warnings_h)
            
            asc = houses.get("asc_deg") if houses else None
            mc = houses.get("mc_deg") if houses else None
            cusps = houses.get("cusps_deg") if houses else None
        
        return {
            "ok": True,
            "method": method,
            "positions": composite_positions,
            "asc": asc,
            "mc": mc,
            "cusps": cusps,
            "meta": {
                "frame": frame,
                "zodiac_mode": zodiac_mode,
                "ayanamsa_deg": ayanamsa_deg,
                "house_system": house_system,
                "warnings": all_warnings,
            },
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": "composite_computation_failed",
            "details": str(e),
            "warnings": all_warnings,
        }

def synastry_report(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    *,
    composite_method: str = "midpoint",
    composite_place_ref: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate comprehensive synastry report including both synastry and composite analysis.
    """
    try:
        # Compute synastry
        synastry_result = compute_synastry(natal_a, natal_b, **kwargs)
        if not synastry_result.get("ok", False):
            return synastry_result
        
        # Compute composite
        composite_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ["frame", "house_system", "ayanamsa_deg", "zodiac_mode"]}
        composite_kwargs.update({
            "method": composite_method,
            "place_ref": composite_place_ref,
        })
        
        composite_result = compute_composite(natal_a, natal_b, **composite_kwargs)
        if not composite_result.get("ok", False):
            # Include composite error but continue with synastry
            synastry_result["composite"] = composite_result
        else:
            synastry_result["composite"] = composite_result
        
        # Add report-level metrics
        aspects_count = len(synastry_result.get("aspects", {}).get("A_to_B", [])) + len(synastry_result.get("aspects", {}).get("B_to_A", []))
        
        synastry_result["metrics"] = {
            "total_aspects": aspects_count,
            "compatibility_score": synastry_result.get("scores", {}).get("total", 0.0),
            "report_generated": True,
        }
        
        # Merge warnings
        all_warnings = synastry_result.get("meta", {}).get("warnings", [])
        if composite_result.get("meta", {}).get("warnings"):
            all_warnings.extend([f"composite_{w}" for w in composite_result["meta"]["warnings"]])
        
        synastry_result["meta"]["warnings"] = all_warnings
        synastry_result["meta"]["report_type"] = "comprehensive"
        
        return synastry_result
        
    except Exception as e:
        return {
            "ok": False,
            "error": "synastry_report_failed",
            "details": str(e),
        }
