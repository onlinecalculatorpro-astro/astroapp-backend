# app/core/western_synastry.py
# -*- coding: utf-8 -*-
"""
Western Synastry & Composite module (v1.0, tropical-only)

Scope
-----
Implements Western (tropical) relationship techniques:
- Planet-to-planet zodiacal aspects
- Declination parallels / contra-parallels
- Antiscia / contra-antiscia
- House overlays (optional)
- Composite charts (midpoint & Davison)

Non-goals (handled elsewhere):
- Vedic: Ashta Koota, Manglik, Nadi, Navamsa logic, ayanamsa shifts, etc.

Public API (Western-only)
-------------------------
compute_western_synastry(natal_a, natal_b, **kwargs) -> dict
compute_western_composite(natal_a, natal_b, **kwargs) -> dict
western_synastry_report(natal_a, natal_b, **kwargs) -> dict

Error taxonomy (compatible)
---------------------------
- validation_error
- synastry_computation_failed
- composite_value_error
- composite_computation_failed
- synastry_report_failed
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional, TypedDict

# ═══════════════════════════════ RESILIENT IMPORTS ═══════════════════════════════

# Ephemeris adapter (required)
try:
    from app.core.ephemeris_adapter import EphemerisAdapter
    _EPHEM_OK = True
    _EPHEM_ERR: Optional[Exception] = None
except Exception as e:  # pragma: no cover
    EphemerisAdapter = None  # type: ignore
    _EPHEM_OK = False
    _EPHEM_ERR = e

# Timescales builder (required)
try:
    from app.core.timescales import build_timescales
    _TS_OK = True
    _TS_ERR: Optional[Exception] = None
except Exception as e:  # pragma: no cover
    build_timescales = None  # type: ignore
    _TS_OK = False
    _TS_ERR = e

# Houses computation (optional for overlays/composites)
try:
    from app.core.houses import compute_houses_with_policy
    _HOUSES_OK = True
    _HOUSES_ERR: Optional[Exception] = None
except Exception as e:  # pragma: no cover
    compute_houses_with_policy = None  # type: ignore
    _HOUSES_OK = False
    _HOUSES_ERR = e

# ═══════════════════════════════ CONSTANTS ═══════════════════════════════════════

MAJOR_BODIES: Tuple[str, ...] = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"
)

# Default orbs (degrees); declination "parallel" also in degrees of declination.
DEFAULT_ORBS: Dict[str, float] = {
    "conjunction": 8.0,
    "opposition": 6.0,
    "trine": 6.0,
    "square": 5.0,
    "sextile": 3.0,
    "quincunx": 2.0,
    "parallel": 1.0,          # declination orb (deg)
    "antiscia": 2.0,          # solstitial reflection tolerance (deg)
}

ASPECT_ANGLES: Dict[str, float] = {
    "conjunction": 0.0,
    "sextile": 60.0,
    "square": 90.0,
    "trine": 120.0,
    "opposition": 180.0,
    "quincunx": 150.0,
}

ASPECT_WEIGHTS: Dict[str, float] = {
    "conjunction": 5.0,
    "trine": 4.0,
    "sextile": 2.5,
    "square": -3.5,
    "opposition": -4.0,
    "quincunx": -1.0,
    "parallel": 2.0,
    "contra-parallel": -1.5,
    "antiscia": 1.5,
    "contra-antiscia": -1.0,
}

# ═══════════════════════════════ TYPES ══════════════════════════════════════════

class Position(TypedDict, total=False):
    name: str
    longitude: float
    latitude: float
    speed: float  # ecliptic longitude speed (deg/day) if provided

class Aspect(TypedDict, total=False):
    planet_a: str
    planet_b: str
    aspect: str
    angle: float
    separation: float     # absolute sep from exact angle OR declination difference
    orb: float            # same units as separation
    applying: bool

# ═══════════════════════════════ UTILITIES ═══════════════════════════════════════

def _normalize_angle(x: float) -> float:
    return x % 360.0

def _angdiff(a: float, b: float) -> float:
    """Shortest absolute angular distance (degrees) between a and b."""
    d = abs(_normalize_angle(b) - _normalize_angle(a))
    return d if d <= 180.0 else 360.0 - d

def _circ_mid(a: float, b: float) -> float:
    """Circular midpoint of two longitudes (degrees)."""
    a = _normalize_angle(a); b = _normalize_angle(b)
    d = _normalize_angle(b - a)
    return _normalize_angle(a + d * 0.5) if d <= 180.0 else _normalize_angle(a - (360.0 - d) * 0.5)

def _antiscia_point(lon: float) -> float:
    """Antiscia reflection across 0° Cancer."""
    return _normalize_angle(180.0 - lon)

@lru_cache(maxsize=256)
def _mean_obliquity_deg(jd_tt: float) -> float:
    """Mean obliquity IAU 2006 (arcsec→deg), cached by jd_tt."""
    T = (jd_tt - 2451545.0) / 36525.0
    eps0 = 84381.406 - 46.836769*T - 0.0001831*T*T + 0.00200340*T*T*T
    return eps0 / 3600.0

def _ecl_to_decl(lon: float, lat: float, jd_tt: float) -> float:
    """Convert ecliptic (λ,β) to declination δ in degrees."""
    eps = math.radians(_mean_obliquity_deg(jd_tt))
    lam = math.radians(lon); beta = math.radians(lat)
    sin_dec = math.sin(beta)*math.cos(eps) + math.cos(beta)*math.sin(eps)*math.sin(lam)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_dec))))

# ═══════════════════════════════ CORE HELPERS ════════════════════════════════════

def _resolve_timescales(
    natal: Dict[str, Any],
    jd_tt: Optional[float],
    jd_ut1: Optional[float],
) -> Tuple[float, float, List[str]]:
    """
    Resolve timescales for a natal chart.
    Returns: (jd_tt, jd_ut1, warnings)
    Raises:
      ValueError: for missing civil-time fields (date/time/place_tz)
      RuntimeError: when timescales unavailable or fail
    """
    warnings: List[str] = []

    if jd_tt is not None and jd_ut1 is not None:
        return float(jd_tt), float(jd_ut1), warnings

    if not _TS_OK:
        raise RuntimeError(f"Timescales builder unavailable: {_TS_ERR}")

    date_str = natal.get("date")
    time_str = natal.get("time")
    tz_name = natal.get("place_tz")

    if not (date_str and time_str and tz_name):
        raise ValueError("missing date/time/place_tz in natal data")

    try:
        ts = build_timescales(str(date_str), str(time_str), str(tz_name), 0.0)  # DUT1=0s
        warnings.append("timescales_computed_with_dut1_0")
        return float(ts.jd_tt), float(ts.jd_ut1), warnings
    except Exception as e:
        raise RuntimeError(f"Failed to resolve timescales: {e}") from e

def _get_positions(
    jd_tt: float,
    place: Optional[Dict[str, Any]],
    frame: str,
    bodies: List[str],
) -> Tuple[List[Position], List[str]]:
    """
    Query planetary positions via EphemerisAdapter.
    Returns: (positions, warnings)
    Each position: {"name", "longitude", "latitude", ["speed"]}
    """
    warnings: List[str] = []
    if not _EPHEM_OK:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPHEM_ERR}")

    adapter = EphemerisAdapter(frame=frame)

    # Determine center and params
    kwargs: Dict[str, Any]
    if place and all(k in place for k in ("latitude", "longitude")):
        kwargs = {
            "jd_tt": float(jd_tt),
            "bodies": bodies,
            "center": "topocentric",
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
            "elevation_m": float(place.get("elev_m", 0.0)),
        }
    else:
        kwargs = {"jd_tt": float(jd_tt), "bodies": bodies, "center": "geocentric"}
        # warn if a place dict exists but missing coords, or place absent
        if not place:
            warnings.append("geocentric_no_coordinates")
        else:
            if "latitude" not in place or "longitude" not in place:
                warnings.append("geocentric_missing_coordinates")

    result: Any = None

    # Prefer batch with velocities
    if hasattr(adapter, "ecliptic_longitudes_and_velocities"):
        try:
            result = adapter.ecliptic_longitudes_and_velocities(**kwargs)  # type: ignore[arg-type]
        except Exception as e:
            warnings.append(f"longitudes_and_velocities_failed_{type(e).__name__}")

    # Fallback to longitudes only
    if result is None and hasattr(adapter, "ecliptic_longitudes"):
        try:
            result = adapter.ecliptic_longitudes(**kwargs)  # type: ignore[arg-type]
        except Exception as e:
            warnings.append(f"longitudes_failed_{type(e).__name__}")

    if result is None:
        raise RuntimeError("No working ephemeris method found")

    # Normalize outputs
    positions: List[Position] = []
    if isinstance(result, dict):
        if "results" in result and isinstance(result["results"], list):
            for item in result["results"]:
                if isinstance(item, dict) and "name" in item:
                    pos: Position = {
                        "name": str(item["name"]),
                        "longitude": float(item.get("longitude", item.get("lon", 0.0))),
                        "latitude": float(item.get("latitude", item.get("lat", 0.0))),
                    }
                    if "speed" in item or "velocity" in item:
                        pos["speed"] = float(item.get("speed", item.get("velocity", 0.0)))
                    positions.append(pos)
        else:
            # Simple dict: { "Sun": 123.45, ... }
            for name, value in result.items():
                if isinstance(value, (int, float)):
                    positions.append({"name": str(name), "longitude": float(value), "latitude": 0.0})
    return positions, warnings

def _compute_houses(
    jd_tt: float,
    jd_ut1: float,
    place: Optional[Dict[str, Any]],
    house_system: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Compute houses (optional). Returns (houses|None, warnings)
    houses keys typically: asc_deg, mc_deg, cusps_deg (list of 12)
    """
    warnings: List[str] = []

    if not place or not all(k in place for k in ("latitude", "longitude")):
        warnings.append("houses_no_coordinates")
        return None, warnings

    if not _HOUSES_OK:
        warnings.append("houses_computation_unavailable")
        return None, warnings

    try:
        result = compute_houses_with_policy(
            jd_tt=float(jd_tt),
            jd_ut1=float(jd_ut1),
            latitude=float(place["latitude"]),
            longitude=float(place["longitude"]),
            elevation_m=float(place.get("elev_m", 0.0)),
            system=str(house_system),
        )
        return result, warnings
    except Exception as e:
        warnings.append(f"houses_computation_failed_{type(e).__name__}")
        return None, warnings

# ═══════════════════════════════ ASPECTS / OVERLAYS / SCORE ══════════════════════

def _find_zodiacal_aspects(
    pa: List[Position],
    pb: List[Position],
    orbs: Dict[str, float],
) -> List[Aspect]:
    out: List[Aspect] = []
    for a in pa:
        for b in pb:
            sep = _angdiff(a["longitude"], b["longitude"])
            # velocity-based applying/separating if available (deg/day)
            rel_speed = float(a.get("speed", 0.0)) - float(b.get("speed", 0.0))
            for name, angle in ASPECT_ANGLES.items():
                orb = orbs.get(name, DEFAULT_ORBS.get(name, 0.0))
                if orb <= 0:
                    continue
                dev = abs(sep - angle)
                if dev <= orb:
                    applying = (rel_speed < 0.0) if "speed" in a or "speed" in b else (sep < angle)
                    out.append({
                        "planet_a": a["name"],
                        "planet_b": b["name"],
                        "aspect": name,
                        "angle": angle,
                        "separation": sep,
                        "orb": dev,
                        "applying": applying,
                    })
    return out

def _find_antiscia_aspects(
    pa: List[Position],
    pb: List[Position],
    orbs: Dict[str, float],
) -> List[Aspect]:
    out: List[Aspect] = []
    orb = orbs.get("antiscia", DEFAULT_ORBS["antiscia"])
    if orb <= 0:
        return out
    for a in pa:
        a_ref = _antiscia_point(a["longitude"])
        for b in pb:
            sep0 = _angdiff(a_ref, b["longitude"])
            if sep0 <= orb:
                out.append({
                    "planet_a": a["name"],
                    "planet_b": b["name"],
                    "aspect": "antiscia",
                    "angle": 0.0,
                    "separation": sep0,
                    "orb": sep0,
                    "applying": False,
                })
            sep180 = _angdiff(_normalize_angle(a_ref + 180.0), b["longitude"])
            if sep180 <= orb:
                out.append({
                    "planet_a": a["name"],
                    "planet_b": b["name"],
                    "aspect": "contra-antiscia",
                    "angle": 180.0,
                    "separation": sep180,
                    "orb": sep180,
                    "applying": False,
                })
    return out

def _find_declination_aspects(
    pa: List[Position],
    pb: List[Position],
    jd_tt: float,
    orbs: Dict[str, float],
) -> List[Aspect]:
    out: List[Aspect] = []
    orb = orbs.get("parallel", DEFAULT_ORBS["parallel"])
    if orb <= 0:
        return out
    for a in pa:
        dec_a = _ecl_to_decl(a["longitude"], float(a.get("latitude", 0.0)), jd_tt)
        for b in pb:
            dec_b = _ecl_to_decl(b["longitude"], float(b.get("latitude", 0.0)), jd_tt)
            d = abs(dec_a - dec_b)
            if d <= orb:
                out.append({
                    "planet_a": a["name"],
                    "planet_b": b["name"],
                    "aspect": "parallel",
                    "angle": 0.0,
                    "separation": d,
                    "orb": d,
                    "applying": False,
                })
            d2 = abs(dec_a + dec_b)  # declinations opposite sign, same magnitude
            if d2 <= orb:
                out.append({
                    "planet_a": a["name"],
                    "planet_b": b["name"],
                    "aspect": "contra-parallel",
                    "angle": 180.0,
                    "separation": d2,
                    "orb": d2,
                    "applying": False,
                })
    return out

def _find_house_number(longitude: float, cusps: List[float]) -> int:
    """Find house index (1..12) from cusps (deg)."""
    lon = _normalize_angle(longitude)
    if not isinstance(cusps, list) or len(cusps) != 12:
        return 1
    for i in range(12):
        c0 = _normalize_angle(cusps[i])
        c1 = _normalize_angle(cusps[(i + 1) % 12])
        if c0 <= c1:
            if c0 <= lon < c1:
                return i + 1
        else:  # wrap
            if lon >= c0 or lon < c1:
                return i + 1
    return 1

def _calc_overlays(positions: List[Position], houses: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    overlays: List[Dict[str, Any]] = []
    if not houses or "cusps_deg" not in houses:
        return overlays
    cusps = houses["cusps_deg"]
    if not isinstance(cusps, list) or len(cusps) != 12:
        return overlays
    for p in positions:
        overlays.append({
            "planet": p["name"],
            "longitude": p["longitude"],
            "house": _find_house_number(p["longitude"], cusps),
        })
    return overlays

def _calc_midpoints(pa: List[Position], pb: List[Position]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    b_map = {p["name"]: p for p in pb}
    for a in pa:
        b = b_map.get(a["name"])
        if b:
            out.append({"planet": a["name"], "longitude": _circ_mid(a["longitude"], b["longitude"])})
    return out

def _score_aspects(aspects: List[Aspect]) -> Tuple[float, Dict[str, float]]:
    total = 0.0
    by: Dict[str, float] = {}
    for asp in aspects:
        kind = asp["aspect"]
        w = ASPECT_WEIGHTS.get(kind, 0.0)
        max_orb = DEFAULT_ORBS.get(kind, 1.0)
        tightness = max(0.0, 1.0 - (asp["orb"] / max_orb))
        s = w * tightness
        total += s
        by[kind] = by.get(kind, 0.0) + s
    return total, by

# ═══════════════════════════════ SYN ASTRY CORE ══════════════════════════════════

def _do_synastry(
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
    house_system: str = "placidus",
    orbs: Optional[Dict[str, float]] = None,
    parallels: bool = True,
    antiscia: bool = True,
) -> Dict[str, Any]:
    """
    Western synastry core (tropical-only).
    Returns dict with aspects/overlays/midpoints/scores/meta.
    """
    warnings_all: List[str] = []
    try:
        eff_orbs = {**DEFAULT_ORBS, **(orbs or {})}

        # Resolve timescales with strict validation semantics
        try:
            jd_tt_a_r, jd_ut1_a_r, wa = _resolve_timescales(natal_a, jd_tt_a, jd_ut1_a)
            jd_tt_b_r, jd_ut1_b_r, wb = _resolve_timescales(natal_b, jd_tt_b, jd_ut1_b)
        except ValueError as ve:
            return {"ok": False, "error": "validation_error", "details": str(ve), "warnings": warnings_all}
        warnings_all.extend(wa); warnings_all.extend(wb)

        # Places (fallback to natal dicts)
        place_a_final = place_a if place_a else natal_a
        place_b_final = place_b if place_b else natal_b

        # Positions
        pos_a, wpa = _get_positions(jd_tt_a_r, place_a_final, frame, list(MAJOR_BODIES))
        pos_b, wpb = _get_positions(jd_tt_b_r, place_b_final, frame, list(MAJOR_BODIES))
        warnings_all.extend(wpa); warnings_all.extend(wpb)

        # Houses (optional)
        houses_a, wha = _compute_houses(jd_tt_a_r, jd_ut1_a_r, place_a_final, house_system)
        houses_b, whb = _compute_houses(jd_tt_b_r, jd_ut1_b_r, place_b_final, house_system)
        warnings_all.extend(wha); warnings_all.extend(whb)

        # Aspects
        ab = _find_zodiacal_aspects(pos_a, pos_b, eff_orbs)
        ba = _find_zodiacal_aspects(pos_b, pos_a, eff_orbs)
        aa = _find_zodiacal_aspects(pos_a, pos_a, eff_orbs)
        bb = _find_zodiacal_aspects(pos_b, pos_b, eff_orbs)

        if antiscia:
            ab += _find_antiscia_aspects(pos_a, pos_b, eff_orbs)
            ba += _find_antiscia_aspects(pos_b, pos_a, eff_orbs)

        if parallels:
            ab += _find_declination_aspects(pos_a, pos_b, jd_tt_a_r, eff_orbs)
            ba += _find_declination_aspects(pos_b, pos_a, jd_tt_b_r, eff_orbs)

        # Overlays
        overlays_a_in_b = _calc_overlays(pos_a, houses_b)
        overlays_b_in_a = _calc_overlays(pos_b, houses_a)

        # Midpoints & score
        mids = _calc_midpoints(pos_a, pos_b)
        all_aspects = ab + ba
        total_score, by_aspect = _score_aspects(all_aspects)

        return {
            "ok": True,
            "meta": {
                "frame": frame,
                "zodiac_mode": "tropical",
                "ayanamsa_deg": 0.0,
                "house_system": house_system,
                "orbs_used": eff_orbs,
                "warnings": warnings_all,
                "timescales": {
                    "chart_a": {"jd_tt": jd_tt_a_r, "jd_ut1": jd_ut1_a_r},
                    "chart_b": {"jd_tt": jd_tt_b_r, "jd_ut1": jd_ut1_b_r},
                },
            },
            "aspects": {
                "A_to_B": ab,
                "B_to_A": ba,
                "intra_A": aa,
                "intra_B": bb,
            },
            "overlays": {
                "A_in_B": overlays_a_in_b,
                "B_in_A": overlays_b_in_a,
            },
            "midpoints": mids,
            "scores": {
                "total": total_score,
                "by_aspect": by_aspect,
                "heuristic_warning": "Scores are heuristic only, not predictive",
            },
        }
    except Exception as e:
        return {"ok": False, "error": "synastry_computation_failed", "details": str(e), "warnings": warnings_all}

# ═══════════════════════════════ COMPOSITE CORE ══════════════════════════════════

def _do_composite(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    *,
    method: str = "midpoint",          # "midpoint" or "davison"
    jd_tt_ref: Optional[float] = None, # reserved (unused)
    jd_ut1_ref: Optional[float] = None,# reserved (unused)
    place_ref: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
) -> Dict[str, Any]:
    warnings_all: List[str] = []

    if method not in ("midpoint", "davison"):
        return {"ok": False, "error": "composite_value_error", "details": f"Unknown composite method: {method}", "warnings": warnings_all}

    try:
        # Resolve timescales for both charts
        jd_tt_a, jd_ut1_a, wa = _resolve_timescales(natal_a, None, None)
        jd_tt_b, jd_ut1_b, wb = _resolve_timescales(natal_b, None, None)
        warnings_all.extend(wa); warnings_all.extend(wb)

        if method == "midpoint":
            # Positions at respective births; midpoint per same-name body
            pos_a, wpa = _get_positions(jd_tt_a, natal_a, frame, list(MAJOR_BODIES))
            pos_b, wpb = _get_positions(jd_tt_b, natal_b, frame, list(MAJOR_BODIES))
            warnings_all.extend(wpa); warnings_all.extend(wpb)

            positions: Dict[str, float] = {}
            b_map = {p["name"]: p for p in pos_b}
            for a in pos_a:
                b = b_map.get(a["name"])
                if b:
                    positions[a["name"]] = _circ_mid(a["longitude"], b["longitude"])

            # Houses midpoint if separately available
            houses_a, wha = _compute_houses(jd_tt_a, jd_ut1_a, natal_a, house_system)
            houses_b, whb = _compute_houses(jd_tt_b, jd_ut1_b, natal_b, house_system)
            warnings_all.extend(wha); warnings_all.extend(whb)

            asc = mc = None
            cusps = None
            if houses_a and houses_b:
                if "asc_deg" in houses_a and "asc_deg" in houses_b:
                    asc = _circ_mid(houses_a["asc_deg"], houses_b["asc_deg"])
                if "mc_deg" in houses_a and "mc_deg" in houses_b:
                    mc = _circ_mid(houses_a["mc_deg"], houses_b["mc_deg"])
                if "cusps_deg" in houses_a and "cusps_deg" in houses_b:
                    ca = houses_a["cusps_deg"]; cb = houses_b["cusps_deg"]
                    if isinstance(ca, list) and isinstance(cb, list) and len(ca) == 12 and len(cb) == 12:
                        cusps = [_circ_mid(a, b) for a, b in zip(ca, cb)]

        else:  # Davison: midpoint time/place, then compute positions/houses there
            jd_tt_mid = (jd_tt_a + jd_tt_b) / 2.0
            jd_ut1_mid = (jd_ut1_a + jd_ut1_b) / 2.0

            place_mid: Optional[Dict[str, Any]] = None
            if all(k in natal_a for k in ("latitude", "longitude")) and all(k in natal_b for k in ("latitude", "longitude")):
                place_mid = {
                    "latitude": (float(natal_a["latitude"]) + float(natal_b["latitude"])) / 2.0,
                    "longitude": (float(natal_a["longitude"]) + float(natal_b["longitude"])) / 2.0,
                    "elev_m": (float(natal_a.get("elev_m", 0.0)) + float(natal_b.get("elev_m", 0.0))) / 2.0,
                }
            elif place_ref:
                place_mid = place_ref
            else:
                warnings_all.append("davison_no_coordinates")

            pos, wp = _get_positions(jd_tt_mid, place_mid, frame, list(MAJOR_BODIES))
            warnings_all.extend(wp)
            positions = {p["name"]: p["longitude"] for p in pos}

            houses, wh = _compute_houses(jd_tt_mid, jd_ut1_mid, place_mid, house_system)
            warnings_all.extend(wh)
            asc = houses.get("asc_deg") if houses else None
            mc = houses.get("mc_deg") if houses else None
            cusps = houses.get("cusps_deg") if houses else None

        return {
            "ok": True,
            "method": method,
            "positions": positions,
            "asc": asc,
            "mc": mc,
            "cusps": cusps,
            "meta": {
                "frame": frame,
                "zodiac_mode": "tropical",
                "ayanamsa_deg": 0.0,
                "house_system": house_system,
                "warnings": warnings_all,
            },
        }

    except Exception as e:
        return {"ok": False, "error": "composite_computation_failed", "details": str(e), "warnings": warnings_all}

# ═══════════════════════════════ PUBLIC WESTERN API ══════════════════════════════

def compute_western_synastry(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Western (tropical) synastry — always tropical.
    Any 'zodiac_mode' or 'ayanamsa' kwargs are ignored.
    """
    # Strip any sidereal/ayanamsa hints to avoid confusion
    kwargs = dict(kwargs)
    kwargs.pop("zodiac_mode", None)
    kwargs.pop("ayanamsa_deg", None)
    return _do_synastry(natal_a, natal_b, **kwargs)

def compute_western_composite(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Western composite chart (midpoint or Davison) — always tropical.
    Valid kwargs include: method, frame, house_system, place_ref.
    """
    kwargs = dict(kwargs)
    kwargs.pop("zodiac_mode", None)
    kwargs.pop("ayanamsa_deg", None)
    return _do_composite(natal_a, natal_b, **kwargs)

def western_synastry_report(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Combined Western synastry + composite report.
    If composite fails, returns synastry with composite error attached.
    """
    try:
        syn = compute_western_synastry(natal_a, natal_b, **kwargs)
        if not syn.get("ok", False):
            return syn
        comp = compute_western_composite(natal_a, natal_b, **kwargs)
        syn["composite"] = comp

        # Metrics
        ab = len(syn.get("aspects", {}).get("A_to_B", []))
        ba = len(syn.get("aspects", {}).get("B_to_A", []))
        syn.setdefault("metrics", {})
        syn["metrics"].update({
            "total_aspects": ab + ba,
            "compatibility_score": syn.get("scores", {}).get("total", 0.0),
            "report_generated": True,
        })

        # Merge warnings
        all_ws = list(syn.get("meta", {}).get("warnings", []))
        if isinstance(comp, dict) and comp.get("meta", {}).get("warnings"):
            all_ws.extend([f"composite_{w}" for w in comp["meta"]["warnings"]])
        syn["meta"]["warnings"] = all_ws
        syn["meta"]["report_type"] = "western_synastry_report"

        return syn

    except Exception as e:
        return {"ok": False, "error": "synastry_report_failed", "details": str(e)}
