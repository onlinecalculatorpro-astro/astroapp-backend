# app/core/synastry.py
# -*- coding: utf-8 -*-
"""
Research-grade Synastry & Composite module (v12.1)

Goals
-----
- Stable, explicit error taxonomy:
  * validation_error                → input/schema issues (e.g., missing date/time/place_tz)
  * synastry_computation_failed     → unexpected runtime error in synastry
  * composite_value_error           → invalid composite method
  * composite_computation_failed    → unexpected runtime error in composite
- Consistent warnings and meta blocks
- Defensive ephemeris & houses calls with graceful degradation
- Sidereal adjustments (ayanamsa) supported
- Heuristic scoring preserved

Public APIs
-----------
compute_synastry(natal_a, natal_b, **kwargs) -> dict
compute_composite(natal_a, natal_b, **kwargs) -> dict
synastry_report(natal_a, natal_b, **kwargs) -> dict
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

# ═══════════════════════════════ RESILIENT IMPORTS ═══════════════════════════════

# Ephemeris adapter (required)
try:
    from app.core.ephemeris_adapter import EphemerisAdapter
    _EPHEMERIS_AVAILABLE = True
    _EPHEMERIS_ERROR = None
except Exception as e:  # pragma: no cover
    EphemerisAdapter = None  # type: ignore
    _EPHEMERIS_AVAILABLE = False
    _EPHEMERIS_ERROR = e

# Timescales builder (required)
try:
    from app.core.timescales import build_timescales
    _TIMESCALES_AVAILABLE = True
    _TIMESCALES_ERROR = None
except Exception as e:  # pragma: no cover
    build_timescales = None  # type: ignore
    _TIMESCALES_AVAILABLE = False
    _TIMESCALES_ERROR = e

# Houses computation (optional for overlays)
try:
    from app.core.houses import compute_houses_with_policy
    _HOUSES_AVAILABLE = True
    _HOUSES_ERROR = None
except Exception as e:  # pragma: no cover
    compute_houses_with_policy = None  # type: ignore
    _HOUSES_AVAILABLE = False
    _HOUSES_ERROR = e

# ═══════════════════════════════ CONSTANTS ═══════════════════════════════════════

MAJOR_BODIES: Tuple[str, ...] = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"
)

DEFAULT_ORBS: Dict[str, float] = {
    "conjunction": 8.0,
    "opposition": 6.0,
    "trine": 6.0,
    "square": 5.0,
    "sextile": 3.0,
    "quincunx": 2.0,
    "parallel": 1.0,   # degrees declination
    "antiscia": 2.0,
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
    "antiscia": 1.5,
}

# ═══════════════════════════════ UTILITIES ═══════════════════════════════════════

def _normalize_angle(degrees: float) -> float:
    """Normalize angle to [0, 360)."""
    return degrees % 360.0


def _angular_separation(a: float, b: float) -> float:
    """Shortest angular separation between two angles."""
    diff = abs(_normalize_angle(b) - _normalize_angle(a))
    return min(diff, 360.0 - diff)


def _circular_midpoint(a: float, b: float) -> float:
    """Circular midpoint on a circle."""
    a_norm = _normalize_angle(a)
    b_norm = _normalize_angle(b)
    diff = _normalize_angle(b_norm - a_norm)
    if diff <= 180.0:
        return _normalize_angle(a_norm + diff * 0.5)
    return _normalize_angle(a_norm - (360.0 - diff) * 0.5)


def _antiscia_point(longitude: float) -> float:
    """Antiscia point (reflection across 0° Cancer)."""
    return _normalize_angle(180.0 - longitude)


def _mean_obliquity(jd_tt: float) -> float:
    """Mean obliquity (IAU 2006 approx) in degrees."""
    T = (jd_tt - 2451545.0) / 36525.0
    eps0 = 84381.406 - 46.836769*T - 0.0001831*T*T + 0.00200340*T*T*T
    return eps0 / 3600.0


def _ecliptic_to_declination(longitude: float, latitude: float, jd_tt: float) -> float:
    """Convert ecliptic (λ,β) to declination δ (deg)."""
    epsilon = math.radians(_mean_obliquity(jd_tt))
    lam = math.radians(longitude)
    beta = math.radians(latitude)
    sin_dec = math.sin(beta) * math.cos(epsilon) + math.cos(beta) * math.sin(epsilon) * math.sin(lam)
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
      RuntimeError: when timescales machinery is unavailable or fails
    """
    warnings: List[str] = []

    # Strict path: use provided timescales
    if jd_tt is not None and jd_ut1 is not None:
        return float(jd_tt), float(jd_ut1), warnings

    if not _TIMESCALES_AVAILABLE:
        raise RuntimeError(f"Timescales builder unavailable: {_TIMESCALES_ERROR}")

    date_str = natal.get("date")
    time_str = natal.get("time")
    tz_name = natal.get("place_tz")

    if not (date_str and time_str and tz_name):
        # Explicit validation error for test harness and clients
        raise ValueError("missing date/time/place_tz in natal data")

    try:
        ts = build_timescales(str(date_str), str(time_str), str(tz_name), 0.0)  # DUT1=0s assumption
        warnings.append("timescales_computed_with_dut1_0")
        return float(ts.jd_tt), float(ts.jd_ut1), warnings
    except Exception as e:
        raise RuntimeError(f"Failed to resolve timescales: {e}") from e


def _get_planet_positions(
    jd_tt: float,
    place: Optional[Dict[str, Any]],
    frame: str,
    bodies: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Query planetary positions via EphemerisAdapter.
    Returns: (positions, warnings) where positions is a list of dicts:
      {"name":<str>, "longitude":<float>, "latitude":<float>, ["speed":<float>]}
    """
    warnings: List[str] = []
    if not _EPHEMERIS_AVAILABLE:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPHEMERIS_ERROR}")

    adapter = EphemerisAdapter(frame=frame)

    # Topo vs geo
    if place and all(k in place for k in ("latitude", "longitude")):
        center = "topocentric"
        kwargs = {
            "jd_tt": float(jd_tt),
            "bodies": bodies,
            "center": center,
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
            "elevation_m": float(place.get("elev_m", 0.0)),
        }
    else:
        center = "geocentric"
        kwargs = {"jd_tt": float(jd_tt), "bodies": bodies, "center": center}
        if place is None:
            warnings.append("geocentric_no_coordinates")

    result: Any = None

    # Prefer modern batch method with velocities
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
    positions: List[Dict[str, Any]] = []
    if isinstance(result, dict):
        if "results" in result and isinstance(result["results"], list):
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
            # Simple dict: { "Sun": 123.45, ... }
            for name, value in result.items():
                if isinstance(value, (int, float)):
                    positions.append({"name": name, "longitude": float(value), "latitude": 0.0})
    return positions, warnings


def _compute_houses(
    jd_tt: float,
    jd_ut1: float,
    place: Optional[Dict[str, Any]],
    house_system: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Compute houses (optional). Returns (houses_dict|None, warnings)
    houses_dict keys typically: asc_deg, mc_deg, cusps_deg (list of 12)
    """
    warnings: List[str] = []

    if not place or not all(k in place for k in ("latitude", "longitude")):
        warnings.append("houses_no_coordinates")
        return None, warnings

    if not _HOUSES_AVAILABLE:
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


def _find_aspects(
    positions_a: List[Dict[str, Any]],
    positions_b: List[Dict[str, Any]],
    orbs: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Zodiacal aspects between two sets of positions."""
    out: List[Dict[str, Any]] = []
    for pa in positions_a:
        for pb in positions_b:
            sep = _angular_separation(pa["longitude"], pb["longitude"])
            for name, angle in ASPECT_ANGLES.items():
                orb = orbs.get(name, DEFAULT_ORBS.get(name, 0.0))
                if orb <= 0:
                    continue
                dev = abs(sep - angle)
                if dev <= orb:
                    out.append({
                        "planet_a": pa["name"],
                        "planet_b": pb["name"],
                        "aspect": name,
                        "angle": angle,
                        "separation": sep,
                        "orb": dev,
                        "applying": sep < angle,  # heuristic
                    })
    return out


def _find_antiscia_aspects(
    positions_a: List[Dict[str, Any]],
    positions_b: List[Dict[str, Any]],
    orbs: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Antiscia & contra-antiscia aspects."""
    out: List[Dict[str, Any]] = []
    orb = orbs.get("antiscia", DEFAULT_ORBS["antiscia"])
    if orb <= 0:
        return out

    for pa in positions_a:
        a_ref = _antiscia_point(pa["longitude"])
        for pb in positions_b:
            sep0 = _angular_separation(a_ref, pb["longitude"])
            if sep0 <= orb:
                out.append({
                    "planet_a": pa["name"],
                    "planet_b": pb["name"],
                    "aspect": "antiscia",
                    "angle": 0.0,
                    "separation": sep0,
                    "orb": sep0,
                    "applying": False,
                })
            sep180 = _angular_separation(a_ref, pb["longitude"] + 180.0)
            if sep180 <= orb:
                out.append({
                    "planet_a": pa["name"],
                    "planet_b": pb["name"],
                    "aspect": "contra-antiscia",
                    "angle": 180.0,
                    "separation": sep180,
                    "orb": sep180,
                    "applying": False,
                })
    return out


def _find_parallel_aspects(
    positions_a: List[Dict[str, Any]],
    positions_b: List[Dict[str, Any]],
    jd_tt: float,
    orbs: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Declination parallel & contra-parallel aspects."""
    out: List[Dict[str, Any]] = []
    orb = orbs.get("parallel", DEFAULT_ORBS["parallel"])
    if orb <= 0:
        return out

    for pa in positions_a:
        dec_a = _ecliptic_to_declination(pa["longitude"], pa.get("latitude", 0.0), jd_tt)
        for pb in positions_b:
            dec_b = _ecliptic_to_declination(pb["longitude"], pb.get("latitude", 0.0), jd_tt)
            d = abs(dec_a - dec_b)
            if d <= orb:
                out.append({
                    "planet_a": pa["name"],
                    "planet_b": pb["name"],
                    "aspect": "parallel",
                    "angle": 0.0,
                    "separation": d,
                    "orb": d,
                    "applying": False,
                })
            d2 = abs(dec_a + dec_b)
            if d2 <= orb:
                out.append({
                    "planet_a": pa["name"],
                    "planet_b": pb["name"],
                    "aspect": "contra-parallel",
                    "angle": 180.0,
                    "separation": d2,
                    "orb": d2,
                    "applying": False,
                })
    return out


def _calculate_house_overlays(
    positions: List[Dict[str, Any]],
    houses: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Which houses planets fall into."""
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


def _find_house_number(longitude: float, cusps: List[float]) -> int:
    """Find house index (1..12)."""
    lon = _normalize_angle(longitude)
    for i in range(12):
        c0 = _normalize_angle(cusps[i])
        c1 = _normalize_angle(cusps[(i + 1) % 12])
        if c0 <= c1:
            if c0 <= lon < c1:
                return i + 1
        else:  # wrap
            if lon >= c0 or lon < c1:
                return i + 1
    return 1  # fallback


def _calculate_midpoints(
    positions_a: List[Dict[str, Any]],
    positions_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Planet-wise midpoints (same-name bodies)."""
    out: List[Dict[str, Any]] = []
    b_by_name = {p["name"]: p for p in positions_b}
    for pa in positions_a:
        pb = b_by_name.get(pa["name"])
        if pb:
            out.append({"planet": pa["name"], "longitude": _circular_midpoint(pa["longitude"], pb["longitude"])})
    return out


def _score_aspects(aspects: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """Heuristic compatibility score."""
    total = 0.0
    by = {}
    for asp in aspects:
        kind = asp["aspect"]
        w = ASPECT_WEIGHTS.get(kind, 0.0)
        max_orb = DEFAULT_ORBS.get(kind, 1.0)
        tightness = max(0.0, 1.0 - (asp["orb"] / max_orb))
        s = w * tightness
        total += s
        by[kind] = by.get(kind, 0.0) + s
    return total, by

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
    Returns dict with aspects/overlays/midpoints/scores/meta.
    """
    all_warnings: List[str] = []
    try:
        # Orbs
        effective_orbs = {**DEFAULT_ORBS, **(orbs or {})}

        # Timescales (with validation semantics)
        try:
            jd_tt_a_r, jd_ut1_a_r, wa = _resolve_timescales(natal_a, jd_tt_a, jd_ut1_a)
            jd_tt_b_r, jd_ut1_b_r, wb = _resolve_timescales(natal_b, jd_tt_b, jd_ut1_b)
        except ValueError as ve:
            # Explicit input error
            return {
                "ok": False,
                "error": "validation_error",
                "details": str(ve),
                "warnings": all_warnings,
            }
        all_warnings.extend(wa)
        all_warnings.extend(wb)

        # Places (fallback to natal dict)
        place_a_final = place_a if place_a else natal_a
        place_b_final = place_b if place_b else natal_b

        # Positions
        pos_a, wpa = _get_planet_positions(jd_tt_a_r, place_a_final, frame, list(MAJOR_BODIES))
        pos_b, wpb = _get_planet_positions(jd_tt_b_r, place_b_final, frame, list(MAJOR_BODIES))
        all_warnings.extend(wpa)
        all_warnings.extend(wpb)

        # Sidereal shift if requested
        if zodiac_mode.lower() == "sidereal" and abs(ayanamsa_deg) > 1e-9:
            for p in pos_a + pos_b:
                p["longitude"] = _normalize_angle(p["longitude"] - ayanamsa_deg)

        # Houses (optional)
        houses_a, wha = _compute_houses(jd_tt_a_r, jd_ut1_a_r, place_a_final, house_system)
        houses_b, whb = _compute_houses(jd_tt_b_r, jd_ut1_b_r, place_b_final, house_system)
        all_warnings.extend(wha)
        all_warnings.extend(whb)

        # Aspects
        ab = _find_aspects(pos_a, pos_b, effective_orbs)
        ba = _find_aspects(pos_b, pos_a, effective_orbs)
        aa = _find_aspects(pos_a, pos_a, effective_orbs)
        bb = _find_aspects(pos_b, pos_b, effective_orbs)

        if antiscia:
            ab += _find_antiscia_aspects(pos_a, pos_b, effective_orbs)
            ba += _find_antiscia_aspects(pos_b, pos_a, effective_orbs)

        if parallels:
            ab += _find_parallel_aspects(pos_a, pos_b, jd_tt_a_r, effective_orbs)
            ba += _find_parallel_aspects(pos_b, pos_a, jd_tt_b_r, effective_orbs)

        # Overlays
        overlays_a_in_b = _calculate_house_overlays(pos_a, houses_b)
        overlays_b_in_a = _calculate_house_overlays(pos_b, houses_a)

        # Midpoints & score
        mids = _calculate_midpoints(pos_a, pos_b)
        all_aspects = ab + ba
        total_score, by_aspect = _score_aspects(all_aspects)

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
    jd_tt_ref: Optional[float] = None,   # reserved (not required here)
    jd_ut1_ref: Optional[float] = None,  # reserved (not required here)
    place_ref: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    ayanamsa_deg: float = 0.0,
    zodiac_mode: str = "tropical",
) -> Dict[str, Any]:
    """
    Compute composite chart using 'midpoint' or 'davison'.
    Error taxonomy:
      - composite_value_error for bad 'method'
      - composite_computation_failed for unexpected errors
    """
    all_warnings: List[str] = []

    # Input validation: method
    if method not in ("midpoint", "davison"):
        return {
            "ok": False,
            "error": "composite_value_error",
            "details": f"Unknown composite method: {method}",
            "warnings": all_warnings,
        }

    try:
        # Resolve timescales for both charts (validation semantics: ValueError -> validation_error not used here)
        jd_tt_a, jd_ut1_a, wa = _resolve_timescales(natal_a, None, None)
        jd_tt_b, jd_ut1_b, wb = _resolve_timescales(natal_b, None, None)
        all_warnings.extend(wa)
        all_warnings.extend(wb)

        if method == "midpoint":
            # Positions at their respective births, then midpoint longitudes per body
            pos_a, wpa = _get_planet_positions(jd_tt_a, natal_a, frame, list(MAJOR_BODIES))
            pos_b, wpb = _get_planet_positions(jd_tt_b, natal_b, frame, list(MAJOR_BODIES))
            all_warnings.extend(wpa)
            all_warnings.extend(wpb)

            if zodiac_mode.lower() == "sidereal" and abs(ayanamsa_deg) > 1e-9:
                for p in pos_a + pos_b:
                    p["longitude"] = _normalize_angle(p["longitude"] - ayanamsa_deg)

            positions: Dict[str, float] = {}
            for pa in pos_a:
                for pb in pos_b:
                    if pa["name"] == pb["name"]:
                        positions[pa["name"]] = _circular_midpoint(pa["longitude"], pb["longitude"])
                        break

            # Houses midpoint if available
            houses_a, wha = _compute_houses(jd_tt_a, jd_ut1_a, natal_a, house_system)
            houses_b, whb = _compute_houses(jd_tt_b, jd_ut1_b, natal_b, house_system)
            all_warnings.extend(wha)
            all_warnings.extend(whb)

            asc = mc = None
            cusps = None
            if houses_a and houses_b:
                if "asc_deg" in houses_a and "asc_deg" in houses_b:
                    asc = _circular_midpoint(houses_a["asc_deg"], houses_b["asc_deg"])
                if "mc_deg" in houses_a and "mc_deg" in houses_b:
                    mc = _circular_midpoint(houses_a["mc_deg"], houses_b["mc_deg"])
                if "cusps_deg" in houses_a and "cusps_deg" in houses_b:
                    ca = houses_a["cusps_deg"]; cb = houses_b["cusps_deg"]
                    if isinstance(ca, list) and isinstance(cb, list) and len(ca) == 12 and len(cb) == 12:
                        cusps = [_circular_midpoint(a, b) for a, b in zip(ca, cb)]

        else:  # davison
            # Midpoint time
            jd_tt_mid = (jd_tt_a + jd_tt_b) / 2.0
            jd_ut1_mid = (jd_ut1_a + jd_ut1_b) / 2.0

            # Midpoint place: average lat/lon if both available, else use place_ref if provided
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
                all_warnings.append("davison_no_coordinates")

            pos, wp = _get_planet_positions(jd_tt_mid, place_mid, frame, list(MAJOR_BODIES))
            all_warnings.extend(wp)

            if zodiac_mode.lower() == "sidereal" and abs(ayanamsa_deg) > 1e-9:
                for p in pos:
                    p["longitude"] = _normalize_angle(p["longitude"] - ayanamsa_deg)

            positions = {p["name"]: p["longitude"] for p in pos}

            houses, wh = _compute_houses(jd_tt_mid, jd_ut1_mid, place_mid, house_system)
            all_warnings.extend(wh)
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
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Combined synastry + composite report.
    - If composite fails, report still returns synastry with composite error payload attached.
    """
    try:
        syn = compute_synastry(natal_a, natal_b, **kwargs)
        if not syn.get("ok", False):
            return syn

        comp_kwargs = {k: v for k, v in kwargs.items() if k in ("frame", "house_system", "ayanamsa_deg", "zodiac_mode")}
        comp_kwargs.update({"method": composite_method, "place_ref": composite_place_ref})

        comp = compute_composite(natal_a, natal_b, **comp_kwargs)
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
        syn["meta"]["report_type"] = "comprehensive"

        return syn

    except Exception as e:
        return {
            "ok": False,
            "error": "synastry_report_failed",
            "details": str(e),
        }
