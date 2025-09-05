# app/core/synastry.py
# -*- coding: utf-8 -*-
"""
Research-grade Synastry & Composite module (v11)

Public APIs
-----------
compute_synastry(
    natal_a, natal_b, *,
    jd_tt_a=None, jd_ut1_a=None,
    jd_tt_b=None, jd_ut1_b=None,
    place_a=None, place_b=None,
    frame="ecliptic-of-date",
    ayanamsa_deg=0.0,
    zodiac_mode="tropical",
    house_system="placidus",
    orbs=None,
    parallels=True,
    antiscia=True,
) -> dict

compute_composite(
    natal_a, natal_b, *,
    method="midpoint"|"davison",
    jd_tt_ref=None, jd_ut1_ref=None,
    place_ref=None,
    frame="ecliptic-of-date",
    house_system="placidus",
    ayanamsa_deg=0.0,
    zodiac_mode="tropical",
) -> dict

synastry_report(...) -> dict
    Convenience wrapper: bundles synastry + composite + light metrics.

Notes & Conventions
-------------------
- Prefers strict timescales (jd_tt, jd_ut1). If any are missing, resolves via
  app.core.timescales.build_timescales(date, time, tz, dut1_seconds=0.0) and
  reports assumptions in meta.warnings.
- Planet longitudes/velocities via app.core.ephemeris_adapter.EphemerisAdapter.
  Default frame is "ecliptic-of-date". Sidereal option subtracts ayanamsa_deg.
- Houses via app.core.houses.compute_houses_with_policy (ASC/MC/Cusps).
- Aspects: zodiacal (0, 60, 90, 120, 180), plus (optional) antiscia/contra and
  declination parallels/contra-parallels.
- House overlays: uses forward-wrap cusp intervals from houses_advanced policy.
- Scoring is HEURISTIC ONLY and flagged as such.

Data Contracts (minimal natal dicts)
------------------------------------
natal := {
    "date": "YYYY-MM-DD",
    "time": "HH:MM[:SS[.fff]]",
    "place_tz": "Area/City",
    "latitude": float,
    "longitude": float,
    "elev_m": float | 0.0,
    "mode": "tropical" | "sidereal"  # optional; default aligns with zodiac_mode
}

Returned Top-level Keys
-----------------------
{
  "meta": {...},  # frames, timescales, house system, warnings
  "aspects": { "A_to_B": [...], "B_to_A": [...], "intra_A": [...], "intra_B": [...] },
  "overlays": { "A_in_B": [...], "B_in_A": [...] },
  "midpoints": { "planets": [...], "axes": {...} },
  "composite": { "method": "...", "positions": {...}, "asc": float, "mc": float, "cusps": [..] },
  "scores": { "heuristic": true, "total": float, "breakdown": {...} }
}

Implementation carefully avoids I/O/HTTP; returns JSON-serializable dicts only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Iterable
from dataclasses import dataclass
import math
import inspect

# ───────────────────────────── Resilient imports ─────────────────────────────
try:
    from app.core.ephemeris_adapter import EphemerisAdapter
except Exception as _e:
    EphemerisAdapter = None  # type: ignore
    _EPH_ADAPTER_IMPORT_ERROR = _e

try:
    # Houses policy façade (preferred strict path)
    from app.core.houses import compute_houses_with_policy as _compute_houses_policy
except Exception as _e:
    _compute_houses_policy = None
    _HOUSES_IMPORT_ERROR = _e

try:
    # Timescales helper (locked API per project notes)
    from app.core.timescales import build_timescales
except Exception as _e:
    build_timescales = None  # type: ignore
    _TS_IMPORT_ERROR = _e

try:
    # If an aspects engine exists, we’ll prefer it; otherwise we use fallback.
    from app.core import aspects as _aspects
except Exception:
    _aspects = None


# ───────────────────────────── Constants & Defaults ──────────────────────────

MAJORS: Tuple[str, ...] = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
)
AXES: Tuple[str, ...] = ("ASC", "MC")

DEFAULT_ORBS: Dict[str, float] = {
    "conjunction": 8.0,
    "opposition": 6.0,
    "trine": 6.0,
    "square": 5.0,
    "sextile": 3.0,
    "quincunx": 2.0,  # optional; not in primary score
    # special
    "parallel_arcmin": 40.0,  # declination parallels default orb (arcminutes)
    "antiscia": 2.0,          # orb for antiscia/contra-antiscia
}

ASPECT_ANGLES: Dict[str, float] = {
    "conjunction": 0.0,
    "sextile": 60.0,
    "square": 90.0,
    "trine": 120.0,
    "opposition": 180.0,
    "quincunx": 150.0,  # optional
}

ASPECT_WEIGHTS: Dict[str, float] = {
    # purely heuristic; flagged in meta
    "conjunction": 5.0,
    "trine": 4.0,
    "sextile": 2.5,
    "square": -3.5,
    "opposition": -4.0,
    "quincunx": -1.0,
    "antiscia": 2.0,          # mild positive
    "contra-antiscia": -2.0,  # mild tension
    "parallel": 2.0,
    "contra-parallel": -2.0,
}


# ───────────────────────────── Utilities ─────────────────────────────────────

def _wrap_deg(x: float) -> float:
    x = math.fmod(x, 360.0)
    return x + 360.0 if x < 0.0 else x

def _delta_deg(a: float, b: float) -> float:
    """Smallest signed separation a→b in degrees (range [-180, +180])."""
    d = _wrap_deg(b) - _wrap_deg(a)
    if d > 180.0:
        d -= 360.0
    elif d < -180.0:
        d += 360.0
    return d

def _abs_sep(a: float, b: float) -> float:
    return abs(_delta_deg(a, b))

def _circ_mid(a: float, b: float) -> float:
    """Circular midpoint along the shortest arc."""
    d = _delta_deg(a, b)
    return _wrap_deg(a + d * 0.5)

def _degmin_to_deg(arcmin: float) -> float:
    return arcmin / 60.0

def _warn(warnings: List[str], msg: str) -> None:
    if msg not in warnings:
        warnings.append(msg)

def _mean_obliquity_iau2006(jd_tt: float) -> float:
    """Mean obliquity (ε) of date (arcdegrees), IAU 2006 (approx), valid for centuries around J2000."""
    T = (jd_tt - 2451545.0) / 36525.0
    # arcseconds
    eps0 = 84381.406 \
           - 46.836769*T \
           - 0.0001831*(T**2) \
           + 0.00200340*(T**3) \
           - 0.000000576*(T**4) \
           - 0.0000000434*(T**5)
    return eps0 / 3600.0

def _declination_from_ecliptic(lon_deg: float, lat_deg: float, jd_tt: float) -> float:
    """Convert ecliptic lon/lat to declination (deg) using mean obliquity."""
    eps = math.radians(_mean_obliquity_iau2006(jd_tt))
    lam = math.radians(_wrap_deg(lon_deg))
    beta = math.radians(lat_deg)
    # sin δ = sin β cos ε + cos β sin ε sin λ
    sin_delta = math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_delta))))

def _antiscia_of(lon_deg: float) -> float:
    """
    Antiscia across the solstitial axis (0° Cancer / 0° Capricorn).
    Mapping: λ' = (180° - λ) mod 360°
    Pairs: Aries↔Virgo, Taurus↔Leo, Gemini↔Cancer, Libra↔Pisces, Scorpio↔Aquarius, Sagittarius↔Capricorn.
    """
    return _wrap_deg(180.0 - _wrap_deg(lon_deg))

def _contra_antiscia_sep(lon1: float, lon2: float) -> float:
    """
    For contra-antiscia, reflect lon1 to antiscia and compare to lon2 + 180 (opposition-like).
    Effectively check |Δ(antiscia(lon1), lon2±180)| with smallest separation.
    """
    a1 = _antiscia_of(lon1)
    return min(_abs_sep(a1, lon2 + 180.0), _abs_sep(a1, lon2 - 180.0))

def _planet_rows(
    jd_tt: float,
    place: Optional[Dict[str, float]],
    frame: str,
    bodies: Iterable[str],
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """
    Query ephemeris adapter for ecliptic-of-date longitudes (and lat/speed if available).
    Returns rows: [{"name":<>, "lon":<deg>, "lat":<deg?>, "speed":<deg/day?>}, ...]
    """
    if EphemerisAdapter is None:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPH_ADAPTER_IMPORT_ERROR}")

    adapter = EphemerisAdapter(frame=frame)
    # attempt the most feature-complete path
    kwargs = {
        "jd_tt": jd_tt,
        "bodies": list(bodies),
    }
    if place and all(k in place for k in ("latitude", "longitude", "elev_m")):
        kwargs.update({
            "center": "topocentric",
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
            "elevation_m": float(place.get("elev_m", 0.0)),
        })
    else:
        kwargs["center"] = "geocentric"
        if place is None:
            _warn(warnings, "topocentric_missing_coords")

    rows: List[Dict[str, Any]] = []

    # Try method resolution progressively (adapter may expose different APIs)
    method_candidates = [
        "ecliptic_longitudes_and_velocities",
        "ecliptic_longitudes",
    ]
    got = False
    for m in method_candidates:
        if hasattr(adapter, m):
            fn = getattr(adapter, m)
            try:
                sig = inspect.signature(fn)
                # Filter kwargs by signature
                call_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
                res = fn(**call_args)
                # Adapter is expected to return rows; tolerate maps by converting
                if isinstance(res, dict):
                    # maps => rows
                    for k, v in res.items():
                        if isinstance(v, (int, float)):
                            rows.append({"name": k, "lon": float(v)})
                        elif isinstance(v, dict) and "lon" in v:
                            r = {"name": k, "lon": float(v["lon"])}
                            if "lat" in v: r["lat"] = float(v["lat"])
                            if "speed" in v: r["speed"] = float(v["speed"])
                            rows.append(r)
                    got = True
                    break
                elif isinstance(res, list):
                    # normalize keys
                    for r in res:
                        if not isinstance(r, dict):  # skip junk
                            continue
                        name = str(r.get("name") or r.get("body") or r.get("id") or "?")
                        lon = float(r.get("lon") or r.get("longitude"))
                        row = {"name": name, "lon": lon}
                        if "lat" in r or "latitude" in r:
                            row["lat"] = float(r.get("lat") or r.get("latitude"))
                        if "speed" in r:
                            row["speed"] = float(r["speed"])
                        rows.append(row)
                    got = True
                    break
            except Exception as e:
                # try next candidate
                _warn(warnings, f"ephemeris_method_failed:{m}:{type(e).__name__}")
                continue

    if not got:
        raise RuntimeError("No usable ephemeris method found on EphemerisAdapter.")

    return rows

def _apply_ayanamsa(rows: List[Dict[str, Any]], ayanamsa_deg: float) -> None:
    if abs(ayanamsa_deg) < 1e-12:
        return
    for r in rows:
        r["lon"] = _wrap_deg(float(r["lon"]) - ayanamsa_deg)

def _compute_declinations(rows: List[Dict[str, Any]], jd_tt: float) -> None:
    for r in rows:
        lat = float(r.get("lat", 0.0))
        r["dec"] = _declination_from_ecliptic(float(r["lon"]), lat, jd_tt)

def _resolve_timescales(
    natal: Dict[str, Any],
    jd_tt: Optional[float],
    jd_ut1: Optional[float],
    warnings: List[str],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Return (jd_tt, jd_ut1, meta_timescales). Uses strict preferred; if missing,
    computes via build_timescales(date, time, tz, dut1=0.0) and warns.
    """
    if jd_tt is not None and jd_ut1 is not None:
        return float(jd_tt), float(jd_ut1), {"jd_tt": float(jd_tt), "jd_ut1": float(jd_ut1), "dut1_assumed": None}

    if build_timescales is None:
        raise RuntimeError("Timescales unavailable and strict values not supplied.")

    date_str = natal.get("date")
    time_str = natal.get("time")
    tz_name  = natal.get("place_tz")
    if not (date_str and time_str and tz_name):
        raise ValueError("Missing date/time/place_tz for timescale resolution.")

    # We do not know DUT1 from inputs; assume 0.0s with explicit transparency.
    ts = build_timescales(date_str=str(date_str), time_str=str(time_str), tz_name=str(tz_name), dut1_seconds=0.0)
    _warn(warnings, "strict_missing→computed_timescales_with_dut1=0.0s")
    meta = {
        "jd_tt": float(ts["jd_tt"]),
        "jd_ut1": float(ts["jd_ut1"]),
        "dut1_assumed": 0.0,
    }
    return float(ts["jd_tt"]), float(ts["jd_ut1"]), meta

def _houses_for(
    jd_tt: float, jd_ut1: float,
    place: Optional[Dict[str, float]],
    house_system: str,
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    if _compute_houses_policy is None:
        _warn(warnings, "houses_policy_unavailable")
        return None
    if not place:
        _warn(warnings, "houses_missing_place")
        return None

    try:
        hs = _compute_houses_policy(
            jd_tt=jd_tt, jd_ut1=jd_ut1,
            latitude=float(place["latitude"]),
            longitude=float(place["longitude"]),
            elevation_m=float(place.get("elev_m", 0.0)),
            system=house_system,
        )
        # Expected shape: {"asc_deg": float, "mc_deg": float, "cusps_deg": [12 floats], ...}
        return hs
    except Exception as e:
        _warn(warnings, f"houses_compute_failed:{type(e).__name__}")
        return None

def _to_place(natal: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    src = override if override is not None else natal
    if src is None:
        return None
    if all(k in src for k in ("latitude", "longitude")):
        return {
            "latitude": float(src["latitude"]),
            "longitude": float(src["longitude"]),
            "elev_m": float(src.get("elev_m", 0.0)),
        }
    return None

def _rows_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["name"]): r for r in rows}

def _find_house(lon: float, cusps_deg: List[float]) -> int:
    """Find 1..12 using forward-wrap intervals [cusp[i], cusp[i+1])."""
    if not cusps_deg or len(cusps_deg) < 12:
        return 0
    cusp = [_wrap_deg(x) for x in cusps_deg]
    # ensure 13th for wrap
    cusp13 = cusp + [cusp[0]]
    lon = _wrap_deg(lon)
    for i in range(12):
        a, b = cusp13[i], cusp13[i+1]
        # interval a→b forward
        if a <= b:
            if a <= lon < b:
                return i + 1
        else:
            # wraps over 360
            if lon >= a or lon < b:
                return i + 1
    return 0

def _zodiacal_aspects(
    rows1: List[Dict[str, Any]],
    rows2: List[Dict[str, Any]],
    orbs: Dict[str, float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    planets1 = [r for r in rows1 if r["name"] in MAJORS]
    planets2 = [r for r in rows2 if r["name"] in MAJORS]
    # pairwise
    for a in planets1:
        for b in planets2:
            sep = _abs_sep(a["lon"], b["lon"])
            for name, angle in ASPECT_ANGLES.items():
                orb = orbs.get(name, DEFAULT_ORBS.get(name, 0.0))
                if orb <= 0:  # disabled
                    continue
                tight = abs(sep - angle)
                if tight <= orb:
                    out.append({
                        "a": a["name"], "b": b["name"],
                        "type": name,
                        "exact_deg": angle,
                        "sep_deg": sep,
                        "orb_deg": tight,
                        "mode": "zodiacal",
                    })
    return out

def _antiscia_aspects(
    rows1: List[Dict[str, Any]],
    rows2: List[Dict[str, Any]],
    orbs: Dict[str, float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    orb = orbs.get("antiscia", DEFAULT_ORBS["antiscia"])
    if orb <= 0:
        return out
    planets1 = [r for r in rows1 if r["name"] in MAJORS]
    planets2 = [r for r in rows2 if r["name"] in MAJORS]
    for a in planets1:
        a_anti = _antiscia_of(a["lon"])
        for b in planets2:
            # antiscia ~ conjunction (0°)
            sep0 = _abs_sep(a_anti, b["lon"])
            if sep0 <= orb:
                out.append({
                    "a": a["name"], "b": b["name"], "type": "antiscia",
                    "exact_deg": 0.0, "sep_deg": sep0, "orb_deg": sep0, "mode": "antiscia",
                })
            # contra-antiscia ~ opposition (180°)
            sep180 = min(_abs_sep(a_anti, b["lon"] + 180.0), _abs_sep(a_anti, b["lon"] - 180.0))
            if sep180 <= orb:
                out.append({
                    "a": a["name"], "b": b["name"], "type": "contra-antiscia",
                    "exact_deg": 180.0, "sep_deg": sep180, "orb_deg": sep180, "mode": "antiscia",
                })
    return out

def _parallel_aspects(
    rows1: List[Dict[str, Any]],
    rows2: List[Dict[str, Any]],
    orbs: Dict[str, float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    arcmin = orbs.get("parallel_arcmin", DEFAULT_ORBS["parallel_arcmin"])
    if arcmin <= 0:
        return out
    orb = _degmin_to_deg(arcmin)
    planets1 = [r for r in rows1 if r["name"] in MAJORS and "dec" in r]
    planets2 = [r for r in rows2 if r["name"] in MAJORS and "dec" in r]
    for a in planets1:
        for b in planets2:
            d1, d2 = float(a["dec"]), float(b["dec"])
            if abs(d1 - d2) <= orb:
                out.append({
                    "a": a["name"], "b": b["name"], "type": "parallel",
                    "exact_deg": 0.0, "sep_deg": abs(d1 - d2), "orb_deg": abs(d1 - d2),
                    "mode": "declination",
                })
            if abs(d1 + d2) <= orb:
                out.append({
                    "a": a["name"], "b": b["name"], "type": "contra-parallel",
                    "exact_deg": 0.0, "sep_deg": abs(d1 + d2), "orb_deg": abs(d1 + d2),
                    "mode": "declination",
                })
    return out

def _intra_aspects(rows: List[Dict[str, Any]], orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    planets = [r for r in rows if r["name"] in MAJORS]
    n = len(planets)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = planets[i], planets[j]
            sep = _abs_sep(a["lon"], b["lon"])
            for name, angle in ASPECT_ANGLES.items():
                orb = orbs.get(name, DEFAULT_ORBS.get(name, 0.0))
                if orb <= 0:
                    continue
                tight = abs(sep - angle)
                if tight <= orb:
                    out.append({
                        "a": a["name"], "b": b["name"],
                        "type": name,
                        "exact_deg": angle, "sep_deg": sep, "orb_deg": tight, "mode": "zodiacal",
                    })
    return out

def _score_aspects(aspects: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    total = 0.0
    by_type: Dict[str, float] = {}
    for a in aspects:
        t = a["type"]
        w = ASPECT_WEIGHTS.get(t, 0.0)
        # “tightness” multiplier: 1.0 at exact, tapering to 0 at orb limit (linear)
        # We need the allowed orb for that type (fallback to DEFAULT_ORBS)
        O = DEFAULT_ORBS.get(t, 1.0)
        if t in ("parallel", "contra-parallel"):
            # Use deg from arcmin default
            O = _degmin_to_deg(DEFAULT_ORBS["parallel_arcmin"])
        if t in ("antiscia", "contra-antiscia"):
            O = DEFAULT_ORBS["antiscia"]
        tight = max(0.0, 1.0 - float(a["orb_deg"]) / max(1e-9, float(O)))
        contrib = w * tight
        total += contrib
        by_type[t] = by_type.get(t, 0.0) + contrib
    return total, by_type

def _midpoints_planets(rowsA: List[Dict[str, Any]], rowsB: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idxA = _rows_index(rowsA); idxB = _rows_index(rowsB)
    out: List[Dict[str, Any]] = []
    for name in MAJORS:
        if name in idxA and name in idxB:
            out.append({"name": name, "lon": _circ_mid(idxA[name]["lon"], idxB[name]["lon"])})
    return out

def _axes_from_houses(hs: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not hs:
        return None, None
    return float(hs.get("asc_deg")) if "asc_deg" in hs else None, \
           float(hs.get("mc_deg")) if "mc_deg" in hs else None

def _midpoints_axes(hA: Optional[Dict[str, Any]], hB: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    ascA, mcA = _axes_from_houses(hA)
    ascB, mcB = _axes_from_houses(hB)
    return {
        "ASC": _circ_mid(ascA, ascB) if (ascA is not None and ascB is not None) else None,
        "MC":  _circ_mid(mcA, mcB)  if (mcA  is not None and mcB  is not None) else None,
    }

def _overlay(planets: List[Dict[str, Any]], homes: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not homes:
        return out
    cusps = homes.get("cusps_deg") or []
    for p in planets:
        h = _find_house(p["lon"], cusps)
        # proximity to nearest cusp (optional; simple min distance)
        cusp_min = None
        if cusps:
            dists = [_abs_sep(p["lon"], c) for c in cusps]
            cusp_min = min(dists) if dists else None
        out.append({
            "planet": p["name"], "lon": float(p["lon"]),
            "house": int(h),
            "orb_to_nearest_cusp_deg": float(cusp_min) if cusp_min is not None else None,
        })
    return out


# ───────────────────────────── Core APIs ─────────────────────────────────────

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
    Compute inter-chart aspects (zodiacal + antiscia/contra + declination parallels),
    house overlays (A→B and B→A), planetary/axis midpoints, and heuristic scores.
    """
    warnings: List[str] = []
    _orbs = {**DEFAULT_ORBS, **(orbs or {})}

    # Timescales (strict preferred; otherwise compute with DUT1=0.0s and warn)
    jd_tt_a, jd_ut1_a, tsA = _resolve_timescales(natal_a, jd_tt_a, jd_ut1_a, warnings)
    jd_tt_b, jd_ut1_b, tsB = _resolve_timescales(natal_b, jd_tt_b, jd_ut1_b, warnings)

    # Places (topocentric if available)
    placeA = _to_place(natal_a, place_a)
    placeB = _to_place(natal_b, place_b)

    # Ephemeris rows for both charts
    rowsA = _planet_rows(jd_tt_a, placeA, frame, MAJORS, warnings)
    rowsB = _planet_rows(jd_tt_b, placeB, frame, MAJORS, warnings)

    # Sidereal mode via ayanamsa offset
    _zmode = (natal_a.get("mode") or natal_b.get("mode") or zodiac_mode or "tropical").lower()
    if _zmode == "sidereal":
        _apply_ayanamsa(rowsA, ayanamsa_deg)
        _apply_ayanamsa(rowsB, ayanamsa_deg)

    # Declinations for parallels
    _compute_declinations(rowsA, jd_tt_a)
    _compute_declinations(rowsB, jd_tt_b)

    # Houses for overlays
    houseA = _houses_for(jd_tt_a, jd_ut1_a, placeA, house_system, warnings)
    houseB = _houses_for(jd_tt_b, jd_ut1_b, placeB, house_system, warnings)

    # Inter-chart aspects
    A_to_B = _zodiacal_aspects(rowsA, rowsB, _orbs)
    B_to_A = _zodiacal_aspects(rowsB, rowsA, _orbs)
    if antiscia:
        A_to_B += _antiscia_aspects(rowsA, rowsB, _orbs)
        B_to_A += _antiscia_aspects(rowsB, rowsA, _orbs)
    if parallels:
        A_to_B += _parallel_aspects(rowsA, rowsB, _orbs)
        B_to_A += _parallel_aspects(rowsB, rowsA, _orbs)

    # Intra-chart
    intra_A = _intra_aspects(rowsA, _orbs)
    intra_B = _intra_aspects(rowsB, _orbs)

    # Overlays
    overlaysAinB = _overlay(rowsA, houseB)
    overlaysBinA = _overlay(rowsB, houseA)

    # Midpoints
    mid_planets = _midpoints_planets(rowsA, rowsB)
    mid_axes = _midpoints_axes(houseA, houseB)

    # Heuristic scores (sum both directions)
    sc_ab, br_ab = _score_aspects(A_to_B)
    sc_ba, br_ba = _score_aspects(B_to_A)
    sc_total = sc_ab + sc_ba

    meta = {
        "frame": frame,
        "zodiac_mode": _zmode,
        "ayanamsa_deg": float(ayanamsa_deg),
        "house_system": house_system,
        "timescales": {
            "A": tsA,
            "B": tsB,
        },
        "warnings": warnings,
        "notes": [
            "scores are heuristic aggregates; not claims of predictive validity"
        ],
    }

    return {
        "meta": meta,
        "aspects": {
            "A_to_B": A_to_B,
            "B_to_A": B_to_A,
            "intra_A": intra_A,
            "intra_B": intra_B,
        },
        "overlays": {
            "A_in_B": overlaysAinB,
            "B_in_A": overlaysBinA,
        },
        "midpoints": {
            "planets": mid_planets,
            "axes": mid_axes,
        },
        "composite": None,  # filled in by compute_composite or synastry_report convenience
        "scores": {
            "heuristic": True,
            "total": sc_total,
            "breakdown": {
                "A_to_B": br_ab,
                "B_to_A": br_ba,
            },
        },
    }


def compute_composite(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    *,
    method: str = "midpoint",      # "midpoint" | "davison"
    jd_tt_ref: Optional[float] = None,
    jd_ut1_ref: Optional[float] = None,
    place_ref: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    ayanamsa_deg: float = 0.0,
    zodiac_mode: str = "tropical",
) -> Dict[str, Any]:
    """
    Composite chart:
      - midpoint: planetary longitudes as circular midpoints of A & B positions at their nativities.
                  ASC/MC/cusps taken as circular midpoints of A/B axes & cusps (if available).
      - davison:  time/space midpoint => compute ephemeris & houses at mid(JD) and mid(latitude, longitude, elevation).
    """
    warnings: List[str] = []

    # Resolve timescales of A and B (needed for davison time midpoint and also for declination, if used)
    jd_tt_a, jd_ut1_a, tsA = _resolve_timescales(natal_a, None, None, warnings)
    jd_tt_b, jd_ut1_b, tsB = _resolve_timescales(natal_b, None, None, warnings)

    placeA = _to_place(natal_a, None)
    placeB = _to_place(natal_b, None)

    # Common mode for zodiac (sidereal offset if requested)
    _zmode = (natal_a.get("mode") or natal_b.get("mode") or zodiac_mode or "tropical").lower()

    if method == "midpoint":
        # Planet rows for both charts
        rowsA = _planet_rows(jd_tt_a, placeA, frame, MAJORS, warnings)
        rowsB = _planet_rows(jd_tt_b, placeB, frame, MAJORS, warnings)
        if _zmode == "sidereal":
            _apply_ayanamsa(rowsA, ayanamsa_deg)
            _apply_ayanamsa(rowsB, ayanamsa_deg)

        # Planetary midpoints
        mids = _midpoints_planets(rowsA, rowsB)
        pos = {r["name"]: float(r["lon"]) for r in mids}

        # Houses: midpoint of axes & cusps when both available; else None
        hA = _houses_for(jd_tt_a, jd_ut1_a, placeA, house_system, warnings)
        hB = _houses_for(jd_tt_b, jd_ut1_b, placeB, house_system, warnings)

        asc_mid = None
        mc_mid = None
        cusps_mid: Optional[List[float]] = None

        if hA and hB:
            asc_mid = _circ_mid(float(hA.get("asc_deg")), float(hB.get("asc_deg"))) \
                if "asc_deg" in hA and "asc_deg" in hB else None
            mc_mid = _circ_mid(float(hA.get("mc_deg")), float(hB.get("mc_deg"))) \
                if "mc_deg" in hA and "mc_deg" in hB else None
            cA = hA.get("cusps_deg") or []
            cB = hB.get("cusps_deg") or []
            if len(cA) == 12 and len(cB) == 12:
                cusps_mid = [_circ_mid(float(a), float(b)) for a, b in zip(cA, cB)]

        return {
            "method": "midpoint",
            "positions": pos,
            "asc": float(asc_mid) if asc_mid is not None else None,
            "mc": float(mc_mid) if mc_mid is not None else None,
            "cusps": cusps_mid,
            "meta": {
                "frame": frame,
                "zodiac_mode": _zmode,
                "ayanamsa_deg": float(ayanamsa_deg),
                "house_system": house_system,
                "timescales": {"A": tsA, "B": tsB},
                "warnings": warnings,
                "notes": ["midpoint composite: axes/cusps are circular midpoints of natal axes (if available)"],
            },
        }

    elif method == "davison":
        # Time midpoint in TT/UT1
        jd_tt_mid = 0.5 * (jd_tt_a + jd_tt_b)
        jd_ut1_mid = 0.5 * (jd_ut1_a + jd_ut1_b)

        # Space midpoint (lat/lon/elev)
        if placeA and placeB:
            place_mid = {
                "latitude": 0.5 * (placeA["latitude"] + placeB["latitude"]),
                "longitude": 0.5 * (placeA["longitude"] + placeB["longitude"]),
                "elev_m": 0.5 * (placeA.get("elev_m", 0.0) + placeB.get("elev_m", 0.0)),
            }
        else:
            # Fallback: use provided reference place if given
            place_mid = place_ref if place_ref else None
            if place_mid is None:
                _warn(warnings, "davison_missing_place→no_topocentric_angles")

        # Ephemeris at midpoint time/place
        rows = _planet_rows(jd_tt_mid, place_mid, frame, MAJORS, warnings)
        if _zmode == "sidereal":
            _apply_ayanamsa(rows, ayanamsa_deg)
        pos = {r["name"]: float(r["lon"]) for r in rows}

        # Houses for davison at time/place midpoint OR reference place if given
        hs = _houses_for(jd_tt_mid, jd_ut1_mid, place_mid, house_system, warnings)

        return {
            "method": "davison",
            "positions": pos,
            "asc": float(hs.get("asc_deg")) if hs and "asc_deg" in hs else None,
            "mc": float(hs.get("mc_deg")) if hs and "mc_deg" in hs else None,
            "cusps": hs.get("cusps_deg") if hs else None,
            "meta": {
                "frame": frame,
                "zodiac_mode": _zmode,
                "ayanamsa_deg": float(ayanamsa_deg),
                "house_system": house_system,
                "timescales_mid": {"jd_tt": jd_tt_mid, "jd_ut1": jd_ut1_mid},
                "warnings": warnings,
                "notes": ["davison composite: time/space midpoint chart"],
            },
        }

    else:
        raise ValueError("Unknown composite method. Use 'midpoint' or 'davison'.")


def synastry_report(
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
    composite_method: str = "midpoint",
    composite_place_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper: runs synastry + composite, and returns a single bundle with light metrics.
    """
    syn = compute_synastry(
        natal_a, natal_b,
        jd_tt_a=jd_tt_a, jd_ut1_a=jd_ut1_a,
        jd_tt_b=jd_tt_b, jd_ut1_b=jd_ut1_b,
        place_a=place_a, place_b=place_b,
        frame=frame, ayanamsa_deg=ayanamsa_deg, zodiac_mode=zodiac_mode,
        house_system=house_system, orbs=orbs, parallels=parallels, antiscia=antiscia,
    )

    comp = compute_composite(
        natal_a, natal_b,
        method=composite_method,
        jd_tt_ref=None, jd_ut1_ref=None,
        place_ref=composite_place_ref,
        frame=frame, house_system=house_system,
        ayanamsa_deg=ayanamsa_deg, zodiac_mode=zodiac_mode,
    )

    # attach composite summary back to synastry-like envelope + report metrics
    bundle = {
        "meta": {
            **syn["meta"],
            "report": {
                "composite_method": composite_method,
            },
        },
        "aspects": syn["aspects"],
        "overlays": syn["overlays"],
        "midpoints": syn["midpoints"],
        "composite": comp,
        "scores": syn["scores"],
    }

    # very light extra metric: average overlay tightness near cusps
    try:
        ainb = bundle["overlays"]["A_in_B"]
        binA = bundle["overlays"]["B_in_A"]
        def _avg(lst):
            xs = [x["orb_to_nearest_cusp_deg"] for x in lst if x.get("orb_to_nearest_cusp_deg") is not None]
            return (sum(xs)/len(xs)) if xs else None
        bundle["metrics"] = {
            "avg_orb_to_cusp_A_in_B": _avg(ainb),
            "avg_orb_to_cusp_B_in_A": _avg(binA),
        }
    except Exception:
        bundle["metrics"] = {}

    return bundle
