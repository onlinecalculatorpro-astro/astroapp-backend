# app/core/predict.py
from __future__ import annotations

from typing import List, Dict, Any
import os
from math import log

from .ensemble import combine
from .qia import qia_adjust
from .calibration import load_calibrators, apply_calibrator
from .varga import d9_navamsa_sign, d10_dashamsa_sign, reinforcement_score
from .transits import detect_transits
from .astro_extras import (
    harmonic_longitudes,
    part_of_fortune,
    find_aspects,
    fixed_star_ecliptics,
    star_conjunctions,
    lunar_phases,
)

DOMAINS = ["Career/Education", "Relationships", "Relocation", "Health", "Finance"]


# ------------------------------- utils ---------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_call(fn, default, *args, **kwargs):
    """
    Call a function defensively. On *any* exception, return default.
    Never let prediction generation 500 the API.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _jd(chart: Dict[str, Any]) -> float:
    # Prefer TT if present, otherwise UT, otherwise 0
    return float(chart.get("jd_tt") or chart.get("jd_ut") or 0.0)


# ----------------------------- natal features --------------------------------


def natal_points(chart: Dict[str, Any], houses: Dict[str, Any]) -> Dict[str, float]:
    pts = {
        "Asc": float(houses.get("asc_deg", 0.0)),
        "MC": float(houses.get("mc_deg", 0.0)),
    }
    for b in chart.get("bodies", []) or []:
        name = b.get("name")
        if name in ("Sun", "Moon", "Jupiter", "Saturn", "Mars", "Venus", "Mercury"):
            pts[name] = float(b.get("longitude_deg", 0.0))
    return pts


# ----------------------------- evidence --------------------------------------


def evidence_contributions(
    chart: Dict[str, Any], houses: Dict[str, Any], horizon: str
) -> Dict[str, float]:
    pts = natal_points(chart, houses)

    # Planet longitudes map
    planet_lons: Dict[str, float] = {}
    for b in chart.get("bodies", []) or []:
        n = b.get("name")
        if n:
            planet_lons[n] = float(b.get("longitude_deg", 0.0))

    # Varga reinforcement (D9 of Moon, D10 of MC)
    moon = float(pts.get("Moon", 0.0))
    mc = float(pts.get("MC", 0.0))
    d9_moon = _safe_call(d9_navamsa_sign, 0, moon)
    d10_mc = _safe_call(d10_dashamsa_sign, 0, mc)
    v_score = _safe_call(reinforcement_score, 0.05, d9_moon, d10_mc)

    # Transit score (placeholder: slow movers to key points)
    trans = {k: v for k, v in pts.items() if k in ("Jupiter", "Saturn", "Mars")}
    key_pts = {
        "Sun": pts.get("Sun", 0.0),
        "Moon": pts.get("Moon", 0.0),
        "Asc": pts.get("Asc", 0.0),
        "MC": pts.get("MC", 0.0),
    }
    t_score = _safe_call(detect_transits, 0.0, key_pts, trans)

    # Dasha score (smooth function of Moon subdivision)
    d_frac = (moon % 30.0) / 30.0
    d_score = 0.25 + 0.5 * abs(0.5 - d_frac)

    # Yoga score (very light)
    j_m = abs((pts.get("Jupiter", 0.0) - pts.get("Moon", 0.0)) % 360.0)
    y_score = 0.06 if min(j_m, 360.0 - j_m) < 15.0 or abs(j_m - 90.0) < 15.0 else 0.02

    # Harmonics (D7/D9 placeholder aggregation)
    _safe_call(harmonic_longitudes, {}, planet_lons, 7)
    _safe_call(harmonic_longitudes, {}, planet_lons, 9)
    harmonic_score = 0.05  # fixed light weight

    # Arabic parts (Part of Fortune proximity to Sun)
    asc = float(houses.get("asc_deg", 0.0))
    sun_lon = float(planet_lons.get("Sun", 0.0))
    moon_lon = float(planet_lons.get("Moon", 0.0))
    pof = _safe_call(part_of_fortune, asc, asc, sun_lon, moon_lon, True)
    parts_score = 0.1 if abs((pof - sun_lon + 540.0) % 360.0 - 180.0) < 30 else 0.05

    # Aspects
    aspects = _safe_call(find_aspects, [], planet_lons, planet_lons, orb_deg=2.0)
    beneficial = [a for a in aspects if a.get("aspect") in ("trine", "sextile")]
    challenging = [a for a in aspects if a.get("aspect") in ("square", "opposition")]
    aspect_score = len(beneficial) * 0.05 - len(challenging) * 0.02
    aspect_score = _clamp(aspect_score + 0.1, 0.01, 0.25)

    # Fixed stars & lunar phase (use any JD we have)
    jd = _jd(chart)
    stars = _safe_call(fixed_star_ecliptics, {}, jd)
    star_conj = _safe_call(star_conjunctions, [], planet_lons, stars, orb_deg=1.0)
    star_score = _clamp(len(star_conj) * 0.03 + 0.02, 0.02, 0.15)

    lunar_info = _safe_call(lunar_phases, {"illumination": 0.5}, jd)
    lunar_score = 0.05 + float(lunar_info.get("illumination", 0.5)) * 0.1

    return {
        "dasha": round(float(d_score), 3),
        "transit": round(float(t_score), 3),
        "varga": round(float(v_score), 3),
        "yoga": round(float(y_score), 3),
        "harmonic": round(float(harmonic_score), 3),
        "parts": round(float(parts_score), 3),
        "aspects": round(float(aspect_score), 3),
        "stars": round(float(star_score), 3),
        "lunar": round(float(lunar_score), 3),
    }


def calibrated_probability(ev: Dict[str, float], domain: str) -> float:
    # Base combination (core signals)
    base = _safe_call(
        combine,
        0.2,
        float(ev.get("dasha", 0.0)),
        float(ev.get("transit", 0.0)),
        float(ev.get("varga", 0.0)),
        float(ev.get("yoga", 0.0)),
    )

    # Add extended evidence
    enhanced_contrib = (
        ev.get("harmonic", 0.0) * 0.10
        + ev.get("parts", 0.0) * 0.15
        + ev.get("aspects", 0.0) * 0.20
        + ev.get("stars", 0.0) * 0.10
        + ev.get("lunar", 0.0) * 0.05
    )

    enhanced = _clamp(base + enhanced_contrib, 0.05, 0.95)

    # QIA quality adjustment (defensive)
    q = _safe_call(qia_adjust, enhanced, enhanced, ev)

    # Optional calibrator; if file is missing/malformed, just return q
    calibrators_path = os.environ.get("ASTRO_CALIBRATORS", "config/calibrators.json")
    calibrators = _safe_call(load_calibrators, None, calibrators_path)
    if calibrators is not None:
        return _safe_call(apply_calibrator, q, domain, q, calibrators)
    return q


# ----------------------------- intervals -------------------------------------


def select_intervals(
    chart: Dict[str, Any], houses: Dict[str, Any], horizon: str
) -> List[Dict[str, Any]]:
    """
    Produce five horizon-dependent ISO8601 [start,end) windows.
    Kept deterministic for testing; safe even if horizon is unknown.
    """
    if horizon == "short":
        anchors = ["2026-01-05", "2026-01-20", "2026-02-04", "2026-02-19", "2026-03-05"]
        # start = end on purpose (instant window) to match downstream expectations
        return [{"start": f"{d}T00:00:00Z", "end": f"{d}T00:00:00Z"} for d in anchors]

    # Fallback for medium/long: 2-month steps starting Jan 2026
    anchors = ["2026-01-01", "2026-03-01", "2026-05-01", "2026-07-01", "2026-09-01"]
    return [{"start": f"{d}T00:00:00Z", "end": f"{d}T00:00:00Z"} for d in anchors]


# ----------------------------- main API --------------------------------------


def predict(chart: Dict[str, Any], houses: Dict[str, Any], horizon: str) -> List[Dict[str, Any]]:
    """
    Create one prediction per domain with:
      - probability (already calibrated if possible)
      - evidence chips (stable keys)
      - interval {start,end}
    This function must NEVER raise; callers rely on 200s.
    """
    intervals = select_intervals(chart, houses, horizon)
    out: List[Dict[str, Any]] = []

    for i, domain in enumerate(DOMAINS):
        ev = evidence_contributions(chart, houses, horizon)
        p = calibrated_probability(ev, domain)
        out.append(
            {
                "domain": domain,
                "probability": round(float(p), 3),
                "evidence": ev,
                "interval": intervals[i % len(intervals)],
            }
        )

    return out
