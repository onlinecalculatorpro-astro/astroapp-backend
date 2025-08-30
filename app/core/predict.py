from __future__ import annotations
from typing import List, Dict, Any
from .ensemble import combine
from .qia import qia_adjust
from .calibration import load_calibrators, apply_calibrator
from .varga import d9_navamsa_sign, d10_dashamsa_sign, reinforcement_score
from .transits import detect_transits
from .astro_extras import (
    harmonic_longitudes, part_of_fortune, find_aspects,
    fixed_star_ecliptics, star_conjunctions, lunar_phases
)
from math import log
import os

DOMAINS = ["Career/Education","Relationships","Relocation","Health","Finance"]

def natal_points(chart: Dict[str, Any], houses: Dict[str, Any]) -> Dict[str, float]:
    pts = {"Asc": houses.get("asc_deg",0.0), "MC": houses.get("mc_deg",0.0)}
    for b in chart.get("bodies",[]):
        if b["name"] in ("Sun","Moon","Jupiter","Saturn","Mars","Venus","Mercury"):
            pts[b["name"]] = float(b["longitude_deg"])
    return pts

def evidence_contributions(chart: Dict[str, Any], houses: Dict[str, Any], horizon: str) -> Dict[str, float]:
    pts = natal_points(chart, houses)
    
    # Extract planetary positions for enhanced calculations
    planet_lons = {body["name"]: float(body["longitude_deg"]) for body in chart.get("bodies", [])}
    
    # Varga reinforcement from D9 (Moon) and D10 (MC approximated as Sun's sign for placeholder)
    moon = pts.get("Moon", 0.0)
    mc = pts.get("MC", 0.0)
    d9_moon = d9_navamsa_sign(moon)
    d10_mc = d10_dashamsa_sign(mc)
    v_score = reinforcement_score(d9_moon, d10_mc)
    
    # Transit score from slow movers toward key points (placeholder uses natal planets as transiting too)
    trans = {k:v for k,v in pts.items() if k in ("Jupiter","Saturn","Mars")}
    t_score = detect_transits({"Sun":pts.get("Sun",0.0), "Moon":pts.get("Moon",0.0), "Asc":pts["Asc"], "MC":pts["MC"]}, trans)
    
    # Dasha score placeholder: based on Moon's nakshatra quarter mapping to a smooth score
    # use fractional part of Moon lon within sign
    d_frac = (moon % 30.0) / 30.0
    d_score = 0.25 + 0.5*abs(0.5 - d_frac)
    
    # Yoga score (placeholder constant small boost)
    y_score = 0.06 if (abs(pts.get("Jupiter",0)-pts.get("Moon",0)) % 90.0) < 15.0 else 0.02
    
    # Enhanced features from astro_extras - with safe error handling
    
    try:
        # Harmonic analysis (D7 and D9)
        h7_lons = harmonic_longitudes(planet_lons, 7)
        h9_lons = harmonic_longitudes(planet_lons, 9)
        harmonic_score = 0.05  # Base score
    except Exception:
        harmonic_score = 0.05
    
    try:
        # Arabic parts
        asc = float(houses.get("asc_deg", 0))
        sun_lon = float(planet_lons.get("Sun", 0))
        moon_lon = float(planet_lons.get("Moon", 0))
        pof = part_of_fortune(asc, sun_lon, moon_lon, True)
        parts_score = 0.1 if abs(pof - sun_lon) < 30 else 0.05
    except Exception:
        parts_score = 0.05
    
    try:
        # Aspect analysis
        aspects = find_aspects(planet_lons, planet_lons, orb_deg=2.0)
        beneficial_aspects = [a for a in aspects if a["aspect"] in ["trine", "sextile"]]
        challenging_aspects = [a for a in aspects if a["aspect"] in ["square", "opposition"]]
        aspect_score = len(beneficial_aspects) * 0.05 - len(challenging_aspects) * 0.02
        aspect_score = max(0.01, min(0.25, aspect_score + 0.1))
    except Exception:
        aspect_score = 0.1
    
    try:
        # Fixed star influences
        stars = fixed_star_ecliptics(chart.get("jd_tt", 0))
        star_conj = star_conjunctions(planet_lons, stars, orb_deg=1.0)
        star_score = min(len(star_conj) * 0.03, 0.15) + 0.02
    except Exception:
        star_score = 0.02
    
    try:
        # Lunar phases
        lunar_info = lunar_phases(chart.get("jd_tt", 0))
        lunar_score = 0.05 + float(lunar_info.get("illumination", 0.5)) * 0.1
    except Exception:
        lunar_score = 0.07
    
    return {
        "dasha": round(float(d_score),3), 
        "transit": round(float(t_score),3), 
        "varga": round(float(v_score),3), 
        "yoga": round(float(y_score),3),
        "harmonic": round(float(harmonic_score),3),
        "parts": round(float(parts_score),3),
        "aspects": round(float(aspect_score),3),
        "stars": round(float(star_score),3),
        "lunar": round(float(lunar_score),3)
    }

def calibrated_probability(ev: Dict[str, float], domain: str) -> float:
    # Combine original evidence
    base = combine(ev["dasha"], ev["transit"], ev["varga"], ev["yoga"])
    
    # Add enhanced evidence with weights
    enhanced_contrib = (
        float(ev.get("harmonic", 0)) * 0.1 +
        float(ev.get("parts", 0)) * 0.15 +
        float(ev.get("aspects", 0)) * 0.2 +
        float(ev.get("stars", 0)) * 0.1 +
        float(ev.get("lunar", 0)) * 0.05
    )
    
    # Combine base with enhanced features
    enhanced_base = float(base) + float(enhanced_contrib)
    enhanced_base = max(0.05, min(0.95, enhanced_base))
    
    q = qia_adjust(enhanced_base, ev)
    calibrators = load_calibrators(os.environ.get("ASTRO_CALIBRATORS","config/calibrators.json"))
    result = apply_calibrator(domain, q, calibrators)
    
    return float(result)

def select_intervals(chart: Dict[str, Any], houses: Dict[str, Any], horizon: str) -> List[Dict[str, Any]]:
    # Placeholder: create horizon-dependent windows in 2026
    if horizon == "short":
        starts = ["2026-01-05","2026-01-20","2026-02-04","2026-02-19","2026-03-05"]
        span_days = 30
    else:
        starts = ["2026-01-01","2026-03-01","2026-05-01","2026-07-01","2026-09-01"]
        span_days = 60
    intervals = []
    for i,s in enumerate(starts):
        e = {"start": f"{s}T00:00:00Z", "end": f"2026-{int(s.split('-')[1]):02d}-{int(s.split('-')[2]):02d}T00:00:00Z"}
        intervals.append(e)
    return intervals

def predict(chart: Dict[str, Any], houses: Dict[str, Any], horizon: str) -> List[Dict[str, Any]]:
    preds = []
    intervals = select_intervals(chart, houses, horizon)
    for i, d in enumerate(DOMAINS):
        try:
            ev = evidence_contributions(chart, houses, horizon)
            p = calibrated_probability(ev, d)
            pred_data = {
                "domain": str(d), 
                "probability": round(float(p), 3), 
                "evidence": ev, 
                "interval": intervals[i % len(intervals)]
            }
            preds.append(pred_data)
        except Exception as e:
            # Fallback prediction if enhanced calculations fail
            fallback_ev = {
                "dasha": 0.5, "transit": 0.3, "varga": 0.05, "yoga": 0.02,
                "harmonic": 0.05, "parts": 0.05, "aspects": 0.1, "stars": 0.02, "lunar": 0.07
            }
            preds.append({
                "domain": str(d), 
                "probability": 0.3, 
                "evidence": fallback_ev, 
                "interval": intervals[i % len(intervals)]
            })
    return preds
