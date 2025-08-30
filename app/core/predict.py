from __future__ import annotations
from typing import List, Dict, Any
from .ensemble import combine
from .qia import qia_adjust
from .calibration import load_calibrators, apply_calibrator
from .varga import d9_navamsa_sign, d10_dashamsa_sign, reinforcement_score
from .transits import detect_transits
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

    return {"dasha": round(d_score,3), "transit": round(t_score,3), "varga": round(v_score,3), "yoga": round(y_score,3)}

def calibrated_probability(ev: Dict[str, float], domain: str) -> float:
    base = combine(ev["dasha"], ev["transit"], ev["varga"], ev["yoga"])
    q = qia_adjust(base, ev)
    calibrators = load_calibrators(os.environ.get("ASTRO_CALIBRATORS","config/calibrators.json"))
    return apply_calibrator(domain, q, calibrators)

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
        ev = evidence_contributions(chart, houses, horizon)
        p = calibrated_probability(ev, d)
        preds.append({"domain": d, "probability": round(p,3), "evidence": ev, "interval": intervals[i % len(intervals)]})
    return preds
