# -*- coding: utf-8 -*-
"""
horary_hybrid.py
----------------
Hybrid (birth + question-moment) prashna core.

Requirements & design
- Birth details are **mandatory** (QuerentBirthData).
- Uses horary_shared.py for dataclasses, chart/house engines, dignity/aspects, etc.
- Produces:
    * composite scoring from multiple components
    * narrative evidence (positives/negatives/neutrals)
    * rough timing windows (heuristic; plug real dasha/transit later)
- Conservative defaults; never crashes if houses engine is unavailable
  (falls back to Equal from ASC via shared helper).

Public API
- analyze_hybrid(inp: HybridPrasnaInput) -> Dict[str, Any]
- analyze_hybrid_enhanced(inp) -> adds {"method": "hybrid"} convenience key
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import asdict
from datetime import datetime, timezone, timedelta

# Shared imports
from horary_shared import (
    # Dataclasses / enums / mappings
    HybridPrasnaInput, QuerentBirthData, QuestionType, ENHANCED_QUESTION_HOUSES,
    TRAD_PLANETS, SIGN_NAMES, SIGN_LORDS,

    # Helpers
    deg_wrap, sign_index, sign_name_from_deg, lord_of_sign,
    angular_sep, shift_sidereal, house_of,
    calculate_planetary_dignity, calculate_aspects,

    # KP
    kp_star_and_sublord,

    # Chart/Houses
    ensure_coords_and_tz, build_chart, compute_houses_from_chart, safe_get_asc,

    # Radicality
    radicality_flags,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BENEFICS = {"Jupiter", "Venus", "Moon", "Mercury"}
_MALEFICS = {"Saturn", "Mars", "Sun"}

def _find_house_of(longitude: float, cusps: List[float]) -> int:
    """Wrap-safe house membership; alias to horary_shared.house_of for clarity."""
    return house_of(longitude, cusps)

def _score_chart_relevance(chart: Dict[str, Any], cusps: List[float], qtype: QuestionType) -> Tuple[float, Dict[str, Any]]:
    """
    Score a single chart for question relevance:
    - Weight planets by (house tier + dignity), with light benefic/malefic adjustments.
    - Return (0..1) score and a per-planet breakdown.
    """
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    primary    = set(mapping.get("primary", []))
    secondary  = set(mapping.get("secondary", []))
    supportive = set(mapping.get("supportive", []))
    obstruct   = set(mapping.get("obstructive", []))

    per_planet = []
    total = 0.0
    max_possible = 0.0

    for b in chart.get("bodies", []):
        nm = b["name"]
        if nm not in TRAD_PLANETS: 
            continue
        lon = float(b["longitude_deg"])
        h   = _find_house_of(lon, cusps)
        dig = calculate_planetary_dignity(lon, nm)

        # Base tier weights
        if h in primary:     tier = 1.00
        elif h in secondary: tier = 0.80
        elif h in supportive:tier = 0.60
        elif h in obstruct:  tier = 0.20
        else:                tier = 0.40

        # Benefic/malefic nudge (very light)
        nud = 0.08 if nm in _BENEFICS else (-0.05 if nm in _MALEFICS else 0.0)

        # Normalize dignity -2..+2 → 0..1 via (dig+2)/4
        dig_n = (dig + 2.0) / 4.0

        # Planet weight (bounded)
        w = max(0.0, min(1.0, 0.55*tier + 0.35*dig_n + nud))

        per_planet.append({
            "planet": nm,
            "longitude_deg": round(lon, 4),
            "sign": sign_name_from_deg(lon),
            "house": h,
            "dignity": dig,
            "weight": round(w, 4),
            "tier": tier,
        })
        total += w
        max_possible += 1.0  # each traditional planet capped to ~1

    score = 0.0 if max_possible <= 0 else max(0.0, min(1.0, total / max_possible))
    return score, {
        "per_planet": per_planet,
        "summary": {"sum": round(total, 3), "max": round(max_possible, 3)}
    }

def _chart_snapshot(chart: Dict[str, Any], cusps: List[float]) -> Dict[str, Any]:
    """Basic snapshot for UI/debug."""
    asc = safe_get_asc(chart)
    moon = next((p for p in chart.get("bodies", []) if p["name"] == "Moon"), None)
    return {
        "ASC_deg": asc,
        "ASC_sign": (sign_name_from_deg(float(asc)) if asc is not None else None),
        "Moon_deg": (moon["longitude_deg"] if moon else None),
        "Moon_sign": (sign_name_from_deg(moon["longitude_deg"]) if moon else None),
        "cusps_deg": cusps,
    }

def _evidence_from_chart(chart: Dict[str, Any], cusps: List[float], qtype: QuestionType) -> Dict[str, List[str]]:
    """
    Build narrative evidence lists: positives/negatives/neutrals.
    Heuristics:
      + Benefics in primary/secondary/supportive houses
      + House lords strengthened by dignity/aspects to Moon
      - Malefics in obstructive houses or afflicting Moon/house lords
    """
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    primary    = set(mapping.get("primary", []))
    secondary  = set(mapping.get("secondary", []))
    supportive = set(mapping.get("supportive", []))
    obstruct   = set(mapping.get("obstructive", []))

    positives: List[str] = []
    negatives: List[str] = []
    neutrals:  List[str] = []

    bodies = chart.get("bodies", [])
    byname = {b["name"]: b for b in bodies}
    moon   = byname.get("Moon")
    moonL  = float(moon["longitude_deg"]) if moon else None

    # House lords from cusp signs
    house_lords: Dict[int, str] = {i+1: lord_of_sign(c) for i, c in enumerate(cusps)}

    for b in bodies:
        nm = b["name"]
        if nm not in TRAD_PLANETS:
            continue
        lon = float(b["longitude_deg"])
        h   = _find_house_of(lon, cusps)
        dig = calculate_planetary_dignity(lon, nm)

        # Moon proximity to house lord?
        if moonL is not None and nm in set(house_lords.values()):
            sep = angular_sep(moonL, lon)
            if sep < 12.0:
                positives.append(f"Moon close to {nm} (house lord) by {round(sep,1)}° — supportive timing.")

        # Planet position evidence
        tag = f"{nm} in H{h} ({sign_name_from_deg(lon)}), dignity {dig:+.0f}"
        if nm in _BENEFICS and (h in (primary | secondary | supportive)):
            positives.append(f"Benefic {tag}.")
        elif nm in _MALEFICS and (h in obstruct):
            negatives.append(f"Malefic {tag} in obstructive house.")
        else:
            neutrals.append(tag + ".")

        # Aspect Moon stress/support
        if moonL is not None and nm != "Moon":
            sep, asp = calculate_aspects(lon, moonL)
            if asp in ("trine","sextile","conjunction") and nm in _BENEFICS:
                positives.append(f"{nm} {asp} Moon — supportive.")
            if asp in ("square","opposition") and nm in _MALEFICS:
                negatives.append(f"{nm} {asp} Moon — stress.")

    return {"positives": positives, "negatives": negatives, "neutrals": neutrals}

def _dasha_support_from_birth(birth_chart: Dict[str, Any]) -> Tuple[float, str]:
    """
    Placeholder: infer 'friendly' vs 'tough' period from Moon sign lord in birth chart.
    Returns (0..1, explanation).
    """
    moon = next((p for p in birth_chart.get("bodies", []) if p["name"] == "Moon"), None)
    if not moon:
        return 0.5, "Moon unknown; neutral period."
    moon_lord = lord_of_sign(float(moon["longitude_deg"]))
    if moon_lord in {"Jupiter","Venus","Moon"}:
        return 0.7, f"Moon sign lord {moon_lord} — generally supportive period."
    if moon_lord in {"Saturn","Mars"}:
        return 0.4, f"Moon sign lord {moon_lord} — somewhat challenging period."
    return 0.55, f"Moon sign lord {moon_lord} — mixed period."

def _harmony_between_charts(birth_chart: Dict[str, Any], q_chart: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Fraction of 'soft' aspects (trine, sextile, conjunction) across the seven classical planets.
    """
    bene, total = 0, 0
    bmap = {p["name"]: p for p in birth_chart.get("bodies", []) if p["name"] in TRAD_PLANETS}
    qmap = {p["name"]: p for p in q_chart.get("bodies", []) if p["name"] in TRAD_PLANETS}

    details = []
    for nm in TRAD_PLANETS:
        if nm in bmap and nm in qmap:
            bL = float(bmap[nm]["longitude_deg"])
            qL = float(qmap[nm]["longitude_deg"])
            sep, asp = calculate_aspects(bL, qL)
            soft = asp in ("trine","sextile","conjunction")
            if soft:
                bene += 1
            total += 1
            details.append({"planet": nm, "sep": round(sep,2), "aspect": asp, "soft": soft})

    frac = (bene / total) if total else 0.0
    return frac, {"per_planet": details, "count_soft": bene, "count_total": total}

def _rough_timing(
    now_iso: str,
    composite: float,
    dasha_support: float,
    qtype: QuestionType,
    q_chart_snap: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Heuristic timing windows based on composite & dasha_support.
    Produces broad windows relative to the question time.
    """
    try:
        now_dt = datetime.fromisoformat(now_iso.replace("Z","")).replace(tzinfo=timezone.utc)
    except Exception:
        now_dt = datetime.now(timezone.utc)

    # Baseline windows by composite strength
    if composite >= 0.70:
        base = [(0, 9), (9, 18)]
        assessment = "near-term likely"
    elif composite >= 0.60:
        base = [(3, 12), (12, 24)]
        assessment = "moderately soon"
    elif composite >= 0.50:
        base = [(6, 18), (18, 30)]
        assessment = "possible, needs conditions"
    else:
        base = [(12, 36)]
        assessment = "deferred or unlikely"

    # Adjust by dasha support (shift earlier/later)
    shift = -2 if dasha_support >= 0.65 else (2 if dasha_support <= 0.45 else 0)
    windows = []
    for a, b in base:
        start = now_dt + timedelta(days=30 * max(0, a + shift))
        end   = now_dt + timedelta(days=30 * max(0, b + shift))
        windows.append({
            "from": start.date().isoformat(),
            "to": end.date().isoformat(),
            "reason": f"{assessment}; dasha support={round(dasha_support,2)}",
        })

    # Add a clue from Moon nakshatra at question time (label only)
    moon_deg = q_chart_snap.get("Moon_deg")
    nak_label = None
    if isinstance(moon_deg, (int, float)):
        star, sub, star_span, _ = kp_star_and_sublord(float(moon_deg))
        nak_label = f"Moon nakshatra={star}, sub={sub}"

    return {
        "assessment": assessment,
        "windows": windows,
        "clues": [nak_label] if nak_label else [],
    }

# ---------------------------------------------------------------------------
# Public analysis
# ---------------------------------------------------------------------------

def analyze_hybrid(inp: HybridPrasnaInput) -> Dict[str, Any]:
    if not inp or not isinstance(inp, HybridPrasnaInput):
        return {"ok": False, "system": "hybrid", "error": "invalid_input"}
    if not inp.querent_birth:
        return {"ok": False, "system": "hybrid", "error": "Querent birth data required"}

    qtype = inp.question_type or QuestionType.JOB

    # ---- Build Question Chart + Houses
    q_chart = build_chart(
        date=inp.question_date, time=inp.question_time, tz=inp.question_tz,
        place=inp.question_place, latitude=inp.question_latitude, longitude=inp.question_longitude,
        zodiac_mode=inp.zodiac_mode, ayanamsa=inp.ayanamsa, topocentric=True,
    )
    q_meta = q_chart.get("meta", {}) or {}
    q_aya  = q_meta.get("ayanamsa_deg")
    q_obs  = q_meta.get("observer") or {}
    q_lat  = float(q_obs.get("latitude", inp.question_latitude or 0.0))
    q_lon  = float(q_obs.get("longitude", inp.question_longitude or 0.0))
    q_houses = compute_houses_from_chart(
        q_chart, latitude=q_lat, longitude=q_lon,
        house_system=inp.house_system, zodiac_mode=inp.zodiac_mode, ayanamsa_deg=q_aya,
    )
    q_cusps = q_houses.get("cusps_deg", []) or []
    q_snap  = _chart_snapshot(q_chart, q_cusps)

    # ---- Build Birth Chart + Houses
    b = inp.querent_birth
    b_chart = build_chart(
        date=b.date, time=b.time, tz=b.tz_name,
        place=b.place, latitude=b.latitude, longitude=b.longitude,
        zodiac_mode=b.zodiac_mode, ayanamsa=b.ayanamsa, topocentric=True,
    )
    b_meta = b_chart.get("meta", {}) or {}
    b_aya  = b_meta.get("ayanamsa_deg")
    b_obs  = b_meta.get("observer") or {}
    b_lat  = float(b_obs.get("latitude", b.latitude or 0.0))
    b_lon  = float(b_obs.get("longitude", b.longitude or 0.0))
    b_houses = compute_houses_from_chart(
        b_chart, latitude=b_lat, longitude=b_lon,
        house_system=inp.house_system, zodiac_mode=b.zodiac_mode, ayanamsa_deg=b_aya,
    )
    b_cusps = b_houses.get("cusps_deg", []) or []
    b_snap  = _chart_snapshot(b_chart, b_cusps)

    # ---- Scores
    birth_score, birth_break = _score_chart_relevance(b_chart, b_cusps, qtype)
    ques_score,  ques_break  = _score_chart_relevance(q_chart, q_cusps, qtype)
    dasha_support, dasha_note = _dasha_support_from_birth(b_chart)
    harmony, harmony_detail   = _harmony_between_charts(b_chart, q_chart)

    # Tunable weights (sum ~1.0)
    W = {"birth": 0.30, "question": 0.35, "dasha": 0.20, "harm": 0.15}
    composite = max(0.0, min(1.0,
        birth_score*W["birth"] + ques_score*W["question"] +
        dasha_support*W["dasha"] + harmony*W["harm"]
    ))

    # ---- Evidence
    ev_q = _evidence_from_chart(q_chart, q_cusps, qtype)
    ev_b = _evidence_from_chart(b_chart, b_cusps, qtype)
    positives = (ev_q["positives"] + ev_b["positives"])[:20]
    negatives = (ev_q["negatives"] + ev_b["negatives"])[:20]
    neutrals  = (ev_q["neutrals"]  + ev_b["neutrals"]) [:20]

    # ---- Judgement bands
    if composite > 0.66:
        answer = "yes"
        confidence = min(0.95, 0.70 + 0.30*(composite-0.66)/(1.0-0.66))
    elif composite < 0.40:
        answer = "no"
        confidence = min(0.95, 0.70 + 0.30*(0.40-composite)/0.40)
    else:
        answer = "uncertain"
        confidence = 0.58 + abs(composite - 0.53) * 0.30

    # ---- Timing windows (heuristic)
    q_timescales = (q_meta.get("timescales") or {})
    # Prefer explicit question datetime if provided; else fallback to now UTC
    q_date = inp.question_date or ""
    q_time = inp.question_time or ""
    q_tz   = inp.question_tz   or "UTC"
    try:
        from zoneinfo import ZoneInfo
        dt_local = datetime.fromisoformat(f"{q_date}T{q_time}").replace(tzinfo=ZoneInfo(q_tz))
        now_iso = dt_local.astimezone(timezone.utc).isoformat()
    except Exception:
        now_iso = datetime.now(timezone.utc).isoformat()

    timing = _rough_timing(
        now_iso=now_iso,
        composite=composite,
        dasha_support=dasha_support,
        qtype=qtype,
        q_chart_snap=q_snap,
    )

    # ---- Radicality (FYI)
    try:
        rad = radicality_flags(q_chart, q_tz, q_date or now_iso[:10], q_time or now_iso[11:19])
    except Exception:
        rad = {"fits": True}

    return {
        "ok": True,
        "system": "hybrid",
        "meta": {
            "question_type": qtype.value,
            "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
            "weights": W,
            "radicality": rad,
        },
        "charts": {
            "birth":  b_snap,
            "question": q_snap,
            "house_system_used": inp.house_system,
            "zodiac_mode": inp.zodiac_mode,
            "ayanamsa": inp.ayanamsa,
        },
        "scores": {
            "birth_chart_relevance": round(birth_score, 3),
            "question_chart_strength": round(ques_score, 3),
            "dasha_support": round(dasha_support, 3),
            "harmony": round(harmony, 3),
            "composite": round(composite, 3),
            "details": {
                "birth_breakdown": birth_break,
                "question_breakdown": ques_break,
                "harmony_detail": harmony_detail,
                "dasha_note": dasha_note,
            }
        },
        "evidence": {
            "positives": positives,
            "negatives": negatives,
            "neutrals":  neutrals,
        },
        "timing": timing,
        "judgement": {
            "answer": answer,
            "confidence": round(confidence, 3),
            "reason": "Hybrid synthesis of birth potential and question-moment factors.",
            "band_explanation": {
                "yes_if":  ">0.66 composite",
                "no_if":   "<0.40 composite",
                "else":    "uncertain band",
            }
        },
        "notes": {
            "disclaimer": "Timing windows are heuristic. For production, replace with full Vimshottari/transit engine.",
            "inputs_used": {
                "question": {
                    "date": inp.question_date, "time": inp.question_time, "tz": inp.question_tz,
                    "place": inp.question_place, "lat": inp.question_latitude, "lon": inp.question_longitude
                },
                "birth": asdict(inp.querent_birth),
            }
        }
    }

def analyze_hybrid_enhanced(inp: HybridPrasnaInput) -> Dict[str, Any]:
    """Thin wrapper that ensures {"method": "hybrid"} is present."""
    res = analyze_hybrid(inp)
    if isinstance(res, dict):
        res.setdefault("method", "hybrid")
    return res

__all__ = [
    "analyze_hybrid",
    "analyze_hybrid_enhanced",
]
