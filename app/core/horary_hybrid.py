# -*- coding: utf-8 -*-
"""
horary_hybrid.py — hybrid pipeline (2025-09-29, patched)
--------------------------------------------------------
Hybrid (birth + question-moment) prashna analysis.

Pipeline
- Step 1: Analyze BIRTH chart with the SAME engines as classical:
          Parāśarī + KP (via horary_classical.analyze_*).
- Step 2: Parāśarī CROSS analysis (birth ↔ question) using house relevance,
          dignity, and inter-chart aspects (soft/hard).
          **Patched:** both charts are aligned to the same zodiac frame first.
- Step 3: Analyze QUESTION (horary) chart: Parāśarī + KP (classical engines).
- Step 4: Fuse Step 2 & Step 3 into the FINAL verdict, with explicit grounds.

Outputs
- Clear, structured results per step + final:
  - judgement {answer, confidence}
  - scores/strengths and compact “grounds”:
      {for: [], against: [], conflicts: [], notes: []}
- Conservative defaults; robust fallbacks (Equal from ASC via shared module).

Public API
- analyze_hybrid(inp: HybridPrasnaInput) -> Dict[str, Any]
- analyze_hybrid_enhanced(inp) -> adds {"method": "hybrid"}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import asdict
from datetime import datetime, timezone

# Shared imports
from .horary_shared import (
    HybridPrasnaInput, QuerentBirthData, HoraryInput, QuestionType, ENHANCED_QUESTION_HOUSES,
    # constants/helpers
    deg_wrap, sign_index, sign_name_from_deg, lord_of_sign,
    angular_sep, house_of,
    calculate_aspects,                      # Ptolemaic aspects
    calc_dignity_rich,                      # rich dignity score
    build_chart, compute_houses_from_chart, safe_get_asc, radicality_flags,
)

# Use the SAME classical engines for consistency (Step 1 & Step 3)
from .horary_classical import (
    analyze_parashari as parashari_classical,
    analyze_kp as kp_classical,
)

# ---------------------------------------------------------------------------
# Local role tags (kept light & traditional for scoring/grounds)
# ---------------------------------------------------------------------------

_BENEFICS: Set[str] = {"Jupiter", "Venus", "Moon"}
_MALEFICS: Set[str] = {"Saturn", "Mars", "Sun"}
_NEUTRAL:  Set[str] = {"Mercury"}  # treated as slight-benefic in weights

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _with_node_aliases(bodies_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Provide both Western ("North Node"/"South Node") and Vedic ("Rahu"/"Ketu") keys.
    """
    d = {b.get("name"): b for b in bodies_list if isinstance(b, dict) and b.get("name")}
    if "North Node" in d and "Rahu" not in d:
        d["Rahu"] = d["North Node"]
    if "South Node" in d and "Ketu" not in d:
        d["Ketu"] = d["South Node"]
    return d

def _mk_reason(source: str, weight: float, text: str) -> Dict[str, Any]:
    return {"source": source, "weight": round(max(0.0, min(1.0, float(weight))), 3), "text": text}

def _top(items: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: x.get("weight", 0.0), reverse=True)[:n]

# ---------------------------------------------------------------------------
# Grounds formatters (Parāśarī & KP)
# ---------------------------------------------------------------------------

def _parashari_grounds_from_result(p_res: Dict[str, Any], mapping: Dict[str, List[int]]) -> Dict[str, Any]:
    """Summarize Parāśarī result into for/against/conflicts/notes grounds."""
    for_list: List[Dict[str, Any]] = []
    against_list: List[Dict[str, Any]] = []
    notes: List[str] = []

    strengths = p_res.get("planetary_analysis", []) or []
    sb = p_res.get("scoring_breakdown", {}) or {}
    primary = set(mapping.get("primary", []))
    secondary = set(mapping.get("secondary", []))
    supportive = set(mapping.get("supportive", []))
    obstructive = set(mapping.get("obstructive", []))

    for s in strengths:
        nm = s.get("planet")
        h  = int(s.get("house", 0) or 0)
        w  = float(s.get("overall_strength", 0.0) or 0.0)
        sig = s.get("sign")
        base = f"{nm} in H{h} ({sig}), strength={w:.2f}"
        if h in primary or h in secondary or h in supportive:
            alpha = 0.6 if h in primary else (0.45 if h in secondary else 0.35)
            for_list.append(_mk_reason("parashari", alpha*w, base))
        elif h in obstructive:
            against_list.append(_mk_reason("parashari", 0.5*w, base + " in obstructive house"))

    if sb.get("moon_proximity_bonus", 0) > 0:
        notes.append("Moon close to key house-lord(s) → timing/strength boost.")
    if sb.get("drishti_bonus", 0) > 0:
        notes.append("Benefic aspects (dṛṣṭi) aiding target houses.")
    if sb.get("drishti_malus", 0) > 0:
        against_list.append(_mk_reason("parashari", 0.12, "Malefic aspects (dṛṣṭi) on target houses"))
    if float(sb.get("radicality_multiplier", 1.0) or 1.0) > 1.0:
        notes.append("Radicality fit (ASC lord vs day/hour) → +10% multiplier.")

    return {
        "for": _top(for_list, 6),
        "against": _top(against_list, 6),
        "conflicts": [],
        "notes": notes[:6],
    }

def _kp_grounds_from_result(kp_res: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize KP result into for/against/conflicts/notes grounds."""
    for_list: List[Dict[str, Any]] = []
    against_list: List[Dict[str, Any]] = []
    notes: List[str] = []

    for ev in kp_res.get("signification_evidence", []) or []:
        ok = bool(ev.get("ok"))
        desc = ev.get("reason", "")
        cusp = ev.get("cusp")
        if ok:
            for_list.append(_mk_reason("kp", 0.35, f"Cusp {cusp}: {desc}"))
        else:
            against_list.append(_mk_reason("kp", 0.35, f"Cusp {cusp}: {desc} → obstructive involvement"))
        if ev.get("ssl_used"):
            notes.append(f"Cusp {cusp}: SSL tie-break used ({ev['ssl_used']}).")

    rp = (kp_res.get("kp_core") or {}).get("ruling_planets", [])
    if rp:
        notes.append(f"KP Ruling Planets: {', '.join(rp)} (minor confidence nudge).")

    return {
        "for": _top(for_list, 6),
        "against": _top(against_list, 6),
        "conflicts": [],
        "notes": notes[:6],
    }

def _fuse_grounds(step2_p: Dict[str, Any], step3_p: Dict[str, Any], step3_kp: Dict[str, Any],
                  kp_answer: str, kp_conf: float) -> Dict[str, Any]:
    """Merge grounds across Step2 (cross Parāśarī) + Step3 (Parāśarī + KP); surface conflicts."""
    for_all = (step2_p.get("for", []) + step3_p.get("for", []) + step3_kp.get("for", []))
    against_all = (step2_p.get("against", []) + step3_p.get("against", []) + step3_kp.get("against", []))
    notes = (step2_p.get("notes", []) + step3_p.get("notes", []) + step3_kp.get("notes", []))
    conflicts = (step2_p.get("conflicts", []) + step3_p.get("conflicts", []) + step3_kp.get("conflicts", []))

    # If KP is strong and contradicts Parāśarī tone, record a conflict
    if kp_answer in {"yes", "no"} and kp_conf >= 0.75:
        tone_parashari_positive = (len(for_all) >= len(against_all))
        if (kp_answer == "yes" and not tone_parashari_positive) or (kp_answer == "no" and tone_parashari_positive):
            conflicts.append(f"Strong KP '{kp_answer.upper()}' vs Parāśarī opposite tone (KP conf={kp_conf:.2f}).")

    return {
        "for": _top(for_all, 6),
        "against": _top(against_all, 6),
        "conflicts": conflicts[:4],
        "notes": notes[:6],
    }

# ---------------------------------------------------------------------------
# Builders (charts / houses)
# ---------------------------------------------------------------------------

def _build_question_bundle(inp: HybridPrasnaInput) -> Tuple[Dict[str, Any], Dict[str, Any], List[float]]:
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
        chart=q_chart, latitude=q_lat, longitude=q_lon,
        house_system=inp.house_system, zodiac_mode=inp.zodiac_mode, ayanamsa_deg=q_aya,
    )
    q_cusps = list(q_houses.get("cusps_deg", []) or [])
    return q_chart, q_houses, q_cusps

def _build_birth_bundle(b: QuerentBirthData, house_system: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[float]]:
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
        chart=b_chart, latitude=b_lat, longitude=b_lon,
        house_system=house_system, zodiac_mode=b.zodiac_mode, ayanamsa_deg=b_aya,
    )
    b_cusps = list(b_houses.get("cusps_deg", []) or [])
    return b_chart, b_houses, b_cusps

def _snapshot(chart: Dict[str, Any], cusps: List[float]) -> Dict[str, Any]:
    asc = safe_get_asc(chart)
    moon = next((p for p in chart.get("bodies", []) if p.get("name") == "Moon"), None)
    return {
        "ASC_deg": asc,
        "ASC_sign": (sign_name_from_deg(float(asc)) if asc is not None else None),
        "Moon_deg": (float(moon["longitude_deg"]) if moon else None),
        "Moon_sign": (sign_name_from_deg(float(moon["longitude_deg"])) if moon else None),
        "cusps_deg": cusps,
    }

# ---------------------------------------------------------------------------
# Frame alignment (NEW): ensure both charts are in the same zodiac for Step 2
# ---------------------------------------------------------------------------

def _shift_list(values: List[float], delta: float) -> List[float]:
    return [deg_wrap(v + delta) for v in values]

def _chart_shifted(chart: Dict[str, Any], delta: float) -> Dict[str, Any]:
    """Return a shallow-copied chart with longitudes (bodies/angles) shifted by delta degrees."""
    if not delta:
        return chart
    out = dict(chart)
    # Bodies
    bodies_new = []
    for b in chart.get("bodies", []) or []:
        b2 = dict(b)
        try:
            b2["longitude_deg"] = deg_wrap(float(b2["longitude_deg"]) + delta)
        except Exception:
            pass
        bodies_new.append(b2)
    out["bodies"] = bodies_new
    # Angles
    ang = dict(chart.get("angles", {}) or {})
    for key in ("asc_deg","mc_deg"):
        if key in ang and ang[key] is not None:
            try:
                ang[key] = deg_wrap(float(ang[key]) + delta)
            except Exception:
                pass
    if ang:
        out["angles"] = ang
    return out

def _align_zodiac_frames_for_cross(
    b_chart: Dict[str, Any], b_cusps: List[float],
    q_chart: Dict[str, Any], q_cusps: List[float],
) -> Tuple[Dict[str, Any], List[float], Dict[str, Any], List[float]]:
    """
    If birth/question charts differ in zodiac mode, convert both to a common frame.
    Policy: choose SIDEREAL if either is sidereal, else tropical.
    Conversion uses each chart's own ayanamsa_deg from meta.
    """
    b_meta = b_chart.get("meta", {}) or {}
    q_meta = q_chart.get("meta", {}) or {}
    b_mode = (b_meta.get("zodiac_mode") or "").lower() or "sidereal"
    q_mode = (q_meta.get("zodiac_mode") or "").lower() or "sidereal"
    if b_mode == q_mode:
        return b_chart, b_cusps, q_chart, q_cusps

    target = "sidereal" if ("sidereal" in (b_mode, q_mode)) else "tropical"

    # Birth transform
    b = b_chart
    bc = list(b_cusps)
    if b_mode != target:
        b_aya = float(b_meta.get("ayanamsa_deg") or 0.0)
        delta = (-b_aya) if target == "sidereal" else (+b_aya)
        b = _chart_shifted(b_chart, delta)
        bc = _shift_list(b_cusps, delta)

    # Question transform
    q = q_chart
    qc = list(q_cusps)
    if q_mode != target:
        q_aya = float(q_meta.get("ayanamsa_deg") or 0.0)
        delta = (-q_aya) if target == "sidereal" else (+q_aya)
        q = _chart_shifted(q_chart, delta)
        qc = _shift_list(q_cusps, delta)

    return b, bc, q, qc

# ---------------------------------------------------------------------------
# Step 2: Parāśarī cross-analysis (birth ↔ question)
# ---------------------------------------------------------------------------

def _house_tier(h: int, mapping: Dict[str, List[int]]) -> float:
    if h in set(mapping.get("primary", [])):    return 1.00
    if h in set(mapping.get("secondary", [])):  return 0.80
    if h in set(mapping.get("supportive", [])): return 0.60
    if h in set(mapping.get("obstructive", [])):return 0.20
    return 0.40

def _planet_weight(nm: str, tier: float, dignity: float) -> float:
    # dignity rough range ~[-2.5..+2.5] → 0..1 normalization
    dig_n = max(0.0, min(1.0, (dignity + 2.5) / 5.0))
    nud = (0.06 if nm in _BENEFICS or nm in _NEUTRAL else (-0.05 if nm in _MALEFICS else 0.0))
    return max(0.0, min(1.0, 0.55*tier + 0.35*dig_n + nud))

def _cross_score_parashari(
    birth_chart: Dict[str, Any], birth_cusps: List[float],
    q_chart: Dict[str, Any], q_cusps: List[float],
    qtype: QuestionType
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate synergy between birth and question charts (aligned to same zodiac):
      A) Question planets placed into BIRTH houses
      B) Birth planets placed into QUESTION houses
      C) Inter-chart aspects (same-name classical planets)
    Returns (score_0..1, breakdown, grounds)
    """
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    bodies_b = _with_node_aliases(birth_chart.get("bodies", []))
    bodies_q = _with_node_aliases(q_chart.get("bodies", []))

    per_item: List[Dict[str, Any]] = []
    total, count = 0.0, 0.0
    reasons_for: List[Dict[str, Any]] = []
    reasons_against: List[Dict[str, Any]] = []
    notes: List[str] = []

    # A) Question planets in BIRTH houses
    for nm, qb in bodies_q.items():
        if nm not in bodies_b:
            continue
        try:
            lon = float(qb["longitude_deg"])
        except Exception:
            continue
        h = house_of(lon, birth_cusps)
        tier = _house_tier(h, mapping)
        dig  = float(calc_dignity_rich(lon, nm))
        w    = _planet_weight(nm, tier, dig)
        per_item.append({"where":"Q→B", "planet":nm, "house":h, "tier":tier, "dignity":round(dig,3), "weight":round(w,4)})
        total += w; count += 1
        txt = f"Q {nm} in BIRTH H{h} ({sign_name_from_deg(lon)}), dignity {dig:+.2f}"
        if h in set(mapping.get("obstructive", [])):
            reasons_against.append(_mk_reason("parashari_cross", 0.45*w, txt + " in obstructive house"))
        else:
            reasons_for.append(_mk_reason("parashari_cross", 0.50*w if h in set(mapping.get("primary", [])) else 0.40*w, txt))

    # B) Birth planets in QUESTION houses
    for nm, bb in bodies_b.items():
        if nm not in bodies_q:
            continue
        try:
            lon = float(bb["longitude_deg"])
        except Exception:
            continue
        h = house_of(lon, q_cusps)
        tier = _house_tier(h, mapping)
        dig  = float(calc_dignity_rich(lon, nm))
        w    = _planet_weight(nm, tier, dig)
        per_item.append({"where":"B→Q", "planet":nm, "house":h, "tier":tier, "dignity":round(dig,3), "weight":round(w,4)})
        total += w; count += 1
        txt = f"B {nm} in QUESTION H{h} ({sign_name_from_deg(lon)}), dignity {dig:+.2f}"
        if h in set(mapping.get("obstructive", [])):
            reasons_against.append(_mk_reason("parashari_cross", 0.40*w, txt + " in obstructive house"))
        else:
            reasons_for.append(_mk_reason("parashari_cross", 0.45*w if h in set(mapping.get("secondary", [])) else 0.35*w, txt))

    # C) Inter-chart aspects (same planet ↔ same planet)
    soft_aspects = {"conjunction","trine","sextile"}
    hard_aspects = {"square","opposition"}
    for nm in list(set(bodies_b.keys()) & set(bodies_q.keys())):
        if nm not in {"Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"}:
            continue
        try:
            bL = float(bodies_b[nm]["longitude_deg"])
            qL = float(bodies_q[nm]["longitude_deg"])
        except Exception:
            continue
        sep, asp = calculate_aspects(bL, qL)
        if asp in soft_aspects:
            reasons_for.append(_mk_reason("parashari_cross", 0.15, f"{nm} {asp} across charts ({sep:.1f}°) — harmony"))
            total += 0.08; count += 1
        elif asp in hard_aspects and nm in _MALEFICS:
            reasons_against.append(_mk_reason("parashari_cross", 0.15, f"{nm} {asp} across charts ({sep:.1f}°) — stress"))
            total += 0.02; count += 1  # mild penalty (bounded)

    score = 0.0 if count == 0 else max(0.0, min(1.0, total / count))

    grounds = {
        "for": _top(reasons_for, 6),
        "against": _top(reasons_against, 6),
        "conflicts": [],
        "notes": notes[:6],
    }
    breakdown = {"items": per_item, "mean": round(score, 3), "count": int(count)}
    return score, breakdown, grounds

# ---------------------------------------------------------------------------
# Scoring normalization & final fusion
# ---------------------------------------------------------------------------

def _parashari_positivity_from_result(p_res: Dict[str, Any]) -> float:
    """
    Map Parāśarī judgement to a 0..1 positivity for fusion.
    Yes with high confidence → close to 1.0; No → close to 0.0; Uncertain → ~0.5.
    """
    j = p_res.get("judgement", {}) or {}
    ans = (j.get("answer") or "").lower()
    conf = float(j.get("confidence", 0.62) or 0.62)
    if ans == "yes":
        return min(1.0, 0.5 + 0.5*conf)
    if ans == "no":
        return max(0.0, 0.5 - 0.5*conf)
    return 0.5

def _kp_signed_signal(kp_res: Dict[str, Any]) -> Tuple[float, str, float]:
    """
    Convert KP verdict to signed signal in [-1..+1] times confidence.
    Returns (signal, answer, confidence).
    """
    j = kp_res.get("judgement", {}) or {}
    ans = (j.get("answer") or "").lower()
    conf = float(j.get("confidence", 0.62) or 0.62)
    if ans == "yes":
        return (+1.0*conf, ans, conf)
    if ans == "no":
        return (-1.0*conf, ans, conf)
    return (0.0, ans, conf)

def _fuse_scores(step2_cross_p: float, step3_parashari_p: float, step3_kp_signal: float,
                 radicality_ok: bool, rp_bonus_cap: float = 0.06, rp_count: int = 0) -> Tuple[str, float, float]:
    """
    Combine Step 2 & Step 3 into a final 0..1 positivity and textual verdict.
    """
    # Base weights (transparent & tunable)
    w_cross = 0.45
    w_p_hor = 0.30
    w_kp    = 0.25  # KP signal already in [-1..+1], we map it to 0..1 contribution below

    # Map KP signal to 0..1 positivity
    kp_pos = 0.5 + 0.5*step3_kp_signal

    # Combine
    pos = w_cross*step2_cross_p + w_p_hor*step3_parashari_p + w_kp*kp_pos

    # Radicality nudge
    if radicality_ok:
        pos = min(1.0, pos * 1.10)

    # RP (Ruling Planets) tiny bias (estimated via rp_count)
    rp_nudge = min(rp_bonus_cap, 0.02 * max(0, rp_count))
    pos = min(1.0, max(0.0, pos + rp_nudge))

    # Verdict bands
    if pos >= 0.66:
        answer, conf = "yes", 0.70 + 0.30*(pos - 0.66)/(1.0-0.66)
    elif pos <= 0.40:
        answer, conf = "no",  0.70 + 0.30*(0.40 - pos)/0.40
    else:
        answer, conf = "uncertain", 0.58 + abs(pos - 0.53) * 0.30

    return answer, round(min(0.97, conf), 3), round(pos, 3)

# ---------------------------------------------------------------------------
# Helpers to call classical engines with proper inputs
# ---------------------------------------------------------------------------

def _hi_from_birth(b: QuerentBirthData, *, qtype: QuestionType, house_system: str) -> HoraryInput:
    return HoraryInput(
        date=b.date, time=b.time, tz_name=b.tz_name,
        place=b.place, latitude=b.latitude, longitude=b.longitude,
        zodiac_mode=b.zodiac_mode, ayanamsa=b.ayanamsa,
        house_system=house_system,
        kp_house_system="placidus", kp_ayanamsa="krishnamurti",
        kp_number=None, kp_number_mode="anchor_asc",
        question_type=qtype, question_text=None, querent_house=1, quesited_house=None
    )

def _hi_from_question(inp: HybridPrasnaInput, *, qtype: QuestionType) -> HoraryInput:
    return HoraryInput(
        date=inp.question_date, time=inp.question_time, tz_name=inp.question_tz,
        place=inp.question_place, latitude=inp.question_latitude, longitude=inp.question_longitude,
        zodiac_mode=inp.zodiac_mode, ayanamsa=inp.ayanamsa,
        house_system=inp.house_system,
        kp_house_system="placidus", kp_ayanamsa="krishnamurti",
        kp_number=getattr(inp, "kp_number", None), kp_number_mode=getattr(inp, "kp_number_mode", "anchor_asc"),
        question_type=qtype, question_text=inp.question_text, querent_house=1, quesited_house=None
    )

# ---------------------------------------------------------------------------
# Public analysis
# ---------------------------------------------------------------------------

def analyze_hybrid(inp: HybridPrasnaInput) -> Dict[str, Any]:
    if not inp or not isinstance(inp, HybridPrasnaInput):
        return {"ok": False, "system": "hybrid", "error": "invalid_input"}
    if not inp.querent_birth:
        return {"ok": False, "system": "hybrid", "error": "Querent birth data required"}

    qtype = inp.question_type or QuestionType.JOB

    # --- Build charts & houses
    q_chart, q_houses, q_cusps = _build_question_bundle(inp)
    b_chart, b_houses, b_cusps = _build_birth_bundle(inp.querent_birth, inp.house_system)

    # Align frames for cross-analysis (sidereal if either is sidereal)
    b_chart_aligned, b_cusps_aligned, q_chart_aligned, q_cusps_aligned = _align_zodiac_frames_for_cross(
        b_chart, b_cusps, q_chart, q_cusps
    )

    # Snapshots (for UI/debug)
    q_snap = _snapshot(q_chart, q_cusps)
    b_snap = _snapshot(b_chart, b_cusps)

    # --- Step 1: Analyze BIRTH (Parāśarī + KP) with the SAME classical engines
    b_hi = _hi_from_birth(inp.querent_birth, qtype=qtype, house_system=inp.house_system)
    b_parashari = parashari_classical(b_hi)
    b_kp        = kp_classical(b_hi)

    # --- Step 2: Parāśarī CROSS (birth ↔ question) on ALIGNED charts
    cross_p_score, cross_break, cross_grounds = _cross_score_parashari(
        birth_chart=b_chart_aligned, birth_cusps=b_cusps_aligned,
        q_chart=q_chart_aligned, q_cusps=q_cusps_aligned,
        qtype=qtype
    )

    # --- Step 3: Analyze QUESTION (Parāśarī + KP) with classical engines
    q_hi = _hi_from_question(inp, qtype=qtype)
    q_parashari = parashari_classical(q_hi)
    q_kp        = kp_classical(q_hi)

    # --- Grounds assembly for Step 1 & 3 Parāśarī/KP (for transparency)
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    b_parashari_grounds = _parashari_grounds_from_result(b_parashari, mapping) if b_parashari.get("ok") else {"for":[], "against":[], "conflicts":[], "notes":["birth parashari error"]}
    b_kp_grounds        = _kp_grounds_from_result(b_kp) if b_kp.get("ok") else {"for":[], "against":[], "conflicts":[], "notes":["birth kp error"]}

    q_parashari_grounds = _parashari_grounds_from_result(q_parashari, mapping) if q_parashari.get("ok") else {"for":[], "against":[], "conflicts":[], "notes":["question parashari error"]}
    q_kp_grounds        = _kp_grounds_from_result(q_kp) if q_kp.get("ok") else {"for":[], "against":[], "conflicts":[], "notes":["question kp error"]}

    # --- Radicality & RP info (for fusion nudges and grounds)
    try:
        rad = radicality_flags(q_chart, inp.question_tz or "UTC",
                               inp.question_date or datetime.now(timezone.utc).date().isoformat(),
                               inp.question_time or datetime.now(timezone.utc).time().replace(microsecond=0).isoformat())
    except Exception:
        rad = {"fits": False}
    rp_list = (q_kp.get("kp_core", {}) or {}).get("ruling_planets", []) if q_kp.get("ok") else []
    rp_support_count = len(rp_list)  # small bias cap applied in fusion

    # --- Step 4: Fusion
    step2_cross_pos = float(cross_p_score)
    step3_parashari_pos = _parashari_positivity_from_result(q_parashari)
    kp_signal, kp_answer, kp_conf = _kp_signed_signal(q_kp)

    final_answer, final_conf, final_pos = _fuse_scores(
        step2_cross_p=step2_cross_pos,
        step3_parashari_p=step3_parashari_pos,
        step3_kp_signal=kp_signal,
        radicality_ok=bool(rad.get("fits")),
        rp_count=rp_support_count
    )

    # Final grounds
    fused_grounds = _fuse_grounds(
        step2_p=cross_grounds,
        step3_p=q_parashari_grounds,
        step3_kp=q_kp_grounds,
        kp_answer=kp_answer, kp_conf=kp_conf
    )

    # Return full structured response
    return {
        "ok": True,
        "system": "hybrid",
        "meta": {
            "question_type": qtype.value,
            "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
            "house_system_used": inp.house_system,
            "zodiac_mode_question": inp.zodiac_mode,
            "ayanamsa_question": inp.ayanamsa,
            "radicality": rad,
        },
        "step1_birth": {
            "parashari": b_parashari,
            "kp": b_kp,
            "grounds": {
                "parashari": b_parashari_grounds,
                "kp": b_kp_grounds,
            },
            "snapshot": b_snap,
        },
        "step2_cross_parashari": {
            "score_0_1": round(step2_cross_pos, 3),
            "breakdown": cross_break,
            "grounds": cross_grounds,
        },
        "step3_question": {
            "parashari": q_parashari,
            "kp": q_kp,
            "grounds": {
                "parashari": q_parashari_grounds,
                "kp": q_kp_grounds,
            },
            "snapshot": q_snap,
        },
        "final": {
            "answer": final_answer,
            "confidence": final_conf,
            "positivity_0_1": final_pos,
            "grounds": fused_grounds,
            "fusion_weights": {"cross_parashari": 0.45, "horary_parashari": 0.30, "kp": 0.25},
            "fusion_notes": [
                "KP signal converted to signed contribution (−1..+1) × confidence → mapped to 0..1.",
                "Radicality fit adds ~10%; KP Ruling Planets add up to +0.06 in total.",
                "If KP (high-conf) contradicts Parāśarī tone, conflict is surfaced in grounds."
            ],
        },
        "notes": {
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
