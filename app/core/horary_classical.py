# -*- coding: utf-8 -*-
"""
horary_classical.py — centralized rewrite (2025-10-01)
------------------------------------------------------
Classical Horary / Prashna systems: Parāśarī and KP (no Hybrid here).

Highlights
- Single source of truth from horary_shared (no drift).
- No double ayanāṁśa: relies on horary_shared.compute_houses_from_chart().
- Robust houses: tries app.core.houses_advanced with multiple JD signatures; safe fallback to Equal-from-ASC.
- Parāśarī (orthodox):
    • Geocentric planets, Whole-Sign default when caller omits house system.
    • Strength table (rich dignity + retro/combust + question-house relevance).
    • Sign-based graha-dṛṣṭi with upacaya tempering.
    • Moon proximity & “radicality” factor.
    • Pañcāṅga snapshot.
    • ***Rule engine integration*** (horary_rules): classical “perfection/block” outcome + timing.
      Scoring is kept as transparency-only and used as a tie-breaker for 'maybe'.
- KP:
    • Star→Sub→SSL chain; cusp decision by sub-lord with SSL tie-break.
    • KP number rotates **all** cusps.
    • Ruling Planets give a small confidence nudge.

Public API
- analyze_parashari(inp: HoraryInput) -> dict
- analyze_kp(inp: HoraryInput) -> dict
- analyze_prasna(inp, method="parashari"|"kp") -> dict
- analyze_prasna_enhanced(inp, method="parashari"|"kp") -> dict (adds {"method": ...})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone

# Shared building blocks
from .horary_shared import (
    # Dataclasses / config
    HoraryInput, QuestionType, ENHANCED_QUESTION_HOUSES, normalize_question_type,

    # Angles & zodiac
    deg_wrap, sign_name_from_deg, lord_of_sign, angular_sep, house_of,

    # KP helpers
    kp_star_sub_sub,

    # Chart & houses
    build_chart, compute_houses_from_chart, safe_get_asc,

    # Drishti
    houses_aspected_by,

    # Dignity & Pañcāṅga & radicality
    calc_dignity_rich, tithi_index, moon_star, radicality_flags,

    # Constants & helpers
    TRAD_PLANETS, COMBUST_DEG, BENEFICS, MALEFICS,
    bodies_by_name_with_nodes, get_house_lords_from_cusps, ayanamsa_from_meta,
)

# Classical rule engine
from .horary_rules import (
    RULES_PARASHARI, build_context_for_parashari, outcome_from_hits,
    timing_estimate_from_moon,
)

# =============================================================================
# Local constants (lean; prefer shared)
# =============================================================================

PARASHARI_PLANETS = list(TRAD_PLANETS)                 # Trad. 7 grahas
PLANETS_WITH_NODES = PARASHARI_PLANETS + ["Rahu", "Ketu"]

# Panchanga / Tithi labels (30)
TITHI_NAMES = [
    "Pratipada","Dvitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Purnima/Amavasya",
]*2

UPACHAYA = {3, 6, 10, 11}

# =============================================================================
# Strength table (Parāśarī)
# =============================================================================

def _house_relevance_by_question(h: int, qtype: Optional[QuestionType]) -> float:
    if not qtype or qtype not in ENHANCED_QUESTION_HOUSES:
        return 0.5
    m = ENHANCED_QUESTION_HOUSES[qtype]
    if h in m.get("primary", []):    return 1.0
    if h in m.get("secondary", []):  return 0.8
    if h in m.get("supportive", []): return 0.6
    if h in m.get("obstructive", []):return 0.2
    return 0.4


def _parashari_strengths(chart: Dict[str, Any], cusps: List[float],
                         question_type: Optional[QuestionType]) -> List[Dict[str, Any]]:
    """Per-planet strength table with rich dignity, house relevance, retro & combustion."""
    out: List[Dict[str, Any]] = []
    bodies = bodies_by_name_with_nodes(chart.get("bodies", []))
    sun_lon: Optional[float] = float(bodies["Sun"]["longitude_deg"]) if "Sun" in bodies else None
    qtype = normalize_question_type(question_type) if question_type else None

    for nm in [p for p in PLANETS_WITH_NODES if p in bodies]:
        b = bodies[nm]
        try:
            lon = float(b["longitude_deg"])
        except Exception:
            continue
        spd = float(b.get("speed_deg_per_day") or b.get("speed") or 0.0)
        h = house_of(lon, cusps)

        # Base dignity (exalt/own/MT; friend/enemy; sandhi/gandanta)
        dig = calc_dignity_rich(lon, nm)

        # Combustion (skip Sun)
        if nm != "Sun" and sun_lon is not None and nm in COMBUST_DEG:
            if angular_sep(lon, sun_lon) <= COMBUST_DEG[nm]:
                dig -= 0.3

        # Retrograde tweak
        if spd < 0:
            if nm in MALEFICS:   dig += 0.25
            elif nm in BENEFICS: dig -= 0.25

        hstr = _house_relevance_by_question(h, qtype)
        overall = (dig + hstr + 2.5) / 5.0  # compress ~0..1

        out.append({
            "planet": nm,
            "longitude_deg": round(lon, 4),
            "sign": sign_name_from_deg(lon),
            "house": h,
            "is_retrograde": spd < 0,
            "dignity_score": round(dig, 3),
            "house_strength": round(hstr, 3),
            "overall_strength": round(max(0.0, min(1.0, overall)), 4)
        })
    return out

# =============================================================================
# Parāśarī entrypoint (orthodox + rules)
# =============================================================================

def analyze_parashari(inp: HoraryInput) -> Dict[str, Any]:
    """
    Parāśarī-style reading of the question chart with:
      - Geocentric planets, Whole-Sign default (if caller omitted house system)
      - Rich dignity scoring
      - House LORD focus + house-based relevance
      - Drishti pressure (sign-based) tempered in upacaya
      - Moon proximity & radicality
      - Pañcāṅga snapshot
      - ***Rule-engine outcome (perfection/block)*** + classical timing
    """
    # Build chart — Parāśarī commonly geocentric
    chart = build_chart(
        date=inp.date, time=inp.time, tz=inp.tz_name,
        place=inp.place, latitude=inp.latitude, longitude=inp.longitude,
        zodiac_mode=inp.zodiac_mode, ayanamsa=inp.ayanamsa, topocentric=False
    )

    meta = chart.get("meta", {}) or {}
    ay_deg = float(inp.ayanamsa_deg) if isinstance(inp.ayanamsa_deg, (int, float)) else ayanamsa_from_meta(chart)
    observer = meta.get("observer") or {}
    la = float(observer.get("latitude", inp.latitude or 0.0))
    lo = float(observer.get("longitude", inp.longitude or 0.0))

    # Houses (no double-shift; Whole-Sign default)
    effective_house_system = (inp.house_system or "whole_sign")
    houses = compute_houses_from_chart(
        chart, latitude=la, longitude=lo,
        house_system=effective_house_system,
        zodiac_mode=inp.zodiac_mode,
        ayanamsa_deg=ay_deg
    )
    cusps = list(houses.get("cusps_deg", []) or [])
    house_lords: Dict[int, str] = get_house_lords_from_cusps(cusps)

    # Core data
    asc_deg = safe_get_asc(chart)
    bodies = bodies_by_name_with_nodes(chart.get("bodies", []))
    moon_lon = (float(bodies["Moon"]["longitude_deg"]) if "Moon" in bodies else None)
    sun_lon  = (float(bodies["Sun"]["longitude_deg"])  if "Sun"  in bodies else None)

    # Question profile
    qtype = normalize_question_type(inp.question_type or QuestionType.KARMA)
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    primary, secondary = mapping.get("primary", []), mapping.get("secondary", [])
    supportive, obstructive = mapping.get("supportive", []), mapping.get("obstructive", [])
    target_pos = set(primary + secondary + supportive)
    target_neg = set(obstructive)

    # Planet strengths
    strengths = _parashari_strengths(chart, cusps, qtype)

    # === Heuristic scoring (transparent-only; rules decide the final outcome) ===
    primary_score = secondary_score = obstruction_score = 0.0

    # Lords of primary/secondary houses
    for h in primary:
        lord = house_lords.get(h)
        if lord:
            lord_data = next((s for s in strengths if s["planet"] == lord), None)
            if lord_data:
                primary_score += lord_data["overall_strength"] * 4.0

    for h in secondary:
        lord = house_lords.get(h)
        if lord:
            lord_data = next((s for s in strengths if s["planet"] == lord), None)
            if lord_data:
                secondary_score += lord_data["overall_strength"] * 2.5

    # Planets by position
    for s in strengths:
        if s["house"] in primary:
            primary_score += s["overall_strength"] * 1.5
        elif s["house"] in secondary:
            secondary_score += s["overall_strength"] * 1.0
        elif s["house"] in supportive:
            secondary_score += s["overall_strength"] * 0.6
        elif s["house"] in obstructive:
            obstruction_score += s["overall_strength"] * 2.0

    # Karakas
    karaka_bonus = 0.0
    for k in mapping.get("karaka", []):
        k_data = next((s for s in strengths if s["planet"] == k), None)
        if k_data:
            if k_data["overall_strength"] > 0.65:
                karaka_bonus += 0.4
            elif k_data["overall_strength"] < 0.35:
                karaka_bonus -= 0.3

    # Drishti pressure (sign-based) + upacaya tempering
    drishti_bonus = drishti_malus = 0.0
    for nm in bodies:
        if nm not in PLANETS_WITH_NODES:
            continue
        tgt = houses_aspected_by(nm, cusps, bodies)
        if not tgt:
            continue
        hits_pos = len(tgt & target_pos)
        hits_neg = len(tgt & target_neg)
        upa_hits_pos = len((tgt & target_pos) & UPACHAYA)

        if nm in BENEFICS:
            drishti_bonus += 0.15 * hits_pos
            drishti_malus += 0.10 * hits_neg
        else:
            harmful = max(0, hits_pos - 0.5 * upa_hits_pos)  # soften in upacaya
            drishti_malus += 0.20 * harmful
            drishti_bonus += 0.03 * upa_hits_pos            # small adaptive bonus
            drishti_bonus += 0.05 * hits_neg                # malefic hitting obstructive helps a bit

    # Moon proximity bonus (to target lords)
    prox_bonus = 0.0
    if moon_lon is not None:
        for pl in {house_lords.get(h) for h in (primary + secondary)}:
            if pl and pl in bodies:
                try:
                    if angular_sep(moon_lon, float(bodies[pl]["longitude_deg"])) < 12.0:
                        prox_bonus += 1.0
                except Exception:
                    pass

    # Radicality & penalties
    date = inp.date or datetime.now(timezone.utc).date().isoformat()
    time_ = inp.time or datetime.now(timezone.utc).time().replace(microsecond=0).isoformat()
    rad = radicality_flags(chart, inp.tz_name or "UTC", date, time_)

    penalties = 0
    # Saturn in 1st
    if any(s["planet"] == "Saturn" and s["house"] == 1 for s in strengths):
        penalties += 1
    # 4+ planets in 1st
    if len([s for s in strengths if s["house"] == 1]) >= 4:
        penalties += 1
    # Debilitated lagna lord
    asc_lord = lord_of_sign(float(asc_deg or 0.0))
    if any(s["planet"] == asc_lord and s["dignity_score"] < -0.3 for s in strengths):
        penalties += 1

    rad_mult = (max(0.75, 1.10 - 0.12 * penalties) if rad.get("fits")
                else max(0.70, 0.95 - 0.10 * penalties))

    # Pañcāṅga
    t_idx = tithi_index(moon_lon, sun_lon)
    moon_star_label = moon_star(chart)

    total_positive = (primary_score + secondary_score + prox_bonus + karaka_bonus + drishti_bonus) * rad_mult
    total_negative = obstruction_score + drishti_malus
    net_score = total_positive - total_negative

    score_answer: str
    if net_score > 1.25:
        score_answer = "yes"
    elif net_score < -1.25:
        score_answer = "no"
    else:
        score_answer = "maybe"

    # === Classical RULE ENGINE ===
    ctx = build_context_for_parashari(
        chart=chart, cusps=cusps,
        house_lords=house_lords,
        mapping=mapping,
        combust_map=COMBUST_DEG,
        querent_house=inp.querent_house or 1,
    )
    rules = RULES_PARASHARI(COMBUST_DEG)
    # We call run_rules() from horary_rules indirectly via outcome_from_hits; run_rules is internal there.
    # Build hits manually by reusing its public constructor: outcome_from_hits() expects RuleHit list.
    # The RULES_PARASHARI provides Rule objects, so we replicate run_rules here to avoid re-import churn.
    # (Safer: import run_rules)
    from .horary_rules import run_rules  # local import to keep top clean
    hits = run_rules(ctx, rules)
    classical_outcome, _ = outcome_from_hits(hits)

    # Timing: Moon applying to quesited lord (1st primary house's lord), if available
    timing = None
    try:
        if moon_lon is not None and primary:
            qlord = house_lords.get(primary[0])
            if qlord and qlord in bodies:
                target_deg = float(bodies[qlord]["longitude_deg"])
                timing = timing_estimate_from_moon(moon_lon, target_deg, primary[0])
    except Exception:
        timing = None

    # Final verdict: rules first, scoring used only to disambiguate 'maybe'
    if classical_outcome in ("yes", "no"):
        final_answer = classical_outcome
        base_conf = 0.78 if classical_outcome == "yes" else 0.76
    else:
        # 'maybe' by rules → lean on score
        final_answer = score_answer
        base_conf = 0.70 if score_answer != "maybe" else 0.62

    # Small adjusts with radicality + proximity evidence
    conf = base_conf
    if rad.get("fits"): conf += 0.04
    if prox_bonus >= 1.0: conf += 0.02
    conf = float(max(0.55, min(0.96, conf)))

    return {
        "ok": True,
        "system": "parashari",
        "meta": {
            "question_type": qtype.value,
            "zodiac_mode": inp.zodiac_mode,
            "ayanamsa": inp.ayanamsa,
            "ayanamsa_deg_used": ay_deg,
            "house_system": effective_house_system,
            "route": "horary/parashari",
            "branch": "horary_core",
            "tz_normalized": inp.tz_name or "UTC",
            "analysis_time": datetime.now(timezone.utc).isoformat(),
        },
        "panchanga": {
            "tithi_index": t_idx,
            "tithi_label": (TITHI_NAMES[t_idx] if t_idx is not None else None),
            "moon_star": moon_star_label,
        },
        "chart_data": {
            "ASC_deg": asc_deg,
            "ASC_sign": sign_name_from_deg(float(asc_deg or 0.0)) if asc_deg is not None else None,
            "Moon_deg": moon_lon,
            "Moon_sign": (sign_name_from_deg(float(moon_lon)) if moon_lon is not None else None),
            "cusps_deg": cusps,
            "house_lords": house_lords,
        },
        "planetary_analysis": strengths,
        "scoring_breakdown": {
            "primary_score": round(primary_score, 3),
            "secondary_score": round(secondary_score, 3),
            "obstruction_score": round(obstruction_score, 3),
            "moon_proximity_bonus": round(prox_bonus, 3),
            "karaka_bonus": round(karaka_bonus, 3),
            "drishti_bonus": round(drishti_bonus, 3),
            "drishti_malus": round(drishti_malus, 3),
            "radicality_multiplier": rad_mult,
            "radicality_penalties": penalties,
            "net_score": round(net_score, 3),
            "score_answer": score_answer,
        },
        "rule_engine": {
            "hits": [h.__dict__ for h in hits],
            "classical_outcome": classical_outcome,
        },
        "timing": timing,  # e.g. {"arc_deg":..., "unit":"weeks", "estimate":N, "basis":"..."}
        "judgement": {"answer": final_answer, "confidence": round(conf, 3)},
    }

# =============================================================================
# KP entrypoint
# =============================================================================

def _kp_signified_houses_base(planet_name: str, chart: Dict[str, Any], cusps: List[float]) -> Set[int]:
    """Occupancy + lordship of cusp signs for the given planet."""
    houses: Set[int] = set()
    bodies = bodies_by_name_with_nodes(chart.get("bodies", []))
    p = bodies.get(planet_name)
    if not p:
        return houses
    try:
        houses.add(house_of(float(p["longitude_deg"]), cusps))
    except Exception:
        pass
    for i, cusp in enumerate(cusps, 1):
        if lord_of_sign(cusp) == planet_name:
            houses.add(i)
    return houses


def _kp_signified_houses_chain(planet_name: str, chart: Dict[str, Any], cusps: List[float]) -> Set[int]:
    """
    KP chain: planet ⇒ star-lord ⇒ sign-lord.
    Nodes act as agents of their star-/sign-lords and the planets they conjoin (±3°) or oppose (7th by sign).
    """
    bodies = bodies_by_name_with_nodes(chart.get("bodies", []))
    p = bodies.get(planet_name)
    if not p:
        return set()
    try:
        lon = float(p["longitude_deg"])
    except Exception:
        return set()

    star_lord, sub_lord, ssl, *_ = kp_star_sub_sub(lon)
    signlord = lord_of_sign(lon)

    sig: Set[int] = set()
    for who in {planet_name, star_lord, signlord}:
        sig |= _kp_signified_houses_base(who, chart, cusps)

    # Node agent rule
    if planet_name in {"Rahu", "Ketu"}:
        sig |= _kp_signified_houses_base(star_lord, chart, cusps)
        sig |= _kp_signified_houses_base(signlord, chart, cusps)
        for other, ob in bodies.items():
            if other == planet_name:
                continue
            try:
                olon = float(ob["longitude_deg"])
            except Exception:
                continue
            from .horary_shared import sign_index  # avoid circulars at top
            if angular_sep(lon, olon) <= 3.0 or (abs(sign_index(lon) - sign_index(olon)) % 12) == 6:
                sig |= _kp_signified_houses_base(other, chart, cusps)
    return sig


def _kp_cusp_sub_lord_eval(
    cusp_house: int,
    chart: Dict[str, Any],
    cusps: List[float],
    positive: Set[int],
    negative: Set[int],
) -> Tuple[bool, Set[int], str, Optional[str]]:
    """Evaluate a cusp by its sub-lord; if borderline, peek at SSL for tie-break."""
    cusp_deg = cusps[cusp_house - 1]
    star_lord, sub_lord, ssl, star_deg, pos_in_star, sub_span, pos_in_sub = kp_star_sub_sub(cusp_deg)
    sig = _kp_signified_houses_chain(sub_lord, chart, cusps)
    good = (len(sig & positive) > 0) and (len(sig & negative) == 0)

    ssl_sig = _kp_signified_houses_chain(ssl, chart, cusps)
    ssl_good = (len(ssl_sig & positive) > 0) and (len(ssl_sig & negative) == 0)
    reason = f"sub={sub_lord}, ssl={ssl}, signified={sorted(sig)}, ssl_signified={sorted(ssl_sig)}"
    chosen_ssl = ssl if (not good and ssl_good) else None
    return (good or ssl_good), (sig if good else (ssl_sig if ssl_good else sig)), reason, chosen_ssl


def _ruling_planets(chart: Dict[str, Any], tz_name: str, date: str, time_: str) -> Set[str]:
    """KP Ruling Planets: day-lord, Moon sign-lord, Moon star-lord, ASC sign-lord, ASC star-lord."""
    bodies = bodies_by_name_with_nodes(chart.get("bodies", []))

    # Day-lord
    try:
        from zoneinfo import ZoneInfo
        dt_local = datetime.fromisoformat(f"{date}T{time_}").replace(tzinfo=ZoneInfo(tz_name))
    except Exception:
        dt_local = datetime.fromisoformat(f"{date}T{time_}").replace(tzinfo=timezone.utc)
    WEEK_TO_LORD = ["Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Sun"]
    daylord = WEEK_TO_LORD[dt_local.weekday()]

    # Moon sign/star
    moon = bodies.get("Moon")
    moon_signlord = moon_starlord = None
    if moon:
        try:
            md = float(moon["longitude_deg"])
            moon_signlord = lord_of_sign(md)
            moon_starlord, *_ = kp_star_sub_sub(md)
        except Exception:
            pass

    # ASC sign/star
    asc_deg = safe_get_asc(chart) or 0.0
    asc_signlord = lord_of_sign(float(asc_deg))
    asc_starlord, *_ = kp_star_sub_sub(float(asc_deg))

    return {p for p in [daylord, moon_signlord, moon_starlord, asc_signlord, asc_starlord] if p}


def analyze_kp(inp: HoraryInput) -> Dict[str, Any]:
    """
    KP (Krishnamurti Paddhati) reading:
      - All calculations in SIDEREAL using KP ayanāṁśa (chart & houses).
      - Decision from primary cusps’ sub-lords (SSL tie-breaker).
      - Ruling Planets add small confidence bias.
    """
    kp_aya = inp.kp_ayanamsa if inp.kp_ayanamsa is not None else "krishnamurti"

    # Chart (SIDEREAL), KP typically topocentric + Placidus
    chart = build_chart(
        date=inp.date, time=inp.time, tz=inp.tz_name,
        place=inp.place, latitude=inp.latitude, longitude=inp.longitude,
        zodiac_mode="sidereal", ayanamsa=kp_aya, topocentric=True
    )
    meta = chart.get("meta", {}) or {}
    ay_deg = meta.get("ayanamsa_deg")
    observer = meta.get("observer") or {}
    la = float(observer.get("latitude", inp.latitude or 0.0))
    lo = float(observer.get("longitude", inp.longitude or 0.0))

    # Houses (KP)
    houses = compute_houses_from_chart(
        chart, latitude=la, longitude=lo,
        house_system=inp.kp_house_system,
        zodiac_mode="sidereal", ayanamsa_deg=ay_deg
    )
    cusps = list(houses.get("cusps_deg", []) or [])
    asc_deg_sid = float(houses.get("asc_deg") or safe_get_asc(chart) or 0.0)

    # KP number anchoring (rotate cusps to align ASC)
    if inp.kp_number and (inp.kp_number_mode or "anchor_asc").lower() == "anchor_asc":
        target = (max(1, min(249, int(inp.kp_number))) - 1) * (360.0 / 249.0)
        delta = deg_wrap(target - asc_deg_sid)
        cusps = [deg_wrap(c + delta) for c in cusps]
        asc_deg_sid = target

    # Decision via cusp sub-lords on primary houses
    qtype = normalize_question_type(inp.question_type or QuestionType.KARMA)
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    pos_h = set(mapping.get("primary", []) + mapping.get("secondary", []) + mapping.get("supportive", []))
    neg_h = set(mapping.get("obstructive", []))

    yes_hits = no_hits = 0
    evidence = []
    for h in mapping.get("primary", []):
        ok, sig, reason, ssl_used = _kp_cusp_sub_lord_eval(h, chart, cusps, pos_h, neg_h)
        evidence.append({"cusp": h, "ok": ok, "reason": reason, "ssl_used": ssl_used})
        yes_hits += int(ok)
        no_hits += int(not ok)

    # Ruling Planets bias
    date = inp.date or datetime.now(timezone.utc).date().isoformat()
    time_ = inp.time or datetime.now(timezone.utc).time().replace(microsecond=0).isoformat()
    rp = _ruling_planets(chart, inp.tz_name or "UTC", date, time_)
    rp_support = 0
    for p in rp:
        sig = _kp_signified_houses_chain(p, chart, cusps)
        if (sig & pos_h) and not (sig & neg_h):
            rp_support += 1

    # Verdict
    if yes_hits > no_hits:
        base_conf = 0.70 + 0.05 * (yes_hits - no_hits)
        base_ans = "yes"
    elif no_hits > yes_hits:
        base_conf = 0.70 + 0.05 * (no_hits - yes_hits)
        base_ans = "no"
    else:
        base_conf = 0.62
        base_ans = "maybe"

    conf = min(0.97, base_conf + min(0.06, 0.02 * rp_support))

    # Cusp table with star/sub/ssl
    detailed_cusps = []
    for i, cusp_deg in enumerate(cusps, 1):
        star_lord, sub_lord, ssl, star_deg, pos_in_star, sub_span, pos_in_sub = kp_star_sub_sub(cusp_deg)
        detailed_cusps.append({
            "house": i,
            "degree": round(cusp_deg, 4),
            "sign": sign_name_from_deg(cusp_deg),
            "star_lord": star_lord,
            "sub_lord": sub_lord,
            "sub_sub_lord": ssl,
            "star_span_deg": round(star_deg, 4),
            "pos_within_star_deg": round(pos_in_star, 4),
            "sub_span_deg": round(sub_span, 4),
            "pos_within_sub_deg": round(pos_in_sub, 4),
        })

    # Planetary star/sub snapshot (trad planets only for concise KP table)
    planetary_kp = []
    for b in chart.get("bodies", []):
        nm = b.get("name")
        if nm not in PARASHARI_PLANETS:
            continue
        try:
            lon = float(b["longitude_deg"])
        except Exception:
            continue
        star_lord, sub_lord, ssl, star_deg, pos_in_star, sub_span, pos_in_sub = kp_star_sub_sub(lon)
        planetary_kp.append({
            "planet": nm,
            "longitude_deg": round(lon, 4),
            "sign": sign_name_from_deg(lon),
            "star_lord": star_lord,
            "sub_lord": sub_lord,
            "sub_sub_lord": ssl,
            "star_span_deg": round(star_deg, 4),
            "pos_within_star_deg": round(pos_in_star, 4),
            "sub_span_deg": round(sub_span, 4),
            "pos_within_sub_deg": round(pos_in_sub, 4),
        })

    return {
        "ok": True,
        "system": "kp",
        "meta": {
            "question_type": qtype.value,
            "ayanamsa": kp_aya,
            "house_system": inp.kp_house_system,
            "kp_number": inp.kp_number,
            "kp_number_mode": inp.kp_number_mode,
            "analysis_time": datetime.now(timezone.utc).isoformat(),
            "route": "horary/kp",
            "branch": "horary_core",
            "tz_normalized": inp.tz_name or "UTC",
        },
        "kp_core": {
            "ASC_deg_sid": asc_deg_sid,
            "ASC_sign": sign_name_from_deg(float(asc_deg_sid or 0.0)),
            "ruling_planets": sorted(list(rp)),
        },
        "cusp_analysis": detailed_cusps,
        "planetary_kp": planetary_kp,
        "signification_evidence": evidence,
        "judgement": {
            "answer": base_ans,
            "confidence": round(conf, 3),
            "method": "KP cusp sub-lords (+SSL & RP bias)",
        },
    }

# =============================================================================
# Public dispatcher
# =============================================================================

def analyze_prasna(inp: HoraryInput, method: str = "parashari") -> Dict[str, Any]:
    """
    method in {"parashari","kp"}.
    (Hybrid is intentionally NOT implemented here.)
    """
    try:
        m = (method or "parashari").strip().lower()
        if m == "kp":
            return analyze_kp(inp)
        return analyze_parashari(inp)
    except Exception as e:
        return {"ok": False, "system": method, "error": str(e), "error_type": type(e).__name__}


def analyze_prasna_enhanced(inp: HoraryInput, method: str = "parashari"):
    """
    Thin wrapper so routes can import analyze_prasna_enhanced().
    Delegates to analyze_prasna() and adds a 'method' key for convenience.
    """
    res = analyze_prasna(inp, method=method)
    if isinstance(res, dict):
        res.setdefault("method", (method or "").lower())
    return res


__all__ = [
    "HoraryInput", "QuestionType", "ENHANCED_QUESTION_HOUSES",
    "analyze_prasna_enhanced",
    "analyze_prasna", "analyze_parashari", "analyze_kp",
]
