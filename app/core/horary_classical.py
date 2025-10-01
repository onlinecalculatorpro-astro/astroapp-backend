# -*- coding: utf-8 -*-
"""
horary_classical.py — centralized rewrite (2025-09-30, patched)
----------------------------------------------------------------
Classical Horary / Prashna systems: Parāśarī and KP (no Hybrid here).

Key points & patches
- **Single source of truth**: pulls core helpers from horary_shared to avoid drift.
- **No double ayanāṁśa**: relies on horary_shared.compute_houses_from_chart() behavior
  (advanced engine gets shifted if sidereal; Whole-Sign/Equal fallbacks use chart’s frame as-is).
- **Robust houses**: tries app.core.houses_advanced with multiple JD signatures; safe fallback to Equal-from-ASC.
- **Parāśarī (orthodox tweaks)**:
    • Geocentric planets, Whole-Sign default when caller omits house system.
    • Rich dignity scoring via shared.calc_dignity_rich (exalt/own/mūlatrikoṇa, friend/enemy,
      sandhi, gandānta), then retro/combust tweaks and house relevance by question type.
    • Sign-based graha-dṛṣṭi using shared.houses_aspected_by with light upacaya tempering.
    • Moon proximity boost to lords of target houses; “radicality” multiplier via shared.radicality_flags.
    • Pañcāṅga snapshot (tithi & Moon nakṣatra star-lord).
    • Verdict labels use “yes / no / maybe” (not “uncertain”).
- **KP**:
    • Nakṣatra star → sub → sub-sub (SSL) via shared.kp_star_sub_sub.
    • Significator chain: planet ⇒ star-lord ⇒ sign-lord (nodes act as agents).
    • Primary cusp evaluation by sub-lord with SSL tie-breaker.
    • KP number anchoring rotates **all cusps** (not just ASC label).
    • Ruling Planets (day-lord, Moon sign/star, ASC sign/star) add a small confidence nudge.

Public API
- analyze_parashari(inp: HoraryInput) -> dict
- analyze_kp(inp: HoraryInput) -> dict
- analyze_prasna(inp, method="parashari"|"kp") -> dict
- analyze_prasna_enhanced(inp, method="parashari"|"kp") -> dict (adds {"method": ...})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone

# Import centralized building blocks from the shared module
from .horary_shared import (
    # Dataclasses / config
    HoraryInput, QuestionType, ENHANCED_QUESTION_HOUSES, normalize_question_type,

    # Zodiac & angles
    deg_wrap, sign_name_from_deg, lord_of_sign, angular_sep, house_of,

    # KP helpers
    kp_star_sub_sub,

    # Chart & houses
    build_chart, compute_houses_from_chart, safe_get_asc,

    # Drishti
    houses_aspected_by,

    # Dignity & panchanga & radicality
    calc_dignity_rich, tithi_index, moon_star, radicality_flags,
)

# =============================================================================
# Local constants (kept minimal; defer to shared data where possible)
# =============================================================================

PARASHARI_PLANETS = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
PLANETS_WITH_NODES = PARASHARI_PLANETS + ["Rahu", "Ketu"]

# Combustion thresholds (deg from Sun) — conservative defaults
COMBUST_DEG = {
    "Moon": 12.0, "Mercury": 12.0, "Venus": 10.0,
    "Mars": 17.0, "Jupiter": 11.0, "Saturn": 15.0,
}

BENEFICS: Set[str] = {"Jupiter", "Venus", "Moon"}            # Mercury often neutral
MALEFICS: Set[str] = {"Saturn", "Mars", "Sun", "Rahu", "Ketu"}  # Sun mild malefic; nodes malefic in horary

# Panchanga / Tithi labels (30)
TITHI_NAMES = [
    "Pratipada","Dvitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Purnima/Amavasya",
]*2

UPACHAYA = {3, 6, 10, 11}

# =============================================================================
# Local helpers
# =============================================================================

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


def _kp_signified_houses_base(planet_name: str, chart: Dict[str, Any], cusps: List[float]) -> Set[int]:
    """Occupancy + lordship of cusp signs for the given planet."""
    houses: Set[int] = set()
    bodies = _with_node_aliases(chart.get("bodies", []))
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
    bodies = _with_node_aliases(chart.get("bodies", []))
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
            # tight conj or sign opposition
            from .horary_shared import sign_index  # defer import
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
    bodies = _with_node_aliases(chart.get("bodies", []))

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

# =============================================================================
# Internal helpers — Parāśarī
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
    bodies = _with_node_aliases(chart.get("bodies", []))
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

        # Base dignity from shared (exalt/own/MT, friends/enemies, sandhi, gandānta)
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
        is_retro = spd < 0
        overall = (dig + hstr + 2.5) / 5.0  # compress to ~0..1

        out.append({
            "planet": nm,
            "longitude_deg": round(lon, 4),
            "sign": sign_name_from_deg(lon),
            "house": h,
            "is_retrograde": is_retro,
            "dignity_score": round(dig, 3),
            "house_strength": round(hstr, 3),
            "overall_strength": round(max(0.0, min(1.0, overall)), 4)
        })
    return out

# =============================================================================
# Parāśarī entrypoint (orthodox adjustments)
# =============================================================================

def analyze_parashari(inp: HoraryInput) -> Dict[str, Any]:
    """
    Parāśarī-style reading of the question chart with:
      - Geocentric planets, Whole-Sign default (if caller omitted house system)
      - Rich dignity scoring
      - House LORDSHIP evaluation (primary focus)
      - House-based relevance for the asked topic
      - Benefic/malefic drishti pressure (tempered in upacaya)
      - Moon proximity & radicality penalties
      - Karaka (natural significator) evaluation
      - Pañcanga snapshot
    """
    # Build question chart — orthodox Parāśarī prefers geocentric planets
    chart = build_chart(
        date=inp.date, time=inp.time, tz=inp.tz_name,
        place=inp.place, latitude=inp.latitude, longitude=inp.longitude,
        zodiac_mode=inp.zodiac_mode, ayanamsa=inp.ayanamsa, topocentric=False
    )

    meta = chart.get("meta", {}) or {}
    ay_from_chart = meta.get("ayanamsa_deg")
    ay_deg = float(inp.ayanamsa_deg) if isinstance(inp.ayanamsa_deg, (int, float)) else ay_from_chart
    observer = meta.get("observer") or {}
    la = float(observer.get("latitude", inp.latitude or 0.0))
    lo = float(observer.get("longitude", inp.longitude or 0.0))

    # Houses/cusps (uses shared semantics; no double shift). Default Whole-Sign if unspecified.
    effective_house_system = (inp.house_system or "whole_sign")
    houses = compute_houses_from_chart(
        chart, latitude=la, longitude=lo,
        house_system=effective_house_system,
        zodiac_mode=inp.zodiac_mode,
        ayanamsa_deg=ay_deg
    )
    cusps = list(houses.get("cusps_deg", []) or [])

    # Core angles + bodies
    asc_deg = safe_get_asc(chart)
    bodies = _with_node_aliases(chart.get("bodies", []))
    moon_lon = (float(bodies["Moon"]["longitude_deg"]) if "Moon" in bodies else None)
    sun_lon  = (float(bodies["Sun"]["longitude_deg"])  if "Sun"  in bodies else None)

    qtype = normalize_question_type(inp.question_type or QuestionType.JOB)
    strengths = _parashari_strengths(chart, cusps, qtype)

    # House lords (by cusp signs)
    house_lords: Dict[int, str] = {i+1: lord_of_sign(c) for i, c in enumerate(cusps)}

    # Get question house mapping
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    primary, secondary = mapping.get("primary", []), mapping.get("secondary", [])
    supportive, obstructive = mapping.get("supportive", []), mapping.get("obstructive", [])
    target_pos = set(primary + secondary + supportive)
    target_neg = set(obstructive)

    # === Scoring ===
    primary_score = secondary_score = obstruction_score = 0.0

    # 1) Lords of primary/secondary houses: primary significators
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

    # 2) Planets by house POSITION (lighter weight)
    for s in strengths:
        if s["house"] in primary:
            primary_score += s["overall_strength"] * 1.5
        elif s["house"] in secondary:
            secondary_score += s["overall_strength"] * 1.0
        elif s["house"] in supportive:
            secondary_score += s["overall_strength"] * 0.6
        elif s["house"] in obstructive:
            obstruction_score += s["overall_strength"] * 2.0

    # 3) Karaka (natural significators)
    karaka_bonus = 0.0
    for k in mapping.get("karaka", []):
        k_data = next((s for s in strengths if s["planet"] == k), None)
        if k_data:
            if k_data["overall_strength"] > 0.65:
                karaka_bonus += 0.4
            elif k_data["overall_strength"] < 0.35:
                karaka_bonus -= 0.3

    # 4) Drishti pressure (sign-based via shared.houses_aspected_by) with upacaya tempering
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
            # Benefic aspecting positive targets is good; aspecting obstructive houses slightly dampens
            drishti_bonus += 0.15 * hits_pos
            drishti_malus += 0.10 * hits_neg
        else:
            # Malefic aspecting positive targets is harmful, but upacaya softens the blow
            # Reduce malus by ~50% for hits that land in upacaya houses; give a tiny adaptive bonus
            harmful = max(0, hits_pos - 0.5 * upa_hits_pos)
            drishti_malus += 0.20 * harmful
            drishti_bonus += 0.03 * upa_hits_pos  # growth-through-challenge flavor
            # Malefic hitting obstructive houses helps a little
            drishti_bonus += 0.05 * hits_neg

    # 5) Moon proximity bonus to lords of target houses
    prox_bonus = 0.0
    if moon_lon is not None:
        for pl in {house_lords.get(h) for h in (primary + secondary)}:
            if pl and pl in bodies:
                try:
                    if angular_sep(moon_lon, float(bodies[pl]["longitude_deg"])) < 12.0:
                        prox_bonus += 1.0
                except Exception:
                    pass

    # 6) Radicality (planetary hour/day & chart-condition flags) + penalties
    date = inp.date or datetime.now(timezone.utc).date().isoformat()
    time_ = inp.time or datetime.now(timezone.utc).time().replace(microsecond=0).isoformat()
    rad = radicality_flags(chart, inp.tz_name or "UTC", date, time_)

    penalties = 0
    # Saturn in 1st house
    if any(s["planet"] == "Saturn" and s["house"] == 1 for s in strengths):
        penalties += 1
    # Overcrowding (4+ planets in 1st)
    if len([s for s in strengths if s["house"] == 1]) >= 4:
        penalties += 1
    # Debilitated Lagna lord
    asc_lord = lord_of_sign(float(asc_deg or 0.0))
    if any(s["planet"] == asc_lord and s["dignity_score"] < -0.3 for s in strengths):
        penalties += 1

    # Apply radicality multiplier with penalties
    if rad.get("fits"):
        rad_mult = max(0.75, 1.10 - 0.12 * penalties)
    else:
        rad_mult = max(0.70, 0.95 - 0.10 * penalties)

    # Pañcāṅga snapshot
    t_idx = tithi_index(moon_lon, sun_lon)
    moon_star_label = moon_star(chart)

    # Final aggregate
    total_positive = (primary_score + secondary_score + prox_bonus + karaka_bonus + drishti_bonus) * rad_mult
    total_negative = obstruction_score + drishti_malus
    net_score = total_positive - total_negative

    if net_score > 1.25:
        answer, conf = "yes", min(0.96, 0.74 + 0.06 * net_score)
    elif net_score < -1.25:
        answer, conf = "no",  min(0.96, 0.74 + 0.06 * abs(net_score))
    else:
        answer, conf = "maybe", 0.62

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
        },
        "judgement": {"answer": answer, "confidence": round(conf, 3)},
    }

# =============================================================================
# KP entrypoint
# =============================================================================

def analyze_kp(inp: HoraryInput) -> Dict[str, Any]:
    """
    KP (Krishnamurti Paddhati) reading:
      - All calculations in SIDEREAL using KP ayanāṁśa (chart & houses).
      - Decision from primary cusps’ sub-lords (SSL tie-breaker).
      - Ruling Planets add small confidence bias.
    """
    kp_aya = inp.kp_ayanamsa if inp.kp_ayanamsa is not None else "krishnamurti"

    # Chart (SIDEREAL) with KP ayanamsa; KP commonly uses Placidus & topocentric in practice
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

    # Houses (KP system, sidereal)
    houses = compute_houses_from_chart(
        chart, latitude=la, longitude=lo,
        house_system=inp.kp_house_system,
        zodiac_mode="sidereal", ayanamsa_deg=ay_deg
    )
    cusps = list(houses.get("cusps_deg", []) or [])
    asc_deg_sid = float(houses.get("asc_deg") or safe_get_asc(chart) or 0.0)

    # KP number anchoring (rotate cusps so ASC equals number-derived degree)
    if inp.kp_number and (inp.kp_number_mode or "anchor_asc").lower() == "anchor_asc":
        target = (max(1, min(249, int(inp.kp_number))) - 1) * (360.0 / 249.0)
        delta = deg_wrap(target - asc_deg_sid)
        cusps = [deg_wrap(c + delta) for c in cusps]
        asc_deg_sid = target

    # Decision via cusp sub-lords on primary houses
    qtype = normalize_question_type(inp.question_type or QuestionType.JOB)
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

    # Planetary star/sub snapshot (trad planets for a concise KP table)
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
