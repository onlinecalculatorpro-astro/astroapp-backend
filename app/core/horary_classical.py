# -*- coding: utf-8 -*-
"""
horary_classical.py — centralized rewrite (2025-09-29)
------------------------------------------------------
Classical Horary / Prashna systems: Parāśarī and KP (no Hybrid here).

Key features
- Sidereal-correct houses: when zodiac_mode == "sidereal", cusps/angles shift by ayanāṁśa
  so lords, KP partitions, and house tests remain consistent.
- Whole-Sign support: house_system="whole_sign" makes houses = signs from Lagna.
- Robust houses: tries app.core.houses_advanced with multiple JD signatures; safe fallback to Equal-from-ASC.
- Parāśarī:
    • Rich dignity scoring (exalt/own/mūlatrikoṇa, friend/enemy, sandhi, retro, combustion),
      normalized to 0..1 per-planet.
    • Vedic graha-dṛṣṭi (sign-based) adds benefic/malefic pressure to target houses.
    • Moon proximity boost to lords of target houses; light “radicality” multiplier.
    • Pañcāṅga snapshot (tithi index & Moon nakṣatra star-lord).
- KP:
    • Nakṣatra star → sub → sub-sub (SSL) partition.
    • Significator chain: planet ⇒ star-lord ⇒ sign-lord (nodes act as agents).
    • Primary cusp evaluation by sub-lord with SSL tie-breaker.
    • KP number anchoring rotates cusps (not only the ASC label).
    • Ruling Planets (day-lord, Moon sign/star, ASC sign/star) add small confidence bias.

Public API
- analyze_parashari(inp: HoraryInput) -> dict
- analyze_kp(inp: HoraryInput) -> dict
- analyze_prasna_enhanced(inp, method="parashari") -> dict
- analyze_prasna(inp, method="parashari"|"kp") -> dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone

# Import centralized building blocks from the shared module
from .horary_shared import (
    # Dataclasses / config
    HoraryInput, QuestionType, ENHANCED_QUESTION_HOUSES, normalize_question_type,

    # Zodiac & angles
    deg_wrap, sign_index, sign_name_from_deg, lord_of_sign, angular_sep,
    house_of,

    # Chart & houses
    build_chart, compute_houses_from_chart,

    # Helpers
    radicality_flags,
)

# =============================================================================
# Local constants (kept here so we don't need to modify horary_shared again)
# =============================================================================

PARASHARI_PLANETS = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"]
PLANETS_WITH_NODES = PARASHARI_PLANETS + ["Rahu","Ketu"]

# Mūlatrikoṇa (sign indices)
MOOLATRIKONA = {
    "Sun": 4,      # Leo
    "Moon": None,  # often omitted in fixed-sign scoring
    "Mars": 0,     # Aries
    "Mercury": 5,  # Virgo
    "Jupiter": 8,  # Sagittarius
    "Venus": 1,    # Taurus
    "Saturn": 10,  # Aquarius
}

# Permanent friendship (simplified)
FRIENDS = {
    "Sun": {"Moon","Mars","Jupiter"},
    "Moon": {"Sun","Mercury"},
    "Mars": {"Sun","Moon","Jupiter"},
    "Mercury": {"Sun","Venus"},
    "Jupiter": {"Sun","Moon","Mars"},
    "Venus": {"Mercury","Saturn"},
    "Saturn": {"Mercury","Venus"},
}
ENEMIES = {
    "Sun": {"Venus","Saturn"},
    "Moon": set(),
    "Mars": {"Mercury"},
    "Mercury": {"Moon"},
    "Jupiter": {"Venus","Mercury"},
    "Venus": {"Sun","Moon"},
    "Saturn": {"Sun","Moon"},
}

# Combustion thresholds (deg from Sun)
COMBUST_DEG = {"Moon": 12.0,"Mercury": 12.0,"Venus": 10.0,"Mars": 17.0,"Jupiter": 11.0,"Saturn": 15.0}

BENEFICS = {"Jupiter","Venus","Moon"}               # Mercury often neutral
MALEFICS = {"Saturn","Mars","Sun","Rahu","Ketu"}    # Sun mild malefic; nodes malefic in horary

# Panchanga / Tithi labels (30)
TITHI_NAMES = [
    "Pratipada","Dvitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dwadashi","Trayodashi","Chaturdashi","Purnima/Amavasya",
]*2

# KP star order & lengths (to build SSL locally)
STAR_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
KP_DASHA_YEARS = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
STAR_LEN_DEG = 360.0 / 27.0
TOTAL_DASHA = 120.0

# =============================================================================
# Local helpers needed only by this module (kept self-contained)
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

def is_sandhi(deg: float, window: float = 1.0) -> bool:
    """Near sign boundary within `window` degrees."""
    d = deg_wrap(deg) % 30.0
    return (d < window) or (30.0 - d < window)

def kp_star_sub_sub(ecl_deg_sidereal: float) -> Tuple[str, str, str, float, float, float, float]:
    """
    Build star → sub → sub-sub (SSL) partition using Vimshottari proportions.
    Returns (star_lord, sub_lord, ssl_lord, star_span, pos_in_star, sub_span, pos_in_sub).
    """
    pos = deg_wrap(ecl_deg_sidereal)
    star_idx = int(pos // STAR_LEN_DEG)
    star_lord = STAR_ORDER[star_idx % 9]
    pos_in_star = pos - STAR_LEN_DEG * star_idx

    # Sub level
    cycle = STAR_ORDER[STAR_ORDER.index(star_lord):] + STAR_ORDER[:STAR_ORDER.index(star_lord)]
    acc = 0.0
    sub_lord = STAR_ORDER[-1]
    sub_start = 0.0
    sub_span = STAR_LEN_DEG
    for lord in cycle:
        portion = STAR_LEN_DEG * (KP_DASHA_YEARS[lord] / TOTAL_DASHA)
        if pos_in_star < acc + portion:
            sub_lord, sub_start, sub_span = lord, acc, portion
            break
        acc += portion
    pos_in_sub = pos_in_star - sub_start

    # Sub-sub from sub_lord
    cycle2 = STAR_ORDER[STAR_ORDER.index(sub_lord):] + STAR_ORDER[:STAR_ORDER.index(sub_lord)]
    acc2 = 0.0
    ssl = STAR_ORDER[-1]
    for lord in cycle2:
        portion2 = sub_span * (KP_DASHA_YEARS[lord] / TOTAL_DASHA)
        if pos_in_sub < acc2 + portion2:
            ssl = lord
            break
        acc2 += portion2

    return (star_lord, sub_lord, ssl, STAR_LEN_DEG, pos_in_star, sub_span, pos_in_sub)

def calc_dignity_rich(long_deg: float, planet: str) -> float:
    """
    Rich dignity score (rough range before normalization ~ [-2.5..+2.5]).
    Components:
      +2.0 exaltation; -2.0 fall; +1.0 own sign; +0.75 moolatrikona
      +0.5 friend sign / -0.5 enemy sign
      -0.2 sandhi penalty
    (Retrograde & combustion applied outside with context.)
    """
    sidx = sign_index(long_deg)
    dign = 0.0

    exalt_sign = {"Sun":0,"Moon":2,"Mars":9,"Mercury":5,"Jupiter":3,"Venus":11,"Saturn":6}
    if planet in exalt_sign and sidx == exalt_sign[planet]:
        dign += 2.0
    elif planet in exalt_sign and sidx == (exalt_sign[planet] + 6) % 12:
        dign -= 2.0

    own = {"Sun":[4], "Moon":[3], "Mars":[0,7], "Mercury":[2,5],
           "Jupiter":[8,11], "Venus":[1,6], "Saturn":[9,10]}
    if planet in own and sidx in own[planet]:
        dign += 1.0

    if MOOLATRIKONA.get(planet) is not None and sidx == MOOLATRIKONA[planet]:
        dign += 0.75

    lord = lord_of_sign(long_deg)
    if planet in FRIENDS and lord in FRIENDS[planet]:
        dign += 0.5
    if planet in ENEMIES and lord in ENEMIES[planet]:
        dign -= 0.5

    if is_sandhi(long_deg, 1.0):
        dign -= 0.2

    return dign

def graha_drishti_offsets(planet: str, include_nodes_special: bool = True) -> Set[int]:
    """
    Vedic drishti (sign-based) as house offsets:
    Everyone: 7th; Mars: 4th & 8th; Jupiter: 5th & 9th; Saturn: 3rd & 10th;
    Nodes optional: 5th & 9th in some traditions.
    """
    base = {7}
    if planet == "Mars": base |= {4, 8}
    elif planet == "Jupiter": base |= {5, 9}
    elif planet == "Saturn": base |= {3, 10}
    elif include_nodes_special and planet in {"Rahu","Ketu"}:
        base |= {5, 9}
    return base

def houses_aspected_by(planet_name: str, cusps: List[float], bodies: Dict[str, Any]) -> Set[int]:
    """Sign-based drishti targets using current house structure."""
    p = bodies.get(planet_name)
    if not p: return set()
    try:
        lon = float(p.get("longitude_deg"))
    except Exception:
        return set()
    h = house_of(lon, cusps)
    targets: Set[int] = set()
    for off in graha_drishti_offsets(planet_name):
        t = ((h - 1 + (off - 1)) % 12) + 1
        targets.add(t)
    return targets

def tithi_index(moon_deg: Optional[float], sun_deg: Optional[float]) -> Optional[int]:
    """Return tithi index 0..29 from Moon-Sun elongation; None if unavailable."""
    if moon_deg is None or sun_deg is None:
        return None
    el = deg_wrap(float(moon_deg) - float(sun_deg))
    return int(el // 12.0)  # 30 tithis

def moon_star(chart: Dict[str, Any]) -> Optional[str]:
    """Return Moon nakshatra (star-lord) name via SSL helper (assumes sidereal Moon)."""
    moon = next((b for b in chart.get("bodies", []) if b.get("name") == "Moon"), None)
    if not moon: return None
    star, *_ = kp_star_sub_sub(float(moon["longitude_deg"]))
    return star

def rotate_cusps_to_target_asc(cusps: List[float], asc_current: float, asc_target: float) -> List[float]:
    """Rotate all cusp degrees so that ASC becomes asc_target (deg)."""
    delta = deg_wrap(asc_target - asc_current)
    return [deg_wrap(c + delta) for c in cusps]

# =============================================================================
# Internal helpers — Parāśarī
# =============================================================================

def _parashari_strengths(chart: Dict[str, Any], cusps: List[float],
                         question_type: Optional[QuestionType]) -> List[Dict[str, Any]]:
    """Per-planet strength table with rich dignity, house relevance, retro & combustion."""
    out: List[Dict[str, Any]] = []
    bodies = _with_node_aliases(chart.get("bodies", []))
    sun_lon: Optional[float] = float(bodies["Sun"]["longitude_deg"]) if "Sun" in bodies else None

    qtype = normalize_question_type(question_type) if question_type else None

    def _house_relevance(h: int) -> float:
        if not qtype or qtype not in ENHANCED_QUESTION_HOUSES:
            return 0.5
        m = ENHANCED_QUESTION_HOUSES[qtype]
        if h in m.get("primary", []):    return 1.0
        if h in m.get("secondary", []):  return 0.8
        if h in m.get("supportive", []): return 0.6
        if h in m.get("obstructive", []):return 0.2
        return 0.4

    for nm in [p for p in PLANETS_WITH_NODES if p in bodies]:
        b = bodies[nm]
        try:
            lon = float(b["longitude_deg"])
        except Exception:
            continue
        spd = float(b.get("speed_deg_per_day") or b.get("speed") or 0.0)
        h = house_of(lon, cusps)

        # Base dignity
        dig = calc_dignity_rich(lon, nm)

        # Combustion (skip Sun)
        if nm != "Sun" and sun_lon is not None and nm in COMBUST_DEG:
            if angular_sep(lon, sun_lon) <= COMBUST_DEG[nm]:
                dig -= 0.3

        # Retrograde tweak
        if spd < 0:
            if nm in MALEFICS:   dig += 0.25
            elif nm in BENEFICS: dig -= 0.25

        hstr = _house_relevance(h)
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
# Parāśarī entrypoint
# =============================================================================

def analyze_parashari(inp: HoraryInput) -> Dict[str, Any]:
    # Build question chart
    chart = build_chart(
        date=inp.date, time=inp.time, tz=inp.tz_name,
        place=inp.place, latitude=inp.latitude, longitude=inp.longitude,
        zodiac_mode=inp.zodiac_mode, ayanamsa=inp.ayanamsa, topocentric=True
    )

    meta = chart.get("meta", {}) or {}
    ay_from_chart = meta.get("ayanamsa_deg")
    ay_deg = float(inp.ayanamsa_deg) if isinstance(inp.ayanamsa_deg, (int, float)) else ay_from_chart
    observer = meta.get("observer") or {}
    la = float(observer.get("latitude", inp.latitude or 0.0))
    lo = float(observer.get("longitude", inp.longitude or 0.0))

    # Houses/cusps
    houses = compute_houses_from_chart(
        chart, latitude=la, longitude=lo,
        house_system=inp.house_system,
        zodiac_mode=inp.zodiac_mode,
        ayanamsa_deg=ay_deg
    )
    cusps = list(houses.get("cusps_deg", []) or [])

    # Core angles + bodies
    asc_deg = chart.get("angles", {}).get("asc_deg", chart.get("asc_deg"))
    bodies = _with_node_aliases(chart.get("bodies", []))
    moon_lon = (float(bodies["Moon"]["longitude_deg"]) if "Moon" in bodies else None)
    sun_lon  = (float(bodies["Sun"]["longitude_deg"])  if "Sun"  in bodies else None)

    qtype = normalize_question_type(inp.question_type or QuestionType.JOB)
    strengths = _parashari_strengths(chart, cusps, qtype)

    # House lords (by cusp signs)
    house_lords: Dict[int, str] = {i+1: lord_of_sign(c) for i, c in enumerate(cusps)}

    # Drishti pressure
    mapping = ENHANCED_QUESTION_HOUSES.get(qtype, {})
    target_pos = set(mapping.get("primary", []) + mapping.get("secondary", []) + mapping.get("supportive", []))
    target_neg = set(mapping.get("obstructive", []))

    drishti_bonus = drishti_malus = 0.0
    for nm in bodies:
        if nm not in PLANETS_WITH_NODES:
            continue
        tgt = houses_aspected_by(nm, cusps, bodies)
        hits_pos = len(tgt & target_pos)
        hits_neg = len(tgt & target_neg)
        if nm in BENEFICS:
            drishti_bonus += 0.15 * hits_pos
            drishti_malus += 0.10 * hits_neg
        else:
            drishti_malus += 0.20 * hits_pos
            drishti_bonus += 0.05 * hits_neg

    # Score buckets from planetary placements
    primary, secondary = mapping.get("primary", []), mapping.get("secondary", [])
    supportive, obstructive = mapping.get("supportive", []), mapping.get("obstructive", [])

    primary_score = secondary_score = obstruction_score = 0.0
    for s in strengths:
        if s["house"] in primary:      primary_score   += s["overall_strength"] * 3.0
        elif s["house"] in secondary:  secondary_score += s["overall_strength"] * 2.0
        elif s["house"] in supportive: secondary_score += s["overall_strength"] * 1.0
        elif s["house"] in obstructive:obstruction_score += s["overall_strength"] * 2.0

    # Moon proximity bonus to lords of target houses
    prox_bonus = 0.0
    if moon_lon is not None:
        for pl in {house_lords.get(h) for h in (primary + secondary)}:
            if pl and pl in bodies:
                try:
                    if angular_sep(moon_lon, float(bodies[pl]["longitude_deg"])) < 12.0:
                        prox_bonus += 1.0
                except Exception:
                    pass

    # Radicality multiplier
    date = inp.date or datetime.now(timezone.utc).date().isoformat()
    time_ = inp.time or datetime.now(timezone.utc).time().replace(microsecond=0).isoformat()
    rad = radicality_flags(chart, inp.tz_name or "UTC", date, time_)
    rad_mult = 1.10 if rad.get("fits") else 1.0

    # Pañcāṅga snapshot
    t_idx = tithi_index(moon_lon, sun_lon)
    moon_star_label = moon_star(chart)

    # Final aggregate
    total_positive = (primary_score + secondary_score + prox_bonus + drishti_bonus) * rad_mult
    total_negative = obstruction_score + drishti_malus
    net_score = total_positive - total_negative

    if net_score > 1.25:
        answer, conf = "yes", min(0.96, 0.74 + 0.06 * net_score)
    elif net_score < -1.25:
        answer, conf = "no",  min(0.96, 0.74 + 0.06 * abs(net_score))
    else:
        answer, conf = "uncertain", 0.62

    return {
        "ok": True,
        "system": "parashari",
        "meta": {
            "question_type": qtype.value,
            "zodiac_mode": inp.zodiac_mode,
            "ayanamsa": inp.ayanamsa,
            "ayanamsa_deg_used": ay_deg,
            "house_system": inp.house_system,
            "analysis_time": datetime.now(timezone.utc).isoformat()
        },
        "panchanga": {
            "tithi_index": t_idx,
            "tithi_label": (TITHI_NAMES[t_idx] if t_idx is not None else None),
            "moon_star": moon_star_label
        },
        "chart_data": {
            "ASC_deg": asc_deg,
            "ASC_sign": sign_name_from_deg(float(asc_deg or 0.0)) if asc_deg is not None else None,
            "Moon_deg": moon_lon,
            "Moon_sign": (sign_name_from_deg(float(moon_lon)) if moon_lon is not None else None),
            "cusps_deg": cusps,
            "house_lords": house_lords
        },
        "planetary_analysis": strengths,
        "scoring_breakdown": {
            "primary_score": round(primary_score, 3),
            "secondary_score": round(secondary_score, 3),
            "obstruction_score": round(obstruction_score, 3),
            "moon_proximity_bonus": round(prox_bonus, 3),
            "drishti_bonus": round(drishti_bonus, 3),
            "drishti_malus": round(drishti_malus, 3),
            "radicality_multiplier": rad_mult,
            "net_score": round(net_score, 3)
        },
        "judgement": {"answer": answer, "confidence": round(conf, 3)}
    }

# =============================================================================
# Internal helpers — KP
# =============================================================================

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
    Nodes act as agents of their star-/sign-lords and the planets they conjoin (±3°) or oppose (7th).
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
    if planet_name in {"Rahu","Ketu"}:
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
            if angular_sep(lon, olon) <= 3.0 or abs((int(lon//30) - int(olon//30)) % 12) == 6:
                sig |= _kp_signified_houses_base(other, chart, cusps)
    return sig

def _kp_cusp_sub_lord_eval(cusp_house: int, chart: Dict[str, Any], cusps: List[float],
                           positive: Set[int], negative: Set[int]) -> Tuple[bool, Set[int], str, Optional[str]]:
    """Evaluate a cusp by its sub-lord; if borderline, peek at SSL for tie-break."""
    cusp_deg = cusps[cusp_house-1]
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
    WEEK_TO_LORD = ["Moon","Mars","Mercury","Jupiter","Venus","Saturn","Sun"]
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
    asc_deg = chart.get("angles", {}).get("asc_deg", chart.get("asc_deg")) or 0.0
    asc_signlord = lord_of_sign(float(asc_deg))
    asc_starlord, *_ = kp_star_sub_sub(float(asc_deg))

    return {p for p in [daylord, moon_signlord, moon_starlord, asc_signlord, asc_starlord] if p}

# =============================================================================
# KP entrypoint
# =============================================================================

def analyze_kp(inp: HoraryInput) -> Dict[str, Any]:
    kp_aya = inp.kp_ayanamsa if inp.kp_ayanamsa is not None else "krishnamurti"

    # Chart (SIDEREAL) with KP ayanamsa
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

    # Houses (KP system)
    houses = compute_houses_from_chart(
        chart, latitude=la, longitude=lo,
        house_system=inp.kp_house_system,
        zodiac_mode="sidereal", ayanamsa_deg=ay_deg
    )
    cusps = list(houses.get("cusps_deg", []) or [])
    asc_deg_sid = float(houses.get("asc_deg"))

    # KP number anchoring (rotate cusps so ASC equals number-derived degree)
    if inp.kp_number and (inp.kp_number_mode or "anchor_asc").lower() == "anchor_asc":
        target = (max(1, min(249, int(inp.kp_number))) - 1) * (360.0 / 249.0)
        cusps = rotate_cusps_to_target_asc(cusps, asc_deg_sid, target)
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
        base_conf = 0.70 + 0.05*(yes_hits - no_hits)
        base_ans = "yes"
    elif no_hits > yes_hits:
        base_conf = 0.70 + 0.05*(no_hits - yes_hits)
        base_ans = "no"
    else:
        base_conf = 0.62
        base_ans = "uncertain"

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
            "analysis_time": datetime.now(timezone.utc).isoformat()
        },
        "kp_core": {
            "ASC_deg_sid": asc_deg_sid,
            "ASC_sign": sign_name_from_deg(float(asc_deg_sid or 0.0)),
            "ruling_planets": sorted(list(rp)),
        },
        "cusp_analysis": detailed_cusps,
        "planetary_kp": planetary_kp,
        "signification_evidence": evidence,
        "judgement": {"answer": base_ans, "confidence": round(conf, 3), "method": "KP cusp sub-lords (+SSL & RP bias)"}
    }

# =============================================================================
# Public dispatcher
# =============================================================================

def analyze_prasna(inp: HoraryInput, method: str = "parashari") -> Dict[str, Any]:
    """
    method in {"parashari","kp"}.
    (Hybrid is intentionally NOT implemented in this classical module.)
    """
    try:
        m = (method or "parashari").strip().lower()
        if m == "kp":
            return analyze_kp(inp)
        return analyze_parashari(inp)
    except Exception as e:
        return {"ok": False, "system": method, "error": str(e), "error_type": type(e).__name__}

def analyze_prasna_enhanced(inp, method: str = "parashari"):
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
