# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional, Set, Tuple
import math

# Reuse your shared helpers/constants
from .horary_shared import (
    deg_wrap, angular_sep, sign_index, house_of, sign_name_from_deg,
    graha_drishti_offsets, houses_aspected_by, EXALTATION_SIGNS, MOOLATRIKONA_SIGNS
)

# ──────────────────────────────────────────────────────────────────────────────
# Small classical helpers
# ──────────────────────────────────────────────────────────────────────────────

MOVABLE = {0, 3, 6, 9}        # Aries, Cancer, Libra, Capricorn
FIXED   = {1, 4, 7, 10}       # Taurus, Leo, Scorpio, Aquarius
DUAL    = {2, 5, 8, 11}       # Gemini, Virgo, Sagittarius, Pisces

def sign_modality_by_deg(deg: float) -> str:
    s = sign_index(deg)
    if s in MOVABLE: return "movable"
    if s in FIXED:   return "fixed"
    return "dual"

def house_angularity(h: int) -> str:
    if h in (1,4,7,10):  return "angular"
    if h in (2,5,8,11):  return "succedent"
    return "cadent"

def is_combust(planet: str, lon_planet: float, lon_sun: Optional[float], combust_map: Dict[str,float]) -> bool:
    if planet == "Sun" or lon_sun is None: return False
    limit = combust_map.get(planet)
    if not isinstance(limit, (int,float)): return False
    return angular_sep(lon_planet, float(lon_sun)) <= float(limit)

def dignity_label(lon: float, planet: str) -> str:
    """Human label: exalted/own/MT/enemy/debilitated/neutral (simple classical readout)."""
    s = sign_index(lon)
    if planet in EXALTATION_SIGNS and s == EXALTATION_SIGNS[planet]: return "exalted"
    if planet in EXALTATION_SIGNS and s == (EXALTATION_SIGNS[planet] + 6) % 12: return "debilitated"
    own = {"Sun":[4], "Moon":[3], "Mars":[0,7], "Mercury":[2,5], "Jupiter":[8,11], "Venus":[1,6], "Saturn":[9,10]}
    if planet in own and s in own[planet]: return "own"
    if planet in MOOLATRIKONA_SIGNS and s == MOOLATRIKONA_SIGNS[planet]: return "moolatrikona"
    return "neutral"

def has_mutual_graha_drishti(a_name: str, b_name: str, cusps: List[float], bodies: Dict[str,Any]) -> bool:
    """Sign-based Parāśarī dṛṣṭi between two planets (mutual)."""
    if a_name not in bodies or b_name not in bodies: return False
    try:
        hA = house_of(float(bodies[a_name]["longitude_deg"]), cusps)
        hB = house_of(float(bodies[b_name]["longitude_deg"]), cusps)
    except Exception:
        return False
    A_targets = houses_aspected_by(a_name, cusps, bodies)
    B_targets = houses_aspected_by(b_name, cusps, bodies)
    return (hB in A_targets) and (hA in B_targets)

def next_applying_arc_deg(moon_deg: float, target_deg: float) -> float:
    """
    Arc until Moon perfects *any* of the standard Ptolemaic aspects (0/60/90/120/180)
    to target by zodiacal order. This mirrors common prasna timing practice.
    """
    aspects = [0.0, 60.0, 90.0, 120.0, 180.0]
    best = 1e9
    for a in aspects:
        # Moon position that would perfect aspect to target: target ± a
        pts = [deg_wrap(target_deg + a), deg_wrap(target_deg - a)]
        for p in pts:
            arc = deg_wrap(p - moon_deg)
            if 0.0 < arc < best:
                best = arc
    return best if best < 1e8 else float("nan")

def timing_units_from_context(house_type: str, sign_mod: str) -> Tuple[str, float]:
    """
    Map arc degrees to a civil unit. Classical heuristics:
      angular/movable → days; angular/fixed → weeks; cadent/fixed → months, etc.
    Returns (unit, scale) where time ≈ arc_deg / scale.
    """
    if house_type == "angular":
        if sign_mod == "movable": return ("days", 1.3)   # ~1.3°/day → days
        if sign_mod == "dual":    return ("days", 0.9)
        return ("weeks", 8.0)     # fixed slows → convert deg to weeks
    if house_type == "succedent":
        if sign_mod == "movable": return ("weeks", 7.0)
        if sign_mod == "dual":    return ("weeks", 5.0)
        return ("months", 30.0)
    # cadent
    if sign_mod == "movable":     return ("weeks", 10.0)
    if sign_mod == "dual":        return ("months", 25.0)
    return ("months", 30.0)

# ──────────────────────────────────────────────────────────────────────────────
# Rule engine
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleHit:
    id: str
    label: str
    effect: str  # "perfect" | "help" | "harm" | "block" | "slow"
    source: str

@dataclass
class Rule:
    id: str
    label: str
    effect: str
    source: str
    applies: Callable[[Dict[str,Any]], bool]

def run_rules(ctx: Dict[str,Any], rules: List[Rule]) -> List[RuleHit]:
    hits: List[RuleHit] = []
    for r in rules:
        try:
            if r.applies(ctx):
                hits.append(RuleHit(r.id, r.label, r.effect, r.source))
        except Exception:
            continue
    return hits

# ──────────────────────────────────────────────────────────────────────────────
# Canonical Parāśarī rule set (starter)
# ──────────────────────────────────────────────────────────────────────────────

def _lord(body_name: str, ctx: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    return (ctx.get("bodies") or {}).get(body_name)

def _house_lord(h: int, ctx: Dict[str,Any]) -> Optional[str]:
    return (ctx.get("house_lords") or {}).get(h)

def _moon(ctx): return (ctx.get("bodies") or {}).get("Moon")
def _sun(ctx):  return (ctx.get("bodies") or {}).get("Sun")

def _lon(b): 
    try: return float(b["longitude_deg"])
    except: return None

def _lord_for_query(ctx: Dict[str,Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (querent_lord, quesited_lord) from mapping."""
    qh = int(ctx.get("querent_house") or 1)
    th = ctx.get("target_houses") or []
    qlord = _house_lord(qh, ctx)
    tlord = _house_lord(th[0], ctx) if th else None
    return qlord, tlord

def RULES_PARASHARI(combust_map: Dict[str,float]) -> List[Rule]:
    # 1) Perfection by conjunction or mutual dṛṣṭi of lords
    def perfection(ctx):
        ql, tl = _lord_for_query(ctx)
        if not (ql and tl): return False
        bodies = ctx["bodies"]; cusps = ctx["cusps"]
        if ql in bodies and tl in bodies:
            la, lb = _lon(bodies[ql]), _lon(bodies[tl])
            if la is None or lb is None: return False
            # same sign & within ~8° (loose)
            if sign_index(la) == sign_index(lb) and angular_sep(la, lb) <= 8.0:
                return True
            # mutual Parāśarī dṛṣṭi
            return has_mutual_graha_drishti(ql, tl, cusps, bodies)
        return False

    # 2) Combustion blocking (either significator)
    def combustion_block(ctx):
        ql, tl = _lord_for_query(ctx)
        sun = _sun(ctx); bodies = ctx["bodies"]
        if not sun or "Sun" not in bodies: return False
        s = _lon(bodies["Sun"])
        for p in (ql, tl):
            if not p or p not in bodies: continue
            lp = _lon(bodies[p])
            if lp is not None and is_combust(p, lp, s, combust_map): 
                return True
        return False

    # 3) Debilitated lagna lord harms
    def lagna_lord_debil(ctx):
        ql, _ = _lord_for_query(ctx)
        if not ql: return False
        b = _lord(ql, ctx); 
        if not b: return False
        return dignity_label(_lon(b), ql) == "debilitated"

    # 4) Saturn in 1st slows/harms
    def saturn_in_first(ctx):
        bodies = ctx["bodies"]; cusps = ctx["cusps"]
        if "Saturn" not in bodies: return False
        try:
            h = house_of(float(bodies["Saturn"]["longitude_deg"]), cusps)
            return h == 1
        except Exception:
            return False

    return [
        Rule("perfection", "Perfection of lords (conjunction or mutual dṛṣṭi)", "perfect",
             "Parāśarī graha dṛṣṭi; PM ‘Outcome’ chapters", perfection),
        Rule("combust_block", "Combust significator blocks perfection", "block",
             "Combustion can spoil result (PM classical dicta)", combustion_block),
        Rule("lagna_lord_debil", "Debilitated lagna lord harms outcome", "harm",
             "Bhava lord dignity (BPHS/PM)", lagna_lord_debil),
        Rule("saturn_in_first", "Saturn in 1st slows & weighs down querent", "slow",
             "PM cautions on lagna afflicted by Saturn", saturn_in_first),
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_context_for_parashari(*, chart: Dict[str,Any], cusps: List[float],
                                house_lords: Dict[int,str],
                                mapping: Dict[str,List[int]],
                                combust_map: Dict[str,float],
                                querent_house: int = 1) -> Dict[str,Any]:
    bodies = {b.get("name"): b for b in (chart.get("bodies") or []) if isinstance(b, dict) and b.get("name")}
    return {
        "chart": chart,
        "bodies": bodies,
        "cusps": cusps,
        "house_lords": house_lords,
        "querent_house": querent_house,
        "target_houses": (mapping.get("primary") or []) + (mapping.get("secondary") or []),
        "combust_map": combust_map,
    }

def outcome_from_hits(hits: List[RuleHit]) -> Tuple[str, List[RuleHit]]:
    """
    Simple, classical priority:
      any 'block' and no 'perfect' → 'no'
      any 'perfect' → 'yes'
      else if 'harm' dominates and no 'help' → 'no'
      else if only 'slow' → 'maybe'
      else → 'maybe'
    """
    have_perfect = any(h.effect == "perfect" for h in hits)
    have_block   = any(h.effect == "block" for h in hits)
    have_help    = any(h.effect == "help"  for h in hits)
    have_harm    = any(h.effect == "harm"  for h in hits)
    if have_block and not have_perfect:
        return "no", hits
    if have_perfect:
        return "yes", hits
    if have_harm and not have_help:
        return "no", hits
    return "maybe", hits

def timing_estimate_from_moon(moon_deg: Optional[float],
                              target_deg: Optional[float],
                              target_house: Optional[int]) -> Optional[Dict[str,Any]]:
    if moon_deg is None or target_deg is None or target_house is None:
        return None
    arc = next_applying_arc_deg(moon_deg, target_deg)
    if not (isinstance(arc, float) and math.isfinite(arc)): 
        return None
    htype = house_angularity(target_house)
    smod  = sign_modality_by_deg(moon_deg)
    unit, scale = timing_units_from_context(htype, smod)
    estimate = max(1, int(round(arc / scale)))
    return {"arc_deg": round(arc,3), "unit": unit, "estimate": estimate,
            "basis": f"Moon applying to significator in {htype}/{smod}"}
