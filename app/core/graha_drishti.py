# app/core/graha_drishti.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Parāśari Graha Dṛṣṭi — canonical, deterministic aspect engine (sign-based)

Scope
- Compute graha→sign and graha→bhāva dṛṣṭi strengths using classical
  Parāśari rules (sign-to-sign). Default strengths are binary (1 or 0).
- Optional variants:
  • "seventh_only" for all grahas (classical fallback in some schools)
  • Nodes rule (Rāhu/Keṭu): off | 7th-only | Jupiter-like (5/9/7) | Mars-like (4/8/7)
- Convenient adaptor from the sidereal chart computed by
  app.core.drik_sidereal_positions.compute_sidereal_chart.

Design notes
- Pure sign-based logic (no degree orbs) to stay faithful and deterministic.
- If you pass houses from houses_advanced, bhāva dṛṣṭi is mapped by the
  *sign at each cusp* (Cusp 1..12).
- All outputs are stable 0..1 floats to play nicely with scoring pipelines.

Public API
    GrahaDrishtiConfig
    default_drishti_config()
    compute_graha_drishti_signs(longitudes_sidereal: dict, config: GrahaDrishtiConfig) -> dict
    compute_graha_drishti_bhavas(
        longitudes_sidereal: dict,
        houses: dict,  # from houses_advanced / drik_sidereal_positions
        config: GrahaDrishtiConfig
    ) -> dict
    drishti_from_sidereal_chart(chart: dict, config: GrahaDrishtiConfig | None = None) -> dict
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

# --- minimal helpers (no circulars) -----------------------------------

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _rashi_index(lon_deg: float) -> int:
    """1..12 for 0° Aries..330° Pisces."""
    return int(math.floor(_norm360(lon_deg) / 30.0)) + 1

# Sanskrit + English labels for introspection / UI (not used in math)
_RASHI_EN = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]
def _rashi_name(idx: int) -> str:
    i = max(1, min(12, int(idx))) - 1
    return _RASHI_EN[i]

# --- config ------------------------------------------------------------

@dataclass(frozen=True)
class GrahaDrishtiConfig:
    """
    mode:
        "parashari"     → Sun/Moon/Mercury/Venus: 7th; Mars: 4/7/8; Jupiter: 5/7/9; Saturn: 3/7/10
        "seventh_only"  → all grahas 7th only
    nodes_rule:
        "off"           → ignore Rāhu/Keṭu
        "seventh_only"  → nodes cast 7th only
        "jupiter_like"  → nodes cast 5/7/9
        "mars_like"     → nodes cast 4/7/8
    include_nodes: kept for convenience; if False, nodes discarded regardless of nodes_rule.
    strength_full: numeric strength for a full dṛṣṭi (default 1.0)
    strength_none: numeric strength for absent dṛṣṭi (default 0.0)
    """
    mode: str = "parashari"
    nodes_rule: str = "seventh_only"
    include_nodes: bool = True
    strength_full: float = 1.0
    strength_none: float = 0.0

def default_drishti_config() -> GrahaDrishtiConfig:
    return GrahaDrishtiConfig()

# --- core mapping tables ----------------------------------------------

# Parāśari offsets (in signs ahead) for full dṛṣṭi from a graha's sign
# Offsets are measured forward (inclusive of 7th for all).
_PARASHARI_SPECIAL: Dict[str, Tuple[int, ...]] = {
    "Sun":      (7,),
    "Moon":     (7,),
    "Mercury":  (7,),
    "Venus":    (7,),
    "Mars":     (4, 7, 8),
    "Jupiter":  (5, 7, 9),
    "Saturn":   (3, 7, 10),
    # Nodes handled by config
}

_SEVENTH_ONLY: Tuple[int, ...] = (7,)

def _nodes_offsets(kind: str) -> Tuple[int, ...]:
    kind = (kind or "").lower()
    if kind == "jupiter_like": return (5, 7, 9)
    if kind == "mars_like":    return (4, 7, 8)
    return (7,)

# --- utilities ---------------------------------------------------------

def _forward_signs(start_idx: int, offsets: Tuple[int, ...]) -> List[int]:
    """Return absolute sign indices (1..12) reached by forward offsets from start_idx."""
    base = int(start_idx)
    out: List[int] = []
    for k in offsets:
        step = int(k) % 12
        j = ((base - 1 + step) % 12) + 1
        out.append(j)
    return out

def _collect_longitudes_sidereal(longitudes_sidereal: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, val in longitudes_sidereal.items():
        try:
            x = float(val)
            if math.isfinite(x):
                out[str(name)] = _norm360(x)
        except Exception:
            continue
    return out

# --- main: signs matrix ------------------------------------------------

def compute_graha_drishti_signs(
    longitudes_sidereal: Dict[str, float],
    config: GrahaDrishtiConfig | None = None
) -> Dict[str, Dict[int, float]]:
    """
    Returns graha → {sign_index: strength} using pure sign dṛṣṭi.
    Strengths are either strength_full or strength_none.
    """
    cfg = config or default_drishti_config()
    lons = _collect_longitudes_sidereal(longitudes_sidereal)

    out: Dict[str, Dict[int, float]] = {}
    for graha, lam in lons.items():
        # filter nodes if requested
        if graha.lower() in ("rahu","ketu","north node","south node"):
            if not cfg.include_nodes:
                continue

        src = _rashi_index(lam)

        if cfg.mode == "seventh_only":
            offs = _SEVENTH_ONLY
        else:  # parashari
            if graha in _PARASHARI_SPECIAL:
                offs = _PARASHARI_SPECIAL[graha]
            elif graha.lower() in ("rahu","ketu","north node","south node"):
                offs = _nodes_offsets(cfg.nodes_rule)
            else:
                offs = _SEVENTH_ONLY

        targets = _forward_signs(src, offs)
        row = {i: cfg.strength_none for i in range(1, 13)}
        for sidx in targets:
            row[sidx] = cfg.strength_full
        out[graha] = row
    return out

# --- bhāva mapping -----------------------------------------------------

def _bhava_signs_from_houses(houses: Dict[str, object]) -> List[int]:
    """
    Extract dominant sign for each bhāva from a houses payload:
    expects {'cusps_deg': [12 floats], ...}. Returns 12 sign indices for bhāvas 1..12.
    """
    cusps = houses.get("cusps_deg") if isinstance(houses, dict) else None
    if not isinstance(cusps, list) or len(cusps) != 12:
        raise ValueError("houses must contain 'cusps_deg' with 12 floats")
    return [_rashi_index(float(cusps[i])) for i in range(12)]

def compute_graha_drishti_bhavas(
    longitudes_sidereal: Dict[str, float],
    houses: Dict[str, object],
    config: GrahaDrishtiConfig | None = None
) -> Dict[str, Dict[int, float]]:
    """
    Map graha dṛṣṭi onto bhāvas 1..12 by taking the sign at each cusp as the
    bhāva's sign. Returns graha → {bhava_number: strength}.
    """
    signs_for_bhava = _bhava_signs_from_houses(houses)
    sign_matrix = compute_graha_drishti_signs(longitudes_sidereal, config=config)

    out: Dict[str, Dict[int, float]] = {}
    for graha, row in sign_matrix.items():
        bh_row: Dict[int, float] = {}
        for i in range(12):  # bhava 1..12
            sidx = signs_for_bhava[i]
            bh_row[i + 1] = float(row.get(sidx, 0.0))
        out[graha] = bh_row
    return out

# --- adaptor from sidereal chart --------------------------------------

def drishti_from_sidereal_chart(
    chart: Dict[str, object],
    config: GrahaDrishtiConfig | None = None
) -> Dict[str, object]:
    """
    Convenience wrapper: take output from app.core.drik_sidereal_positions.compute_sidereal_chart
    and return both signs- and bhāva-level dṛṣṭi matrices when possible.
    """
    cfg = config or default_drishti_config()
    bodies = chart.get("bodies_sidereal", []) if isinstance(chart, dict) else []
    long_map: Dict[str, float] = {}
    for r in bodies:
        try:
            nm = str(r["name"])
            lon = float(r["longitude_sidereal_deg"])
            if math.isfinite(lon):
                long_map[nm] = lon
        except Exception:
            continue

    signs = compute_graha_drishti_signs(long_map, cfg)

    bhavas = None
    houses = chart.get("houses") if isinstance(chart, dict) else None
    if isinstance(houses, dict) and "cusps_deg" in houses:
        try:
            bhavas = compute_graha_drishti_bhavas(long_map, houses, cfg)
        except Exception:
            bhavas = None

    return {
        "config": {
            "mode": cfg.mode,
            "nodes_rule": cfg.nodes_rule,
            "include_nodes": cfg.include_nodes,
            "strength_full": cfg.strength_full,
            "strength_none": cfg.strength_none,
        },
        "drishti_signs": signs,
        "drishti_bhavas": bhavas,
    }
