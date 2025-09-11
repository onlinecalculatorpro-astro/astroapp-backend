# app/core/varga_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Varga (Divisional) Charts — precise, sidereal-friendly core

What this provides
- Deterministic mapping of a longitude to varga sign + degree-in-sign
- Implemented: D1 (Rāśi), D2 (Hora — Parāśara), D3 (Drekkāṇa), D9 (Navāṃśa),
  D12 (Dvādaśāṃśa)
- Chart builders for a dict of graha longitudes
- Exact boundary policy and numeric guards for 100% repeatability

Notes
- Feed this module *nirāyaṇa* (sidereal) longitudes if you want sidereal vargas.
  (Use app.core.ayanamsa or panchanga helpers upstream.)
- Degree-in-varga-sign is computed by scaling the fraction within the selected
  part to 0..30°. This is the standard presentation used by most software.

Public API
    varga_position(lon_deg, varga="d9") -> dict
    build_varga_chart(lon_map: dict[str,float], varga="d9") -> dict[str, dict]
    group_by_sign(varga_map: dict[str, dict]) -> dict[str, list[str]]

Implemented vargas
    "d1"  Rāśi
    "d2"  Hora (Parāśara: Sun=Leo, Moon=Cancer; odd/even halves)
    "d3"  Drekkāṇa (10° parts; odd→1/5/9, even→9/1/5)
    "d9"  Navāṃśa (movable start=self, fixed start=9th, dual start=5th)
    "d12" Dvādaśāṃśa (2.5° parts starting from the same sign)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, NamedTuple, Optional
import math

# ────────────────────────────────────────────────────────────────────────
# Constants & helpers
# ────────────────────────────────────────────────────────────────────────

TAU = 2.0 * math.pi
DEG = math.pi / 180.0
EPS = 1e-12  # guard for boundary snaps (sub-microarcsecond in degrees)

SIGN_NAMES = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _sign_index(lon: float) -> int:
    """0..11 (0=Aries). Exact 360 wraps to 0."""
    x = _norm360(lon)
    # Treat exact 360.0 as 0.0
    if abs(x - 360.0) < EPS:
        x = 0.0
    i = int(math.floor(x / 30.0)) % 12
    return i

def _deg_in_sign(lon: float) -> float:
    x = _norm360(lon)
    if abs(x - 360.0) < EPS:
        x = 0.0
    s = _sign_index(x)
    d = x - 30.0 * s
    # Snap 30.000.. down to 0.0 of next sign consistently
    if d >= 30.0 - EPS:
        d = 0.0
        s = (s + 1) % 12
    return d

def _is_odd_sign(sign_idx: int) -> bool:
    """Odd sign in classical sense (Aries=1, Gemini=3,...)."""
    return (sign_idx % 2) == 0  # Aries index 0 ⇒ odd

def _modality(sign_idx: int) -> str:
    """movable|fixed|dual based on sign index 0..11."""
    m = sign_idx % 3
    return ("movable" if m == 0 else "fixed" if m == 1 else "dual")

# ────────────────────────────────────────────────────────────────────────
# Core varga primitives
# ────────────────────────────────────────────────────────────────────────

class VargaPos(NamedTuple):
    varga: str
    sign_index: int           # 0..11 in the varga chart
    sign_name: str            # name of that sign
    degree_in_sign: float     # 0..30
    part_index: int           # 1..N within the natal sign
    part_size_deg: float      # 30/N
    natal_sign_index: int     # 0..11 of the natal rāśi
    natal_degree_in_sign: float  # 0..30 within natal sign

# Generic “which part” finder with strict boundary handling
def _part_index_in_sign(deg_in_sign: float, N: int) -> Tuple[int, float]:
    """
    Return 1..N index of the part and the fractional position within that
    part [0,1). Right boundary belongs to the next part (except final, which
    rolls over via the caller's sign logic).
    """
    w = 30.0 / float(N)
    # clamp tiny negatives / endings
    x = deg_in_sign
    if x < 0.0 and x > -EPS:
        x = 0.0
    if x > 30.0 - EPS and x <= 30.0 + EPS:
        x = 30.0 - EPS
    k = int(math.floor(x / w))
    if k >= N:
        k = N - 1
    start = k * w
    frac = (x - start) / w
    return (k + 1), frac  # 1-based

# Degree scaling to varga sign
def _scaled_degree_in_varga(frac_within_part: float) -> float:
    """Map fractional position in the chosen part to 0..30 in the varga sign."""
    # Boundaries safeguarded
    f = min(max(frac_within_part, 0.0), 1.0 - 1e-15)
    return 30.0 * f

# ────────────────────────────────────────────────────────────────────────
# Specific varga mappers
# ────────────────────────────────────────────────────────────────────────

def _d1_map(sign_i: int, part_i: int) -> int:
    return sign_i

def _d2_parashara(sign_i: int, part_i: int) -> int:
    """
    Hora (Parāśara):
      - Odd signs: 0..15° → Sun (Leo), 15..30° → Moon (Cancer)
      - Even signs: 0..15° → Moon (Cancer), 15..30° → Sun (Leo)
    part_i: 1 for first 15°, 2 for second 15°
    """
    odd = _is_odd_sign(sign_i)
    if odd:
        return 4 if part_i == 1 else 3  # Leo or Cancer
    else:
        return 3 if part_i == 1 else 4

def _d3_map(sign_i: int, part_i: int) -> int:
    """
    Drekkāṇa (10° each):
      Odd signs: 1st=self, 2nd=5th, 3rd=9th
      Even signs: 1st=9th, 2nd=self, 3rd=5th
    """
    odd = _is_odd_sign(sign_i)
    if odd:
        offs = [0, 4, 8]
    else:
        offs = [8, 0, 4]
    return (sign_i + offs[part_i - 1]) % 12

def _start_for_navamsa(sign_i: int) -> int:
    """
    Navāṃśa start sign per modality:
      movable → self
      fixed   → 9th from self (+8)
      dual    → 5th from self (+4)
    """
    mod = _modality(sign_i)
    if mod == "movable":
        return sign_i
    if mod == "fixed":
        return (sign_i + 8) % 12
    return (sign_i + 4) % 12  # dual

def _d9_map(sign_i: int, part_i: int) -> int:
    start = _start_for_navamsa(sign_i)
    return (start + (part_i - 1)) % 12

def _d12_map(sign_i: int, part_i: int) -> int:
    """
    Dvādaśāṃśa:
      12 parts of 2.5°, sequential signs starting from the natal sign.
    """
    return (sign_i + (part_i - 1)) % 12

# Registry for implemented vargas
_VARGA_DEF: Dict[str, Tuple[int, Any]] = {
    "d1":  (1,  _d1_map),
    "d2":  (2,  _d2_parashara),
    "d3":  (3,  _d3_map),
    "d9":  (9,  _d9_map),
    "d12": (12, _d12_map),
}
_ALIASES = {
    "rasi": "d1",
    "hora": "d2",
    "drekkana": "d3",
    "navamsa": "d9",
    "dwadasamsa": "d12",
    "dvadasamsa": "d12",
}

def _resolve_key(key: str) -> str:
    k = str(key or "").strip().lower()
    return _ALIASES.get(k, k)

# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────

def varga_position(lon_deg: float, *, varga: str = "d9") -> Dict[str, Any]:
    """
    Map a longitude to its varga placement (sign + degree-in-sign).

    Parameters
    ----------
    lon_deg : float
        Ecliptic longitude in degrees (0..360).
    varga : str
        One of: "d1","d2","d3","d9","d12" (aliases: rasi,hora,drekkana,navamsa,dvadasamsa)

    Returns
    -------
    dict with keys:
        varga, sign_index, sign_name, degree_in_sign,
        part_index, part_size_deg, natal_sign_index, natal_degree_in_sign
    """
    vk = _resolve_key(varga)
    if vk not in _VARGA_DEF:
        raise NotImplementedError(f"Varga '{varga}' not implemented yet")

    N, mapper = _VARGA_DEF[vk]
    natal_si = _sign_index(lon_deg)
    natal_d  = _deg_in_sign(lon_deg)
    part_i, frac = _part_index_in_sign(natal_d, N)
    varga_si = int(mapper(natal_si, part_i))  # 0..11
    deg_out  = _scaled_degree_in_varga(frac)

    return {
        "varga": vk,
        "sign_index": varga_si,
        "sign_name": SIGN_NAMES[varga_si],
        "degree_in_sign": float(deg_out),
        "part_index": int(part_i),
        "part_size_deg": float(30.0 / N),
        "natal_sign_index": natal_si,
        "natal_degree_in_sign": float(natal_d),
    }

def build_varga_chart(longitudes_deg: Dict[str, float], *, varga: str = "d9") -> Dict[str, Dict[str, Any]]:
    """
    Compute varga placements for a mapping {name: longitude_deg}.
    Returns {name: varga_position_dict}.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, lon in longitudes_deg.items():
        try:
            out[name] = varga_position(float(lon), varga=varga)
        except Exception as e:
            out[name] = {"error": str(e), "input": float(lon)}
    return out

def group_by_sign(varga_map: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Group bodies by varga sign name from a build_varga_chart output.
    """
    bins: Dict[str, List[str]] = {s: [] for s in SIGN_NAMES}
    for name, rec in varga_map.items():
        if not isinstance(rec, dict):
            continue
        si = rec.get("sign_index")
        if isinstance(si, int) and 0 <= si < 12:
            bins[SIGN_NAMES[si]].append(name)
    # prune empties for compactness
    return {k: v for k, v in bins.items() if v}

# ────────────────────────────────────────────────────────────────────────
# Self-checks (deterministic examples)
# ────────────────────────────────────────────────────────────────────────

def _self_test_examples() -> List[Tuple[str, Dict[str, Any]]]:
    """
    A few canonical placements that can be regression-checked.
    (All examples are Aries sign unless stated otherwise.)
    """
    ex: List[Tuple[str, Dict[str, Any]]] = []

    # D9: Aries 0°00' → Aries 0°00' (1st navāṃśa of a movable sign starts at self)
    ex.append(("d9_aries_0", varga_position(0.0, varga="d9")))

    # D9: Aries 10°00' → Cancer 0°00'
    # 10° = 3rd boundary → 4th navāṃśa; movable start=self → Aries + 3 = Cancer
    ex.append(("d9_aries_10", varga_position(10.0, varga="d9")))

    # D3: Aries 15° → odd sign → 2nd drekkāṇa ⇒ 5th sign from Aries = Leo
    ex.append(("d3_aries_15", varga_position(15.0, varga="d3")))

    # D2: Taurus 7° → even sign first hora = Moon (Cancer)
    ex.append(("d2_taurus_7", varga_position(30.0 + 7.0, varga="d2")))

    # D12: Gemini 5° → part index floor(5/2.5)=2 ⇒ Gemini + 1 = Cancer
    ex.append(("d12_gem_5", varga_position(60.0 + 5.0, varga="d12")))

    return ex

if __name__ == "__main__":
    for k, rec in _self_test_examples():
        print(k, "→", rec)
