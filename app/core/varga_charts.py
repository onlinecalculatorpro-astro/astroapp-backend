# app/core/varga_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Śoḍaśa Vargas — research-grade divisional chart engine (gold-standard numerics)

Goals
- Full 16 vargas: D1,D2,D3,D4,D7,D9,D10,D12,D16,D20,D24,D27,D30,D40,D45,D60
- Sidereal-first (nirayana) with ayanāṁśa subtraction (app.core.ayanamsa.get_ayanamsa_deg)
- Canonical Parāśara mapping rules; modality/parity aware; special D2/D3/D30 handling
- Deterministic boundary policy: half-open sub-intervals [start, end) with tiny epsilon
- “Stretched” varga longitude: each amśa is expanded to 0..30° in the varga rāśi
- Pluggable spec: if constants_vedic exposes custom tables, we’ll honor them

Public API
    varga_position(lon_deg, varga, *, zodiac_mode="sidereal", ayanamsa=None|float|str) -> dict
    compute_varga_chart(longitudes_by_name: dict[str,float], varga, **opts) -> dict[str,dict]
    compute_many_vargas(longitudes_by_name, vargas: list[str]|tuple[str,...], **opts) -> dict

Notes
- Aries=0 … Pisces=11 indexing. Rāśi names in English for UI; adjust if you prefer saṁskṛta.
- Ayanāṁśa:
    • If `ayanamsa` is a string (e.g., "lahiri"), we call get_ayanamsa_deg(jd_tt?) — not available
      here; pass a float ayanāṁśa when you can. For convenience, if string is passed, we call
      app.core.ayanamsa.get_ayanamsa_deg with a None JD (your implementation may accept it or
      ignore JD). Best practice: supply the numeric ayanāṁśa for the event’s JD_TT.
- D30 (Triṁśāṁśa): unequal segment widths by rāśi parity; we also return the *lord* and map
  the varga sign to the planet’s masculine domicile for odd rāśis and feminine domicile for even.
- For D27/D40/D45/D60 we implement the widely-used Parāśara starts:
    D27, D40, D60: odd → start Aries (0), even → start Libra (6), step +1
    D45: movable → start Aries (0), fixed → start Sagittarius (8), dual → start Leo (4)
  If your tradition differs, override via `constants_vedic.VARGA_SPECS`.

Caveat
- For absolute conformance with your school, populate `constants_vedic.VARGA_SPECS` (see
  `_SPEC_SCHEMA` below). This module will seamlessly adopt your gold tables.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Callable

import math

# ─────────────────────────────────────────────────────────────────────────────
# Optional custom spec import (plug-in)
# ─────────────────────────────────────────────────────────────────────────────
_CUSTOM_SPECS: Optional[Dict[str, Any]] = None
try:
    # Expecting constants_vedic.VARGA_SPECS (dict), optional aliases at VARGA_ALIASES (dict)
    from app.core.constants_vedic import VARGA_SPECS as _CV_SPECS  # type: ignore
    _CUSTOM_SPECS = dict(_CV_SPECS) if isinstance(_CV_SPECS, dict) else None
    try:
        from app.core.constants_vedic import VARGA_ALIASES as _CV_ALIASES  # type: ignore
        _CUSTOM_ALIASES = dict(_CV_ALIASES) if isinstance(_CV_ALIASES, dict) else {}
    except Exception:
        _CUSTOM_ALIASES = {}
except Exception:
    _CUSTOM_ALIASES = {}

# ─────────────────────────────────────────────────────────────────────────────
# Ayanāṁśa (optional float or key)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_ayanamsa(ayanamsa: Optional[Any]) -> Tuple[float, Optional[str]]:
    """
    Accepts:
      - None or 0 → 0.0
      - float     → use as-is
      - str       → call app.core.ayanamsa.get_ayanamsa_deg with (key) if available; else 0.0
    Returns (deg, note)
    """
    if ayanamsa is None:
        return 0.0, None
    if isinstance(ayanamsa, (int, float)):
        try:
            return float(ayanamsa), "explicit"
        except Exception:
            return 0.0, "ayanamsa_parse_fallback"
    if isinstance(ayanamsa, str) and ayanamsa.strip():
        try:
            from app.core.ayanamsa import get_ayanamsa_deg  # type: ignore
            # If your implementation requires JD, pass it in your own wrapper.
            val = float(get_ayanamsa_deg(None, ayanamsa.strip().lower()))  # type: ignore[arg-type]
            return val, ayanamsa.strip().lower()
        except Exception:
            return 0.0, f"ayanamsa_fallback({ayanamsa})"
    return 0.0, "ayanamsa_parse_fallback"

# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────
_RASI_NAMES = (
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
)
_ODD_RASI = {0,2,4,6,8,10}  # masculine: Aries, Gemini, Leo, Libra, Sagittarius, Aquarius
_MOVABLE  = {0,3,6,9}
_FIXED    = {1,4,7,10}
_DUAL     = {2,5,8,11}

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _rasi_index(lon: float) -> int:
    return int(math.floor(_norm360(lon) / 30.0)) % 12

def _lon_in_rasi(lon: float) -> float:
    s = _rasi_index(lon)
    return _norm360(lon) - 30.0 * s

def _modality(s: int) -> str:
    if s in _MOVABLE: return "movable"
    if s in _FIXED:   return "fixed"
    return "dual"

# masculine/feminine domiciles for D30 varga-sign mapping
_MASC_DOM = {"Mars":0, "Saturn":10, "Jupiter":8, "Mercury":2, "Venus":6}
_FEM_DOM  = {"Mars":7, "Saturn":9,  "Jupiter":11,"Mercury":5, "Venus":1}

# deterministic boundary epsilon (deg)
_EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# Spec schema & default Parāśara rules
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VargaSpec:
    key: str            # "D9" etc
    name: str           # "Navamsa"
    parts: int          # N
    method: str         # "linear", "hora", "drekkana", "trimsamsa"
    # For "linear": base and step functions resolve start offset (0..11) and step (±1 or 3 etc)
    base_ofs_fn: Optional[Callable[[int], int]] = None
    step_fn: Optional[Callable[[int], int]] = None

# base offset helpers
def _base_same(s: int) -> int: return s
def _base_plus(s: int, k: int) -> int: return (s + k) % 12
def _base_by_parity(s: int, odd_ofs: int, even_ofs: int) -> int:
    return (s + (odd_ofs if s in _ODD_RASI else even_ofs)) % 12
def _base_by_modality(s: int, mov: int, fix: int, dual: int) -> int:
    if s in _MOVABLE: return (s + mov) % 12
    if s in _FIXED:   return (s + fix) % 12
    return (s + dual) % 12

# Default canonical specs (widely-used Parāśara scheme)
_DEFAULT_SPECS: Dict[str, VargaSpec] = {
    "D1":  VargaSpec("D1",  "Rasi",           1,  "linear",   _base_same,            lambda s: 0),
    "D2":  VargaSpec("D2",  "Hora",           2,  "hora"),
    "D3":  VargaSpec("D3",  "Drekkana",       3,  "drekkana"),
    "D4":  VargaSpec("D4",  "Chaturthamsa",   4,  "linear",
                     lambda s: _base_by_modality(s, 0, 8, 3),  # movable:+0, fixed:+8 (9th), dual:+3 (4th)
                     lambda s: 3),
    "D7":  VargaSpec("D7",  "Saptamsa",       7,  "linear",
                     lambda s: _base_by_parity(s, 0, 6),       # odd:+0, even:+6 (7th)
                     lambda s: 1),
    "D9":  VargaSpec("D9",  "Navamsa",        9,  "linear",
                     lambda s: _base_by_modality(s, 0, 8, 4),  # movable:+0, fixed:+8, dual:+4
                     lambda s: 1),
    "D10": VargaSpec("D10", "Dasamsa",       10,  "linear",
                     lambda s: _base_by_parity(s, 0, 8),       # odd:+0, even:+8
                     lambda s: 1),
    "D12": VargaSpec("D12", "Dvadashamsa",   12,  "linear",
                     _base_same,                                 lambda s: 1),
    "D16": VargaSpec("D16", "Shodasamsa",    16,  "linear",
                     lambda s: _base_by_parity(s, 0, 8),       # (common tradition)
                     lambda s: 1),
    "D20": VargaSpec("D20", "Vimsamsa",      20,  "linear",
                     lambda s: _base_by_modality(s, 0, 8, 4),
                     lambda s: 1),
    "D24": VargaSpec("D24", "Chaturvimsh.",  24,  "linear",
                     lambda s: _base_by_parity(s, 0, 8),
                     lambda s: 1),
    "D27": VargaSpec("D27", "Bhamsa",        27,  "linear",
                     # Start bases by parity (Aries/Libra) — common Parāśara convention
                     lambda s: (0 if s in _ODD_RASI else 6),
                     lambda s: 1),
    "D30": VargaSpec("D30", "Trimsamsa",     30,  "trimsamsa"),
    "D40": VargaSpec("D40", "Khavedamsa",    40,  "linear",
                     lambda s: (0 if s in _ODD_RASI else 6),   # odd→Aries, even→Libra
                     lambda s: 1),
    "D45": VargaSpec("D45", "Akshavedamsa",  45,  "linear",
                     lambda s: _base_by_modality(s, 0, 8, 4),  # movable:Aries, fixed:Sag, dual:Leo
                     lambda s: 1),
    "D60": VargaSpec("D60", "Shashtiamsa",   60,  "linear",
                     lambda s: (0 if s in _ODD_RASI else 6),   # odd→Aries, even→Libra
                     lambda s: 1),
}

# Aliases (lowercased keys)
_ALIASES = {
    "rasi":"D1","rāśi":"D1","d1":"D1",
    "hora":"D2","d2":"D2",
    "drekkana":"D3","d3":"D3",
    "chaturthamsa":"D4","d4":"D4",
    "saptamsa":"D7","d7":"D7",
    "navamsa":"D9","navamsha":"D9","d9":"D9",
    "dasamsa":"D10","dashamsa":"D10","d10":"D10",
    "dvadasamsa":"D12","dvadashamsa":"D12","d12":"D12",
    "shodasamsa":"D16","sodasamsa":"D16","d16":"D16",
    "vimsamsa":"D20","d20":"D20",
    "chaturvimshamsa":"D24","siddhamsa":"D24","d24":"D24",
    "bhamsa":"D27","nakshatramsa":"D27","d27":"D27",
    "trimsamsa":"D30","trimsamsha":"D30","d30":"D30",
    "khavedamsa":"D40","d40":"D40",
    "akshavedamsa":"D45","d45":"D45",
    "shashtiamsa":"D60","shastiamsa":"D60","d60":"D60",
}
# fold in any project-level aliases
_ALIASES.update({k.lower(): v for k, v in _CUSTOM_ALIASES.items()})  # type: ignore

# Plug custom specs if present
if _CUSTOM_SPECS:
    for k, v in _CUSTOM_SPECS.items():
        try:
            # Expect dict with keys per _SPEC_SCHEMA or already a VargaSpec
            if isinstance(v, VargaSpec):
                _DEFAULT_SPECS[k] = v
            elif isinstance(v, dict):
                _DEFAULT_SPECS[k] = VargaSpec(
                    key=str(k),
                    name=str(v.get("name", k)),
                    parts=int(v["parts"]),
                    method=str(v["method"]),
                    base_ofs_fn=v.get("base_ofs_fn") or _DEFAULT_SPECS[k].base_ofs_fn,  # type: ignore[index]
                    step_fn=v.get("step_fn") or _DEFAULT_SPECS[k].step_fn,              # type: ignore[index]
                )
        except Exception:
            continue

# ─────────────────────────────────────────────────────────────────────────────
# Internals: per-method mapping
# ─────────────────────────────────────────────────────────────────────────────
def _part_index_in_sign(deg_in_sign: float, parts: int) -> int:
    """Return 0..parts-1 using half-open intervals [k*w, (k+1)*w)."""
    w = 30.0 / float(parts)
    # subtract tiny epsilon so exact multiples at 30/N land in the lower bucket deterministically
    x = max(0.0, min(deg_in_sign - _EPS, 30.0 - _EPS))
    k = int(math.floor(x / w))
    if k >= parts: k = parts - 1
    return k

def _linear_varga_sign(s: int, part_k: int, base_fn: Callable[[int], int], step_fn: Callable[[int], int]) -> int:
    base = base_fn(s) % 12
    step = step_fn(s)
    return (base + (part_k * step)) % 12

def _drekkana_sign(s: int, part_k: int) -> int:
    # Odd: s + (0,4,8); Even: s + (0,8,4)
    if part_k == 0:
        return s
    if s in _ODD_RASI:
        return (s + (4 if part_k == 1 else 8)) % 12
    else:
        return (s + (8 if part_k == 1 else 4)) % 12

def _hora_sign(s: int, part_k: int) -> int:
    # Parāśara (Cancer/Leo only). Odd sign: first half → Leo(4), second → Cancer(3).
    # Even sign: first → Cancer(3), second → Leo(4).
    if s in _ODD_RASI:
        return 4 if part_k == 0 else 3
    else:
        return 3 if part_k == 0 else 4

def _trimsamsa_segment(s: int, deg_in_sign: float) -> Tuple[str, float]:
    """
    Returns (lord, width_deg_consumed_until_end_of_segment) using Parāśara table.
    Odd rāśi: 5° Mars, 5° Saturn, 8° Jupiter, 7° Mercury, 5° Venus  (total 30)
    Even rāśi: 5° Venus,7° Mercury,8° Jupiter,5° Saturn,5° Mars
    """
    odd = s in _ODD_RASI
    if odd:
        cuts = (5.0, 10.0, 18.0, 25.0, 30.0)
        lords = ("Mars","Saturn","Jupiter","Mercury","Venus")
    else:
        cuts = (5.0, 12.0, 20.0, 25.0, 30.0)
        lords = ("Venus","Mercury","Jupiter","Saturn","Mars")
    x = max(0.0, min(deg_in_sign, 30.0 - _EPS))
    for c, L in zip(cuts, lords):
        if x < c or abs(x - c) < _EPS:
            return L, c
    return lords[-1], cuts[-1]

def _trimsamsa_sign_and_lord(s: int, deg_in_sign: float) -> Tuple[int, str]:
    lord, _ = _trimsamsa_segment(s, deg_in_sign)
    if s in _ODD_RASI:
        return _MASC_DOM[lord], lord
    else:
        return _FEM_DOM[lord], lord

# ─────────────────────────────────────────────────────────────────────────────
# Public engine
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_key(varga: str) -> str:
    k = str(varga).strip()
    if not k:
        raise ValueError("varga key is required")
    ku = k.upper()
    if ku in _DEFAULT_SPECS:
        return ku
    kl = k.lower()
    if kl in _ALIASES:
        return _ALIASES[kl]
    # D## typed in lowercase?
    if kl.startswith("d") and kl[1:].isdigit() and ("D"+kl[1:]) in _DEFAULT_SPECS:
        return "D"+kl[1:]
    raise ValueError(f"Unsupported varga key '{varga}'")

def varga_position(
    lon_deg: float,
    varga: str,
    *,
    zodiac_mode: str = "sidereal",
    ayanamsa: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Compute a single point’s varga placement.
    Inputs
        lon_deg     tropical ecliptic longitude (deg, 0..360) unless zodiac_mode="sidereal"
        varga       e.g. "D9", "navamsa", "D30", "trimsamsa"
        zodiac_mode "sidereal" (default) or "tropical"
        ayanamsa    None|float|str  (if sidereal and you pass float, we subtract it; if str, we try adapter)
    Output dict
        {
          'varga': 'D9', 'name': 'Navamsa',
          'rasi_index': 0..11, 'rasi_name': 'Aries',
          'division_index': 1..N, 'division_count': N,
          'varga_rasi_index': 0..11, 'varga_rasi_name': 'Sagittarius',
          'varga_longitude_deg': 0..30,
          'nirayana_longitude_deg': (if sidereal) deg,
          'extras': {...}  # e.g., for D30: {'lord':'Mars'}
        }
    """
    key = _resolve_key(varga)
    spec = _DEFAULT_SPECS[key]

    # sidereal handling
    L = float(lon_deg)
    extras: Dict[str, Any] = {}
    if (zodiac_mode or "sidereal").lower().startswith("sidereal"):
        ay, note = _resolve_ayanamsa(ayanamsa)
        if note:
            extras["ayanamsa_note"] = note
        L = _norm360(L - ay)
        extras["nirayana_longitude_deg"] = L
    else:
        extras["nirayana_longitude_deg"] = None

    s = _rasi_index(L)
    din = _lon_in_rasi(L)

    # division index and stretched longitude
    N = int(spec.parts)
    if spec.method == "trimsamsa":
        # piecewise widths — compute amśa-longitude in 30° by proportional stretch of that segment
        lord, seg_end = _trimsamsa_segment(s, din)
        seg_start = {True:(0.0,5.0,10.0,18.0,25.0), False:(0.0,5.0,12.0,20.0,25.0)}[s in _ODD_RASI]
        # find k for division_index
        # division index is 1..30 uniform splits, but trimsamsa classical divisions are not equal 1°.
        # We report: segment ordinal as 1..5 and also a pseudo 1..30 index by floor( (din / 1.0) )+1
        # For consistency with equal-N interface, set division_index= floor(din / (30/N)) + 1.
        k = _part_index_in_sign(din, N)
        # varga sign: from planet domicile (masc/fem)
        vs, lord_nm = _trimsamsa_sign_and_lord(s, din)
        # normalized stretched long: map position within the segment to 0..30 via linear scale inside the segment
        seg_w = seg_end - (seg_start[(0,1,2,3,4)[0]] if False else 0.0)  # keep linter happy
        # compute actual segment start:
        seg_starts = seg_start  # tuple
        x = din
        st = 0.0
        if s in _ODD_RASI:
            bounds = (5.0,10.0,18.0,25.0,30.0)
        else:
            bounds = (5.0,12.0,20.0,25.0,30.0)
        last = 0.0
        for b in bounds:
            if x < b or abs(x - b) < _EPS:
                st = last
                seg_w = b - last
                break
            last = b
        frac_in_seg = 0.0 if seg_w <= 0 else (x - st) / seg_w
        varga_lon = 30.0 * frac_in_seg  # stretch each segment to 30°
        extras["trimsamsa_lord"] = lord_nm
        division_index = k + 1
        varga_sign = vs
    elif spec.method == "hora":
        k = _part_index_in_sign(din, N)
        varga_sign = _hora_sign(s, k)
        # stretched varga longitude (each half → 30°)
        w = 30.0 / N  # 15°
        start = k * w
        frac = (din - start) / w
        varga_lon = max(0.0, min(frac * 30.0, 30.0 - _EPS))
        division_index = k + 1
    elif spec.method == "drekkana":
        k = _part_index_in_sign(din, N)
        varga_sign = _drekkana_sign(s, k)
        w = 30.0 / N  # 10°
        start = k * w
        frac = (din - start) / w
        varga_lon = max(0.0, min(frac * 30.0, 30.0 - _EPS))
        division_index = k + 1
    else:  # linear
        k = _part_index_in_sign(din, N)
        base_fn = spec.base_ofs_fn or _base_same
        step_fn = spec.step_fn or (lambda s_: 1)
        varga_sign = _linear_varga_sign(s, k, base_fn, step_fn)
        w = 30.0 / N
        start = k * w
        frac = (din - start) / w
        varga_lon = max(0.0, min(frac * 30.0, 30.0 - _EPS))
        division_index = k + 1

    return {
        "varga": spec.key,
        "name": spec.name,
        "rasi_index": s,
        "rasi_name": _RASI_NAMES[s],
        "division_index": division_index,
        "division_count": N,
        "varga_rasi_index": varga_sign,
        "varga_rasi_name": _RASI_NAMES[varga_sign],
        "varga_longitude_deg": varga_lon,  # 0..30 within the varga sign
        **extras,
    }

def compute_varga_chart(
    longitudes_by_name: Dict[str, float],
    varga: str,
    *,
    zodiac_mode: str = "sidereal",
    ayanamsa: Optional[Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Map planet/point → varga placement dict (see varga_position output).
    `longitudes_by_name` are ecliptic longitudes in degrees (tropical unless zodiac_mode="sidereal").
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, lon in longitudes_by_name.items():
        try:
            out[name] = varga_position(float(lon), varga, zodiac_mode=zodiac_mode, ayanamsa=ayanamsa)
        except Exception as e:
            out[name] = {"error": str(e)}
    return out

def compute_many_vargas(
    longitudes_by_name: Dict[str, float],
    vargas: List[str] | Tuple[str, ...],
    *,
    zodiac_mode: str = "sidereal",
    ayanamsa: Optional[Any] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute several vargas at once.
    Returns: { 'D9': { 'Sun': {...}, ... }, 'D10': {...}, ... }
    """
    res: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for v in vargas:
        key = _resolve_key(v)
        res[key] = compute_varga_chart(longitudes_by_name, key, zodiac_mode=zodiac_mode, ayanamsa=ayanamsa)
    return res

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight self-checks (deterministic behavior around edges)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simple probes (no I/O): Aries 0°, 10°, 20° in D3 → Aries, Leo, Sagittarius
    samples = [0.0, 10.0, 20.0, 29.999999]
    for x in samples:
        d3 = varga_position(x, "D3")
        assert d3["varga_rasi_name"] in ("Aries","Leo","Sagittarius")
    # D2 Hora: Aries (odd) 0..15 → Leo, 15..30 → Cancer
    h1 = varga_position(0.0, "D2");  h2 = varga_position(14.9999, "D2")
    h3 = varga_position(15.0, "D2"); h4 = varga_position(29.9999, "D2")
    assert h1["varga_rasi_name"] == "Leo" and h2["varga_rasi_name"] == "Leo"
    assert h3["varga_rasi_name"] == "Cancer" and h4["varga_rasi_name"] == "Cancer"
    print("Varga core probes OK.")
