# app/core/varga_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Śoḍaśa Vargas — divisional chart engine (optimized & robust)

Compatibility
-------------
- Public API and return shapes are preserved:
    varga_position(lon_deg, varga, *, zodiac_mode="sidereal", ayanamsa=None|float|str) -> dict
    compute_varga_chart(longitudes_by_name: dict[str,float], varga, **opts) -> dict[str,dict]
    compute_many_vargas(longitudes_by_name, vargas: list[str]|tuple[str,...], **opts) -> dict
- Epsilon policy and boundary semantics are maintained.
- More defensive input handling; no silent Aries 0° fallbacks.

Improvements
------------
- Single-point hot path kept tight; shared factors cached.
- Robust ayanāṁśa resolver (float|str|None) with notes.
- Safe normalization & clamping of degrees (0 ≤ x < 360).
- Deterministic bucket assignment at boundaries using small ε.
- Extra tolerance for bad/missing inputs inside batch calls:
  per-point errors reported as {"error": "..."} instead of crashing.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Callable

import math

# ─────────────────────────────────────────────────────────────────────────────
# Optional custom spec import (plug-in)
# ─────────────────────────────────────────────────────────────────────────────
_CUSTOM_SPECS: Optional[Dict[str, Any]] = None
try:
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
      - float     → use as-is (explicit)
      - str       → app.core.ayanamsa.get_ayanamsa_deg(key) if available; else 0.0 (fallback note)
    Returns (deg, note|None)
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
            val = float(get_ayanamsa_deg(None, ayanamsa.strip().lower()))  # type: ignore[arg-type]
            return val, ayanamsa.strip().lower()
        except Exception:
            return 0.0, f"ayanamsa_fallback({ayanamsa})"
    return 0.0, "ayanamsa_parse_fallback"

# ─────────────────────────────────────────────────────────────────────────────
# Core helpers & constants
# ─────────────────────────────────────────────────────────────────────────────
_RASI_NAMES = (
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
)
_ODD_RASI = {0,2,4,6,8,10}  # Aries, Gemini, Leo, Libra, Sagittarius, Aquarius
_MOVABLE  = {0,3,6,9}
_FIXED    = {1,4,7,10}
_DUAL     = {2,5,8,11}

_INV_30 = 1.0 / 30.0
# deterministic boundary epsilon (deg) — tiny, but avoids upper bucket leak
_EPS = 1e-12

def _norm360(x: float) -> float:
    # Python % already yields remainder in [0,360) for positive divisors
    try:
        return float(x) % 360.0
    except Exception:
        return 0.0

def _rasi_index(lon: float) -> int:
    # _norm360 ∈ [0,360); int(...) is floor for non-negative values
    return int(_norm360(lon) * (1.0/30.0)) % 12

def _lon_in_rasi(lon: float) -> float:
    s = _rasi_index(lon)
    return _norm360(lon) - 30.0 * s

def _base_same(s: int) -> int: return s

def _base_by_parity(s: int, odd_ofs: int, even_ofs: int) -> int:
    return (s + (odd_ofs if s in _ODD_RASI else even_ofs)) % 12

def _base_by_modality(s: int, mov: int, fix: int, dual: int) -> int:
    if s in _MOVABLE: return (s + mov) % 12
    if s in _FIXED:   return (s + fix) % 12
    return (s + dual) % 12

# D30 Triṁśāṁśa mapping helpers
_MASC_DOM = {"Mars":0, "Saturn":10, "Jupiter":8, "Mercury":2, "Venus":6}
_FEM_DOM  = {"Mars":7, "Saturn":9,  "Jupiter":11,"Mercury":5, "Venus":1}

_TRIM_ODD_BOUNDS  = (5.0, 10.0, 18.0, 25.0, 30.0)
_TRIM_EVEN_BOUNDS = (5.0, 12.0, 20.0, 25.0, 30.0)
_TRIM_ODD_LORDS   = ("Mars","Saturn","Jupiter","Mercury","Venus")
_TRIM_EVEN_LORDS  = ("Venus","Mercury","Jupiter","Saturn","Mars")

# Cache for per-N factors: (scale=N/30, width=30/N)
_PARTS_FACTORS: Dict[int, Tuple[float, float]] = {}
def _factors_for_parts(n: int) -> Tuple[float, float]:
    n = int(n)
    if n <= 0:
        # never divide by zero; default to 1 to keep math safe
        n = 1
    f = _PARTS_FACTORS.get(n)
    if f is None:
        f = (float(n) * _INV_30, 30.0 / float(n))
        _PARTS_FACTORS[n] = f
    return f

# ─────────────────────────────────────────────────────────────────────────────
# Spec schema & Parāśara defaults
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VargaSpec:
    key: str            # "D9"
    name: str           # "Navamsa"
    parts: int          # N
    method: str         # "linear" | "hora" | "drekkana" | "trimsamsa"
    base_ofs_fn: Optional[Callable[[int], int]] = None
    step_fn: Optional[Callable[[int], int]] = None

_DEFAULT_SPECS: Dict[str, VargaSpec] = {
    "D1":  VargaSpec("D1",  "Rasi",           1,  "linear",   _base_same,            lambda s: 0),
    "D2":  VargaSpec("D2",  "Hora",           2,  "hora"),
    "D3":  VargaSpec("D3",  "Drekkana",       3,  "drekkana"),
    "D4":  VargaSpec("D4",  "Chaturthamsa",   4,  "linear",
                     lambda s: _base_by_modality(s, 0, 8, 3),
                     lambda s: 3),
    "D7":  VargaSpec("D7",  "Saptamsa",       7,  "linear",
                     lambda s: _base_by_parity(s, 0, 6),
                     lambda s: 1),
    "D9":  VargaSpec("D9",  "Navamsa",        9,  "linear",
                     lambda s: _base_by_modality(s, 0, 8, 4),
                     lambda s: 1),
    "D10": VargaSpec("D10", "Dasamsa",       10,  "linear",
                     lambda s: _base_by_parity(s, 0, 8),
                     lambda s: 1),
    "D12": VargaSpec("D12", "Dvadashamsa",   12,  "linear",
                     _base_same,                                 lambda s: 1),
    "D16": VargaSpec("D16", "Shodasamsa",    16,  "linear",
                     lambda s: _base_by_parity(s, 0, 8),
                     lambda s: 1),
    "D20": VargaSpec("D20", "Vimsamsa",      20,  "linear",
                     lambda s: _base_by_modality(s, 0, 8, 4),
                     lambda s: 1),
    "D24": VargaSpec("D24", "Chaturvimsh.",  24,  "linear",
                     lambda s: _base_by_parity(s, 0, 8),
                     lambda s: 1),
    "D27": VargaSpec("D27", "Bhamsa",        27,  "linear",
                     lambda s: (0 if s in _ODD_RASI else 6),
                     lambda s: 1),
    "D30": VargaSpec("D30", "Trimsamsa",     30,  "trimsamsa"),
    "D40": VargaSpec("D40", "Khavedamsa",    40,  "linear",
                     lambda s: (0 if s in _ODD_RASI else 6),
                     lambda s: 1),
    "D45": VargaSpec("D45", "Akshavedamsa",  45,  "linear",
                     lambda s: _base_by_modality(s, 0, 8, 4),
                     lambda s: 1),
    "D60": VargaSpec("D60", "Shashtiamsa",   60,  "linear",
                     lambda s: (0 if s in _ODD_RASI else 6),
                     lambda s: 1),
}

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
_ALIASES.update({k.lower(): v for k, v in _CUSTOM_ALIASES.items()})  # type: ignore

# Plug custom specs, preserving API
if _CUSTOM_SPECS:
    for k, v in _CUSTOM_SPECS.items():
        try:
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
# Internals: per-method mapping (fast)
# ─────────────────────────────────────────────────────────────────────────────
def _part_index_in_sign(deg_in_sign: float, parts: int) -> int:
    """
    Return 0..parts-1 using half-open intervals [k*w, (k+1)*w).
    Uses scale N/30 to avoid a division in the hot path.
    """
    parts = int(parts) if parts else 1
    scale, _width = _factors_for_parts(parts)
    # subtract tiny epsilon so exact multiples at 30/N land in the lower bucket deterministically
    x = max(0.0, min(float(deg_in_sign) - _EPS, 30.0 - _EPS))
    k = int(math.floor(x * scale))
    if k >= parts:
        k = parts - 1
    if k < 0:
        k = 0
    return k

def _linear_varga_sign(s: int, part_k: int, base_fn: Callable[[int], int], step_fn: Callable[[int], int]) -> int:
    base = int(base_fn(int(s))) % 12
    step = int(step_fn(int(s)))
    return (base + (part_k * step)) % 12

def _drekkana_sign(s: int, part_k: int) -> int:
    # Odd: s + (0,4,8); Even: s + (0,8,4)
    s = int(s) % 12
    if part_k == 0:
        return s
    if s in _ODD_RASI:
        return (s + (4 if part_k == 1 else 8)) % 12
    else:
        return (s + (8 if part_k == 1 else 4)) % 12

def _hora_sign(s: int, part_k: int) -> int:
    # Parāśara (Cancer/Leo only).
    # Odd sign: first half → Leo(4), second → Cancer(3).
    # Even sign: first → Cancer(3), second → Leo(4).
    s = int(s) % 12
    if s in _ODD_RASI:
        return 4 if part_k == 0 else 3
    else:
        return 3 if part_k == 0 else 4

def _trimsamsa_segment(s: int, deg_in_sign: float) -> Tuple[str, float]:
    """
    Returns (lord, bound_deg) using Parāśara table.
    Odd rāśi: 5° Mars, 5° Saturn, 8° Jupiter, 7° Mercury, 5° Venus
    Even rāśi: 5° Venus, 7° Mercury, 8° Jupiter, 5° Saturn, 5° Mars
    """
    odd = (int(s) % 12) in _ODD_RASI
    cuts  = _TRIM_ODD_BOUNDS  if odd else _TRIM_EVEN_BOUNDS
    lords = _TRIM_ODD_LORDS   if odd else _TRIM_EVEN_LORDS
    x = max(0.0, min(float(deg_in_sign), 30.0 - _EPS))
    last = 0.0
    for c, L in zip(cuts, lords):
        if x < c or abs(x - c) < _EPS:
            return L, c
        last = c
    # Should not reach; return last segment as guard
    return lords[-1], cuts[-1]

def _trimsamsa_sign_and_lord(s: int, deg_in_sign: float) -> Tuple[int, str]:
    lord, _ = _trimsamsa_segment(s, deg_in_sign)
    if (int(s) % 12) in _ODD_RASI:
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
    if kl.startswith("d") and kl[1:].isdigit() and ("D"+kl[1:]) in _DEFAULT_SPECS:
        return "D"+kl[1:]
    raise ValueError(f"Unsupported varga key '{varga}'")

def _position_given_nirayana(L: float, spec: VargaSpec, *, include_nirayana: bool) -> Dict[str, Any]:
    """
    Takes *sidereal/nirayana* longitude L (0..360) and a VargaSpec,
    returns placement fields with deterministic boundary policy.
    """
    L = _norm360(L)
    s = _rasi_index(L)
    din = _lon_in_rasi(L)
    N = int(spec.parts)
    # method branches
    if spec.method == "trimsamsa":
        k = _part_index_in_sign(din, N)  # 0..29
        vs, lord_nm = _trimsamsa_sign_and_lord(s, din)

        # stretched longitude: linear within the segment scaled to 0..30
        odd = s in _ODD_RASI
        bounds = _TRIM_ODD_BOUNDS if odd else _TRIM_EVEN_BOUNDS
        last = 0.0
        st = 0.0
        seg_w = 30.0
        x = din
        for b in bounds:
            if x < b or abs(x - b) < _EPS:
                st = last
                seg_w = b - last
                break
            last = b
        frac = 0.0 if seg_w <= 0.0 else (x - st) / seg_w
        varga_lon = max(0.0, min(frac * 30.0, 30.0 - _EPS))
        out = {
            "rasi_index": s,
            "rasi_name": _RASI_NAMES[s],
            "division_index": k + 1,
            "division_count": N,
            "varga_rasi_index": vs,
            "varga_rasi_name": _RASI_NAMES[vs],
            "varga_longitude_deg": varga_lon,
            "trimsamsa_lord": lord_nm,
        }
    elif spec.method == "hora":
        k = _part_index_in_sign(din, N)  # 0 or 1
        vs = _hora_sign(s, k)
        varga_lon = din * float(N) - (k * 30.0)  # 0..30
        if varga_lon < 0.0: varga_lon = 0.0
        if varga_lon > (30.0 - _EPS): varga_lon = 30.0 - _EPS
        out = {
            "rasi_index": s,
            "rasi_name": _RASI_NAMES[s],
            "division_index": k + 1,
            "division_count": N,
            "varga_rasi_index": vs,
            "varga_rasi_name": _RASI_NAMES[vs],
            "varga_longitude_deg": varga_lon,
        }
    elif spec.method == "drekkana":
        k = _part_index_in_sign(din, N)  # 0..2
        vs = _drekkana_sign(s, k)
        varga_lon = din * float(N) - (k * 30.0)
        if varga_lon < 0.0: varga_lon = 0.0
        if varga_lon > (30.0 - _EPS): varga_lon = 30.0 - _EPS
        out = {
            "rasi_index": s,
            "rasi_name": _RASI_NAMES[s],
            "division_index": k + 1,
            "division_count": N,
            "varga_rasi_index": vs,
            "varga_rasi_name": _RASI_NAMES[vs],
            "varga_longitude_deg": varga_lon,
        }
    else:
        # linear family: D1, D4, D7, D9, D10, D12, D16, D20, D24, D27, D40, D45, D60
        k = _part_index_in_sign(din, N)
        base_fn = spec.base_ofs_fn or _base_same
        step_fn = spec.step_fn or (lambda _s: 1)
        vs = _linear_varga_sign(s, k, base_fn, step_fn)
        varga_lon = din * float(N) - (k * 30.0)
        if varga_lon < 0.0: varga_lon = 0.0
        if varga_lon > (30.0 - _EPS): varga_lon = 30.0 - _EPS
        out = {
            "rasi_index": s,
            "rasi_name": _RASI_NAMES[s],
            "division_index": k + 1,
            "division_count": N,
            "varga_rasi_index": vs,
            "varga_rasi_name": _RASI_NAMES[vs],
            "varga_longitude_deg": varga_lon,
        }

    out["nirayana_longitude_deg"] = float(L) if include_nirayana else None
    return out

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
        ayanamsa    None|float|str
    Output
        Dict with keys:
          rasi_index, rasi_name, division_index (1..N), division_count (=N),
          varga_rasi_index, varga_rasi_name, varga_longitude_deg (0..30),
          trimsamsa_lord (only for D30),
          nirayana_longitude_deg (sidereal value used by engine).
    """
    key = _resolve_key(varga)
    spec = _DEFAULT_SPECS[key]

    # Normalize input longitude first for safety
    try:
        L = float(lon_deg)
    except Exception:
        raise ValueError("invalid input longitude")

    sidereal = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    extras: Dict[str, Any] = {}
    if sidereal:
        ay, note = _resolve_ayanamsa(ayanamsa)
        if note:
            extras["ayanamsa_note"] = note
        L = _norm360(L - ay)
    else:
        L = _norm360(L)

    core = _position_given_nirayana(L, spec, include_nirayana=True)
    core.update({"varga": spec.key, "name": spec.name})
    core.update(extras)
    return core

# ─────────────────────────────────────────────────────────────────────────────
# Batch APIs (optimized & defensive)
# ─────────────────────────────────────────────────────────────────────────────
def compute_varga_chart(
    longitudes_by_name: Dict[str, float],
    varga: str,
    *,
    zodiac_mode: str = "sidereal",
    ayanamsa: Optional[Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Map planet/point → varga placement dict (varga_position output).
    Resolves spec & ayanāṁśa once; tolerates bad inputs per-point.
    """
    key = _resolve_key(varga)
    spec = _DEFAULT_SPECS[key]

    sidereal = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    ay_val = 0.0
    ay_note: Optional[str] = None
    if sidereal:
        ay_val, ay_note = _resolve_ayanamsa(ayanamsa)

    out: Dict[str, Dict[str, Any]] = {}
    setitem = out.__setitem__

    for name, lon in longitudes_by_name.items():
        nm = str(name)
        try:
            L = _norm360(float(lon) - ay_val) if sidereal else _norm360(float(lon))
            core = _position_given_nirayana(L, spec, include_nirayana=True)
            core.update({"varga": spec.key, "name": spec.name})
            if sidereal and ay_note:
                core["ayanamsa_note"] = ay_note
            setitem(nm, core)
        except Exception as e:
            setitem(nm, {"error": str(e)})
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
    Returns: { 'D9': {'Sun': {...}, ...}, 'D10': {...}, ... }
    Optimized to reuse nirayana/tropical longitudes and resilient to bad inputs.
    """
    # Resolve ayanāṁśa once if sidereal
    sidereal = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    ay_val = 0.0
    ay_note: Optional[str] = None
    if sidereal:
        ay_val, ay_note = _resolve_ayanamsa(ayanamsa)

    # Pre-normalize longitudes once
    Lmap: Dict[str, float] = {}
    for name, lon in longitudes_by_name.items():
        nm = str(name)
        try:
            Lmap[nm] = _norm360(float(lon) - ay_val) if sidereal else _norm360(float(lon))
        except Exception:
            # keep it out; per-var step will report error for this name
            pass

    res: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for v in vargas:
        try:
            key = _resolve_key(v)
            spec = _DEFAULT_SPECS[key]
        except Exception as e:
            res[str(v)] = {"_error": {"error": str(e)}}
            continue

        sub: Dict[str, Dict[str, Any]] = {}
        setsub = sub.__setitem__
        for name in longitudes_by_name.keys():
            nm = str(name)
            if nm not in Lmap:
                setsub(nm, {"error": "invalid_longitude"})
                continue
            try:
                core = _position_given_nirayana(Lmap[nm], spec, include_nirayana=True)
                core.update({"varga": spec.key, "name": spec.name})
                if sidereal and ay_note:
                    core["ayanamsa_note"] = ay_note
                setsub(nm, core)
            except Exception as e:
                setsub(nm, {"error": str(e)})
        res[key] = sub
    return res

# ─────────────────────────────────────────────────────────────────────────────
# Lightweight self-checks
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # D3 quick probes: Aries 0°, 10°, 20° → Aries, Leo, Sagittarius
    samples = [0.0, 10.0, 20.0, 29.999999]
    for x in samples:
        d3 = varga_position(x, "D3")
        assert d3["varga_rasi_name"] in ("Aries","Leo","Sagittarius")

    # D2 Hora: Aries (odd) 0..15 → Leo, 15..30 → Cancer
    h1 = varga_position(0.0, "D2")
    h2 = varga_position(14.9999, "D2")
    h3 = varga_position(15.0, "D2")
    h4 = varga_position(29.9999, "D2")
    assert h1["varga_rasi_name"] == "Leo" and h2["varga_rasi_name"] == "Leo"
    assert h3["varga_rasi_name"] == "Cancer" and h4["varga_rasi_name"] == "Cancer"

    print("Varga engine probes OK.")
