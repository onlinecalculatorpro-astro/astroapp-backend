# app/core/shadbala.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Śaḍbala engine (wired to your project stack)

Primary public API (stable):
    compute_shadbala(payload: dict) -> dict

Optional helpers (public but stable-by-intent):
    clear_caches() -> None

Notes
-----
- Uses app.core.astronomy.compute_chart for high-precision longitudes, speeds,
  and angles (ASC/MC). That module already routes through ephemeris_adapter.
- Optionally uses app.core.houses_advanced (if present) for house cusps and
  better Kendrādi/Dig/Kāla components; falls back to whole-sign safely.
- Optionally uses app.core.varga_charts for simple Saptavargaja-like bonus.
- Computes a practical subset out-of-the-box: Naisargika, Dig, Uchcha,
  Kendrādi (approx), and Cheshtā. Drik & full Kāla are scaffolded with
  graceful “not_computed” markers so you can extend formulas later.
- All scores are in Śaṣṭiāṁśa units (0–60). Totals sum only computed parts.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import math
import os

# ────────────────────────────── Required core ────────────────────────────────
try:
    # single source of truth (positions/angles/timescales)
    from app.core.astronomy import compute_chart as _compute_chart
except Exception as e:
    raise RuntimeError(f"shadbala: astronomy.compute_chart import failed: {e}")

# Optional helpers (graceful degrade when missing)
try:
    from app.core.houses_advanced import compute_houses as _compute_houses  # type: ignore
except Exception:
    _compute_houses = None  # type: ignore

try:
    from app.core.varga_charts import compute_many_vargas as _compute_many_vargas  # type: ignore
except Exception:
    _compute_many_vargas = None  # type: ignore

# ────────────────────────────── Config / constants ───────────────────────────
# Planet set (names as produced by astronomy module)
_PLANETS = ("Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn")
_POINTS_OPTIONAL = ("North Node", "South Node")

# Naisargika bala (Śaṣṭiāṁśa)
_NAISARGIKA: Dict[str, float] = {
    "Sun": 60.0, "Moon": 51.0, "Venus": 43.0, "Jupiter": 34.0,
    "Mercury": 26.0, "Mars": 17.0, "Saturn": 9.0,
    # Nodes are not standard here; leave undefined
}

# Exaltation (absolute zodiac degrees, 0° Aries = 0.0)
# Aries..Pisces starts at: [0,30,60,90,120,150,180,210,240,270,300,330]
def _z(sign_index: int, deg_in_sign: float) -> float:
    return (30.0 * sign_index + deg_in_sign) % 360.0

_EXALT: Dict[str, float] = {
    "Sun":     _z(0, 10.0),   # Aries 10°
    "Moon":    _z(1, 3.0),    # Taurus 3°
    "Mars":    _z(9, 28.0),   # Capricorn 28°
    "Mercury": _z(5, 15.0),   # Virgo 15°
    "Jupiter": _z(3, 5.0),    # Cancer 5°
    "Venus":   _z(11, 27.0),  # Pisces 27°
    "Saturn":  _z(6, 20.0),   # Libra 20°
    # Nodes (often used in some schools)
    "North Node": _z(1, 20.0),  # Taurus 20°
    "South Node": _z(7, 20.0),  # Scorpio 20°
}

# Dig bala preferred directions (use ecliptic longitudes of angles)
# Max at preferred; min at opposite (linear 0..60 by angular separation).
def _wrap360(x: float) -> float:
    v = float(x) % 360.0
    return 0.0 if abs(v) < 1e-12 else v

def _ang_sep(a: float, b: float) -> float:
    d = abs(_wrap360(a) - _wrap360(b))
    return d if d <= 180.0 else 360.0 - d

# ─────────────────────────────── Fallback houses ─────────────────────────────
def _whole_sign_cusps(asc_deg: float) -> List[float]:
    """12 whole-sign cusps starting at Ascendant sign start."""
    sign0 = int(_wrap360(asc_deg) // 30)  # 0..11
    base = 30.0 * sign0
    return [(_wrap360(base + 30.0 * k)) for k in range(12)]

def _house_index_from_cusps(lon: float, cusps: List[float]) -> int:
    """
    Return 1..12 house index for a longitude using *sign-like* half-open bins.
    Assumes monotonic 12 cusps, 30° spaced (whole-sign or equal).
    """
    L = _wrap360(lon)
    # Find the cusp immediately preceding L moving forward from cusp[0]
    for i in range(12):
        a = cusps[i]
        b = cusps[(i + 1) % 12]
        span = (b - a) % 360.0
        delta = (L - a) % 360.0
        if delta < span or math.isclose(delta, span, abs_tol=1e-12):
            return i + 1  # 1-based
    return 1

def _class_of_house(h: int) -> str:
    """Kendra (1,4,7,10), Panaphara (2,5,8,11), Apoklima (3,6,9,12)."""
    h = int(h)
    if h in (1, 4, 7, 10): return "kendra"
    if h in (2, 5, 8, 11): return "panaphara"
    return "apoklima"

# ───────────────────────────── Component calculators ──────────────────────────
def _naisargika_bala(name: str) -> Optional[float]:
    return _NAISARGIKA.get(name)

def _uchcha_bala(name: str, lon: Optional[float]) -> Optional[float]:
    """Linear 60 at exaltation, 0 at debilitation (opposite point)."""
    if lon is None:
        return None
    ex = _EXALT.get(name)
    if ex is None:
        return None
    d = _ang_sep(lon, ex)  # 0..180
    return max(0.0, 60.0 * (1.0 - d / 180.0))

def _dig_bala(name: str, lon: Optional[float], asc: Optional[float], mc: Optional[float]) -> Optional[float]:
    if lon is None or asc is None or mc is None:
        return None
    asc = _wrap360(asc)
    mc  = _wrap360(mc)
    desc = _wrap360(asc + 180.0)
    ic   = _wrap360(mc + 180.0)

    pref_map = {
        "Sun": mc, "Mars": mc,
        "Jupiter": asc, "Mercury": asc,
        "Saturn": desc,
        "Moon": ic, "Venus": ic,
    }
    pref = pref_map.get(name)
    if pref is None:
        return None
    d = _ang_sep(lon, pref)  # 0..180
    return max(0.0, 60.0 * (1.0 - d / 180.0))

def _kendradi_bala(house_idx: Optional[int]) -> Optional[float]:
    """Simple Kendrādi: Kendra=60, Panaphara=30, Apoklima=15."""
    if house_idx is None:
        return None
    cls = _class_of_house(house_idx)
    if cls == "kendra": return 60.0
    if cls == "panaphara": return 30.0
    return 15.0

def _cheshta_bala(name: str, speed_deg_per_day: Optional[float]) -> Optional[float]:
    """
    Simplified Cheshtā bala:
      - retrograde (v<0) → 60
      - near-station (|v|<eps) → 45
      - direct (v>0) → 30
    NOTE: This is intentionally simple and safe; refine with per-planet
    dynamic ranges if you later plug in your advanced model.
    """
    if speed_deg_per_day is None:
        return None
    v = float(speed_deg_per_day)
    eps = 0.05  # ~3' / day
    if v < -eps: return 60.0
    if abs(v) <= eps: return 45.0
    return 30.0

# Placeholder stubs: return None and “not_computed” so totals exclude them
def _kala_bala_stub(*_a, **_k) -> Optional[float]: return None
def _drik_bala_stub(*_a, **_k) -> Optional[float]: return None

# ───────────────────────────── Varga-based bonus (optional) ──────────────────
def _varga_bonus(longitudes_by_name: Dict[str, float], *, mode: str, ayanamsa: Any, vargas: List[str]) -> Dict[str, float]:
    """
    Tiny, opinionated “varga bonus” (0..60, scaled) to hint Saptavargaja Bala.
    Counts how often a planet is in own sign or exaltation across the requested
    vargas. If varga module is missing, returns empty dict.
    """
    out: Dict[str, float] = {}
    if _compute_many_vargas is None or not vargas:
        return out
    try:
        charts = _compute_many_vargas(longitudes_by_name, vargas, zodiac_mode=("sidereal" if mode=="sidereal" else "tropical"), ayanamsa=ayanamsa)
    except Exception:
        return out

    # simple sign rulers
    rulers = {
        0:"Mars",1:"Venus",2:"Mercury",3:"Moon",4:"Sun",5:"Mercury",
        6:"Venus",7:"Mars",8:"Jupiter",9:"Saturn",10:"Saturn",11:"Jupiter",
    }

    # Accumulate hits: own or exalted in that varga
    counts: Dict[str, int] = {k: 0 for k in longitudes_by_name.keys()}
    total_slots = 0
    for _vk, chart in charts.items():
        if not isinstance(chart, dict): continue
        for nm, cell in chart.items():
            if not isinstance(cell, dict): continue
            if "error" in cell: continue
            vrasi = cell.get("varga_rasi_index")
            if isinstance(vrasi, int):
                total_slots += 1
                owner = rulers.get(int(vrasi))
                if owner and owner == nm:
                    counts[nm] = counts.get(nm, 0) + 1
                # treat exact exalted house as hit as well
                ex = _EXALT.get(nm)
                if isinstance(ex, (int, float)):
                    # if varga sign equals exaltation sign
                    if int(ex // 30) == int(vrasi):
                        counts[nm] = counts.get(nm, 0) + 1

    # Scale to max 60
    slots_per_planet = max(1, len(vargas) * 2)  # own+exalted checks
    for nm, c in counts.items():
        out[nm] = max(0.0, min(60.0, 60.0 * (float(c) / float(slots_per_planet))))
    return out

# ───────────────────────────── Results model / helpers ───────────────────────
@dataclass
class _Comp:
    value: Optional[float]
    note: Optional[str] = None

def _sum_components(parts: Dict[str, _Comp]) -> Tuple[float, float, Dict[str, Any]]:
    total = 0.0
    maxsum = 0.0
    details: Dict[str, Any] = {}
    for k, c in parts.items():
        if c.value is not None:
            total += float(c.value)
            maxsum += 60.0
            details[k] = {"value": float(c.value), "note": c.note}
        else:
            details[k] = {"value": None, "note": c.note or "not_computed"}
    return total, maxsum, details

# ───────────────────────────── Public API ────────────────────────────────────
def compute_shadbala(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Śaḍbala-like strengths.

    Input
    -----
    payload: dict
        Pass-through to astronomy.compute_chart with optional extras:
        - include_components: list[str] to limit components
              default = ["naisargika","uchcha","dig","kendradi","cheshta","kala","drik","varga_bonus"]
        - vargas: list[str] for the varga bonus (e.g., ["D1","D9","D30"])
        - house_system: forwarded to houses_advanced.compute_houses if available

    Output
    ------
    dict shaped as:
      {
        "bala": {
          "Sun": {"components": {...}, "total": x, "max": y, "pct": x/y, ...},
          ...
        },
        "meta": {...},
        "warnings": [...],
      }
    """
    warnings: List[str] = []
    include_default = ["naisargika","uchcha","dig","kendradi","cheshta","kala","drik","varga_bonus"]
    include: List[str] = payload.get("include_components") or include_default
    include = [str(x).strip().lower() for x in include]

    # 1) Core chart (ALL correctness derives from here)
    chart = _compute_chart(payload)
    mode = str(chart.get("mode", payload.get("mode","tropical"))).strip().lower()
    meta = dict(chart.get("meta", {}))
    angles = chart.get("angles", {}) or {}
    asc = angles.get("asc_deg")
    mc  = angles.get("mc_deg")

    # Grab longitudes & speeds for planets (names as-is from astronomy)
    longs: Dict[str, float] = {}
    speeds: Dict[str, Optional[float]] = {}
    for row in (chart.get("bodies") or []):
        try:
            nm = str(row.get("name"))
            lon = float(row.get("longitude_deg", row.get("lon")))
            longs[nm] = _wrap360(lon)
            spd = row.get("speed_deg_per_day", row.get("speed"))
            speeds[nm] = (float(spd) if isinstance(spd, (int,float)) and math.isfinite(float(spd)) else None)
        except Exception:
            continue
    # Nodes (optional)
    for row in (chart.get("points") or []):
        try:
            nm = str(row.get("name"))
            lon = float(row.get("longitude_deg", row.get("lon")))
            longs[nm] = _wrap360(lon)
        except Exception:
            continue

    # 2) Houses (preferred via houses_advanced; fallback = whole-sign)
    house_system = payload.get("house_system")
    cusps: List[float] = []
    if _compute_houses is not None:
        try:
            # expected signature: compute_houses(chart_like_dict, **opts) -> dict with 'cusps':[12 floats]
            hv = _compute_houses({"angles": {"asc": asc, "mc": mc}, **meta}, system=house_system, payload=payload)  # type: ignore
            cusps = list(hv.get("cusps", [])) if isinstance(hv, dict) else []
        except Exception as e:
            warnings.append(f"houses_advanced_failed:{type(e).__name__}")
    if (not cusps) and isinstance(asc, (int,float)):
        cusps = _whole_sign_cusps(float(asc))
        warnings.append("houses_fallback_whole_sign")

    # 3) Optional varga-based bonus (very light heuristic)
    varga_bonus: Dict[str, float] = {}
    if "varga_bonus" in include:
        try:
            varga_list = payload.get("vargas") or ["D1","D9","D30"]
            varga_bonus = _varga_bonus(longs, mode=mode, ayanamsa=payload.get("ayanamsa"), vargas=varga_list)
        except Exception as e:
            warnings.append(f"varga_bonus_failed:{type(e).__name__}")

    # 4) Assemble per-planet components
    results: Dict[str, Any] = {}
    all_names = [p for p in _PLANETS if p in longs] + [q for q in _POINTS_OPTIONAL if q in longs]

    for nm in all_names:
        lon = longs.get(nm)
        spd = speeds.get(nm)

        # resolve house index if cusps available
        hidx: Optional[int] = None
        if lon is not None and len(cusps) == 12:
            try:
                hidx = _house_index_from_cusps(lon, cusps)
            except Exception:
                hidx = None

        comps: Dict[str, _Comp] = {}

        if "naisargika" in include:
            comps["naisargika"] = _Comp(_naisargika_bala(nm), None if nm in _NAISARGIKA else "not_standard_for_body")

        if "uchcha" in include:
            comps["uchcha"] = _Comp(_uchcha_bala(nm, lon), None if _EXALT.get(nm) is not None else "no_exaltation_defined")

        if "dig" in include:
            comps["dig"] = _Comp(_dig_bala(nm, lon, asc, mc), None if asc is not None and mc is not None else "angles_missing")

        if "kendradi" in include:
            note = None if hidx is not None else "houses_missing"
            comps["kendradi"] = _Comp(_kendradi_bala(hidx), note)

        if "cheshta" in include:
            comps["cheshta"] = _Comp(_cheshta_bala(nm, spd), None if spd is not None else "speed_missing")

        if "kala" in include:
            comps["kala"] = _Comp(_kala_bala_stub(), "not_computed")

        if "drik" in include:
            comps["drik"] = _Comp(_drik_bala_stub(), "not_computed")

        if "varga_bonus" in include:
            vb = varga_bonus.get(nm)
            comps["varga_bonus"] = _Comp(vb if vb is not None else None,
                                         None if vb is not None else ("disabled" if not _compute_many_vargas else "not_available"))

        total, maxsum, details = _sum_components(comps)
        pct = (total / maxsum) if maxsum > 0 else None

        results[nm] = {
            "components": details,
            "total": total,
            "max": maxsum,
            "pct": pct,
        }

    # 5) meta + warnings
    out_meta: Dict[str, Any] = {
        "module": "shadbala(core)",
        "mode": mode,
        "angles": {"asc": asc, "mc": mc},
        "houses_source": ("houses_advanced" if _compute_houses and len(cusps) == 12 else "whole_sign_fallback"),
        "varga_bonus": {"enabled": bool(_compute_many_vargas and "varga_bonus" in include)},
        "source_chart": {
            "center": chart.get("meta", {}).get("center", chart.get("center")),
            "frame": chart.get("meta", {}).get("frame"),
            "ayanamsa_deg": chart.get("meta", {}).get("ayanamsa_deg"),
        },
        "version": 1,
    }

    # include cusps if present (helps debugging / routing)
    if len(cusps) == 12:
        out_meta["house_cusps_deg"] = [float(_wrap360(x)) for x in cusps]

    out = {
        "bala": results,
        "meta": out_meta,
        "warnings": list(set(warnings + list(chart.get("warnings") or []))),
    }
    return out


def clear_caches() -> None:
    """Currently stateless; placeholder for future memoization."""
    pass


__all__ = ["compute_shadbala", "clear_caches"]
