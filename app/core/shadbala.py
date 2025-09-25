# app/core/shadbala.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Śaḍbala engine (project-integrated)

Public API
----------
compute_shadbala(payload: dict) -> dict
clear_caches() -> None

Notes
-----
- Uses app.core.astronomy.compute_chart for positions/angles/timescales.
- Optionally uses app.core.houses_advanced for cusps; otherwise falls back
  to whole-sign safely.
- Optionally uses app.core.varga_charts for a tiny Saptavargaja-like bonus.
- Scores are in Śaṣṭiāṁśa (0..60). Totals sum only computed parts.

Payload (minimal)
-----------------
{
  "date": "YYYY-MM-DD",
  "time": "HH:MM[:SS]",
  "tz": "IANA/Zone",
  "latitude": <float>,
  "longitude": <float>,

  # Optional
  "zodiac_mode": "sidereal" | "tropical",
  "ayanamsa": "lahiri" | "krishnamurti" | <float>,
  "elevation_m": <float>,
  "house_system": "placidus" | "koch" | "sripati" | "whole_sign" | ...,
  "angles": {"asc": <deg>, "mc": <deg>},           # overrides for dig/houses
  "house_cusps_deg": [12 floats],                  # precomputed cusps
  # or { "houses": { "cusps_deg" | "cusps": [12 floats] } }
  "include_components": [
      "naisargika","uchcha","dig","kendradi","cheshta","kala","drik","varga_bonus"
  ],
  "vargas": ["D1","D9","D30", ...]                 # for tiny varga bonus
}
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

# ────────────────────────────── Required core ────────────────────────────────
try:
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
_PLANETS = ("Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn")
_POINTS_OPTIONAL = ("North Node", "South Node")

_NAISARGIKA: Dict[str, float] = {
    "Sun": 60.0, "Moon": 51.0, "Venus": 43.0, "Jupiter": 34.0,
    "Mercury": 26.0, "Mars": 17.0, "Saturn": 9.0,
}

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
    "North Node": _z(1, 20.0),
    "South Node": _z(7, 20.0),
}

def _wrap360(x: float) -> float:
    v = float(x) % 360.0
    return 0.0 if abs(v) < 1e-12 else v

def _ang_sep(a: float, b: float) -> float:
    d = abs(_wrap360(a) - _wrap360(b))
    return d if d <= 180.0 else 360.0 - d

# ───────────────────────────── Houses helpers ────────────────────────────────
def _whole_sign_cusps(asc_deg: float) -> List[float]:
    """12 whole-sign cusps starting at ascendant sign start."""
    sign0 = int(_wrap360(asc_deg) // 30)  # 0..11
    base = 30.0 * sign0
    return [_wrap360(base + 30.0 * k) for k in range(12)]

def _house_index_from_cusps(lon: float, cusps: List[float]) -> int:
    """Return 1..12 house index using sign-like half-open bins over cusps."""
    L = _wrap360(lon)
    for i in range(12):
        a = cusps[i]
        b = cusps[(i + 1) % 12]
        span = (b - a) % 360.0
        delta = (L - a) % 360.0
        if delta < span or math.isclose(delta, span, abs_tol=1e-12):
            return i + 1
    return 1

def _class_of_house(h: int) -> str:
    if h in (1, 4, 7, 10): return "kendra"
    if h in (2, 5, 8, 11): return "panaphara"
    return "apoklima"

# ───────────────────────────── Components ────────────────────────────────────
def _naisargika_bala(name: str) -> Optional[float]:
    return _NAISARGIKA.get(name)

def _uchcha_bala(name: str, lon: Optional[float]) -> Optional[float]:
    if lon is None: return None
    ex = _EXALT.get(name)
    if ex is None: return None
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
    if pref is None: return None
    d = _ang_sep(lon, pref)
    return max(0.0, 60.0 * (1.0 - d / 180.0))

def _kendradi_bala(house_idx: Optional[int]) -> Optional[float]:
    if house_idx is None: return None
    cls = _class_of_house(house_idx)
    if cls == "kendra": return 60.0
    if cls == "panaphara": return 30.0
    return 15.0

def _cheshta_bala(_name: str, speed_deg_per_day: Optional[float]) -> Optional[float]:
    if speed_deg_per_day is None: return None
    v = float(speed_deg_per_day)
    eps = 0.05  # ~3' / day
    if v < -eps: return 60.0
    if abs(v) <= eps: return 45.0
    return 30.0

# Stubs (extend later if desired)
def _kala_bala_stub(*_a, **_k) -> Optional[float]: return None
def _drik_bala_stub(*_a, **_k) -> Optional[float]: return None

# ───────────────────────────── Varga bonus (optional) ────────────────────────
def _varga_bonus(longitudes_by_name: Dict[str, float], *, mode: str, ayanamsa: Any, vargas: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if _compute_many_vargas is None or not vargas:
        return out
    try:
        charts = _compute_many_vargas(
            longitudes_by_name,
            vargas,
            zodiac_mode=("sidereal" if mode == "sidereal" else "tropical"),
            ayanamsa=ayanamsa,
        )
    except Exception:
        return out

    rulers = {
        0:"Mars",1:"Venus",2:"Mercury",3:"Moon",4:"Sun",5:"Mercury",
        6:"Venus",7:"Mars",8:"Jupiter",9:"Saturn",10:"Saturn",11:"Jupiter",
    }

    counts: Dict[str, int] = {k: 0 for k in longitudes_by_name.keys()}
    for _vk, chart in (charts or {}).items():
        if not isinstance(chart, dict): continue
        for nm, cell in chart.items():
            if not isinstance(cell, dict): continue
            if "error" in cell: continue
            vrasi = cell.get("varga_rasi_index")
            if not isinstance(vrasi, int): continue
            owner = rulers.get(int(vrasi))
            if owner and owner == nm:
                counts[nm] = counts.get(nm, 0) + 1
            ex = _EXALT.get(nm)
            if isinstance(ex, (int, float)) and int(ex // 30) == int(vrasi):
                counts[nm] = counts.get(nm, 0) + 1

    slots_per_planet = max(1, len(vargas) * 2)  # own + exalted checks
    for nm, c in counts.items():
        out[nm] = max(0.0, min(60.0, 60.0 * (float(c) / float(slots_per_planet))))
    return out

# ───────────────────────────── Results helpers ────────────────────────────────
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

# ───────────────────────────── Input extraction ───────────────────────────────
def _pick_angles(payload: Dict[str, Any], chart: Dict[str, Any]) -> Dict[str, Optional[float]]:
    # Prefer explicit values from payload
    p_ang = payload.get("angles") or {}
    asc = p_ang.get("asc") or p_ang.get("asc_deg") or payload.get("asc") or payload.get("asc_deg")
    mc  = p_ang.get("mc")  or p_ang.get("mc_deg")  or payload.get("mc")  or payload.get("mc_deg")

    # Else fall back to chart angles
    c_ang = (chart.get("angles") or {})
    asc = asc if isinstance(asc, (int, float)) else c_ang.get("asc_deg")
    mc  = mc  if isinstance(mc,  (int, float)) else c_ang.get("mc_deg")
    try:
        asc = float(asc) if asc is not None else None
    except Exception:
        asc = None
    try:
        mc = float(mc) if mc is not None else None
    except Exception:
        mc = None
    return {"asc": asc, "mc": mc}

def _pick_cusps(payload: Dict[str, Any]) -> List[float]:
    """
    Accept precomputed house cusps from payload in any of these keys:
    - house_cusps_deg (flat list)
    - houses.cusps_deg or houses.cusps
    - cusps_deg or cusps (legacy flat)
    Returns [] if not present/valid.
    """
    # direct flat list
    direct = payload.get("house_cusps_deg")
    if isinstance(direct, (list, tuple)) and len(direct) == 12:
        try:
            return [float(_wrap360(x)) for x in direct]
        except Exception:
            pass

    # nested under "houses"
    houses_obj = payload.get("houses") or {}
    for key in ("cusps_deg", "cusps"):
        arr = houses_obj.get(key)
        if isinstance(arr, (list, tuple)) and len(arr) == 12:
            try:
                return [float(_wrap360(x)) for x in arr]
            except Exception:
                pass

    # legacy top-level
    for key in ("cusps_deg", "cusps"):
        arr = payload.get(key)
        if isinstance(arr, (list, tuple)) and len(arr) == 12:
            try:
                return [float(_wrap360(x)) for x in arr]
            except Exception:
                pass

    return []

# ───────────────────────────── Public API ────────────────────────────────────
def compute_shadbala(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Śaḍbala-like strengths (safe, extensible subset).
    """
    warnings: List[str] = []

    include_default = ["naisargika", "uchcha", "dig", "kendradi", "cheshta", "kala", "drik", "varga_bonus"]
    include = payload.get("include_components") or include_default
    include = [str(x).strip().lower() for x in include]

    # 1) Core chart (all correctness derives from here)
    chart = _compute_chart(payload)
    mode = str(
        chart.get("mode")
        or payload.get("zodiac_mode")
        or payload.get("mode")
        or "tropical"
    ).strip().lower()
    meta = dict(chart.get("meta", {}))  # may include center/frame/ayanamsa_deg etc.

    # Angles (from payload or chart)
    ang = _pick_angles(payload, chart)
    asc, mc = ang["asc"], ang["mc"]

    # Longitudes & speeds
    longs: Dict[str, float] = {}
    speeds: Dict[str, Optional[float]] = {}
    for row in (chart.get("bodies") or []):
        try:
            nm = str(row.get("name"))
            lon = float(row.get("longitude_deg", row.get("lon")))
            longs[nm] = _wrap360(lon)
            spd = row.get("speed_deg_per_day", row.get("speed"))
            speeds[nm] = (float(spd) if isinstance(spd, (int, float)) and math.isfinite(float(spd)) else None)
        except Exception:
            continue
    for row in (chart.get("points") or []):
        try:
            nm = str(row.get("name"))
            lon = float(row.get("longitude_deg", row.get("lon")))
            longs[nm] = _wrap360(lon)
        except Exception:
            continue

    # 2) Houses (preferred via houses_advanced; payload cusps allowed; fallback = whole-sign)
    house_system = payload.get("house_system")
    cusps: List[float] = []
    precomputed_source = False

    # Accept precomputed cusps (e.g., from /ops/calculate)
    cusps_from_payload = _pick_cusps(payload)
    if len(cusps_from_payload) == 12:
        cusps = cusps_from_payload
        precomputed_source = True

    # Else try advanced house engine
    if (not cusps) and _compute_houses is not None:
        try:
            hv = _compute_houses({"angles": {"asc": asc, "mc": mc}, **meta},
                                 system=house_system, payload=payload)  # type: ignore
            cusps = list(hv.get("cusps", [])) if isinstance(hv, dict) else []
        except Exception as e:
            warnings.append(f"houses_advanced_failed:{type(e).__name__}")

    # Else whole-sign from ASC
    if (not cusps) and isinstance(asc, (int, float)):
        cusps = _whole_sign_cusps(float(asc))
        warnings.append("houses_fallback_whole_sign")

    # 3) Optional varga-based bonus
    varga_bonus: Dict[str, float] = {}
    if "varga_bonus" in include:
        try:
            varga_list = payload.get("vargas") or ["D1", "D9", "D30"]
            varga_bonus = _varga_bonus(longs, mode=mode, ayanamsa=payload.get("ayanamsa"), vargas=varga_list)
        except Exception as e:
            warnings.append(f"varga_bonus_failed:{type(e).__name__}")

    # 4) Assemble per-planet components
    results: Dict[str, Any] = {}
    names = [p for p in _PLANETS if p in longs] + [q for q in _POINTS_OPTIONAL if q in longs]

    for nm in names:
        lon = longs.get(nm)
        spd = speeds.get(nm)

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
            comps["kendradi"] = _Comp(_kendradi_bala(hidx), None if hidx is not None else "houses_missing")

        if "cheshta" in include:
            comps["cheshta"] = _Comp(_cheshta_bala(nm, spd), None if spd is not None else "speed_missing")

        if "kala" in include:
            comps["kala"] = _Comp(_kala_bala_stub(), "not_computed")

        if "drik" in include:
            comps["drik"] = _Comp(_drik_bala_stub(), "not_computed")

        if "varga_bonus" in include:
            vb = varga_bonus.get(nm)
            comps["varga_bonus"] = _Comp(
                vb if vb is not None else None,
                None if vb is not None else ("disabled" if not _compute_many_vargas else "not_available"),
            )

        total, maxsum, details = _sum_components(comps)
        pct = (total / maxsum) if maxsum > 0 else None

        results[nm] = {
            "components": details,
            "total": total,
            "max": maxsum,
            "pct": pct,
        }

    # 5) Meta + warnings (after building all planets)
    house_src = (
        "payload"
        if (len(cusps) == 12 and precomputed_source)
        else (
            "houses_advanced"
            if (len(cusps) == 12 and _compute_houses is not None)
            else ("whole_sign_fallback" if len(cusps) == 12 else "none")
        )
    )

    out_meta: Dict[str, Any] = {
        "module": "shadbala(core)",
        "mode": mode,
        "angles": {"asc": asc, "mc": mc},
        "houses_source": house_src,
        "varga_bonus": {"enabled": bool(_compute_many_vargas and "varga_bonus" in include)},
        "source_chart": {
            "center": chart.get("meta", {}).get("center", chart.get("center")),
            "frame": chart.get("meta", {}).get("frame"),
            "ayanamsa_deg": chart.get("meta", {}).get("ayanamsa_deg"),
        },
        "version": 1,
    }

    if len(cusps) == 12:
        # ensure wrapped/float cusps in meta for debugging/routing
        out_meta["house_cusps_deg"] = [float(_wrap360(x)) for x in cusps]

    warnings_out = list(set(warnings + list(chart.get("warnings") or [])))

    return {
        "ok": True,
        "bala": results,
        "meta": out_meta,
        "warnings": warnings_out,
    }


def clear_caches() -> None:
    """Currently stateless; placeholder for future memoization."""
    pass


__all__ = ["compute_shadbala", "clear_caches"]
