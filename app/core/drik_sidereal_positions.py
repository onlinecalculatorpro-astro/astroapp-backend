# app/core/drik_sidereal_positions.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Drik Sidereal Positions — high-precision Vedic backbone

Purpose
- Produce *nirayana* (sidereal) longitudes for planets/points with the same
  numerical rigor as your astronomy stack (PyERFA GAST/true ε, strict time scales).
- Attach Vedic metadata (rāśi index/name, nakṣatra + pāda via KP-style split).
- Optionally compute houses (Sripati/Equal/Whole/… via houses_advanced) and
  assign planets to bhāvas (chalit-ready).
- One clean facade other Vedic modules (dasha, yoga, ashtakavarga, etc.) can call.

Numerical policy
- Delegates all ephemeris & timekeeping to app.core.astronomy (gold pipeline).
- Houses via app.core.houses_advanced (strict jd_tt & jd_ut1 required).
- No hidden shortcuts; ayanāṃśa subtraction only from astronomy result.

Public API
    compute_sidereal_positions(payload: dict) -> dict
    compute_sidereal_chart(payload: dict) -> dict
    rashi_index(lon_deg: float) -> int          # 1..12
    rashi_name(idx: int, *, style: str = "sanskrit") -> str
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

# Core numerics: ephemeris + angles + ayanāṃśa + strict timescales
from app.core.astronomy import compute_chart as _astro_compute

# Houses, KP helpers, and house assignment
from app.core.houses_advanced import (
    PreciseHouseCalculator,
    compute_house_system as _compute_house_system,
    assign_houses as _assign_houses,
    kp_assign_for_points as _kp_assign_for_points,
)

# ─────────────────────────────── helpers ───────────────────────────────

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

# 1..12 (Mesha..Meena)
def rashi_index(lon_deg: float) -> int:
    return int(math.floor(_norm360(lon_deg) / 30.0)) + 1

_RASHI_SANSKRIT = [
    "Meṣa","Vṛṣabha","Mithuna","Karkaṭa","Siṁha","Kanyā",
    "Tulā","Vṛścika","Dhanu","Makara","Kumbha","Mīna"
]
_RASHI_EN = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

def rashi_name(idx: int, *, style: str = "sanskrit") -> str:
    i = max(1, min(12, int(idx))) - 1
    return _RASHI_SANSKRIT[i] if str(style).lower().startswith("sansk") else _RASHI_EN[i]

# Classical sign lords (Parāśara / traditional)
_RASHI_LORD = {
    1: "Mars", 2: "Venus", 3: "Mercury", 4: "Moon", 5: "Sun", 6: "Mercury",
    7: "Venus", 8: "Mars", 9: "Jupiter", 10: "Saturn", 11: "Saturn", 12: "Jupiter",
}

@dataclass
class _Site:
    latitude: Optional[float]
    longitude: Optional[float]

# ─────────────────────────── core transformation ───────────────────────────

def _decorate_bodies_sidereal(rows: List[Dict[str, Any]], *, ay_deg: float) -> List[Dict[str, Any]]:
    """
    Convert astronomy.py rows (tropical λ, speed) → sidereal + Vedic labels.
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        nm = str(row["name"])
        lam_trop = float(row["longitude_deg"])
        lam_sid = _norm360(lam_trop - ay_deg)
        r_idx = rashi_index(lam_sid)
        out.append({
            "name": nm,
            "longitude_sidereal_deg": lam_sid,
            "longitude_tropical_deg": lam_trop,
            "speed_deg_per_day": (float(row.get("speed_deg_per_day")) if isinstance(row.get("speed_deg_per_day"), (int, float)) else None),
            "rashi_index": r_idx,
            "rashi_name_sanskrit": rashi_name(r_idx, style="sanskrit"),
            "rashi_name_english": rashi_name(r_idx, style="english"),
            "rashi_lord": _RASHI_LORD.get(r_idx),
        })
    return out

def _kp_for_payload(points_deg: Dict[str, float], *, ay_deg: float) -> Dict[str, Dict[str, Any]]:
    return _kp_assign_for_points(points_deg, zodiac_mode="sidereal", ayanamsa_deg=float(ay_deg))

def _extract_times(out_ast: Dict[str, Any]) -> Tuple[float, float, float]:
    return float(out_ast["jd_ut"]), float(out_ast["jd_tt"]), float(out_ast["jd_ut1"])

# ───────────────────────── public API: positions only ─────────────────────────

def compute_sidereal_positions(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (any one of):
      - Provide jd_tt/jd_ut/jd_ut1 directly; OR
      - date, time, tz (astronomy.py resolves strict timescales)

    Options:
      - bodies: list[str] (default classic 10)
      - points: may include "North Node"/"South Node" (astronomy computes tropical then we siderealize)
      - ayanamsa: key string (default from astronomy CFG or "lahiri")

    Output:
      {
        ok, jd_ut, jd_tt, jd_ut1, ayanamsa_deg,
        bodies_sidereal: [{name, longitude_sidereal_deg, rashi_index, ...}],
        points_sidereal: [{name, longitude_sidereal_deg, ...}],
        kp: { <name>: {nakshatra_index, pada, ...}, ... },
        meta, warnings
      }
    """
    # Force sidereal mode in the astronomy engine; preserve any other user keys
    astro_in = dict(payload)
    astro_in["mode"] = "sidereal"
    out_ast = _astro_compute(astro_in)

    ay = float(out_ast.get("ayanamsa_deg") or 0.0)
    jd_ut, jd_tt, jd_ut1 = _extract_times(out_ast)

    # Bodies → sidereal + rāśi labeling
    bodies = out_ast.get("bodies", [])
    bodies_sd = _decorate_bodies_sidereal(bodies, ay_deg=ay)

    # Points (e.g., lunar nodes)
    pts = out_ast.get("points", [])
    points_sd: List[Dict[str, Any]] = []
    for p in pts:
        if p.get("longitude_deg") is None:
            points_sd.append({"name": p.get("name"), "longitude_sidereal_deg": None})
            continue
        lam_sid = _norm360(float(p["longitude_deg"]))  # astronomy already subtracted ay in sidereal mode
        points_sd.append({"name": p.get("name"), "longitude_sidereal_deg": lam_sid})

    # KP nakṣatra/pāda for all relevant longitudes (planets + angles if present)
    kp_inputs: Dict[str, float] = {}
    for r in bodies_sd:
        if isinstance(r.get("longitude_sidereal_deg"), (int, float)):
            kp_inputs[r["name"]] = float(r["longitude_sidereal_deg"])
    # angles if astronomy provided asc/mc in sidereal frame
    asc = out_ast.get("angles", {}).get("asc_deg")
    mc = out_ast.get("angles", {}).get("mc_deg")
    if isinstance(asc, (int, float)):
        kp_inputs["Ascendant"] = float(asc)
    if isinstance(mc, (int, float)):
        kp_inputs["Midheaven"] = float(mc)
    kp = _kp_for_payload(kp_inputs, ay_deg=ay)

    # response
    return {
        "ok": True,
        "jd_ut": jd_ut, "jd_tt": jd_tt, "jd_ut1": jd_ut1,
        "ayanamsa_deg": ay,
        "bodies_sidereal": bodies_sd,
        "points_sidereal": points_sd,
        "kp": kp,
        "meta": out_ast.get("meta", {}),
        "warnings": out_ast.get("warnings", []),
    }

# ───────────────────────── public API: full chart (+ houses) ─────────────────────────

def compute_sidereal_chart(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Superset that also computes houses and bhāva assignments.

    Extra inputs:
      - house_system: str (default "sripati"; any supported by houses_advanced)
      - latitude, longitude (degrees); strictly required for houses
      - topocentric: bool (astronomy handles ABER topocentric for planets; houses use jd_tt/jd_ut1)
    """
    resp = compute_sidereal_positions(payload)
    if not resp.get("ok"):
        return resp

    ay = float(resp["ayanamsa_deg"])
    jd_ut  = float(resp["jd_ut"])
    jd_tt  = float(resp["jd_tt"])
    jd_ut1 = float(resp["jd_ut1"])

    lat = payload.get("latitude")
    lon = payload.get("longitude")
    house_sys = str(payload.get("house_system", "sripati")).strip().lower()

    houses_block: Optional[Dict[str, Any]] = None
    bhava_assign: Optional[Dict[str, int]] = None
    kp_cusps: Optional[Dict[str, Dict[str, Any]]] = None

    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        # Strict house calculator (will raise if jd_tt/jd_ut1 missing; we have them)
        calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=False, enable_validation=False)
        hd = calc.calculate_houses(
            latitude=float(lat), longitude=float(lon),
            jd_ut=jd_ut, house_system=house_sys, jd_tt=jd_tt, jd_ut1=jd_ut1
        )
        houses_block = {
            "system": hd.system,
            "ascendant": float(hd.ascendant),
            "midheaven": float(hd.midheaven),
            "cusps_deg": [float(x) for x in hd.cusps],
            "vertex": hd.vertex, "eastpoint": hd.eastpoint,
            "warnings": hd.warnings,
        }

        # Bhāva assignment (chalit-style interval mapping of ecliptic λ)
        planet_lons = []
        for r in resp["bodies_sidereal"]:
            lam = r.get("longitude_sidereal_deg")
            if isinstance(lam, (int, float)):
                planet_lons.append(float(lam))
            else:
                planet_lons.append(float("nan"))
        bh_nums = _assign_houses(planet_lons, houses_block["cusps_deg"])
        bhava_assign = {resp["bodies_sidereal"][i]["name"]: int(bh_nums[i]) for i in range(len(bh_nums))}

        # KP labeling for cusps as well
        cusp_map = {f"Cusp{i+1}": float(houses_block["cusps_deg"][i]) for i in range(12)}
        kp_cusps = _kp_for_payload(cusp_map, ay_deg=ay)

    # Attach bhāvas to body rows (if available)
    if bhava_assign:
        for r in resp["bodies_sidereal"]:
            r["bhava"] = bhava_assign.get(r["name"])

    out = dict(resp)
    if houses_block is not None:
        out["houses"] = houses_block
    if kp_cusps is not None:
        out.setdefault("kp", {}).update(kp_cusps)
    return out
