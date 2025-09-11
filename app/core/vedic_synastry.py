# app/core/vedic_synastry.py
# -*- coding: utf-8 -*-
"""
Vedic Synastry (Kundli Milan) module (v1.0, sidereal-only)

Implements:
- Ashta Koota scoring (Varna, Vashya, Tara, Yoni, Graha Maitri, Gana, Bhakoot, Nadi)
- Manglik (Mangal Dosha) check (lagna-based if available, else Moon-based heuristic)
- Clean warnings/meta and stable error taxonomy

Public API
----------
compute_vedic_synastry(natal_a, natal_b, **kwargs) -> dict
compute_vedic_guna_milan(natal_a, natal_b, **kwargs) -> dict
vedic_synastry_report(natal_a, natal_b, **kwargs) -> dict
compute_vedic_composite(...) -> dict (returns composite_value_error; not a Vedic concept)

Error taxonomy (consistent)
---------------------------
- validation_error
- synastry_computation_failed
- composite_value_error
- synastry_report_failed
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# ───────── Resilient Imports ─────────
try:
    from app.core.timescales import build_timescales
    _TS_OK = True
    _TS_ERR = None
except Exception as e:  # pragma: no cover
    build_timescales = None  # type: ignore
    _TS_OK = False
    _TS_ERR = e

try:
    from app.core.ephemeris_adapter import EphemerisAdapter
    _EPH_OK = True
    _EPH_ERR = None
except Exception as e:  # pragma: no cover
    EphemerisAdapter = None  # type: ignore
    _EPH_OK = False
    _EPH_ERR = e

# Houses optional (for lagna-based Manglik)
try:
    from app.core.houses import compute_houses_with_policy
    _HOUSES_OK = True
    _HOUSES_ERR = None
except Exception as e:  # pragma: no cover
    compute_houses_with_policy = None  # type: ignore
    _HOUSES_OK = False
    _HOUSES_ERR = e


# ───────── Constants / Tables ─────────

SIGNS = (
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
)

SIGN_LORD = {
    "Aries":"Mars","Taurus":"Venus","Gemini":"Mercury","Cancer":"Moon",
    "Leo":"Sun","Virgo":"Mercury","Libra":"Venus","Scorpio":"Mars",
    "Sagittarius":"Jupiter","Capricorn":"Saturn","Aquarius":"Saturn","Pisces":"Jupiter",
}

# Natural friendships (classical)
FRIENDS = {
    "Sun": {"Moon","Mars","Jupiter"},
    "Moon": {"Sun","Mercury"},
    "Mars": {"Sun","Moon","Jupiter"},
    "Mercury": {"Sun","Venus"},
    "Jupiter": {"Sun","Moon","Mars"},
    "Venus": {"Mercury","Saturn"},
    "Saturn": {"Mercury","Venus"},
}
NEUTRAL = {
    "Sun": {"Mercury","Venus"},
    "Moon": {"Mars","Jupiter","Venus","Saturn"},
    "Mars": {"Venus","Saturn"},
    "Mercury": {"Mars","Jupiter","Saturn"},
    "Jupiter": {"Saturn"},
    "Venus": {"Jupiter"},
    "Saturn": {"Jupiter"},
}
ENEMIES = {
    "Sun": {"Saturn"},
    "Moon": set(),
    "Mars": {"Mercury"},
    "Mercury": {"Moon"},
    "Jupiter": {"Venus","Mercury"},
    "Venus": {"Sun","Moon","Mars"},
    "Saturn": {"Sun","Moon","Mars"},
}

# Nakshatra list in order with meta: (name, gana, nadi, yoni)
# Gana: Deva, Manushya, Rakshasa
# Nadi: Adi, Madhya, Antya
# Yoni: classic animal mapping
NAKSHATRAS = [
    ("Ashwini",      "Deva","Adi",    "Horse"),
    ("Bharani",      "Manushya","Madhya","Elephant"),
    ("Krittika",     "Rakshasa","Antya","Sheep"),
    ("Rohini",       "Manushya","Adi",  "Serpent"),
    ("Mrigashira",   "Deva","Madhya","Serpent"),
    ("Ardra",        "Manushya","Antya","Dog"),
    ("Punarvasu",    "Deva","Adi",    "Cat"),
    ("Pushya",       "Deva","Madhya","Sheep"),
    ("Ashlesha",     "Rakshasa","Antya","Cat"),
    ("Magha",        "Rakshasa","Adi",  "Rat"),
    ("Purva Phalguni","Manushya","Madhya","Rat"),
    ("Uttara Phalguni","Manushya","Antya","Cow"),
    ("Hasta",        "Deva","Adi",    "Buffalo"),
    ("Chitra",       "Rakshasa","Madhya","Tiger"),
    ("Swati",        "Deva","Antya","Buffalo"),
    ("Vishakha",     "Rakshasa","Adi","Tiger"),
    ("Anuradha",     "Deva","Madhya","Deer"),
    ("Jyeshtha",     "Rakshasa","Antya","Deer"),
    ("Mula",         "Rakshasa","Adi","Dog"),
    ("Purva Ashadha","Manushya","Madhya","Monkey"),
    ("Uttara Ashadha","Manushya","Antya","Mongoose"),
    ("Shravana",     "Deva","Adi","Monkey"),
    ("Dhanishtha",   "Rakshasa","Madhya","Lion"),
    ("Shatabhisha",  "Rakshasa","Antya","Horse"),
    ("Purva Bhadrapada","Manushya","Adi","Lion"),
    ("Uttara Bhadrapada","Manushya","Madhya","Cow"),
    ("Revati",       "Deva","Antya","Elephant"),
]

# Vashya categories (by sign)
VASHYA_GROUP = {
    "Leo":"Chatushpad", "Aries":"Chatushpad", "Taurus":"Chatushpad",
    "Cancer":"Jalachara", "Pisces":"Jalachara","Scorpio":"Jalachara",
    "Gemini":"Manava","Virgo":"Manava","Libra":"Manava","Aquarius":"Manava",
    "Sagittarius":"Vanachara","Capricorn":"Vanachara",
}

# Varna order (Brahmin > Kshatriya > Vaishya > Shudra)
VARNA_ORDER = {"Cancer":"Brahmin","Scorpio":"Brahmin","Pisces":"Brahmin",
               "Aries":"Kshatriya","Leo":"Kshatriya","Sagittarius":"Kshatriya",
               "Taurus":"Vaishya","Virgo":"Vaishya","Capricorn":"Vaishya",
               "Gemini":"Shudra","Libra":"Shudra","Aquarius":"Shudra"}
VARNA_RANK = {"Brahmin":3,"Kshatriya":2,"Vaishya":1,"Shudra":0}

# Points per Koota (total 36)
KootaPoints = {
    "Varna": 1.0, "Vashya": 2.0, "Tara": 3.0, "Yoni": 4.0,
    "GrahaMaitri": 5.0, "Gana": 6.0, "Bhakoot": 7.0, "Nadi": 8.0,
}

PASS_THRESHOLD = 18.0  # Commonly used acceptance threshold


# ───────── Small helpers ─────────

def _normalize_angle(deg: float) -> float:
    return deg % 360.0

def _sign_index(lon: float) -> int:
    return int(_normalize_angle(lon) // 30)  # 0..11

def _nakshatra_index(lon_sidereal: float) -> int:
    # Each Nakshatra = 13°20' = 13.333333... deg
    part = 13.333333333333334
    return int(_normalize_angle(lon_sidereal) // part)  # 0..26

def _pada_index(lon_sidereal: float) -> int:
    # Pada = 3°20' = 3.3333333333333335 deg, 1..4
    part = 3.3333333333333335
    return int((_normalize_angle(lon_sidereal) % 13.333333333333334) // part) + 1

def _resolve_timescales(natal: Dict[str, Any], jd_tt: Optional[float], jd_ut1: Optional[float]) -> Tuple[float, float, List[str]]:
    warnings: List[str] = []
    if jd_tt is not None and jd_ut1 is not None:
        return float(jd_tt), float(jd_ut1), warnings
    if not _TS_OK:
        raise RuntimeError(f"Timescales builder unavailable: {_TS_ERR}")
    date_str = natal.get("date"); time_str = natal.get("time"); tz_name = natal.get("place_tz")
    if not (date_str and time_str and tz_name):
        raise ValueError("missing date/time/place_tz in natal data")
    ts = build_timescales(str(date_str), str(time_str), str(tz_name), 0.0)
    warnings.append("timescales_computed_with_dut1_0")
    return float(ts.jd_tt), float(ts.jd_ut1), warnings

def _get_moon_longitude(jd_tt: float, place: Optional[Dict[str, Any]], frame: str = "ecliptic-of-date") -> Tuple[float, List[str]]:
    if not _EPH_OK:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPH_ERR}")
    adapter = EphemerisAdapter(frame=frame)
    warnings: List[str] = []
    kwargs: Dict[str, Any]
    if place and all(k in place for k in ("latitude","longitude")):
        kwargs = {"jd_tt": float(jd_tt), "bodies": ["Moon"], "center": "topocentric",
                  "latitude": float(place["latitude"]), "longitude": float(place["longitude"]),
                  "elevation_m": float(place.get("elev_m", 0.0))}
    else:
        kwargs = {"jd_tt": float(jd_tt), "bodies": ["Moon"], "center": "geocentric"}
        if not place:
            warnings.append("geocentric_no_coordinates")
        else:
            warnings.append("geocentric_missing_coordinates")
    result = None
    if hasattr(adapter, "ecliptic_longitudes"):
        try:
            result = adapter.ecliptic_longitudes(**kwargs)  # type: ignore[arg-type]
        except Exception as e:
            warnings.append(f"moon_longitude_failed_{type(e).__name__}")
    if not isinstance(result, dict):
        raise RuntimeError("Moon longitude not available")
    # Accept either {"Moon": deg} or {"results":[{"name":"Moon","longitude":deg}]}
    if "results" in result and isinstance(result["results"], list) and result["results"]:
        moon_lon = float(result["results"][0].get("longitude", result["results"][0].get("lon", 0.0)))
    else:
        moon_lon = float(result.get("Moon", 0.0))
    return _normalize_angle(moon_lon), warnings

def _apply_ayanamsa(longitude_tropical: float, ayanamsa_deg: float) -> float:
    return _normalize_angle(longitude_tropical - float(ayanamsa_deg))

def _moon_context(natal: Dict[str, Any], jd_tt: float, ayanamsa_deg: float, frame: str = "ecliptic-of-date") -> Dict[str, Any]:
    lon_trop, w = _get_moon_longitude(jd_tt, natal, frame)
    lon_sid = _apply_ayanamsa(lon_trop, ayanamsa_deg)
    sidx = _sign_index(lon_sid)
    nidx = _nakshatra_index(lon_sid)
    return {
        "warnings": w,
        "lon_tropical": lon_trop,
        "lon_sidereal": lon_sid,
        "sign_index": sidx,
        "sign": SIGNS[sidx],
        "nakshatra_index": nidx,
        "nakshatra": NAKSHATRAS[nidx][0],
        "pada": _pada_index(lon_sid),
        "lord": SIGN_LORD[SIGNS[sidx]],
    }

# ───────── Koota calculators ─────────

def _varna_points(sign_a: str, sign_b: str) -> float:
    va = VARNA_ORDER[sign_a]; vb = VARNA_ORDER[sign_b]
    return KootaPoints["Varna"] if VARNA_RANK[va] >= VARNA_RANK[vb] else 0.0

def _vashya_points(sign_a: str, sign_b: str) -> float:
    # Simple rule-set: same group = full, friendly = partial, else 0
    ga = VASHYA_GROUP[sign_a]; gb = VASHYA_GROUP[sign_b]
    if ga == gb:
        return KootaPoints["Vashya"]
    # Friendly pairs (broad heuristic commonly used)
    friendly = {
        ("Chatushpad","Vanachara"), ("Jalachara","Manava"), ("Manava","Jalachara"),
    }
    return KootaPoints["Vashya"] * 0.5 if (ga, gb) in friendly or (gb, ga) in friendly else 0.0

def _tara_points(nidx_a: int, nidx_b: int) -> float:
    # Count from A's janma to B's; if count % 9 in {2,4,6,8} → good; {3,5,7} → mixed; {1,0} → poor
    count = (nidx_b - nidx_a) % 27
    mod = (count + 1) % 9  # traditional counting includes start
    if mod in {2,4,6,8}:
        return KootaPoints["Tara"]
    if mod in {3,5,7}:
        return KootaPoints["Tara"] * 0.5
    return 0.0

def _yoni_points(nidx_a: int, nidx_b: int) -> float:
    y_a = NAKSHATRAS[nidx_a][3]; y_b = NAKSHATRAS[nidx_b][3]
    if y_a == y_b:
        # Same yoni traditionally full/near-full; many traditions penalize enemity pairs (omitted for brevity)
        return KootaPoints["Yoni"]
    # Simple enemy pairs set (sample; extend as needed)
    enemy_pairs = {("Cat","Rat"), ("Dog","Deer"), ("Tiger","Monkey"), ("Cow","Tiger")}
    if (y_a, y_b) in enemy_pairs or (y_b, y_a) in enemy_pairs:
        return 0.0
    return KootaPoints["Yoni"] * 0.5

def _graha_maitri_points(lord_a: str, lord_b: str) -> float:
    if lord_b in FRIENDS.get(lord_a, set()):
        return KootaPoints["GrahaMaitri"]
    if lord_b in NEUTRAL.get(lord_a, set()) or lord_a in NEUTRAL.get(lord_b, set()):
        return KootaPoints["GrahaMaitri"] * 0.5
    if lord_b in ENEMIES.get(lord_a, set()):
        return 0.0
    return KootaPoints["GrahaMaitri"] * 0.5

def _gana_points(nidx_a: int, nidx_b: int) -> float:
    g_a = NAKSHATRAS[nidx_a][1]; g_b = NAKSHATRAS[nidx_b][1]
    if g_a == g_b:
        return KootaPoints["Gana"]
    # Deva with Manushya ok (partial), Manushya with Rakshasa partial; Deva with Rakshasa poor
    if (g_a, g_b) in {("Deva","Manushya"),("Manushya","Deva"),("Manushya","Rakshasa"),("Rakshasa","Manushya")}:
        return KootaPoints["Gana"] * 0.5
    return 0.0

def _bhakoot_points(sidx_a: int, sidx_b: int) -> float:
    # Distance in signs (A->B); problematic: 2/12, 3/11, 5/9 in some traditions (gendered variants exist).
    dist = (sidx_b - sidx_a) % 12
    if dist in {2,12,3,11,5,9}:
        return 0.0
    return KootaPoints["Bhakoot"]

def _nadi_points(nidx_a: int, nidx_b: int) -> float:
    n_a = NAKSHATRAS[nidx_a][2]; n_b = NAKSHATRAS[nidx_b][2]
    if n_a == n_b:
        return 0.0  # same Nadi → no points
    return KootaPoints["Nadi"]

def _ashta_koota_scores(ctx_a: Dict[str, Any], ctx_b: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
    sidx_a, sidx_b = ctx_a["sign_index"], ctx_b["sign_index"]
    nidx_a, nidx_b = ctx_a["nakshatra_index"], ctx_b["nakshatra_index"]
    lord_a, lord_b = ctx_a["lord"], ctx_b["lord"]
    varna  = _varna_points(SIGNS[sidx_a], SIGNS[sidx_b])
    vashya = _vashya_points(SIGNS[sidx_a], SIGNS[sidx_b])
    tara   = _tara_points(nidx_a, nidx_b)
    yoni   = _yoni_points(nidx_a, nidx_b)
    maitri = _graha_maitri_points(lord_a, lord_b)
    gana   = _gana_points(nidx_a, nidx_b)
    bhakoot= _bhakoot_points(sidx_a, sidx_b)
    nadi   = _nadi_points(nidx_a, nidx_b)
    scores = {
        "Varna": varna, "Vashya": vashya, "Tara": tara, "Yoni": yoni,
        "GrahaMaitri": maitri, "Gana": gana, "Bhakoot": bhakoot, "Nadi": nadi,
    }
    total = sum(scores.values())
    return scores, total

# ───────── Manglik (simple) ─────────

def _manglik_check(natal: Dict[str, Any], jd_tt: float, jd_ut1: float) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Returns ({basis, houses_hit, is_manglik}, warnings) or (None, warnings) if indeterminate.
    - If houses available: use lagna houses (1,4,7,8,12) for Mars occupancy
    - Else: use Moon-lagna signs as surrogate (same indices)
    """
    warnings: List[str] = []
    # Try houses
    if _HOUSES_OK and all(k in natal for k in ("latitude","longitude")):
        try:
            hs = compute_houses_with_policy(
                jd_tt=float(jd_tt), jd_ut1=float(jd_ut1),
                latitude=float(natal["latitude"]), longitude=float(natal["longitude"]),
                elevation_m=float(natal.get("elev_m", 0.0)), system="placidus",
            )
            # Need Mars ecliptic longitude to place in house
            if not _EPH_OK:
                warnings.append("manglik_ephemeris_unavailable")
                return None, warnings
            adapter = EphemerisAdapter(frame="ecliptic-of-date")
            res = adapter.ecliptic_longitudes(jd_tt=float(jd_tt), bodies=["Mars"], center="topocentric",
                                              latitude=float(natal["latitude"]), longitude=float(natal["longitude"]),
                                              elevation_m=float(natal.get("elev_m", 0.0)))
            mars_lon = float(res.get("Mars")) if isinstance(res, dict) and "Mars" in res else float(res["results"][0]["longitude"])
            # Determine house
            cusps = hs.get("cusps_deg")
            if isinstance(cusps, list) and len(cusps) == 12:
                house = _house_of_longitude(mars_lon, cusps)
                is_m = house in {1,4,7,8,12}
                return {"basis":"lagna","house":house,"is_manglik":is_m}, warnings
        except Exception:
            warnings.append("manglik_houses_computation_failed")
    # Fallback: Moon-lagna by sign distance
    try:
        moon_lon_trop, w1 = _get_moon_longitude(jd_tt, natal, frame="ecliptic-of-date")
        warnings.extend(w1)
        ay = float(natal.get("ayanamsa_deg", 0.0))
        moon_sid = _apply_ayanamsa(moon_lon_trop, ay)
        mars_trop, w2 = _get_planet_lon(jd_tt, natal, "Mars")
        warnings.extend(w2)
        mars_sid = _apply_ayanamsa(mars_trop, ay)
        # Houses as 30° sectors from Moon-long (Chandra-lagna)
        moon_sign0 = _sign_index(moon_sid)
        mars_sign  = _sign_index(mars_sid)
        dist = (mars_sign - moon_sign0) % 12
        is_m = dist in {0,3,6,7,11}  # 1,4,7,8,12 from Moon-lagna
        return {"basis":"moon_lagna","sign_distance":int(dist),"is_manglik":is_m}, warnings
    except Exception:
        warnings.append("manglik_moon_fallback_failed")
        return None, warnings

def _get_planet_lon(jd_tt: float, place: Optional[Dict[str, Any]], body: str) -> Tuple[float, List[str]]:
    if not _EPH_OK:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPH_ERR}")
    adapter = EphemerisAdapter(frame="ecliptic-of-date")
    warnings: List[str] = []
    kwargs: Dict[str, Any]
    if place and all(k in place for k in ("latitude","longitude")):
        kwargs = {"jd_tt": float(jd_tt), "bodies": [body], "center": "topocentric",
                  "latitude": float(place["latitude"]), "longitude": float(place["longitude"]),
                  "elevation_m": float(place.get("elev_m", 0.0))}
    else:
        kwargs = {"jd_tt": float(jd_tt), "bodies": [body], "center": "geocentric"}
        warnings.append("geocentric_no_coordinates")
    result = None
    if hasattr(adapter, "ecliptic_longitudes"):
        result = adapter.ecliptic_longitudes(**kwargs)  # type: ignore[arg-type]
    if not isinstance(result, dict):
        raise RuntimeError("planet longitude not available")
    if "results" in result and isinstance(result["results"], list) and result["results"]:
        lon = float(result["results"][0].get("longitude", result["results"][0].get("lon", 0.0)))
    else:
        lon = float(result.get(body, 0.0))
    return _normalize_angle(lon), warnings

def _house_of_longitude(lon: float, cusps: List[float]) -> int:
    L = _normalize_angle(lon)
    for i in range(12):
        c0 = _normalize_angle(cusps[i]); c1 = _normalize_angle(cusps[(i+1)%12])
        if c0 <= c1:
            if c0 <= L < c1: return i+1
        else:
            if L >= c0 or L < c1: return i+1
    return 1

# ───────── Public Engines ─────────

def compute_vedic_guna_milan(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    *,
    jd_tt_a: Optional[float] = None,
    jd_ut1_a: Optional[float] = None,
    jd_tt_b: Optional[float] = None,
    jd_ut1_b: Optional[float] = None,
    ayanamsa_deg: float = 0.0,
) -> Dict[str, Any]:
    """
    Standalone Ashta Koota scorer (sidereal).
    """
    warnings_all: List[str] = []
    try:
        try:
            a_jd_tt, a_jd_ut1, wa = _resolve_timescales(natal_a, jd_tt_a, jd_ut1_a)
            b_jd_tt, b_jd_ut1, wb = _resolve_timescales(natal_b, jd_tt_b, jd_ut1_b)
        except ValueError as ve:
            return {"ok": False, "error": "validation_error", "details": str(ve), "warnings": warnings_all}
        warnings_all.extend(wa); warnings_all.extend(wb)

        if abs(ayanamsa_deg) < 1e-9:
            warnings_all.append("ayanamsa_missing_or_zero")

        ctx_a = _moon_context({**natal_a, "ayanamsa_deg": ayanamsa_deg}, a_jd_tt, ayanamsa_deg)
        ctx_b = _moon_context({**natal_b, "ayanamsa_deg": ayanamsa_deg}, b_jd_tt, ayanamsa_deg)
        warnings_all.extend(ctx_a["warnings"]); warnings_all.extend(ctx_b["warnings"])

        scores, total = _ashta_koota_scores(ctx_a, ctx_b)

        return {
            "ok": True,
            "koota_scores": scores,
            "total": total,
            "max_points": 36.0,
            "pass_threshold": PASS_THRESHOLD,
            "passed": total >= PASS_THRESHOLD,
            "meta": {
                "zodiac_mode": "sidereal",
                "ayanamsa_deg": float(ayanamsa_deg),
                "timescales": {
                    "chart_a": {"jd_tt": a_jd_tt, "jd_ut1": a_jd_ut1},
                    "chart_b": {"jd_tt": b_jd_tt, "jd_ut1": b_jd_ut1},
                },
                "moon": {
                    "chart_a": {k: v for k, v in ctx_a.items() if k != "warnings"},
                    "chart_b": {k: v for k, v in ctx_b.items() if k != "warnings"},
                },
                "warnings": warnings_all,
            },
        }
    except Exception as e:
        return {"ok": False, "error": "synastry_computation_failed", "details": str(e), "warnings": warnings_all}

def compute_vedic_synastry(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Full Vedic synastry: Ashta Koota + Manglik summary.
    kwargs:
      - ayanamsa_deg: float (required for accurate sidereal; warning if 0)
    """
    ay = float(kwargs.get("ayanamsa_deg", 0.0))
    res = compute_vedic_guna_milan(natal_a, natal_b,
                                   jd_tt_a=kwargs.get("jd_tt_a"), jd_ut1_a=kwargs.get("jd_ut1_a"),
                                   jd_tt_b=kwargs.get("jd_tt_b"), jd_ut1_b=kwargs.get("jd_ut1_b"),
                                   ayanamsa_deg=ay)
    if not res.get("ok", False):
        return res

    # Manglik (lagna if possible, else Moon-lagna fallback)
    try:
        a_jd_tt = res["meta"]["timescales"]["chart_a"]["jd_tt"]
        a_jd_ut1 = res["meta"]["timescales"]["chart_a"]["jd_ut1"]
        b_jd_tt = res["meta"]["timescales"]["chart_b"]["jd_tt"]
        b_jd_ut1 = res["meta"]["timescales"]["chart_b"]["jd_ut1"]

        mang_a, wa = _manglik_check({**natal_a, "ayanamsa_deg": ay}, a_jd_tt, a_jd_ut1)
        mang_b, wb = _manglik_check({**natal_b, "ayanamsa_deg": ay}, b_jd_tt, b_jd_ut1)
        res["meta"]["warnings"].extend(wa or []); res["meta"]["warnings"].extend(wb or [])

        res["manglik"] = {"chart_a": mang_a, "chart_b": mang_b}
    except Exception as e:
        res["meta"]["warnings"].append(f"manglik_evaluation_failed:{e}")

    res["meta"]["report_type"] = "vedic_synastry"
    return res

def vedic_synastry_report(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Vedic synastry report (Koota + Manglik). There is no Vedic 'composite' chart;
    this function focuses on compatibility & risk flags customary in Vedic practice.
    """
    try:
        syn = compute_vedic_synastry(natal_a, natal_b, **kwargs)
        # metrics
        if syn.get("ok", False):
            syn.setdefault("metrics", {})
            syn["metrics"].update({
                "koota_total": syn.get("total", 0.0),
                "passed_threshold": syn.get("passed", False),
            })
        return syn
    except Exception as e:
        return {"ok": False, "error": "synastry_report_failed", "details": str(e)}

def compute_vedic_composite(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Not a Vedic technique. Provided for API parity; always returns a structured error.
    """
    return {
        "ok": False,
        "error": "composite_value_error",
        "details": "Composite/Davison charts are not standard Vedic techniques.",
        "warnings": ["use western_synastry for composites if needed"],
    }
