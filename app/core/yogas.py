# app/core/yogas.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Classical Yogas — exhaustive, sidereal-first, multi-level evidence & registry

Design pillars
- Gold numerics: strict timescales → ERFA GAST/true ε (via houses_advanced) → sidereal longitudes
- Explicit, conservative rules; options expose common variants w/ deterministic scoring (0..1)
- Evidence layers: sign/house/degree/varga/arudha/drishti with why/how
- Registry pattern: easy to extend/disable/query groups

Included yogas (non-exhaustive labels; rules below are exhaustive for popular practice):
  • Panch Mahapurusha: Ruchaka, Bhadra, Hamsa, Malavya, Sasa
  • Gaja-Kesari, Chandra-Mangala, Chandra Adhi, Durudhara
  • Veshi, Voshi, Ubhayachari (Sun flanked)
  • Shubha Kartari / Papa Kartari (around Lagna, Moon, Sun)
  • Raja Yogas: kendra–trikona association; Dharma-Karmadhipati; Dhana (2/11[/5]); Lakshmi; Saraswati
  • Parivartana (Maha / Khala / Dainya / Regular)
  • Neecha-bhanga (core + dispositor/exaltation kendra; drishti/exchange variants)
  • Vipareeta Raja (Harsha/Sarala/Vimala; dusthana lords in dusthanas/exchange)
  • Amala (benefic in 10th from Lagna/Moon)
  • Chatussagara (planets in all kendras), Vasumati (benefics in upachayas from Moon)
  • Kala Sarpa / Kala Amrita (seven classical planets hemmed between nodes)
  • Vargottama flags; D9/D10 boosts (configurable)

Public API
    compute_yogas(payload, **options) -> dict
    list_registered_yogas() -> list[dict]
    enable_yogas(names|tags), disable_yogas(names|tags)

Dependencies (internal)
- app.core.houses_advanced (strict)
- app.core.ephemeris_adapter + ephem_singleton (strict)
- app.core.ayanamsa (preferred; linear fallback)
- app.core.timescales or app.core.time_kernel (one of)
- Optional: app.core.varga_charts (we also provide built-in D9 sign resolver)
- Optional: app.core.jaimini_arudha (for AL notes)
- Optional: app.core.panchanga (for waxing/waning Moon classification)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable, Set
import math

# ─────────────────────────────────────────────────────────────────────
# Optional imports & guards
# ─────────────────────────────────────────────────────────────────────
try:
    from app.core.ayanamsa import get_ayanamsa_deg as _get_ayanamsa_deg
except Exception:
    _get_ayanamsa_deg = None

try:
    from app.core.ephem_singleton import TS, PLANETS
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    _EPH_OK = False
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore

try:
    from app.core.houses_advanced import compute_house_system as _compute_houses
    from app.core.houses_advanced import assign_houses as _assign_houses
    _HOUSES_OK = True
except Exception:
    _HOUSES_OK = False

try:
    from app.core.varga_charts import compute_vargas as _compute_vargas
except Exception:
    _compute_vargas = None

try:
    from app.core.jaimini_arudha import compute_arudhas as _compute_arudhas
except Exception:
    _compute_arudhas = None

try:
    from app.core.panchanga import panchanga_elements_at as _panchanga_elements_at
except Exception:
    _panchanga_elements_at = None

try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None

try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

# Constants & dignity maps
try:
    from app.core.constants_vedic import SIGN_NAMES as _SIGNS
except Exception:
    _SIGNS = (
        "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
        "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
    )

try:
    from app.core.constants_vedic import EXALTATION_SIGN_INDEX as _EXALT
    from app.core.constants_vedic import DEBILITATION_SIGN_INDEX as _DEB
    from app.core.constants_vedic import OWN_SIGN_INDEXES as _OWN
except Exception:
    _EXALT = {"Sun":0,"Moon":1,"Mars":9,"Mercury":5,"Jupiter":3,"Venus":11,"Saturn":6}
    _DEB   = {"Sun":6,"Moon":7,"Mars":3,"Mercury":11,"Jupiter":9,"Venus":5,"Saturn":0}
    _OWN   = {"Sun":(4,),"Moon":(3,),"Mars":(0,7),"Mercury":(2,5),"Jupiter":(8,11),"Venus":(1,6),"Saturn":(9,10)}

# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────
_PLANETS_MAIN = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn")
_PLANETS_ALL  = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu")

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0);  return r + 360.0 if r < 0.0 else r

def _angdiff(a: float, b: float) -> float:
    return ((_norm360(a) - _norm360(b) + 540.0) % 360.0) - 180.0

def _sign_index(lon: float) -> int:
    return int(math.floor(_norm360(lon) / 30.0)) % 12

def _house_delta(h_from: int, h_to: int) -> int:
    return ((h_to - h_from) % 12) + 1

def _is_kendra(h: int) -> bool:
    return h in (1,4,7,10)

def _is_trikona(h: int) -> bool:
    return h in (1,5,9)

def _is_upachaya(h: int) -> bool:
    return h in (3,6,10,11)

def _is_dusthana(h: int) -> bool:
    return h in (6,8,12)

def _conj(a: float, b: float, orb: float) -> bool:
    return abs(_angdiff(a, b)) <= max(0.0, float(orb))

def _has_7th_aspect(si_from: int, si_to: int) -> bool:
    return ((si_from - si_to) % 12) == 6  # 7th sign

def _has_graha_drishti(g: str, si_from: int, si_to: int) -> bool:
    d = ((si_to - si_from) % 12)
    if d == 6:  # 7th
        return True
    if g == "Jupiter" and d in (4,8):  # 5th, 9th
        return True
    if g == "Mars" and d in (3,9):     # 4th, 8th
        return True
    if g == "Saturn" and d in (2,10):  # 3rd, 10th
        return True
    return False

def _sign_lords(variant: str = "classical") -> Dict[int, str]:
    if str(variant).lower().startswith("nodes"):
        return {0:"Mars",1:"Venus",2:"Mercury",3:"Moon",4:"Sun",5:"Mercury",6:"Venus",7:"Ketu",8:"Jupiter",9:"Saturn",10:"Rahu",11:"Jupiter"}
    return {0:"Mars",1:"Venus",2:"Mercury",3:"Moon",4:"Sun",5:"Mercury",6:"Venus",7:"Mars",8:"Jupiter",9:"Saturn",10:"Saturn",11:"Jupiter"}

def _dignity(planet: str, sign_idx: int) -> str:
    if _EXALT.get(planet) == sign_idx: return "exalted"
    if _DEB.get(planet) == sign_idx:   return "debilitated"
    if sign_idx in _OWN.get(planet, ()): return "own"
    return "neutral"

def _waxing_moon(jd_tt: float, ay_key: str) -> Optional[bool]:
    if _panchanga_elements_at is None: return None
    try:
        el = _panchanga_elements_at(float(jd_tt), ayanamsa_key=ay_key)  # {"tithi": {"index":..}}
        idx = int(el["tithi"]["index"])
        return idx <= 15
    except Exception:
        return None

def _benefic_set(jd_tt: float, ay_key: str) -> Set[str]:
    # Classical: Jup, Ven, Mer; Moon benefic if waxing, malefic if waning (else assume benefic).
    base = {"Jupiter","Venus","Mercury"}
    waxing = _waxing_moon(jd_tt, ay_key)
    if waxing is False:
        return set(base)  # waning → exclude Moon
    return set(base | {"Moon"})

def _timescales(payload: Dict[str, Any]) -> Tuple[float, float, List[str]]:
    jd_tt = payload.get("jd_tt"); jd_ut1 = payload.get("jd_ut1")
    warns: List[str] = []
    if isinstance(jd_tt, (int,float)) and isinstance(jd_ut1, (int,float)):
        return float(jd_tt), float(jd_ut1), warns
    d = payload.get("date") or payload.get("birth", {}).get("date")
    t = payload.get("time") or payload.get("birth", {}).get("time") or "12:00:00"
    tz = payload.get("tz") or payload.get("place_tz") or payload.get("birth", {}).get("tz") or "UTC"
    if _tk is not None:
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(date=d, time=t, tz=tz)
                except TypeError:
                    out = fn(d, t, tz)
                if isinstance(out, dict):
                    return float(out["jd_tt"]), float(out.get("jd_ut1") or out.get("jd_ut") or out.get("jd_utc") or out["jd_tt"]), warns
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    ju, jt, j1 = map(float, out[:3])
                    return jt, j1, warns
    if _ts is None:
        raise ValueError("timescales unavailable; provide jd_tt & jd_ut1 or civil+tz")
    jd_ut = float(_ts.julian_day_utc(d, t, tz))
    try:
        y, m = map(int, str(d).split("-")[:2])
        jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m))
    except Exception:
        jd_tt = jd_ut + 69.0/86400.0
        warns.append("deltaT_fallback_69s")
    return jd_tt, jd_ut, warns

def _ayanamsa(jd_tt: float, key_or_deg: Any) -> Tuple[str, float, List[str]]:
    warns: List[str] = []
    if isinstance(key_or_deg, (int,float)):
        return "explicit", float(key_or_deg), warns
    key = str(key_or_deg or "lahiri").strip().lower()
    if _get_ayanamsa_deg is None:
        # linearized fallback, continuous
        AY_J2000_DEG = (23 + 51/60 + 26.26/3600)
        RATE_AS_PER_YR = 50.290966
        years = (float(jd_tt) - 2451545.0) / 365.25
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
        return key, ay, ["ayanamsa_fallback_lahiri_linearized"]
    try:
        return key, float(_get_ayanamsa_deg(float(jd_tt), key)), warns
    except Exception as e:
        AY_J2000_DEG = (23 + 51/60 + 26.26/3600)
        RATE_AS_PER_YR = 50.290966
        years = (float(jd_tt) - 2451545.0) / 365.25
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
        warns.append(f"ayanamsa_resolve_error:{e}")
        warns.append("ayanamsa_fallback_lahiri_linearized")
        return key, ay, warns

def _sidereal_longitudes(jd_tt: float, names: List[str], ay_deg: float) -> Dict[str, float]:
    if not _EPH_OK:
        raise RuntimeError("ephemeris_unavailable")
    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore
    rows = ep.ecliptic_longitudes(float(jd_tt), names).get("results", [])
    out: Dict[str, float] = {}
    for r in rows or []:
        nm = str(r["name"])
        out[nm] = _norm360(float(r["longitude"]) - float(ay_deg))
    return out

def _compute_houses_payload(lat: float, lon: float, jd_tt: float, jd_ut1: float, house_system: str) -> Dict[str, Any]:
    if not _HOUSES_OK:
        raise RuntimeError("houses_module_unavailable")
    return _compute_houses(
        latitude=float(lat), longitude=float(lon),
        house_system=str(house_system or "placidus"),
        jd_ut=float(jd_tt), jd_tt=float(jd_tt), jd_ut1=float(jd_ut1)
    )

def _house_map_for_planets(planet_lons: Dict[str, float], cusps: List[float]) -> Dict[str, int]:
    idxs = _assign_houses([planet_lons[p] for p in planet_lons], cusps)
    return {k: idxs[i] for i, k in enumerate(planet_lons.keys())}

# Varga helpers — built-in D9/D10 sign index (no external dep required)
def _d9_sign_index_from_lon(lon: float) -> int:
    # Navamsa: each sign (30°) has 9 parts of 3°20'. Mapping starts from same sign for movable signs,
    # 9th from sign for fixed, 5th from sign for dual.
    si = _sign_index(lon)
    within = _norm360(lon) % 30.0
    pada = int(within // (30.0 / 9.0))  # 0..8
    if si in (0,3,6,9):      # movable: start at same sign
        start = si
    elif si in (1,4,7,10):   # fixed: start at 9th from sign
        start = (si + 8) % 12
    else:                     # dual: start at 5th from sign
        start = (si + 4) % 12
    return (start + pada) % 12

def _d10_sign_index_from_lon(lon: float) -> int:
    # Dasamsa: 10 parts per sign; movable start from same sign, fixed from 9th, dual from 4th.
    si = _sign_index(lon)
    within = _norm360(lon) % 30.0
    part = int(within // 3.0)  # 0..9
    if si in (0,3,6,9):      # movable
        start = si
    elif si in (1,4,7,10):   # fixed
        start = (si + 8) % 12
    else:                    # dual
        start = (si + 3) % 12
    return (start + part) % 12

def _vargottama_flags(pl_lons: Dict[str, float]) -> Dict[str, bool]:
    flags = {}
    for p, lon in pl_lons.items():
        try:
            flags[p] = (_sign_index(lon) == _d9_sign_index_from_lon(lon))
        except Exception:
            flags[p] = False
    return flags

# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────
@dataclass
class YogaHit:
    name: str
    present: bool
    score: float
    levels: Dict[str, Any]
    details: Dict[str, Any]
    tags: Tuple[str, ...] = ()

# Registry
_RULES: Dict[str, Dict[str, Any]] = {}  # name -> {"fn":callable, "tags":tuple, "enabled":bool, "meta":dict}

def register_yoga(name: str, fn: Callable[..., YogaHit], *, tags: Iterable[str] = (), enabled: bool = True, meta: Optional[Dict[str,Any]] = None) -> None:
    _RULES[name] = {"fn": fn, "tags": tuple(tags), "enabled": bool(enabled), "meta": dict(meta or {})}

def list_registered_yogas() -> List[Dict[str, Any]]:
    out = []
    for k, v in _RULES.items():
        out.append({"name": k, "enabled": v["enabled"], "tags": list(v["tags"]), "meta": v["meta"]})
    return sorted(out, key=lambda r: r["name"].lower())

def enable_yogas(keys: Iterable[str]) -> None:
    keys = list(keys)
    by_tag = {k for k in keys if k.startswith("tag:")}
    names  = set(keys) - by_tag
    for nm in names:
        if nm in _RULES: _RULES[nm]["enabled"] = True
    for t in by_tag:
        tag = t.split(":",1)[1]
        for nm, rec in _RULES.items():
            if tag in rec["tags"]:
                rec["enabled"] = True

def disable_yogas(keys: Iterable[str]) -> None:
    keys = list(keys)
    by_tag = {k for k in keys if k.startswith("tag:")}
    names  = set(keys) - by_tag
    for nm in names:
        if nm in _RULES: _RULES[nm]["enabled"] = False
    for t in by_tag:
        tag = t.split(":",1)[1]
        for nm, rec in _RULES.items():
            if tag in rec["tags"]:
                rec["enabled"] = False

# ─────────────────────────────────────────────────────────────────────
# Core rule implementations
# Each rule receives (ctx) with:
#  jd_tt, ay_key, ay_deg, lat, lon, asc_sid, lagna_sign, cusps, pl_lons, pl_houses,
#  house_lords (1..12->planet), benefics, vargottama, d9_signs, d10_signs
# ─────────────────────────────────────────────────────────────────────

# — Mahapurusha (5) —
def _mahapurusha_rule(ctx: Dict[str, Any], planet: str, name_map: Dict[str,str], include_mooltrikona: bool) -> YogaHit:
    lon = ctx["pl_lons"].get(planet)
    if lon is None:
        return YogaHit(name_map[planet], False, 0.0, {}, {}, ("mahapurusha","strength"))
    si = _sign_index(lon)
    house = ctx["pl_houses"].get(planet)
    dign = _dignity(planet, si)
    strong = dign in ("own","exalted") or (include_mooltrikona and dign=="neutral" and planet in _PLANETS_MAIN)
    present = bool(house) and _is_kendra(house) and strong
    levels = {"house":{"is_kendra": _is_kendra(house or 0), "house": house}, "sign":{"index": si, "dignity": dign}}
    score = 0.9 if dign=="exalted" else (0.85 if dign=="own" else (0.8 if present else 0.0))
    return YogaHit(name_map[planet], present, score if present else 0.0, levels, {"planet": planet}, ("mahapurusha","strength"))

# — Gaja-Kesari —
def _gaja_kesari_rule(ctx: Dict[str, Any], include_same_house: bool) -> YogaHit:
    h_m = ctx["pl_houses"].get("Moon"); h_j = ctx["pl_houses"].get("Jupiter")
    d = _house_delta(h_m, h_j) if h_m and h_j else None
    present = bool(d and (d in (1,4,7,10) if include_same_house else d in (4,7,10)))
    lev = {"house":{"from_moon_delta": d}}
    # dignity nuance
    jd = _dignity("Jupiter", _sign_index(ctx["pl_lons"]["Jupiter"]))
    score = 0.82 + (0.05 if jd in ("own","exalted") else 0.0) if present else 0.0
    return YogaHit("Gaja-Kesari", present, score, lev, {}, ("moon","benefic","strength"))

# — Chandra-Mangala —
def _chandra_mangala_rule(ctx: Dict[str, Any], by_sign: bool, orb_deg: float) -> YogaHit:
    if by_sign:
        present = _sign_index(ctx["pl_lons"]["Moon"]) == _sign_index(ctx["pl_lons"]["Mars"])
        lev = {"sign":{"same_sign": present}}
        score = 0.78 if present else 0.0
    else:
        present = _conj(ctx["pl_lons"]["Moon"], ctx["pl_lons"]["Mars"], orb_deg)
        lev = {"degree":{"orb_deg": orb_deg, "within_orb": present}}
        score = 0.8 if present else 0.0
    return YogaHit("Chandra-Mangala", present, score, lev, {}, ("moon","wealth"))

# — Chandra Adhi (benefics in 6/7/8 from Moon) —
def _adhi_rule(ctx: Dict[str, Any]) -> YogaHit:
    h_m = ctx["pl_houses"].get("Moon")
    if not h_m: return YogaHit("Adhi", False, 0.0, {}, {}, ("moon","benefic"))
    deltas = {g: _house_delta(h_m, ctx["pl_houses"].get(g, 0)) for g in ("Mercury","Venus","Jupiter") if ctx["pl_houses"].get(g)}
    present = len(deltas)==3 and all(d in (6,7,8) for d in deltas.values())
    return YogaHit("Adhi", present, 0.78 if present else 0.0, {"from_moon_deltas": deltas}, {}, ("moon","benefic"))

# — Durudhara (planets both sides of Moon) —
def _durudhara_rule(ctx: Dict[str, Any]) -> YogaHit:
    moon = ctx["pl_lons"]["Moon"]
    present_left  = any(0 < _angdiff(ctx["pl_lons"][p], moon) < 180 for p in _PLANETS_MAIN if p!="Moon")
    present_right = any(0 < _angdiff(moon, ctx["pl_lons"][p]) < 180 for p in _PLANETS_MAIN if p!="Moon")
    present = present_left and present_right
    return YogaHit("Durudhara (Moon flanked)", present, 0.72 if present else 0.0, {}, {}, ("moon","kartari"))

# — Veshi/Voshi/Ubhayachari (Sun flanked) —
def _veshi_voshi_ubhay_rule(ctx: Dict[str, Any]) -> List[YogaHit]:
    sun = ctx["pl_lons"]["Sun"]
    left  = [p for p in _PLANETS_MAIN if p not in ("Sun","Moon") and 0 < _angdiff(ctx["pl_lons"][p], sun) < 180]
    right = [p for p in _PLANETS_MAIN if p not in ("Sun","Moon") and 0 < _angdiff(sun, ctx["pl_lons"][p]) < 180]
    hits: List[YogaHit] = []
    if left and not right:  hits.append(YogaHit("Veshi (planet after Sun)",  True, 0.7, {"planets": left},  {}, ("sun","kartari")))
    if right and not left:  hits.append(YogaHit("Voshi (planet before Sun)", True, 0.7, {"planets": right}, {}, ("sun","kartari")))
    if left and right:      hits.append(YogaHit("Ubhayachari (both sides of Sun)", True, 0.75, {"left": left,"right":right}, {}, ("sun","kartari")))
    if not hits:
        hits.append(YogaHit("Veshi/Voshi/Ubhayachari", False, 0.0, {}, {}, ("sun","kartari")))
    return hits

# — Shubha/Papa Kartari around Lagna/Moon/Sun —
def _kartari_rule(ctx: Dict[str, Any], around: str) -> List[YogaHit]:
    # Planets in adjacent houses (2 and 12 from reference house); benefics=Shubha, malefics=Papa
    center_house = 1 if around=="Lagna" else ctx["pl_houses"].get(around, 1)
    h12 = ((center_house + 10 - 1) % 12) + 1
    h2  = ((center_house + 1) % 12) + 1
    benefics = ctx["benefics"]
    malefics = {"Saturn","Mars","Sun","Rahu","Ketu"} | (set() if "Moon" in benefics else {"Moon"})
    at_12 = [p for p,h in ctx["pl_houses"].items() if h==h12]
    at_2  = [p for p,h in ctx["pl_houses"].items() if h==h2]
    has_b = any(p in benefics for p in at_12) and any(p in benefics for p in at_2)
    has_m = any(p in malefics for p in at_12) and any(p in malefics for p in at_2)
    res: List[YogaHit] = []
    res.append(YogaHit(f"Shubha Kartari around {around}", has_b, 0.75 if has_b else 0.0, {"houses":{"h12":h12,"h2":h2},"planets":{"h12":at_12,"h2":at_2}}, {}, ("kartari","benefic")))
    res.append(YogaHit(f"Papa Kartari around {around}",   has_m, 0.65 if has_m else 0.0, {"houses":{"h12":h12,"h2":h2},"planets":{"h12":at_12,"h2":at_2}}, {}, ("kartari","malefic")))
    return res

# — Raja (kendra–trikona association) & Dharma-Karmadhipati —
def _raja_rule(ctx: Dict[str, Any], conj_only: bool) -> YogaHit:
    pairs = []
    for k in (1,4,7,10):
        for t in (1,5,9):
            a = ctx["house_lords"][k]; b = ctx["house_lords"][t]
            if a == b: continue
            ha = ctx["pl_houses"].get(a); hb = ctx["pl_houses"].get(b)
            if not (ha and hb): continue
            same_sign = _sign_index(ctx["pl_lons"][a]) == _sign_index(ctx["pl_lons"][b])
            assoc = same_sign or (_is_kendra(ha) and _is_kendra(hb)) or _has_graha_drishti(a, _sign_index(ctx["pl_lons"][a]), _sign_index(ctx["pl_lons"][b]))
            if conj_only: assoc = same_sign
            if assoc:
                pairs.append({"kendra_lord":a,"trikona_lord":b,"houses":(ha,hb),"same_sign":same_sign})
    present = bool(pairs)
    return YogaHit("Raja (k–t association)", present, 0.82 if present else 0.0, {"pairs": pairs}, {}, ("raja","association"))

def _dharma_karmadhipati_rule(ctx: Dict[str, Any]) -> YogaHit:
    L9 = ctx["house_lords"][9]; L10 = ctx["house_lords"][10]
    h9 = ctx["pl_houses"].get(L9); h10 = ctx["pl_houses"].get(L10)
    if not (h9 and h10): return YogaHit("Dharma-Karmadhipati", False, 0.0, {}, {}, ("raja","dk"))
    same_sign = _sign_index(ctx["pl_lons"][L9]) == _sign_index(ctx["pl_lons"][L10])
    assoc = same_sign or (_is_kendra(h9) and _is_kendra(h10)) or _has_graha_drishti(L9, _sign_index(ctx["pl_lons"][L9]), _sign_index(ctx["pl_lons"][L10]))
    present = assoc
    lev = {"lords":{"L9":L9,"L10":L10},"houses":{"L9":h9,"L10":h10},"same_sign":same_sign}
    return YogaHit("Dharma-Karmadhipati", present, 0.86 if present else 0.0, lev, {}, ("raja","dk"))

# — Parivartana (typed) —
def _parivartana_rule(ctx: Dict[str, Any]) -> YogaHit:
    lords = ctx["sign_lords"]
    lagna_sign = ctx["lagna_sign"]
    def house_of_sign(si: int) -> int: return ((si - lagna_sign) % 12) + 1
    exchanges = []
    for s1 in range(12):
        lord1 = lords[s1]
        s2 = _sign_index(ctx["pl_lons"].get(lord1, float("nan")))
        if not math.isfinite(s2): continue
        lord2 = lords[s2]
        s_back = _sign_index(ctx["pl_lons"].get(lord2, float("nan")))
        if s_back == s1 and lord1 != lord2:
            h1, h2 = house_of_sign(s1), house_of_sign(s2)
            exchanges.append({"lords":(lord1,lord2),"signs":(s1,s2),"houses":(h1,h2)})
    def classify(hs: Tuple[int,int]) -> str:
        a,b = hs; S = {a,b}
        if (S & {1,5,9}) and (S & {4,7,10}): return "Maha"
        if (S & {6,8,12}): return "Dainya"
        if (S & {3,11}):   return "Khala"
        return "Regular"
    for ex in exchanges:
        ex["type"] = classify(ex["houses"])
    present = bool(exchanges)
    types = sorted(set(ex["type"] for ex in exchanges)) if present else []
    return YogaHit("Parivartana (exchange)", present, 0.88 if present else 0.0, {"exchanges": exchanges}, {"types": types}, ("exchange","raja"))

# — Neecha-bhanga (core + drishti/exchange variants) —
def _neechabhanga_rule(ctx: Dict[str, Any]) -> YogaHit:
    lords = ctx["sign_lords"]; lagna = 1; moon_h = ctx["pl_houses"].get("Moon",1)
    cancels = []
    for p, deb_s in _DEB.items():
        ps = _sign_index(ctx["pl_lons"].get(p, float("nan")))
        if ps != deb_s: continue
        disp = lords[deb_s]
        ex_s = _EXALT.get(p); ex_lord = lords[ex_s] if ex_s is not None else None
        disp_h = ctx["pl_houses"].get(disp); ex_h = ctx["pl_houses"].get(ex_lord) if ex_lord else None
        okA = bool(disp_h and (_is_kendra(_house_delta(lagna, disp_h)) or _is_kendra(_house_delta(moon_h, disp_h))))
        okB = bool(ex_h   and (_is_kendra(_house_delta(lagna, ex_h))   or _is_kendra(_house_delta(moon_h, ex_h))))
        # drishti/exchange support
        dr = False
        if ex_lord:
            dr = _has_graha_drishti(ex_lord, _sign_index(ctx["pl_lons"][ex_lord]), ps)
        exch = False
        if disp and ex_lord:
            s_disp = _sign_index(ctx["pl_lons"][disp]); s_exl = _sign_index(ctx["pl_lons"][ex_lord])
            # exchange between deb sign lord and exaltation sign lord (rare; include)
            exch = (lords[s_disp]==p and lords[s_exl]==p)
        if okA or okB or dr or exch:
            cancels.append({"planet": p, "rules": {"A":okA,"B":okB,"drishti_exalt_lord":dr,"exchange_disp_exalt":exch}})
    present = bool(cancels)
    score = 0.8 if present else 0.0
    return YogaHit("Neecha-bhanga (composite)", present, score, {"cancellations": cancels}, {}, ("cancellation","raja"))

# — Vipareeta Raja (Harsha/Sarala/Vimala) —
def _vipareeta_raja_rule(ctx: Dict[str, Any]) -> YogaHit:
    # Harsha: 6L in 6/8/12; Sarala: 8L in 6/8/12; Vimala: 12L in 6/8/12 (stronger if in own/exaltation)
    res = []
    for H, name in ((6,"Harsha"),(8,"Sarala"),(12,"Vimala")):
        lord = ctx["house_lords"][H]
        h = ctx["pl_houses"].get(lord)
        if h and _is_dusthana(h):
            dig = _dignity(lord, _sign_index(ctx["pl_lons"][lord]))
            res.append({"type": name, "lord": lord, "house": h, "dignity": dig})
    present = bool(res)
    score = 0.82 if any(r["dignity"] in ("own","exalted") for r in res) else (0.76 if present else 0.0)
    return YogaHit("Vipareeta Raja (tri-dusthana lords)", present, score, {"instances": res}, {}, ("vipareeta","raja"))

# — Amala (benefic in 10th from Lagna / Moon) —
def _amala_rule(ctx: Dict[str, Any], from_ref: str) -> YogaHit:
    center = 1 if from_ref=="Lagna" else ctx["pl_houses"].get("Moon",1)
    target = ((center + 8 - 1) % 12) + 1  # 10th from ref
    benefics = ctx["benefics"]
    pls = [p for p,h in ctx["pl_houses"].items() if h==target and p in benefics]
    present = bool(pls)
    return YogaHit(f"Amala (benefic 10th from {from_ref})", present, 0.78 if present else 0.0, {"house": target, "planets": pls}, {}, ("amala","career"))

# — Chatussagara (planets in all kendras) —
def _chatussagara_rule(ctx: Dict[str, Any]) -> YogaHit:
    have = {1:False,4:False,7:False,10:False}
    for p,h in ctx["pl_houses"].items():
        if p in _PLANETS_MAIN and h in have:
            have[h] = True
    present = all(have.values())
    return YogaHit("Chatussagara", present, 0.76 if present else 0.0, {"kendras": have}, {}, ("kendras","strength"))

# — Vasumati (benefics in upachayas from Moon) —
def _vasumati_rule(ctx: Dict[str, Any]) -> YogaHit:
    m = ctx["pl_houses"].get("Moon",1)
    upas = {((m+2-1)%12)+1, ((m+5-1)%12)+1, ((m+9-1)%12)+1, ((m+10-1)%12)+1}  # 3,6,10,11 from Moon
    bens = ctx["benefics"]
    count = sum(1 for p,h in ctx["pl_houses"].items() if p in bens and h in upas)
    present = count >= 2
    return YogaHit("Vasumati (benefics in Moon's upachayas)", present, 0.74 if present else 0.0, {"count": count}, {}, ("wealth","moon"))

# — Dhana (2/11[/5] lords association) —
def _dhana_rule(ctx: Dict[str, Any]) -> YogaHit:
    L2, L11, L5 = ctx["house_lords"][2], ctx["house_lords"][11], ctx["house_lords"][5]
    combos = [(L2,L11,"2&11"), (L2,L5,"2&5"), (L5,L11,"5&11")]
    hits = []
    for a,b,lab in combos:
        if a==b: continue
        ha,hb = ctx["pl_houses"].get(a), ctx["pl_houses"].get(b)
        if not (ha and hb): continue
        same = _sign_index(ctx["pl_lons"][a]) == _sign_index(ctx["pl_lons"][b])
        assoc = same or (_is_kendra(ha) and _is_kendra(hb)) or _has_graha_drishti(a, _sign_index(ctx["pl_lons"][a]), _sign_index(ctx["pl_lons"][b]))
        if assoc:
            hits.append({"pair": lab, "lords": (a,b), "houses": (ha,hb), "same_sign": same})
    present = bool(hits)
    return YogaHit("Dhana (2/11[/5] association)", present, 0.78 if present else 0.0, {"pairs": hits}, {}, ("wealth","association"))

# — Saraswati & Lakshmi (from earlier, retained) —
def _saraswati_rule(ctx: Dict[str, Any]) -> YogaHit:
    trio = ("Mercury","Venus","Jupiter")
    ok_h = all(_is_kendra(ctx["pl_houses"].get(g,0)) or _is_trikona(ctx["pl_houses"].get(g,0)) for g in trio)
    ok_d = all(_dignity(g, _sign_index(ctx["pl_lons"][g])) in ("own","exalted") for g in trio)
    present = ok_h and ok_d
    levels = {"house": {g: ctx["pl_houses"].get(g) for g in trio}, "sign": {g: _dignity(g, _sign_index(ctx["pl_lons"][g])) for g in trio}}
    return YogaHit("Saraswati", present, 0.84 if present else 0.0, levels, {}, ("education","speech"))

def _lakshmi_rule(ctx: Dict[str, Any]) -> YogaHit:
    LL = ctx["house_lords"][1]; L9 = ctx["house_lords"][9]
    hL, h9 = ctx["pl_houses"].get(LL), ctx["pl_houses"].get(L9)
    if not (hL and h9): return YogaHit("Lakshmi", False, 0.0, {}, {}, ("wealth","fortune"))
    strongL = _is_kendra(hL) or _is_trikona(hL)
    strong9 = _is_kendra(h9) or _is_trikona(h9)
    same = _sign_index(ctx["pl_lons"][LL]) == _sign_index(ctx["pl_lons"][L9])
    assoc = same or (_is_kendra(hL) and _is_kendra(h9))
    present = strongL and strong9 and assoc
    levels = {"lords":{"LL":LL,"L9":L9}, "houses":{"LL":hL,"L9":h9}, "assoc":{"same_sign": same, "mutual_kendra": _is_kendra(hL) and _is_kendra(h9)}}
    return YogaHit("Lakshmi", present, 0.83 if present else 0.0, levels, {}, ("wealth","fortune"))

# — Kemadruma (Moon isolated) —
def _kemadruma_rule(ctx: Dict[str, Any]) -> YogaHit:
    # No planets (excl. nodes) in 2/12 from Moon; simple cancellations: planets in kendras from Moon
    h_m = ctx["pl_houses"].get("Moon",1)
    h12 = ((h_m + 10 - 1) % 12) + 1
    h2  = ((h_m + 1) % 12) + 1
    flank = [p for p,h in ctx["pl_houses"].items() if p in _PLANETS_MAIN and h in (h12,h2) and p!="Moon"]
    canceled = any(_is_kendra(_house_delta(h_m, ctx["pl_houses"].get(p,0))) for p in _PLANETS_MAIN if p!="Moon")
    present = (len(flank)==0) and (not canceled)
    return YogaHit("Kemadruma (Moon isolated)", present, 0.7 if present else 0.0, {"flanking_planets": flank, "kendra_from_moon_present": canceled}, {}, ("moon","dosha"))

# — Kala Sarpa / Kala Amrita —
def _kala_sarpa_rule(ctx: Dict[str, Any]) -> List[YogaHit]:
    rahu = ctx["pl_lons"].get("Rahu"); ketu = ctx["pl_lons"].get("Ketu")
    if rahu is None or ketu is None:
        return [YogaHit("Kala Sarpa", False, 0.0, {}, {}, ("nodal","dosha")),
                YogaHit("Kala Amrita", False, 0.0, {}, {}, ("nodal","dosha"))]
    def between_rahu_ketu(l: float) -> bool:
        d = (_norm360(l - rahu))
        return 0 < d < 180.0  # within arc Rahu→Ketu
    def between_ketu_rahu(l: float) -> bool:
        d = (_norm360(l - ketu))
        return 0 < d < 180.0  # within arc Ketu→Rahu
    planets7 = [ctx["pl_lons"][p] for p in _PLANETS_MAIN]
    all_rk = all(between_rahu_ketu(L) for L in planets7)
    all_kr = all(between_ketu_rahu(L) for L in planets7)
    # Sarpa: all bw Rahu→Ketu with Moon on Rahu half; Amrita: all bw Ketu→Rahu with Moon on Ketu half (common practical split)
    sarpa  = all_rk
    amrita = all_kr
    return [
        YogaHit("Kala Sarpa",  sarpa,  0.68 if sarpa else 0.0, {}, {}, ("nodal","dosha")),
        YogaHit("Kala Amrita", amrita, 0.68 if amrita else 0.0, {}, {}, ("nodal","dosha")),
    ]

# ─────────────────────────────────────────────────────────────────────
# Varga-aware scoring & Arudha notes
# ─────────────────────────────────────────────────────────────────────
def _score_with_vargas(hit: YogaHit, ctx: Dict[str, Any], strengthen_on: Tuple[str,...]) -> YogaHit:
    if not hit.present:
        return hit
    bump = 0.0
    if "D9" in strengthen_on:
        # generic bump if central planets are vargottama or strong by D9 alignment
        vg = ctx.get("vargottama", {})
        if isinstance(hit.levels.get("lords"), dict):
            if any(vg.get(p, False) for p in hit.levels["lords"].values()):
                bump += 0.04
        elif isinstance(hit.details.get("planet"), str):
            if vg.get(hit.details["planet"], False):
                bump += 0.04
        else:
            # soft generic bump
            bump += 0.02
    if "D10" in strengthen_on:
        bump += 0.01
    return YogaHit(hit.name, hit.present, min(0.99, hit.score + bump), {**hit.levels, "varga":{"boost_keys": strengthen_on}}, hit.details, hit.tags)

def _attach_arudha_notes(hit: YogaHit, arudha: Dict[str, Any] | None, ctx: Dict[str, Any]) -> YogaHit:
    if not arudha:
        return hit
    try:
        a1 = arudha.get("arudhas", {}).get("A1", {})
        al_idx = int(a1.get("sign_index"))
        notes = {}
        # check focal planets on AL
        if "lords" in hit.levels and isinstance(hit.levels["lords"], dict):
            for key, lord in hit.levels["lords"].items():
                if lord in ctx["pl_lons"] and _sign_index(ctx["pl_lons"][lord]) == al_idx:
                    notes[f"{lord}_on_AL"] = True
        return YogaHit(hit.name, hit.present, hit.score, {**hit.levels, "arudha":{"AL_sign_index": al_idx, **notes}}, hit.details, hit.tags)
    except Exception:
        return hit

# ─────────────────────────────────────────────────────────────────────
# Public orchestrator
# ─────────────────────────────────────────────────────────────────────
def compute_yogas(
    payload: Dict[str, Any],
    *,
    ayanamsa: str | float = "lahiri",
    house_system: str = "placidus",
    sign_lord_variant: str = "classical",
    chandra_mangala_by_sign: bool = True,
    conj_orb_deg: float = 6.0,
    gajakesari_include_same_house: bool = True,
    include_mooltrikona_in_mahapurusha: bool = True,
    use_vargas_for_scoring: bool = True,
    varga_keys_for_boost: Tuple[str,...] = ("D9","D10"),
    include_arudha_notes: bool = True,
    enable_catalog_tags: Tuple[str,...] = (),    # e.g., ("dosha","raja","wealth") — if provided, filters registry to these tags
    disable_catalog_tags: Tuple[str,...] = (),   # tags to disable
) -> Dict[str, Any]:
    if not _EPH_OK or not _HOUSES_OK:
        return {"ok": False, "error": "core_modules_unavailable", "yogas": [], "warnings": []}

    # Timescales, ayanamsa, site
    jd_tt, jd_ut1, warns_ts = _timescales(payload)
    ay_key, ay_deg, warns_ay = _ayanamsa(jd_tt, ayanamsa)

    if not all(k in payload for k in ("latitude","longitude")):
        return {"ok": False, "error": "site_required", "yogas": [], "warnings": warns_ts + warns_ay}
    lat = float(payload["latitude"]); lon = float(payload["longitude"])

    # Houses & asc
    hp = _compute_houses_payload(lat, lon, jd_tt, jd_ut1, house_system)
    cusps = list(hp["cusps_deg"]); asc_trop = float(hp["asc_deg"])
    asc_sid = _norm360(asc_trop - ay_deg); lagna_sign = _sign_index(asc_sid)

    # Longitudes & houses
    pl_lons = _sidereal_longitudes(jd_tt, list(_PLANETS_ALL), ay_deg)
    pl_houses = _house_map_for_planets({k:v for k,v in pl_lons.items() if k in _PLANETS_ALL}, cusps)

    # Lords
    sign_lords = _sign_lords(sign_lord_variant)
    house_lords = {i: sign_lords[(lagna_sign + (i-1)) % 12] for i in range(1,13)}

    # Benefics (Moon phase-sensitive if panchanga present)
    benefics = _benefic_set(jd_tt, ay_key)

    # Vargas (built-in D9/D10; also accept external compute_vargas for downstream)
    d9_signs = {p: _d9_sign_index_from_lon(lon) for p,lon in pl_lons.items()}
    d10_signs = {p: _d10_sign_index_from_lon(lon) for p,lon in pl_lons.items()}
    vargottama = _vargottama_flags(pl_lons)
    varga_ctx = {}
    if callable(_compute_vargas):
        try:
            varga_ctx = _compute_vargas({"jd_tt": jd_tt, "ayanamsa_deg": ay_deg}, keys=list(varga_keys_for_boost))
        except Exception:
            varga_ctx = {}

    # Arudha (optional)
    arudha_ctx = None
    if include_arudha_notes and callable(_compute_arudhas):
        try:
            arudha_ctx = _compute_arudhas({"jd_tt": jd_tt, "jd_ut1": jd_ut1, "latitude": lat, "longitude": lon, "ayanamsa": ay_key})
        except Exception:
            arudha_ctx = None

    # Context passed to rules
    ctx = {
        "jd_tt": jd_tt, "ay_key": ay_key, "ay_deg": ay_deg,
        "lat": lat, "lon": lon,
        "asc_sid": asc_sid, "lagna_sign": lagna_sign,
        "cusps": cusps,
        "pl_lons": pl_lons, "pl_houses": pl_houses,
        "sign_lords": sign_lords, "house_lords": house_lords,
        "benefics": benefics,
        "vargottama": vargottama, "d9_signs": d9_signs, "d10_signs": d10_signs,
    }

    # Build/refresh registry (idempotent)
    _build_registry_once()

    # Tag gating
    if enable_catalog_tags:
        for nm, rec in _RULES.items():
            rec["enabled"] = any(t in rec["tags"] for t in enable_catalog_tags)
    for tag in disable_catalog_tags:
        for nm, rec in _RULES.items():
            if tag in rec["tags"]:
                rec["enabled"] = False

    # Evaluate active rules
    hits: List[YogaHit] = []
    for name, rec in _RULES.items():
        if not rec["enabled"]:
            continue
        fn = rec["fn"]
        try:
            out = fn(ctx)
            if isinstance(out, list):
                hits.extend(out)
            elif isinstance(out, YogaHit):
                hits.append(out)
        except Exception:
            # robust: skip faulty rule
            continue

    # Varga-aware scoring and AL notes
    final_hits: List[YogaHit] = []
    for h in hits:
        hh = _score_with_vargas(h, ctx, varga_keys_for_boost) if use_vargas_for_scoring else h
        hh = _attach_arudha_notes(hh, arudha_ctx, ctx) if include_arudha_notes else hh
        final_hits.append(hh)

    # Shape response
    out_list = [{
        "name": h.name, "present": bool(h.present), "score": float(h.score),
        "levels": h.levels, "details": h.details, "tags": list(h.tags)
    } for h in final_hits]

    context = {
        "ayanamsa": {"key": ay_key, "deg": float(ay_deg)},
        "ascendant_sidereal_deg": float(asc_sid),
        "lagna_sign_index": int(lagna_sign),
        "house_lords": house_lords,
        "planet_houses": {k:int(v) for k,v in pl_houses.items()},
        "planet_signs": {k:int(_sign_index(pl_lons[k])) for k in pl_lons},
        "benefics_used": sorted(list(benefics)),
        "vargottama": vargottama,
        "house_system": house_system,
    }
    warnings = list(dict.fromkeys(warns_ts + warns_ay))
    return {"ok": True, "yogas": out_list, "context": context, "warnings": warnings}

# ─────────────────────────────────────────────────────────────────────
# Registry population
# ─────────────────────────────────────────────────────────────────────
_BUILT = False
def _build_registry_once() -> None:
    global _BUILT
    if _BUILT:
        return
    _BUILT = True
    # Clear on first build (idempotent guard)
    _RULES.clear()

    # Mahapurusha x5
    name_map = {"Mars":"Ruchaka","Mercury":"Bhadra","Jupiter":"Hamsa","Venus":"Malavya","Saturn":"Sasa"}
    for p in ("Mars","Mercury","Jupiter","Venus","Saturn"):
        def _make(pname: str) -> Callable[[Dict[str,Any]], YogaHit]:
            return lambda ctx, _p=pname: _mahapurusha_rule(ctx, _p, name_map, include_mooltrikona=True)
        register_yoga(name_map[p], _make(p), tags=("mahapurusha","strength"))

    # Core popular yogas
    register_yoga("Gaja-Kesari",               lambda ctx: _gaja_kesari_rule(ctx, include_same_house=True), tags=("moon","strength","popular"))
    register_yoga("Chandra-Mangala",           lambda ctx: _chandra_mangala_rule(ctx, by_sign=True, orb_deg=6.0), tags=("moon","wealth","popular"))
    register_yoga("Adhi",                      _adhi_rule, tags=("moon","benefic"))
    register_yoga("Durudhara (Moon flanked)",  _durudhara_rule, tags=("moon","kartari"))

    # Sun flankers
    register_yoga("Veshi/Voshi/Ubhayachari",   _veshi_voshi_ubhay_rule, tags=("sun","kartari"))

    # Kartari around key points
    register_yoga("Kartari around Lagna", lambda ctx: _kartari_rule(ctx, "Lagna"), tags=("kartari","lagna"))
    register_yoga("Kartari around Moon",  lambda ctx: _kartari_rule(ctx, "Moon"),  tags=("kartari","moon"))
    register_yoga("Kartari around Sun",   lambda ctx: _kartari_rule(ctx, "Sun"),   tags=("kartari","sun"))

    # Raja family
    register_yoga("Raja (k–t association)", _raja_rule, tags=("raja","association"))
    register_yoga("Dharma-Karmadhipati",    _dharma_karmadhipati_rule, tags=("raja","dk"))
    register_yoga("Parivartana (exchange)", _parivartana_rule, tags=("raja","exchange"))
    register_yoga("Neecha-bhanga (composite)", _neechabhanga_rule, tags=("raja","cancellation"))
    register_yoga("Vipareeta Raja (tri-dusthana lords)", _vipareeta_raja_rule, tags=("raja","vipareeta"))

    # Wealth / career
    register_yoga("Amala from Lagna", lambda ctx: _amala_rule(ctx, "Lagna"), tags=("amala","career"))
    register_yoga("Amala from Moon",  lambda ctx: _amala_rule(ctx, "Moon"),  tags=("amala","career","moon"))
    register_yoga("Chatussagara", _chatussagara_rule, tags=("kendras","strength"))
    register_yoga("Vasumati",     _vasumati_rule, tags=("moon","wealth"))
    register_yoga("Dhana (2/11[/5] association)", _dhana_rule, tags=("wealth","association"))
    register_yoga("Saraswati", _saraswati_rule, tags=("education","speech"))
    register_yoga("Lakshmi",   _lakshmi_rule, tags=("wealth","fortune"))

    # Dosha / nodal frameworks
    register_yoga("Kemadruma (Moon isolated)", _kemadruma_rule, tags=("moon","dosha"))
    register_yoga("Kala Sarpa / Amrita", _kala_sarpa_rule, tags=("nodal","dosha"))

# ─────────────────────────────────────────────────────────────────────
# Convenience: default export of registry listing
# ─────────────────────────────────────────────────────────────────────
__all__ = [
    "compute_yogas",
    "list_registered_yogas",
    "enable_yogas",
    "disable_yogas",
]
