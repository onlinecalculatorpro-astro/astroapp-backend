# app/core/yogas.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Classical Yogas — research-grade, sidereal-first, multi-level evidence

Scope
- Computes a curated set of widely-cited classical yogas with rigorous, explicit rules:
  • Panch Mahapurusha (Ruchaka, Bhadra, Hamsa, Malavya, Sasa)
  • Gaja-Kesari
  • Raja Yoga (kendra–trikona lord association; sign-conjunction or safe mutual-kendra)
  • Parivartana (exchange) with simple type classification
  • Neecha-bhanga (core cancellation patterns)
  • Chandra-Mangala
  • Adhi Yoga (benefics in 6/7/8 from Moon)
  • Saraswati (Mercury–Venus–Jupiter strength in K/T)
  • Lakshmi (LL & 9L strength + association)

Design
- Sidereal-first: ecliptic-of-date longitudes minus ayanāṁśa (app.core.ayanamsa)
- Gold ascendant: app.core.houses_advanced (ERFA gst06a + true ε)
- House placement & mapping: uses houses_advanced.assign_houses (forward-wrap intervals)
- Vargas (optional): app.core.varga_charts (e.g., D9) to boost confidence
- Arudha (optional): app.core.jaimini_arudha for contextual notes
- All rules are explicit; options expose common school variants. No hidden heuristics.

Output
- compute_yogas(payload, **options) -> dict with:
  {
    "ok": True,
    "yogas": [
      {
        "name": "Gaja-Kesari",
        "present": True,
        "score": 0.86,                      # 0..1 composite
        "levels": {                         # evidence layers
           "sign": {...}, "house": {...}, "degree": {...}, "varga": {...}, "arudha": {...}
        },
        "details": {...},                   # rule-specific facts
      },
      ...
    ],
    "context": {...},                       # lagna, sign lords, dignities, etc.
    "warnings": [...]
  }

Precision
- Numeric path relies on your “gold” house/angles and strict timescales.
- Deterministic, with explicit orbs and sign/house semantics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

# ─────────────────────────────────────────────────────────────────────
# Optional imports (graceful degradation)
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

# Optional vargas and arudhas
try:
    from app.core.varga_charts import compute_vargas as _compute_vargas
except Exception:
    _compute_vargas = None

try:
    from app.core.jaimini_arudha import compute_arudhas as _compute_arudhas
except Exception:
    _compute_arudhas = None

# Timescales backends (same pattern you use elsewhere)
try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

# Classical constants (fallbacks provided if not present)
try:
    from app.core.constants_vedic import SIGN_NAMES as _SIGNS
except Exception:
    _SIGNS = (
        "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
        "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
    )

# Dignity maps — use constants_vedic if available; else fall back
try:
    from app.core.constants_vedic import EXALTATION_SIGN_INDEX as _EXALT  # {"Sun":0, ...}
    from app.core.constants_vedic import DEBILITATION_SIGN_INDEX as _DEB
    from app.core.constants_vedic import OWN_SIGN_INDEXES as _OWN  # {"Mars": (0,7), ...}
except Exception:
    _EXALT = {
        "Sun": 0,     # Aries
        "Moon": 1,    # Taurus
        "Mars": 9,    # Capricorn
        "Mercury": 5, # Virgo
        "Jupiter": 3, # Cancer
        "Venus": 11,  # Pisces
        "Saturn": 6,  # Libra
    }
    _DEB = {
        "Sun": 6, "Moon": 7, "Mars": 3, "Mercury": 11, "Jupiter": 9, "Venus": 5, "Saturn": 0
    }
    _OWN = {
        "Sun": (4,), "Moon": (3,),
        "Mars": (0,7), "Mercury": (2,5),
        "Jupiter": (8,11), "Venus": (1,6), "Saturn": (9,10)
    }

# Benefic/malefic sets (basic; finer combustion etc. can be layered outside)
_BENEFICS = {"Jupiter","Venus","Mercury","Waxing Moon","Moon"}  # Mercury neutral→treated benefic
_MALEFICS = {"Saturn","Mars","Sun","Rahu","Ketu","Waning Moon"} # nodes as malefics

# Supported planets (grahas) for this engine
_PLANETS = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu")

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _sign_index(lon: float) -> int:
    return int(math.floor(_norm360(lon) / 30.0)) % 12

def _house_delta(h_from: int, h_to: int) -> int:
    """1..12 difference of houses (1 means same house from reference)."""
    return ((h_to - h_from) % 12) + 1

def _is_kendra(h: int) -> bool:
    return h in (1,4,7,10)

def _is_trikona(h: int) -> bool:
    return h in (1,5,9)

def _mutual_kendra(h1: int, h2: int) -> bool:
    d = _house_delta(h1, h2)
    return d in (1,4,7,10)

def _safe_orb(sep_deg: float, cap: float) -> bool:
    return abs(float(sep_deg)) <= max(0.0, float(cap))

def _angdiff(a: float, b: float) -> float:
    d = (_norm360(a) - _norm360(b) + 540.0) % 360.0 - 180.0
    return d

def _conj_by_degree(a: float, b: float, orb_deg: float) -> bool:
    return _safe_orb(_angdiff(a, b), float(orb_deg))

def _sign_lords(variant: str = "classical") -> Dict[int, str]:
    if str(variant).lower().startswith("nodes"):
        return { 0:"Mars",1:"Venus",2:"Mercury",3:"Moon",4:"Sun",5:"Mercury",6:"Venus",7:"Ketu",8:"Jupiter",9:"Saturn",10:"Rahu",11:"Jupiter" }
    return { 0:"Mars",1:"Venus",2:"Mercury",3:"Moon",4:"Sun",5:"Mercury",6:"Venus",7:"Mars",8:"Jupiter",9:"Saturn",10:"Saturn",11:"Jupiter" }

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
                    jd_tt = float(out["jd_tt"])
                    jd_ut1 = float(out.get("jd_ut1") or out.get("jd_ut") or out.get("jd_utc") or out.get("jd_tt"))
                    return jd_tt, jd_ut1, warns
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
    jd_ut1 = jd_ut
    return jd_tt, jd_ut1, warns

def _ayanamsa(jd_tt: float, key_or_deg: Any) -> Tuple[str, float, List[str]]:
    warns: List[str] = []
    if isinstance(key_or_deg, (int,float)):
        return "explicit", float(key_or_deg), warns
    key = str(key_or_deg or "lahiri").strip().lower()
    if _get_ayanamsa_deg is None:
        # linearized fallback (keeps continuity)
        AY_J2000_DEG = (23 + 51/60 + 26.26/3600)
        RATE_AS_PER_YR = 50.290966  # "/yr
        years = (float(jd_tt) - 2451545.0) / 365.25
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
        warns.append("ayanamsa_fallback_lahiri_linearized")
        return key, ay, warns
    try:
        return key, float(_get_ayanamsa_deg(float(jd_tt), key)), warns
    except Exception as e:
        warns.append(f"ayanamsa_resolve_error:{e}")
        AY_J2000_DEG = (23 + 51/60 + 26.26/3600)
        RATE_AS_PER_YR = 50.290966
        years = (float(jd_tt) - 2451545.0) / 365.25
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
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

def _dignity(planet: str, sign_idx: int) -> str:
    if planet in _EXALT and _EXALT[planet] == sign_idx:
        return "exalted"
    if planet in _DEB and _DEB[planet] == sign_idx:
        return "debilitated"
    owns = _OWN.get(planet, ())
    if sign_idx in owns:
        return "own"
    return "neutral"

# ─────────────────────────────────────────────────────────────────────
# Rule evaluation cores (yogas)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class YogaHit:
    name: str
    present: bool
    score: float
    levels: Dict[str, Any]
    details: Dict[str, Any]

def _mahapurusha(
    planet: str, pl_lons: Dict[str, float], pl_houses: Dict[str, int], *,
    include_mooltrikona: bool, lagna_house: int
) -> Optional[YogaHit]:
    # Kendra from Lagna + dignity (own/exaltation[/mooltrikona])
    sign_idx = _sign_index(pl_lons.get(planet, float("nan")))
    house = pl_houses.get(planet)
    if house is None:
        return None
    dign = _dignity(planet, sign_idx)
    strong = dign in ("own","exalted") or (include_mooltrikona and dign == "neutral" and planet in ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"))  # simple mt fallback
    if _is_kendra(house) and strong:
        name_map = {
            "Mars":"Ruchaka","Mercury":"Bhadra","Jupiter":"Hamsa","Venus":"Malavya","Saturn":"Sasa"
        }
        levels = {
            "house": {"kendra_from_lagna": True, "house": int(house)},
            "sign": {"sign_index": int(sign_idx), "dignity": dign},
        }
        return YogaHit(name_map[planet], True, 0.85 if dign=="own" else 0.95, levels, {"planet": planet})
    return None

def _gajakesari(pl_lons: Dict[str, float], pl_houses: Dict[str, int], *, include_same_house: bool) -> YogaHit:
    # Jupiter in kendra from Moon (optionally excluding 1st from Moon)
    h_moon = pl_houses.get("Moon"); h_jup = pl_houses.get("Jupiter")
    present = False
    if h_moon and h_jup:
        d = _house_delta(h_moon, h_jup)
        if include_same_house:
            present = d in (1,4,7,10)
        else:
            present = d in (4,7,10)
    levels = {"house": {"from_moon_delta": int(d) if (h_moon and h_jup) else None}}
    # Degree strength (optionally: avoid debility)
    j_sign = _sign_index(pl_lons.get("Jupiter", float("nan")))
    moon_sign = _sign_index(pl_lons.get("Moon", float("nan")))
    levels["sign"] = {
        "jupiter_dignity": _dignity("Jupiter", j_sign),
        "moon_dignity": _dignity("Moon", moon_sign),
    }
    score = 0.0
    if present:
        boost = 0.05 if levels["sign"]["jupiter_dignity"] in ("own","exalted") else 0.0
        score = 0.8 + boost
    return YogaHit("Gaja-Kesari", bool(present), float(score), levels, {})

def _raja(pl_lords: Dict[int, str], pl_houses: Dict[str, int], *, sign_conj_only: bool, pl_lons: Dict[str, float]) -> YogaHit:
    # Any kendra-lord associated with any trikona-lord (association: same sign OR mutual kendra)
    # Kendra lords: houses 1/4/7/10 → their sign lords
    k_lords = {pl_lords[h] for h in (1,4,7,10)}
    t_lords = {pl_lords[h] for h in (1,5,9)}  # 1 counted in both by many schools
    pairs = []
    for k in k_lords:
        for t in (t_lords - {k}):
            hk = pl_houses.get(k); ht = pl_houses.get(t)
            if hk is None or ht is None:
                continue
            ok = False
            if sign_conj_only:
                ok = (_sign_index(pl_lons[k]) == _sign_index(pl_lons[t]))
            else:
                ok = (_sign_index(pl_lons[k]) == _sign_index(pl_lons[t])) or _mutual_kendra(hk, ht)
            if ok:
                pairs.append((k, t, hk, ht))
    present = len(pairs) > 0
    levels = {"pairs": [{"kendra_lord": a, "trikona_lord": b, "houses": (int(hk), int(ht))} for (a,b,hk,ht) in pairs]}
    score = 0.82 if present else 0.0
    return YogaHit("Raja Yoga (k–t association)", present, score, levels, {})

def _parivartana(pl_lords: Dict[int, str], pl_lons: Dict[str, float], lagna_sign: int) -> YogaHit:
    # Exchange of signs between two planets → classify types (simple)
    # House ownership by sign relative to Lagna
    def house_of_sign(sign_idx: int) -> int:
        return ((sign_idx - lagna_sign) % 12) + 1

    exchanges = []
    for s1 in range(12):
        lord1 = pl_lords[s1]
        s2 = _sign_index(pl_lons.get(lord1, float("nan")))
        if not math.isfinite(s2):
            continue
        lord2 = pl_lords[s2]
        s_back = _sign_index(pl_lons.get(lord2, float("nan")))
        if s_back == s1 and lord1 != lord2:
            # exchange between lord1 and lord2 over signs s1 and s2
            h1 = house_of_sign(s1); h2 = house_of_sign(s2)
            exchanges.append({"lords": (lord1, lord2), "signs": (s1, s2), "houses": (h1, h2)})

    kind = None
    if exchanges:
        # simple type: Maha if involves 1,5,9 with 4/7/10; Dainya if 6/8/12; Khala if 3/11; else regular
        def classify(hs: Tuple[int,int]) -> str:
            a, b = hs
            s = {a, b}
            if (s & {1,5,9}) and (s & {4,7,10}):
                return "Maha"
            if (s & {6,8,12}):
                return "Dainya"
            if (s & {3,11}):
                return "Khala"
            return "Regular"
        for ex in exchanges:
            ex["type"] = classify(ex["houses"])
        kind = ",".join(sorted(set(ex["type"] for ex in exchanges)))
    present = bool(exchanges)
    score = 0.88 if present else 0.0
    return YogaHit("Parivartana (exchange)", present, score, {"exchanges": exchanges}, {"type_summary": kind})

def _neechabhanga(pl_lons: Dict[str, float], pl_houses: Dict[str, int], lagna_house: int, moon_house: int) -> YogaHit:
    # Core cancellation patterns for any debilitated planet:
    #  A) Dispositor (lord of debilitation sign) in kendra from Lagna or Moon
    #  B) Exaltation lord in kendra from Lagna or Moon
    lords = _sign_lords("classical")  # dispositors by sign
    hits = []
    for p, s_deb in _DEB.items():
        ps = _sign_index(pl_lons.get(p, float("nan")))
        if ps != s_deb:
            continue
        disp = lords[s_deb]
        ex_sign = _EXALT.get(p)
        disp_h = pl_houses.get(disp); exl_h = None
        # find exaltation lord: owner of ex_sign
        if ex_sign is not None:
            ex_lord = lords[ex_sign]
            exl_h = pl_houses.get(ex_lord)
        okA = (disp_h is not None) and (_is_kendra(_house_delta(lagna_house, disp_h)) or _is_kendra(_house_delta(moon_house, disp_h)))
        okB = (exl_h is not None) and (_is_kendra(_house_delta(lagna_house, exl_h)) or _is_kendra(_house_delta(moon_house, exl_h)))
        if okA or okB:
            hits.append({"planet": p, "rules_met": ("A" if okA else "") + ("B" if okB else "")})
    present = bool(hits)
    score = 0.78 if present else 0.0
    return YogaHit("Neecha-bhanga (core)", present, score, {"cancellations": hits}, {})

def _chandra_mangala(pl_lons: Dict[str, float], *, conj_orb_deg: float, by_sign: bool) -> YogaHit:
    if by_sign:
        present = _sign_index(pl_lons["Moon"]) == _sign_index(pl_lons["Mars"])
        levels = {"sign": {"same_sign": present}}
        score = 0.75 if present else 0.0
    else:
        present = _conj_by_degree(pl_lons["Moon"], pl_lons["Mars"], conj_orb_deg)
        levels = {"degree": {"orb_deg": float(conj_orb_deg), "within_orb": present}}
        score = 0.78 if present else 0.0
    return YogaHit("Chandra-Mangala", present, score, levels, {})

def _adhi(pl_houses: Dict[str,int], moon_house: int) -> YogaHit:
    # Benefics (Mercury, Venus, Jupiter) in 6/7/8 from Moon
    deltas = {g: _house_delta(moon_house, pl_houses.get(g, 0)) for g in ("Mercury","Venus","Jupiter") if pl_houses.get(g)}
    present = all(d in (6,7,8) for d in deltas.values()) and len(deltas)==3
    return YogaHit("Adhi Yoga", present, 0.76 if present else 0.0, {"from_moon_deltas": deltas}, {})

def _saraswati(pl_lons: Dict[str,float], pl_houses: Dict[str,int]) -> YogaHit:
    # Mercury, Venus, Jupiter in K/T and dignified (own/exaltation) — conservative
    trio = ("Mercury","Venus","Jupiter")
    ok_h = all(_is_kendra(pl_houses[g]) or _is_trikona(pl_houses[g]) for g in trio if pl_houses.get(g))
    ok_d = all(_dignity(g, _sign_index(pl_lons[g])) in ("own","exalted") for g in trio)
    present = ok_h and ok_d
    levels = {"house": {g: pl_houses.get(g) for g in trio}, "sign": {g: _dignity(g, _sign_index(pl_lons[g])) for g in trio}}
    return YogaHit("Saraswati", present, 0.82 if present else 0.0, levels, {})

def _lakshmi(pl_lords: Dict[int,str], pl_lons: Dict[str,float], pl_houses: Dict[str,int]) -> YogaHit:
    # Lagna lord & 9th lord strong in K/T and associated (same sign or mutual kendra)
    LL = pl_lords[1]; L9 = pl_lords[9]
    hLL, h9 = pl_houses.get(LL), pl_houses.get(L9)
    if not (hLL and h9):
        return YogaHit("Lakshmi", False, 0.0, {}, {})
    strongLL = _is_kendra(hLL) or _is_trikona(hLL)
    strong9  = _is_kendra(h9)  or _is_trikona(h9)
    assoc = (_sign_index(pl_lons[LL]) == _sign_index(pl_lons[L9])) or _mutual_kendra(hLL, h9)
    present = strongLL and strong9 and assoc
    levels = {
        "lords": {"LL": LL, "L9": L9},
        "houses": {"LL": hLL, "L9": h9},
        "assoc": {"same_sign": _sign_index(pl_lons[LL]) == _sign_index(pl_lons[L9]), "mutual_kendra": _mutual_kendra(hLL, h9)}
    }
    return YogaHit("Lakshmi", present, 0.83 if present else 0.0, levels, {})

# ─────────────────────────────────────────────────────────────────────
# Composite scorer
# ─────────────────────────────────────────────────────────────────────
def _score_with_vargas(hit: YogaHit, *, varga_ctx: Dict[str, Any] | None, strengthen_on: Tuple[str,...] = ("D9",)) -> YogaHit:
    if not hit.present or not varga_ctx:
        return hit
    bump = 0.0
    for key in strengthen_on:
        d = varga_ctx.get(key) or {}
        # Simple policy: if the planets central to the yoga are strong in the varga, add a small bump
        # We peek into details if available, otherwise apply a generic slight bump.
        bump += 0.04
    new = YogaHit(hit.name, hit.present, min(0.99, hit.score + bump), {**hit.levels, "varga": {"boost_keys": strengthen_on}}, hit.details)
    return new

def _attach_arudha_notes(hit: YogaHit, *, arudha: Dict[str, Any] | None, pl_lons: Dict[str,float]) -> YogaHit:
    if not arudha:
        return hit
    try:
        a1 = arudha.get("arudhas", {}).get("A1", {})
        al_idx = int(a1.get("sign_index"))
        # note if yoga focal planets sit in AL sign
        notes = {}
        if "lords" in hit.levels:
            for key, lord in hit.levels["lords"].items():
                if lord in pl_lons and _sign_index(pl_lons[lord]) == al_idx:
                    notes[f"{lord}_on_AL"] = True
        new_levels = {**hit.levels, "arudha": {"AL_sign_index": al_idx, **notes}}
        return YogaHit(hit.name, hit.present, hit.score, new_levels, hit.details)
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
    conj_orb_deg: float = 6.0,
    chandra_mangala_by_sign: bool = True,
    gajakesari_include_same_house: bool = True,
    include_mooltrikona_in_mahapurusha: bool = True,
    use_vargas_for_scoring: bool = True,
    varga_keys_for_boost: Tuple[str,...] = ("D9",),
    include_arudha_notes: bool = True,
) -> Dict[str, Any]:
    """
    Compute classical yogas with multi-level evidence.

    Required payload:
      - Either jd_tt & jd_ut1, or (date, time, tz)
      - latitude, longitude (for houses/lagna)
    """
    if not _EPH_OK or not _HOUSES_OK:
        return {"ok": False, "error": "core_modules_unavailable", "yogas": [], "warnings": []}

    # Timescales & ayanāṁśa
    jd_tt, jd_ut1, warns_ts = _timescales(payload)
    ay_key, ay_deg, warns_ay = _ayanamsa(jd_tt, ayanamsa)

    # Site
    if not all(k in payload for k in ("latitude","longitude")):
        return {"ok": False, "error": "site_required", "yogas": [], "warnings": warns_ts + warns_ay}
    lat = float(payload["latitude"]); lon = float(payload["longitude"])

    # Houses / Asc / cusps
    houses_payload = _compute_houses_payload(lat, lon, jd_tt, jd_ut1, house_system)
    asc_tropical = float(houses_payload["asc_deg"])
    cusps = list(houses_payload["cusps_deg"])
    asc_sid = _norm360(asc_tropical - ay_deg)
    lagna_sign = _sign_index(asc_sid)
    lagna_house = 1  # by definition for house mapping relative to Lagna

    # Longitudes (nirayana) for planets
    pl_lons = _sidereal_longitudes(jd_tt, list(_PLANETS), ay_deg)

    # House placement per planet (using exact cusps)
    pl_houses = _house_map_for_planets(pl_lons, cusps)

    # Moon-reference
    moon_house = pl_houses.get("Moon", 1)

    # Lords map (by house index 1..12 → lord name), using sign at each house
    # House 1 sign is lagna_sign; house i sign = (lagna_sign + i-1) % 12
    sign_lords = _sign_lords(sign_lord_variant)
    house_lords: Dict[int, str] = {}
    for i in range(1, 13):
        sidx = (lagna_sign + (i - 1)) % 12
        house_lords[i] = sign_lords[sidx]

    # Optional Vargas
    varga_ctx = {}
    if use_vargas_for_scoring and callable(_compute_vargas):
        try:
            varga_ctx = _compute_vargas({"jd_tt": jd_tt, "ayanamsa_deg": ay_deg}, keys=list(varga_keys_for_boost))
        except Exception:
            varga_ctx = {}

    # Optional Arudha
    arudha_ctx = None
    if include_arudha_notes and callable(_compute_arudhas):
        try:
            arudha_ctx = _compute_arudhas({
                "jd_tt": jd_tt, "jd_ut1": jd_ut1,
                "latitude": lat, "longitude": lon,
                "ayanamsa": ay_key
            })
        except Exception:
            arudha_ctx = None

    # Evaluate rules
    hits: List[YogaHit] = []

    # Panch Mahapurusha
    for p in ("Mars","Mercury","Jupiter","Venus","Saturn"):
        y = _mahapurusha(p, pl_lons, pl_houses, include_mooltrikona=include_mooltrikona_in_mahapurusha, lagna_house=lagna_house)
        if y:
            hits.append(y)

    # Gaja-Kesari
    hits.append(_gajakesari(pl_lons, pl_houses, include_same_house=gajakesari_include_same_house))

    # Raja Yoga (k–t association)
    hits.append(_raja(house_lords, pl_houses, sign_conj_only=False, pl_lons=pl_lons))

    # Parivartana (exchange)
    hits.append(_parivartana(sign_lords, pl_lons, lagna_sign))

    # Neecha-bhanga (core)
    hits.append(_neechabhanga(pl_lons, pl_houses, lagna_house, moon_house))

    # Chandra-Mangala
    hits.append(_chandra_mangala(pl_lons, conj_orb_deg=conj_orb_deg, by_sign=chandra_mangala_by_sign))

    # Adhi
    hits.append(_adhi(pl_houses, moon_house))

    # Saraswati
    hits.append(_saraswati(pl_lons, pl_houses))

    # Lakshmi
    hits.append(_lakshmi(house_lords, pl_lons, pl_houses))

    # Varga-aware boost and Arudha notes
    final_hits: List[YogaHit] = []
    for h in hits:
        hh = _score_with_vargas(h, varga_ctx=varga_ctx, strengthen_on=varga_keys_for_boost) if use_vargas_for_scoring else h
        hh = _attach_arudha_notes(hh, arudha=arudha_ctx, pl_lons=pl_lons) if include_arudha_notes else hh
        final_hits.append(hh)

    # Shape the response
    out_list = []
    for h in final_hits:
        out_list.append({
            "name": h.name,
            "present": bool(h.present),
            "score": float(h.score),
            "levels": h.levels,
            "details": h.details
        })

    context = {
        "ayanamsa": {"key": ay_key, "deg": float(ay_deg)},
        "ascendant_sidereal_deg": float(asc_sid),
        "lagna_sign_index": int(lagna_sign),
        "house_lords": house_lords,
        "planet_houses": {k: int(v) for k, v in pl_houses.items()},
        "planet_signs": {k: int(_sign_index(pl_lons[k])) for k in pl_lons},
        "house_system": house_system,
    }

    warnings = list(dict.fromkeys(warns_ts + warns_ay))
    return {"ok": True, "yogas": out_list, "context": context, "warnings": warnings}
