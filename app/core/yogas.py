# app/core/yogas.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Classical Yogas — single pipeline (civil birth details → timescales → ephemeris → houses → yogas).

Hardened version
----------------
• Conservative defaults (caller can loosen via options):
  - chandra_mangala_by_sign=False (use degree-orb unless explicitly requested)
  - gajakesari_include_same_house=False (require 4/7/10 from Moon)
  - include_mooltrikona_in_mahapurusha=False (only own/exalted by default)
• Mahāpuruṣa mooltrikona uses a real checker only if available; otherwise it's ignored with a warning.
• Tag gating is *per-call* (no global flag flips).
• Varga scoring uses sidereal inputs consistently.
• Registry passes an `opts` dict to each rule; no capture at definition time.
• Ephemeris names are canonicalized (Sun..Saturn, Rahu/Ketu), and the opposite node is synthesized if missing.
• House indices from assign_houses are normalized to 1..12.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable, Set, Union
import math
import os
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────
# Optional imports & guards
# ─────────────────────────────────────────────────────────────────────
try:
    from app.core.ayanamsa import get_ayanamsa_deg as _get_ayanamsa_deg
except Exception:
    _get_ayanamsa_deg = None

# Ephemeris (singleton adapter)
_EPH_OK = True
try:
    # TS/PLANETS imported for compatibility elsewhere; not required here.
    from app.core.ephem_singleton import TS, PLANETS  # noqa: F401
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
except Exception:
    _EPH_OK = False
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore

# Houses façade
_HOUSES_OK = True
try:
    from app.core.house import compute_houses_with_policy as _compute_houses_with_policy
except Exception:
    _HOUSES_OK = False
    _compute_houses_with_policy = None  # type: ignore

# House assignment helper
_ASSIGN_OK = True
try:
    from app.core.houses_advanced import assign_houses as _assign_houses
except Exception:
    _ASSIGN_OK = False
    def _assign_houses(_lstA, _lstB):  # type: ignore
        raise RuntimeError("assign_houses unavailable")

# Panchanga (optional waxing/waning Moon)
try:
    from app.core.panchanga import panchanga_elements_at as _panchanga_elements_at
except Exception:
    _panchanga_elements_at = None

# Vargas
_VARGA_OK = True
try:
    from app.core.varga_charts import compute_many_vargas as _compute_many_vargas
except Exception:
    _VARGA_OK = False
    _compute_many_vargas = None  # type: ignore

# timescales backends
_ts = None
_tk = None
try:
    from app.core import timescales as _ts  # fallback backend
except Exception:
    _ts = None
try:
    from app.core import time_kernel as _tk  # preferred backend
except Exception:
    _tk = None

# Constants & dignity maps
try:
    from app.core.constants_vedic import SIGN_NAMES as _SIGNS  # noqa: F401
except Exception:
    _SIGNS = (
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
    )

try:
    from app.core.constants_vedic import EXALTATION_SIGN_INDEX as _EXALT
    from app.core.constants_vedic import DEBILITATION_SIGN_INDEX as _DEB
    from app.core.constants_vedic import OWN_SIGN_INDEXES as _OWN
except Exception:
    _EXALT = {"Sun": 0, "Moon": 1, "Mars": 9, "Mercury": 5, "Jupiter": 3, "Venus": 11, "Saturn": 6}
    _DEB = {"Sun": 6, "Moon": 7, "Mars": 3, "Mercury": 11, "Jupiter": 9, "Venus": 5, "Saturn": 0}
    _OWN = {"Sun": (4,), "Moon": (3,), "Mars": (0, 7), "Mercury": (2, 5), "Jupiter": (8, 11), "Venus": (1, 6), "Saturn": (9, 10)}

# Optional mooltrikona checker (used only if present)
try:
    from app.core.constants_vedic import is_mooltrikona_position as _is_mt_pos  # type: ignore
except Exception:
    _is_mt_pos = None  # type: ignore

# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────
_PLANETS_MAIN: Tuple[str, ...] = ("Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn")
_PLANETS_ALL: Tuple[str, ...]  = ("Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu")

def _env_dut1_seconds() -> float:
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST",
                                    os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _angdiff(a: float, b: float) -> float:
    return ((_norm360(a) - _norm360(b) + 540.0) % 360.0) - 180.0

def _sign_index(lon: float) -> int:
    return int(math.floor(_norm360(lon) / 30.0)) % 12

def _house_delta(h_from: Optional[int], h_to: Optional[int]) -> Optional[int]:
    if not h_from or not h_to:
        return None
    return ((h_to - h_from) % 12) + 1

def _is_kendra(h: int) -> bool:   return h in (1, 4, 7, 10)
def _is_trikona(h: int) -> bool:  return h in (1, 5, 9)
def _is_upachaya(h: int) -> bool: return h in (3, 6, 10, 11)
def _is_dusthana(h: int) -> bool: return h in (6, 8, 12)

def _conj(a: float, b: float, orb: float) -> bool:
    return abs(_angdiff(a, b)) <= max(0.0, float(orb))

def _has_graha_drishti(g: str, si_from: int, si_to: int) -> bool:
    d = ((si_to - si_from) % 12)
    if d == 6: return True
    if g == "Jupiter" and d in (4, 8): return True
    if g == "Mars"    and d in (3, 9): return True
    if g == "Saturn"  and d in (2,10): return True
    return False

def _sign_lords(variant: str = "classical") -> Dict[int, str]:
    if str(variant).lower().startswith("nodes"):
        return {0:"Mars",1:"Venus",2:"Mercury",3:"Moon",4:"Sun",5:"Mercury",6:"Venus",7:"Ketu",8:"Jupiter",9:"Saturn",10:"Rahu",11:"Jupiter"}
    return {0:"Mars",1:"Venus",2:"Mercury",3:"Moon",4:"Sun",5:"Mercury",6:"Venus",7:"Mars",8:"Jupiter",9:"Saturn",10:"Saturn",11:"Jupiter"}

def _dignity(planet: str, sign_idx: int) -> str:
    if _EXALT.get(planet) == sign_idx: return "exalted"
    if _DEB.get(planet)  == sign_idx: return "debilitated"
    if sign_idx in _OWN.get(planet, ()): return "own"
    return "neutral"

def _is_mooltrikona(planet: str, lon_sidereal: float) -> bool:
    if callable(_is_mt_pos):
        try:
            return bool(_is_mt_pos(planet, float(lon_sidereal)))
        except Exception:
            return False
    return False

@lru_cache(maxsize=4096)
def _waxing_moon_cached(jd_tt_bucket: int, ay_key: str) -> Optional[bool]:
    if _panchanga_elements_at is None: return None
    try:
        el = _panchanga_elements_at(float(jd_tt_bucket), ayanamsa_key=ay_key)
        idx = int(el["tithi"]["index"])
        return idx <= 15
    except Exception:
        return None

def _benefic_set(jd_tt: float, ay_key: str) -> Set[str]:
    base = {"Jupiter", "Venus", "Mercury"}
    waxing = _waxing_moon_cached(int(round(jd_tt)), ay_key)
    if waxing is False: return set(base)
    return base | {"Moon"}

# ─────────────────────────────────────────────────────────────────────
# Timescales, Ayanāṁśa & Ephemeris (cached)
# ─────────────────────────────────────────────────────────────────────
def _require_birth_details(payload: Dict[str, Any]) -> Tuple[str, str, str, float, float]:
    d  = payload.get("date")     or payload.get("birth", {}).get("date")
    t  = payload.get("time")     or payload.get("birth", {}).get("time")
    tz = payload.get("tz")       or payload.get("birth", {}).get("tz") or payload.get("place_tz")
    lat = payload.get("latitude")  or payload.get("birth", {}).get("latitude")
    lon = payload.get("longitude") or payload.get("birth", {}).get("longitude")
    if not all((d, t, tz, lat, lon)):
        raise ValueError("birth_details_required (date, time, tz, latitude, longitude)")
    return str(d), str(t), str(tz), float(lat), float(lon)

def _timescales_from_civil(date: str, time: str, tz: str) -> Tuple[float, float, List[str]]:
    warns: List[str] = []
    dut1 = _env_dut1_seconds()

    if _tk is not None:
        for fname in ("timescales_from_civil", "compute_timescales", "build_timescales", "to_timescales", "from_civil"):
            fn = getattr(_tk, fname, None)
            if not callable(fn):
                continue
            try:
                if fname == "build_timescales":
                    out = fn(date=date, time=time, tz=tz, dut1_seconds=dut1)
                else:
                    try:
                        out = fn(date=date, time=time, tz=tz, dut1_seconds=dut1)
                    except TypeError:
                        out = fn(date=date, time=time, tz=tz)
            except TypeError:
                try:
                    if fname == "build_timescales":
                        out = fn(date, time, tz, dut1)
                    else:
                        try:
                            out = fn(date, time, tz, dut1)
                        except TypeError:
                            out = fn(date, time, tz)
                except TypeError:
                    continue

            if isinstance(out, dict):
                jt = float(out.get("jd_tt") or out.get("tt") or 0.0)
                ju = float(out.get("jd_ut1") or out.get("ut1") or jt)
                return jt, ju, warns
            if isinstance(out, (list, tuple)) and len(out) >= 3:
                ju, jt = float(out[1]), float(out[0])
                return jt, ju, warns

    if _ts is None:
        raise RuntimeError("timescales_unavailable")

    try:
        _bts = getattr(_ts, "build_timescales", None)
        if callable(_bts):
            try:
                out = _bts(date=date, time=time, tz=tz, dut1_seconds=dut1)
            except TypeError:
                out = _bts(date, time, tz, dut1)
            if isinstance(out, dict):
                jt = float(out.get("jd_tt") or out.get("tt") or 0.0)
                ju = float(out.get("jd_ut1") or out.get("ut1") or jt)
                return jt, ju, warns
    except Exception:
        pass

    jd_ut = float(_ts.julian_day_utc(date, time, tz))  # type: ignore[attr-defined]
    try:
        y, m = map(int, str(date).split("-")[:2])
        jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m))  # type: ignore[attr-defined]
    except Exception:
        jd_tt = jd_ut + 69.0 / 86400.0  # ~ΔT fallback
        warns.append("deltaT_fallback_69s")
    return jd_tt, jd_ut, warns

@lru_cache(maxsize=4096)
def _ayanamsa_cached(jd_tt_bucket: int, key_or_deg: str) -> Tuple[str, float, Tuple[str, ...]]:
    warns: List[str] = []
    try:
        val = float(key_or_deg)
        return "explicit", val, tuple(warns)
    except Exception:
        pass
    key = str(key_or_deg or "lahiri").strip().lower()
    if _get_ayanamsa_deg is None:
        AY_J2000_DEG = (23 + 51/60 + 26.26/3600)
        RATE_AS_PER_YR = 50.290966
        years = (float(jd_tt_bucket)/1e6 - 2451545.0) / 365.25
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
        warns.append("ayanamsa_fallback_lahiri_linearized")
        return key, ay, tuple(warns)
    try:
        ay = float(_get_ayanamsa_deg(float(jd_tt_bucket)/1e6, key))
        return key, ay, tuple(warns)
    except Exception as e:
        AY_J2000_DEG = (23 + 51/60 + 26.26/3600)
        RATE_AS_PER_YR = 50.290966
        years = (float(jd_tt_bucket)/1e6 - 2451545.0) / 365.25
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
        warns.append(f"ayanamsa_resolve_error:{e}")
        warns.append("ayanamsa_fallback_lahiri_linearized")
        return key, ay, tuple(warns)

# ─────────────────────────────────────────────────────────────────────
# Ephemeris (canonicalization + caching)
# ─────────────────────────────────────────────────────────────────────
_EPHEM: Optional[EphemerisAdapter] = None

def _ephem() -> EphemerisAdapter:
    global _EPHEM
    if _EPHEM is None:
        if not _EPH_OK:
            raise RuntimeError("ephemeris_unavailable")
        try:
            _EPHEM = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date"))
        except TypeError:
            _EPHEM = EphemerisAdapter()  # type: ignore
    return _EPHEM

def _canon_planet_key(s: Union[str, None]) -> Optional[str]:
    if s is None:
        return None
    k = str(s).strip().lower().replace("(true)", "").replace("(mean)", "").strip()
    base = {
        "sun": "Sun", "sol": "Sun",
        "moon": "Moon", "luna": "Moon",
        "mercury": "Mercury",
        "venus": "Venus",
        "mars": "Mars",
        "jupiter": "Jupiter",
        "saturn": "Saturn",
        # nodes (lots of spellings)
        "rahu": "Rahu", "north node": "Rahu", "ascending node": "Rahu", "true node": "Rahu", "mean node": "Rahu",
        "ketu": "Ketu", "south node": "Ketu", "descending node": "Ketu",
    }
    return base.get(k)

@lru_cache(maxsize=4096)
def _ecliptic_lons_cached(jd_tt_bucket: int, names_key: Tuple[str, ...]) -> Dict[str, float]:
    # ask adapter; canonicalize names; synthesize missing node
    res = _ephem().ecliptic_longitudes(float(jd_tt_bucket)/1e6, list(names_key))
    rows = (res or {}).get("results", []) if isinstance(res, dict) else []
    out: Dict[str, float] = {}
    for r in rows or []:
        nm = _canon_planet_key(r.get("name"))
        if nm:
            try:
                out[nm] = float(r["longitude"])
            except Exception:
                continue
    # if only one node present, synthesize the other
    if "Rahu" in out and "Ketu" not in out:
        out["Ketu"] = _norm360(out["Rahu"] + 180.0)
    elif "Ketu" in out and "Rahu" not in out:
        out["Rahu"] = _norm360(out["Ketu"] + 180.0)
    return out

def _sidereal_longitudes(jd_tt: float, names: List[str], ay_deg: float) -> Dict[str, float]:
    jd_bucket = int(round(float(jd_tt) * 1e6))  # microday bucket
    lons_trop = _ecliptic_lons_cached(jd_bucket, tuple(names))
    return {nm: _norm360(lon - float(ay_deg)) for nm, lon in lons_trop.items()}

# ─────────────────────────────────────────────────────────────────────
# Houses & mapping
# ─────────────────────────────────────────────────────────────────────
def _compute_houses_payload(lat: float, lon: float, jd_tt: float, jd_ut1: float, house_system: str) -> Dict[str, Any]:
    if not (_HOUSES_OK and callable(_compute_houses_with_policy)):
        raise RuntimeError("houses_module_unavailable")
    return _compute_houses_with_policy(  # type: ignore[misc]
        lat=float(lat),
        lon=float(lon),
        system=str(house_system or "placidus"),
        jd_tt=float(jd_tt),
        jd_ut1=float(jd_ut1),
        jd_ut=float(jd_ut1),
        diagnostics=False,
        validation=False,
    )

def _house_map_for_planets(planet_lons: Dict[str, float], cusps: List[float]) -> Dict[str, int]:
    idxs = _assign_houses([planet_lons[p] for p in planet_lons], cusps)
    # normalize to 1..12
    ints = [int(round(i)) for i in idxs]
    if all(0 <= i <= 11 for i in ints):
        ints = [((i % 12) + 1) for i in ints]
    else:
        ints = [i if 1 <= i <= 12 else (((i - 1) % 12) + 1) for i in ints]
    return {k: ints[i] for i, k in enumerate(planet_lons.keys())}

# ─────────────────────────────────────────────────────────────────────
# Vargas (D9/D10 + vargottama)
# ─────────────────────────────────────────────────────────────────────
def _varga_context(
    pl_lons_sidereal: Dict[str, float],
    ay_key: str,
    enable: bool,
    varga_keys: Tuple[str, ...],
) -> Tuple[Dict[str, bool], Dict[str, int], Dict[str, int], List[str]]:
    warns: List[str] = []
    if not (enable and _VARGA_OK and callable(_compute_many_vargas)):
        return {}, {}, {}, warns
    try:
        # Inputs are SIDERAL; tell the engine and pass the key
        res = _compute_many_vargas(pl_lons_sidereal, list(varga_keys), zodiac_mode="sidereal", ayanamsa=ay_key)  # type: ignore[misc]
    except Exception as e:
        warns.append(f"varga_compute_error:{e}")
        return {}, {}, {}, warns

    d9: Dict[str, int] = {}
    d10: Dict[str, int] = {}
    for name, sub in (res.get("D9") or {}).items():
        if isinstance(sub, dict) and "varga_rasi_index" in sub:
            d9[str(name)] = int(sub["varga_rasi_index"])
    for name, sub in (res.get("D10") or {}).items():
        if isinstance(sub, dict) and "varga_rasi_index" in sub:
            d10[str(name)] = int(sub["varga_rasi_index"])

    vargottama: Dict[str, bool] = {}
    for p, lon in pl_lons_sidereal.items():
        try:
            d1 = _sign_index(lon)
            vargottama[p] = (d1 == d9.get(p, -99))
        except Exception:
            vargottama[p] = False

    return vargottama, d9, d10, warns

# ─────────────────────────────────────────────────────────────────────
# Data structures & Registry
# ─────────────────────────────────────────────────────────────────────
@dataclass
class YogaHit:
    name: str
    present: bool
    score: float
    levels: Dict[str, Any]
    details: Dict[str, Any]
    tags: Tuple[str, ...] = ()

RuleFn = Callable[[Dict[str, Any], Dict[str, Any]], Union[YogaHit, List[YogaHit]]]
_RULES: Dict[str, Dict[str, Any]] = {}  # name -> {"fn":RuleFn, "tags":tuple, "enabled":bool, "meta":dict}
_BUILT = False

def register_yoga(
    name: str,
    fn: RuleFn,
    *,
    tags: Iterable[str] = (),
    enabled: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    _RULES[name] = {"fn": fn, "tags": tuple(tags), "enabled": bool(enabled), "meta": dict(meta or {})}

# ─────────────────────────────────────────────────────────────────────
# Core rule implementations
# ─────────────────────────────────────────────────────────────────────
def _mahapurusha_rule_for(planet: str, title: str) -> RuleFn:
    def _fn(ctx: Dict[str, Any], opts: Dict[str, Any]) -> YogaHit:
        lon = ctx["pl_lons"].get(planet)
        if lon is None:
            return YogaHit(title, False, 0.0, {}, {}, ("mahapurusha", "strength"))
        si = _sign_index(lon)
        house = ctx["pl_houses"].get(planet)
        dign = _dignity(planet, si)
        mt_opt = bool(opts.get("mahapurusha_include_mooltrikona", False))
        mt_ok = mt_opt and _is_mooltrikona(planet, lon)
        strong = dign in ("own", "exalted") or mt_ok
        present = bool(house) and _is_kendra(house) and strong
        levels = {
            "house": {"is_kendra": _is_kendra(house or 0), "house": house},
            "sign": {"index": si, "dignity": dign, "mooltrikona": bool(mt_ok)}
        }
        score = 0.9 if dign == "exalted" else (0.85 if dign == "own" else (0.83 if mt_ok else 0.0))
        return YogaHit(title, present, score if present else 0.0, levels, {"planet": planet}, ("mahapurusha", "strength"))
    return _fn

def _gaja_kesari_rule(ctx: Dict[str, Any], opts: Dict[str, Any]) -> YogaHit:
    include_same = bool(opts.get("gajakesari_include_same_house", False))
    h_m = ctx["pl_houses"].get("Moon");  h_j = ctx["pl_houses"].get("Jupiter")
    d = _house_delta(h_m, h_j)
    present = bool(d and (d in (1,4,7,10) if include_same else d in (4,7,10)))
    lev = {"house": {"from_moon_delta": d, "include_same_house": include_same}}
    jd = _dignity("Jupiter", _sign_index(ctx["pl_lons"].get("Jupiter", 0.0)))
    score = (0.82 + (0.05 if jd in ("own","exalted") else 0.0)) if present else 0.0
    return YogaHit("Gaja-Kesari", present, score, lev, {}, ("moon", "benefic", "strength"))

def _chandra_mangala_rule(ctx: Dict[str, Any], opts: Dict[str, Any]) -> YogaHit:
    by_sign = bool(opts.get("chandra_mangala_by_sign", False))
    orb_deg = float(opts.get("conj_orb_deg", 6.0))
    m  = ctx["pl_lons"].get("Moon");  ma = ctx["pl_lons"].get("Mars")
    if m is None or ma is None:
        return YogaHit("Chandra-Mangala", False, 0.0, {}, {}, ("moon", "wealth"))
    if by_sign:
        present = _sign_index(m) == _sign_index(ma)
        lev = {"sign": {"same_sign": present}}
        score = 0.78 if present else 0.0
    else:
        present = _conj(m, ma, orb_deg)
        lev = {"degree": {"orb_deg": orb_deg, "within_orb": present}}
        score = 0.8 if present else 0.0
    return YogaHit("Chandra-Mangala", present, score, lev, {}, ("moon", "wealth"))

def _adhi_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    h_m = ctx["pl_houses"].get("Moon")
    if not h_m: return YogaHit("Adhi", False, 0.0, {}, {}, ("moon", "benefic"))
    deltas = {g: _house_delta(h_m, ctx["pl_houses"].get(g)) for g in ("Mercury","Venus","Jupiter") if ctx["pl_houses"].get(g)}
    present = len(deltas) == 3 and all(d in (6,7,8) for d in deltas.values())
    return YogaHit("Adhi", present, 0.78 if present else 0.0, {"from_moon_deltas": deltas}, {}, ("moon", "benefic"))

def _durudhara_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    moon = ctx["pl_lons"].get("Moon")
    if moon is None: return YogaHit("Durudhara (Moon flanked)", False, 0.0, {}, {}, ("moon", "kartari"))
    left  = any(0 < _angdiff(ctx["pl_lons"][p], moon) < 180 for p in _PLANETS_MAIN if p != "Moon")
    right = any(0 < _angdiff(moon, ctx["pl_lons"][p]) < 180 for p in _PLANETS_MAIN if p != "Moon")
    present = left and right
    return YogaHit("Durudhara (Moon flanked)", present, 0.72 if present else 0.0, {}, {}, ("moon", "kartari"))

def _veshi_voshi_ubhay_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> List[YogaHit]:
    sun = ctx["pl_lons"].get("Sun")
    if sun is None: return [YogaHit("Veshi/Voshi/Ubhayachari", False, 0.0, {}, {}, ("sun", "kartari"))]
    left  = [p for p in _PLANETS_MAIN if p not in ("Sun","Moon") and 0 < _angdiff(ctx["pl_lons"][p], sun) < 180]
    right = [p for p in _PLANETS_MAIN if p not in ("Sun","Moon") and 0 < _angdiff(sun, ctx["pl_lons"][p]) < 180]
    hits: List[YogaHit] = []
    if left and not right:  hits.append(YogaHit("Veshi (planet after Sun)", True, 0.7, {"planets": left}, {}, ("sun", "kartari")))
    if right and not left:  hits.append(YogaHit("Voshi (planet before Sun)", True, 0.7, {"planets": right}, {}, ("sun", "kartari")))
    if left and right:      hits.append(YogaHit("Ubhayachari (both sides of Sun)", True, 0.75, {"left": left, "right": right}, {}, ("sun", "kartari")))
    if not hits:            hits.append(YogaHit("Veshi/Voshi/Ubhayachari", False, 0.0, {}, {}, ("sun", "kartari")))
    return hits

def _kartari_rule(around: str) -> RuleFn:
    def _fn(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> List[YogaHit]:
        center_house = 1 if around == "Lagna" else ctx["pl_houses"].get(around, 1)
        h12 = ((center_house + 10 - 1) % 12) + 1
        h2  = ((center_house + 1  - 1) % 12) + 1
        benefics = ctx["benefics"]
        malefics = {"Saturn","Mars","Sun","Rahu","Ketu"} | (set() if "Moon" in benefics else {"Moon"})
        at_12 = [p for p, h in ctx["pl_houses"].items() if h == h12]
        at_2  = [p for p, h in ctx["pl_houses"].items() if h == h2]
        has_b = any(p in benefics for p in at_12) and any(p in benefics for p in at_2)
        has_m = any(p in malefics for p in at_12) and any(p in malefics for p in at_2)
        return [
            YogaHit(f"Shubha Kartari around {around}", has_b, 0.75 if has_b else 0.0,
                    {"houses":{"h12":h12,"h2":h2}, "planets":{"h12":at_12,"h2":at_2}}, {}, ("kartari","benefic")),
            YogaHit(f"Papa Kartari around {around}",  has_m, 0.65 if has_m else 0.0,
                    {"houses":{"h12":h12,"h2":h2}, "planets":{"h12":at_12,"h2":at_2}}, {}, ("kartari","malefic")),
        ]
    return _fn

def _raja_rule(ctx: Dict[str, Any], opts: Dict[str, Any]) -> YogaHit:
    conj_only = bool(opts.get("raja_conj_only", False))
    pairs = []
    for k in (1,4,7,10):
        for t in (1,5,9):
            a = ctx["house_lords"][k]; b = ctx["house_lords"][t]
            if a == b: continue
            ha = ctx["pl_houses"].get(a); hb = ctx["pl_houses"].get(b)
            if not (ha and hb): continue
            same_sign = _sign_index(ctx["pl_lons"][a]) == _sign_index(ctx["pl_lons"][b])
            assoc = same_sign or (_is_kendra(ha) and _is_kendra(hb)) or _has_graha_drishti(
                a, _sign_index(ctx["pl_lons"][a]), _sign_index(ctx["pl_lons"][b])
            )
            if conj_only: assoc = same_sign
            if assoc: pairs.append({"kendra_lord": a, "trikona_lord": b, "houses": (ha, hb), "same_sign": same_sign})
    present = bool(pairs)
    return YogaHit("Raja (k–t association)", present, 0.82 if present else 0.0, {"pairs": pairs}, {}, ("raja","association"))

def _dharma_karmadhipati_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    L9 = ctx["house_lords"][9]; L10 = ctx["house_lords"][10]
    h9 = ctx["pl_houses"].get(L9); h10 = ctx["pl_houses"].get(L10)
    if not (h9 and h10): return YogaHit("Dharma-Karmadhipati", False, 0.0, {}, {}, ("raja","dk"))
    same_sign = _sign_index(ctx["pl_lons"][L9]) == _sign_index(ctx["pl_lons"][L10])
    assoc = same_sign or (_is_kendra(h9) and _is_kendra(h10)) or _has_graha_drishti(
        L9, _sign_index(ctx["pl_lons"][L9]), _sign_index(ctx["pl_lons"][L10])
    )
    lev = {"lords":{"L9":L9,"L10":L10}, "houses":{"L9":h9,"L10":h10}, "same_sign":same_sign}
    return YogaHit("Dharma-Karmadhipati", assoc, 0.86 if assoc else 0.0, lev, {}, ("raja","dk"))

def _parivartana_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    lords = ctx["sign_lords"]; lagna_sign = ctx["lagna_sign"]
    def house_of_sign(si: int) -> int: return ((si - lagna_sign) % 12) + 1
    exchanges = []
    for s1 in range(12):
        lord1 = lords[s1]; s2 = _sign_index(ctx["pl_lons"].get(lord1, float("nan")))
        if not math.isfinite(s2): continue
        lord2 = lords[s2]; s_back = _sign_index(ctx["pl_lons"].get(lord2, float("nan")))
        if s_back == s1 and lord1 != lord2:
            h1, h2 = house_of_sign(s1), house_of_sign(s2)
            exchanges.append({"lords": (lord1, lord2), "signs": (s1, s2), "houses": (h1, h2)})
    def classify(hs: Tuple[int,int]) -> str:
        a,b = hs; S = {a,b}
        if (S & {1,5,9}) and (S & {4,7,10}): return "Maha"
        if (S & {6,8,12}): return "Dainya"
        if (S & {3,11}):   return "Khala"
        return "Regular"
    for ex in exchanges: ex["type"] = classify(ex["houses"])
    present = bool(exchanges); types = sorted(set(ex["type"] for ex in exchanges)) if present else []
    return YogaHit("Parivartana (exchange)", present, 0.88 if present else 0.0, {"exchanges": exchanges}, {"types": types}, ("exchange","raja"))

def _neechabhanga_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    lords = ctx["sign_lords"]; lagna = 1; moon_h = ctx["pl_houses"].get("Moon", 1)
    cancels = []
    for p, deb_s in _DEB.items():
        ps = _sign_index(ctx["pl_lons"].get(p, float("nan")))
        if ps != deb_s: continue
        disp = lords[deb_s]; ex_s = _EXALT.get(p); ex_lord = lords[ex_s] if ex_s is not None else None
        disp_h = ctx["pl_houses"].get(disp); ex_h = ctx["pl_houses"].get(ex_lord) if ex_lord else None
        okA = bool(disp_h and (_is_kendra(_house_delta(lagna, disp_h)) or _is_kendra(_house_delta(moon_h, disp_h))))
        okB = bool(ex_h   and (_is_kendra(_house_delta(lagna, ex_h))   or _is_kendra(_house_delta(moon_h, ex_h))))
        dr = bool(ex_lord and _has_graha_drishti(ex_lord, _sign_index(ctx["pl_lons"][ex_lord]), ps))
        exch = False
        if disp and ex_lord:
            s_disp = _sign_index(ctx["pl_lons"][disp]); s_exl = _sign_index(ctx["pl_lons"][ex_lord])
            exch = (lords[s_disp] == p and lords[s_exl] == p)
        if okA or okB or dr or exch:
            cancels.append({"planet": p, "rules": {"A": okA, "B": okB, "drishti_exalt_lord": dr, "exchange_disp_exalt": exch}})
    present = bool(cancels); score = 0.8 if present else 0.0
    return YogaHit("Neecha-bhanga (composite)", present, score, {"cancellations": cancels}, {}, ("cancellation","raja"))

def _vipareeta_raja_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    res = []
    for H, name in ((6,"Harsha"), (8,"Sarala"), (12,"Vimala")):
        lord = ctx["house_lords"][H]; h = ctx["pl_houses"].get(lord)
        if h and _is_dusthana(h):
            dig = _dignity(lord, _sign_index(ctx["pl_lons"][lord]))
            res.append({"type": name, "lord": lord, "house": h, "dignity": dig})
    present = bool(res)
    score = 0.82 if any(r["dignity"] in ("own","exalted") for r in res) else (0.76 if present else 0.0)
    return YogaHit("Vipareeta Raja (tri-dusthana lords)", present, score, {"instances": res}, {}, ("vipareeta","raja"))

def _amala_rule(from_ref: str) -> RuleFn:
    def _fn(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
        center = 1 if from_ref == "Lagna" else ctx["pl_houses"].get("Moon", 1)
        target = ((center + 8 - 1) % 12) + 1
        benefics = ctx["benefics"]
        pls = [p for p, h in ctx["pl_houses"].items() if h == target and p in benefics]
        present = bool(pls)
        return YogaHit(f"Amala (benefic 10th from {from_ref})", present, 0.78 if present else 0.0, {"house": target, "planets": pls}, {}, ("amala","career"))
    return _fn

def _chatussagara_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    have = {1: False, 4: False, 7: False, 10: False}
    for p, h in ctx["pl_houses"].items():
        if p in _PLANETS_MAIN and h in have: have[h] = True
    present = all(have.values())
    return YogaHit("Chatussagara", present, 0.76 if present else 0.0, {"kendras": have}, {}, ("kendras","strength"))

def _vasumati_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    m = ctx["pl_houses"].get("Moon", 1)
    upas = {((m+2-1)%12)+1, ((m+5-1)%12)+1, ((m+9-1)%12)+1, ((m+10-1)%12)+1}
    bens = ctx["benefics"]
    count = sum(1 for p,h in ctx["pl_houses"].items() if p in bens and h in upas)
    present = count >= 2
    return YogaHit("Vasumati (benefics in Moon's upachayas)", present, 0.74 if present else 0.0, {"count": count}, {}, ("wealth","moon"))

def _dhana_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    L2, L11, L5 = ctx["house_lords"][2], ctx["house_lords"][11], ctx["house_lords"][5]
    combos = [(L2,L11,"2&11"), (L2,L5,"2&5"), (L5,L11,"5&11")]
    hits = []
    for a,b,lab in combos:
        if a == b: continue
        ha, hb = ctx["pl_houses"].get(a), ctx["pl_houses"].get(b)
        if not (ha and hb): continue
        same = _sign_index(ctx["pl_lons"][a]) == _sign_index(ctx["pl_lons"][b])
        assoc = same or (_is_kendra(ha) and _is_kendra(hb)) or _has_graha_drishti(a, _sign_index(ctx["pl_lons"][a]), _sign_index(ctx["pl_lons"][b]))
        if assoc:
            hits.append({"pair": lab, "lords": (a,b), "houses": (ha,hb), "same_sign": same})
    present = bool(hits)
    return YogaHit("Dhana (2/11[/5] association)", present, 0.78 if present else 0.0, {"pairs": hits}, {}, ("wealth","association"))

def _saraswati_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    trio = ("Mercury","Venus","Jupiter")
    ok_h = all(_is_kendra(ctx["pl_houses"].get(g,0)) or _is_trikona(ctx["pl_houses"].get(g,0)) for g in trio)
    ok_d = all(_dignity(g, _sign_index(ctx["pl_lons"][g])) in ("own","exalted") for g in trio)
    present = ok_h and ok_d
    levels = {"house": {g: ctx["pl_houses"].get(g) for g in trio},
              "sign":  {g: _dignity(g, _sign_index(ctx["pl_lons"][g])) for g in trio}}
    return YogaHit("Saraswati", present, 0.84 if present else 0.0, levels, {}, ("education","speech"))

def _lakshmi_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    LL = ctx["house_lords"][1]; L9 = ctx["house_lords"][9]
    hL, h9 = ctx["pl_houses"].get(LL), ctx["pl_houses"].get(L9)
    if not (hL and h9): return YogaHit("Lakshmi", False, 0.0, {}, {}, ("wealth","fortune"))
    strongL = _is_kendra(hL) or _is_trikona(hL); strong9 = _is_kendra(h9) or _is_trikona(h9)
    same = _sign_index(ctx["pl_lons"][LL]) == _sign_index(ctx["pl_lons"][L9])
    assoc = same or (_is_kendra(hL) and _is_kendra(h9))
    present = strongL and strong9 and assoc
    levels = {"lords":{"LL":LL,"L9":L9}, "houses":{"LL":hL,"L9":h9}, "assoc":{"same_sign":same, "mutual_kendra": _is_kendra(hL) and _is_kendra(h9)}}
    return YogaHit("Lakshmi", present, 0.83 if present else 0.0, levels, {}, ("wealth","fortune"))

def _kemadruma_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> YogaHit:
    h_m = ctx["pl_houses"].get("Moon", 1)
    h12 = ((h_m + 10 - 1) % 12) + 1; h2 = ((h_m + 1 - 1) % 12) + 1
    flank = [p for p,h in ctx["pl_houses"].items() if p in _PLANETS_MAIN and h in (h12,h2) and p != "Moon"]
    canceled = any(_is_kendra(_house_delta(h_m, ctx["pl_houses"].get(p,0)) or 0) for p in _PLANETS_MAIN if p != "Moon")
    present = (len(flank) == 0) and (not canceled)
    return YogaHit("Kemadruma (Moon isolated)", present, 0.7 if present else 0.0, {"flanking_planets": flank, "kendra_from_moon_present": canceled}, {}, ("moon","dosha"))

def _kala_sarpa_rule(ctx: Dict[str, Any], _opts: Dict[str, Any]) -> List[YogaHit]:
    rahu = ctx["pl_lons"].get("Rahu"); ketu = ctx["pl_lons"].get("Ketu")
    if rahu is None or ketu is None:
        return [YogaHit("Kala Sarpa", False, 0.0, {}, {}, ("nodal","dosha")),
                YogaHit("Kala Amrita", False, 0.0, {}, {}, ("nodal","dosha"))]
    def between_rahu_ketu(l: float) -> bool:
        d = (_norm360(l - rahu));  return 0 < d < 180.0
    def between_ketu_rahu(l: float) -> bool:
        d = (_norm360(l - ketu));  return 0 < d < 180.0
    planets7 = [ctx["pl_lons"][p] for p in _PLANETS_MAIN]
    all_rk = all(between_rahu_ketu(L) for L in planets7); all_kr = all(between_ketu_rahu(L) for L in planets7)
    sarpa, amrita = all_rk, all_kr
    return [YogaHit("Kala Sarpa", sarpa, 0.68 if sarpa else 0.0, {}, {}, ("nodal","dosha")),
            YogaHit("Kala Amrita", amrita, 0.68 if amrita else 0.0, {}, {}, ("nodal","dosha"))]

# ─────────────────────────────────────────────────────────────────────
# Varga-aware scoring (quick bumps)
# ─────────────────────────────────────────────────────────────────────
def _score_with_vargas(hit: YogaHit, ctx: Dict[str, Any], strengthen_on: Tuple[str, ...]) -> YogaHit:
    if not hit.present: return hit
    bump = 0.0; vg = ctx.get("vargottama", {})
    if "D9" in strengthen_on:
        if isinstance(hit.levels.get("lords"), dict):
            if any(vg.get(p, False) for p in hit.levels["lords"].values()): bump += 0.04
        elif isinstance(hit.details.get("planet"), str):
            if vg.get(hit.details["planet"], False): bump += 0.04
        else:
            bump += 0.02
    if "D10" in strengthen_on: bump += 0.01
    sc = min(0.99, hit.score + bump) if hit.present else 0.0
    return YogaHit(hit.name, hit.present, sc, {**hit.levels, "varga": {"boost_keys": strengthen_on}}, hit.details, hit.tags)

# ─────────────────────────────────────────────────────────────────────
# Registry build
# ─────────────────────────────────────────────────────────────────────
def _ensure_registry() -> None:
    global _BUILT
    if _BUILT:
        return
    _BUILT = True
    _RULES.clear()

    # — Mahapurusha (5) —
    name_map = {"Mars": "Ruchaka", "Mercury": "Bhadra", "Jupiter": "Hamsa", "Venus": "Malavya", "Saturn": "Sasa"}
    for p in ("Mars", "Mercury", "Jupiter", "Venus", "Saturn"):
        register_yoga(name_map[p], _mahapurusha_rule_for(p, name_map[p]), tags=("mahapurusha", "strength", "popular"))

    # Core popular yogas
    register_yoga("Gaja-Kesari",            _gaja_kesari_rule, tags=("moon", "strength", "popular"))
    register_yoga("Chandra-Mangala",        _chandra_mangala_rule, tags=("moon", "wealth", "popular"))
    register_yoga("Adhi",                   _adhi_rule, tags=("moon", "benefic"))
    register_yoga("Durudhara (Moon flanked)", _durudhara_rule, tags=("moon", "kartari"))

    # Sun flankers
    register_yoga("Veshi/Voshi/Ubhayachari", _veshi_voshi_ubhay_rule, tags=("sun", "kartari"))

    # Kartari
    register_yoga("Kartari around Lagna",   _kartari_rule("Lagna"), tags=("kartari", "lagna"))
    register_yoga("Kartari around Moon",    _kartari_rule("Moon"),  tags=("kartari", "moon"))
    register_yoga("Kartari around Sun",     _kartari_rule("Sun"),   tags=("kartari", "sun"))

    # Raja family
    register_yoga("Raja (k–t association)", _raja_rule, tags=("raja", "association"))
    register_yoga("Dharma-Karmadhipati",    _dharma_karmadhipati_rule, tags=("raja", "dk"))
    register_yoga("Parivartana (exchange)", _parivartana_rule, tags=("raja", "exchange"))
    register_yoga("Neecha-bhanga (composite)", _neechabhanga_rule, tags=("raja", "cancellation"))
    register_yoga("Vipareeta Raja (tri-dusthana lords)", _vipareeta_raja_rule, tags=("raja", "vipareeta"))

    # Wealth / career
    register_yoga("Amala from Lagna",       _amala_rule("Lagna"), tags=("amala", "career"))
    register_yoga("Amala from Moon",        _amala_rule("Moon"),  tags=("amala", "career", "moon"))
    register_yoga("Chatussagara",           _chatussagara_rule, tags=("kendras", "strength"))
    register_yoga("Vasumati",               _vasumati_rule, tags=("moon", "wealth"))
    register_yoga("Dhana (2/11[/5] association)", _dhana_rule, tags=("wealth", "association"))
    register_yoga("Saraswati",              _saraswati_rule, tags=("education", "speech"))
    register_yoga("Lakshmi",                _lakshmi_rule, tags=("wealth", "fortune"))

    # Dosha / nodal frameworks
    register_yoga("Kemadruma (Moon isolated)", _kemadruma_rule, tags=("moon", "dosha"))
    register_yoga("Kala Sarpa / Amrita",       _kala_sarpa_rule, tags=("nodal", "dosha"))

# ─────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────
def list_registered_yogas() -> List[Dict[str, Any]]:
    _ensure_registry()
    out = []
    for k, v in _RULES.items():
        out.append({"name": k, "enabled": v["enabled"], "tags": list(v["tags"]), "meta": v["meta"]})
    return sorted(out, key=lambda r: r["name"].lower())

def enable_yogas(keys: Iterable[str]) -> None:
    _ensure_registry()
    keys = list(keys)
    by_tag = {k for k in keys if str(k).startswith("tag:")}
    names = set(keys) - by_tag
    for nm in names:
        if nm in _RULES: _RULES[nm]["enabled"] = True
    for t in by_tag:
        tag = str(t).split(":", 1)[1]
        for _, rec in _RULES.items():
            if tag in rec["tags"]: rec["enabled"] = True

def disable_yogas(keys: Iterable[str]) -> None:
    _ensure_registry()
    keys = list(keys)
    by_tag = {k for k in keys if str(k).startswith("tag:")}
    names = set(keys) - by_tag
    for nm in names:
        if nm in _RULES: _RULES[nm]["enabled"] = False
    for t in by_tag:
        tag = str(t).split(":", 1)[1]
        for _, rec in _RULES.items():
            if tag in rec["tags"]:
                rec["enabled"] = False

# ─────────────────────────────────────────────────────────────────────
# Public orchestrator (Mode-C only)
# ─────────────────────────────────────────────────────────────────────
def compute_yogas(
    payload: Dict[str, Any],
    *,
    ayanamsa: str | float = "lahiri",
    house_system: str = "placidus",
    sign_lord_variant: str = "classical",
    chandra_mangala_by_sign: bool = False,
    conj_orb_deg: float = 6.0,
    gajakesari_include_same_house: bool = False,
    include_mooltrikona_in_mahapurusha: bool = False,
    use_vargas_for_scoring: bool = True,
    varga_keys_for_boost: Tuple[str, ...] = ("D9","D10"),
    include_arudha_notes: bool = False,  # reserved
    enable_catalog_tags: Tuple[str, ...] = (),
    disable_catalog_tags: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    _ensure_registry()

    warnings: List[str] = []
    try:
        date, time, tz, lat_f, lon_f = _require_birth_details(payload)
    except Exception as e:
        return {"ok": False, "error": str(e), "yogas": [], "warnings": []}

    # Timescales
    try:
        jd_tt, jd_ut1, warns_ts = _timescales_from_civil(date, time, tz)
        warnings.extend(warns_ts)
    except Exception as e:
        return {"ok": False, "error": f"timescales_unavailable:{e}", "yogas": [], "warnings": warnings}

    # Ayanāṁśa
    ay_key, ay_deg, warns_ay = _ayanamsa_cached(int(round(jd_tt * 1e6)), str(ayanamsa))
    warnings.extend(list(warns_ay))

    # Houses
    try:
        hp = _compute_houses_payload(lat_f, lon_f, jd_tt, jd_ut1, house_system)
    except Exception as e:
        return {"ok": False, "error": f"houses_unavailable:{e}", "yogas": [], "warnings": warnings}
    cusps   = list(hp["cusps_deg"])
    asc_trop = float(hp["asc_deg"])
    asc_sid  = _norm360(asc_trop - ay_deg)
    lagna_sign = _sign_index(asc_sid)

    # Longitudes (sidereal) & houses
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "yogas": [], "warnings": warnings}
    try:
        pl_lons = _sidereal_longitudes(jd_tt, list(_PLANETS_ALL), ay_deg)
    except Exception as e:
        return {"ok": False, "error": f"ephemeris_unavailable:{e}", "yogas": [], "warnings": warnings}

    if not _ASSIGN_OK:
        return {"ok": False, "error": "assign_houses_unavailable", "yogas": [], "warnings": warnings}
    pl_houses = _house_map_for_planets({k: v for k, v in pl_lons.items() if k in _PLANETS_ALL}, cusps)

    # Lords, benefics
    sign_lords  = _sign_lords(sign_lord_variant)
    house_lords = {i: sign_lords[(lagna_sign + (i-1)) % 12] for i in range(1,13)}
    benefics    = _benefic_set(jd_tt, ay_key)

    # Vargas (optional)
    vargottama, d9_signs, d10_signs, warns_v = _varga_context(pl_lons, ay_key, use_vargas_for_scoring, varga_keys_for_boost)
    warnings.extend(warns_v)

    # Build evaluation context
    ctx = {
        "jd_tt": jd_tt, "ay_key": ay_key, "ay_deg": ay_deg,
        "lat": lat_f, "lon": lon_f,
        "asc_sid": asc_sid, "lagna_sign": lagna_sign,
        "cusps": cusps,
        "pl_lons": pl_lons, "pl_houses": pl_houses,
        "sign_lords": sign_lords, "house_lords": house_lords,
        "benefics": benefics,
        "vargottama": vargottama, "d9_signs": d9_signs, "d10_signs": d10_signs,
    }

    # Per-call options dict for rules
    opts = {
        "chandra_mangala_by_sign": bool(chandra_mangala_by_sign),
        "conj_orb_deg": float(conj_orb_deg),
        "gajakesari_include_same_house": bool(gajakesari_include_same_house),
        "mahapurusha_include_mooltrikona": bool(include_mooltrikona_in_mahapurusha),
        "raja_conj_only": False,  # reserved knob
    }
    if opts["mahapurusha_include_mooltrikona"] and not callable(_is_mt_pos):
        warnings.append("mahapurusha_mooltrikona_option_ignored_no_checker")

    # Per-call tag gating WITHOUT mutating registry state
    enable_tags = tuple(enable_catalog_tags or ())
    disable_tags = set(disable_catalog_tags or ())
    def _should_run(name: str, rec: Dict[str, Any]) -> bool:
        if not rec.get("enabled", True):
            return False
        rtags = set(rec.get("tags", ()))
        if enable_tags:
            if not (rtags & set(enable_tags)):
                return False
        if rtags & disable_tags:
            return False
        return True

    # Evaluate rules
    hits: List[YogaHit] = []
    for name, rec in _RULES.items():
        if not _should_run(name, rec):
            continue
        fn: RuleFn = rec["fn"]
        try:
            out = fn(ctx, opts)
            if isinstance(out, list): hits.extend(out)
            elif isinstance(out, YogaHit): hits.append(out)
        except Exception:
            # robust: skip faulty rule
            continue

    # Optional varga-aware scoring bumps
    final_hits: List[YogaHit] = []
    if use_vargas_for_scoring:
        for h in hits:
            final_hits.append(_score_with_vargas(h, ctx, varga_keys_for_boost))
    else:
        final_hits = hits

    yogas_list = [{
        "name": h.name, "present": bool(h.present), "score": float(h.score),
        "levels": h.levels, "details": h.details, "tags": list(h.tags)
    } for h in final_hits]

    context = {
        "ayanamsa": {"key": ctx["ay_key"], "deg": float(ctx["ay_deg"]) if isinstance(ctx["ay_deg"], (int,float)) and math.isfinite(float(ctx["ay_deg"])) else None},
        "ascendant_sidereal_deg": float(ctx["asc_sid"]),
        "lagna_sign_index": int(ctx["lagna_sign"]),
        "house_lords": ctx["house_lords"],
        "planet_houses": {k: int(v) for k, v in ctx["pl_houses"].items()},
        "planet_signs": {k: int(_sign_index(ctx["pl_lons"][k])) for k in ctx["pl_lons"]},
        "vargottama": ctx.get("vargottama", {}),
        "varga_signs": {"D9": ctx.get("d9_signs", {}), "D10": ctx.get("d10_signs", {})},
        "house_system": house_system,
        "lat": ctx.get("lat"), "lon": ctx.get("lon"),
    }

    return {"ok": True, "yogas": yogas_list, "context": context, "warnings": list(dict.fromkeys(warnings))}

# ─────────────────────────────────────────────────────────────────────
# Convenience: export list + toggles
# ─────────────────────────────────────────────────────────────────────
__all__ = [
    "compute_yogas",
    "list_registered_yogas",
    "enable_yogas",
    "disable_yogas",
]
