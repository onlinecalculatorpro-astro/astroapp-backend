# app/core/vedic_predictive.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Literal, Optional
import math
from datetime import datetime

from app.core.common_predictive import norm360, sign_index, angdiff, compute_houses, timescales_from_civil
from app.core.ephem_singleton import TS, PLANETS

try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    _EPH_OK = False
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore

# ── Dasha (Vimshottari) ─────────────────────────────────────────────────────
_VIM_ORDER = ["ketu","venus","sun","moon","mars","rahu","jupiter","saturn","mercury"]
_VIM_YEARS = {"ketu":7,"venus":20,"sun":6,"moon":10,"mars":7,"rahu":18,"jupiter":16,"saturn":19,"mercury":17}
_NAK_WIDTH = 360.0/27.0

def _nirayana(lon_tropical: float, ayanamsa_deg: float) -> float:
    return norm360(lon_tropical - ayanamsa_deg)

def _nak_index(nirayana_lon: float) -> int:
    return int(math.floor(nirayana_lon / _NAK_WIDTH))

def _nak_lord(idx: int) -> str:
    return _VIM_ORDER[idx % 9]

def _cycle_from(lord: str) -> List[str]:
    i = _VIM_ORDER.index(lord)
    return _VIM_ORDER[i:] + _VIM_ORDER[:i]

@dataclass
class DashaPeriod:
    start_jd_tt: float
    end_jd_tt: float
    level: int
    lord: str
    parent_chain: Tuple[str, ...]
    meta: Dict[str, Any]

def vimsottari_dasha(*, birth_jd_tt: float, moon_lon_tropical_deg: float,
                     ayanamsa_deg: float = 0.0, levels: int = 3, span_years: float = 120.0) -> List[DashaPeriod]:
    if levels < 1: levels = 1
    if levels > 3: levels = 3
    moon_nir = _nirayana(moon_lon_tropical_deg, ayanamsa_deg)
    idx = _nak_index(moon_nir)
    lord0 = _nak_lord(idx)
    pos_in_nak = moon_nir - idx * _NAK_WIDTH
    rem_frac = max(0.0, min(1.0, (_NAK_WIDTH - pos_in_nak) / _NAK_WIDTH))
    def y2d(y: float) -> float: return y * 365.2425
    cycle = _cycle_from(lord0)
    t = birth_jd_tt
    periods: List[DashaPeriod] = []
    for i, lord in enumerate(cycle):
        years = float(_VIM_YEARS[lord])
        frac = rem_frac if i == 0 else 1.0
        start = t; end = start + y2d(years * frac)
        periods.append(DashaPeriod(start, end, 1, lord, (lord,), {"years": years, "frac": frac}))
        t = end
        if (end - birth_jd_tt) >= y2d(span_years) + 1e-9:
            break
    def expand(parent: DashaPeriod, level: int) -> List[DashaPeriod]:
        if level > levels: return []
        subs = _cycle_from(parent.lord)
        out: List[DashaPeriod] = []
        total_days = parent.end_jd_tt - parent.start_jd_tt
        t0 = parent.start_jd_tt
        for lord in subs:
            frac = _VIM_YEARS[lord] / 120.0
            dur = total_days * frac
            seg = DashaPeriod(t0, t0 + dur, level, lord, parent.parent_chain + (lord,) if level>1 else (parent.lord, lord), {"frac": frac})
            out.append(seg); t0 += dur
        return out
    result = list(periods)
    if levels >= 2:
        b2: List[DashaPeriod] = []
        for p in periods: b2.extend(expand(p, 2))
        result.extend(b2)
        if levels >= 3:
            b3: List[DashaPeriod] = []
            for p in b2: b3.extend(expand(p, 3))
            result.extend(b3)
    result.sort(key=lambda d: (d.start_jd_tt, d.level))
    return result

def predict_dasha_periods(*, natal_chart: Dict[str, Any], start_date: datetime, end_date: datetime,
                          dasha_system: str = "vimshottari", include_antardasha: bool = True) -> Dict[str, Any]:
    if dasha_system.lower() not in ("vimshottari", "vimsottari", "vimshottari"):
        return {"ok": False, "error": "unsupported_dasha"}
    # Resolve natal JD_TT (either provided or via timescales)
    if "jd_tt" in natal_chart:
        birth_jd_tt = float(natal_chart["jd_tt"])
    else:
        from datetime import datetime as _dt
        d = _dt.strptime(str(natal_chart.get("date")), "%Y-%m-%d").date()
        t = str(natal_chart.get("time") or "00:00:00")
        tz = str(natal_chart.get("place_tz") or natal_chart.get("timezone") or "UTC")
        ts = timescales_from_civil(d.strftime("%Y-%m-%d"), t, tz)
        birth_jd_tt = float(ts["jd_tt"])

    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore
    mm_rows = ep.ecliptic_longitudes(birth_jd_tt, ["Moon"]).get("results", [])
    moon_lon_trop = float(mm_rows[0]["longitude"]) if mm_rows else 0.0

    periods = vimsottari_dasha(
        birth_jd_tt=birth_jd_tt, moon_lon_tropical_deg=moon_lon_trop,
        ayanamsa_deg=float(natal_chart.get("ayanamsa_deg", 0.0)),
        levels=(3 if include_antardasha else 1), span_years=120.0,
    )
    def jd_to_iso(j: float) -> str:
        unix = (j - 2440587.5) * 86400.0
        return datetime.utcfromtimestamp(unix).isoformat() + "Z"
    out = [{
        "start_jd_tt": p.start_jd_tt, "end_jd_tt": p.end_jd_tt,
        "start_date": jd_to_iso(p.start_jd_tt), "end_date": jd_to_iso(p.end_jd_tt),
        "level": p.level, "mahadasha_lord": p.parent_chain[0] if p.parent_chain else p.lord,
        "chain": list(p.parent_chain), "meta": p.meta,
    } for p in periods]
    return {"ok": True, "periods": out, "system": "vimshottari"}

# ── Vargas ───────────────────────────────────────────────────────────────────
EXALT_SIGN = {"sun":0,"moon":1,"mars":9,"mercury":5,"jupiter":3,"venus":11,"saturn":6}
OWN_SIGNS = {"sun":[4],"moon":[3],"mars":[0,7],"mercury":[2,5],"jupiter":[8,11],"venus":[1,6],"saturn":[9,10]}

def _to_nirayana(lon: float, zodiac_mode: str, ayanamsa_deg: float) -> float:
    return norm360(lon - (ayanamsa_deg if zodiac_mode.startswith("sidereal") else 0.0))

def _hora_d2_sign(lon_nir: float) -> int:
    s = sign_index(lon_nir); deg_in_sign = (lon_nir % 30.0); odd = (s % 2 == 0)
    return (4 if deg_in_sign < 15.0 else 3) if odd else (3 if deg_in_sign < 15.0 else 4)

def _drekkana_d3_sign(lon_nir: float) -> int:
    s = sign_index(lon_nir); slot = int((lon_nir % 30.0) // 10.0); odd = (s % 2 == 0)
    start = s if odd else (s + 2) % 12
    return (start + 4 * slot) % 12

def _navamsa_d9_sign(lon_nir: float) -> int:
    s = sign_index(lon_nir); part = int((lon_nir % 30.0) // (30.0/9.0))
    movable={0,3,6,9}; fixed={1,4,7,10}
    base = s if s in movable else ((s + 8) % 12 if s in fixed else (s + 4) % 12)
    return (base + part) % 12

def _dasamsa_d10_sign(lon_nir: float) -> int:
    s = sign_index(lon_nir); part = int((lon_nir % 30.0) // 3.0); odd = (s % 2 == 0)
    base = s if odd else (s + 8) % 12
    return (base + part) % 12

def _dvadasamsa_d12_sign(lon_nir: float) -> int:
    s = sign_index(lon_nir); part = int((lon_nir % 30.0) // (30.0/12.0))
    return (s + part) % 12

def compute_vargas_for_point(*, lon_deg: float, zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
                             ayanamsa_deg: float = 0.0, include: Iterable[str] = ("D1","D2","D3","D9","D10","D12")) -> Dict[str, int]:
    L = _to_nirayana(lon_deg, zodiac_mode, ayanamsa_deg)
    d: Dict[str, int] = {}
    if "D1"  in include: d["D1"]  = sign_index(L)
    if "D2"  in include: d["D2"]  = _hora_d2_sign(L)
    if "D3"  in include: d["D3"]  = _drekkana_d3_sign(L)
    if "D9"  in include: d["D9"]  = _navamsa_d9_sign(L)
    if "D10" in include: d["D10"] = _dasamsa_d10_sign(L)
    if "D12" in include: d["D12"] = _dvadasamsa_d12_sign(L)
    return d

def compute_vargas(*, points_deg: Dict[str, float], zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
                   ayanamsa_deg: float = 0.0, include: Iterable[str] = ("D1","D2","D3","D9","D10","D12")) -> Dict[str, Dict[str, int]]:
    return {name: compute_vargas_for_point(lon_deg=lon, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg, include=include)
            for name, lon in points_deg.items()}

# ── Yogas ────────────────────────────────────────────────────────────────────
def house_index_for_longitude(cusps_deg: List[float], lon_deg: float) -> int:
    if len(cusps_deg) != 12:
        raise ValueError("cusps_deg must be 12 values")
    c = [norm360(x) for x in cusps_deg]
    lam = norm360(lon_deg)
    for i in range(12):
        start = c[i]; end = norm360(c[(i + 1) % 12])
        span = norm360(end - start); delta = norm360(lam - start)
        if delta < span or span == 0.0: return i + 1
    return 12

def is_kendra(h: int) -> bool: return h in (1,4,7,10)

def in_own_or_exaltation(planet: str, sign_idx: int) -> bool:
    p = planet.lower()
    if EXALT_SIGN.get(p, -1) == sign_idx: return True
    return sign_idx in OWN_SIGNS.get(p, [])

def detect_panch_mahapurusha(points_deg: Dict[str, float], cusps_deg: List[float]) -> List[Dict[str, Any]]:
    yogas: List[Dict[str, Any]] = []
    for p, name in [("mars","Ruchaka"), ("mercury","Bhadra"), ("jupiter","Hamsa"), ("venus","Malavya"), ("saturn","Shasha")]:
        if p not in points_deg: continue
        lon = points_deg[p]; s = sign_index(lon); h = house_index_for_longitude(cusps_deg, lon)
        if is_kendra(h) and in_own_or_exaltation(p, s):
            yogas.append({"yoga": name, "planet": p, "house": h, "sign_index": s})
    return yogas

def detect_gajakesari(points_deg: Dict[str, float], cusps_deg: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if "moon" in points_deg and "jupiter" in points_deg:
        from_house = house_index_for_longitude(cusps_deg, points_deg["moon"])
        to_house   = house_index_for_longitude(cusps_deg, points_deg["jupiter"])
        diff = ((to_house - from_house) % 12) or 12
        if diff in (1,4,7,10):
            out.append({"yoga": "Gajakesari", "from": "Moon", "to": "Jupiter", "offset_houses": diff})
    return out

def detect_chandra_mangal(points_deg: Dict[str, float], max_orb_deg: float = 8.0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if "moon" in points_deg and "mars" in points_deg:
        sep = abs(angdiff(points_deg["moon"], points_deg["mars"]))
        if sep <= max_orb_deg:
            out.append({"yoga": "Chandra-Mangal", "orb_deg": sep})
    return out

def detect_parivartana(points_deg: Dict[str, float]) -> List[Dict[str, Any]]:
    owner: Dict[int, str] = {}
    for pl, signs in OWN_SIGNS.items():
        for s in signs: owner[s] = pl
    loc_owner: Dict[str, str] = {}
    for pl, lon in points_deg.items():
        s = sign_index(lon); loc_owner[pl] = owner.get(s, "")
    checked = set(); out: List[Dict[str, Any]] = []
    for a, lord_b in loc_owner.items():
        if not lord_b or lord_b == a: continue
        if (a, lord_b) in checked or (lord_b, a) in checked: continue
        if loc_owner.get(lord_b) == a:
            out.append({"yoga": "Parivartana", "pair": (a, lord_b)})
            checked.add((a, lord_b))
    return out

def detect_yogas(*, points_deg: Dict[str, float], cusps_deg: List[float],
                 include: Iterable[str] = ("panch_mahapurusha","gajakesari","chandra_mangal","parivartana"),
                 orbs: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if "panch_mahapurusha" in include: out.extend(detect_panch_mahapurusha(points_deg, cusps_deg))
    if "gajakesari" in include: out.extend(detect_gajakesari(points_deg, cusps_deg))
    if "chandra_mangal" in include: out.extend(detect_chandra_mangal(points_deg, max_orb_deg=(orbs or {}).get("chandra_mangal", 8.0)))
    if "parivartana" in include: out.extend(detect_parivartana(points_deg))
    out.sort(key=lambda x: (x.get("yoga",""), x.get("planet",""), tuple(x.get("pair",()))))
    return out
