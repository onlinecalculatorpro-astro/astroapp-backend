# app/core/predictive.py
from __future__ import annotations

"""
Predictive Toolkit — Transits • Dasha • Varga • Yoga • Validation (single file)

Additions in this revision
- Transit performance:
  • In-scan memo cache for lon(body, t) → avoids duplicate ephemeris queries
  • Reuse of previous window’s endpoints (t1 → next t0) halves fetches
  • Zero-finder reuses cached longitudes; bisection no longer re-queries same times
- Validation for time series:
  • permutation_pvalue_corr(..., perm_mode=...) with:
      - "iid": classic shuffle (optionally stratified; previous default)
      - "within": shuffle labels within strata/blocks (preserves block means)
      - "circular": circular shifts within each group using time order (preserves serial correlation)
  • evaluate_univariate(..., perm_mode=..., group_by=..., use_time=True) to pass groups & times

Empirical stance:
Traditional techniques are exposed exactly and reproducibly, but **no claim** of
predictive validity is made. Use the validation utilities here to audit any
hypothesis out-of-sample with multiple-testing control.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, Callable
import math
import random
import logging

# ───────────────────── repo-local precise backends ─────────────────────
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig, rows_to_maps, get_node_longitude
except Exception as e:
    raise RuntimeError(f"predictive: ephemeris backend unavailable: {e}") from e

try:
    from app.core.house import compute_houses_with_policy as _compute_houses_policy
    _HAS_POLICY = True
except Exception:
    _HAS_POLICY = False
    from app.core.houses import asc_mc_houses as _asc_mc_houses

try:
    from app.core.validators import resolve_timescales_from_civil_erfa as _ts_resolve
except Exception:
    _ts_resolve = None

log = logging.getLogger(__name__)

# =============================================================================
# Common math helpers
# =============================================================================

TAU = 360.0
EPS = 1e-12

def norm360(x: float) -> float:
    r = x % TAU
    return r + TAU if r < 0.0 else r

def wrap180(x: float) -> float:
    v = ((x + 180.0) % 360.0) - 180.0
    return v if v != -180.0 else 180.0

def angdiff(a: float, b: float) -> float:
    return wrap180(a - b)

def sign_index(lon_deg: float) -> int:
    return int(math.floor(norm360(lon_deg) / 30.0)) % 12

def is_finite(*xs: float) -> bool:
    return all(math.isfinite(float(x)) for x in xs)

# =============================================================================
# ASPECT ENGINE
# =============================================================================

AspectKind = Literal["zodiacal", "antiscia", "contra-antiscia", "parallel", "contra-parallel"]

@dataclass(frozen=True)
class AspectSpec:
    name: str
    angle: float
    orb_deg: float
    kind: AspectKind = "zodiacal"
    weight: float = 1.0

MAJOR_ASPECTS: Tuple[AspectSpec, ...] = (
    AspectSpec("Conjunction", 0.0, 8.0),
    AspectSpec("Opposition", 180.0, 8.0),
    AspectSpec("Trine", 120.0, 6.0),
    AspectSpec("Square", 90.0, 6.0),
    AspectSpec("Sextile", 60.0, 4.0),
)

MINOR_ASPECTS: Tuple[AspectSpec, ...] = (
    AspectSpec("Quincunx", 150.0, 3.0),
    AspectSpec("Semisextile", 30.0, 2.0),
    AspectSpec("Semisquare", 45.0, 2.0),
    AspectSpec("Sesquisquare", 135.0, 2.0),
)

def antiscia_longitude(lon: float) -> float:
    return norm360(180.0 - lon)

def contra_antiscia_longitude(lon: float) -> float:
    return norm360(360.0 - lon)

@dataclass(frozen=True)
class ParallelSpec:
    name: str
    orb_deg: float
    kind: AspectKind

DEFAULT_PARALLELS: Tuple[ParallelSpec, ...] = (
    ParallelSpec("Parallel", 1.0, "parallel"),
    ParallelSpec("Contra-Parallel", 1.0, "contra-parallel"),
)

def _zodiacal_separation(a: float, b: float, target: float) -> float:
    return wrap180(angdiff(a, b) - target)

def _match_antiscia(a: float, b: float, spec: AspectSpec) -> Tuple[bool, float]:
    b_image = antiscia_longitude(b) if spec.kind == "antiscia" else contra_antiscia_longitude(b)
    d = wrap180(a - b_image)
    return (abs(d) <= spec.orb_deg), d

# =============================================================================
# TRANSITS (with caching & reduced ephemeris calls)
# =============================================================================

@dataclass
class TransitEvent:
    jd_tt: float
    body: str
    target: str
    aspect: str
    kind: AspectKind
    separation_deg: float
    applying: bool
    exact: bool
    meta: Dict[str, Any]

class TransitEngine:
    """
    Performance-aware transit finder:
    - One batch call per boundary (t0, t1) per step for all movers
    - Carry forward lons(t1) → next-step lons(t0)
    - In-scan cache for lon(body, t) during refinement; no duplicate lookups
    """

    def __init__(
        self,
        *,
        ephem: Optional[EphemerisAdapter] = None,
        frame: str = "ecliptic-of-date",
        topocentric: bool = False,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        elevation_m: Optional[float] = None,
    ):
        self.ephem = ephem or EphemerisAdapter(EphemConfig(frame=frame))
        self.frame = frame
        self.obs = dict(
            topocentric=bool(topocentric),
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m,
        )

    # ---------- ephemeris wrappers ----------
    def _lon_map(self, jd_tt: float, names: List[str]) -> Dict[str, float]:
        r = self.ephem.ecliptic_longitudes(jd_tt, names, **self.obs)
        return rows_to_maps(r.get("results", []))["longitudes"]

    # ---------- root finding ----------
    @staticmethod
    def _refine_zero(f, t0, t1, *, max_iter=32, tol_days=1e-6) -> float:
        f0 = f(t0); f1 = f(t1)
        if not (math.isfinite(f0) and math.isfinite(f1)):
            return (t0 + t1) / 2.0
        if f0 == 0.0: return t0
        if f1 == 0.0: return t1
        if f0 * f1 > 0.0:
            return (t0 + t1) / 2.0
        a, b = (t0, t1)
        fa, fb = (f0, f1)
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = f(m)
            if abs(fm) == 0.0 or (b - a) <= tol_days:
                return m
            if fa * fm <= 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    # ---------- scans ----------
    def scan_aspects(
        self,
        *,
        jd_start_tt: float,
        jd_end_tt: float,
        movers: List[str],
        targets: Dict[str, float],
        aspects: Iterable[AspectSpec] = MAJOR_ASPECTS,
        step_minutes: float = 30.0,
        include_antiscia: bool = False,
        antiscia_orb_deg: float = 2.0,
    ) -> List[TransitEvent]:
        if jd_end_tt <= jd_start_tt:
            return []

        # prepare aspect set
        asp_list: List[AspectSpec] = list(aspects)
        if include_antiscia:
            asp_list.append(AspectSpec("Antiscia", 0.0, antiscia_orb_deg, kind="antiscia"))
            asp_list.append(AspectSpec("Contra-Antiscia", 0.0, antiscia_orb_deg, kind="contra-antiscia"))

        dt = step_minutes / (60.0 * 24.0)
        events: List[TransitEvent] = []

        # in-scan cache: {(name, t): lon_deg}
        lon_cache: Dict[Tuple[str, float], float] = {}

        def _lon_cached(name: str, t: float) -> float:
            key = (name, float(t))
            v = lon_cache.get(key)
            if v is not None:
                return v
            # fetch single (rare) – normally we batch-fill below
            v = self._lon_map(t, [name]).get(name)
            if v is None:
                raise RuntimeError(f"no ephemeris for {name}@{t}")
            lon_cache[key] = float(v)
            return lon_cache[key]

        # make separation function using cache
        def make_sep(body: str, target_lon: float, spec: AspectSpec):
            if spec.kind == "zodiacal":
                return lambda t: _zodiacal_separation(_lon_cached(body, t), target_lon, spec.angle)
            elif spec.kind == "antiscia":
                b_image = antiscia_longitude(target_lon)
                return lambda t: wrap180(_lon_cached(body, t) - b_image)
            else:
                b_image = contra_antiscia_longitude(target_lon)
                return lambda t: wrap180(_lon_cached(body, t) - b_image)

        # initial boundary
        t0 = jd_start_tt
        l0 = self._lon_map(t0, movers)
        for m, v in l0.items():
            lon_cache[(m, float(t0))] = float(v)

        while t0 < jd_end_tt - 1e-12:
            t1 = min(t0 + dt, jd_end_tt)
            # compute l1 once
            l1 = self._lon_map(t1, movers)
            for m, v in l1.items():
                lon_cache[(m, float(t1))] = float(v)

            for body in movers:
                lon0 = l0.get(body); lon1 = l1.get(body)
                if lon0 is None or lon1 is None:
                    continue
                for tgt_name, tgt_lon in targets.items():
                    for spec in asp_list:
                        if spec.kind == "zodiacal":
                            s0 = _zodiacal_separation(lon0, tgt_lon, spec.angle)
                            s1 = _zodiacal_separation(lon1, tgt_lon, spec.angle)
                        elif spec.kind == "antiscia":
                            img = antiscia_longitude(tgt_lon)
                            s0 = wrap180(lon0 - img); s1 = wrap180(lon1 - img)
                        else:
                            img = contra_antiscia_longitude(tgt_lon)
                            s0 = wrap180(lon0 - img); s1 = wrap180(lon1 - img)

                        if not (math.isfinite(s0) and math.isfinite(s1)):
                            continue
                        if abs(s0) > 120.0 or abs(s1) > 120.0:
                            continue
                        if s0 == 0.0 or s1 == 0.0 or (s0 * s1) < 0.0:
                            f = make_sep(body, tgt_lon, spec)
                            t_exact = self._refine_zero(f, t0, t1, tol_days=1e-6)
                            lon_now = _lon_cached(body, t_exact)
                            sep = (
                                _zodiacal_separation(lon_now, tgt_lon, spec.angle)
                                if spec.kind == "zodiacal"
                                else wrap180(lon_now - (antiscia_longitude(tgt_lon) if spec.kind == "antiscia" else contra_antiscia_longitude(tgt_lon)))
                            )
                            epsd = 5.0 / (24.0 * 60.0)  # 5 minutes
                            before = f(t_exact - epsd)
                            applying = (abs(before) > abs(sep))
                            events.append(
                                TransitEvent(
                                    jd_tt=float(t_exact),
                                    body=body,
                                    target=tgt_name,
                                    aspect=spec.name,
                                    kind=spec.kind,
                                    separation_deg=float(sep),
                                    applying=bool(applying),
                                    exact=abs(sep) <= 1e-6,
                                    meta={"orb_deg": spec.orb_deg, "angle": spec.angle},
                                )
                            )

            # carry forward: next l0 is current l1 (no extra ephemeris call)
            t0 = t1
            l0 = l1

        events.sort(key=lambda e: (e.jd_tt, e.body, e.target, e.aspect))
        return events

    def find_ingresses(
        self,
        *,
        jd_start_tt: float,
        jd_end_tt: float,
        movers: List[str],
        step_minutes: float = 60.0
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        dt = step_minutes / (60.0 * 24.0)
        t0 = jd_start_tt
        lon_cache: Dict[Tuple[str,float], float] = {}
        l0 = self._lon_map(t0, movers)
        for m, v in l0.items(): lon_cache[(m, t0)] = float(v)
        last_sign: Dict[str, Optional[int]] = {m: sign_index(l0[m]) for m in movers if m in l0}
        while t0 < jd_end_tt - 1e-12:
            t1 = min(t0 + dt, jd_end_tt)
            l1 = self._lon_map(t1, movers)
            for m, v in l1.items(): lon_cache[(m, t1)] = float(v)
            for m in movers:
                if m not in l0 or m not in l1: continue
                s0 = last_sign.get(m)
                s1 = sign_index(l1[m])
                if s0 is None: 
                    last_sign[m] = s1
                elif s1 != s0:
                    edge = (s1 * 30.0)
                    def f(t: float) -> float:
                        # simple cached lon
                        if (m, t) not in lon_cache:
                            lon_cache[(m,t)] = self._lon_map(t, [m]).get(m, float("nan"))
                        return wrap180(lon_cache[(m,t)] - edge)
                    t_exact = self._refine_zero(f, t0, t1, tol_days=1e-6)
                    out.append({"jd_tt": float(t_exact), "body": m, "sign": s1})
                    last_sign[m] = s1
            t0 = t1
            l0 = l1
        out.sort(key=lambda r: (r["jd_tt"], r["body"]))
        return out

    def find_stations(
        self,
        *,
        jd_start_tt: float,
        jd_end_tt: float,
        movers: List[str],
        step_minutes: float = 60.0,
    ) -> List[Dict[str, Any]]:
        dt = step_minutes / (60.0 * 24.0)
        out: List[Dict[str, Any]] = []
        t0 = jd_start_tt
        # cache both endpoints for all movers
        l0 = self._lon_map(t0, movers)
        while t0 < jd_end_tt - 1e-12:
            t1 = min(t0 + dt, jd_end_tt)
            l1 = self._lon_map(t1, movers)
            h = min(0.5 * dt, 0.25 / 24.0)
            for m in movers:
                if m not in l0 or m not in l1: continue
                def vel(t: float) -> float:
                    # estimate velocity using cached endpoints + single fetches only if needed
                    def L(tt: float) -> float:
                        if tt == t0 and m in l0: return l0[m]
                        if tt == t1 and m in l1: return l1[m]
                        return self._lon_map(tt, [m]).get(m, float("nan"))
                    lp = L(t + h); lm = L(t - h)
                    return angdiff(lp, lm) / (2.0 * h)
                v0 = vel(t0); v1 = vel(t1)
                if not (math.isfinite(v0) and math.isfinite(v1)):
                    continue
                if v0 == 0.0 or v1 == 0.0 or (v0 * v1) < 0.0:
                    t_exact = self._refine_zero(vel, t0, t1, tol_days=1e-6)
                    out.append({"jd_tt": float(t_exact), "body": m, "retrograde": vel(t_exact + 1e-4) < 0.0})
            t0 = t1
            l0 = l1
        out.sort(key=lambda r: (r["jd_tt"], r["body"]))
        return out

# =============================================================================
# VIMSOTTARI DASHA, VARGA, YOGAS (unchanged core)
# =============================================================================

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

def vimsottari_dasha(
    *,
    birth_jd_tt: float,
    moon_lon_tropical_deg: float,
    ayanamsa_deg: float = 0.0,
    levels: int = 3,
    span_years: float = 120.0
) -> List[DashaPeriod]:
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

# Varga helpers
EXALT_SIGN = {"sun":0,"moon":1,"mars":9,"mercury":5,"jupiter":3,"venus":11,"saturn":6}
OWN_SIGNS = {
    "sun":[4],"moon":[3],"mars":[0,7],"mercury":[2,5],"jupiter":[8,11],"venus":[1,6],"saturn":[9,10]
}

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

def compute_vargas_for_point(
    *,
    lon_deg: float,
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa_deg: float = 0.0,
    include: Iterable[str] = ("D1","D2","D3","D9","D10","D12"),
) -> Dict[str, int]:
    L = _to_nirayana(lon_deg, zodiac_mode, ayanamsa_deg)
    d: Dict[str, int] = {}
    if "D1" in include: d["D1"] = sign_index(L)
    if "D2" in include: d["D2"] = _hora_d2_sign(L)
    if "D3" in include: d["D3"] = _drekkana_d3_sign(L)
    if "D9" in include: d["D9"] = _navamsa_d9_sign(L)
    if "D10" in include: d["D10"] = _dasamsa_d10_sign(L)
    if "D12" in include: d["D12"] = _dvadasamsa_d12_sign(L)
    return d

def compute_vargas(
    *,
    points_deg: Dict[str, float],
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa_deg: float = 0.0,
    include: Iterable[str] = ("D1","D2","D3","D9","D10","D12"),
) -> Dict[str, Dict[str, int]]:
    return {name: compute_vargas_for_point(lon_deg=lon, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg, include=include)
            for name, lon in points_deg.items()}

# Yogas (selected)
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
        h_moon = house_index_for_longitude(cusps_deg, points_deg["moon"])
        h_jup  = house_index_for_longitude(cusps_deg, points_deg["jupiter"])
        diff = ((h_jup - h_moon) % 12) or 12
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

def detect_yogas(
    *,
    points_deg: Dict[str, float],
    cusps_deg: List[float],
    include: Iterable[str] = ("panch_mahapurusha","gajakesari","chandra_mangal","parivartana"),
    orbs: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    orbs = orbs or {}
    out: List[Dict[str, Any]] = []
    if "panch_mahapurusha" in include: out.extend(detect_panch_mahapurusha(points_deg, cusps_deg))
    if "gajakesari" in include: out.extend(detect_gajakesari(points_deg, cusps_deg))
    if "chandra_mangal" in include: out.extend(detect_chandra_mangal(points_deg, max_orb_deg=orbs.get("chandra_mangal", 8.0)))
    if "parivartana" in include: out.extend(detect_parivartana(points_deg))
    out.sort(key=lambda x: (x.get("yoga",""), x.get("planet",""), tuple(x.get("pair",()))))
    return out

# =============================================================================
# House & timescale helpers
# =============================================================================

def compute_houses(
    *,
    latitude: float,
    longitude: float,
    jd_tt: float,
    jd_ut1: float,
    system: str = "placidus"
) -> Dict[str, Any]:
    if _HAS_POLICY:
        pay = _compute_houses_policy(lat=latitude, lon=longitude, system=system, jd_tt=jd_tt, jd_ut1=jd_ut1, jd_ut=jd_ut1)
        return {"asc": float(pay["asc"]), "mc": float(pay["mc"]), "cusps": [float(x) for x in pay["cusps"]]}
    asc, mc, cusps = _asc_mc_houses(system, latitude, longitude, jd_tt=jd_tt, jd_ut1=jd_ut1, jd_ut=jd_ut1)
    return {"asc": float(asc), "mc": float(mc), "cusps": [float(x) for x in cusps]}

def timescales_from_civil(date_yyyy_mm_dd: str, time_hh_mm_ss: str, place_tz: str) -> Dict[str, float]:
    if _ts_resolve is None:
        raise RuntimeError("Timescale resolver unavailable; pass jd_tt/jd_ut1 directly.")
    from datetime import datetime
    d = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").date()
    return _ts_resolve(d, time_hh_mm_ss, place_tz)

# =============================================================================
# VALIDATION (with time-series–aware permutations)
# =============================================================================

FeatureFn = Callable[[Dict[str, Any], EphemerisAdapter], Dict[str, float]]

def pearson_corr(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0 or len(y) != n:
        return 0.0
    sx = sy = sxx = syy = sxy = 0.0
    for xi, yi in zip(x, y):
        sx += xi; sy += yi
        sxx += xi*xi; syy += yi*yi; sxy += xi*yi
    num = n*sxy - sx*sy
    denx = n*sxx - sx*sx
    deny = n*syy - sy*sy
    if denx <= 0.0 or deny <= 0.0:
        return 0.0
    return num / math.sqrt(denx * deny)

def _groups_from_ids(ids: List[Any]) -> List[List[int]]:
    buckets: Dict[Any, List[int]] = {}
    for i, gid in enumerate(ids):
        buckets.setdefault(gid, []).append(i)
    return list(buckets.values())

def permutation_pvalue_corr(
    x: List[float],
    y: List[int],
    *,
    n_perm: int = 2000,
    strata: Optional[List[Any]] = None,
    perm_mode: Literal["iid","within","circular"] = "iid",
    times: Optional[List[float]] = None,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Two-sided permutation p-value for correlation.
    perm_mode:
      - "iid": shuffle labels across all (or within strata if provided).
      - "within": shuffle labels within each stratum/group (preserve group means).
      - "circular": for each group (from strata), sort by 'times' and apply a random
                    circular shift to the label vector (preserves serial correlation).
    """
    rnd = random.Random(seed)
    y = [int(v) for v in y]
    r_obs = pearson_corr(x, [float(v) for v in y])
    if not math.isfinite(r_obs): return 0.0, 1.0
    if n_perm <= 0: return r_obs, 1.0

    indices = list(range(len(y)))

    # Build grouping
    if strata is None:
        groups = [indices]
    else:
        groups = _groups_from_ids(strata)

    # Optionally precompute orderings per group by time for circular shifts
    order_in_group: Dict[int, List[int]] = {}
    if perm_mode == "circular":
        if times is None:
            # fallback to within-group iid if no times
            perm_mode = "within"
        else:
            # group-local indices sorted by time
            pos = {i: t for i, t in enumerate(times)}
            for gi, g in enumerate(groups):
                order_in_group[gi] = sorted(g, key=lambda i: pos.get(i, 0.0))

    extreme = 0
    abs_obs = abs(r_obs)
    y_work = y[:]

    for _ in range(n_perm):
        if perm_mode == "iid":
            if strata is None:
                rnd.shuffle(y_work)
            else:
                # shuffle within each stratum
                for g in groups:
                    vals = [y_work[i] for i in g]
                    rnd.shuffle(vals)
                    for i, v in zip(g, vals): y_work[i] = v
        elif perm_mode == "within":
            for g in groups:
                vals = [y_work[i] for i in g]
                rnd.shuffle(vals)
                for i, v in zip(g, vals): y_work[i] = v
        else:  # circular
            for gi, g in enumerate(groups):
                ord_idx = order_in_group.get(gi, g[:])
                if not ord_idx: continue
                k = rnd.randrange(len(ord_idx))  # shift amount
                shifted = ord_idx[k:] + ord_idx[:k]
                # write back labels by mapping original order→shifted order
                vals = [y_work[i] for i in ord_idx]
                for i, v in zip(shifted, vals):
                    y_work[i] = v

        r_perm = pearson_corr(x, [float(v) for v in y_work])
        if abs(r_perm) >= abs_obs - 1e-15:
            extreme += 1

    p = (extreme + 1.0) / (n_perm + 1.0)
    return r_obs, p

def bh_fdr(pvals: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0]*m; min_q = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        j = order[-rank]; pi = pvals[j]
        qj = pi * m / (m - rank + 1)
        if qj < min_q: min_q = qj
        q[j] = min_q
    rejected = [qv <= alpha for qv in q]
    return q, rejected

@dataclass
class EvalResult:
    feature: str
    n: int
    effect_r: float
    p_perm: float
    q_fdr: float
    accepted: bool

def evaluate_univariate(
    records: List[Dict[str, Any]],
    feature_fn: FeatureFn,
    *,
    ephem: Optional[EphemerisAdapter] = None,
    n_perm: int = 2000,
    alpha: float = 0.05,
    stratify_by: Optional[str] = None,    # prior behavior (kept)
    group_by: Optional[str] = None,       # preferred: group id for "within"/"circular"
    perm_mode: Literal["iid","within","circular"] = "iid",
    use_time: bool = True,                # use rec['jd_tt'] as time for circular mode
    seed: Optional[int] = None
) -> List[EvalResult]:
    ep = ephem or EphemerisAdapter(EphemConfig(frame="ecliptic-of-date"))
    y: List[int] = []; strata: List[Any] = []; rows: List[Dict[str, float]] = []; times: List[float] = []

    for rec in records:
        y.append(int(rec.get("outcome", 0)))
        gid = rec.get(group_by) if group_by else (rec.get(stratify_by) if stratify_by else None)
        strata.append(gid)
        times.append(float(rec.get("jd_tt", 0.0)))
        feats = feature_fn(rec, ep)
        rows.append(feats)

    names: List[str] = sorted({k for r in rows for k in r.keys()})
    results: List[EvalResult] = []
    pvals: List[float] = []; effects: List[float] = []; ns: List[int] = []

    for name in names:
        x: List[float] = []; yy: List[int] = []; ss: List[Any] = []; tt: List[float] = []
        for i, r in enumerate(rows):
            if name in r and math.isfinite(r[name]):
                x.append(float(r[name])); yy.append(y[i]); ss.append(strata[i]); tt.append(times[i])
        if len(x) < 8 or len(set(yy)) < 2:
            effects.append(0.0); pvals.append(1.0); ns.append(len(x)); continue
        r_obs, p = permutation_pvalue_corr(
            x, yy, n_perm=n_perm,
            strata=ss if (perm_mode != "iid") else (ss if stratify_by else None),
            perm_mode=perm_mode,
            times=tt if (use_time and perm_mode == "circular") else None,
            seed=seed
        )
        effects.append(r_obs); pvals.append(p); ns.append(len(x))

    qvals, flags = bh_fdr(pvals, alpha=alpha) if names else ([], [])
    for name, n, r, p, q, ok in zip(names, ns, effects, pvals, qvals, flags):
        results.append(EvalResult(feature=name, n=n, effect_r=r, p_perm=p, q_fdr=q, accepted=ok))
    results.sort(key=lambda e: (e.q_fdr, e.p_perm, -abs(e.effect_r), e.feature))
    return results

def holdout_replicate(
    records: List[Dict[str, Any]],
    feature_fn: FeatureFn,
    *,
    train_frac: float = 0.7,
    alpha: float = 0.05,
    n_perm_train: int = 2000,
    n_perm_test: int = 4000,
    perm_mode: Literal["iid","within","circular"] = "iid",
    group_by: Optional[str] = None,
    use_time: bool = True,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    rnd = random.Random(seed)
    idx = list(range(len(records))); rnd.shuffle(idx)
    cut = max(1, int(len(idx) * train_frac))
    tr_idx = set(idx[:cut])
    train = [records[i] for i in range(len(records)) if i in tr_idx]
    test  = [records[i] for i in range(len(records)) if i not in tr_idx]

    train_res = evaluate_univariate(
        train, feature_fn,
        n_perm=n_perm_train, alpha=alpha,
        perm_mode=perm_mode, group_by=group_by, use_time=use_time, seed=seed
    )
    selected = [r.feature for r in train_res if r.accepted]

    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date"))
    y_test: List[int] = []; strata: List[Any] = []; rows: List[Dict[str, float]] = []; times: List[float] = []
    for rec in test:
        y_test.append(int(rec.get("outcome", 0)))
        strata.append(rec.get(group_by) if group_by else None)
        times.append(float(rec.get("jd_tt", 0.0)))
        rows.append(feature_fn(rec, ep))

    detailed: List[Dict[str, Any]] = []; replicated = 0
    for name in selected:
        x: List[float] = []; yy: List[int] = []; ss: List[Any] = []; tt: List[float] = []
        for i, r in enumerate(rows):
            if name in r and math.isfinite(r[name]):
                x.append(float(r[name])); yy.append(y_test[i]); ss.append(strata[i]); tt.append(times[i])
        if len(x) < 8 or len(set(yy)) < 2:
            detailed.append({"feature": name, "n": len(x), "p_perm": 1.0, "effect_r": 0.0, "replicated": False})
            continue
        r_obs, p = permutation_pvalue_corr(
            x, yy, n_perm=n_perm_test,
            strata=ss if (perm_mode != "iid") else None,
            perm_mode=perm_mode,
            times=tt if (use_time and perm_mode == "circular") else None,
            seed=seed
        )
        ok = p <= alpha
        if ok: replicated += 1
        detailed.append({"feature": name, "n": len(x), "p_perm": p, "effect_r": r_obs, "replicated": ok})

    rate = (replicated / max(1, len(selected))) if selected else 0.0
    return {
        "train_results": [r.__dict__ for r in train_res],
        "selected_features": selected,
        "test_details": detailed,
        "replication_rate": rate,
        "n_train": len(train),
        "n_test": len(test),
    }

# =============================================================================
# Built-in feature builders (unchanged)
# =============================================================================

def feature_transit_proximity(
    movers: List[str],
    targets_key: str = "natal_longitudes",
    *,
    aspects: Iterable[AspectSpec] = MAJOR_ASPECTS,
    orb_deg: float = 1.0
) -> FeatureFn:
    label = f"tr_any_major_within_{int(round(orb_deg))}deg"
    def _fn(rec: Dict[str, Any], ephem: EphemerisAdapter) -> Dict[str, float]:
        jd = float(rec["jd_tt"]); targets: Dict[str, float] = rec.get(targets_key, {}) or {}
        if not targets: return {}
        try:
            r = ephem.ecliptic_longitudes(jd, movers)
            lmap = rows_to_maps(r.get("results", []))["longitudes"]
        except Exception:
            lmap = {}
            for m in movers:
                lrow = ephem.ecliptic_longitudes(jd, [m]).get("results", [])
                if lrow: lmap[m] = float(lrow[0]["longitude"])
        hit = 0
        for _, lm in lmap.items():
            for _, lt in targets.items():
                for spec in aspects:
                    if abs(_zodiacal_separation(lm, lt, spec.angle)) <= max(0.0, orb_deg):
                        hit = 1; break
                if hit: break
            if hit: break
        return {label: float(hit)}
    return _fn

def feature_dasha_lords_onehot(level: int = 1) -> FeatureFn:
    tag = f"dashaL{level}_"
    def _fn(rec: Dict[str, Any], _ephem: EphemerisAdapter) -> Dict[str, float]:
        bj = float(rec["birth_jd_tt"]); t  = float(rec["jd_tt"]); ay = float(rec.get("ayanamsa_deg", 0.0))
        moon_trop = float(rec["natal_longitudes"]["moon"])
        periods = vimsottari_dasha(birth_jd_tt=bj, moon_lon_tropical_deg=moon_trop, ayanamsa_deg=ay, levels=max(1,level))
        lord = None
        for p in periods:
            if p.level == level and p.start_jd_tt - 1e-9 <= t <= p.end_jd_tt + 1e-9:
                lord = p.lord; break
        if lord is None: return {}
        feats = {tag + L: (1.0 if L == lord else 0.0) for L in _VIM_ORDER}
        return feats
    return _fn

def feature_yoga_flags(yoga_names: Iterable[str] = ("panch_mahapurusha","gajakesari","chandra_mangal","parivartana")) -> FeatureFn:
    ynames = tuple(yoga_names)
    def _fn(rec: Dict[str, Any], _ephem: EphemerisAdapter) -> Dict[str, float]:
        pts = rec.get("natal_longitudes") or {}; cusps = rec.get("natal_cusps") or []
        if not pts or len(cusps) != 12: return {}
        found = detect_yogas(points_deg=pts, cusps_deg=cusps, include=ynames)
        active = {f"yoga_{y}": 0.0 for y in ynames}
        for item in found:
            nm = item.get("yoga","").lower().replace(" ","_"); key = f"yoga_{nm}"
            if key in active: active[key] = 1.0
        return active
    return _fn

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Aspects
    "AspectSpec", "AspectKind", "MAJOR_ASPECTS", "MINOR_ASPECTS",
    "antiscia_longitude", "contra_antiscia_longitude",
    # Transits
    "TransitEngine", "TransitEvent",
    # Dasha
    "DashaPeriod", "vimsottari_dasha",
    # Varga
    "compute_vargas_for_point", "compute_vargas",
    # Yogas
    "detect_yogas", "house_index_for_longitude",
    # Houses & timescales helpers
    "compute_houses", "timescales_from_civil",
    # Validation (time-series aware)
    "EvalResult", "evaluate_univariate", "bh_fdr", "holdout_replicate", "permutation_pvalue_corr",
    "feature_transit_proximity", "feature_dasha_lords_onehot", "feature_yoga_flags",
]
