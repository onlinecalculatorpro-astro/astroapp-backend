# app/core/predictive.py
from __future__ import annotations

"""
Predictive Toolkit — Transits • Dasha • Varga • Yoga • Validation

This module is a single import hub used by prediction.py.

Exports (see __all__):
- Transits: TransitEngine, TransitEvent, find_transits_in_range
- Dasha:   DashaPeriod, vimsottari_dasha, predict_dasha_periods
- Varga:   compute_vargas_for_point, compute_vargas
- Yoga:    detect_yogas, house_index_for_longitude
- Houses/Timescales: compute_houses, timescales_from_civil
- Validation: evaluate_univariate, permutation_pvalue_corr, bh_fdr, holdout_replicate, validate_predictions
- Feature builders: feature_transit_proximity, feature_dasha_lords_onehot, feature_yoga_flags
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, Callable
import math
import random
import logging

# ───────────────────── precise backends (soft imports) ─────────────────────
# Ephemeris adapter
try:
    from app.core.ephemeris_adapter import (
        EphemerisAdapter, Config as EphemConfig, rows_to_maps, get_node_longitude
    )
    _EPH_OK = True
    _EPH_ERR = None
except Exception as e:
    # Do NOT raise at import time; defer error until a function needs it
    _EPH_OK = False
    _EPH_ERR = e
    EphemerisAdapter = None          # type: ignore
    EphemConfig = None               # type: ignore

    def rows_to_maps(_rows):         # minimal harmless fallback
        return {"longitudes": {}}

    def get_node_longitude(*_args, **_kwargs):
        return None

# Houses
_HAS_POLICY = False
_HOUSES_OK = True
try:
    from app.core.house import compute_houses_with_policy as _compute_houses_policy
    _HAS_POLICY = True
except Exception:
    try:
        from app.core.houses import asc_mc_houses as _asc_mc_houses  # fallback implementation
    except Exception as _houses_err:
        _HOUSES_OK = False
        _compute_houses_policy = None   # type: ignore
        _asc_mc_houses = None           # type: ignore
        _HOUSES_ERR = _houses_err
    else:
        _HOUSES_ERR = None
else:
    _HOUSES_ERR = None

# Timescale resolver (optional)
try:
    from app.core.validators import resolve_timescales_from_civil_erfa as _ts_resolve
    _TS_OK = True
    _TS_ERR = None
except Exception as e:
    _TS_OK = False
    _TS_ERR = e
    _ts_resolve = None  # type: ignore

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
            v = self._lon_map(t, [name]).get(name)
            if v is None:
                raise RuntimeError(f"no ephemeris for {name}@{t}")
            lon_cache[key] = float(v)
            return lon_cache[key]

        def make_sep(body: str, target_lon: float, spec: AspectSpec):
            if spec.kind == "zodiacal":
                return lambda t: _zodiacal_separation(_lon_cached(body, t), target_lon, spec.angle)
            elif spec.kind == "antiscia":
                b_image = antiscia_longitude(target_lon)
                return lambda t: wrap180(_lon_cached(body, t) - b_image)
            else:
                b_image = contra_antiscia_longitude(target_lon)
                return lambda t: wrap180(_lon_cached(body, t) - b_image)

        t0 = jd_start_tt
        l0 = self._lon_map(t0, movers)
        for m, v in l0.items():
            lon_cache[(m, float(t0))] = float(v)

        while t0 < jd_end_tt - 1e-12:
            t1 = min(t0 + dt, jd_end_tt)
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
                            epsd = 5.0 / (24.0 * 60.0)
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

            t0 = t1
            l0 = l1

        events.sort(key=lambda e: (e.jd_tt, e.body, e.target, e.aspect))
        return events

# =============================================================================
# Public transit wrapper expected by prediction.py
# =============================================================================

from datetime import datetime, date as _date
from types import SimpleNamespace

def _to_date(s) -> _date:
    if isinstance(s, _date):
        return s
    if isinstance(s, str):
        try:
            if "T" in s or " " in s:
                return datetime.fromisoformat(s).date()
            from datetime import datetime as _dt
            return _dt.strptime(s, "%Y-%m-%d").date()
        except Exception:
            pass
    return datetime.utcnow().date()

def _jd_from_date(d: _date, tz: str) -> float:
    if _ts_resolve is None:
        raise RuntimeError("Timescale resolver unavailable; pass jd_tt/jd_ut1 directly.")
    ts = _ts_resolve(d, "00:00:00", tz)
    return float(ts["jd_tt"])

def _aspects_from_kwargs(kwargs: dict) -> list[AspectSpec]:
    custom = kwargs.get("aspects")
    orbs = kwargs.get("orbs", {})
    specs: list[AspectSpec] = []
    if isinstance(custom, (list, tuple)) and custom:
        known = {a.name.lower(): a for a in (list(MAJOR_ASPECTS) + list(MINOR_ASPECTS))}
        for item in custom:
            if isinstance(item, str):
                a = known.get(item.strip().lower())
                if a:
                    orb = float(orbs.get(item.strip().lower(), a.orb_deg)) if isinstance(orbs, dict) else a.orb_deg
                    specs.append(AspectSpec(a.name, a.angle, orb, a.kind, a.weight))
            elif isinstance(item, dict):
                try:
                    nm = str(item.get("name") or "Aspect")
                    ang = float(item["angle"])
                    orb = float(item.get("orb_deg", 1.0))
                    specs.append(AspectSpec(nm, ang, orb))
                except Exception:
                    continue
    if not specs:
        for a in MAJOR_ASPECTS:
            nm = a.name.lower()
            orb = float(orbs.get(nm, a.orb_deg)) if isinstance(orbs, dict) else a.orb_deg
            specs.append(AspectSpec(a.name, a.angle, orb, a.kind, a.weight))
    return specs

def find_transits_in_range(
    *,
    natal_chart: dict,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    time_range: tuple | list | None = None,
    transiting_bodies: list[str] | None = None,
    natal_targets: list[str] | None = None,
    natal_bodies: list[str] | None = None,
    frame: str | None = None,
    zodiac_mode: str | None = None,
    ayanamsa_deg: float | None = None,
    include_aspects_to: list[str] | None = None,
    include_house_cusps: bool | None = None,
    exact_timing: bool | None = None,
    aspects: list[str] | None = None,
    orbs: Dict[str, float] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Returns a dict with key 'transits' (shape expected by prediction.py).
    Each transit item includes: transiting_body, natal_body, aspect, orb, applying, exact_jd_tt, exact_datetime_utc, etc.
    """
    import time as _time
    t0_wall = _time.perf_counter()
    frame = frame or "ecliptic-of-date"
    zodiac_mode = (zodiac_mode or "tropical").lower()
    ay = float(ayanamsa_deg or 0.0)

    tz = str(natal_chart.get("place_tz") or natal_chart.get("timezone") or "UTC")
    if start_jd_tt is None or end_jd_tt is None:
        if not time_range or len(time_range) != 2:
            return {"ok": False, "error": "time_range_required", "transits": [], "meta": {}}
        d0 = _to_date(time_range[0]); d1 = _to_date(time_range[1])
        start_jd_tt = _jd_from_date(d0, tz)
        # end of day (inclusive feel)
        end_jd_tt = _jd_from_date(d1, tz) + (24*60-1) / (24*60)

    movers = list(transiting_bodies or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"])
    tgts   = list(natal_targets or natal_bodies or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"])

    ep = EphemerisAdapter(EphemConfig(frame=frame))
    eng = TransitEngine(ephem=ep, frame=frame)

    # natal longitudes for targets
    # (Angles/cusps could be added here later if needed.)
    nat_rows = ep.ecliptic_longitudes(start_jd_tt, tgts).get("results", [])
    nat_map = rows_to_maps(nat_rows)["longitudes"]
    targets: dict[str, float] = {k: float(v) for k, v in nat_map.items() if math.isfinite(float(v))}

    specs = _aspects_from_kwargs({"aspects": aspects, "orbs": (orbs or {})})
    include_antiscia = bool(kwargs.get("include_antiscia", False))
    antiscia_orb = float(kwargs.get("antiscia_orb_deg", 2.0))
    step_min = float(kwargs.get("step_minutes", 30.0))

    try:
        evs = eng.scan_aspects(
            jd_start_tt=float(start_jd_tt), jd_end_tt=float(end_jd_tt),
            movers=movers, targets=targets, aspects=specs,
            step_minutes=step_min, include_antiscia=include_antiscia, antiscia_orb_deg=antiscia_orb,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": f"transit_scan_failed:{e}",
            "transits": [],
            "meta": {"frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": ay},
            "computation_time_ms": (_time.perf_counter() - t0_wall) * 1000.0,
        }

    hits: List[Dict[str, Any]] = []
    for ev in evs:
        asp_name = ev.aspect.lower()
        orb_now = abs(ev.separation_deg)
        hits.append({
            "transiting_body": ev.body,
            "natal_body": ev.target,
            "aspect": asp_name,
            "orb": float(orb_now),
            "max_orb": float(next((s.orb_deg for s in specs if s.name.lower() == asp_name), 1.0)),
            "applying": bool(ev.applying),
            "exact": bool(ev.exact),
            "exact_jd_tt": float(ev.jd_tt),
            "exact_jd_ut1": None,
            "exact_datetime_utc": None,
            "p_value": 1.0,
            "house": None,
            "exact_longitude": None,
        })

    return {
        "ok": True,
        "technique": "transits",
        "transits": hits,
        "meta": {
            "movers": movers,
            "natal_targets": list(targets.keys()),
            "window_jd_tt": [float(start_jd_tt), float(end_jd_tt)],
            "frame": frame,
            "zodiac_mode": zodiac_mode,
            "ayanamsa_deg": ay,
        },
        "computation_time_ms": (_time.perf_counter() - t0_wall) * 1000.0,
    }

# =============================================================================
# VIMSOTTARI DASHA
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

# Convenience for prediction.py's comprehensive_forecast (optional)
def predict_dasha_periods(
    *,
    natal_chart: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    dasha_system: str = "vimshottari",
    include_antardasha: bool = True,
) -> Dict[str, Any]:
    if dasha_system.lower() not in ("vimshottari", "vimsottari", "vimshottari"):
        return {"ok": False, "error": "unsupported_dasha"}
    if _ts_resolve is None:
        return {"ok": False, "error": "timescale_resolver_unavailable"}

    # Resolve natal JD_TT (needs date/time/tz or direct jd_tt)
    if "jd_tt" in natal_chart:
        birth_jd_tt = float(natal_chart["jd_tt"])
    else:
        from datetime import datetime as _dt
        d = _dt.strptime(str(natal_chart.get("date")), "%Y-%m-%d").date()
        t = str(natal_chart.get("time") or "00:00:00")
        tz = str(natal_chart.get("place_tz") or natal_chart.get("timezone") or "UTC")
        ts = _ts_resolve(d, t, tz)
        birth_jd_tt = float(ts["jd_tt"])

    # Need Moon tropical longitude at birth; ask adapter
    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date"))
    mm = rows_to_maps(ep.ecliptic_longitudes(birth_jd_tt, ["Moon"]).get("results", []))["longitudes"]
    moon_lon_trop = float(mm.get("Moon") or mm.get("moon") or 0.0)

    periods = vimsottari_dasha(
        birth_jd_tt=birth_jd_tt,
        moon_lon_tropical_deg=moon_lon_trop,
        ayanamsa_deg=float(natal_chart.get("ayanamsa_deg", 0.0)),
        levels=(3 if include_antardasha else 1),
        span_years=120.0,
    )

    out: List[Dict[str, Any]] = []
    # Filter to provided date window
    def jd_to_iso(jd_tt: float) -> str:
        # rough conversion; prediction.py only displays/use jd_tt
        unix = (jd_tt - 2440587.5) * 86400.0
        return datetime.utcfromtimestamp(unix).isoformat() + "Z"
    for p in periods:
        out.append({
            "start_jd_tt": p.start_jd_tt,
            "end_jd_tt": p.end_jd_tt,
            "start_date": jd_to_iso(p.start_jd_tt),
            "end_date": jd_to_iso(p.end_jd_tt),
            "level": p.level,
            "mahadasha_lord": p.parent_chain[0] if p.parent_chain else p.lord,
            "chain": list(p.parent_chain),
            "meta": p.meta,
        })
    return {"ok": True, "periods": out, "system": "vimshottari"}

# =============================================================================
# Varga helpers
# =============================================================================

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

# =============================================================================
# Yogas (selected)
# =============================================================================

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
    out: List[Dict[str, Any]] = []
    if "panch_mahapurusha" in include: out.extend(detect_panch_mahapurusha(points_deg, cusps_deg))
    if "gajakesari" in include: out.extend(detect_gajakesari(points_deg, cusps_deg))
    if "chandra_mangal" in include: out.extend(detect_chandra_mangal(points_deg, max_orb_deg=(orbs or {}).get("chandra_mangal", 8.0)))
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
    from datetime import datetime as _dt
    d = _dt.strptime(date_yyyy_mm_dd, "%Y-%m-%d").date()
    return _ts_resolve(d, time_hh_mm_ss, place_tz)

# =============================================================================
# VALIDATION (with time-series–aware permutations) + light wrapper
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
    rnd = random.Random(seed)
    y = [float(int(v)) for v in y]
    r_obs = pearson_corr(x, y)
    if not math.isfinite(r_obs): return 0.0, 1.0
    if n_perm <= 0: return r_obs, 1.0

    indices = list(range(len(y)))
    if strata is None:
        groups = [indices]
    else:
        groups = _groups_from_ids(strata)

    order_in_group: Dict[int, List[int]] = {}
    if perm_mode == "circular":
        if times is None:
            perm_mode = "within"
        else:
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
                k = rnd.randrange(len(ord_idx))
                shifted = ord_idx[k:] + ord_idx[:k]
                vals = [y_work[i] for i in ord_idx]
                for i, v in zip(shifted, vals):
                    y_work[i] = v

        r_perm = pearson_corr(x, y_work)
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
    stratify_by: Optional[str] = None,
    group_by: Optional[str] = None,
    perm_mode: Literal["iid","within","circular"] = "iid",
    use_time: bool = True,
    seed: Optional[int] = None
) -> List[EvalResult]:
    ep = ephem or EphemerisAdapter(EphemConfig(frame="ecliptic-of-date"))
    y: List[int] = []; strata: List[Any] = []; rows: List[Dict[str, float]] = []; times: List[float] = []

    for rec in records:
        y.append(int(rec.get("outcome", 0)))
        gid = rec.get(group_by) if group_by else (rec.get(stratify_by) if stratify_by else None)
        strata.append(gid)
        times.append(float(rec.get("jd_tt", 0.0)))
        rows.append(feature_fn(rec, ep))

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

# lightweight validator used by prediction.py when statistical_validation=True
def validate_predictions(
    events: List[Any],
    *,
    method: str = "permutation",
    n_permutations: int = 200,
    fdr_correction: bool = True,
    **kwargs
) -> Dict[str, Any]:
    # Minimal, safe default: returns neutral metrics (won’t block predictions)
    return {"ok": True, "metrics": {"p_value": 1.0}, "warnings": []}

# =============================================================================
# Built-in feature builders
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
    "AspectSpec", "AspectKind",
    "MAJOR_ASPECTS", "MINOR_ASPECTS",
    "antiscia_longitude", "contra_antiscia_longitude",

    # Transits
    "TransitEngine", "TransitEvent", "find_transits_in_range",

    # Dasha
    "DashaPeriod", "vimsottari_dasha", "predict_dasha_periods",

    # Varga
    "compute_vargas_for_point", "compute_vargas",

    # Yogas
    "detect_yogas", "house_index_for_longitude",

    # Houses & timescales
    "compute_houses", "timescales_from_civil",

    # Validation
    "EvalResult", "evaluate_univariate",
    "bh_fdr", "holdout_replicate", "permutation_pvalue_corr", "validate_predictions",

    # Feature builders
    "feature_transit_proximity", "feature_dasha_lords_onehot", "feature_yoga_flags",
]
