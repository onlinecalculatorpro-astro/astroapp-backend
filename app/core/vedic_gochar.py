# app/core/vedic_gochar.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Vedic Gochar (Transits) — precision & performance, sidereal-first.

Focus (Vedic):
- Graha dṛṣṭi gochar: planet→planet hits by classical dṛṣṭi schema
  • All planets full 7th; Mars (4,7,8), Jupiter (5,7,9), Saturn (3,7,10)
  • Degree-true refinement (roots at k*30° axes) with robust bracketing
- Ingresses: rāśi (sign) and nakṣatra boundary crossings (27 equal arcs)
- Stations: retrograde/direct via velocity zero-crossing
- Feature builder: feature_drishti_proximity (distance to dṛṣṭi axis)
- Sidereal pipeline: default on; ayanāṁśa subtraction supported

Design targets:
- Deterministic ephemeris batching & JD→longitude caching (like Western)
- Brent-like safeguarded refinement in JD days
- Degree-precise gochar (not whole-sign only), with lineage-friendly knobs
- Clean integration with constants_vedic and common_predictive utilities
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, Callable, Union
import math

from app.core.ephem_singleton import TS, PLANETS
from app.core.common_predictive import (
    norm360, wrap180, angdiff, PROF, _to_date, _jd_from_date, _ts_resolve
)
from app.core.constants_vedic import (
    graha_drishti_schema, drishti_strength_factor, nakshatra_index, NAKSHATRAS_27
)

try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    _EPH_OK = False
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Public exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Core event types & engine
    "GocharEvent", "VedicTransitEngine",
    # Gochar scanners
    "find_gochar_in_range",
    # Ingresses (rāśi & nakṣatra) and stations
    "find_rashi_ingresses_in_range", "find_nakshatra_ingresses_in_range", "find_stations_in_range",
    # Feature builder
    "feature_drishti_proximity",
]


# ─────────────────────────────────────────────────────────────────────────────
# Event model
# ─────────────────────────────────────────────────────────────────────────────
DrishtiKind = Literal["7th", "3rd", "4th", "5th", "8th", "9th", "10th"]

@dataclass
class GocharEvent:
    jd_tt: float
    body: str
    target: str
    drishti: DrishtiKind
    axis_deg: float           # k*30° axis at which separation centers
    separation_deg: float     # signed distance to axis at solution (≈0)
    applying: bool
    exact: bool
    weight: float
    meta: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Engine (caching, ephemeris, refinement)
# ─────────────────────────────────────────────────────────────────────────────
class VedicTransitEngine:
    """
    Performance-aware Vedic gochar finder:
    - Batch ephemeris per time boundary for all movers
    - Carry forward lons(t1) → next iteration's l0
    - In-scan JD cache for lon(body, t) with pruning headroom
    - Brent-like zero finder (guarded) with JD-day tolerance
    - Sidereal mode enabled by default (ayanāṁśa subtract)
    """
    _LON_CACHE_MAX = 2000

    @staticmethod
    def _speed_est(body: str) -> float:
        b = (body or "").lower()
        if b == "moon": return 14.0
        if b in ("mercury", "venus"): return 1.6
        if b == "mars": return 0.9
        if b == "sun": return 1.0
        if b in ("jupiter", "saturn"): return 0.2
        return 0.1

    @staticmethod
    def _prune_lon_cache(lon_cache: Dict[Tuple[str, float], float], *, max_size: int) -> None:
        n = len(lon_cache)
        if n <= max_size: return
        remove = (n - max_size) + max(1, max_size // 4)
        for k in sorted(lon_cache.keys(), key=lambda kk: kk[1])[:remove]:
            lon_cache.pop(k, None)

    def __init__(
        self,
        *,
        ephem: Optional[EphemerisAdapter] = None,
        frame: str = "ecliptic-of-date",
        topocentric: bool = False,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        elevation_m: Optional[float] = None,
        prebatch_refinement: bool = False,
        lon_cache_max: Optional[int] = None,
        treat_nodes_like_saturn: bool = False,
    ):
        if not _EPH_OK:
            raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")
        self.ephem = ephem or EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
        self.frame = frame
        self.obs = dict(topocentric=bool(topocentric), latitude=latitude, longitude=longitude, elevation_m=elevation_m)
        self.prebatch_refinement = bool(prebatch_refinement)
        self._lon_cache_max = int(lon_cache_max) if isinstance(lon_cache_max, int) and lon_cache_max > 256 else self._LON_CACHE_MAX
        self.sidereal_mode: bool = True
        self.ayanamsa_deg: float = 0.0
        self.treat_nodes_like_saturn = bool(treat_nodes_like_saturn)

    # ephemeris wrappers
    def _lon_map(self, jd_tt: float, names: List[str]) -> Dict[str, float]:
        # Map Vedic names for nodes to adapter canonical names if needed
        req = [("North Node" if n.lower() == "rahu" else "South Node" if n.lower() == "ketu" else n) for n in names]
        res = self.ephem.ecliptic_longitudes(jd_tt, req, **self.obs).get("results", [])
        PROF["ephem_calls"] += 1
        got = {}
        for row in res or []:
            nm = str(row["name"])
            if nm == "North Node": nm = "Rahu"
            if nm == "South Node": nm = "Ketu"
            got[nm] = float(row["longitude"])
        if self.sidereal_mode:
            ay = self.ayanamsa_deg
            return {k: norm360(v - ay) for k, v in got.items()}
        return got

    def _lon_cached(self, name: str, t: float, lon_cache: Dict[Tuple[str, float], float]) -> float:
        key = (name, float(t))
        v = lon_cache.get(key)
        if v is not None:
            return v
        got = self._lon_map(t, [name])
        if name not in got:
            raise RuntimeError(f"no ephemeris for {name}@{t}")
        v = float(got[name])
        lon_cache[key] = v
        self._prune_lon_cache(lon_cache, max_size=self._lon_cache_max)
        return v

    def _preload_lons(self, body: str, times: Iterable[float], lon_cache: Dict[Tuple[str, float], float]) -> None:
        for t in times:
            key = (body, float(t))
            if key in lon_cache:
                continue
            got = self._lon_map(float(t), [body])
            v = got.get(body)
            if v is not None:
                lon_cache[key] = float(v)
        self._prune_lon_cache(lon_cache, max_size=self._lon_cache_max)

    # safeguarded root finder (Brent-like)
    @staticmethod
    def _refine_zero_brent(f: Callable[[float], float], a: float, b: float, fa: float, fb: float, *, max_iter: int = 32, tol_days: float = 1e-6) -> float:
        from app.core.common_predictive import PROF as _PROF
        _PROF["refinements"] += 1
        if fa == 0.0: return a
        if fb == 0.0: return b
        if fa * fb > 0.0:
            aa, bb = a, b
            for _ in range(max_iter):
                m = 0.5 * (aa + bb)
                fm = f(m)
                if fm == 0.0 or (bb - aa) <= tol_days:
                    return m
                if fa * fm <= 0:
                    bb, fb = m, fm
                else:
                    aa, fa = m, fm
            return 0.5 * (aa + bb)
        c, fc = a, fa
        d = e = b - a
        for _ in range(max_iter):
            if fb == 0.0: return b
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            m = 0.5 * (a + b)
            if abs(b - a) <= tol_days:
                return b
            if fa != fc and fb != fc:
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + (b * fa * fc) / ((fb - fa) * (fb - fc)) + (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                s = b - fb * (b - a) / (fb - fa)
            cond = not ((3 * a + b) / 4 < s < b if a < b else b < s < (3 * a + b) / 4)
            cond |= (e and abs(s - b) >= abs(e) / 2)
            cond |= (not e and abs(s - b) >= abs(d) / 2)
            cond |= (abs(e) < tol_days)
            cond |= (abs(d) < tol_days)
            if cond:
                s = m
                d = e = b - a
            else:
                d, e = e, b - s
            fs = f(s)
            c, fc = a, fa
            if (fa * fs) < 0:
                b, fb = s, fs
            else:
                a, fa = s, fs
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
        return b

    # ─────────────────────────────────────────────────────────────────────────
    # Graha dṛṣṭi scan
    # ─────────────────────────────────────────────────────────────────────────
    def scan_drishti(
        self,
        *,
        jd_start_tt: float,
        jd_end_tt: float,
        movers: List[str],
        targets: Dict[str, float],
        orb_deg: float = 12.0,
        orb_map: Optional[Dict[str, float]] = None,
        step_minutes: Union[str, float, int] = "auto",
        include_nodes: bool = False,
    ) -> List[GocharEvent]:
        if jd_end_tt <= jd_start_tt or not movers or not targets:
            return []

        # movers / nodes handling for schema
        def _schema_for(planet: str) -> Dict[int, float]:
            p = planet
            if planet in ("Rahu", "Ketu") and self.treat_nodes_like_saturn:
                p = "Saturn"
            sch = graha_drishti_schema(p)
            if planet in ("Rahu", "Ketu") and not (self.treat_nodes_like_saturn or include_nodes):
                # default: only 7th for nodes if included=False -> effectively skip
                sch = ({7: 1.0} if include_nodes else {})
            return sch

        # step selection
        auto = isinstance(step_minutes, str) and step_minutes.lower() == "auto"
        if auto:
            caps_min: List[int] = []
            for m in movers:
                n = (m or "").lower()
                if n == "moon": caps_min.append(10)
                elif n in ("mercury","venus","mars"): caps_min.append(30)
                elif n in ("sun","jupiter","saturn"): caps_min.append(90)
                else: caps_min.append(180)
            step_minutes = max(5, min(caps_min) if caps_min else 30)
        try:
            dt = float(step_minutes) / (60.0 * 24.0)
        except Exception:
            dt = 30.0 / (60.0 * 24.0)

        events: List[GocharEvent] = []
        dedupe: set[Tuple[str, str, str, int]] = set()
        lon_cache: Dict[Tuple[str, float], float] = {}

        # preload left boundary
        t0 = float(jd_start_tt)
        l0 = self._lon_map(t0, movers)
        for m, v in l0.items():
            lon_cache[(m, t0)] = float(v)
        self._prune_lon_cache(lon_cache, max_size=self._lon_cache_max)

        def _lon_cached_local(name: str, t: float) -> float:
            return self._lon_cached(name, t, lon_cache)

        # scanning
        while t0 < jd_end_tt - 1e-12:
            t1 = float(min(t0 + dt, jd_end_tt))
            l1 = self._lon_map(t1, movers)
            for m, v in l1.items():
                lon_cache[(m, t1)] = float(v)
            self._prune_lon_cache(lon_cache, max_size=self._lon_cache_max)

            for body in movers:
                # skip node schema if not included
                if body in ("Rahu","Ketu") and not (include_nodes or self.treat_nodes_like_saturn):
                    continue

                lon0 = l0.get(body); lon1 = l1.get(body)
                if lon0 is None or lon1 is None:
                    continue
                max_change_possible = self._speed_est(body) * dt
                schema = _schema_for(body)
                if not schema:
                    continue

                for tgt_name, tgt_lon in targets.items():
                    for k, weight in schema.items():
                        axis = (k * 30.0) % 360.0
                        # residual to axis at t
                        def make_sep(b: str, tgt: float, ax: float) -> Callable[[float], float]:
                            return lambda t: wrap180(angdiff(_lon_cached_local(b, t), tgt) - ax)

                        s0 = wrap180(angdiff(lon0, tgt_lon) - axis)
                        s1 = wrap180(angdiff(lon1, tgt_lon) - axis)
                        if not (math.isfinite(s0) and math.isfinite(s1)):
                            continue
                        if abs(s0) > 120.0 and abs(s1) > 120.0:
                            continue

                        sign_change = (s0 == 0.0) or (s1 == 0.0) or ((s0 * s1) < 0.0)
                        _orb = float(orb_map.get(body, orb_deg)) if isinstance(orb_map, dict) else float(orb_deg)
                        guard = max(0.15, min(_orb * 0.33, max_change_possible * 1.5))
                        near = (min(abs(s0), abs(s1)) <= (_orb + guard))
                        if not (sign_change or near):
                            continue
                        if min(abs(s0), abs(s1)) > (_orb + max_change_possible):
                            continue

                        f = make_sep(body, tgt_lon, axis)
                        if self.prebatch_refinement:
                            mid = 0.5 * (t0 + t1)
                            t13 = t0 + (t1 - t0) / 3.0
                            t23 = t0 + 2.0 * (t1 - t0) / 3.0
                            t38 = t0 + (t1 - t0) * 3.0 / 8.0
                            t58 = t0 + (t1 - t0) * 5.0 / 8.0
                            self._preload_lons(body, (t0, t1, mid, t13, t23, t38, t58), lon_cache)

                        t_exact = self._refine_zero_brent(f, t0, t1, s0, s1, tol_days=1e-6)
                        lon_now = _lon_cached_local(body, t_exact)
                        sep = wrap180(angdiff(lon_now, tgt_lon) - axis)
                        applying = (abs(s1) < abs(s0))
                        # dedupe at 1-second buckets
                        bucket = int(math.floor(t_exact * 86400.0 + 0.5))
                        key = (body, tgt_name, f"{k}th", bucket)
                        if key in dedupe:
                            continue
                        dedupe.add(key)

                        events.append(GocharEvent(
                            jd_tt=float(t_exact),
                            body=body,
                            target=tgt_name,
                            drishti=(f"{k}th" if k != 7 else "7th"),  # normalize label
                            axis_deg=float(axis),
                            separation_deg=float(sep),
                            applying=bool(applying),
                            exact=abs(sep) <= 1e-6,
                            weight=float(weight),
                            meta={"orb_deg": _orb, "k_house": int(k)},
                        ))

            t0 = t1
            l0 = l1

        events.sort(key=lambda e: (e.jd_tt, e.body, e.target, e.axis_deg))
        return events


# ─────────────────────────────────────────────────────────────────────────────
# Public wrappers
# ─────────────────────────────────────────────────────────────────────────────
def _sign_index(lon: float) -> int:
    return int(math.floor(norm360(lon) / 30.0)) % 12

def find_gochar_in_range(
    *,
    natal_chart: dict,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    time_range: tuple | list | None = None,
    transiting_bodies: list[str] | None = None,   # default 7 grahas; nodes optional
    natal_targets: list[str] | None = None,
    frame: str | None = None,
    zodiac_mode: str | None = None,
    ayanamsa_deg: float | None = None,
    orb_deg: float | None = None,
    orb_map: Dict[str, float] | None = None,
    include_nodes: bool | None = None,
    treat_nodes_like_saturn: bool | None = None,
    step_minutes: Union[str, float, int] = "auto",
    **kwargs,
) -> Dict[str, Any]:
    """Route-friendly Vedic gochar wrapper."""
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "gochar": [], "meta": {}}

    frame = frame or "ecliptic-of-date"
    zodiac_mode = (zodiac_mode or "sidereal").lower()
    ay = float(ayanamsa_deg or 0.0)
    use_nodes = bool(include_nodes or False)

    tz = str(natal_chart.get("place_tz") or natal_chart.get("timezone") or "UTC")
    if start_jd_tt is None or end_jd_tt is None:
        if not time_range or len(time_range) != 2:
            return {"ok": False, "error": "time_range_required", "gochar": [], "meta": {}}
        d0 = _to_date(time_range[0]); d1 = _to_date(time_range[1])
        start_jd_tt = _jd_from_date(d0, tz, _ts_resolve)
        end_jd_tt   = _jd_from_date(d1, tz, _ts_resolve) + (24*60-1) / (24*60)

    movers = list(transiting_bodies or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"])
    if use_nodes:
        if "Rahu" not in movers: movers.append("Rahu")
        if "Ketu" not in movers: movers.append("Ketu")
    tgts   = list(natal_targets or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"])

    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = VedicTransitEngine(
        ephem=ep, frame=frame, prebatch_refinement=bool(kwargs.get("prebatch_refinement", False)),
        treat_nodes_like_saturn=bool(treat_nodes_like_saturn or False),
    )
    eng.sidereal_mode = not zodiac_mode.startswith("tropical")
    eng.ayanamsa_deg = ay

    # natal targets at left boundary
    nat_rows = ep.ecliptic_longitudes(float(start_jd_tt), [("North Node" if n=="Rahu" else "South Node" if n=="Ketu" else n) for n in tgts]).get("results", [])
    PROF["ephem_calls"] += 1
    nat_map0 = {}
    for row in nat_rows or []:
        nm = str(row["name"])
        if nm == "North Node": nm = "Rahu"
        if nm == "South Node": nm = "Ketu"
        nat_map0[nm] = float(row["longitude"])
    if eng.sidereal_mode:
        targets = {k: norm360(v - ay) for k, v in nat_map0.items()}
    else:
        targets = dict(nat_map0)

    try:
        evs = eng.scan_drishti(
            jd_start_tt=float(start_jd_tt), jd_end_tt=float(end_jd_tt),
            movers=movers, targets=targets, orb_deg=float(orb_deg or 12.0),
            orb_map=orb_map, step_minutes=step_minutes, include_nodes=use_nodes,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": f"gochar_scan_failed:{e}",
            "gochar": [],
            "meta": {"frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": ay, "prof": dict(PROF)},
        }

    hits: List[Dict[str, Any]] = []
    for ev in evs:
        hits.append({
            "transiting_body": ev.body,
            "natal_body": ev.target,
            "drishti": ev.drishti,
            "axis_deg": float(ev.axis_deg),
            "orb": abs(float(ev.separation_deg)),
            "max_orb": float(ev.meta.get("orb_deg", 12.0)),
            "weight": float(ev.weight),
            "applying": bool(ev.applying),
            "exact": bool(ev.exact),
            "exact_jd_tt": float(ev.jd_tt),
            "exact_jd_ut1": None,
            "exact_datetime_utc": None,
        })

    return {
        "ok": True, "technique": "gochar_drishti", "gochar": hits,
        "meta": {
            "movers": movers, "natal_targets": list(targets.keys()),
            "window_jd_tt": [float(start_jd_tt), float(end_jd_tt)],
            "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": ay, "prof": dict(PROF),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ingresses: rāśi and nakṣatra
# ─────────────────────────────────────────────────────────────────────────────
def find_rashi_ingresses_in_range(
    *,
    start_jd_tt: float,
    end_jd_tt: float,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "sidereal",
    ayanamsa_deg: float = 0.0,
    step_minutes: Union[str, float, int] = "auto",
    topocentric: bool = False,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float | None = None,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "ingresses": [], "meta": {}}
    if end_jd_tt <= start_jd_tt:
        return {"ok": True, "ingresses": [], "meta": {}}

    movers = list(movers or ["Sun","Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    auto = isinstance(step_minutes, str) and step_minutes.lower() == "auto"
    if auto:
        caps_min: List[int] = []
        for m in movers:
            n = (m or "").lower()
            if n == "moon": caps_min.append(10)
            elif n in ("mercury","venus","mars"): caps_min.append(30)
            elif n in ("sun","jupiter","saturn"): caps_min.append(90)
            else: caps_min.append(180)
        step_minutes = max(5, min(caps_min) if caps_min else 60)
    try:
        dt = float(step_minutes) / (60.0 * 24.0)
    except Exception:
        dt = 60.0 / (60.0 * 24.0)

    events: List[Dict[str, Any]] = []
    lon_cache: Dict[Tuple[str, float], float] = {}
    def lon_at(body: str, t: float) -> float:
        return eng._lon_cached(body, float(t), lon_cache)

    def sign_change(body: str, t0: float, t1: float) -> Optional[Tuple[float, int]]:
        l0 = lon_at(body, t0); l1 = lon_at(body, t1)
        s0 = _sign_index(l0); s1 = _sign_index(l1)
        if s0 == s1:
            return None
        # refine f(t) = wrap180((lon % 30) - 0)
        def f(t: float) -> float:
            return wrap180((lon_at(body, t) % 30.0) - 0.0)
        v0 = wrap180((l0 % 30.0) - 0.0)
        v1 = wrap180((l1 % 30.0) - 0.0)
        t_exact = eng._refine_zero_brent(f, t0, t1, v0, v1, tol_days=1e-6)
        sign_to = _sign_index(lon_at(body, t_exact))
        return (t_exact, sign_to)

    t = float(start_jd_tt)
    while t < end_jd_tt - 1e-12:
        t2 = float(min(t + dt, end_jd_tt))
        for b in movers:
            sc = sign_change(b, t, t2)
            if sc:
                t_exact, sign_to = sc
                events.append({"body": b, "exact_jd_tt": float(t_exact), "sign_to": int(sign_to)})
        t = t2

    events.sort(key=lambda r: (r["exact_jd_tt"], r["body"]))
    return {
        "ok": True, "ingresses": events,
        "meta": {"movers": movers, "window_jd_tt": [float(start_jd_tt), float(end_jd_tt)],
                 "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg)}
    }


def find_nakshatra_ingresses_in_range(
    *,
    start_jd_tt: float,
    end_jd_tt: float,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "sidereal",
    ayanamsa_deg: float = 0.0,
    step_minutes: Union[str, float, int] = "auto",
    topocentric: bool = False,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float | None = None,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "ingresses": [], "meta": {}}
    if end_jd_tt <= start_jd_tt:
        return {"ok": True, "ingresses": [], "meta": {}}

    movers = list(movers or ["Moon","Sun","Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    width = 360.0 / 27.0  # 13°20′

    auto = isinstance(step_minutes, str) and step_minutes.lower() == "auto"
    if auto:
        caps_min: List[int] = []
        for m in movers:
            n = (m or "").lower()
            if n == "moon": caps_min.append(5)
            elif n in ("mercury","venus","mars"): caps_min.append(20)
            else: caps_min.append(60)
        step_minutes = max(2, min(caps_min) if caps_min else 20)
    try:
        dt = float(step_minutes) / (60.0 * 24.0)
    except Exception:
        dt = 20.0 / (60.0 * 24.0)

    events: List[Dict[str, Any]] = []
    lon_cache: Dict[Tuple[str, float], float] = {}
    def lon_at(body: str, t: float) -> float:
        return eng._lon_cached(body, float(t), lon_cache)

    def nak_change(body: str, t0: float, t1: float) -> Optional[Tuple[float, int]]:
        l0 = lon_at(body, t0); l1 = lon_at(body, t1)
        n0 = nakshatra_index(l0); n1 = nakshatra_index(l1)
        if n0 == n1:
            return None
        # refine f(t) = wrap180((lon % width) - 0)
        def f(t: float) -> float:
            return wrap180((lon_at(body, t) % width) - 0.0)
        v0 = wrap180((l0 % width) - 0.0)
        v1 = wrap180((l1 % width) - 0.0)
        t_exact = eng._refine_zero_brent(f, t0, t1, v0, v1, tol_days=5e-7)
        idx = nakshatra_index(lon_at(body, t_exact))
        return (t_exact, idx)

    t = float(start_jd_tt)
    while t < end_jd_tt - 1e-12:
        t2 = float(min(t + dt, end_jd_tt))
        for b in movers:
            sc = nak_change(b, t, t2)
            if sc:
                t_exact, idx = sc
                events.append({"body": b, "exact_jd_tt": float(t_exact), "nakshatra_index": int(idx), "nakshatra_name": NAKSHATRAS_27[(idx-1)%27]})
        t = t2

    events.sort(key=lambda r: (r["exact_jd_tt"], r["body"]))
    return {
        "ok": True, "ingresses": events,
        "meta": {"movers": movers, "window_jd_tt": [float(start_jd_tt), float(end_jd_tt)],
                 "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg)}
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stations — velocity zero-crossing (retrograde/direct)
# ─────────────────────────────────────────────────────────────────────────────
def find_stations_in_range(
    *,
    start_jd_tt: float,
    end_jd_tt: float,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "sidereal",
    ayanamsa_deg: float = 0.0,
    step_minutes: Union[str, float, int] = "auto",
    topocentric: bool = False,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float | None = None,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "stations": [], "meta": {}}
    if end_jd_tt <= start_jd_tt:
        return {"ok": True, "stations": [], "meta": {}}

    movers = list(movers or ["Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    auto = isinstance(step_minutes, str) and step_minutes.lower() == "auto"
    if auto:
        caps_min: List[int] = []
        for m in movers:
            n = (m or "").lower()
            if n in ("mercury","venus"): caps_min.append(60)
            elif n in ("mars",):         caps_min.append(120)
            else:                        caps_min.append(180)
        step_minutes = max(15, min(caps_min) if caps_min else 120)
    try:
        dt = float(step_minutes) / (60.0 * 24.0)
    except Exception:
        dt = 120.0 / (60.0 * 24.0)

    events: List[Dict[str, Any]] = []
    lon_cache: Dict[Tuple[str, float], float] = {}
    def lon_at(body: str, t: float) -> float:
        return eng._lon_cached(body, float(t), lon_cache)

    def vel(body: str, t: float, h_days: float = 1.0 / (24.0 * 12.0)) -> float:
        l1 = lon_at(body, t - h_days)
        l2 = lon_at(body, t + h_days)
        d = wrap180(l2 - l1)
        return d / (2.0 * h_days)

    def zero_cross(body: str, t0: float, t1: float) -> Optional[Tuple[float, str]]:
        v0 = vel(body, t0); v1 = vel(body, t1)
        if not (math.isfinite(v0) and math.isfinite(v1)):
            return None
        if v0 == 0.0: return (t0, "station")
        if v1 == 0.0: return (t1, "station")
        if (v0 * v1) > 0.0: return None
        def f(t: float) -> float: return vel(body, t)
        t_exact = eng._refine_zero_brent(f, t0, t1, v0, v1, tol_days=5e-6)
        pre = vel(body, t_exact - 2.0 / (24.0 * 12.0))
        post = vel(body, t_exact + 2.0 / (24.0 * 12.0))
        if pre > 0.0 and post < 0.0: k = "retrograde"
        elif pre < 0.0 and post > 0.0: k = "direct"
        else: k = "station"
        return (t_exact, k)

    t = float(start_jd_tt)
    while t < end_jd_tt - 1e-12:
        t2 = float(min(t + dt, end_jd_tt))
        for b in movers:
            zc = zero_cross(b, t, t2)
            if zc:
                t_exact, kind = zc
                events.append({"body": b, "exact_jd_tt": float(t_exact), "kind": kind})
        t = t2

    events.sort(key=lambda r: (r["exact_jd_tt"], r["body"]))
    return {
        "ok": True, "stations": events,
        "meta": {"movers": movers, "window_jd_tt": [float(start_jd_tt), float(end_jd_tt)],
                 "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg)}
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────────────────────────────────────
def feature_drishti_proximity(
    *,
    hits: List[Dict[str, Any]],
    cap_deg: float = 12.0
) -> List[float]:
    """
    Build a simple 0..1 proximity feature per hit (1 at axis, →0 at cap).
    """
    out: List[float] = []
    cap = max(1e-6, float(cap_deg))
    for h in hits:
        orb = abs(float(h.get("orb", float("inf"))))
        if not math.isfinite(orb):
            continue
        score = max(0.0, 1.0 - (orb / cap))
        out.append(score)
    return out
