# app/core/western_predictive.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Western Predictive Toolkit — precision & performance

Focus (Western):
- Transits: TransitEngine (scan_aspects + Brent refinement), TransitEvent
- Ingresses: sign-boundary crossings (0°,30°,...,330°)
- Stations: retrograde/direct stations via velocity zero-crossing
- Evaluation / Holdout: simple univariate metrics & replication
- Stats: Pearson r, permutation p-value for r, BH-FDR
- Feature builder: feature_transit_proximity (distance to nearest hit)

Design targets:
- Correct ecliptic-of-date pipeline; clean sidereal toggle (ayanāṃśa subtract)
- Batch ephemeris calls per time step for ALL movers
- In-scan JD→longitude cache with pruning headroom
- Brent-like refinement with bracket safety & tolerance in JD-days
- Deterministic dedupe at 1-second JD buckets
- Minimal allocations in hot paths; avoid repeated dict creates
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, Callable, Union

import math
from datetime import datetime

from app.core.ephem_singleton import TS, PLANETS
from app.core.common_predictive import (
    norm360, wrap180, angdiff, PROF, _to_date, _jd_from_date, _ts_resolve
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
    # Aspect specs / helpers
    "AspectKind", "AspectSpec", "MAJOR_ASPECTS", "MINOR_ASPECTS",
    "antiscia_longitude", "contra_antiscia_longitude",
    # Core transit types
    "TransitEvent", "TransitEngine", "find_transits_in_range",
    # Ingresses & stations
    "find_ingresses_in_range", "find_stations_in_range",
    # Western evaluation/holdout + stats helpers
    "evaluate_univariate", "holdout_replicate", "validate_predictions",
    "pearson_corr", "permutation_pvalue_corr", "bh_fdr",
    # Feature builders (Western)
    "feature_transit_proximity",
]


# ─────────────────────────────────────────────────────────────────────────────
# Aspect specification
# ─────────────────────────────────────────────────────────────────────────────
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

def _zodiacal_separation(a: float, b: float, target: float) -> float:
    return wrap180(angdiff(a, b) - target)

def antiscia_longitude(lon: float) -> float:
    return norm360(180.0 - lon)

def contra_antiscia_longitude(lon: float) -> float:
    return norm360(360.0 - lon)


# ─────────────────────────────────────────────────────────────────────────────
# Transit core
# ─────────────────────────────────────────────────────────────────────────────
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
    - Batch ephemeris per boundary time for all movers
    - Carry forward lons(t1) → next iteration's l0
    - In-scan JD cache for lon(body, t) during refinement; prune with headroom
    - Brent-like zero finder (guarded) with JD-day tolerance
    - Optional Sidereal mode via ayanāṃśa subtraction
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
        return 0.1  # outers default

    @staticmethod
    def _prune_lon_cache(lon_cache: Dict[Tuple[str, float], float], *, max_size: int) -> None:
        n = len(lon_cache)
        if n <= max_size:
            return
        # Remove oldest by JD; leave headroom to reduce churn
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
    ):
        if not _EPH_OK:
            raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")
        self.ephem = ephem or EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
        self.frame = frame
        self.obs = dict(topocentric=bool(topocentric), latitude=latitude, longitude=longitude, elevation_m=elevation_m)
        self.prebatch_refinement = bool(prebatch_refinement)
        self._lon_cache_max = int(lon_cache_max) if isinstance(lon_cache_max, int) and lon_cache_max > 256 else self._LON_CACHE_MAX
        self.sidereal_mode: bool = False
        self.ayanamsa_deg: float = 0.0

    # ── ephemeris wrappers ────────────────────────────────────────────────────
    def _lon_map(self, jd_tt: float, names: List[str]) -> Dict[str, float]:
        res = self.ephem.ecliptic_longitudes(jd_tt, names, **self.obs).get("results", [])
        PROF["ephem_calls"] += 1
        if not res:
            return {}
        if self.sidereal_mode:
            ay = self.ayanamsa_deg
            return {row["name"]: norm360(float(row["longitude"]) - ay) for row in res}
        return {row["name"]: float(row["longitude"]) for row in res}

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

    # ── root refiner (Brent variant; bracketed or guarded) ────────────────────
    @staticmethod
    def _refine_zero_brent(f: Callable[[float], float], a: float, b: float, fa: float, fb: float, *, max_iter: int = 32, tol_days: float = 1e-6) -> float:
        from app.core.common_predictive import PROF as _PROF
        _PROF["refinements"] += 1
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            # not bracketed → guarded bisection-like shrink
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
        # classic Brent
        c, fc = a, fa
        d = e = b - a
        for _ in range(max_iter):
            if fb == 0.0:
                return b
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            m = 0.5 * (a + b)
            if abs(b - a) <= tol_days:
                return b
            if fa != fc and fb != fc:
                # inverse quadratic interpolation
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + (b * fa * fc) / ((fb - fa) * (fb - fc)) + (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                # secant
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

    # ── main scan ─────────────────────────────────────────────────────────────
    def scan_aspects(
        self,
        *,
        jd_start_tt: float,
        jd_end_tt: float,
        movers: List[str],
        targets: Dict[str, float],
        aspects: Iterable[AspectSpec] = MAJOR_ASPECTS,
        step_minutes: Union[str, float, int] = "auto",
        include_antiscia: bool = False,
        antiscia_orb_deg: float = 2.0,
    ) -> List[TransitEvent]:
        if jd_end_tt <= jd_start_tt or not movers or not targets:
            return []

        asp_list: List[AspectSpec] = list(aspects)
        if include_antiscia:
            aorb = float(antiscia_orb_deg)
            asp_list.append(AspectSpec("Antiscia", 0.0, aorb, kind="antiscia"))
            asp_list.append(AspectSpec("Contra-Antiscia", 0.0, aorb, kind="contra-antiscia"))

        tgt_img_anti = {k: antiscia_longitude(v) for k, v in targets.items()}
        tgt_img_contra = {k: contra_antiscia_longitude(v) for k, v in targets.items()}

        # compute step
        auto = isinstance(step_minutes, str) and step_minutes.lower() == "auto"
        if auto:
            caps_min: List[int] = []
            for m in movers:
                n = (m or "").lower()
                if n == "moon": caps_min.append(10)
                elif n in ("mercury", "venus", "mars"): caps_min.append(30)
                elif n in ("sun", "jupiter", "saturn"): caps_min.append(90)
                else: caps_min.append(180)
            step_minutes = max(5, min(caps_min) if caps_min else 30)
        try:
            dt = float(step_minutes) / (60.0 * 24.0)
        except Exception:
            dt = 30.0 / (60.0 * 24.0)

        events: List[TransitEvent] = []
        dedupe: set[Tuple[str, str, str, int]] = set()
        lon_cache: Dict[Tuple[str, float], float] = {}

        def _lon_cached_local(name: str, t: float) -> float:
            return self._lon_cached(name, t, lon_cache)

        def make_sep(body: str, tgt_name: str, tgt_lon: float, spec: AspectSpec):
            if spec.kind == "zodiacal":
                ang = float(spec.angle)
                return lambda t: wrap180(angdiff(_lon_cached_local(body, t), tgt_lon) - ang)
            elif spec.kind == "antiscia":
                img = tgt_img_anti[tgt_name]
                return lambda t: wrap180(_lon_cached_local(body, t) - img)
            else:
                img = tgt_img_contra[tgt_name]
                return lambda t: wrap180(_lon_cached_local(body, t) - img)

        # preload left boundary
        t0 = float(jd_start_tt)
        l0 = self._lon_map(t0, movers)
        for m, v in l0.items():
            lon_cache[(m, t0)] = float(v)
        self._prune_lon_cache(lon_cache, max_size=self._lon_cache_max)

        while t0 < jd_end_tt - 1e-12:
            t1 = float(min(t0 + dt, jd_end_tt))
            l1 = self._lon_map(t1, movers)
            for m, v in l1.items():
                lon_cache[(m, t1)] = float(v)
            self._prune_lon_cache(lon_cache, max_size=self._lon_cache_max)

            for body in movers:
                lon0 = l0.get(body); lon1 = l1.get(body)
                if lon0 is None or lon1 is None:
                    continue
                max_change_possible = self._speed_est(body) * dt

                for tgt_name, tgt_lon in targets.items():
                    for spec in asp_list:
                        if spec.kind == "zodiacal":
                            s0 = _zodiacal_separation(lon0, tgt_lon, spec.angle)
                            s1 = _zodiacal_separation(lon1, tgt_lon, spec.angle)
                        elif spec.kind == "antiscia":
                            img = tgt_img_anti[tgt_name]; s0 = wrap180(lon0 - img); s1 = wrap180(lon1 - img)
                        else:
                            img = tgt_img_contra[tgt_name]; s0 = wrap180(lon0 - img); s1 = wrap180(lon1 - img)

                        if not (math.isfinite(s0) and math.isfinite(s1)):
                            continue
                        if abs(s0) > 120.0 and abs(s1) > 120.0:
                            continue

                        sign_change = (s0 == 0.0) or (s1 == 0.0) or ((s0 * s1) < 0.0)
                        guard = max(0.15, min(spec.orb_deg * 0.33, max_change_possible * 1.5))
                        near = (min(abs(s0), abs(s1)) <= (spec.orb_deg + guard))
                        if not (sign_change or near):
                            continue
                        if min(abs(s0), abs(s1)) > (spec.orb_deg + max_change_possible):
                            continue

                        f = make_sep(body, tgt_name, tgt_lon, spec)
                        if self.prebatch_refinement:
                            # warm cache at midpoints (reduces ephem calls inside refinement)
                            mid = 0.5 * (t0 + t1)
                            t13 = t0 + (t1 - t0) / 3.0
                            t23 = t0 + 2.0 * (t1 - t0) / 3.0
                            t38 = t0 + (t1 - t0) * 3.0 / 8.0
                            t58 = t0 + (t1 - t0) * 5.0 / 8.0
                            self._preload_lons(body, (t0, t1, mid, t13, t23, t38, t58), lon_cache)

                        t_exact = self._refine_zero_brent(f, t0, t1, s0, s1, tol_days=1e-6)
                        lon_now = _lon_cached_local(body, t_exact)
                        if spec.kind == "zodiacal":
                            sep = _zodiacal_separation(lon_now, tgt_lon, spec.angle)
                        else:
                            img_now = tgt_img_anti[tgt_name] if spec.kind == "antiscia" else tgt_img_contra[tgt_name]
                            sep = wrap180(lon_now - img_now)

                        applying = (abs(s1) < abs(s0))
                        # Deduplicate at 1-second JD buckets
                        bucket = int(math.floor(t_exact * 86400.0 + 0.5))
                        key = (body, tgt_name, spec.name, bucket)
                        if key in dedupe:
                            continue
                        dedupe.add(key)

                        events.append(TransitEvent(
                            jd_tt=float(t_exact),
                            body=body,
                            target=tgt_name,
                            aspect=spec.name,
                            kind=spec.kind,
                            separation_deg=float(sep),
                            applying=bool(applying),
                            exact=abs(sep) <= 1e-6,
                            meta={"orb_deg": spec.orb_deg, "angle": spec.angle},
                        ))

            t0 = t1
            l0 = l1

        events.sort(key=lambda e: (e.jd_tt, e.body, e.target, e.aspect))
        return events


# ─────────────────────────────────────────────────────────────────────────────
# Public transit wrapper (routes use this)
# ─────────────────────────────────────────────────────────────────────────────
def _aspects_from_kwargs(kwargs: dict) -> List[AspectSpec]:
    custom = kwargs.get("aspects")
    orbs = kwargs.get("orbs", {})
    specs: List[AspectSpec] = []
    if isinstance(custom, (list, tuple)) and custom:
        known = {a.name.lower(): a for a in (list(MAJOR_ASPECTS) + list(MINOR_ASPECTS))}
        for item in custom:
            if isinstance(item, str):
                key = item.strip().lower()
                a = known.get(key)
                if a:
                    orb = float(orbs.get(key, a.orb_deg)) if isinstance(orbs, dict) else a.orb_deg
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
    include_aspects_to: list[str] | None = None,  # reserved for houses/angles
    include_house_cusps: bool | None = None,      # reserved
    exact_timing: bool | None = None,             # reserved
    aspects: list[str] | None = None,
    orbs: Dict[str, float] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """Route-friendly wrapper. Keeps prior response shape/keys."""
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "transits": [], "meta": {}}

    frame = frame or "ecliptic-of-date"
    zodiac_mode = (zodiac_mode or "tropical").lower()
    ay = float(ayanamsa_deg or 0.0)

    tz = str(natal_chart.get("place_tz") or natal_chart.get("timezone") or "UTC")
    if start_jd_tt is None or end_jd_tt is None:
        if not time_range or len(time_range) != 2:
            return {"ok": False, "error": "time_range_required", "transits": [], "meta": {}}
        d0 = _to_date(time_range[0]); d1 = _to_date(time_range[1])
        start_jd_tt = _jd_from_date(d0, tz, _ts_resolve)
        # include end day up to 23:59
        end_jd_tt   = _jd_from_date(d1, tz, _ts_resolve) + (24*60-1) / (24*60)

    movers = list(transiting_bodies or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"])
    tgts   = list(natal_targets or natal_bodies or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"])

    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = TransitEngine(ephem=ep, frame=frame)
    eng.sidereal_mode = zodiac_mode.startswith("sidereal")
    eng.ayanamsa_deg = ay

    nat_rows = ep.ecliptic_longitudes(float(start_jd_tt), tgts).get("results", [])
    PROF["ephem_calls"] += 1
    nat_map = {row["name"]: float(row["longitude"]) for row in nat_rows} if nat_rows else {}
    targets: Dict[str, float] = {}
    if eng.sidereal_mode:
        for k, v in nat_map.items():
            val = norm360(float(v) - ay)
            if math.isfinite(val): targets[k] = val
    else:
        for k, v in nat_map.items():
            val = float(v)
            if math.isfinite(val): targets[k] = val

    specs = _aspects_from_kwargs({"aspects": aspects, "orbs": (orbs or {})})
    include_antiscia = bool(kwargs.get("include_antiscia", False))
    antiscia_orb = float(kwargs.get("antiscia_orb_deg", 2.0))
    step_arg = kwargs.get("step_minutes", "auto")

    try:
        evs = eng.scan_aspects(
            jd_start_tt=float(start_jd_tt), jd_end_tt=float(end_jd_tt),
            movers=movers, targets=targets, aspects=specs,
            step_minutes=step_arg, include_antiscia=include_antiscia, antiscia_orb_deg=antiscia_orb,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": f"transit_scan_failed:{e}",
            "transits": [],
            "meta": {"frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": ay, "prof": dict(PROF)},
        }

    # build response
    specs_by_name = {s.name.lower(): s for s in specs}
    hits: List[Dict[str, Any]] = []
    for ev in evs:
        asp_name = ev.aspect.lower()
        spec = specs_by_name.get(asp_name)
        orb_now = abs(ev.separation_deg)
        hits.append({
            "transiting_body": ev.body, "natal_body": ev.target, "aspect": asp_name,
            "orb": float(orb_now),
            "max_orb": float(spec.orb_deg if spec else 1.0),
            "applying": bool(ev.applying), "exact": bool(ev.exact),
            "exact_jd_tt": float(ev.jd_tt), "exact_jd_ut1": None, "exact_datetime_utc": None,
            "p_value": 1.0, "house": None, "exact_longitude": None,
        })

    return {
        "ok": True, "technique": "transits", "transits": hits,
        "meta": {
            "movers": movers, "natal_targets": list(targets.keys()),
            "window_jd_tt": [float(start_jd_tt), float(end_jd_tt)],
            "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": ay, "prof": dict(PROF),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ingresses — sign-boundary crossing detection
# ─────────────────────────────────────────────────────────────────────────────
def _sign_index(lon: float) -> int:
    # 0..11 (0° Aries start convention)
    return int(math.floor(norm360(lon) / 30.0)) % 12

def find_ingresses_in_range(
    *,
    start_jd_tt: float,
    end_jd_tt: float,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "tropical",
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

    movers = list(movers or ["Sun","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"])
    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = TransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "tropical").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    # step compute similar to transits (auto)
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
        # refine: zero of f(t) = wrap180((lon % 30) - 0)
        def f(t: float) -> float:
            return wrap180((lon_at(body, t) % 30.0) - 0.0)
        # compute initial values
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
                events.append({
                    "body": b,
                    "exact_jd_tt": float(t_exact),
                    "sign_to": int(sign_to),  # 0..11
                })
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
    zodiac_mode: str = "tropical",
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

    movers = list(movers or ["Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"])
    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = TransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "tropical").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    # step similar to ingresses
    auto = isinstance(step_minutes, str) and step_minutes.lower() == "auto"
    if auto:
        caps_min: List[int] = []
        for m in movers:
            n = (m or "").lower()
            if n in ("mercury","venus"): caps_min.append(60)    # fast apparent switches
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

    # finite-difference velocity (deg/day) using small symmetric delta
    def vel(body: str, t: float, h_days: float = 1.0 / (24.0 * 12.0)) -> float:
        # h ≈ 5 minutes default; safe for all movers
        l1 = lon_at(body, t - h_days)
        l2 = lon_at(body, t + h_days)
        d = wrap180(l2 - l1)
        return d / (2.0 * h_days)

    def zero_cross(body: str, t0: float, t1: float) -> Optional[Tuple[float, str]]:
        v0 = vel(body, t0); v1 = vel(body, t1)
        if not (math.isfinite(v0) and math.isfinite(v1)):
            return None
        if v0 == 0.0:
            sign = "station"  # degenerate
            return (t0, sign)
        if v1 == 0.0:
            sign = "station"
            return (t1, sign)
        if (v0 * v1) > 0.0:
            return None
        # refine root of dv/dt = 0 (velocity sign change -> direct/retro)
        def f(t: float) -> float:
            return vel(body, t)
        t_exact = eng._refine_zero_brent(f, t0, t1, v0, v1, tol_days=5e-6)
        # classify direction around station
        pre = vel(body, t_exact - 2.0 / (24.0 * 12.0))
        post = vel(body, t_exact + 2.0 / (24.0 * 12.0))
        if pre > 0.0 and post < 0.0:
            k = "retrograde"
        elif pre < 0.0 and post > 0.0:
            k = "direct"
        else:
            k = "station"
        return (t_exact, k)

    t = float(start_jd_tt)
    while t < end_jd_tt - 1e-12:
        t2 = float(min(t + dt, end_jd_tt))
        for b in movers:
            zc = zero_cross(b, t, t2)
            if zc:
                t_exact, kind = zc
                events.append({
                    "body": b,
                    "exact_jd_tt": float(t_exact),
                    "kind": kind,  # "retrograde" | "direct" | "station"
                })
        t = t2

    events.sort(key=lambda r: (r["exact_jd_tt"], r["body"]))
    return {
        "ok": True, "stations": events,
        "meta": {"movers": movers, "window_jd_tt": [float(start_jd_tt), float(end_jd_tt)],
                 "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg)}
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation / Holdout (lightweight, route-friendly)
# ─────────────────────────────────────────────────────────────────────────────
def pearson_corr(x: List[float], y: List[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return float("nan")
    sx = sy = sxx = syy = sxy = 0.0
    for i in range(n):
        xi = float(x[i]); yi = float(y[i])
        sx += xi; sy += yi
        sxx += xi * xi; syy += yi * yi; sxy += xi * yi
    mx = sx / n; my = sy / n
    vx = max(sxx / n - mx * mx, 0.0); vy = max(syy / n - my * my, 0.0)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    r = (sxy / n - mx * my) / math.sqrt(vx * vy)
    return max(-1.0, min(1.0, r))

def permutation_pvalue_corr(x: List[float], y: List[float], *, repeats: int = 2000, seed: int = 0) -> float:
    import random
    r_obs = pearson_corr(x, y)
    if not math.isfinite(r_obs):
        return 1.0
    rng = random.Random(int(seed))
    y_perm = list(y)
    count = 0
    for _ in range(max(100, int(repeats))):
        rng.shuffle(y_perm)
        r = pearson_corr(x, y_perm)
        if abs(r) >= abs(r_obs):
            count += 1
    return (count + 1) / (repeats + 1)  # add-one smoothing

def bh_fdr(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], float]:
    m = len(p_values)
    indexed = sorted([(float(p), i) for i, p in enumerate(p_values)], key=lambda t: t[0])
    thresh = 0.0
    keep = [False] * m
    for k, (p, i) in enumerate(indexed, start=1):
        t = alpha * k / m
        if p <= t:
            thresh = p
            keep[i] = True
    return keep, thresh

def evaluate_univariate(pred: List[float], truth: List[float]) -> Dict[str, Any]:
    r = pearson_corr(pred, truth)
    p = permutation_pvalue_corr(pred, truth, repeats=2000, seed=42)
    return {"r": r, "p_perm": p, "ok": math.isfinite(r)}

def holdout_replicate(features: List[List[float]], truth: List[float], *, seed: int = 0, test_frac: float = 0.33) -> Dict[str, Any]:
    import random
    n = min(len(truth), len(features))
    idx = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(idx)
    cut = max(1, int(n * max(0.05, min(0.95, float(test_frac)))))
    test_idx = set(idx[:cut])
    # simple linear comb feature (mean across features) as baseline
    pred = []
    for i in range(n):
        row = features[i]
        if row:
            pred.append(sum(float(v) for v in row) / len(row))
        else:
            pred.append(0.0)
    pred_train = [pred[i] for i in range(n) if i not in test_idx]
    truth_train = [truth[i] for i in range(n) if i not in test_idx]
    pred_test = [pred[i] for i in range(n) if i in test_idx]
    truth_test = [truth[i] for i in range(n) if i in test_idx]
    train = evaluate_univariate(pred_train, truth_train)
    test = evaluate_univariate(pred_test, truth_test)
    return {"train": train, "test": test, "n": n, "n_test": len(pred_test)}

def validate_predictions(datasets: List[Tuple[List[List[float]], List[float]]]) -> Dict[str, Any]:
    results = []
    for i, (X, y) in enumerate(datasets):
        res = holdout_replicate(X, y, seed=123 + i, test_frac=0.33)
        results.append(res)
    ok = all(r["test"]["ok"] for r in results if isinstance(r, dict) and "test" in r)
    return {"ok": ok, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder (Western): proximity to nearest transit hit
# ─────────────────────────────────────────────────────────────────────────────
def feature_transit_proximity(
    *,
    hits: List[Dict[str, Any]],
    aspect_name: Optional[str] = None,
    cap_deg: float = 8.0,
) -> List[float]:
    """
    Build a simple 0..1 proximity feature per hit (1 at exact, →0 at cap).
    If aspect_name given, filter by that aspect.
    """
    out: List[float] = []
    cap = max(1e-6, float(cap_deg))
    for h in hits:
        if aspect_name and (str(h.get("aspect", "")).lower() != str(aspect_name).lower()):
            continue
        orb = abs(float(h.get("orb", float("inf"))))
        if not math.isfinite(orb):
            continue
        score = max(0.0, 1.0 - (orb / cap))
        out.append(score)
    return out
