# app/core/western_predictive.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

import math
from datetime import datetime

from app.core.ephem_singleton import TS, PLANETS
from app.core.common_predictive import (
    norm360, wrap180, angdiff, PROF, _to_date, _jd_from_date
)

try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    _EPH_OK = False
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore

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

def antiscia_longitude(lon: float) -> float: return norm360(180.0 - lon)
def contra_antiscia_longitude(lon: float) -> float: return norm360(360.0 - lon)

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
        if len(lon_cache) <= max_size: return
        keep = max_size - max(1, max_size // 4)
        for k in sorted(lon_cache.keys(), key=lambda kk: kk[1])[: len(lon_cache) - keep]:
            lon_cache.pop(k, None)

    def __init__(self, *, ephem: Optional[EphemerisAdapter] = None, frame: str = "ecliptic-of-date",
                 topocentric: bool = False, latitude: Optional[float] = None, longitude: Optional[float] = None,
                 elevation_m: Optional[float] = None, prebatch_refinement: bool = False, lon_cache_max: Optional[int] = None):
        self.ephem = ephem or EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
        self.frame = frame
        self.obs = dict(topocentric=bool(topocentric), latitude=latitude, longitude=longitude, elevation_m=elevation_m)
        self.prebatch_refinement = bool(prebatch_refinement)
        self._lon_cache_max = int(lon_cache_max) if isinstance(lon_cache_max, int) and lon_cache_max > 256 else self._LON_CACHE_MAX
        self.sidereal_mode: bool = False
        self.ayanamsa_deg: float = 0.0

    def _lon_map(self, jd_tt: float, names: List[str]) -> Dict[str, float]:
        res = self.ephem.ecliptic_longitudes(jd_tt, names, **self.obs).get("results", [])
        PROF["ephem_calls"] += 1
        if not res: return {}
        if self.sidereal_mode:
            ay = self.ayanamsa_deg
            return {row["name"]: norm360(float(row["longitude"]) - ay) for row in res}
        return {row["name"]: float(row["longitude"]) for row in res}

    def _lon_cached(self, name: str, t: float, lon_cache: Dict[Tuple[str, float], float]) -> float:
        key = (name, float(t))
        if key in lon_cache:
            return lon_cache[key]
        got = self._lon_map(t, [name])
        if name not in got:
            raise RuntimeError(f"no ephemeris for {name}@{t}")
        lon_cache[key] = float(got[name])
        self._prune_lon_cache(lon_cache, max_size=self._lon_cache_max)
        return lon_cache[key]

    def _preload_lons(self, body: str, times: Iterable[float], lon_cache: Dict[Tuple[str, float], float]) -> None:
        for t in times:
            key = (body, float(t))
            if key in lon_cache: continue
            got = self._lon_map(float(t), [body])
            if body in got: lon_cache[key] = float(got[body])
        self._prune_lon_cache(lon_cache, max_size=self._lon_cache_max)

    @staticmethod
    def _refine_zero_brent(f, a, b, fa, fb, *, max_iter=32, tol_days=1e-6) -> float:
        from app.core.common_predictive import PROF as _PROF
        _PROF["refinements"] += 1
        if fa == 0.0: return a
        if fb == 0.0: return b
        if fa * fb > 0.0:
            aa, bb = a, b
            for _ in range(max_iter):
                m = 0.5 * (aa + bb)
                fm = f(m)
                if fm == 0.0 or (bb - aa) <= tol_days: return m
                if fa * fm <= 0: bb, fb = m, fm
                else: aa, fa = m, fm
            return 0.5 * (aa + bb)
        c, fc = a, fa
        d = e = b - a
        for _ in range(max_iter):
            if fb == 0.0: return b
            if abs(fa) < abs(fb): a, b = b, a; fa, fb = fb, fa
            m = 0.5 * (a + b); tol = tol_days
            if abs(b - a) <= tol: return b
            if fa != fc and fb != fc:
                s = (a*fb*fc)/((fa - fb)*(fa - fc)) + (b*fa*fc)/((fb - fa)*(fb - fc)) + (c*fa*fb)/((fc - fa)*(fc - fb))
            else:
                s = b - fb*(b - a)/(fb - fa)
            cond = not ((3*a + b)/4 < s < b if a < b else b < s < (3*a + b)/4)
            cond |= (e and abs(s - b) >= abs(e)/2)
            cond |= (not e and abs(s - b) >= abs(d)/2)
            cond |= (abs(e) < tol)
            cond |= (abs(d) < tol)
            if cond: s = m; d = e = b - a
            else: d, e = e, b - s
            fs = f(s)
            c, fc = a, fa
            if (fa * fs) < 0: b, fb = s, fs
            else: a, fa = s, fs
            if abs(fa) < abs(fb): a, b = b, a; fa, fb = fb, fa
        return b

    def scan_aspects(self, *, jd_start_tt: float, jd_end_tt: float, movers: List[str], targets: Dict[str, float],
                     aspects: Iterable[AspectSpec] = MAJOR_ASPECTS, step_minutes: float = "auto",
                     include_antiscia: bool = False, antiscia_orb_deg: float = 2.0) -> List[TransitEvent]:
        if jd_end_tt <= jd_start_tt: return []
        asp_list: List[AspectSpec] = list(aspects)
        if include_antiscia:
            asp_list.append(AspectSpec("Antiscia", 0.0, antiscia_orb_deg, kind="antiscia"))
            asp_list.append(AspectSpec("Contra-Antiscia", 0.0, antiscia_orb_deg, kind="contra-antiscia"))

        tgt_img_anti = {k: antiscia_longitude(v) for k, v in targets.items()}
        tgt_img_contra = {k: contra_antiscia_longitude(v) for k, v in targets.items()}

        auto = (isinstance(step_minutes, str) and step_minutes.lower() == "auto") or (float(step_minutes) <= 0.0)
        if auto:
            caps_min = []
            for m in movers:
                n = m.lower()
                if n == "moon": caps_min.append(10)
                elif n in ("mercury","venus","mars"): caps_min.append(30)
                elif n in ("sun","jupiter","saturn"): caps_min.append(90)
                else: caps_min.append(180)
            step_minutes = max(5, min(caps_min) if caps_min else 30)
        dt = float(step_minutes) / (60.0 * 24.0)

        events: List[TransitEvent] = []
        dedupe: set[Tuple[str, str, str, int]] = set()
        lon_cache: Dict[Tuple[str, float], float] = {}

        def _lon_cached(name: str, t: float) -> float:
            return self._lon_cached(name, t, lon_cache)

        def make_sep(body: str, tgt_name: str, tgt_lon: float, spec: AspectSpec):
            if spec.kind == "zodiacal":
                ang = float(spec.angle)
                return lambda t: wrap180(angdiff(_lon_cached(body, t), tgt_lon) - ang)
            elif spec.kind == "antiscia":
                img = tgt_img_anti[tgt_name]
                return lambda t: wrap180(_lon_cached(body, t) - img)
            else:
                img = tgt_img_contra[tgt_name]
                return lambda t: wrap180(_lon_cached(body, t) - img)

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
                if lon0 is None or lon1 is None: continue
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
                        if not (math.isfinite(s0) and math.isfinite(s1)): continue
                        if abs(s0) > 120.0 and abs(s1) > 120.0: continue
                        sign_change = (s0 == 0.0) or (s1 == 0.0) or ((s0 * s1) < 0.0)
                        guard = max(0.15, min(spec.orb_deg * 0.33, max_change_possible * 1.5))
                        near = (min(abs(s0), abs(s1)) <= (spec.orb_deg + guard))
                        if not (sign_change or near): continue
                        if min(abs(s0), abs(s1)) > (spec.orb_deg + max_change_possible): continue

                        f = make_sep(body, tgt_name, tgt_lon, spec)
                        if self.prebatch_refinement:
                            mid = 0.5 * (t0 + t1)
                            t13 = t0 + (t1 - t0) / 3.0
                            t23 = t0 + 2.0 * (t1 - t0) / 3.0
                            t38 = t0 + (t1 - t0) * 3.0 / 8.0
                            t58 = t0 + (t1 - t0) * 5.0 / 8.0
                            self._preload_lons(body, (t0, t1, mid, t13, t23, t38, t58), lon_cache)

                        t_exact = self._refine_zero_brent(f, t0, t1, s0, s1, tol_days=1e-6)
                        lon_now = _lon_cached(body, t_exact)
                        if spec.kind == "zodiacal":
                            sep = _zodiacal_separation(lon_now, tgt_lon, spec.angle)
                        else:
                            img_now = tgt_img_anti[tgt_name] if spec.kind == "antiscia" else tgt_img_contra[tgt_name]
                            sep = wrap180(lon_now - img_now)
                        applying = (abs(s1) < abs(s0))
                        bucket = int(round(t_exact * 86400.0))
                        key = (body, tgt_name, spec.name, bucket)
                        if key in dedupe: continue
                        dedupe.add(key)
                        events.append(TransitEvent(
                            jd_tt=float(t_exact), body=body, target=tgt_name, aspect=spec.name, kind=spec.kind,
                            separation_deg=float(sep), applying=bool(applying), exact=abs(sep) <= 1e-6,
                            meta={"orb_deg": spec.orb_deg, "angle": spec.angle},
                        ))

            t0 = t1
            l0 = l1

        events.sort(key=lambda e: (e.jd_tt, e.body, e.target, e.aspect))
        return events

# Public wrapper used by routes
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
    *, natal_chart: dict, start_jd_tt: float | None = None, end_jd_tt: float | None = None,
    time_range: tuple | list | None = None, transiting_bodies: list[str] | None = None,
    natal_targets: list[str] | None = None, natal_bodies: list[str] | None = None,
    frame: str | None = None, zodiac_mode: str | None = None, ayanamsa_deg: float | None = None,
    include_aspects_to: list[str] | None = None, include_house_cusps: bool | None = None,
    exact_timing: bool | None = None, aspects: list[str] | None = None, orbs: Dict[str, float] | None = None, **kwargs,
) -> Dict[str, Any]:
    import time as _time
    from app.core.common_predictive import _ts_resolve
    t0_wall = _time.perf_counter()
    frame = frame or "ecliptic-of-date"
    zodiac_mode = (zodiac_mode or "tropical").lower()
    ay = float(ayanamsa_deg or 0.0)

    tz = str(natal_chart.get("place_tz") or natal_chart.get("timezone") or "UTC")
    if start_jd_tt is None or end_jd_tt is None:
        if not time_range or len(time_range) != 2:
            return {"ok": False, "error": "time_range_required", "transits": [], "meta": {}}
        d0 = _to_date(time_range[0]); d1 = _to_date(time_range[1])
        start_jd_tt = _jd_from_date(d0, tz, _ts_resolve)
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

    hits: List[Dict[str, Any]] = []
    for ev in evs:
        asp_name = ev.aspect.lower()
        orb_now = abs(ev.separation_deg)
        hits.append({
            "transiting_body": ev.body, "natal_body": ev.target, "aspect": asp_name,
            "orb": float(orb_now),
            "max_orb": float(next((s.orb_deg for s in specs if s.name.lower() == asp_name), 1.0)),
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
