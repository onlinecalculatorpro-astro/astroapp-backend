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
- Deterministic ephemeris batching & JD→longitude caching
- Brent-like safeguarded refinement in JD days
- Degree-precise gochar (not whole-sign only), with lineage-friendly knobs
- Clean integration with constants_vedic and common_predictive utilities

Changelog vs previous version:
- FIX: Natal targets now come from natal epoch (longitudes map or natal JD),
       not from the gochar window start.
- PERF: Lean LRU for lon cache, smarter auto steps, guarded preloads,
        fewer adapter calls, cheaper velocity sampling for stations.
- API:  Optional `target_lon_map` override; consistent node naming; safer PROF.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, Callable, Union
import math
from collections import OrderedDict

# Timescales / Ephemeris backbone
from app.core.ephem_singleton import TS, PLANETS
from app.core.common_predictive import (
    norm360, wrap180, angdiff, PROF, _to_date, _jd_from_date, _ts_resolve
)
from app.core.constants_vedic import (
    graha_drishti_schema, drishti_strength_factor, nakshatra_index, NAKSHATRAS_27
)

# Optional high-precision civil→JD helpers (used if available for natal JD)
try:
    from app.core.validators import normalize_chart_payload as _normalize_chart_payload  # type: ignore
except Exception:
    _normalize_chart_payload = None  # type: ignore

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
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────
def _safe_prof_inc(key: str, inc: int = 1) -> None:
    try:
        PROF[key] = int(PROF.get(key, 0)) + int(inc)
    except Exception:
        pass

def _sign_index(lon: float) -> int:
    return int(math.floor(norm360(lon) / 30.0)) % 12

def _node_canon(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    nl = n.lower()
    if nl in ("north node", "rahu"):
        return "Rahu"
    if nl in ("south node", "ketu"):
        return "Ketu"
    return n

def _batch_map_nodes(q: List[str]) -> List[str]:
    out: List[str] = []
    for n in q:
        nl = (n or "").lower()
        if nl == "rahu": out.append("North Node")
        elif nl == "ketu": out.append("South Node")
        else: out.append(n)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Engine (caching, ephemeris, refinement)
# ─────────────────────────────────────────────────────────────────────────────
class _LRUCache(OrderedDict):
    """Tiny LRU for (name, jd_tt) → longitude degrees."""
    def __init__(self, maxsize: int = 2000):
        super().__init__()
        self.maxsize = int(max(256, maxsize))

    def get_put(self, key: Tuple[str, float], getter: Callable[[], float]) -> float:
        if key in self:
            v = super().pop(key)  # move to end
            super().__setitem__(key, v)
            return v
        v = float(getter())
        super().__setitem__(key, v)
        if len(self) > self.maxsize:
            # drop ~25% oldest in one go (amortized)
            drop = max(1, self.maxsize // 4)
            for _ in range(drop):
                try:
                    self.popitem(last=False)
                except KeyError:
                    break
        return v


class VedicTransitEngine:
    """
    Performance-aware Vedic gochar finder:
    - Batch ephemeris per time boundary for all movers
    - LRU cache for lon(body, t) with amortized pruning
    - Brent-like zero finder (guarded) with JD-day tolerance
    - Sidereal mode enabled by default (ayanāṁśa subtract)
    """
    _DEFAULT_LRU = 2000

    @staticmethod
    def _speed_est(body: str) -> float:
        b = (body or "").lower()
        if b == "moon": return 14.0
        if b in ("mercury", "venus"): return 1.6
        if b == "mars": return 0.9
        if b == "sun": return 1.0
        if b in ("jupiter", "saturn"): return 0.2
        return 0.1

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
        self._lon_cache = _LRUCache(int(lon_cache_max) if isinstance(lon_cache_max, int) and lon_cache_max > 256 else self._DEFAULT_LRU)
        self.sidereal_mode: bool = True
        self.ayanamsa_deg: float = 0.0
        self.treat_nodes_like_saturn = bool(treat_nodes_like_saturn)

    # ephemeris wrappers
    def _lon_map(self, jd_tt: float, names: List[str]) -> Dict[str, float]:
        req = _batch_map_nodes(names)
        res = self.ephem.ecliptic_longitudes(float(jd_tt), req, **self.obs).get("results", [])
        _safe_prof_inc("ephem_calls", 1)
        got: Dict[str, float] = {}
        for row in res or []:
            nm = _node_canon(str(row["name"]))
            got[nm] = float(row["longitude"])
        if self.sidereal_mode:
            ay = self.ayanamsa_deg
            return {k: norm360(v - ay) for k, v in got.items()}
        return got

    def _lon_cached(self, name: str, t: float) -> float:
        key = (_node_canon(name), float(t))
        return self._lon_cache.get_put(key, lambda: self._lon_map(t, [key[0]])[key[0]])

    def _preload_lons(self, body: str, times: Iterable[float]) -> None:
        name = _node_canon(body)
        miss: List[float] = []
        for t in times:
            key = (name, float(t))
            if key not in self._lon_cache:
                miss.append(float(t))
        if not miss:
            return
        got = self._lon_map(miss[0], [name]) if len(miss) == 1 else None  # warm one
        if got and name in got:
            self._lon_cache.get_put((name, float(miss[0])), lambda: got[name])
        for t in miss[1:]:
            _ = self._lon_cached(name, float(t))  # populate by normal path

    # safeguarded root finder (Brent-like)
    @staticmethod
    def _refine_zero_brent(f: Callable[[float], float], a: float, b: float, fa: float, fb: float, *, max_iter: int = 32, tol_days: float = 1e-6) -> float:
        from app.core.common_predictive import PROF as _PROF
        try:
            _PROF["refinements"] = int(_PROF.get("refinements", 0)) + 1
        except Exception:
            pass

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
            p = _node_canon(planet)
            if p in ("Rahu", "Ketu") and self.treat_nodes_like_saturn:
                p = "Saturn"
            sch = graha_drishti_schema(p)
            if _node_canon(planet) in ("Rahu", "Ketu") and not (self.treat_nodes_like_saturn or include_nodes):
                sch = ({7: 1.0} if include_nodes else {})
            return sch

        # step selection (tighter for faster movers)
        auto = isinstance(step_minutes, str) and step_minutes.lower() == "auto"
        if auto:
            caps_min: List[int] = []
            for m in movers:
                n = (m or "").lower()
                if n == "moon": caps_min.append(8)
                elif n in ("mercury","venus","mars"): caps_min.append(20)
                elif n in ("sun","jupiter","saturn"): caps_min.append(60)
                else: caps_min.append(120)
            step_minutes = max(4, min(caps_min) if caps_min else 20)
        try:
            dt = float(step_minutes) / (60.0 * 24.0)
        except Exception:
            dt = 20.0 / (60.0 * 24.0)

        events: List[GocharEvent] = []
        dedupe: set[Tuple[str, str, str, int]] = set()

        # preload left boundary in one batch
        t0 = float(jd_start_tt)
        l0 = self._lon_map(t0, movers)
        for m, v in l0.items():
            _ = self._lon_cache.get_put((_node_canon(m), t0), lambda vv=v: float(vv))

        def _lon_cached_local(name: str, t: float) -> float:
            return self._lon_cached(name, t)

        # scanning
        while t0 < jd_end_tt - 1e-12:
            t1 = float(min(t0 + dt, jd_end_tt))
            l1 = self._lon_map(t1, movers)
            for m, v in l1.items():
                _ = self._lon_cache.get_put((_node_canon(m), t1), lambda vv=v: float(vv))

            for body in movers:
                if _node_canon(body) in ("Rahu","Ketu") and not (include_nodes or self.treat_nodes_like_saturn):
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
                        guard = max(0.12, min(_orb * 0.33, max_change_possible * 1.4))
                        near = (min(abs(s0), abs(s1)) <= (_orb + guard))
                        if not (sign_change or near):
                            continue
                        if min(abs(s0), abs(s1)) > (_orb + max_change_possible):
                            continue

                        f = make_sep(body, tgt_lon, axis)
                        if self.prebatch_refinement:
                            # inexpensive preload at a couple of interior points
                            mid = 0.5 * (t0 + t1)
                            self._preload_lons(body, (t0, mid, t1))

                        t_exact = self._refine_zero_brent(f, t0, t1, s0, s1, tol_days=1e-6)
                        lon_now = _lon_cached_local(body, t_exact)
                        sep = wrap180(angdiff(lon_now, tgt_lon) - axis)
                        applying = (abs(s1) < abs(s0))
                        # dedupe at 1-second buckets
                        bucket = int(math.floor(t_exact * 86400.0 + 0.5))
                        key = (_node_canon(body), _node_canon(tgt_name), f"{k}th", bucket)
                        if key in dedupe:
                            continue
                        dedupe.add(key)

                        events.append(GocharEvent(
                            jd_tt=float(t_exact),
                            body=_node_canon(body),
                            target=_node_canon(tgt_name),
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
def _pick_natal_targets_map(
    *,
    ep: EphemerisAdapter,
    eng: VedicTransitEngine,
    natal_chart: Dict[str, Any],
    tgts: List[str],
    ay: float,
) -> Dict[str, float]:
    """
    Select natal target longitudes (sidereal/tropical aligned with engine).
    Preference order:
    1) Explicit longitudes: natal_chart["longitudes"] or ["ecliptic_longitudes"]
    2) natal_chart["natal_jd_tt"] or ["jd_tt"] or ["jd_utc"] (converted)
    3) Derive from natal date/time/place via validators (if available)
    """
    # (1) explicit maps
    for key in ("longitudes", "ecliptic_longitudes"):
        m = natal_chart.get(key)
        if isinstance(m, dict) and m:
            got = {_node_canon(k): float(v) for k, v in m.items() if k is not None}
            return {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in got.items()}

    # (2) JD first
    natal_jd_tt = None
    if isinstance(natal_chart.get("natal_jd_tt"), (int, float)):
        natal_jd_tt = float(natal_chart["natal_jd_tt"])
    elif isinstance(natal_chart.get("jd_tt"), (int, float)):
        natal_jd_tt = float(natal_chart["jd_tt"])
    elif isinstance(natal_chart.get("jd_utc"), (int, float)):
        # Assume jd_utc ≈ jd_tt for fallback (small error; acceptable as last resort)
        natal_jd_tt = float(natal_chart["jd_utc"])

    if isinstance(natal_jd_tt, float):
        rows = ep.ecliptic_longitudes(float(natal_jd_tt), _batch_map_nodes(tgts)).get("results", [])
        _safe_prof_inc("ephem_calls", 1)
        got = {_node_canon(str(r["name"])): float(r["longitude"]) for r in rows or []}
        return {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in got.items()}

    # (3) Try to derive via validator if available (civil inputs in chart)
    if callable(_normalize_chart_payload):
        try:
            norm, _, _ = _normalize_chart_payload(
                dict(
                    date=natal_chart.get("date") or natal_chart.get("dob"),
                    time=natal_chart.get("time") or natal_chart.get("tob"),
                    place=natal_chart.get("place") or natal_chart.get("location") or "",
                    tz_name=natal_chart.get("place_tz") or natal_chart.get("timezone") or "UTC",
                ),
                compute_timescales=True,
                include_jd_utc=True,
            )
            natal_jd_tt = float(norm.get("timescales", {}).get("jd_tt"))
            if isinstance(natal_jd_tt, float):
                rows = ep.ecliptic_longitudes(natal_jd_tt, _batch_map_nodes(tgts)).get("results", [])
                _safe_prof_inc("ephem_calls", 1)
                got = {_node_canon(str(r["name"])): float(r["longitude"]) for r in rows or []}
                return {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in got.items()}
        except Exception:
            pass

    # (4) Fallback: compute targets at left window boundary (worst-case; warn)
    _safe_prof_inc("natal_targets_fallback", 1)
    return {}


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
    target_lon_map: Dict[str, float] | None = None,
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
        # cover full end date till 23:59
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

    # Natal targets (fixed at natal epoch)
    if isinstance(target_lon_map, dict) and target_lon_map:
        nat_map = {_node_canon(k): float(v) for k, v in target_lon_map.items()}
        targets = {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in nat_map.items()}
    else:
        targets = _pick_natal_targets_map(ep=ep, eng=eng, natal_chart=natal_chart, tgts=tgts, ay=ay)
        if not targets:
            # last-ditch: compute at window start (warn via PROF)
            _safe_prof_inc("gochar_fallback_targets_used", 1)
            rows = ep.ecliptic_longitudes(float(start_jd_tt), _batch_map_nodes(tgts)).get("results", [])
            _safe_prof_inc("ephem_calls", 1)
            nat_map = {_node_canon(str(r["name"])): float(r["longitude"]) for r in rows or []}
            targets = {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in nat_map.items()}

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
            "exact_jd_ut1": None,        # fill via timescales pipeline if desired
            "exact_datetime_utc": None,  # idem
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
    lon_cache = _LRUCache()

    def lon_at(body: str, t: float) -> float:
        return eng._lon_cache.get_put((_node_canon(body), float(t)), lambda: eng._lon_map(float(t), [_node_canon(body)])[_node_canon(body)])

    def sign_change(body: str, t0: float, t1: float) -> Optional[Tuple[float, int]]:
        l0 = lon_at(body, t0); l1 = lon_at(body, t1)
        s0 = _sign_index(l0); s1 = _sign_index(l1)
        if s0 == s1:
            return None
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
                events.append({"body": _node_canon(b), "exact_jd_tt": float(t_exact), "sign_to": int(sign_to)})
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
    def lon_at(body: str, t: float) -> float:
        return eng._lon_cached(body, float(t))

    def nak_change(body: str, t0: float, t1: float) -> Optional[Tuple[float, int]]:
        l0 = lon_at(body, t0); l1 = lon_at(body, t1)
        n0 = nakshatra_index(l0); n1 = nakshatra_index(l1)
        if n0 == n1:
            return None
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
                events.append({"body": _node_canon(b), "exact_jd_tt": float(t_exact), "nakshatra_index": int(idx), "nakshatra_name": NAKSHATRAS_27[(idx-1)%27]})
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

    def lon_at(body: str, t: float) -> float:
        return eng._lon_cached(body, float(t))

    # Smaller half-window for central difference, scaled with dt
    def vel(body: str, t: float, h_days: float) -> float:
        l1 = lon_at(body, t - h_days)
        l2 = lon_at(body, t + h_days)
        d = wrap180(l2 - l1)
        return d / (2.0 * h_days)

    def zero_cross(body: str, t0: float, t1: float) -> Optional[Tuple[float, str]]:
        # sample velocity at endpoints with modest h
        h = max(1.0 / (24.0 * 24.0), dt * 0.5)  # >= 1 hour window, or half step
        v0 = vel(body, t0, h); v1 = vel(body, t1, h)
        if not (math.isfinite(v0) and math.isfinite(v1)):
            return None
        if v0 == 0.0: return (t0, "station")
        if v1 == 0.0: return (t1, "station")
        if (v0 * v1) > 0.0: return None
        def f(t: float) -> float: return vel(body, t, h)
        t_exact = eng._refine_zero_brent(f, t0, t1, v0, v1, tol_days=5e-6)
        pre = vel(body, t_exact - 2.0 * h, h)
        post = vel(body, t_exact + 2.0 * h, h)
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
                events.append({"body": _node_canon(b), "exact_jd_tt": float(t_exact), "kind": kind})
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
        try:
            orb = abs(float(h.get("orb", float("inf"))))
        except Exception:
            continue
        if not math.isfinite(orb):
            continue
        score = max(0.0, 1.0 - (orb / cap))
        out.append(score)
    return out
