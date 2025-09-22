# app/core/vedic_gochar.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Vedic Gochar (Transits) — astronomy.py–compatible time model.

Public surface (used by routes):
  • gochar_drishti(**kwargs)
  • ingresses_rashi(**kwargs)
  • ingresses_nakshatra(**kwargs)
  • stations_retro_direct(**kwargs)
  • feature_drishti_proximity(hits, cap_deg)

Internals (can be used elsewhere):
  • find_gochar_in_range(...)
  • find_rashi_ingresses_in_range(...)
  • find_nakshatra_ingresses_in_range(...)
  • find_stations_in_range(...)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, Callable, Union
import os
import math
from collections import OrderedDict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Ephemeris backbone: shared singletons
try:
    from app.core.ephem_singleton import TS, PLANETS  # Skyfield TimeScale + bodies
except Exception:
    TS = None
    PLANETS = None

try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    _EPH_OK = False
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore

# Optional helpers (drishti schema + nakshatras)
try:
    from app.core.constants_vedic import (
        graha_drishti_schema, drishti_strength_factor,
        nakshatra_index, NAKSHATRAS_27
    )
except Exception:
    def graha_drishti_schema(name: str) -> Dict[int, float]:  # type: ignore
        n = (name or "").strip().lower()
        base = {7: 1.0}
        if n == "mars": base.update({4: 0.75, 8: 0.75})
        elif n == "jupiter": base.update({5: 0.75, 9: 0.75})
        elif n == "saturn": base.update({3: 0.75, 10: 0.75})
        return base
    def drishti_strength_factor(_: str, __: int) -> float:  # type: ignore
        return 1.0
    def nakshatra_index(lon: float) -> int:  # type: ignore
        # 27 equal arcs across 360°
        w = 360.0 / 27.0
        return int(math.floor((lon % 360.0) / w)) + 1
    NAKSHATRAS_27 = tuple(f"Nakshatra {i+1}" for i in range(27))  # type: ignore

# Civil→JD helpers (astronomy.py family)
try:
    from app.core import time_kernel as _tk  # preferred flexible adapter
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts  # optional helpers (ΔT etc.)
except Exception:
    _ts = None


# ──────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────
_NODE_ALIAS = {
    "north node": "North Node", "rahu": "Rahu", "true node": "North Node",
    "mean node": "North Node",  "south node": "South Node", "ketu": "Ketu",
}
def _node_canon(nm: str) -> str:
    if not nm: return nm
    key = nm.strip().lower()
    return _NODE_ALIAS.get(key, nm)

def _batch_map_nodes(q: Iterable[str]) -> List[str]:
    out: List[str] = []
    for n in q:
        nl = (n or "").strip().lower()
        if nl == "rahu": out.append("Rahu")
        elif nl == "ketu": out.append("Ketu")
        else: out.append(n)
    return out

def norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    if r < 0.0: r += 360.0
    return 0.0 if abs(r) < 1e-12 else r

def wrap180(x: float) -> float:
    return ((float(x) + 180.0) % 360.0) - 180.0

def angdiff(a2: float, a1: float) -> float:
    return ((a2 - a1 + 540.0) % 360.0) - 180.0


# ──────────────────────────────────────────────────────────────────────
# Timescale model (no “resolver unavailable” surprises)
# ──────────────────────────────────────────────────────────────────────
def _env_dut1_seconds() -> float:
    for k in ("ASTRO_DUT1_BROADCAST", "ASTRO_DUT1", "OCP_DUT1_SECONDS"):
        v = os.getenv(k)
        if v not in (None, ""):
            try: return float(v)
            except Exception: pass
    return 0.0

def _clamp_dut1(x: float) -> float:
    return max(-0.9, min(0.9, float(x)))

def _jd_tt_from_utc_jd(ju: float, y: int, m: int) -> float:
    if _ts and hasattr(_ts, "jd_tt_from_utc_jd"):
        try: return float(_ts.jd_tt_from_utc_jd(float(ju), int(y), int(m)))
        except Exception: pass
    # astronomy.py fallback: ΔT ≈ 69 s
    return float(ju) + (69.0 / 86400.0)

def _jd_utc_via_stdlib(d: str, t: str, tz: str) -> float:
    # Normalize time to HH:MM:SS
    parts = (t or "").split(":")
    timestr = t if len(parts) >= 3 else ((t + ":00") if len(parts) == 2 else (t + ":00:00"))
    dt_local = datetime.fromisoformat(f"{d}T{timestr}").replace(tzinfo=ZoneInfo(tz))
    dt_utc = dt_local.astimezone(timezone.utc)
    Y, M, D = dt_utc.year, dt_utc.month, dt_utc.day
    h = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600 + dt_utc.microsecond/3.6e9
    if M <= 2: Y -= 1; M += 12
    A = Y // 100
    B = 2 - A + A // 4
    JD0 = int(365.25*(Y + 4716)) + int(30.6001*(M + 1)) + D + B - 1524.5
    return JD0 + h/24.0

def _civil_to_jd_utc(date: str, time: str, tz: str) -> float:
    # Preferred: time_kernel with flexible signatures
    if _tk:
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
            fn = getattr(_tk, fname, None)
            if not callable(fn): continue
            try:
                out = fn(date=date, time=time, tz=tz)  # type: ignore[call-arg]
            except Exception:
                try: out = fn(date, time, tz)  # type: ignore[misc]
                except Exception: out = None
            if isinstance(out, dict):
                ju = out.get("jd_utc", out.get("jd_ut"))
                if isinstance(ju, (int, float)): return float(ju)
            if isinstance(out, (list, tuple)) and out and isinstance(out[0], (int,float)):
                return float(out[0])
    # Next: timescales helper
    if _ts and hasattr(_ts, "julian_day_utc"):
        try: return float(_ts.julian_day_utc(date, time, tz))
        except Exception: pass
    # Fallback: stdlib
    return _jd_utc_via_stdlib(date, time, tz)

def _first_of_day_jd_utc(date: str, tz: str) -> float:
    return _civil_to_jd_utc(date, "00:00:00", tz)

def _last_of_day_jd_utc(date: str, tz: str) -> float:
    return _civil_to_jd_utc(date, "23:59:59", tz)

def _parse_dates_from_body(body: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    # Supports: time_range[0,1], or date_from/date_to, or from/to
    if isinstance(body.get("time_range"), (list, tuple)) and len(body["time_range"]) >= 2:
        return str(body["time_range"][0]), str(body["time_range"][1])
    d0 = (body.get("date_from") or body.get("from") or None)
    d1 = (body.get("date_to")   or body.get("to")   or None)
    return (str(d0) if d0 else None, str(d1) if d1 else None)

def _tz_from_body(body: Dict[str, Any]) -> str:
    tz = body.get("place_tz") or body.get("tz") or "UTC"
    tzs = str(tz).strip()
    return tzs if tzs else "UTC"

def _window_from_body_to_jd_tt(body: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Convert any accepted window shape into (start_jd_tt, end_jd_tt).
    Returns meta with jd_utc window, tz, dut1_seconds used.
    """
    # Numeric fast-paths
    if isinstance(body.get("jd_tt_window"), (list, tuple)) and len(body["jd_tt_window"]) == 2:
        try:
            a, b = map(float, body["jd_tt_window"])
            return a, b, {"from": "jd_tt_window"}
        except Exception:
            pass
    if isinstance(body.get("start_jd_tt"), (int, float)) and isinstance(body.get("end_jd_tt"), (int, float)):
        return float(body["start_jd_tt"]), float(body["end_jd_tt"]), {"from": "start_end_jd_tt"}

    # Civil path
    tz = _tz_from_body(body)
    d0, d1 = _parse_dates_from_body(body)
    if not (isinstance(d0, str) and isinstance(d1, str) and d0.strip() and d1.strip()):
        return None, None, {"error": "time_range_required"}

    try:
        ju0 = _first_of_day_jd_utc(d0, tz)
        ju1 = _last_of_day_jd_utc(d1, tz)
    except Exception as e:
        return None, None, {"error": f"jd_utc_failed:{e}"}

    # TT via helper (+69s fallback baked in _jd_tt_from_utc_jd)
    try: Y0, M0 = map(int, d0.split("-")[:2])
    except Exception: Y0, M0 = 2000, 1
    try: Y1, M1 = map(int, d1.split("-")[:2])
    except Exception: Y1, M1 = 2000, 1

    jt0 = _jd_tt_from_utc_jd(ju0, Y0, M0)
    jt1 = _jd_tt_from_utc_jd(ju1, Y1, M1)

    # Deterministic UT1 (meta trace only)
    dut1 = body.get("dut1") if isinstance(body.get("dut1"), (int,float)) else body.get("dut1_seconds")
    if isinstance(dut1, (int, float, str)) and str(dut1).strip() != "":
        try: dut1_used = _clamp_dut1(float(dut1))
        except Exception: dut1_used = _clamp_dut1(_env_dut1_seconds())
    else:
        dut1_used = _clamp_dut1(_env_dut1_seconds())

    meta = {"from": "civil", "tz": tz, "jd_utc_window": [float(ju0), float(ju1)], "dut1_seconds": float(dut1_used)}
    return float(jt0), float(jt1), meta


# ──────────────────────────────────────────────────────────────────────
# Event model + engine
# ──────────────────────────────────────────────────────────────────────
DrishtiKind = Literal["7th", "3rd", "4th", "5th", "8th", "9th", "10th"]

@dataclass
class GocharEvent:
    jd_tt: float
    body: str
    target: str
    drishti: DrishtiKind
    axis_deg: float
    separation_deg: float
    applying: bool
    exact: bool
    weight: float
    meta: Dict[str, Any]

class _LRUCache(OrderedDict):
    """Tiny LRU for (name, jd_tt) → longitude degrees."""
    def __init__(self, maxsize: int = 2000):
        super().__init__()
        self.maxsize = int(max(256, maxsize))
    def get_put(self, key: Tuple[str, float], getter: Callable[[], float]) -> float:
        if key in self:
            v = super().pop(key); super().__setitem__(key, v); return v
        v = float(getter()); super().__setitem__(key, v)
        if len(self) > self.maxsize:
            drop = max(1, self.maxsize // 4)
            for _ in range(drop):
                try: self.popitem(last=False)
                except KeyError: break
        return v

class VedicTransitEngine:
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
        self._lon_cache = _LRUCache(int(lon_cache_max) if isinstance(lon_cache_max, int) and lon_cache_max > 256 else 2000)
        self.sidereal_mode: bool = True
        self.ayanamsa_deg: float = 0.0
        self.treat_nodes_like_saturn = bool(treat_nodes_like_saturn)

    # ephemeris wrappers
    def _lon_map(self, jd_tt: float, names: List[str]) -> Dict[str, float]:
        req = _batch_map_nodes(names)
        rows = self.ephem.ecliptic_longitudes(float(jd_tt), req, **self.obs).get("results", [])  # type: ignore
        got: Dict[str, float] = {}
        for r in rows or []:
            nm = _node_canon(str(r["name"]))
            got[nm] = float(r["longitude"])
        if self.sidereal_mode:
            ay = self.ayanamsa_deg
            return {k: norm360(v - ay) for k, v in got.items()}
        return got

    def _lon_cached(self, name: str, t: float) -> float:
        key = (_node_canon(name), float(t))
        return self._lon_cache.get_put(key, lambda: self._lon_map(t, [key[0]])[key[0]])

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
    def _refine_zero_brent(f: Callable[[float], float], a: float, b: float, fa: float, fb: float, *, max_iter: int = 32, tol_days: float = 1e-6) -> float:
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

    # Graha dṛṣṭi scan
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

        def _schema_for(planet: str) -> Dict[int, float]:
            p = _node_canon(planet)
            if p in ("Rahu", "Ketu") and self.treat_nodes_like_saturn:
                p = "Saturn"
            sch = graha_drishti_schema(p)
            if _node_canon(planet) in ("Rahu", "Ketu") and not (self.treat_nodes_like_saturn or include_nodes):
                sch = ({7: 1.0} if include_nodes else {})
            return sch

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

        # preload
        t0 = float(jd_start_tt)
        l0 = self._lon_map(t0, movers)
        for m, v in l0.items():
            _ = self._lon_cache.get_put((_node_canon(m), t0), lambda vv=v: float(vv))

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
                        def fsep(t: float) -> float:
                            return wrap180(angdiff(self._lon_cached(body, t), tgt_lon) - axis)
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
                        t_exact = self._refine_zero_brent(fsep, t0, t1, s0, s1, tol_days=1e-6)
                        lon_now = self._lon_cached(body, t_exact)
                        sep = wrap180(angdiff(lon_now, tgt_lon) - axis)
                        applying = (abs(s1) < abs(s0))
                        bucket = int(math.floor(t_exact * 86400.0 + 0.5))
                        key = (_node_canon(body), _node_canon(tgt_name), f"{k}th", bucket)
                        if key in dedupe:
                            continue
                        dedupe.add(key)
                        events.append(GocharEvent(
                            jd_tt=float(t_exact),
                            body=_node_canon(body),
                            target=_node_canon(tgt_name),
                            drishti=(f"{k}th" if k != 7 else "7th"),
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


# ──────────────────────────────────────────────────────────────────────
# Public scanners
# ──────────────────────────────────────────────────────────────────────
def _resolve_window_or_error(body: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    a, b, meta = _window_from_body_to_jd_tt(body)
    if not (isinstance(a, float) and isinstance(b, float) and b > a):
        return None, None, {"error": meta.get("error", "invalid_time_window")}
    return float(a), float(b), meta

def _pick_natal_targets_map(
    *, ep: EphemerisAdapter, eng: VedicTransitEngine, natal_chart: Dict[str, Any],
    tgts: List[str], ay: float
) -> Dict[str, float]:
    # explicit longitudes
    for key in ("longitudes", "ecliptic_longitudes"):
        m = natal_chart.get(key)
        if isinstance(m, dict) and m:
            got = {_node_canon(k): float(v) for k, v in m.items() if k is not None}
            return {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in got.items()}
    # use natal_jd_tt if provided
    natal_jd_tt = None
    for k in ("natal_jd_tt","jd_tt","jd_utc"):
        if isinstance(natal_chart.get(k), (int, float)):
            natal_jd_tt = float(natal_chart[k]); break
    if isinstance(natal_jd_tt, float):
        rows = ep.ecliptic_longitudes(float(natal_jd_tt), _batch_map_nodes(tgts)).get("results", [])
        got = {_node_canon(str(r["name"])): float(r["longitude"]) for r in rows or []}
        return {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in got.items()}
    # fallback: empty; caller may compute at window start
    return {}

def find_gochar_in_range(
    *,
    natal_chart: dict,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    time_range: tuple | list | None = None,
    transiting_bodies: list[str] | None = None,
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
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "gochar": [], "meta": {}}

    # Allow function-style inputs OR route-body-style payload
    body_like: Dict[str, Any] = dict(natal_chart or {})
    if time_range and len(time_range) >= 2:
        body_like["time_range"] = [time_range[0], time_range[1]]
    if start_jd_tt is not None and end_jd_tt is not None:
        body_like["start_jd_tt"] = start_jd_tt
        body_like["end_jd_tt"] = end_jd_tt

    a, b, meta_ts = _resolve_window_or_error(body_like if body_like else {})
    if not (isinstance(a, float) and isinstance(b, float)):
        return {"ok": False, "error": meta_ts.get("error", "invalid_time_window"), "gochar": [], "meta": meta_ts}

    frame = frame or "ecliptic-of-date"
    zodiac_mode = (zodiac_mode or "sidereal").lower()
    ay = float(ayanamsa_deg or 0.0)
    use_nodes = bool(include_nodes or False)

    movers = list(transiting_bodies or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"])
    if use_nodes:
        if "Rahu" not in movers: movers.append("Rahu")
        if "Ketu" not in movers: movers.append("Ketu")
    tgts = list(natal_targets or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"])

    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        prebatch_refinement=bool(kwargs.get("prebatch_refinement", False)),
        treat_nodes_like_saturn=bool(treat_nodes_like_saturn or False),
    )
    eng.sidereal_mode = not zodiac_mode.startswith("tropical")
    eng.ayanamsa_deg = ay

    if isinstance(target_lon_map, dict) and target_lon_map:
        nat_map = {_node_canon(k): float(v) for k, v in target_lon_map.items()}
        targets = {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in nat_map.items()}
    else:
        targets = _pick_natal_targets_map(ep=ep, eng=eng, natal_chart=natal_chart or {}, tgts=tgts, ay=ay)
        if not targets:
            # last-ditch: compute at left edge
            rows = ep.ecliptic_longitudes(float(a), _batch_map_nodes(tgts)).get("results", [])
            nat_map = {_node_canon(str(r["name"])): float(r["longitude"]) for r in rows or []}
            targets = {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in nat_map.items()}

    try:
        evs = eng.scan_drishti(
            jd_start_tt=float(a), jd_end_tt=float(b),
            movers=movers, targets=targets, orb_deg=float(orb_deg or 12.0),
            orb_map=orb_map, step_minutes=step_minutes, include_nodes=use_nodes,
        )
    except Exception as e:
        return {"ok": False, "error": f"gochar_scan_failed:{e}", "gochar": [], "meta": {"window_jd_tt": [a, b]}}

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
        "meta": {"movers": movers, "natal_targets": list(targets.keys()),
                 "window_jd_tt": [float(a), float(b)],
                 "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": ay, **meta_ts},
    }

def _auto_step_minutes_for_signs(movers: List[str]) -> float:
    caps: List[int] = []
    for m in movers:
        n = (m or "").lower()
        if n == "moon": caps.append(10)
        elif n in ("mercury","venus","mars"): caps.append(30)
        elif n in ("sun","jupiter","saturn"): caps.append(90)
        else: caps.append(180)
    return float(max(5, min(caps) if caps else 60))

def _auto_step_minutes_for_nak(movers: List[str]) -> float:
    caps: List[int] = []
    for m in movers:
        n = (m or "").lower()
        if n == "moon": caps.append(5)
        elif n in ("mercury","venus","mars"): caps.append(20)
        else: caps.append(60)
    return float(max(2, min(caps) if caps else 20))

def _sign_index(lon: float) -> int:
    return int(math.floor(norm360(lon) / 30.0)) % 12

def find_rashi_ingresses_in_range(
    *,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "sidereal",
    ayanamsa_deg: float = 0.0,
    step_minutes: Union[str, float, int] = "auto",
    topocentric: bool = False,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float | None = None,
    **body_like,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "ingresses": [], "meta": {}}

    # Allow civil via body_like (date_from/date_to/time_range/tz)
    if start_jd_tt is None or end_jd_tt is None:
        a, b, meta_ts = _resolve_window_or_error(body_like)
        if not (isinstance(a, float) and isinstance(b, float)):
            return {"ok": False, "error": meta_ts.get("error","invalid_time_window"), "ingresses": [], "meta": meta_ts}
    else:
        a, b = float(start_jd_tt), float(end_jd_tt)
        meta_ts = {"from": "jd_tt"}

    if b <= a:
        return {"ok": True, "ingresses": [], "meta": {"window_jd_tt": [a,b], **meta_ts}}

    movers = list(movers or ["Sun","Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
        step_minutes = _auto_step_minutes_for_signs(movers)
    try:
        dt = float(step_minutes) / (60.0 * 24.0)
    except Exception:
        dt = 60.0 / (60.0 * 24.0)

    events: List[Dict[str, Any]] = []

    def lon_at(body: str, t: float) -> float:
        return eng._lon_cached(body, float(t))

    def sign_change(body: str, t0: float, t1: float) -> Optional[Tuple[float, int]]:
        l0 = lon_at(body, t0); l1 = lon_at(body, t1)
        s0 = _sign_index(l0); s1 = _sign_index(l1)
        if s0 == s1: return None
        def f(t: float) -> float:
            return wrap180((lon_at(body, t) % 30.0) - 0.0)
        v0 = wrap180((l0 % 30.0) - 0.0)
        v1 = wrap180((l1 % 30.0) - 0.0)
        t_exact = eng._refine_zero_brent(f, t0, t1, v0, v1, tol_days=1e-6)
        sign_to = _sign_index(lon_at(body, t_exact))
        return (t_exact, sign_to)

    t = float(a)
    while t < b - 1e-12:
        t2 = float(min(t + dt, b))
        for m in movers:
            sc = sign_change(m, t, t2)
            if sc:
                te, s_to = sc
                events.append({"body": _node_canon(m), "exact_jd_tt": float(te), "sign_to": int(s_to)})
        t = t2

    events.sort(key=lambda r: (r["exact_jd_tt"], r["body"]))
    return {"ok": True, "ingresses": events,
            "meta": {"movers": movers, "window_jd_tt": [float(a), float(b)],
                     "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg), **meta_ts}}

def find_nakshatra_ingresses_in_range(
    *m,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "sidereal",
    ayanamsa_deg: float = 0.0,
    step_minutes: Union[str, float, int] = "auto",
    topocentric: bool = False,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float | None = None,
    **body_like,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "ingresses": [], "meta": {}}

    if start_jd_tt is None or end_jd_tt is None:
        a, b, meta_ts = _resolve_window_or_error(body_like)
        if not (isinstance(a, float) and isinstance(b, float)):
            return {"ok": False, "error": meta_ts.get("error","invalid_time_window"), "ingresses": [], "meta": meta_ts}
    else:
        a, b = float(start_jd_tt), float(end_jd_tt)
        meta_ts = {"from": "jd_tt"}

    if b <= a:
        return {"ok": True, "ingresses": [], "meta": {"window_jd_tt": [a,b], **meta_ts}}

    movers = list(movers or ["Moon","Sun","Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    width = 360.0 / 27.0

    if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
        step_minutes = _auto_step_minutes_for_nak(movers)
    try:
        dt = float(step_minutes) / (60.0 * 24.0)
    except Exception:
        dt = 20.0 / (60.0 * 24.0)

    events: List[Dict[str, Any]] = []

    def lon_at(body: str, t: float) -> float:
        return eng._lon_cached(body, float(t))

    def nak_change(body: str, t0: float, t1: float) -> Optional[Tuple[float, int]]:
        l0 = lon_at(body, t0); l1 = lon_at(body, t1)
        # index already respects ayanamsa via eng.sidereal_mode adjustment in lon_at
        n0 = nakshatra_index(l0); n1 = nakshatra_index(l1)
        if n0 == n1: return None
        def f(t: float) -> float:
            return wrap180((lon_at(body, t) % width) - 0.0)
        v0 = wrap180((l0 % width) - 0.0)
        v1 = wrap180((l1 % width) - 0.0)
        t_exact = eng._refine_zero_brent(f, t0, t1, v0, v1, tol_days=5e-7)
        idx = nakshatra_index(lon_at(body, t_exact))
        return (t_exact, idx)

    t = float(a)
    while t < b - 1e-12:
        t2 = float(min(t + dt, b))
        for m in movers:
            sc = nak_change(m, t, t2)
            if sc:
                te, idx = sc
                events.append({
                    "body": _node_canon(m),
                    "exact_jd_tt": float(te),
                    "nakshatra_index": int(idx),
                    "nakshatra_name": NAKSHATRAS_27[(idx-1)%27],
                })
        t = t2

    events.sort(key=lambda r: (r["exact_jd_tt"], r["body"]))
    return {"ok": True, "ingresses": events,
            "meta": {"movers": movers, "window_jd_tt": [float(a), float(b)],
                     "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg), **meta_ts}}

def find_stations_in_range(
    *,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "sidereal",
    ayanamsa_deg: float = 0.0,
    step_minutes: Union[str, float, int] = "auto",
    topocentric: bool = False,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float | None = None,
    **body_like,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "stations": [], "meta": {}}

    if start_jd_tt is None or end_jd_tt is None:
        a, b, meta_ts = _resolve_window_or_error(body_like)
        if not (isinstance(a, float) and isinstance(b, float)):
            return {"ok": False, "error": meta_ts.get("error","invalid_time_window"), "stations": [], "meta": meta_ts}
    else:
        a, b = float(start_jd_tt), float(end_jd_tt)
        meta_ts = {"from": "jd_tt"}

    if b <= a:
        return {"ok": True, "stations": [], "meta": {"window_jd_tt": [a,b], **meta_ts}}

    movers = list(movers or ["Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
        caps: List[int] = []
        for m in movers:
            n = (m or "").lower()
            if n in ("mercury","venus"): caps.append(60)
            elif n in ("mars",):         caps.append(120)
            else:                        caps.append(180)
        step_minutes = max(15, min(caps) if caps else 120)
    try:
        dt = float(step_minutes) / (60.0 * 24.0)
    except Exception:
        dt = 120.0 / (60.0 * 24.0)

    events: List[Dict[str, Any]] = []

    def lon_at(body: str, t: float) -> float:
        return eng._lon_cached(body, float(t))

    def vel(body: str, t: float, h_days: float) -> float:
        l1 = lon_at(body, t - h_days)
        l2 = lon_at(body, t + h_days)
        d = wrap180(l2 - l1)
        return d / (2.0 * h_days)

    def zero_cross(body: str, t0: float, t1: float) -> Optional[Tuple[float, str]]:
        h = max(1.0 / (24.0 * 24.0), dt * 0.5)  # ≥ 1 hour or half-step
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

    t = float(a)
    while t < b - 1e-12:
        t2 = float(min(t + dt, b))
        for m in movers:
            zc = zero_cross(m, t, t2)
            if zc:
                te, kind = zc
                events.append({"body": _node_canon(m), "exact_jd_tt": float(te), "kind": kind})
        t = t2

    events.sort(key=lambda r: (r["exact_jd_tt"], r["body"]))
    return {"ok": True, "stations": events,
            "meta": {"movers": movers, "window_jd_tt": [float(a), float(b)],
                     "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg), **meta_ts}}


# ──────────────────────────────────────────────────────────────────────
# Feature builder
# ──────────────────────────────────────────────────────────────────────
def feature_drishti_proximity(*, hits: List[Dict[str, Any]], cap_deg: float = 12.0) -> List[float]:
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


# ──────────────────────────────────────────────────────────────────────
# Thin wrappers named exactly like what routes import
# ──────────────────────────────────────────────────────────────────────
def _extract_observer(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize geocentric/topocentric args from routes."""
    obs_mode = (kwargs.get("observer") or "").strip().lower()
    topo = bool(kwargs.get("topocentric", False) or obs_mode == "topocentric")
    return dict(
        topocentric=topo,
        latitude=kwargs.get("latitude"),
        longitude=kwargs.get("longitude"),
        elevation_m=kwargs.get("elevation_m"),
    )

def gochar_drishti(
    *,
    natal_chart: Dict[str, Any],
    date_from: str | None = None,
    date_to: str | None = None,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    jd_tt_window: List[float] | Tuple[float, float] | None = None,
    transiting_bodies: List[str] | None = None,
    natal_targets: List[str] | None = None,
    zodiac_mode: str = "sidereal",
    ayanamsa: float | str | None = None,
    frame: str = "ecliptic-of-date",
    include_nodes: bool = False,
    treat_nodes_like_saturn: bool = False,
    orb_deg: float = 12.0,
    orb_map: Dict[str, float] | None = None,
    step_minutes: Union[str, float, int] = "auto",
    prebatch_refinement: bool = False,
    tz_name: str | None = None,
    **kwargs,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable"}

    body_like: Dict[str, Any] = {}
    if isinstance(jd_tt_window, (list, tuple)) and len(jd_tt_window) >= 2:
        body_like["jd_tt_window"] = [jd_tt_window[0], jd_tt_window[1]]
    if start_jd_tt is not None and end_jd_tt is not None:
        body_like["start_jd_tt"] = start_jd_tt
        body_like["end_jd_tt"] = end_jd_tt
    if date_from and date_to:
        body_like.update({"date_from": date_from, "date_to": date_to, "tz": tz_name or kwargs.get("tz") or kwargs.get("place_tz")})

    ay_deg = float(ayanamsa) if isinstance(ayanamsa, (int, float, str)) and str(ayanamsa).replace(".","",1).isdigit() else 0.0
    obs = _extract_observer(kwargs)

    res = find_gochar_in_range(
        natal_chart=natal_chart,
        transiting_bodies=transiting_bodies,
        natal_targets=natal_targets,
        frame=frame,
        zodiac_mode=zodiac_mode,
        ayanamsa_deg=ay_deg,
        orb_deg=orb_deg,
        orb_map=orb_map,
        include_nodes=include_nodes,
        treat_nodes_like_saturn=treat_nodes_like_saturn,
        step_minutes=step_minutes,
        time_range=[date_from, date_to] if (date_from and date_to) else None,
        start_jd_tt=body_like.get("start_jd_tt"),
        end_jd_tt=body_like.get("end_jd_tt"),
        **obs,
        **body_like,
        prebatch_refinement=prebatch_refinement,
    )
    return res

def ingresses_rashi(
    *,
    movers: List[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    jd_tt_window: List[float] | Tuple[float, float] | None = None,
    zodiac_mode: str = "sidereal",
    ayanamsa: float | str | None = None,
    frame: str = "ecliptic-of-date",
    step_minutes: Union[str, float, int] = "auto",
    tz_name: str | None = None,
    **kwargs,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable"}

    body_like: Dict[str, Any] = {}
    if isinstance(jd_tt_window, (list, tuple)) and len(jd_tt_window) >= 2:
        body_like["jd_tt_window"] = [jd_tt_window[0], jd_tt_window[1]]
    if start_jd_tt is not None and end_jd_tt is not None:
        body_like["start_jd_tt"] = start_jd_tt
        body_like["end_jd_tt"] = end_jd_tt
    if date_from and date_to:
        body_like.update({"date_from": date_from, "date_to": date_to, "tz": tz_name or kwargs.get("tz") or kwargs.get("place_tz")})

    ay_deg = float(ayanamsa) if isinstance(ayanamsa, (int, float, str)) and str(ayanamsa).replace(".","",1).isdigit() else 0.0
    obs = _extract_observer(kwargs)

    return find_rashi_ingresses_in_range(
        movers=movers,
        frame=frame,
        zodiac_mode=zodiac_mode,
        ayanamsa_deg=ay_deg,
        step_minutes=step_minutes,
        start_jd_tt=body_like.get("start_jd_tt"),
        end_jd_tt=body_like.get("end_jd_tt"),
        **obs,
        **body_like,
    )

def ingresses_nakshatra(
    *,
    movers: List[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    jd_tt_window: List[float] | Tuple[float, float] | None = None,
    zodiac_mode: str = "sidereal",
    ayanamsa: float | str | None = None,
    frame: str = "ecliptic-of-date",
    step_minutes: Union[str, float, int] = "auto",
    tz_name: str | None = None,
    **kwargs,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable"}

    body_like: Dict[str, Any] = {}
    if isinstance(jd_tt_window, (list, tuple)) and len(jd_tt_window) >= 2:
        body_like["jd_tt_window"] = [jd_tt_window[0], jd_tt_window[1]]
    if start_jd_tt is not None and end_jd_tt is not None:
        body_like["start_jd_tt"] = start_jd_tt
        body_like["end_jd_tt"] = end_jd_tt
    if date_from and date_to:
        body_like.update({"date_from": date_from, "date_to": date_to, "tz": tz_name or kwargs.get("tz") or kwargs.get("place_tz")})

    ay_deg = float(ayanamsa) if isinstance(ayanamsa, (int, float, str)) and str(ayanamsa).replace(".","",1).isdigit() else 0.0
    obs = _extract_observer(kwargs)

    return find_nakshatra_ingresses_in_range(
        movers=movers,
        frame=frame,
        zodiac_mode=zodiac_mode,
        ayanamsa_deg=ay_deg,
        step_minutes=step_minutes,
        start_jd_tt=body_like.get("start_jd_tt"),
        end_jd_tt=body_like.get("end_jd_tt"),
        **obs,
        **body_like,
    )

def stations_retro_direct(
    *,
    movers: List[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    jd_tt_window: List[float] | Tuple[float, float] | None = None,
    zodiac_mode: str = "sidereal",
    ayanamsa: float | str | None = None,
    frame: str = "ecliptic-of-date",
    step_minutes: Union[str, float, int] = "auto",
    tz_name: str | None = None,
    **kwargs,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable"}

    body_like: Dict[str, Any] = {}
    if isinstance(jd_tt_window, (list, tuple)) and len(jd_tt_window) >= 2:
        body_like["jd_tt_window"] = [jd_tt_window[0], jd_tt_window[1]]
    if start_jd_tt is not None and end_jd_tt is not None:
        body_like["start_jd_tt"] = start_jd_tt
        body_like["end_jd_tt"] = end_jd_tt
    if date_from and date_to:
        body_like.update({"date_from": date_from, "date_to": date_to, "tz": tz_name or kwargs.get("tz") or kwargs.get("place_tz")})

    ay_deg = float(ayanamsa) if isinstance(ayanamsa, (int, float, str)) and str(ayanamsa).replace(".","",1).isdigit() else 0.0
    obs = _extract_observer(kwargs)

    return find_stations_in_range(
        movers=movers,
        frame=frame,
        zodiac_mode=zodiac_mode,
        ayanamsa_deg=ay_deg,
        step_minutes=step_minutes,
        start_jd_tt=body_like.get("start_jd_tt"),
        end_jd_tt=body_like.get("end_jd_tt"),
        **obs,
        **body_like,
    )
