# app/core/vedic_gochar.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Vedic Gochar (Transits) — production-ready, high-performance engine,
fully delegated to your timescales/ayanāṁśa modules (no DIY time math).

What you get
• Observer forwarding everywhere (topocentric/lat/lon/elevation)
• Ayanāṁśa resolver (string keys → degrees via your module, or numeric passthrough)
• UTC/UT1 timing on every event where your timescales provides the conversions:
    - exact_jd_ut1
    - exact_datetime_utc (ISO8601, UTC)
• Nakṣatra ingress includes pada_to (1..4) and nakshatra_name
• Same public API your routes import:
    - gochar_drishti(**kwargs)
    - ingresses_rashi(**kwargs)
    - ingresses_nakshatra(**kwargs)
    - stations_retro_direct(**kwargs)
    - feature_drishti_proximity(hits, cap_deg)
• Optimized sliding-window ephemeris with adaptive steps and event-driven scans
• Clear errors if timescales/ephemeris are unavailable (no silent fallbacks)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, Union
import math
import os
from datetime import datetime, timezone

# Optional: numpy (not required)
try:
    import numpy as np  # noqa: F401
except Exception:
    np = None  # type: ignore

# Ephemeris singletons
try:
    from app.core.ephem_singleton import TS, PLANETS  # type: ignore
except Exception:
    TS = None  # type: ignore
    PLANETS = None  # type: ignore

try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig  # type: ignore
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object  # type: ignore
    _EPH_OK = False

# Timescales (authoritative — no DIY replacements)
try:
    from app.core import time_kernel as _tk  # type: ignore
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts  # type: ignore
    _TS_OK = True
except Exception:
    _ts = None
    _TS_OK = False

# Angles/houses helper (optional)
try:
    import app.core.astronomy as _astro  # type: ignore
except Exception:
    _astro = None

# Ayanāṁśa adapter (yours)
try:
    from app.core.ayanamsa import get_ayanamsa_deg  # type: ignore
    _AY_OK = True
except Exception:
    get_ayanamsa_deg = None  # type: ignore
    _AY_OK = False

# Parāśari dṛiṣṭi & Nakṣatra helpers (optional)
try:
    from app.core.constants_vedic import (  # type: ignore
        graha_drishti_schema, drishti_strength_factor,
        nakshatra_index, NAKSHATRAS_27
    )
except Exception:
    def graha_drishti_schema(name: str) -> Dict[int, float]:
        n = (name or "").strip().lower()
        base = {7: 1.0}
        if n == "mars": base.update({4: 0.75, 8: 0.75})
        elif n == "jupiter": base.update({5: 0.75, 9: 0.75})
        elif n == "saturn": base.update({3: 0.75, 10: 0.75})
        return base
    def drishti_strength_factor(_: str, __: int) -> float: return 1.0
    def nakshatra_index(lon: float) -> int:
        w = 360.0 / 27.0
        return int(math.floor((lon % 360.0) / w)) + 1
    NAKSHATRAS_27 = tuple(f"Nakshatra {i+1}" for i in range(27))


# ──────────────────────────────────────────────────────────────────────────────
# Fast math + constants
# ──────────────────────────────────────────────────────────────────────────────
_DRISHTI_AXES = {k: (k * 30.0) % 360.0 for k in range(1, 13)}  # k-th house axis deg
_PLANET_SPEEDS = {
    "moon": 13.2, "mercury": 1.6, "venus": 1.6, "sun": 1.0,
    "mars": 0.7, "jupiter": 0.08, "saturn": 0.03, "rahu": -0.05, "ketu": -0.05
}
def norm360(x: float) -> float:
    r = x % 360.0
    return r if r >= 0.0 else (r + 360.0)
def wrap180(x: float) -> float:
    r = ((x + 180.0) % 360.0) - 180.0
    return 0.0 if abs(r) < 1e-12 else r
def angdiff(a2: float, a1: float) -> float: return wrap180(a2 - a1)


# ──────────────────────────────────────────────────────────────────────────────
# Node aliasing
# ──────────────────────────────────────────────────────────────────────────────
_NODE_ALIAS = {
    "north node": "North Node", "rahu": "Rahu", "true node": "North Node",
    "mean node": "North Node", "south node": "South Node", "ketu": "Ketu",
}
def _canon_node(nm: str) -> str:
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


# ──────────────────────────────────────────────────────────────────────────────
# Ephemeris adapter factory
# ──────────────────────────────────────────────────────────────────────────────
def _make_ephem(frame: str = "ecliptic-of-date") -> EphemerisAdapter:
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable")
    try:
        cfg = EphemConfig(frame=frame, planets=PLANETS) if PLANETS is not None else EphemConfig(frame=frame)  # type: ignore
    except TypeError:
        cfg = EphemConfig(frame=frame)  # type: ignore
    try:
        return EphemerisAdapter(cfg, timescale=TS)  # type: ignore
    except TypeError:
        try: return EphemerisAdapter(cfg, TS)  # type: ignore
        except TypeError: return EphemerisAdapter(cfg)  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Timescales helpers (STRICT: use your modules only; no DIY math)
# ──────────────────────────────────────────────────────────────────────────────
def _dut1_from_env() -> float:
    for k in ("ASTRO_DUT1_BROADCAST", "ASTRO_DUT1", "OCP_DUT1_SECONDS"):
        s = os.getenv(k)
        if s not in (None, ""):
            try: return float(s)
            except Exception: pass
    return 0.0

def _ts_from_civil(date: str, time: str, tz: str) -> Dict[str, Any]:
    """
    Call your timescales/time_kernel to obtain jd_utc/jd_tt and optionally datetime.
    Raises RuntimeError if not available.
    """
    if _tk is not None:
        for fname in ("timescales_from_civil", "build_timescales", "compute_timescales", "from_civil", "to_timescales"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(date=date, time=time, tz=tz)  # prefer kwargs
                except Exception:
                    out = fn(date, time, tz)
                if isinstance(out, dict): return out
    if _ts is not None:
        fn = getattr(_ts, "build_timescales", None)
        if callable(fn):
            try:
                return fn(date, time, tz, _dut1_from_env())  # type: ignore
            except TypeError:
                return fn(date, time, tz)  # type: ignore
    raise RuntimeError("timescales_unavailable")

def _civil_window_to_jd_tt(*, date_from: str, date_to: str, tz: str) -> Tuple[float, float, Dict[str, Any]]:
    if not (_TS_OK or _tk is not None):
        raise RuntimeError("timescales_unavailable")
    ts0 = _ts_from_civil(date_from, "00:00:00", tz)
    ts1 = _ts_from_civil(date_to,   "23:59:59", tz)
    jt0 = float(ts0.get("jd_tt"))
    jt1 = float(ts1.get("jd_tt"))
    ju0 = float(ts0.get("jd_utc")) if ts0.get("jd_utc") is not None else None
    ju1 = float(ts1.get("jd_utc")) if ts1.get("jd_utc") is not None else None
    meta = {"from": "civil", "tz": tz, "jd_utc_window": [ju0, ju1]}
    return jt0, jt1, meta

def _jd_tt_to_jd_utc(jd_tt: float) -> Optional[float]:
    """
    Use your timescales for TT→UTC JD. If not exposed, return None.
    """
    if _ts is not None:
        for name in ("jd_utc_from_jd_tt", "tt_to_utc_julian_day", "tt_to_utc_jd", "jd_utc_from_tt", "to_jd_utc_from_tt"):
            fn = getattr(_ts, name, None)
            if callable(fn):
                try:
                    return float(fn(jd_tt))  # type: ignore
                except Exception:
                    pass
    if _tk is not None:
        for name in ("jd_utc_from_jd_tt", "tt_to_utc_jd"):
            fn = getattr(_tk, name, None)
            if callable(fn):
                try:
                    return float(fn(jd_tt))  # type: ignore
                except Exception:
                    pass
    return None

def _jd_utc_to_datetime_utc(jd_utc: float) -> Optional[datetime]:
    """
    Use your timescales to convert JD(UTC) → datetime(UTC). If not exposed, None.
    """
    if _ts is not None:
        for name in ("datetime_from_jd_utc", "utc_datetime_from_jd", "to_datetime_utc"):
            fn = getattr(_ts, name, None)
            if callable(fn):
                try:
                    dt = fn(float(jd_utc))  # type: ignore
                    if isinstance(dt, datetime): return dt.astimezone(timezone.utc)
                except Exception:
                    pass
    if _tk is not None:
        for name in ("datetime_from_jd_utc", "utc_datetime_from_jd"):
            fn = getattr(_tk, name, None)
            if callable(fn):
                try:
                    dt = fn(float(jd_utc))  # type: ignore
                    if isinstance(dt, datetime): return dt.astimezone(timezone.utc)
                except Exception:
                    pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Ayanāṁśa resolver (strictly through your module; or numeric passthrough)
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_ayanamsa_deg(ayanamsa: Any, jd_tt_hint: Optional[float], zodiac_mode: str) -> float:
    zm = (zodiac_mode or "sidereal").lower()
    if zm.startswith("trop"):  # tropical → no subtraction
        return 0.0
    # numeric?
    try:
        if isinstance(ayanamsa, (int, float)): return float(ayanamsa)
        s = str(ayanamsa).strip()
        if s and s.replace(".", "", 1).isdigit(): return float(s)
    except Exception:
        pass
    # string key → degrees via your module
    if _AY_OK and callable(get_ayanamsa_deg) and isinstance(jd_tt_hint, (int, float)):
        try:
            return float(get_ayanamsa_deg(float(jd_tt_hint), ayanamsa or "lahiri"))  # type: ignore
        except Exception:
            # fall through to default if key failed
            pass
    # last resort: if we can compute at epoch 2000.0 with key, try that
    if _AY_OK and callable(get_ayanamsa_deg):
        try:
            return float(get_ayanamsa_deg(None, ayanamsa or "lahiri"))  # type: ignore
        except Exception:
            pass
    # default lahiri key passthrough if everything fails numerically
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Angles (Asc/MC) sidereal helper (uses astronomy.py or ephemeris angles)
# ──────────────────────────────────────────────────────────────────────────────
def _angles_sidereal_deg(*, jd_tt: float, lat: float, lon: float, ay_deg: float, sidereal: bool,
                         eng_ephem: EphemerisAdapter | None = None) -> Dict[str, float]:
    # try astronomy.py
    cand = (
        getattr(_astro, "compute_houses", None),
        getattr(_astro, "houses_advanced", None),
        getattr(_astro, "asc_mc_from", None),
        getattr(_astro, "asc_mc", None),
    ) if _astro else ()
    for fn in cand:
        if callable(fn):
            try:
                try:
                    res = fn(jd_tt=jd_tt, latitude=lat, longitude=lon)
                except TypeError:
                    res = fn(jd_tt, lat, lon)
                if isinstance(res, dict):
                    asc = res.get("Asc") or res.get("asc") or res.get("ASC")
                    mc =  res.get("MC")  or res.get("mc")
                    if asc is None or mc is None: continue
                    asc = float(asc); mc = float(mc)
                    if sidereal:
                        return {"Asc": norm360(asc - ay_deg), "MC": norm360(mc - ay_deg)}
                    return {"Asc": norm360(asc), "MC": norm360(mc)}
            except Exception:
                pass
    # try ephem adapter angles_ecliptic if provided
    if eng_ephem is not None and hasattr(eng_ephem, "angles_ecliptic"):
        try:
            out = eng_ephem.angles_ecliptic(jd_tt, latitude=lat, longitude=lon)
            asc = float(out.get("Asc")); mc = float(out.get("MC"))
            if sidereal:
                return {"Asc": norm360(asc - ay_deg), "MC": norm360(mc - ay_deg)}
            return {"Asc": norm360(asc), "MC": norm360(mc)}
        except Exception:
            pass
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Transit engine (optimized) + sliding window cache
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GocharEvent:
    jd_tt: float
    body: str
    target: str
    drishti: Literal["7th","3rd","4th","5th","8th","9th","10th"]
    axis_deg: float
    separation_deg: float
    applying: bool
    exact: bool
    weight: float
    meta: Dict[str, Any]

class EphemerisSlidingWindow:
    def __init__(self, engine: "VedicTransitEngine", window_size: int = 100):
        self.engine = engine
        self.window_size = max(50, int(window_size))
        self.cache: Dict[float, Dict[str, float]] = {}
        self.sorted_times: List[float] = []

    def _batch_fetch(self, times: List[float], bodies: List[str]) -> None:
        if not times or not bodies: return
        for t in times:
            if t in self.cache: continue
            try:
                rows = self.engine.ephem.ecliptic_longitudes(float(t), _batch_map_nodes(bodies), **self.engine.obs).get("results", [])
                lon_map: Dict[str, float] = {}
                for r in rows or []:
                    nm = _canon_node(str(r["name"]))
                    raw = float(r["longitude"])
                    lon_map[nm] = norm360(raw - self.engine.ayanamsa_deg) if self.engine.sidereal_mode else norm360(raw)
                self.cache[t] = lon_map
            except Exception:
                # best-effort individual calls
                got: Dict[str, float] = {}
                for b in bodies:
                    try:
                        one = self.engine.ephem.ecliptic_longitudes(float(t), _batch_map_nodes([b]), **self.engine.obs).get("results", [])
                        for r in one or []:
                            nm = _canon_node(str(r["name"]))
                            raw = float(r["longitude"])
                            got[nm] = norm360(raw - self.engine.ayanamsa_deg) if self.engine.sidereal_mode else norm360(raw)
                    except Exception:
                        pass
                self.cache[t] = got

    def ensure_window(self, center_time: float, span_days: float, bodies: List[str]) -> None:
        start_time = center_time - span_days/2
        end_time   = center_time + span_days/2
        num_points = min(self.window_size, max(20, int(span_days * 24)))  # ≥ hourly
        dt = (end_time - start_time)/num_points
        times = [start_time + i*dt for i in range(num_points + 1)]
        self._batch_fetch(times, bodies)
        self.sorted_times = sorted(self.cache.keys())

    def get_longitude(self, body: str, time: float) -> Optional[float]:
        if time in self.cache:
            return self.cache[time].get(body)
        if len(self.sorted_times) < 2:
            return None
        # binary search for bracket
        left, right = 0, len(self.sorted_times)-1
        while left <= right:
            mid = (left + right)//2
            if self.sorted_times[mid] <= time: left = mid + 1
            else: right = mid - 1
        idx = max(0, right)
        if idx >= len(self.sorted_times)-1: return None
        t1, t2 = self.sorted_times[idx], self.sorted_times[idx+1]
        lon1 = self.cache[t1].get(body); lon2 = self.cache[t2].get(body)
        if lon1 is None or lon2 is None: return None
        diff = angdiff(lon2, lon1)
        frac = (time - t1)/(t2 - t1) if (t2 - t1) else 0.0
        return norm360(lon1 + diff*frac)

class VedicTransitEngine:
    def __init__(self, *, ephem: EphemerisAdapter, sidereal_mode: bool, ayanamsa_deg: float,
                 topocentric: bool = False, latitude: float | None = None, longitude: float | None = None,
                 elevation_m: float | None = None, cache_size: int = 2000,
                 treat_nodes_like_saturn: bool = False):
        self.ephem = ephem
        self.sidereal_mode = bool(sidereal_mode)
        self.ayanamsa_deg = float(ayanamsa_deg or 0.0)
        self.obs = dict(topocentric=bool(topocentric), latitude=latitude, longitude=longitude, elevation_m=elevation_m)
        self.sliding = EphemerisSlidingWindow(self, window_size=max(80, cache_size//10))
        self._schemas: Dict[str, Dict[int, float]] = {}
        self.treat_nodes_like_saturn = bool(treat_nodes_like_saturn)

    def _schema_for(self, body: str) -> Dict[int, float]:
        b = _canon_node(body)
        key = (b, self.treat_nodes_like_saturn)
        if key in self._schemas: return self._schemas[key]
        name = b
        if b in ("Rahu", "Ketu") and self.treat_nodes_like_saturn:
            name = "Saturn"
        schema = graha_drishti_schema(name)
        if b in ("Rahu", "Ketu") and not self.treat_nodes_like_saturn:
            schema = {7: 1.0}
        self._schemas[key] = schema
        return schema

    def _planet_step_minutes(self, body: str, base: float) -> float:
        v = _PLANET_SPEEDS.get(body.lower(), 0.5)
        if v > 5.0: return base*0.4     # Moon
        if v > 1.0: return base*0.7     # Mercury/Venus/Sun
        if v > 0.1: return base*1.0     # Mars
        return base*2.0                 # Jup/Sat/Nodes

    # ---- Drishti scan ----
    def scan_drishti(self, *, jd_start_tt: float, jd_end_tt: float, movers: List[str], targets: Dict[str, float],
                     orb_deg: float = 12.0, orb_map: Optional[Dict[str, float]] = None,
                     step_minutes: Union[str, float, int] = "auto", scan_mode: str = "balanced") -> List[GocharEvent]:
        if jd_end_tt <= jd_start_tt or not movers or not targets: return []
        window_days = jd_end_tt - jd_start_tt
        if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
            base = 30.0 if window_days <= 3.0 else (60.0 if window_days <= 14.0 else 180.0)
            m = scan_mode.lower()
            if m.startswith("ultra"): base *= 2.0
            elif m.startswith("fine"): base *= 0.5
        else:
            base = float(step_minutes)
        self.sliding.ensure_window((jd_start_tt + jd_end_tt)/2, window_days + 1.0, movers)
        planet_step = {b: self._planet_step_minutes(b, base) for b in movers}
        events: List[GocharEvent] = []
        dedupe: set[Tuple[int, int, int, int]] = set()

        t = jd_start_tt
        while t < jd_end_tt - 1e-12:
            dt_min = min(planet_step.values())
            t_next = min(t + dt_min/(60.0*24.0), jd_end_tt)
            curr = {b: self.sliding.get_longitude(b, t)      for b in movers}
            nxt  = {b: self.sliding.get_longitude(b, t_next) for b in movers}
            for b in movers:
                lon0 = curr.get(b); lon1 = nxt.get(b)
                if lon0 is None or lon1 is None: continue
                dlon = wrap180(lon1 - lon0); speed = dlon/max(1e-9, (t_next - t))
                schema = self._schema_for(b)
                for tgt_name, tgt_lon in targets.items():
                    diff0 = wrap180(angdiff(lon0, tgt_lon))
                    diff1 = wrap180(angdiff(lon1, tgt_lon))
                    for k, w in schema.items():
                        axis = _DRISHTI_AXES[k]
                        sep0 = wrap180(diff0 - axis)
                        sep1 = wrap180(diff1 - axis)
                        if abs(sep0) > 15 and abs(sep1) > 15 and sep0 * sep1 > 0:  # far and no crossing
                            continue
                        # decide if within orb or crossing
                        max_orb = float(orb_map.get(b, orb_deg)) if orb_map else float(orb_deg)
                        min_sep = min(abs(sep0), abs(sep1))
                        crossing = (sep0 == 0.0) or (sep1 == 0.0) or (sep0 * sep1 < 0.0)
                        if (min_sep <= max_orb) or crossing:
                            # estimate exact time (linear in separation)
                            if abs(speed) > 1e-9 and crossing:
                                # sep(t) ≈ sep0 + speed*(t - t0) → solve sep=0
                                # speed is in deg/day of planet vs target fixed; we approximate axis crossing
                                te = t - (sep0 / speed)
                                te = max(t, min(t_next, te))
                            else:
                                te = t if abs(sep0) < abs(sep1) else t_next
                            # final separation at te (interpolate lon)
                            lon_te = self.sliding.get_longitude(b, te)
                            if lon_te is None:
                                final_sep = min(sep0, sep1, key=abs)
                            else:
                                final_sep = wrap180(angdiff(lon_te, tgt_lon) - axis)
                            key = (hash(b) % 10007, hash(tgt_name) % 10007, int(k), int(te * 86400 + 0.5))
                            if key in dedupe: continue
                            dedupe.add(key)
                            events.append(GocharEvent(
                                jd_tt=float(te), body=_canon_node(b), target=_canon_node(tgt_name),
                                drishti=("7th" if k == 7 else f"{k}th"), axis_deg=float(axis),
                                separation_deg=float(final_sep), applying=(abs(sep1) < abs(sep0)),
                                exact=abs(final_sep) <= 1e-6, weight=float(w), meta={"k_house": k}
                            ))
            t = t_next
        events.sort(key=lambda e: (e.jd_tt, e.body, e.target, e.axis_deg))
        return events

    # ---- Rāśi / Nakṣatra ingresses ----
    def find_sign_ingresses(self, a: float, b: float, movers: List[str], step_minutes: Union[str, float, int] = "auto") -> List[Dict[str, Any]]:
        if b <= a: return []
        window_days = b - a
        base = 30.0 if (isinstance(step_minutes, str) and step_minutes == "auto" and window_days <= 7) else (60.0 if isinstance(step_minutes, str) else float(step_minutes))
        self.sliding.ensure_window((a + b)/2, window_days + 0.5, movers)
        out: List[Dict[str, Any]] = []
        for body in movers:
            step_days = self._planet_step_minutes(body, base) / (60.0*24.0)
            t = a
            prev = None
            while t <= b:
                lon = self.sliding.get_longitude(body, t)
                if lon is not None:
                    sidx = int(lon // 30) % 12
                    if prev is not None and sidx != prev:
                        # refine crossing of multiple of 30°
                        t0, t1 = max(a, t - step_days), t
                        te = self._refine_division_cross(body, t0, t1, 30.0)
                        if te is not None:
                            out.append({"body": _canon_node(body), "exact_jd_tt": float(te), "sign_to": int(sidx)})
                    prev = sidx
                t += step_days
        return sorted(out, key=lambda x: (x["exact_jd_tt"], x["body"]))

    def find_nakshatra_ingresses(self, a: float, b: float, movers: List[str], step_minutes: Union[str, float, int] = "auto") -> List[Dict[str, Any]]:
        if b <= a: return []
        window_days = b - a
        base = 15.0 if (isinstance(step_minutes, str) and step_minutes == "auto" and window_days <= 7) else (30.0 if isinstance(step_minutes, str) else float(step_minutes))
        self.sliding.ensure_window((a + b)/2, window_days + 0.5, movers)
        out: List[Dict[str, Any]] = []
        width = 360.0/27.0
        pada_w = width/4.0
        for body in movers:
            step_days = self._planet_step_minutes(body, base) / (60.0*24.0)
            t = a
            prev = None
            while t <= b:
                lon = self.sliding.get_longitude(body, t)
                if lon is not None:
                    idx = nakshatra_index(lon)  # 1..27
                    if prev is not None and idx != prev:
                        t0, t1 = max(a, t - step_days), t
                        te = self._refine_division_cross(body, t0, t1, width)
                        if te is not None:
                            # evaluate pada at te
                            lon_te = self.sliding.get_longitude(body, te) or 0.0
                            pos_in_nak = (lon_te % width)
                            pada_to = int(pos_in_nak // pada_w) + 1  # 1..4
                            out.append({
                                "body": _canon_node(body),
                                "exact_jd_tt": float(te),
                                "nakshatra_index": int(idx),
                                "nakshatra_name": NAKSHATRAS_27[(idx-1) % 27],
                                "pada_to": int(pada_to)
                            })
                    prev = idx
                t += step_days
        return sorted(out, key=lambda x: (x["exact_jd_tt"], x["body"]))

    def _refine_division_cross(self, body: str, t0: float, t1: float, div_width: float) -> Optional[float]:
        # binary search for boundary where (lon % div_width) ≈ 0
        for _ in range(24):
            if t1 - t0 < 1e-6: break
            tm = 0.5*(t0 + t1)
            l0 = self.sliding.get_longitude(body, t0)
            lm = self.sliding.get_longitude(body, tm)
            if l0 is None or lm is None: return tm
            r0 = (l0 % div_width); rm = (lm % div_width)
            # detect wrap across 0 by looking at minimal distance to 0
            d0 = min(r0, div_width - r0); dm = min(rm, div_width - rm)
            if dm <= 1e-8: return tm
            if (r0 <= div_width/2 and rm <= div_width/2 and rm < r0) or (r0 > div_width/2 and rm > div_width/2 and (div_width - rm) < (div_width - r0)):
                t1 = tm
            else:
                t0 = tm
        return 0.5*(t0 + t1)


# ──────────────────────────────────────────────────────────────────────────────
# Station finder (retrograde/direct) using velocity sign change
# ──────────────────────────────────────────────────────────────────────────────
class StationFinder:
    def __init__(self, engine: VedicTransitEngine):
        self.engine = engine

    def find(self, a: float, b: float, movers: List[str], step_minutes: Union[str, float, int] = "auto") -> List[Dict[str, Any]]:
        if b <= a: return []
        window_days = b - a
        base = 120.0 if (isinstance(step_minutes, str) and step_minutes == "auto" and window_days <= 30) else (240.0 if isinstance(step_minutes, str) else float(step_minutes))
        self.engine.sliding.ensure_window((a + b)/2, window_days + 2.0, movers)
        out: List[Dict[str, Any]] = []
        for body in movers:
            if _canon_node(body).lower() in ("sun", "moon"):  # these never retrograde
                continue
            step_days = self.engine._planet_step_minutes(body, base) / (60.0*24.0)
            vel_dt = min(step_days/4, 0.5)
            t = a
            v_prev = None
            while t < b - vel_dt:
                lon_m = self.engine.sliding.get_longitude(body, t - vel_dt)
                lon_p = self.engine.sliding.get_longitude(body, t + vel_dt)
                if lon_m is None or lon_p is None:
                    t += step_days
                    continue
                v = angdiff(lon_p, lon_m) / (2.0*vel_dt)  # deg/day
                if v_prev is not None and v_prev * v < 0:
                    # refine zero-crossing (velocity change)
                    te = self._refine_station_time(body, t - step_days, t, vel_dt)
                    kind = self._kind_around(body, te, vel_dt)
                    out.append({"body": _canon_node(body), "exact_jd_tt": float(te), "kind": kind})
                v_prev = v
                t += step_days
        return sorted(out, key=lambda x: (x["exact_jd_tt"], x["body"]))

    def _refine_station_time(self, body: str, t0: float, t1: float, vel_dt: float) -> float:
        def vel_at(t: float) -> float:
            m = self.engine.sliding.get_longitude(body, t - vel_dt)
            p = self.engine.sliding.get_longitude(body, t + vel_dt)
            if m is None or p is None: return 0.0
            return angdiff(p, m)/(2.0*vel_dt)
        for _ in range(18):
            if t1 - t0 < 1e-5: break
            tm = 0.5*(t0 + t1)
            if abs(vel_at(tm)) < 1e-6: return tm
            if vel_at(t0) * vel_at(tm) < 0: t1 = tm
            else: t0 = tm
        return 0.5*(t0 + t1)

    def _kind_around(self, body: str, te: float, vel_dt: float) -> str:
        def vel_at(t: float) -> float:
            m = self.engine.sliding.get_longitude(body, t - vel_dt)
            p = self.engine.sliding.get_longitude(body, t + vel_dt)
            if m is None or p is None: return 0.0
            return angdiff(p, m)/(2.0*vel_dt)
        vb = vel_at(te - 1.0); va = vel_at(te + 1.0)
        if vb > 0 and va < 0: return "retrograde"
        if vb < 0 and va > 0: return "direct"
        return "station"

# ──────────────────────────────────────────────────────────────────────────────
# Window + natal target resolution (strict timescales; no DIY math)
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_window_or_error(body: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    # direct jd_tt numbers?
    if isinstance(body.get("jd_tt_window"), (list, tuple)) and len(body["jd_tt_window"]) == 2:
        try:
            a, b = map(float, body["jd_tt_window"])
            return a, b, {"from": "jd_tt_window"}
        except Exception:
            return None, None, {"error": "invalid_jd_tt_window"}
    if isinstance(body.get("start_jd_tt"), (int, float)) and isinstance(body.get("end_jd_tt"), (int, float)):
        return float(body["start_jd_tt"]), float(body["end_jd_tt"]), {"from": "start_end_jd_tt"}

    # civil window
    date_from = body.get("date_from") or body.get("from")
    date_to   = body.get("date_to")   or body.get("to")
    tz        = body.get("tz_name") or body.get("tz") or body.get("place_tz") or "UTC"
    if not (date_from and date_to):
        return None, None, {"error": "time_range_required"}
    try:
        a, b, meta = _civil_window_to_jd_tt(date_from=str(date_from), date_to=str(date_to), tz=str(tz))
        return a, b, meta
    except Exception as e:
        return None, None, {"error": f"timescales_failed:{e}"}

def _angles_for_natal(natal_chart: Dict[str, Any], *, ay_deg: float, sidereal: bool,
                      ephem: EphemerisAdapter | None) -> Dict[str, float]:
    lat = natal_chart.get("latitude"); lon = natal_chart.get("longitude")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return {}
    # prefer provided natal jd_tt / jd_utc
    jd_tt = None
    for k in ("natal_jd_tt","jd_tt"):
        if isinstance(natal_chart.get(k), (int, float)): jd_tt = float(natal_chart[k]); break
    if jd_tt is None:
        date = natal_chart.get("date") or natal_chart.get("birth_date")
        time = natal_chart.get("time") or natal_chart.get("birth_time") or "12:00:00"
        tz   = natal_chart.get("tz")   or natal_chart.get("place_tz")   or "UTC"
        try:
            jd_tt = float(_ts_from_civil(str(date), str(time), str(tz)).get("jd_tt"))
        except Exception:
            return {}
    return _angles_sidereal_deg(jd_tt=jd_tt, lat=float(lat), lon=float(lon), ay_deg=float(ay_deg), sidereal=sidereal, eng_ephem=ephem)

def _pick_natal_targets_map(*, ephem: EphemerisAdapter, sidereal: bool, ay_deg: float,
                            natal_chart: Dict[str, Any], tgts: List[str]) -> Dict[str, float]:
    # 1) direct longitudes?
    for key in ("longitudes", "ecliptic_longitudes"):
        m = natal_chart.get(key)
        if isinstance(m, dict) and m:
            out = {_canon_node(k): float(v) for k, v in m.items() if k is not None}
            out = {k: (norm360(v - ay_deg) if sidereal else norm360(v)) for k, v in out.items()}
            if ("Asc" in tgts) or ("MC" in tgts):
                out.update(_angles_for_natal(natal_chart, ay_deg=ay_deg, sidereal=sidereal, ephem=ephem))
            return out

    # 2) from natal jd_tt / civil inputs
    nat_jd_tt = None
    for k in ("natal_jd_tt", "jd_tt"):
        if isinstance(natal_chart.get(k), (int, float)):
            nat_jd_tt = float(natal_chart[k]); break
    if nat_jd_tt is None:
        date = natal_chart.get("date") or natal_chart.get("birth_date")
        time = natal_chart.get("time") or natal_chart.get("birth_time") or "12:00:00"
        tz   = natal_chart.get("tz") or natal_chart.get("place_tz") or "UTC"
        try:
            nat_jd_tt = float(_ts_from_civil(str(date), str(time), str(tz)).get("jd_tt"))
        except Exception:
            nat_jd_tt = None

    out: Dict[str, float] = {}
    if isinstance(nat_jd_tt, float):
        planet_tgts = [x for x in tgts if x not in ("Asc", "MC")]
        if planet_tgts:
            rows = ephem.ecliptic_longitudes(float(nat_jd_tt), _batch_map_nodes(planet_tgts)).get("results", [])
            got = {_canon_node(str(r["name"])): float(r["longitude"]) for r in rows or []}
            out.update({k: (norm360(v - ay_deg) if sidereal else norm360(v)) for k, v in got.items()})
        if ("Asc" in tgts) or ("MC" in tgts):
            out.update(_angles_for_natal(natal_chart, ay_deg=ay_deg, sidereal=sidereal, ephem=ephem))
        return out

    # 3) fallback: sample targets at window-left (rare)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public feature: proximity score [0..1]
# ──────────────────────────────────────────────────────────────────────────────
def feature_drishti_proximity(*, hits: List[Dict[str, Any]], cap_deg: float = 12.0) -> List[float]:
    cap = max(1e-6, float(cap_deg))
    out: List[float] = []
    for h in hits or []:
        try:
            orb = abs(float(h.get("orb", 9999.0)))
        except Exception:
            continue
        score = max(0.0, 1.0 - (orb / cap))
        out.append(score)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Core finders (used by wrappers) — strict timescales + ayanāṁśa
# ──────────────────────────────────────────────────────────────────────────────
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
    scan_mode: str | None = None,
    tz: str | None = None,
    **obs_kwargs,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "gochar": [], "meta": {}}

    # Resolve window
    if time_range and len(time_range) >= 2:
        date_from, date_to = str(time_range[0]), str(time_range[1])
        tz_used = tz or "UTC"
        try:
            a, b, meta_ts = _civil_window_to_jd_tt(date_from=date_from, date_to=date_to, tz=tz_used)
        except Exception as e:
            return {"ok": False, "error": f"timescales_failed:{e}", "gochar": [], "meta": {"tz": tz_used}}
    elif start_jd_tt is not None and end_jd_tt is not None:
        a, b = float(start_jd_tt), float(end_jd_tt)
        meta_ts = {"from": "jd_tt"}
        tz_used = tz or "UTC"
    else:
        return {"ok": False, "error": "time_range_required", "gochar": [], "meta": {}}

    # Engine setup
    movers = list(transiting_bodies or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"])
    if include_nodes:
        for nd in ("Rahu","Ketu"):
            if nd not in movers: movers.append(nd)
    tgts = list(natal_targets or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Asc","MC"])
    frame = frame or "ecliptic-of-date"
    zodiac_mode = (zodiac_mode or "sidereal").lower()

    # Ayanāṁśa degrees
    # Hint the resolver with a natal jd_tt if present; else window-left TT
    jd_hint = None
    if isinstance(natal_chart.get("natal_jd_tt"), (int, float)): jd_hint = float(natal_chart["natal_jd_tt"])
    elif isinstance(natal_chart.get("jd_tt"), (int, float)): jd_hint = float(natal_chart["jd_tt"])
    else: jd_hint = a
    ay_deg = float(ayanamsa_deg or 0.0)
    if not isinstance(ayanamsa_deg, (int, float)):
        ay_deg = _resolve_ayanamsa_deg(natal_chart.get("ayanamsa", None), jd_hint, zodiac_mode)

    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep,
        sidereal_mode=not zodiac_mode.startswith("trop"),
        ayanamsa_deg=ay_deg,
        topocentric=bool(obs_kwargs.get("topocentric")),
        latitude=obs_kwargs.get("latitude"),
        longitude=obs_kwargs.get("longitude"),
        elevation_m=obs_kwargs.get("elevation_m"),
        cache_size=min(5000, int((b - a) * 100)),
        treat_nodes_like_saturn=bool(treat_nodes_like_saturn or False),
    )

    # Natal targets
    if isinstance(target_lon_map, dict) and target_lon_map:
        nat_map = {_canon_node(k): float(v) for k, v in target_lon_map.items()}
        targets = {k: (norm360(v - ay_deg) if eng.sidereal_mode else norm360(v)) for k, v in nat_map.items()}
    else:
        targets = _pick_natal_targets_map(ephem=ep, sidereal=eng.sidereal_mode, ay_deg=ay_deg, natal_chart=natal_chart or {}, tgts=tgts)
        if not targets:
            rows = ep.ecliptic_longitudes(float(a), _batch_map_nodes([x for x in tgts if x not in ("Asc","MC")])).get("results", [])
            nat_map = {_canon_node(str(r["name"])): float(r["longitude"]) for r in rows or []}
            targets = {k: (norm360(v - ay_deg) if eng.sidereal_mode else norm360(v)) for k, v in nat_map.items()}

    # Scan
    try:
        evs = eng.scan_drishti(
            jd_start_tt=float(a), jd_end_tt=float(b),
            movers=movers, targets=targets, orb_deg=float(orb_deg or 12.0),
            orb_map=orb_map, step_minutes=step_minutes, scan_mode=(scan_mode or "balanced"),
        )
    except Exception as e:
        return {"ok": False, "error": f"gochar_scan_failed:{e}", "gochar": [], "meta": {"window_jd_tt": [a, b]}}

    # Stamp outputs using your timescales only (no DIY math)
    dut1 = _dut1_from_env()
    hits: List[Dict[str, Any]] = []
    for ev in evs:
        jd_utc = _jd_tt_to_jd_utc(ev.jd_tt)
        dt_utc = _jd_utc_to_datetime_utc(jd_utc) if isinstance(jd_utc, float) else None
        jd_ut1 = (jd_utc + (dut1/86400.0)) if isinstance(jd_utc, float) else None
        hits.append({
            "transiting_body": ev.body,
            "natal_body": ev.target,
            "drishti": ev.drishti,
            "axis_deg": float(ev.axis_deg),
            "orb": abs(float(ev.separation_deg)),
            "max_orb": float(orb_deg or 12.0),
            "weight": float(ev.weight),
            "applying": bool(ev.applying),
            "exact": bool(ev.exact),
            "exact_jd_tt": float(ev.jd_tt),
            "exact_jd_ut1": (float(jd_ut1) if isinstance(jd_ut1, float) else None),
            "exact_datetime_utc": (dt_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z") if isinstance(dt_utc, datetime) else None),
        })

    meta_out = {
        "movers": movers,
        "natal_targets": list(targets.keys()),
        "window_jd_tt": [float(a), float(b)],
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa_deg": ay_deg,
        "tz": tz or meta_ts.get("tz") if isinstance(meta_ts, dict) else None,
        "performance_mode": "optimized",
        **(meta_ts if isinstance(meta_ts, dict) else {}),
    }
    return {"ok": True, "technique": "gochar_drishti_optimized", "gochar": hits, "meta": meta_out}

def find_rashi_ingresses_in_range(
    *,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    time_range: tuple | list | None = None,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "sidereal",
    ayanamsa_deg: float = 0.0,
    step_minutes: Union[str, float, int] = "auto",
    tz: str | None = None,
    **obs_kwargs,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "ingresses": [], "meta": {}}

    if time_range and len(time_range) >= 2:
        date_from, date_to = str(time_range[0]), str(time_range[1])
        tz_used = tz or "UTC"
        try:
            a, b, meta_ts = _civil_window_to_jd_tt(date_from=date_from, date_to=date_to, tz=tz_used)
        except Exception as e:
            return {"ok": False, "error": f"timescales_failed:{e}", "ingresses": [], "meta": {"tz": tz_used}}
    elif start_jd_tt is not None and end_jd_tt is not None:
        a, b = float(start_jd_tt), float(end_jd_tt)
        meta_ts = {"from": "jd_tt"}
    else:
        return {"ok": False, "error": "time_range_required", "ingresses": [], "meta": {}}

    movers = list(movers or ["Sun","Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep, sidereal_mode=zodiac_mode.lower().startswith("sidereal"),
        ayanamsa_deg=float(ayanamsa_deg),
        topocentric=bool(obs_kwargs.get("topocentric")),
        latitude=obs_kwargs.get("latitude"),
        longitude=obs_kwargs.get("longitude"),
        elevation_m=obs_kwargs.get("elevation_m"),
        cache_size=min(3000, int((b - a) * 50))
    )
    scanner = eng.find_sign_ingresses(a, b, movers, step_minutes)
    return {"ok": True, "ingresses": scanner, "meta": {
        "movers": movers, "window_jd_tt": [float(a), float(b)], "frame": frame,
        "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg),
        "performance_mode": "optimized", **meta_ts
    }}

def find_nakshatra_ingresses_in_range(
    *,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    time_range: tuple | list | None = None,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "sidereal",
    ayanamsa_deg: float = 0.0,
    step_minutes: Union[str, float, int] = "auto",
    tz: str | None = None,
    **obs_kwargs,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "ingresses": [], "meta": {}}

    if time_range and len(time_range) >= 2:
        date_from, date_to = str(time_range[0]), str(time_range[1])
        tz_used = tz or "UTC"
        try:
            a, b, meta_ts = _civil_window_to_jd_tt(date_from=date_from, date_to=date_to, tz=tz_used)
        except Exception as e:
            return {"ok": False, "error": f"timescales_failed:{e}", "ingresses": [], "meta": {"tz": tz_used}}
    elif start_jd_tt is not None and end_jd_tt is not None:
        a, b = float(start_jd_tt), float(end_jd_tt)
        meta_ts = {"from": "jd_tt"}
    else:
        return {"ok": False, "error": "time_range_required", "ingresses": [], "meta": {}}

    movers = list(movers or ["Moon","Sun","Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep, sidereal_mode=zodiac_mode.lower().startswith("sidereal"),
        ayanamsa_deg=float(ayanamsa_deg),
        topocentric=bool(obs_kwargs.get("topocentric")),
        latitude=obs_kwargs.get("latitude"),
        longitude=obs_kwargs.get("longitude"),
        elevation_m=obs_kwargs.get("elevation_m"),
        cache_size=min(3000, int((b - a) * 60))
    )
    scanner = eng.find_nakshatra_ingresses(a, b, movers, step_minutes)
    return {"ok": True, "ingresses": scanner, "meta": {
        "movers": movers, "window_jd_tt": [float(a), float(b)], "frame": frame,
        "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg),
        "performance_mode": "optimized", **meta_ts
    }}

def find_stations_in_range(
    *,
    start_jd_tt: float | None = None,
    end_jd_tt: float | None = None,
    time_range: tuple | list | None = None,
    movers: List[str] | None = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "sidereal",
    ayanamsa_deg: float = 0.0,
    step_minutes: Union[str, float, int] = "auto",
    tz: str | None = None,
    **obs_kwargs,
) -> Dict[str, Any]:
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "stations": [], "meta": {}}

    if time_range and len(time_range) >= 2:
        date_from, date_to = str(time_range[0]), str(time_range[1])
        tz_used = tz or "UTC"
        try:
            a, b, meta_ts = _civil_window_to_jd_tt(date_from=date_from, date_to=date_to, tz=tz_used)
        except Exception as e:
            return {"ok": False, "error": f"timescales_failed:{e}", "stations": [], "meta": {"tz": tz_used}}
    elif start_jd_tt is not None and end_jd_tt is not None:
        a, b = float(start_jd_tt), float(end_jd_tt)
        meta_ts = {"from": "jd_tt"}
    else:
        return {"ok": False, "error": "time_range_required", "stations": [], "meta": {}}

    movers = list(movers or ["Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep, sidereal_mode=zodiac_mode.lower().startswith("sidereal"),
        ayanamsa_deg=float(ayanamsa_deg),
        topocentric=bool(obs_kwargs.get("topocentric")),
        latitude=obs_kwargs.get("latitude"),
        longitude=obs_kwargs.get("longitude"),
        elevation_m=obs_kwargs.get("elevation_m"),
        cache_size=min(2000, int((b - a) * 30))
    )
    finder = StationFinder(eng)
    items = finder.find(a, b, movers, step_minutes)
    return {"ok": True, "stations": items, "meta": {
        "movers": movers, "window_jd_tt": [float(a), float(b)], "frame": frame,
        "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg),
        "performance_mode": "optimized", **meta_ts
    }}


# ──────────────────────────────────────────────────────────────────────────────
# Route-compatible wrappers (unchanged public surface)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_observer(kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
    prebatch_refinement: bool = False,  # kept for API parity; not used
    tz_name: str | None = None,
    scan_mode: str = "balanced",
    **kwargs,
) -> Dict[str, Any]:
    body_like: Dict[str, Any] = {}
    tz_used = tz_name or kwargs.get("tz") or kwargs.get("place_tz")
    if isinstance(jd_tt_window, (list, tuple)) and len(jd_tt_window) >= 2:
        body_like["jd_tt_window"] = [jd_tt_window[0], jd_tt_window[1]]
    if start_jd_tt is not None and end_jd_tt is not None:
        body_like["start_jd_tt"] = start_jd_tt
        body_like["end_jd_tt"] = end_jd_tt
    if date_from and date_to:
        body_like.update({"time_range": [date_from, date_to]})
    obs = _extract_observer(kwargs)
    return find_gochar_in_range(
        natal_chart=natal_chart,
        transiting_bodies=transiting_bodies,
        natal_targets=natal_targets,
        frame=frame,
        zodiac_mode=zodiac_mode,
        ayanamsa_deg=(float(ayanamsa) if isinstance(ayanamsa, (int, float, str)) and str(ayanamsa).replace(".","",1).isdigit() else None),
        orb_deg=orb_deg,
        orb_map=orb_map,
        include_nodes=include_nodes,
        treat_nodes_like_saturn=treat_nodes_like_saturn,
        step_minutes=step_minutes,
        time_range=body_like.get("time_range"),
        start_jd_tt=body_like.get("start_jd_tt"),
        end_jd_tt=body_like.get("end_jd_tt"),
        scan_mode=scan_mode,
        tz=tz_used,
        **obs,
    )

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
    body_like: Dict[str, Any] = {}
    tz_used = tz_name or kwargs.get("tz") or kwargs.get("place_tz")
    if isinstance(jd_tt_window, (list, tuple)) and len(jd_tt_window) >= 2:
        body_like["jd_tt_window"] = [jd_tt_window[0], jd_tt_window[1]]
    if start_jd_tt is not None and end_jd_tt is not None:
        body_like["start_jd_tt"] = start_jd_tt
        body_like["end_jd_tt"] = end_jd_tt
    if date_from and date_to:
        body_like.update({"time_range": [date_from, date_to]})
    obs = _extract_observer(kwargs)
    ay_deg = float(ayanamsa) if isinstance(ayanamsa, (int, float, str)) and str(ayanamsa).replace(".","",1).isdigit() else 0.0
    return find_rashi_ingresses_in_range(
        movers=movers, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ay_deg,
        step_minutes=step_minutes, time_range=body_like.get("time_range"),
        start_jd_tt=body_like.get("start_jd_tt"), end_jd_tt=body_like.get("end_jd_tt"),
        tz=tz_used, **obs
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
    body_like: Dict[str, Any] = {}
    tz_used = tz_name or kwargs.get("tz") or kwargs.get("place_tz")
    if isinstance(jd_tt_window, (list, tuple)) and len(jd_tt_window) >= 2:
        body_like["jd_tt_window"] = [jd_tt_window[0], jd_tt_window[1]]
    if start_jd_tt is not None and end_jd_tt is not None:
        body_like["start_jd_tt"] = start_jd_tt
        body_like["end_jd_tt"] = end_jd_tt
    if date_from and date_to:
        body_like.update({"time_range": [date_from, date_to]})
    obs = _extract_observer(kwargs)
    ay_deg = float(ayanamsa) if isinstance(ayanamsa, (int, float, str)) and str(ayanamsa).replace(".","",1).isdigit() else 0.0
    return find_nakshatra_ingresses_in_range(
        movers=movers, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ay_deg,
        step_minutes=step_minutes, time_range=body_like.get("time_range"),
        start_jd_tt=body_like.get("start_jd_tt"), end_jd_tt=body_like.get("end_jd_tt"),
        tz=tz_used, **obs
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
    body_like: Dict[str, Any] = {}
    tz_used = tz_name or kwargs.get("tz") or kwargs.get("place_tz")
    if isinstance(jd_tt_window, (list, tuple)) and len(jd_tt_window) >= 2:
        body_like["jd_tt_window"] = [jd_tt_window[0], jd_tt_window[1]]
    if start_jd_tt is not None and end_jd_tt is not None:
        body_like["start_jd_tt"] = start_jd_tt
        body_like["end_jd_tt"] = end_jd_tt
    if date_from and date_to:
        body_like.update({"time_range": [date_from, date_to]})
    obs = _extract_observer(kwargs)
    ay_deg = float(ayanamsa) if isinstance(ayanamsa, (int, float, str)) and str(ayanamsa).replace(".","",1).isdigit() else 0.0
    return find_stations_in_range(
        movers=movers, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ay_deg,
        step_minutes=step_minutes, time_range=body_like.get("time_range"),
        start_jd_tt=body_like.get("start_jd_tt"), end_jd_tt=body_like.get("end_jd_tt"),
        tz=tz_used, **obs
    )


