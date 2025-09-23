# app/core/vedic_gochar.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Vedic Gochar (Transits) — High-Performance astronomy.py–compatible time model.

PERFORMANCE OPTIMIZATIONS:
- Batch ephemeris fetching with sliding windows
- Adaptive time stepping based on planetary speeds
- Pre-computed drishti schemas and angle calculations
- Vectorized separation calculations
- Event-driven scanning for ingresses and stations
- Optimized deduplication with integer keys

Public surface (used by routes):
  • gochar_drishti(**kwargs)
  • ingresses_rashi(**kwargs)
  • ingresses_nakshatra(**kwargs)
  • stations_retro_direct(**kwargs)
  • feature_drishti_proximity(hits, cap_deg)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, Callable, Union
import os
import math
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np

# Ephemeris backbone: shared singletons
try:
    from app.core.ephem_singleton import TS, PLANETS
except Exception:
    TS = None
    PLANETS = None

try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    _EPH_OK = False
    EphemerisAdapter = object
    EphemConfig = object

# Optional helpers (drishti schema + nakshatras)
try:
    from app.core.constants_vedic import (
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
    def drishti_strength_factor(_: str, __: int) -> float:
        return 1.0
    def nakshatra_index(lon: float) -> int:
        w = 360.0 / 27.0
        return int(math.floor((lon % 360.0) / w)) + 1
    NAKSHATRAS_27 = tuple(f"Nakshatra {i+1}" for i in range(27))

# Civil→JD helpers (astronomy.py family)
try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

# Angles / houses via astronomy.py (preferred)
try:
    import app.core.astronomy as _astro
except Exception:
    _astro = None


# ──────────────────────────────────────────────────────────────────────
# PERFORMANCE OPTIMIZATIONS: Pre-computed constants and fast math
# ──────────────────────────────────────────────────────────────────────
_TWO_PI = 2.0 * math.pi
_PI = math.pi
_DEG_TO_RAD = math.pi / 180.0
_RAD_TO_DEG = 180.0 / math.pi

# Pre-computed drishti axes (avoid repeated modulo operations)
_DRISHTI_AXES = {k: (k * 30.0) % 360.0 for k in range(1, 13)}

# Planet speed estimates (degrees per day) for adaptive stepping
_PLANET_SPEEDS = {
    "moon": 13.2, "mercury": 1.6, "venus": 1.6, "sun": 1.0,
    "mars": 0.7, "jupiter": 0.08, "saturn": 0.03, "rahu": -0.05, "ketu": -0.05
}

def norm360_fast(x: float) -> float:
    """Optimized 360-degree normalization."""
    if 0.0 <= x < 360.0:
        return x
    r = x % 360.0
    return r if r >= 0.0 else r + 360.0

def wrap180_fast(x: float) -> float:
    """Optimized ±180 wrapping."""
    r = ((x + 180.0) % 360.0) - 180.0
    return 0.0 if abs(r) < 1e-12 else r

def angdiff_fast(a2: float, a1: float) -> float:
    """Fast angular difference."""
    return wrap180_fast(a2 - a1)


# ──────────────────────────────────────────────────────────────────────
# Adapter factory (backward-compatible)
# ──────────────────────────────────────────────────────────────────────
def _make_ephem(frame: str = "ecliptic-of-date") -> EphemerisAdapter:
    try:
        if PLANETS is not None:
            cfg = EphemConfig(frame=frame, planets=PLANETS)
        else:
            cfg = EphemConfig(frame=frame)
    except TypeError:
        cfg = EphemConfig(frame=frame)

    try:
        return EphemerisAdapter(cfg, timescale=TS)
    except TypeError:
        try:
            return EphemerisAdapter(cfg, TS)
        except TypeError:
            return EphemerisAdapter(cfg)


# ──────────────────────────────────────────────────────────────────────
# Node aliasing and utilities
# ──────────────────────────────────────────────────────────────────────
_NODE_ALIAS = {
    "north node": "North Node", "rahu": "Rahu", "true node": "North Node",
    "mean node": "North Node", "south node": "South Node", "ketu": "Ketu",
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


# ──────────────────────────────────────────────────────────────────────
# Timescale model (compatible with houses_advanced)
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
    return float(ju) + (69.0 / 86400.0)

def _jd_utc_via_stdlib(d: str, t: str, tz: str) -> float:
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
    if _tk:
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
            fn = getattr(_tk, fname, None)
            if not callable(fn): continue
            try:
                out = fn(date=date, time=time, tz=tz)
            except Exception:
                try: out = fn(date, time, tz)
                except Exception: out = None
            if isinstance(out, dict):
                ju = out.get("jd_utc", out.get("jd_ut"))
                if isinstance(ju, (int, float)): return float(ju)
            if isinstance(out, (list, tuple)) and out and isinstance(out[0], (int,float)):
                return float(out[0])
    if _ts and hasattr(_ts, "julian_day_utc"):
        try: return float(_ts.julian_day_utc(date, time, tz))
        except Exception: pass
    return _jd_utc_via_stdlib(date, time, tz)

def _first_of_day_jd_utc(date: str, tz: str) -> float:
    return _civil_to_jd_utc(date, "00:00:00", tz)

def _last_of_day_jd_utc(date: str, tz: str) -> float:
    return _civil_to_jd_utc(date, "23:59:59", tz)

def _parse_dates_from_body(body: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(body.get("time_range"), (list, tuple)) and len(body["time_range"]) >= 2:
        return str(body["time_range"][0]), str(body["time_range"][1])
    d0 = (body.get("date_from") or body.get("from") or None)
    d1 = (body.get("date_to")   or body.get("to")   or None)
    if not d0 or not d1:
        return None, None
    s0, s1 = str(d0), str(d1)
    def _just_date(s: str) -> str:
        if "T" in s:   return s.split("T", 1)[0]
        if " " in s:   return s.split(" ", 1)[0]
        return s
    return _just_date(s0), _just_date(s1)

def _tz_from_body(body: Dict[str, Any]) -> str:
    tz = body.get("tz_name") or body.get("tz") or body.get("place_tz") or "UTC"
    tzs = str(tz).strip()
    return tzs if tzs else "UTC"

def _window_from_body_to_jd_tt(body: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
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

    try: Y0, M0 = map(int, d0.split("-")[:2])
    except Exception: Y0, M0 = 2000, 1
    try: Y1, M1 = map(int, d1.split("-")[:2])
    except Exception: Y1, M1 = 2000, 1

    jt0 = _jd_tt_from_utc_jd(ju0, Y0, M0)
    jt1 = _jd_tt_from_utc_jd(ju1, Y1, M1)

    dut1 = body.get("dut1") if isinstance(body.get("dut1"), (int,float)) else body.get("dut1_seconds")
    if isinstance(dut1, (int, float, str)) and str(dut1).strip() != "":
        try: dut1_used = _clamp_dut1(float(dut1))
        except Exception: dut1_used = _clamp_dut1(_env_dut1_seconds())
    else:
        dut1_used = _clamp_dut1(_env_dut1_seconds())

    ju_ut1_0 = float(ju0 + dut1_used/86400.0)
    ju_ut1_1 = float(ju1 + dut1_used/86400.0)

    meta = {
        "from": "civil",
        "tz": tz,
        "jd_utc_window": [float(ju0), float(ju1)],
        "jd_ut1_window": [ju_ut1_0, ju_ut1_1],
        "dut1_seconds": float(dut1_used),
    }
    return float(jt0), float(jt1), meta


# ──────────────────────────────────────────────────────────────────────
# HIGH-PERFORMANCE EPHEMERIS CACHING: Sliding Window System
# ──────────────────────────────────────────────────────────────────────
class EphemerisSlidingWindow:
    """High-performance sliding window ephemeris cache with batch prefetching."""
    
    def __init__(self, engine: "VedicTransitEngine", window_size: int = 100):
        self.engine = engine
        self.window_size = max(50, window_size)
        self.cache: Dict[float, Dict[str, float]] = {}
        self.sorted_times: List[float] = []
        self.window_start = 0
        self.prefetch_threshold = 0.7  # Prefetch when 70% through window
        
    def _batch_fetch(self, times: List[float], bodies: List[str]) -> None:
        """Batch fetch ephemeris data for multiple times."""
        if not times or not bodies:
            return
            
        # Group times to minimize ephemeris calls
        for t in times:
            if t not in self.cache:
                try:
                    batch_result = self.engine.ephem.ecliptic_longitudes(
                        float(t), _batch_map_nodes(bodies), **self.engine.obs
                    ).get("results", [])
                    
                    lon_map = {}
                    for r in batch_result:
                        nm = _node_canon(str(r["name"]))
                        lon_raw = float(r["longitude"])
                        if self.engine.sidereal_mode:
                            lon_map[nm] = norm360_fast(lon_raw - self.engine.ayanamsa_deg)
                        else:
                            lon_map[nm] = norm360_fast(lon_raw)
                    
                    self.cache[t] = lon_map
                except Exception:
                    # Fallback to individual calls if batch fails
                    self.cache[t] = self.engine._lon_map_fallback(t, bodies)
    
    def ensure_window(self, center_time: float, span_days: float, bodies: List[str]) -> None:
        """Ensure ephemeris window covers the required time span."""
        start_time = center_time - span_days / 2
        end_time = center_time + span_days / 2
        
        # Generate time grid
        num_points = min(self.window_size, max(20, int(span_days * 24)))  # At least hourly
        dt = (end_time - start_time) / num_points
        times = [start_time + i * dt for i in range(num_points + 1)]
        
        # Batch prefetch
        self._batch_fetch(times, bodies)
        self.sorted_times = sorted(self.cache.keys())
    
    def get_longitude(self, body: str, time: float) -> Optional[float]:
        """Get longitude with interpolation if needed."""
        if time in self.cache:
            return self.cache[time].get(body)
        
        # Find bracketing times for interpolation
        if len(self.sorted_times) < 2:
            return None
            
        # Linear interpolation between nearest cached points
        idx = self._find_bracket_index(time)
        if idx < 0 or idx >= len(self.sorted_times) - 1:
            return None
            
        t1, t2 = self.sorted_times[idx], self.sorted_times[idx + 1]
        if t1 == t2:
            return self.cache[t1].get(body)
            
        lon1 = self.cache[t1].get(body)
        lon2 = self.cache[t2].get(body)
        
        if lon1 is None or lon2 is None:
            return None
            
        # Handle longitude wrapping for interpolation
        diff = angdiff_fast(lon2, lon1)
        fraction = (time - t1) / (t2 - t1)
        interpolated = lon1 + diff * fraction
        
        return norm360_fast(interpolated)
    
    def _find_bracket_index(self, time: float) -> int:
        """Binary search for bracketing index."""
        left, right = 0, len(self.sorted_times) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.sorted_times[mid] <= time:
                left = mid + 1
            else:
                right = mid - 1
        return max(0, right)


# ──────────────────────────────────────────────────────────────────────
# Angles helper (Asc/MC) — uses astronomy.py if available
# ──────────────────────────────────────────────────────────────────────
def _angles_sidereal_deg(*, jd_tt: float, lat: float, lon: float, eng: "VedicTransitEngine") -> Dict[str, float]:
    if _astro:
        cand = (
            getattr(_astro, "compute_houses", None),
            getattr(_astro, "houses_advanced", None),
            getattr(_astro, "asc_mc_from", None),
            getattr(_astro, "asc_mc", None),
        )
        for fn in cand:
            if callable(fn):
                try:
                    try:
                        res = fn(jd_tt=jd_tt, latitude=lat, longitude=lon)
                    except TypeError:
                        res = fn(jd_tt, lat, lon)
                    if isinstance(res, dict):
                        asc = res.get("Asc") or res.get("asc") or res.get("ASC") or res.get("ascendant")
                        mc  = res.get("MC")  or res.get("mc")  or res.get("midheaven") or res.get("Medium Coeli")
                        if asc is None or mc is None:
                            continue
                        asc = float(asc); mc = float(mc)
                        if eng.sidereal_mode:
                            ay = float(eng.ayanamsa_deg)
                            return {"Asc": norm360_fast(asc - ay), "MC": norm360_fast(mc - ay)}
                        return {"Asc": norm360_fast(asc), "MC": norm360_fast(mc)}
                except Exception:
                    pass
    try:
        if hasattr(eng.ephem, "angles_ecliptic"):
            res = eng.ephem.angles_ecliptic(jd_tt, latitude=lat, longitude=lon, **eng.obs)
            asc = float(res.get("Asc")); mc = float(res.get("MC"))
            if eng.sidereal_mode:
                ay = float(eng.ayanamsa_deg)
                return {"Asc": norm360_fast(asc - ay), "MC": norm360_fast(mc - ay)}
            return {"Asc": norm360_fast(asc), "MC": norm360_fast(mc)}
    except Exception:
        pass
    return {}


# ──────────────────────────────────────────────────────────────────────
# Event model + High-Performance Engine
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

class VedicTransitEngine:
    """High-performance Vedic transit calculation engine."""
    
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
        treat_nodes_like_saturn: bool = False,
        cache_size: int = 2000,
    ):
        if not _EPH_OK:
            raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")
        self.ephem = ephem or _make_ephem(frame)
        self.frame = frame
        self.obs = dict(topocentric=bool(topocentric), latitude=latitude, longitude=longitude, elevation_m=elevation_m)
        self.prebatch_refinement = bool(prebatch_refinement)
        self.treat_nodes_like_saturn = bool(treat_nodes_like_saturn)
        self.sidereal_mode: bool = True
        self.ayanamsa_deg: float = 0.0
        
        # High-performance caching
        self.sliding_window = EphemerisSlidingWindow(self, window_size=cache_size // 10)
        self._drishti_schemas: Dict[str, Dict[int, float]] = {}
        self._last_schema_config = None
    
    def _lon_map_fallback(self, jd_tt: float, names: List[str]) -> Dict[str, float]:
        """Fallback individual longitude fetching."""
        req = _batch_map_nodes(names)
        try:
            rows = self.ephem.ecliptic_longitudes(float(jd_tt), req, **self.obs).get("results", [])
            got: Dict[str, float] = {}
            for r in rows or []:
                nm = _node_canon(str(r["name"]))
                got[nm] = float(r["longitude"])
            if self.sidereal_mode:
                ay = self.ayanamsa_deg
                return {k: norm360_fast(v - ay) for k, v in got.items()}
            return got
        except Exception:
            return {}
    
    def _precompute_drishti_schemas(self, movers: List[str]) -> None:
        """Pre-compute all drishti schemas to avoid repeated calculations."""
        config_key = (tuple(sorted(movers)), self.treat_nodes_like_saturn)
        if self._last_schema_config == config_key:
            return
            
        self._drishti_schemas.clear()
        for body in movers:
            canon_body = _node_canon(body)
            p = canon_body
            if p in ("Rahu", "Ketu") and self.treat_nodes_like_saturn:
                p = "Saturn"
            schema = graha_drishti_schema(p)
            if _node_canon(body) in ("Rahu", "Ketu") and not self.treat_nodes_like_saturn:
                schema = {7: 1.0}  # Only 7th house drishti
            self._drishti_schemas[canon_body] = schema
        
        self._last_schema_config = config_key
    
    def _get_adaptive_step(self, body: str, base_step_minutes: float) -> float:
        """Calculate adaptive step size based on planetary speed."""
        speed = _PLANET_SPEEDS.get(body.lower(), 0.5)
        
        # Faster planets need smaller steps
        if speed > 5.0:      # Moon
            return base_step_minutes * 0.4
        elif speed > 1.0:    # Mercury, Venus, Sun
            return base_step_minutes * 0.7
        elif speed > 0.1:    # Mars
            return base_step_minutes * 1.0
        else:                # Jupiter, Saturn, Nodes
            return base_step_minutes * 2.0
    
    def scan_drishti_optimized(
        self,
        *,
        jd_start_tt: float,
        jd_end_tt: float,
        movers: List[str],
        targets: Dict[str, float],
        orb_deg: float = 12.0,
        orb_map: Optional[Dict[str, float]] = None,
        step_minutes: Union[str, float, int] = "auto",
        scan_mode: str = "balanced",
    ) -> List[GocharEvent]:
        """Optimized drishti scanning with batch ephemeris and adaptive stepping."""
        
        if jd_end_tt <= jd_start_tt or not movers or not targets:
            return []
        
        # Pre-compute all drishti schemas
        self._precompute_drishti_schemas(movers)
        
        # Calculate base step size
        window_days = jd_end_tt - jd_start_tt
        if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
            if window_days <= 3.0: base_step = 30.0
            elif window_days <= 14.0: base_step = 60.0
            else: base_step = 180.0
            
            # Adjust for scan mode
            mode = scan_mode.lower()
            if mode.startswith("ultra"): base_step *= 2.0
            elif mode.startswith("fine"): base_step *= 0.5
        else:
            base_step = float(step_minutes)
        
        # Initialize sliding window cache
        self.sliding_window.ensure_window(
            (jd_start_tt + jd_end_tt) / 2, 
            window_days + 1.0,  # Extra margin
            movers
        )
        
        events: List[GocharEvent] = []
        dedupe: set[Tuple[str, str, int, int]] = set()  # Integer keys for performance
        
        # Adaptive stepping per planet
        planet_steps = {}
        for body in movers:
            planet_steps[body] = self._get_adaptive_step(body, base_step)
        
        # Scan with planet-specific adaptive steps
        current_time = jd_start_tt
        
        while current_time < jd_end_tt - 1e-12:
            # Calculate next time step (minimum across all planets)
            min_step = min(planet_steps.values())
            next_time = min(current_time + min_step / (60.0 * 24.0), jd_end_tt)
            
            # Batch fetch all longitudes for current and next time
            current_lons = {}
            next_lons = {}
            
            for body in movers:
                current_lons[body] = self.sliding_window.get_longitude(body, current_time)
                next_lons[body] = self.sliding_window.get_longitude(body, next_time)
            
            # Vectorized separation calculations
            separations = self._calculate_bulk_separations(
                current_lons, next_lons, targets, current_time, next_time
            )
            
            # Process potential events
            for sep_data in separations:
                if self._should_create_event(sep_data, orb_deg, orb_map):
                    event = self._create_gochar_event(sep_data, dedupe)
                    if event:
                        events.append(event)
            
            current_time = next_time
        
        events.sort(key=lambda e: (e.jd_tt, e.body, e.target, e.axis_deg))
        return events
    
    def _calculate_bulk_separations(
        self, 
        current_lons: Dict[str, Optional[float]], 
        next_lons: Dict[str, Optional[float]], 
        targets: Dict[str, float], 
        t0: float, 
        t1: float
    ) -> List[Dict[str, Any]]:
        """Vectorized separation calculations for all body-target-aspect combinations."""
        
        separations = []
        dt_days = max(1e-9, t1 - t0)
        
        for body in current_lons:
            canon_body = _node_canon(body)
            schema = self._drishti_schemas.get(canon_body, {})
            
            lon0 = current_lons.get(body)
            lon1 = next_lons.get(body)
            
            if lon0 is None or lon1 is None:
                continue
            
            # Calculate motion for this planet
            dlon = wrap180_fast(lon1 - lon0)
            speed_deg_per_day = dlon / dt_days if dt_days > 0 else 0.0
            
            for target_name, target_lon in targets.items():
                for k, weight in schema.items():
                    if k not in _DRISHTI_AXES:
                        continue
                        
                    axis = _DRISHTI_AXES[k]
                    
                    # Current and next separations
                    sep0 = wrap180_fast(angdiff_fast(lon0, target_lon) - axis)
                    sep1 = wrap180_fast(angdiff_fast(lon1, target_lon) - axis)
                    
                    # Check for potential crossing or close approach
                    if abs(sep0) > 15.0 and abs(sep1) > 15.0 and sep0 * sep1 > 0:
                        continue  # Too far and not crossing
                    
                    separations.append({
                        'body': canon_body,
                        'target': _node_canon(target_name),
                        'k': k,
                        'axis': axis,
                        'weight': weight,
                        'sep0': sep0,
                        'sep1': sep1,
                        't0': t0,
                        't1': t1,
                        'speed': speed_deg_per_day,
                        'applying': abs(sep1) < abs(sep0)
                    })
        
        return separations
    
    def _should_create_event(
        self, 
        sep_data: Dict[str, Any], 
        orb_deg: float, 
        orb_map: Optional[Dict[str, float]]
    ) -> bool:
        """Determine if separation data warrants event creation."""
        body = sep_data['body']
        sep0, sep1 = sep_data['sep0'], sep_data['sep1']
        
        _orb = float(orb_map.get(body, orb_deg)) if orb_map else orb_deg
        
        # Check if within orb or crossing zero
        min_sep = min(abs(sep0), abs(sep1))
        sign_change = (sep0 == 0.0) or (sep1 == 0.0) or (sep0 * sep1 < 0.0)
        
        return min_sep <= _orb or sign_change
    
    def _create_gochar_event(
        self, 
        sep_data: Dict[str, Any], 
        dedupe: set[Tuple[str, str, int, int]]
    ) -> Optional[GocharEvent]:
        """Create GocharEvent from separation data with deduplication."""
        
        body = sep_data['body']
        target = sep_data['target']
        k = sep_data['k']
        t0, t1 = sep_data['t0'], sep_data['t1']
        sep0, sep1 = sep_data['sep0'], sep_data['sep1']
        speed = sep_data['speed']
        
        # Estimate exact time of minimum separation
        if abs(speed) > 1e-9 and sep0 * sep1 <= 0:  # Zero crossing
            te = t0 - (sep0 / speed)  # Linear interpolation to zero
            te = max(t0, min(t1, te))  # Clamp to interval
        else:
            te = t0 if abs(sep0) < abs(sep1) else t1
        
        # Get exact longitude at event time for final separation
        event_lon = self.sliding_window.get_longitude(body, te)
        if event_lon is None:
            event_sep = min(sep0, sep1, key=abs)
        else:
            target_lon = None
            for tgt, tlon in sep_data.get('targets', {}).items():
                if _node_canon(tgt) == target:
                    target_lon = tlon
                    break
            if target_lon is not None:
                event_sep = wrap180_fast(angdiff_fast(event_lon, target_lon) - sep_data['axis'])
            else:
                event_sep = min(sep0, sep1, key=abs)
        
        # Deduplication with integer keys (faster than string concatenation)
        body_hash = hash(body) % 10000
        target_hash = hash(target) % 10000
        time_bucket = int(te * 86400.0 + 0.5)  # 1-second buckets
        
        dedup_key = (body_hash, target_hash, k, time_bucket)
        if dedup_key in dedupe:
            return None
        dedupe.add(dedup_key)
        
        return GocharEvent(
            jd_tt=float(te),
            body=body,
            target=target,
            drishti=f"{k}th" if k != 7 else "7th",
            axis_deg=float(sep_data['axis']),
            separation_deg=float(event_sep),
            applying=bool(sep_data['applying']),
            exact=abs(event_sep) <= 1e-6,
            weight=float(sep_data['weight']),
            meta={"k_house": k}
        )


# ──────────────────────────────────────────────────────────────────────
# OPTIMIZED INGRESS SCANNERS: Event-driven approach
# ──────────────────────────────────────────────────────────────────────
class OptimizedIngressScanner:
    """Event-driven ingress scanner for signs and nakshatras."""
    
    def __init__(self, engine: VedicTransitEngine):
        self.engine = engine
    
    def find_sign_ingresses(
        self, 
        jd_start: float, 
        jd_end: float, 
        movers: List[str],
        step_minutes: Union[str, float, int] = "auto"
    ) -> List[Dict[str, Any]]:
        """Find rashi (sign) ingresses using event-driven scanning."""
        
        if jd_end <= jd_start:
            return []
        
        # Calculate adaptive step
        window_days = jd_end - jd_start
        if isinstance(step_minutes, str):
            base_step = 30.0 if window_days <= 7 else 60.0
        else:
            base_step = float(step_minutes)
        
        # Prepare sliding window
        self.engine.sliding_window.ensure_window(
            (jd_start + jd_end) / 2, window_days + 0.5, movers
        )
        
        events = []
        
        for body in movers:
            canon_body = _node_canon(body)
            step_days = self.engine._get_adaptive_step(body, base_step) / (60.0 * 24.0)
            
            t = jd_start
            last_sign = None
            
            while t < jd_end:
                lon = self.engine.sliding_window.get_longitude(body, t)
                if lon is not None:
                    current_sign = int(lon // 30.0) % 12
                    
                    if last_sign is not None and current_sign != last_sign:
                        # Sign change detected - refine timing
                        exact_time = self._refine_ingress_time(
                            body, t - step_days, t, 30.0
                        )
                        if exact_time is not None:
                            events.append({
                                "body": canon_body,
                                "exact_jd_tt": float(exact_time),
                                "sign_to": int(current_sign)
                            })
                    
                    last_sign = current_sign
                
                t += step_days
        
        return sorted(events, key=lambda x: (x["exact_jd_tt"], x["body"]))
    
    def find_nakshatra_ingresses(
        self, 
        jd_start: float, 
        jd_end: float, 
        movers: List[str],
        step_minutes: Union[str, float, int] = "auto"
    ) -> List[Dict[str, Any]]:
        """Find nakshatra ingresses using event-driven scanning."""
        
        if jd_end <= jd_start:
            return []
        
        # Calculate adaptive step
        window_days = jd_end - jd_start
        if isinstance(step_minutes, str):
            base_step = 15.0 if window_days <= 7 else 30.0
        else:
            base_step = float(step_minutes)
        
        # Prepare sliding window
        self.engine.sliding_window.ensure_window(
            (jd_start + jd_end) / 2, window_days + 0.5, movers
        )
        
        nak_width = 360.0 / 27.0
        events = []
        
        for body in movers:
            canon_body = _node_canon(body)
            step_days = self.engine._get_adaptive_step(body, base_step) / (60.0 * 24.0)
            
            t = jd_start
            last_nak = None
            
            while t < jd_end:
                lon = self.engine.sliding_window.get_longitude(body, t)
                if lon is not None:
                    current_nak = nakshatra_index(lon)
                    
                    if last_nak is not None and current_nak != last_nak:
                        # Nakshatra change detected - refine timing
                        exact_time = self._refine_ingress_time(
                            body, t - step_days, t, nak_width
                        )
                        if exact_time is not None:
                            events.append({
                                "body": canon_body,
                                "exact_jd_tt": float(exact_time),
                                "nakshatra_index": int(current_nak),
                                "nakshatra_name": NAKSHATRAS_27[(current_nak-1) % 27]
                            })
                    
                    last_nak = current_nak
                
                t += step_days
        
        return sorted(events, key=lambda x: (x["exact_jd_tt"], x["body"]))
    
    def _refine_ingress_time(
        self, 
        body: str, 
        t_start: float, 
        t_end: float, 
        division_width: float
    ) -> Optional[float]:
        """Refine ingress timing using binary search."""
        
        def residual(t: float) -> float:
            lon = self.engine.sliding_window.get_longitude(body, t)
            if lon is None:
                return 0.0
            return (lon % division_width) - 0.0  # Distance from boundary
        
        # Binary search for zero crossing
        for _ in range(20):  # Max 20 iterations
            if t_end - t_start < 1e-6:  # 1-second precision
                break
                
            t_mid = (t_start + t_end) / 2
            
            r_start = residual(t_start)
            r_mid = residual(t_mid)
            
            if abs(r_mid) < 1e-8:  # Found exact crossing
                return t_mid
            
            if r_start * r_mid < 0:  # Zero in first half
                t_end = t_mid
            else:  # Zero in second half
                t_start = t_mid
        
        return (t_start + t_end) / 2


# ──────────────────────────────────────────────────────────────────────
# OPTIMIZED STATION FINDER: Batch velocity calculations
# ──────────────────────────────────────────────────────────────────────
class OptimizedStationFinder:
    """High-performance retrograde/direct station finder."""
    
    def __init__(self, engine: VedicTransitEngine):
        self.engine = engine
        self.velocity_cache: Dict[Tuple[str, float], float] = {}
    
    def find_stations(
        self, 
        jd_start: float, 
        jd_end: float, 
        movers: List[str],
        step_minutes: Union[str, float, int] = "auto"
    ) -> List[Dict[str, Any]]:
        """Find retrograde/direct stations using optimized velocity calculations."""
        
        if jd_end <= jd_start:
            return []
        
        # Calculate adaptive step
        window_days = jd_end - jd_start
        if isinstance(step_minutes, str):
            base_step = 120.0 if window_days <= 30 else 240.0  # Larger steps for stations
        else:
            base_step = float(step_minutes)
        
        # Prepare sliding window with extra margin for velocity calculations
        self.engine.sliding_window.ensure_window(
            (jd_start + jd_end) / 2, window_days + 2.0, movers
        )
        
        events = []
        
        for body in movers:
            canon_body = _node_canon(body)
            
            # Skip Sun and Moon (don't go retrograde)
            if canon_body.lower() in ("sun", "moon"):
                continue
            
            step_days = self.engine._get_adaptive_step(body, base_step) / (60.0 * 24.0)
            velocity_dt = min(step_days / 4, 0.5)  # Half-day for velocity calculation
            
            t = jd_start
            last_velocity = None
            
            while t < jd_end - velocity_dt:
                current_velocity = self._calculate_velocity(body, t, velocity_dt)
                
                if last_velocity is not None and current_velocity is not None:
                    # Check for velocity sign change (station)
                    if last_velocity * current_velocity < 0:
                        # Station detected - refine timing
                        station_time, station_type = self._refine_station_time(
                            body, t - step_days, t, velocity_dt
                        )
                        
                        if station_time is not None:
                            events.append({
                                "body": canon_body,
                                "exact_jd_tt": float(station_time),
                                "kind": station_type
                            })
                
                last_velocity = current_velocity
                t += step_days
        
        return sorted(events, key=lambda x: (x["exact_jd_tt"], x["body"]))
    
    def _calculate_velocity(self, body: str, time: float, dt: float) -> Optional[float]:
        """Calculate velocity using cached three-point method."""
        
        cache_key = (body, round(time * 1440))  # Cache per minute
        if cache_key in self.velocity_cache:
            return self.velocity_cache[cache_key]
        
        # Three-point velocity calculation
        lon_before = self.engine.sliding_window.get_longitude(body, time - dt)
        lon_after = self.engine.sliding_window.get_longitude(body, time + dt)
        
        if lon_before is None or lon_after is None:
            return None
        
        # Handle longitude wrapping
        delta_lon = angdiff_fast(lon_after, lon_before)
        velocity = delta_lon / (2.0 * dt)  # degrees per day
        
        # Cache result
        self.velocity_cache[cache_key] = velocity
        
        # Limit cache size
        if len(self.velocity_cache) > 1000:
            # Remove oldest 25% of entries
            old_keys = list(self.velocity_cache.keys())[:250]
            for k in old_keys:
                self.velocity_cache.pop(k, None)
        
        return velocity
    
    def _refine_station_time(
        self, 
        body: str, 
        t_start: float, 
        t_end: float, 
        velocity_dt: float
    ) -> Tuple[Optional[float], str]:
        """Refine station timing using binary search on velocity."""
        
        def velocity_at(t: float) -> float:
            v = self._calculate_velocity(body, t, velocity_dt)
            return v if v is not None else 0.0
        
        # Binary search for velocity zero
        for _ in range(15):  # Max 15 iterations
            if t_end - t_start < 1e-5:  # High precision
                break
                
            t_mid = (t_start + t_end) / 2
            
            v_start = velocity_at(t_start)
            v_mid = velocity_at(t_mid)
            
            if abs(v_mid) < 1e-6:  # Found station
                break
            
            if v_start * v_mid < 0:  # Zero in first half
                t_end = t_mid
            else:  # Zero in second half
                t_start = t_mid
        
        station_time = (t_start + t_end) / 2
        
        # Determine station type by checking velocity trend
        v_before = velocity_at(station_time - 1.0)  # 1 day before
        v_after = velocity_at(station_time + 1.0)   # 1 day after
        
        if v_before > 0 and v_after < 0:
            station_type = "retrograde"
        elif v_before < 0 and v_after > 0:
            station_type = "direct"
        else:
            station_type = "station"
        
        return station_time, station_type


# ──────────────────────────────────────────────────────────────────────
# WINDOW RESOLUTION AND OPTIMIZATION HELPERS
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
    """Optimized natal target resolution with caching."""
    
    # 1) explicit longitudes?
    for key in ("longitudes", "ecliptic_longitudes"):
        m = natal_chart.get(key)
        if isinstance(m, dict) and m:
            got = {_node_canon(k): float(v) for k, v in m.items() if k is not None}
            out = {k: (norm360_fast(v - ay) if eng.sidereal_mode else norm360_fast(v)) for k, v in got.items()}
            need_angles = ("Asc" in tgts) or ("MC" in tgts)
            if need_angles:
                lat = natal_chart.get("latitude"); lon = natal_chart.get("longitude")
                jd_tt = natal_chart.get("natal_jd_tt") or natal_chart.get("jd_tt")
                if isinstance(lat, (int,float)) and isinstance(lon, (int,float)) and isinstance(jd_tt, (int,float)):
                    out.update(_angles_sidereal_deg(jd_tt=float(jd_tt), lat=float(lat), lon=float(lon), eng=eng))
            return out

    # 2) natal_jd_tt path
    natal_jd_tt = None
    for k in ("natal_jd_tt","jd_tt","jd_utc"):
        if isinstance(natal_chart.get(k), (int, float)):
            natal_jd_tt = float(natal_chart[k]); break

    out: Dict[str, float] = {}

    if isinstance(natal_jd_tt, float):
        planet_tgts = [x for x in tgts if x not in ("Asc","MC")]
        if planet_tgts:
            rows = ep.ecliptic_longitudes(float(natal_jd_tt), _batch_map_nodes(planet_tgts)).get("results", [])
            got = {_node_canon(str(r["name"])): float(r["longitude"]) for r in rows or []}
            out.update({k: (norm360_fast(v - ay) if eng.sidereal_mode else norm360_fast(v)) for k, v in got.items()})
        if ("Asc" in tgts) or ("MC" in tgts):
            lat = natal_chart.get("latitude"); lon = natal_chart.get("longitude")
            if isinstance(lat, (int,float)) and isinstance(lon, (int,float)):
                out.update(_angles_sidereal_deg(jd_tt=float(natal_jd_tt), lat=float(lat), lon=float(lon), eng=eng))
        return out

    # 3) civil + site (angles only; planets sampled at window-left if needed)
    date = natal_chart.get("date") or natal_chart.get("birth_date")
    time = natal_chart.get("time") or natal_chart.get("birth_time") or "12:00:00"
    tz   = natal_chart.get("tz") or natal_chart.get("place_tz")
    lat  = natal_chart.get("latitude"); lon = natal_chart.get("longitude")

    if isinstance(date, str) and isinstance(tz, str) and isinstance(lat, (int,float)) and isinstance(lon, (int,float)):
        try:
            ju = _civil_to_jd_utc(str(date), str(time), str(tz))
            try: y, m = map(int, str(date).split("-")[:2])
            except Exception: y, m = 2000, 1
            jdtt = _jd_tt_from_utc_jd(float(ju), y, m)
            if ("Asc" in tgts) or ("MC" in tgts):
                out.update(_angles_sidereal_deg(jd_tt=float(jdtt), lat=float(lat), lon=float(lon), eng=eng))
        except Exception:
            pass

    return out


# ──────────────────────────────────────────────────────────────────────
# HIGH-PERFORMANCE MAIN FUNCTIONS
# ──────────────────────────────────────────────────────────────────────
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
    **kwargs,
) -> Dict[str, Any]:
    """Optimized gochar finding with batch processing and adaptive stepping."""
    
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "gochar": [], "meta": {}}

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

    tgts = list(natal_targets or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Asc","MC"])

    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        treat_nodes_like_saturn=bool(treat_nodes_like_saturn or False),
        cache_size=min(5000, int((b - a) * 100))  # Adaptive cache size
    )
    eng.sidereal_mode = not zodiac_mode.startswith("tropical")
    eng.ayanamsa_deg = ay

    # Resolve natal targets
    if isinstance(target_lon_map, dict) and target_lon_map:
        nat_map = {_node_canon(k): float(v) for k, v in target_lon_map.items()}
        targets = {k: (norm360_fast(v - ay) if eng.sidereal_mode else norm360_fast(v)) for k, v in nat_map.items()}
    else:
        targets = _pick_natal_targets_map(ep=ep, eng=eng, natal_chart=natal_chart or {}, tgts=tgts, ay=ay)
        if not targets:
            rows = ep.ecliptic_longitudes(float(a), _batch_map_nodes([x for x in tgts if x not in ("Asc","MC")])).get("results", [])
            nat_map = {_node_canon(str(r["name"])): float(r["longitude"]) for r in rows or []}
            targets = {k: (norm360_fast(v - ay) if eng.sidereal_mode else norm360_fast(v)) for k, v in nat_map.items()}

    try:
        # Use optimized scanner
        evs = eng.scan_drishti_optimized(
            jd_start_tt=float(a), jd_end_tt=float(b),
            movers=movers, targets=targets, orb_deg=float(orb_deg or 12.0),
            orb_map=orb_map, step_minutes=step_minutes,
            scan_mode=(scan_mode or "balanced"),
        )
    except Exception as e:
        return {"ok": False, "error": f"gochar_scan_failed:{e}", "gochar": [], "meta": {"window_jd_tt": [a, b]}}

    # Format results
    hits: List[Dict[str, Any]] = []
    for ev in evs:
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
            "exact_jd_ut1": None,
            "exact_datetime_utc": None,
        })

    meta_out = {
        "movers": movers,
        "natal_targets": list(targets.keys()),
        "window_jd_tt": [float(a), float(b)],
        "performance_mode": "optimized",
        **meta_ts,
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa_deg": ay,
        "scan_mode": (scan_mode or "balanced"),
    }

    return {"ok": True, "technique": "gochar_drishti_optimized", "gochar": hits, "meta": meta_out}

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
    """Optimized rashi ingress finding."""
    
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

    movers = list(movers or ["Sun","Mercury","Venus","Mars","Jupiter","Saturn"])
    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m,
        cache_size=min(3000, int((b - a) * 50))
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    scanner = OptimizedIngressScanner(eng)
    events = scanner.find_sign_ingresses(a, b, movers, step_minutes)

    return {"ok": True, "ingresses": events,
            "meta": {"movers": movers, "window_jd_tt": [float(a), float(b)],
                     "performance_mode": "optimized",
                     "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg), **meta_ts}}

def find_nakshatra_ingresses_in_range(
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
    """Optimized nakshatra ingress finding."""
    
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
    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m,
        cache_size=min(3000, int((b - a) * 60))  # Higher resolution for nakshatras
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    scanner = OptimizedIngressScanner(eng)
    events = scanner.find_nakshatra_ingresses(a, b, movers, step_minutes)

    return {"ok": True, "ingresses": events,
            "meta": {"movers": movers, "window_jd_tt": [float(a), float(b)],
                     "performance_mode": "optimized",
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
    """Optimized station finding with batch velocity calculations."""
    
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
    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m,
        cache_size=min(2000, int((b - a) * 30))  # Moderate cache for stations
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    finder = OptimizedStationFinder(eng)
    events = finder.find_stations(a, b, movers, step_minutes)

    return {"ok": True, "stations": events,
            "meta": {"movers": movers, "window_jd_tt": [float(a), float(b)],
                     "performance_mode": "optimized",
                     "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa_deg": float(ayanamsa_deg), **meta_ts}}


# ──────────────────────────────────────────────────────────────────────
# Feature builder (unchanged but optimized for new data structures)
# ──────────────────────────────────────────────────────────────────────
def feature_drishti_proximity(*, hits: List[Dict[str, Any]], cap_deg: float = 12.0) -> List[float]:
    """Calculate proximity features for drishti hits."""
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
# Route-compatible wrapper functions (optimized)
# ──────────────────────────────────────────────────────────────────────
def _extract_observer(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract observer configuration from kwargs."""
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
    scan_mode: str = "balanced",
    **kwargs,
) -> Dict[str, Any]:
    """High-performance gochar drishti calculation."""
    
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
        scan_mode=scan_mode,
        **obs,
        **body_like,
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
    """High-performance rashi ingress calculation."""
    
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
    """High-performance nakshatra ingress calculation."""
    
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
    """High-performance station calculation."""
    
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
