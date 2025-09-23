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

Internals:
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

# Angles / houses via astronomy.py (preferred)
try:
    import app.core.astronomy as _astro
except Exception:
    _astro = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────
# Adapter factory (backward-compatible; never passes timescale into Config)
# ──────────────────────────────────────────────────────────────────────
def _make_ephem(frame: str = "ecliptic-of-date") -> EphemerisAdapter:
    # Build Config without 'timescale'
    try:
        if PLANETS is not None:
            cfg = EphemConfig(frame=frame, planets=PLANETS)  # type: ignore
        else:
            cfg = EphemConfig(frame=frame)  # type: ignore
    except TypeError:
        cfg = EphemConfig(frame=frame)  # type: ignore

    # Build Adapter with best available signature
    try:
        return EphemerisAdapter(cfg, timescale=TS)  # type: ignore[arg-type]
    except TypeError:
        try:
            return EphemerisAdapter(cfg, TS)  # type: ignore[misc]
        except TypeError:
            return EphemerisAdapter(cfg)  # type: ignore


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
# Timescale model (strict-friendly metadata like houses_advanced)
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
                out = fn(date=date, time=time, tz=tz)  # type: ignore[call-arg]
            except Exception:
                try: out = fn(date, time, tz)  # type: ignore[misc]
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
                        res = fn(jd_tt, lat, lon)  # type: ignore[misc]
                    if isinstance(res, dict):
                        asc = res.get("Asc") or res.get("asc") or res.get("ASC") or res.get("ascendant")
                        mc  = res.get("MC")  or res.get("mc")  or res.get("midheaven") or res.get("Medium Coeli")
                        if asc is None or mc is None:
                            continue
                        asc = float(asc); mc = float(mc)
                        if eng.sidereal_mode:
                            ay = float(eng.ayanamsa_deg)
                            return {"Asc": norm360(asc - ay), "MC": norm360(mc - ay)}
                        return {"Asc": norm360(asc), "MC": norm360(mc)}
                except Exception:
                    pass
    try:
        if hasattr(eng.ephem, "angles_ecliptic"):
            res = eng.ephem.angles_ecliptic(jd_tt, latitude=lat, longitude=lon, **eng.obs)  # type: ignore[misc]
            asc = float(res.get("Asc")); mc = float(res.get("MC"))
            if eng.sidereal_mode:
                ay = float(eng.ayanamsa_deg)
                return {"Asc": norm360(asc - ay), "MC": norm360(mc - ay)}
            return {"Asc": norm360(asc), "MC": norm360(mc)}
    except Exception:
        pass
    return {}


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
        self.ephem = ephem or _make_ephem(frame)
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
    def _refine_zero_brent(
        f: Callable[[float], float], a: float, b: float, fa: float, fb: float, *,
        max_iter: int = 32, tol_days: float = 1e-6
    ) -> float:
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


# ──────────────────────────────────────────────────────────────────────
# Window-aware AUTO step helpers (shared by all scanners)
# ──────────────────────────────────────────────────────────────────────
def _window_days(a: float, b: float) -> float:
    return max(0.0, float(b) - float(a))

def _baseline_step_by_window(days: float) -> int:
    """Return baseline minutes from window length (≤3d → 60m, 4–14d → 180m, >14d → 720m)."""
    if days <= 3.0: return 60
    if days <= 14.0: return 180
    return 720

def _baseline_step_by_window_fast(days: float) -> int:
    """Faster baseline for long spans (≤3d → 120m, 4–14d → 360m, >14d → 1440m)."""
    if days <= 3.0: return 120
    if days <= 14.0: return 360
    return 1440

def _min_cap_by_body(kind: str, name: str) -> int:
    """Planet caps per scan kind (minutes)."""
    n = (name or "").strip().lower()
    if kind == "drishti":
        if n == "moon": return 8
        if n in ("mercury","venus","mars"): return 20
        if n in ("sun","jupiter","saturn"): return 60
        if n in ("rahu","ketu","north node","south node"): return 90
        return 120
    if kind == "signs":
        if n == "moon": return 10
        if n in ("mercury","venus","mars"): return 30
        if n in ("sun","jupiter","saturn"): return 90
        return 180
    if kind == "nak":
        if n == "moon": return 5
        if n in ("mercury","venus","mars"): return 20
        return 60
    if kind == "stations":
        if n in ("mercury","venus"): return 60
        if n == "mars": return 120
        return 180  # jupiter/saturn/others
    return 60

def _auto_step_minutes(
    kind: str, movers: List[str], a: float, b: float, *,
    min_floor: int, fallback: int, scan_mode: str = "balanced"
) -> float:
    days = _window_days(a, b)
    base = _baseline_step_by_window_fast(days) if str(scan_mode).lower().startswith("fast") \
           else _baseline_step_by_window(days)
    caps: List[int] = [ _min_cap_by_body(kind, m) for m in (movers or []) ] or [fallback]
    step = max(base, min(caps))
    if str(scan_mode).lower().startswith("fine"):
        step = max(min_floor, int(step * 0.6))  # tighten in fine mode
    return float(max(min_floor, step))


# ──────────────────────────────────────────────────────────────────────
# Graha dṛṣṭi scan
# ──────────────────────────────────────────────────────────────────────
def _schema_for(planet: str, *, include_nodes: bool, treat_nodes_like_saturn: bool) -> Dict[int, float]:
    p = _node_canon(planet)
    if p in ("Rahu", "Ketu") and treat_nodes_like_saturn:
        p = "Saturn"
    sch = graha_drishti_schema(p)
    if _node_canon(planet) in ("Rahu", "Ketu") and not (treat_nodes_like_saturn or include_nodes):
        sch = ({7: 1.0} if include_nodes else {})
    return sch

class VedicTransitEngine(VedicTransitEngine):  # extend with scans
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
        scan_mode: str = "fast",          # "ultra" | "fast" | "balanced" | "fine"
        snap_policy: str = "near",        # "never" | "near" | "always"
    ) -> List[GocharEvent]:
        if jd_end_tt <= jd_start_tt or not movers or not targets:
            return []

        # window-aware AUTO (favor larger steps in faster modes)
        if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
            base = _auto_step_minutes("drishti", movers, jd_start_tt, jd_end_tt, min_floor=6, fallback=30)
            if str(scan_mode).lower().startswith("ultra"):   base *= 1.75
            elif str(scan_mode).lower().startswith("fast"):  base *= 1.3
            step_minutes = float(base)
        try:
            dt = float(step_minutes) / (60.0 * 24.0)
        except Exception:
            dt = 30.0 / (60.0 * 24.0)

        mode = str(scan_mode).lower()
        ultra = mode.startswith("ultra")
        fast  = ultra or mode.startswith("fast")
        fine  = mode.startswith("fine")

        # tighter cap → fewer ephemeris reads for snap checks
        max_refine_per_step = 2 if ultra else (3 if fast else (6 if fine else 4))
        # only snap when close
        snap = str(snap_policy).lower()
        snap_near = (snap == "near")
        snap_never = (snap == "never")

        events: List[GocharEvent] = []
        dedupe: set[Tuple[str, str, str, int]] = set()

        # preload t0
        t0 = float(jd_start_tt)
        l0 = self._lon_map(t0, movers)
        for m, v in l0.items():
            _ = self._lon_cache.get_put((_node_canon(m), t0), lambda vv=v: float(vv))

        while t0 < jd_end_tt - 1e-12:
            t1 = float(min(t0 + dt, jd_end_tt))
            # preload t1
            l1 = self._lon_map(t1, movers)
            for m, v in l1.items():
                _ = self._lon_cache.get_put((_node_canon(m), t1), lambda vv=v: float(vv))

            refinements_left = max_refine_per_step

            for body in movers:
                bcanon = _node_canon(body)
                if bcanon in ("Rahu", "Ketu") and not (include_nodes or self.treat_nodes_like_saturn):
                    continue

                lon0 = l0.get(body); lon1 = l1.get(body)
                if lon0 is None or lon1 is None:
                    continue

                dlon = wrap180(lon1 - lon0)
                dt_days = max(1e-9, t1 - t0)
                dsep_dt = dlon / dt_days  # deg/day (linearized)

                # broad guard: how far can separation move this step
                max_change_possible = abs(dsep_dt) * dt_days + 0.3

                schema = _schema_for(body, include_nodes=include_nodes, treat_nodes_like_saturn=self.treat_nodes_like_saturn)
                if not schema:
                    continue

                for tgt_name, tgt_lon in targets.items():
                    for k, weight in schema.items():
                        if refinements_left <= 0:
                            break

                        axis = (k * 30.0) % 360.0
                        s0 = wrap180(angdiff(lon0, tgt_lon) - axis)
                        s1 = wrap180(angdiff(lon1, tgt_lon) - axis)

                        _orb = float(orb_map.get(body, orb_deg)) if isinstance(orb_map, dict) else float(orb_deg)

                        # hard far-prune by orb reachability
                        if min(abs(s0), abs(s1)) > (_orb + max_change_possible):
                            continue

                        sign_change = (s0 == 0.0) or (s1 == 0.0) or ((s0 * s1) < 0.0)
                        if ultra and not sign_change:
                            continue

                        # linear solve for zero (or nearest point in step)
                        if abs(dsep_dt) < 1e-9:
                            if not sign_change:
                                continue
                            te_lin = 0.5 * (t0 + t1)
                        else:
                            te_lin = t0 - (s0 / dsep_dt)
                            if te_lin < t0 or te_lin > t1:
                                if fast and not sign_change:
                                    continue
                                te_lin = max(t0, min(t1, te_lin))

                        # estimate separation at te_lin without a read
                        # sep_lin ≈ s0 + dsep_dt*(te_lin - t0)
                        sep_lin = s0 + dsep_dt * (te_lin - t0)

                        # decide whether to do an actual ephemeris read ("snap")
                        do_snap = (snap == "always")
                        if snap_near and abs(sep_lin) <= min(_orb, 0.5):  # only snap when promising
                            do_snap = True

                        if do_snap and refinements_left > 0 and not snap_never:
                            refinements_left -= 1
                            lon_now = self._lon_cached(body, te_lin)
                            sep_now = wrap180(angdiff(lon_now, tgt_lon) - axis)

                            # single Newton correction if still not close
                            if abs(dsep_dt) > 1e-9 and abs(sep_now) > 0.03:
                                te_corr = te_lin - (sep_now / dsep_dt)
                                if t0 <= te_corr <= t1:
                                    lon_now = self._lon_cached(body, te_corr)
                                    sep_now = wrap180(angdiff(lon_now, tgt_lon) - axis)
                                    te_lin = te_corr
                        else:
                            # trust linear prediction (zero extra ephemeris calls)
                            sep_now = sep_lin

                        if abs(sep_now) > _orb and not sign_change:
                            continue

                        applying = (abs(s1) < abs(s0))
                        bucket = int(math.floor(te_lin * 86400.0 + 0.5))  # 1s bucket
                        key = (bcanon, _node_canon(tgt_name), f"{k}th", bucket)
                        if key in dedupe:
                            continue
                        dedupe.add(key)

                        events.append(GocharEvent(
                            jd_tt=float(te_lin),
                            body=bcanon,
                            target=_node_canon(tgt_name),
                            drishti=(f"{k}th" if k != 7 else "7th"),
                            axis_deg=float(axis),
                            separation_deg=float(sep_now),
                            applying=bool(applying),
                            exact=abs(sep_now) <= 1e-6,
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
    # 1) explicit longitudes?
    for key in ("longitudes", "ecliptic_longitudes"):
        m = natal_chart.get(key)
        if isinstance(m, dict) and m:
            got = {_node_canon(k): float(v) for k, v in m.items() if k is not None}
            out = {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in got.items()}
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
            out.update({k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in got.items()})
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
            rows = ep.ecliptic_longitudes(float(a), _batch_map_nodes([x for x in tgts if x not in ("Asc","MC")])).get("results", [])
            nat_map = {_node_canon(str(r["name"])): float(r["longitude"]) for r in rows or []}
            targets = {k: (norm360(v - ay) if eng.sidereal_mode else norm360(v)) for k, v in nat_map.items()}

    try:
        evs = eng.scan_drishti(
            jd_start_tt=float(a), jd_end_tt=float(b),
            movers=movers, targets=targets, orb_deg=float(orb_deg or 12.0),
            orb_map=orb_map, step_minutes=step_minutes, include_nodes=use_nodes,
            scan_mode=(scan_mode or "balanced"),
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

    meta_out = {
        "movers": movers,
        "natal_targets": list(targets.keys()),
        "window_jd_tt": [float(a), float(b)],
        **meta_ts,
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa_deg": ay,
        "scan_mode": (scan_mode or "balanced"),
    }

    return {"ok": True, "technique": "gochar_drishti", "gochar": hits, "meta": meta_out}


# ──────────────────────────────────────────────────────────────────────
# Ingress: signs (rāśi)
# ──────────────────────────────────────────────────────────────────────
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
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
        step_minutes = _auto_step_minutes("signs", movers, a, b, min_floor=5, fallback=60)
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


# ──────────────────────────────────────────────────────────────────────
# Ingress: nakshatra (27)
# ──────────────────────────────────────────────────────────────────────
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
    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    width = 360.0 / 27.0

    if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
        step_minutes = _auto_step_minutes("nak", movers, a, b, min_floor=2, fallback=20)
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


# ──────────────────────────────────────────────────────────────────────
# Stations (retro/direct)
# ──────────────────────────────────────────────────────────────────────
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
    ep = _make_ephem(frame)
    eng = VedicTransitEngine(
        ephem=ep, frame=frame,
        topocentric=topocentric, latitude=latitude, longitude=longitude, elevation_m=elevation_m
    )
    eng.sidereal_mode = (zodiac_mode or "sidereal").lower().startswith("sidereal")
    eng.ayanamsa_deg = float(ayanamsa_deg)

    if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
        step_minutes = _auto_step_minutes("stations", movers, a, b, min_floor=15, fallback=120)
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
# Thin wrappers imported by routes
# ──────────────────────────────────────────────────────────────────────
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
    prebatch_refinement: bool = False,
    tz_name: str | None = None,
    scan_mode: str = "balanced",
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
        scan_mode=scan_mode,
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
