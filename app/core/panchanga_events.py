# app/core/panchanga_events.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Pañcāṅga Events — high-precision change times & timelines (research-grade)

What this provides:
- find_panchanga_changes(...)  → exact boundary times (JD_TT) for Tithi / Nakṣatra / Yoga / Karaṇa
- panchanga_timeline(...)      → contiguous segments with start/end JD_TT for each element
- compute_panchanga_events(...)→ route-friendly orchestrator over civil dates or JD windows

Numerics / design:
- Sidereal by subtraction of ayanāṁśa (app.core.ayanamsa.get_ayanamsa_deg) at each evaluation
- Sun & Moon ecliptic-of-date longitudes via EphemerisAdapter (batched; local LRU cache)
- Boundary detection with a safe scan (small fixed step) + Brent-like guarded refinement on
  f(t) = wrap180(A(t) − threshold), where A(t) is the relevant phase angle
- Deterministic dedupe at 1-second JD buckets for multiple elements firing close together
- Timeline builder closes the last segment at window end (or next boundary if inside window)

Elements and phase definitions (nirāyaṇa, i.e., sidereal longitudes):
- Tithi:     A_T = norm360( Moon − Sun ); boundaries every S_T = 12°
- Karaṇa:    A_K = norm360( Moon − Sun ); boundaries every S_K = 6°
- Nakṣatra:  A_N = norm360( Moon );      boundaries every S_N = 360/27 = 13⅓°
- Yoga:      A_Y = norm360( Moon + Sun );boundaries every S_Y = 360/27

Public API:
    find_panchanga_changes(jd_start_tt, jd_end_tt, *, ayanamsa_key="lahiri", step_minutes="auto",
                           elements=("tithi","nakshatra","yoga","karana")) -> dict
    panchanga_timeline(jd_start_tt, jd_end_tt, *, ayanamsa_key="lahiri", **kwargs) -> dict
    compute_panchanga_events(payload: dict) -> dict

Notes:
- This module does not compute sunrise/sunset directly; for day slicing or muhūrtas,
  use app.core.panchanga.sunrise_sunset_for_julian_day / muhurta_windows.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union, NamedTuple
import math

# ── Optional time helpers (consistent with astronomy.py / panchanga.py) ──
try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

# Ephemeris + ayanāṁśa + constants
from app.core.ephem_singleton import TS, PLANETS
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    _EPH_OK = False

from app.core.ayanamsa import get_ayanamsa_deg
from app.core.constants_vedic import NAKSHATRAS_27 as _NAK_27
from app.core.panchanga import panchanga_elements_at as _elements_at   # for names/pada, kept consistent

__all__ = [
    "find_panchanga_changes",
    "panchanga_timeline",
    "compute_panchanga_events",
]

# ────────────────────────────────────────────────────────────────────────
# Tiny math helpers
# ────────────────────────────────────────────────────────────────────────

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _wrap180(x: float) -> float:
    return ((float(x) + 180.0) % 360.0) - 180.0

# ────────────────────────────────────────────────────────────────────────
# Tithi / Yoga names (kept aligned with app/core/panchanga.py)
# ────────────────────────────────────────────────────────────────────────

_TITHI_BASE_15 = [
    "Pratipadā","Dvitīyā","Tṛtīyā","Caturthī","Pañcamī","Ṣaṣṭhī","Saptamī","Aṣṭamī","Navamī","Daśamī",
    "Ekādaśī","Dvādaśī","Trayodaśī","Caturdaśī","Paurṇimā/Amāvasyā seed",
]
_YOGA_27 = [
    "Viṣkambha","Prīti","Āyuṣmān","Saubhāgya","Śobhana","Atigaṇḍa","Sukarmā","Dhṛti","Śūla",
    "Gaṇḍa","Vṛddhi","Dhruva","Vyāghāta","Harṣaṇa","Vajra","Siddhi","Vyatīpāta","Vāriyana",
    "Parigha","Śiva","Siddha","Sādhya","Śubha","Śukla","Brahmā","Indra","Vaidhṛti",
]

# Karaṇa names (7 cara + 4 sthira) — identical rule as in panchanga.py
_CHARA_KARANAS = ["Bava","Bālava","Kaulava","Taitila","Gara","Vaṇij","Viṣṭi (Bhadrā)"]
_STHIRA_KARANAS = ["Kiṁstughna","Śakuni","Catuṣpāda","Nāga"]

def _karana_name_from_number(n: int) -> str:
    if n <= 1:
        return _STHIRA_KARANAS[0]
    if 2 <= n <= 57:
        return _CHARA_KARANAS[(n - 2) % 7]
    if n == 58:
        return _STHIRA_KARANAS[1]
    if n == 59:
        return _STHIRA_KARANAS[2]
    return _STHIRA_KARANAS[3]

# ────────────────────────────────────────────────────────────────────────
# Timescales helper (civil → JD) mirroring panchanga.py behavior
# ────────────────────────────────────────────────────────────────────────

class _TSOut(NamedTuple):
    jd_ut: float
    jd_tt: float
    jd_ut1: float
    warnings: List[str]

def _timescales_from_civil(date: str, time: str, tz: str, *, dut1_seconds: float = 0.0) -> _TSOut:
    warns: List[str] = []
    if _tk is not None:
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(date=date, time=time, tz=tz, dut1=dut1_seconds)
                except TypeError:
                    out = fn(date, time, tz, dut1_seconds)
                if isinstance(out, dict):
                    return _TSOut(float(out.get("jd_ut") or out.get("jd_utc")), float(out["jd_tt"]), float(out["jd_ut1"]), warns)
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    ju, jt, j1 = map(float, out[:3])
                    return _TSOut(ju, jt, j1, warns)
    if _ts is None:
        raise ValueError("timescales module not available")
    try:
        jd_ut = float(_ts.julian_day_utc(date, time, tz))
    except Exception as e:
        raise ValueError(f"Failed to compute JD_UTC from {date} {time} {tz}: {e}")
    try:
        y, m = map(int, date.split("-")[:2])
    except Exception:
        y, m = 2000, 1
    try:
        jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m))
    except Exception:
        jd_tt = jd_ut + 69.0/86400.0
        warns.append("deltaT_fallback_69s")
    jd_ut1 = jd_ut + float(dut1_seconds or 0.0)/86400.0
    return _TSOut(jd_ut, jd_tt, jd_ut1, warns)

# ────────────────────────────────────────────────────────────────────────
# Ephemeris context & caches
# ────────────────────────────────────────────────────────────────────────

class _EphemCtx:
    def __init__(self, frame: str = "ecliptic-of-date"):
        if not _EPH_OK:
            raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")
        self.ephem = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore
        self._lon_cache: Dict[float, Tuple[float, float]] = {}  # jd_tt → (sun_lon, moon_lon)
        self._max = 4096

    def sun_moon(self, jd_tt: float) -> Tuple[float, float]:
        t = float(jd_tt)
        v = self._lon_cache.get(t)
        if v is not None:
            return v
        rows = self.ephem.ecliptic_longitudes(t, ["Sun","Moon"]).get("results", [])
        if not rows:
            raise RuntimeError("ephemeris returned no Sun/Moon results")
        m = {str(r["name"]): float(r["longitude"]) for r in rows}
        out = (m["Sun"], m["Moon"])
        self._lon_cache[t] = out
        if len(self._lon_cache) > self._max:
            # prune oldest ~25%
            for k in sorted(self._lon_cache.keys())[: max(1, self._max // 4)]:
                self._lon_cache.pop(k, None)
        return out

# small ayanāṁśa cache (quantize at 1e-7 d)
def _ayan_deg_cached(jd_tt: float, key: str, _cache: Dict[Tuple[float,str], float] = {}) -> float:
    q = round(float(jd_tt) / 1e-7) * 1e-7
    ck = (q, key)
    v = _cache.get(ck)
    if v is not None:
        return v
    v = float(get_ayanamsa_deg(q, key))
    _cache[ck] = v
    return v

# ────────────────────────────────────────────────────────────────────────
# Root refiner (guarded Brent)
# ────────────────────────────────────────────────────────────────────────

def _brent_zero(f, a: float, b: float, fa: float, fb: float, *, tol: float = 5e-7, max_iter: int = 64) -> float:
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa * fb > 0.0:
        lo, hi = a, b
        flo, fhi = fa, fb
        for _ in range(max_iter):
            m = 0.5*(lo+hi)
            fm = f(m)
            if fm == 0.0 or abs(hi-lo) <= tol:
                return m
            if flo * fm <= 0:
                hi, fhi = m, fm
            else:
                lo, flo = m, fm
        return 0.5*(lo+hi)
    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if abs(fb) < abs(fa):
            a, b = b, a; fa, fb = fb, fa
        m = 0.5*(a + b)
        if abs(b - a) <= tol:
            return b
        if fa != fc and fb != fc:
            s = (a*fb*fc)/((fa-fb)*(fa-fc)) + (b*fa*fc)/((fb-fa)*(fb-fc)) + (c*fa*fb)/((fc-fa)*(fc-fb))
        else:
            s = b - fb*(b-a)/(fb-fa)
        cond = not ((3*a + b)/4 < s < b if a < b else b < s < (3*a + b)/4)
        cond |= (abs(s - b) >= abs(e)/2)
        cond |= (abs(e) < tol) or (abs(d) < tol)
        if cond:
            s = m
            d = e = b - a
        else:
            d, e = e, b - s
        fs = f(s)
        c, fc = a, fa
        if fa*fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
    return b

# ────────────────────────────────────────────────────────────────────────
# Phase engines for each element
# ────────────────────────────────────────────────────────────────────────

class _PhaseEngine:
    """
    Encapsulates A(t) and step size S for a given element; supplies:
      - angle_at(jd_tt)   → A(t) in [0,360)
      - grid_step_deg     → S in degrees
      - label(idx, ...)   → human label/details for timeline/change
      - index_from_angle  → integer index (1-based where standard)
    """
    def __init__(self, kind: str, ay_key: str, ephem: _EphemCtx):
        self.kind = kind
        self.ay_key = ay_key
        self.ephem = ephem
        if kind == "tithi":
            self.S = 12.0
        elif kind == "karana":
            self.S = 6.0
        else:
            self.S = 360.0 / 27.0

    def angle_at(self, jd_tt: float) -> float:
        sun_t, moon_t = self.ephem.sun_moon(jd_tt)
        ay = _ayan_deg_cached(jd_tt, self.ay_key)
        if self.kind == "nakshatra":
            return _norm360(moon_t - ay)
        elif self.kind == "yoga":
            return _norm360((moon_t + sun_t) - ay)  # (nirāyaṇa Moon + Sun)
        else:
            # tithi/karana use Moon − Sun (nirāyaṇa)
            return _norm360((moon_t - sun_t) - ay)

    @property
    def grid_step_deg(self) -> float:
        return self.S

    def index_from_angle(self, ang: float) -> int:
        if self.kind == "tithi":
            return int(math.floor(_norm360(ang) / 12.0)) + 1  # 1..30
        if self.kind == "nakshatra":
            return int(math.floor(_norm360(ang) / (360.0/27.0))) + 1  # 1..27
        if self.kind == "yoga":
            return int(math.floor(_norm360(ang) / (360.0/27.0))) + 1  # 1..27
        # karana number 1..60 following classical sequence: floor(2*tithi_float)+1
        k = int(math.floor(_norm360(ang) / 6.0)) + 1
        return max(1, min(60, k))

    def label(self, idx: int, *, angle: Optional[float] = None) -> Dict[str, Any]:
        if self.kind == "tithi":
            paksha = "Śukla" if idx <= 15 else "Kṛṣṇa"
            n = (idx - 1) % 15
            nm = _TITHI_BASE_15[n if n < 14 else 14]
            if n == 14:
                nm = "Paurṇimā" if paksha == "Śukla" else "Amāvasyā"
            return {"index": idx, "name": f"{paksha} {nm}", "paksha": paksha}
        if self.kind == "nakshatra":
            name = _NAK_27[(idx - 1) % 27]
            # derive pada if instantaneous angle provided
            if angle is not None:
                width = 360.0 / 27.0
                off = _norm360(angle) - ((idx - 1) * width)
                pada = int(math.floor((off / width) * 4.0)) + 1
            else:
                pada = None
            return {"index": idx, "name": name, "pada": pada}
        if self.kind == "yoga":
            return {"index": idx, "name": _YOGA_27[(idx - 1) % 27]}
        # karana
        return {"number": idx, "name": _karana_name_from_number(idx)}

# ────────────────────────────────────────────────────────────────────────
# Core scan + refinement
# ────────────────────────────────────────────────────────────────────────

def _auto_step_minutes(elements: Iterable[str]) -> int:
    elems = {e.lower().strip() for e in elements}
    # Ensure we never skip the fastest boundary: karaṇa ~ every ~11–12h
    # Safe defaults: 60 min for everything, 30 min if karaṇa included.
    return 30 if "karana" in elems else 60

def _threshold_after(ang: float, S: float) -> float:
    """Return the next grid boundary > ang on [0,360), as an absolute angle in [0,360)."""
    a = _norm360(ang)
    k_next = int(math.floor(a / S)) + 1
    thr = k_next * S
    return 0.0 if abs(thr - 360.0) < 1e-12 else _norm360(thr)

def _refine_boundary(engine: _PhaseEngine, t0: float, t1: float, *, S: float, a0: float, a1: float) -> float:
    """Refine exact boundary inside [t0,t1] for angle A crossing next grid multiple."""
    thr = _threshold_after(a0, S)
    def f(t: float) -> float:
        return _wrap180(engine.angle_at(t) - thr)
    v0 = _wrap180(a0 - thr)
    v1 = _wrap180(a1 - thr)
    return _brent_zero(f, t0, t1, v0, v1, tol=5e-7)

def find_panchanga_changes(
    jd_start_tt: float,
    jd_end_tt: float,
    *,
    ayanamsa_key: str = "lahiri",
    step_minutes: Union[str, float, int] = "auto",
    elements: Iterable[str] = ("tithi","nakshatra","yoga","karana"),
    frame: str = "ecliptic-of-date",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return change lists per element within [start, end):
      { "tithi": [ {jd_tt, index, name, ...}, ...], "nakshatra": [...], ... }
    """
    if not _EPH_OK:
        return {"tithi": [], "nakshatra": [], "yoga": [], "karana": []}
    if jd_end_tt <= jd_start_tt:
        return {"tithi": [], "nakshatra": [], "yoga": [], "karana": []}

    elems = [e.lower().strip() for e in elements if isinstance(e, str)]
    ctx = _EphemCtx(frame=frame)
    engines = {k: _PhaseEngine(k, ayanamsa_key, ctx) for k in elems}

    # step
    if isinstance(step_minutes, str) and step_minutes.lower() == "auto":
        step_minutes = _auto_step_minutes(elems)
    try:
        dt = float(step_minutes) / (24.0 * 60.0)
    except Exception:
        dt = _auto_step_minutes(elems) / (24.0 * 60.0)

    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ("tithi","nakshatra","yoga","karana")}
    dedupe: set[Tuple[str, int]] = set()  # (element, jd_bucket_sec)

    # initialize angles & indices
    t = float(jd_start_tt)
    a_prev: Dict[str, float] = {}
    idx_prev: Dict[str, int] = {}
    for k, eng in engines.items():
        a = eng.angle_at(t)
        a_prev[k] = a
        idx_prev[k] = eng.index_from_angle(a)

    while t < jd_end_tt - 1e-12:
        t2 = min(t + dt, jd_end_tt)
        for k, eng in engines.items():
            a0 = a_prev[k]
            i0 = idx_prev[k]
            a1 = eng.angle_at(t2)
            i1 = eng.index_from_angle(a1)
            if i1 != i0:
                # refine exact boundary inside [t, t2]
                t_exact = _refine_boundary(eng, t, t2, S=eng.grid_step_deg, a0=a0, a1=a1)
                # index after boundary == next index
                idx_new = ((i0 % (60 if k=="karana" else (27 if k in ("nakshatra","yoga") else 30))) + 1)
                # produce label, include pada for nakshatra using instantaneous angle at t_exact
                ang_exact = eng.angle_at(t_exact)
                lbl = eng.label(idx_new, angle=ang_exact)
                bucket = int(math.floor(t_exact * 86400.0 + 0.5))
                dkey = (k, bucket)
                if dkey not in dedupe:
                    dedupe.add(dkey)
                    rec = {"jd_tt": float(t_exact), **lbl}
                    out[k].append(rec)
            # update prev
            a_prev[k] = a1
            idx_prev[k] = i1
        t = t2

    # sort for deterministic order
    for k in out:
        out[k].sort(key=lambda r: r["jd_tt"])
    return out

# ────────────────────────────────────────────────────────────────────────
# Timeline builder
# ────────────────────────────────────────────────────────────────────────

def _timeline_from_changes(
    jd_start_tt: float,
    jd_end_tt: float,
    initial_idx: int,
    changes: List[Dict[str, Any]],
    *,
    element: str,
    engine: _PhaseEngine
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    cur_start = float(jd_start_tt)
    cur_idx = int(initial_idx)
    # helper to materialize label (with pada if applicable)
    def _label_at(idx: int, t_ref: float) -> Dict[str, Any]:
        ang = engine.angle_at(t_ref)
        return engine.label(idx, angle=ang)

    for ch in changes:
        t_sw = float(ch["jd_tt"])
        if t_sw <= cur_start + 1e-12:
            cur_idx = int((cur_idx % (60 if element=="karana" else (27 if element in ("nakshatra","yoga") else 30))) + 1)
            cur_start = t_sw
            continue
        lbl = _label_at(cur_idx, (cur_start + t_sw)/2.0)
        seg = {"start_jd_tt": cur_start, "end_jd_tt": t_sw, **lbl}
        segments.append(seg)
        # step to next
        cur_idx = int((cur_idx % (60 if element=="karana" else (27 if element in ("nakshatra","yoga") else 30))) + 1)
        cur_start = t_sw

    # tail
    lbl_tail = _label_at(cur_idx, (cur_start + float(jd_end_tt))/2.0)
    segments.append({"start_jd_tt": cur_start, "end_jd_tt": float(jd_end_tt), **lbl_tail})
    return segments

def panchanga_timeline(
    jd_start_tt: float,
    jd_end_tt: float,
    *,
    ayanamsa_key: str = "lahiri",
    elements: Iterable[str] = ("tithi","nakshatra","yoga","karana"),
    frame: str = "ecliptic-of-date",
    step_minutes: Union[str, float, int] = "auto",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build contiguous segments [start,end) with labels for each requested element.
    """
    if not _EPH_OK or jd_end_tt <= jd_start_tt:
        return {k: [] for k in ("tithi","nakshatra","yoga","karana")}
    changes = find_panchanga_changes(
        jd_start_tt, jd_end_tt, ayanamsa_key=ayanamsa_key,
        step_minutes=step_minutes, elements=elements, frame=frame
    )
    ctx = _EphemCtx(frame=frame)
    engines = {k: _PhaseEngine(k, ayanamsa_key, ctx) for k in [e for e in elements]}
    # initial indices at window start
    initial: Dict[str, int] = {}
    for k, eng in engines.items():
        initial[k] = eng.index_from_angle(eng.angle_at(jd_start_tt))

    out: Dict[str, List[Dict[str, Any]]] = {}
    for k in ("tithi","nakshatra","yoga","karana"):
        if k not in engines:
            out[k] = []
            continue
        out[k] = _timeline_from_changes(jd_start_tt, jd_end_tt, initial[k], changes.get(k, []), element=k, engine=engines[k])
    return out

# ────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────

def compute_panchanga_events(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (either JD window OR civil start):
      - jd_start_tt, jd_end_tt  (preferred for precision windows)
      - OR date="YYYY-MM-DD", tz="Area/City", days=int>=1  → window = local midnight..+days
      - ayanamsa (str) default "lahiri"
      - elements: list subset of ["tithi","nakshatra","yoga","karana"]
      - include_timeline: bool (default True)
      - step_minutes: "auto" | number (scan step; refinement is always sub-second)
      - frame: ephemeris frame (default "ecliptic-of-date")
    Output:
      {
        ok, window_jd_tt:[start,end], ayanamsa, changes:{...}, timeline:{...}, meta:{...}
      }
    """
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "changes": {}, "timeline": {}, "meta": {}}

    ay_key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
    elements = payload.get("elements") or ["tithi","nakshatra","yoga","karana"]
    frame = str(payload.get("frame") or "ecliptic-of-date")
    include_tl = bool(payload.get("include_timeline", True))
    step_minutes = payload.get("step_minutes", "auto")

    jd_start_tt = payload.get("jd_start_tt")
    jd_end_tt = payload.get("jd_end_tt")
    warnings: List[str] = []

    if not (isinstance(jd_start_tt, (int, float)) and isinstance(jd_end_tt, (int, float))):
        # build from civil date (+days)
        d = str(payload.get("date"))
        tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
        days = int(payload.get("days", 1))
        ts0 = _timescales_from_civil(d, "00:00:00", tz)
        jd_start_tt = ts0.jd_tt
        warnings.extend(ts0.warnings)
        # end at next midnight of (date + days)
        y, m, dd = [int(x) for x in d.split("-")]
        import datetime as _dt
        d_end = str((_dt.date(y, m, dd) + _dt.timedelta(days=max(1, days))))
        ts1 = _timescales_from_civil(d_end, "00:00:00", tz)
        jd_end_tt = ts1.jd_tt
        warnings.extend(ts1.warnings)

    start = float(jd_start_tt); end = float(jd_end_tt)
    if end <= start:
        return {"ok": True, "changes": {k: [] for k in ("tithi","nakshatra","yoga","karana")}, "timeline": {}, "window_jd_tt": [start, end], "meta": {"ayanamsa": ay_key}, "warnings": warnings}

    changes = find_panchanga_changes(start, end, ayanamsa_key=ay_key, step_minutes=step_minutes, elements=elements, frame=frame)
    timeline = panchanga_timeline(start, end, ayanamsa_key=ay_key, elements=elements, frame=frame, step_minutes=step_minutes) if include_tl else {}

    return {
        "ok": True,
        "window_jd_tt": [start, end],
        "ayanamsa": ay_key,
        "changes": changes,
        "timeline": timeline,
        "meta": {"frame": frame},
        "warnings": warnings,
    }
