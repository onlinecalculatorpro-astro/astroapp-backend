# app/core/vimshottari_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Vimśottarī Daśā — Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa

Enhancements:
- method: "sidereal" (default) or "tropical"
- observer: "geocentric" (default) or "topocentric" (needs lat/lon)
- ayanamsa: "lahiri" (default) or any supported key (fagan_bradley, krishnamurti, raman, yukteswar, devore, ...)
- Boundary distance helper for near-nakṣatra-edge warnings
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
import os
import math
from functools import lru_cache

# ── Optional time helpers ─────────────────────────────────────────────────────
try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

# ── Ephemeris & ayanāṁśa ─────────────────────────────────────────────────────
from app.core.ephem_singleton import TS, PLANETS  # singleton config
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    _EPH_OK = False

try:
    # Your module should support multiple ayanamsa keys; default is lahiri.
    from app.core.ayanamsa import get_ayanamsa_deg as _ayan
except Exception:
    _ayan = None

# Optional override table for nakṣatra lords (27 entries)
try:
    from app.core.constants_vedic import NAKSHATRA_LORDS_27 as _NAK_LORDS  # type: ignore
    _NAK_LORDS = [str(x).title() for x in _NAK_LORDS]
    if len(_NAK_LORDS) != 27:
        _NAK_LORDS = []
except Exception:
    _NAK_LORDS = []

# ── Constants ─────────────────────────────────────────────────────────────────
VIM_ORDER: Tuple[str, ...] = ("Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury")
VIM_YEARS: Dict[str, int] = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
_RATIOS: Dict[str, float] = {k: VIM_YEARS[k] / 120.0 for k in VIM_ORDER}
_NAK_WIDTH = 360.0 / 27.0  # 13°20′

# Default year length (days) for Vimśottarī arithmetic; configurable via env
VIM_YEAR_DAYS = float(os.getenv("VIM_YEAR_DAYS", "365.25"))

_EPS = 1e-12

# ── Small helpers ─────────────────────────────────────────────────────────────
def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _kahan_sum(vals: Iterable[float]) -> float:
    s = 0.0; c = 0.0
    for v in vals:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

def _norm_method(v: Any, default: str = "sidereal") -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("sidereal","nirayana","nirāyaṇa","sid","s"): return "sidereal"
        if s in ("tropical","sayana","sāyana","trop","t"):    return "tropical"
    return default

def _norm_observer(v: Any, default: str = "geocentric") -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("geocentric","geo","center"):    return "geocentric"
        if s in ("topocentric","apparent","obs"): return "topocentric"
    return default

def _norm_coord_mode(topocentric_flag: Any, mode_or_observer: Any) -> str:
    # Accept either booleans (legacy topocentric) or a string observer/coordinate_mode
    if isinstance(topocentric_flag, bool):
        return "topocentric" if topocentric_flag else "geocentric"
    if isinstance(mode_or_observer, str):
        return _norm_observer(mode_or_observer, default="geocentric")
    return "geocentric"

def _norm_ayanamsa(v: Any) -> str:
    # Default LAHIRI as requested
    if v is None: return "lahiri"
    s = str(v).strip().lower()
    return s or "lahiri"

@lru_cache(maxsize=16)
def _cycle(start_lord: str) -> Tuple[str, ...]:
    s = start_lord.strip().title()
    if s not in VIM_ORDER:
        raise ValueError(f"unknown dasha lord: {start_lord}")
    i = VIM_ORDER.index(s)
    return tuple(VIM_ORDER[i:] + VIM_ORDER[:i])

@lru_cache(maxsize=16)
def _cycle_ratios(order: Tuple[str, ...]) -> Tuple[float, ...]:
    return tuple(_RATIOS[l] for l in order)

def _nak_lord_from_index(idx0: int) -> str:
    if _NAK_LORDS:
        return _NAK_LORDS[idx0]
    return VIM_ORDER[idx0 % 9]  # Ashwini → Ketu

# ── Timescales ────────────────────────────────────────────────────────────────
def _timescales_from_civil(date: str, time: str, tz: str) -> Tuple[float, float, float]:
    """
    Returns (jd_ut, jd_tt, jd_ut1). We accept any available backend.
    """
    if _tk is not None:
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(date=date, time=time, tz=tz, dut1=0.0)
                except TypeError:
                    out = fn(date, time, tz, 0.0)
                if isinstance(out, dict):
                    jd_ut = float(out.get("jd_ut") or out.get("jd_utc"))
                    return jd_ut, float(out["jd_tt"]), float(out.get("jd_ut1", jd_ut))
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    ju, jt, j1 = map(float, out[:3]); return ju, jt, j1
    if _ts is None:
        raise ValueError("timescales module not available")
    jd_ut = float(_ts.julian_day_utc(date, time, tz))
    try:
        y, m = map(int, date.split("-")[:2])
        jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m))
    except Exception:
        jd_tt = jd_ut + 69.0/86400.0
    return jd_ut, jd_tt, jd_ut

# ── Ephemeris compatibility + cache ───────────────────────────────────────────
_EPH_ADAPTER = None  # cached instance

def _make_ephem(frame: str = "ecliptic-of-date"):
    """
    Build EphemerisAdapter regardless of whether 'timescale'/'planets'
    are accepted by EphemConfig/EphemerisAdapter in this deployment.
    Tries several signatures gracefully.
    """
    try:
        cfg = EphemConfig(frame=frame, timescale=TS, planets=PLANETS)  # type: ignore[arg-type]
        try:    return EphemerisAdapter(cfg)  # type: ignore[call-arg]
        except TypeError: pass
    except TypeError:
        cfg = None  # type: ignore[assignment]

    if cfg is None:
        try:    cfg = EphemConfig(frame=frame)  # type: ignore[call-arg]
        except TypeError:
            cfg = None  # type: ignore[assignment]

    if cfg is not None:
        try:    return EphemerisAdapter(cfg, timescale=TS, planets=PLANETS)  # type: ignore[call-arg]
        except TypeError:
            try:    return EphemerisAdapter(cfg)  # type: ignore[call-arg]
            except TypeError: pass

    try:        return EphemerisAdapter(frame=frame, timescale=TS, planets=PLANETS)  # type: ignore[call-arg]
    except TypeError:
        try:    return EphemerisAdapter(frame=frame)  # type: ignore[call-arg]
        except TypeError as e:
            raise RuntimeError(f"EphemerisAdapter incompatible signatures: {e}")

def _get_ephem():
    global _EPH_ADAPTER
    if _EPH_ADAPTER is None:
        _EPH_ADAPTER = _make_ephem("ecliptic-of-date")
    return _EPH_ADAPTER

# ── Moon longitude (tropical), with geo/topo option ──────────────────────────
def _moon_longitude_tropical(
    jd_tt: float,
    *,
    topocentric: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
    warnings: Optional[List[str]] = None,
) -> float:
    """
    Get Moon ecliptic longitude (tropical). Geocentric by default.
    If topocentric=True and lat/lon provided, attempts an observer-based call.
    Falls back to geocentric with a warning if the adapter lacks support.
    """
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")

    ep = _get_ephem()

    if topocentric and latitude is not None and longitude is not None:
        rows = []
        try:
            rows = ep.ecliptic_longitudes(
                float(jd_tt), ["Moon"],
                observer={
                    "lat": float(latitude), "lon": float(longitude),
                    "elevation_m": float(elevation_m or 0.0), "topocentric": True
                }
            ).get("results", [])
        except TypeError:
            try:
                rows = ep.ecliptic_longitudes(
                    float(jd_tt), ["Moon"],
                    lat=float(latitude), lon=float(longitude),
                    elevation_m=float(elevation_m or 0.0), topocentric=True
                ).get("results", [])
            except TypeError:
                try:
                    # Some adapters expose an "apparent_*" API
                    rows = ep.apparent_ecliptic_longitudes(  # type: ignore[attr-defined]
                        float(jd_tt), ["Moon"],
                        lat=float(latitude), lon=float(longitude),
                        elevation_m=float(elevation_m or 0.0)
                    ).get("results", [])
                except Exception:
                    rows = []
        if rows:
            return float(rows[0]["longitude"])
        else:
            if warnings is not None:
                warnings.append("topocentric_unavailable:fallback_to_geocentric")

    rows = ep.ecliptic_longitudes(float(jd_tt), ["Moon"]).get("results", [])
    if not rows:
        raise RuntimeError("ephemeris returned no Moon longitude")
    return float(rows[0]["longitude"])

def _ayanamsa_deg(jd_tt: float, ayanamsa_key: Optional[Any]) -> float:
    # Accept numeric ayanamsa or string key (lahiri default via _norm_ayanamsa)
    if isinstance(ayanamsa_key, (int, float)):
        return float(ayanamsa_key)
    key = _norm_ayanamsa(ayanamsa_key)
    if _ayan is not None:
        try:
            return float(_ayan(jd_tt, key))
        except Exception:
            return 0.0
    return 0.0

# ── Nakṣatra computation on chosen basis ─────────────────────────────────────
@dataclass
class NakshatraInfo:
    index: int              # 1..27
    lord: str               # "Ketu".. "Mercury"
    start_deg: float        # start of segment (sidereal or tropical)
    offset_deg: float       # position within current 13°20′ segment
    fraction_left: float    # 0..1
    moon_basis_deg: float   # longitude used for segmentation (sidereal or tropical)
    basis: str              # "sidereal" or "tropical"

def nakshatra_info_from_jd(
    jd_tt: float,
    *,
    method: str = "sidereal",         # "sidereal" (default) or "tropical"
    ayanamsa: Optional[Any] = "lahiri",
    topocentric: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
    warnings: Optional[List[str]] = None,
) -> NakshatraInfo:
    """
    Compute nakṣatra from Moon longitude:
      - method="sidereal": segment on nirāyaṇa Moon (tropical - ayanāṁśa)
      - method="tropical": segment directly on tropical Moon
    """
    method = _norm_method(method, default="sidereal")

    moon_trop = _moon_longitude_tropical(
        jd_tt,
        topocentric=topocentric,
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
        warnings=warnings,
    )
    if method == "sidereal":
        ay = _ayanamsa_deg(jd_tt, ayanamsa)
        base = _norm360(moon_trop - ay)
    else:
        base = _norm360(moon_trop)

    idx0 = int(math.floor(base / _NAK_WIDTH))  # 0..26
    start = idx0 * _NAK_WIDTH
    offset = base - start
    frac_left = max(0.0, min(1.0, (_NAK_WIDTH - offset) / _NAK_WIDTH))
    lord = _nak_lord_from_index(idx0)
    return NakshatraInfo(
        index=idx0 + 1, lord=lord, start_deg=start, offset_deg=offset,
        fraction_left=frac_left, moon_basis_deg=base, basis=method
    )

# ── Partitioning and tree build ───────────────────────────────────────────────
def _partition_by_weights(start: float, end: float, order: Tuple[str, ...]) -> List[Tuple[str, float, float]]:
    span = float(end - start)
    ratios = _cycle_ratios(order)
    durs = [span * r for r in ratios]
    tail = span - _kahan_sum(durs[:-1])
    durs[-1] = tail
    out = []
    t = start
    for lord, d in zip(order, durs):
        a = t; b = t + d
        out.append((lord, a, b))
        t = b
    out[-1] = (out[-1][0], out[-1][1], end)
    return out

def _build_children_windowed(parent_lord: str, a: float, b: float, level: int, max_level: int, w0: float, w1: float) -> List[Dict[str, Any]]:
    if level >= max_level:
        return []
    order = _cycle(parent_lord)
    parts = _partition_by_weights(a, b, order)
    out: List[Dict[str, Any]] = []
    for lord, aa, bb in parts:
        c = _clip_interval(aa, bb, w0, w1)
        if c is None:
            continue
        ca, cb = c
        node = {"level": level + 1, "lord": lord, "start_jd_tt": ca, "end_jd_tt": cb, "children": []}
        node["children"] = _build_children_windowed(lord, aa, bb, level + 1, max_level, ca, cb)
        out.append(node)
    return out

def _maha_stream(start_maha_lord: str, maha0_start: float, jd_end: float, *, year_days: float) -> List[Tuple[str, float, float]]:
    seq = []
    t = float(maha0_start)
    lord_idx = VIM_ORDER.index(start_maha_lord)
    cap = 9 * 2 + 2  # two cycles worst-case
    for _ in range(cap):
        lord = VIM_ORDER[(lord_idx) % 9]
        dur_days = VIM_YEARS[lord] * year_days
        a = t; b = a + dur_days
        seq.append((lord, a, b))
        t = b; lord_idx += 1
        if a > jd_end + 365.0:
            break
    return seq

def _clip_interval(a: float, b: float, w0: float, w1: float) -> Optional[Tuple[float, float]]:
    aa = max(a, w0); bb = min(b, w1)
    if bb <= aa + _EPS:
        return None
    return (aa, bb)

# ── Public API ────────────────────────────────────────────────────────────────
def generate_vimshottari_tree(
    *,
    birth_jd_tt: float | None = None,
    date: str | None = None,
    time: str | None = None,
    tz: str | None = None,
    ayanamsa: Optional[Any] = "lahiri",
    method: str = "sidereal",           # NEW
    levels: int = 5,
    span_years: float | None = 120.0,
    end_jd_tt: float | None = None,
    year_days: float | None = None,
    # Place + toggle for topocentric
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float | None = None,
    topocentric: bool = False,
    coordinate_mode: Optional[str] = None,  # alias for observer/coordinate mode
) -> Dict[str, Any]:
    """
    Build Vimśottarī daśā tree from birth onward, clipped to the requested window.

    Window selection:
      - If end_jd_tt is provided, window = [birth_jd_tt, end_jd_tt)
      - Else if span_years is provided, window = [birth_jd_tt, birth_jd_tt + span_years*year_days)
      - Else default span_years=120.

    New behavior:
      - method: "sidereal" (default) or "tropical"
      - coordinate_mode/topocentric: "geocentric" (default) or "topocentric" (needs lat/lon)
    """
    if levels < 1 or levels > 5:
        raise ValueError("levels must be between 1 and 5")

    warns: List[str] = []

    method = _norm_method(method, default="sidereal")
    ay_key = _norm_ayanamsa(ayanamsa)
    coord_mode = _norm_coord_mode(topocentric, coordinate_mode)
    topo = (coord_mode == "topocentric")

    # Resolve birth time
    if birth_jd_tt is None:
        if not (isinstance(date, str) and isinstance(time, str) and isinstance(tz, str)):
            raise ValueError("Provide birth_jd_tt or (date, time, tz)")
        _, birth_jd_tt, _ = _timescales_from_civil(date, time, tz)

    jd0 = float(birth_jd_tt)
    yd = float(year_days if isinstance(year_days, (int, float)) else VIM_YEAR_DAYS)

    # Moon & nakṣatra at birth (basis + geo/topo)
    info = nakshatra_info_from_jd(
        jd_tt=jd0,
        method=method,
        ayanamsa=ay_key,
        topocentric=topo,
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
        warnings=warns,
    )
    maha_lord0 = info.lord
    full_maha_days = VIM_YEARS[maha_lord0] * yd
    balance_days = info.fraction_left * full_maha_days
    elapsed_days = full_maha_days - balance_days
    maha0_start = jd0 - elapsed_days

    # Window end
    if isinstance(end_jd_tt, (int, float)):
        jd1 = float(end_jd_tt)
    else:
        sy = float(span_years if isinstance(span_years, (int, float)) else 120.0)
        jd1 = jd0 + sy * yd

    # Build Mahādaśā stream (may start before birth)
    maha_seq = _maha_stream(maha_lord0, maha0_start, jd1, year_days=yd)

    # Assemble tree with clipping to [jd0, jd1) and window-aware children
    out_periods: List[Dict[str, Any]] = []
    for lord, a, b in maha_seq:
        clipped = _clip_interval(a, b, jd0, jd1)
        if clipped is None:
            continue
        ca, cb = clipped
        root = {"level": 1, "lord": lord, "start_jd_tt": ca, "end_jd_tt": cb, "children": []}
        root["children"] = _build_children_windowed(lord, a, b, level=1, max_level=levels, w0=ca, w1=cb)
        out_periods.append(root)

    # Distance to nearest nakṣatra boundary (deg)
    boundary_dist = min(info.offset_deg, _NAK_WIDTH - info.offset_deg)

    out: Dict[str, Any] = {
        "ok": True,
        "birth_jd_tt": jd0,
        "moon_basis_deg": info.moon_basis_deg,  # longitude used for segmentation
        "nakshatra": {
            "index": info.index,
            "lord": info.lord,
            "fraction_left": info.fraction_left,
            "offset_deg": info.offset_deg,
            "width_deg": _NAK_WIDTH,
            "boundary_distance_deg": boundary_dist,
            "basis": info.basis,  # "sidereal" | "tropical"
        },
        "year_days": yd,
        "levels": int(levels),
        "periods": out_periods,
        "meta": {
            "method": method,
            "coordinate_mode": coord_mode,       # "geocentric" | "topocentric"
            "ayanamsa_key": ay_key,              # e.g., "lahiri" (default)
            "observer": {
                "latitude": latitude, "longitude": longitude, "elevation_m": elevation_m
            },
        },
    }
    if warns:
        out["warnings"] = warns
    return out

def current_dasha_at(
    *,
    tree: Dict[str, Any] | None = None,
    birth_jd_tt: float | None = None,
    query_jd_tt: float | None = None,
    date: str | None = None,
    time: str | None = None,
    tz: str | None = None,
    ayanamsa: Optional[Any] = "lahiri",
    levels: int = 5,
    year_days: float | None = None,
) -> Dict[str, Any]:
    """
    Return current (Mahā..Prāṇa) at the query moment.

    If a prebuilt 'tree' is not provided, a 130y window is built around birth.
    """
    if query_jd_tt is None:
        if not (isinstance(date, str) and isinstance(time, str) and isinstance(tz, str)):
            raise ValueError("Provide query_jd_tt or (date,time,tz)")
        _, query_jd_tt, _ = _timescales_from_civil(date, time, tz)
    t = float(query_jd_tt)

    if tree is None:
        if birth_jd_tt is None:
            raise ValueError("Provide tree or birth_jd_tt")
        yd = float(year_days if isinstance(year_days, (int, float)) else VIM_YEAR_DAYS)
        tree = generate_vimshottari_tree(
            birth_jd_tt=float(birth_jd_tt),
            ayanamsa=_norm_ayanamsa(ayanamsa),
            levels=int(levels),
            span_years=130.0,
            year_days=yd
        )

    out: Dict[str, Any] = {"ok": False, "levels": int(levels)}
    path: List[str] = []

    def find_level(nodes: List[Dict[str, Any]], when: float) -> Optional[Dict[str, Any]]:
        for n in nodes:
            if n["start_jd_tt"] - _EPS <= when < n["end_jd_tt"] - _EPS:
                return n
        return None

    periods = tree.get("periods", []) if isinstance(tree, dict) else []
    n = find_level(periods, t)
    if not n:
        return {"ok": False, "error": "query_outside_window"}
    path.append(n["lord"])
    for _lvl in range(2, int(levels) + 1):
        kids = n.get("children", [])
        n2 = find_level(kids, t)
        if not n2:
            break
        path.append(n2["lord"])
        n = n2

    out.update({"ok": True, "path": path, "start_jd_tt": n["start_jd_tt"], "end_jd_tt": n["end_jd_tt"]})
    keys = ["maha","antar","pratyantar","sukshma","prana"]
    for i, lord in enumerate(path):
        out[keys[i]] = lord
    return out

def flatten_periods(periods: List[Dict[str, Any]], *, level: int) -> List[Dict[str, Any]]:
    """
    Flatten to a list at a chosen depth (1..5). Each row includes the full path.
    """
    if level < 1 or level > 5:
        raise ValueError("level must be 1..5")
    rows: List[Dict[str, Any]] = []

    def walk(node: Dict[str, Any], path: List[str]) -> None:
        p2 = path + [node["lord"]]
        if node["level"] == level:
            rows.append({
                "level": level,
                "path": p2,
                "start_jd_tt": node["start_jd_tt"],
                "end_jd_tt": node["end_jd_tt"],
                "lord": node["lord"],
            })
        for ch in node.get("children", []):
            walk(ch, p2)

    for m in periods:
        walk(m, [])
    rows.sort(key=lambda r: (r["start_jd_tt"], r["path"]))
    return rows

def compute_vimshottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route-friendly wrapper.

    Accepted payload keys (any subset):
        # Primary civic time
        date, time, tz
        birth_jd_tt                       # alternative to (date,time,tz)

        # Basis & observer
        method: "sidereal"|"tropical"     # default "sidereal"
        observer: "geocentric"|"topocentric"  # default "geocentric"
        topocentric: bool                 # legacy alias
        coordinate_mode: str              # legacy alias
        ayanamsa: string|float            # default "lahiri"

        # Site (needed if topocentric)
        latitude|lat, longitude|lon, elevation_m|elevation

        # Controls
        levels: 1..5                      # default 5
        span_years: float                 # default 120
        end_jd_tt: float                  # optional
        year_days: float                  # default env or 365.25
        query_jd_tt | (q_date,q_time,q_tz)# optional → 'current'
        flatten_level: 1..5               # optional → flat list at chosen level
    """
    birth_jd_tt = payload.get("birth_jd_tt")
    date = payload.get("date"); time = payload.get("time"); tz = payload.get("tz")

    # Basis
    method = _norm_method(payload.get("method", payload.get("mode", "sidereal")), default="sidereal")
    ay = _norm_ayanamsa(payload.get("ayanamsa", payload.get("ayanamsa_key", "lahiri")))

    # Observer
    observer = _norm_observer(payload.get("observer", "geocentric"))
    coord_mode = payload.get("coordinate_mode", observer)
    topo = (_norm_coord_mode(payload.get("topocentric"), coord_mode) == "topocentric")

    # Site
    lat = payload.get("latitude", payload.get("lat"))
    lon = payload.get("longitude", payload.get("lon"))
    elev = payload.get("elevation_m", payload.get("elevation"))

    # Controls
    levels = int(payload.get("levels", 5))
    span_years = payload.get("span_years", 120.0)
    end_jd_tt = payload.get("end_jd_tt")
    year_days = payload.get("year_days")

    # Depth optimization for flatten
    build_levels = levels
    flat_level = payload.get("flatten_level")
    if isinstance(flat_level, int) and 1 <= flat_level <= 5:
        build_levels = max(1, min(5, int(flat_level)))

    tree = generate_vimshottari_tree(
        birth_jd_tt=birth_jd_tt,
        date=date, time=time, tz=tz,
        ayanamsa=ay, method=method, levels=build_levels,
        span_years=span_years, end_jd_tt=end_jd_tt, year_days=year_days,
        latitude=float(lat) if isinstance(lat, (int, float, str)) and str(lat).strip() else None,
        longitude=float(lon) if isinstance(lon, (int, float, str)) and str(lon).strip() else None,
        elevation_m=float(elev) if isinstance(elev, (int, float, str)) and str(elev).strip() else None,
        topocentric=topo,
        coordinate_mode=("topocentric" if topo else "geocentric"),
    )

    out: Dict[str, Any] = {"ok": True, "tree": tree}

    # Bubble up warnings
    if isinstance(tree, dict) and tree.get("warnings"):
        out["warnings"] = list(tree["warnings"])

    # Current at query (optional)
    q_jd = payload.get("query_jd_tt")
    if q_jd is None and all(k in payload for k in ("q_date","q_time","q_tz")):
        _, q_jd, _ = _timescales_from_civil(payload["q_date"], payload["q_time"], payload["q_tz"])
    if q_jd is not None:
        out["current"] = current_dasha_at(
            tree=tree, query_jd_tt=float(q_jd), levels=levels, ayanamsa=ay, year_days=year_days
        )

    # Flatten (optional)
    if isinstance(flat_level, int) and 1 <= flat_level <= 5:
        out["flat"] = flatten_periods(tree.get("periods", []), level=int(flat_level))

    return out

# ── Self-checks ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    jd0 = 2451545.0  # J2000 TT
    t_geo_sid = generate_vimshottari_tree(birth_jd_tt=jd0, method="sidereal", ayanamsa="lahiri", levels=5, span_years=120.0)
    assert t_geo_sid["ok"] and t_geo_sid["periods"], "Tree generation failed (sidereal)"

    t_geo_trop = generate_vimshottari_tree(birth_jd_tt=jd0, method="tropical", levels=5, span_years=120.0)
    assert t_geo_trop["ok"] and t_geo_trop["periods"], "Tree generation failed (tropical)"

    L1 = flatten_periods(t_geo_sid["periods"], level=1)
    for i in range(1, len(L1)):
        assert abs(L1[i-1]["end_jd_tt"] - L1[i]["start_jd_tt"]) < 1e-6
    print("Vimśottarī flexible core probes OK.")
