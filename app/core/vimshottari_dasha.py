# app/core/vimshottari_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Vimśottarī Daśā — Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa

Features
- Full 5-tier sub-period structure with exact parent→child proportional timing.
- Sidereal-first: Moon’s nirāyaṇa longitude at birth sets the starting Mahādaśā
  (nakṣatra lord) and the running balance (nakṣatra remainder).
- Deterministic boundary policy: half-open intervals [start, end); last child in
  each tier snaps to the parent’s end to avoid floating drift.
- Windowed generation: build from a requested time window (e.g., from birth for
  N years, or up to a given end JD).
- Utilities:
    • current_dasha_at(birth, query) → lords at all 5 levels
    • flatten(periods, level=n) → flat timeline lists for UI

Conventions
- Year length for Vimśottarī is configurable (default 365.25 days) via parameter
  or environment variable VIM_YEAR_DAYS.
- Order & years (canonical Parāśara/KP):
    Ketu 7, Venus 20, Sun 6, Moon 10, Mars 7, Rahu 18, Jupiter 16, Saturn 19, Mercury 17.
- Antardaśā sequence inside a Mahādaśā starts with the Mahādaśā lord, then follows
  the standard 9-lord cycle; child durations are proportional to parent × (years/120).

Inputs
- You can pass a birth JD_TT directly, or civil date/time/tz (uses timescales).
- Ayanāṁśa: pass a float (deg) or a key (e.g., "lahiri"). If a key is provided,
  we call app.core.ayanamsa.get_ayanamsa_deg(jd_tt, key).
- Ephemeris: app.core.ephemeris_adapter.EphemerisAdapter (ecliptic-of-date).

Outputs
- Period dicts have: {'level':1..5,'lord':"Ketu"..,"start_jd_tt","end_jd_tt","children":[...]}
- Flatteners return rows with 'path' = [maha,antar,...] and clipped [start,end).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Literal
import os
import math

# ── Optional time helpers (same pattern as panchanga.py) ──
try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

# ── Ephemeris & ayanāṁśa ──
from app.core.ephem_singleton import TS, PLANETS  # singleton config
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    _EPH_OK = False

try:
    from app.core.ayanamsa import get_ayanamsa_deg as _ayan
except Exception:
    _ayan = None

# ─────────────────────────────────────────────────────────────────────────────
# Constants & helpers
# ─────────────────────────────────────────────────────────────────────────────
VIM_ORDER: Tuple[str, ...] = ("Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury")
VIM_YEARS: Dict[str, int] = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}
_NAK_WIDTH = 360.0 / 27.0  # 13°20'

# Default year length (days) for Vimśottarī arithmetic; configurable via env
VIM_YEAR_DAYS = float(os.getenv("VIM_YEAR_DAYS", "365.25"))

_EPS = 1e-12

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

def _rotate_cycle(start_lord: str) -> List[str]:
    s = start_lord.strip().title()
    if s not in VIM_ORDER:
        raise ValueError(f"unknown dasha lord: {start_lord}")
    i = VIM_ORDER.index(s)
    return list(VIM_ORDER[i:] + VIM_ORDER[:i])

# Optional constants_vedic override for nakṣatra lords (27-long list)
try:
    from app.core.constants_vedic import NAKSHATRA_LORDS_27 as _NAK_LORDS  # type: ignore
    _NAK_LORDS = [str(x).title() for x in _NAK_LORDS]  # normalize
    if len(_NAK_LORDS) != 27:
        _NAK_LORDS = []
except Exception:
    _NAK_LORDS = []

# ─────────────────────────────────────────────────────────────────────────────
# Timescales
# ─────────────────────────────────────────────────────────────────────────────
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
                    ju, jt, j1 = map(float, out[:3])
                    return ju, jt, j1
    if _ts is None:
        raise ValueError("timescales module not available")
    jd_ut = float(_ts.julian_day_utc(date, time, tz))
    # monthly ΔT if available
    try:
        y, m = map(int, date.split("-")[:2])
        jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m))
    except Exception:
        jd_tt = jd_ut + 69.0/86400.0
    return jd_ut, jd_tt, jd_ut  # no UT1 drift by default

# ─────────────────────────────────────────────────────────────────────────────
# Moon nirāyaṇa longitude & nakṣatra / balance
# ─────────────────────────────────────────────────────────────────────────────
def _moon_longitude_tropical(jd_tt: float) -> float:
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")
    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore
    rows = ep.ecliptic_longitudes(float(jd_tt), ["Moon"]).get("results", [])
    if not rows:
        raise RuntimeError("ephemeris returned no Moon longitude")
    return float(rows[0]["longitude"])

def _ayanamsa_deg(jd_tt: float, ayanamsa: Optional[Any]) -> float:
    if isinstance(ayanamsa, (int, float)):
        return float(ayanamsa)
    if isinstance(ayanamsa, str) and _ayan is not None:
        try:
            return float(_ayan(jd_tt, ayanamsa.strip().lower()))
        except Exception:
            return 0.0
    return 0.0

def _nak_lord_from_index(idx0: int) -> str:
    if _NAK_LORDS:
        return _NAK_LORDS[idx0]
    # Cycle by 9, starting Ashwini → Ketu
    return VIM_ORDER[idx0 % 9]

@dataclass
class NakshatraInfo:
    index: int              # 1..27
    lord: str               # "Ketu".. "Mercury"
    start_deg: float        # start of nak in nirayana (deg)
    offset_deg: float       # position within nak (deg, 0..13°20')
    fraction_left: float    # 0..1
    moon_nirayana_deg: float

def nakshatra_info_from_jd(jd_tt: float, *, ayanamsa: Optional[Any] = "lahiri") -> NakshatraInfo:
    moon_trop = _moon_longitude_tropical(jd_tt)
    ay = _ayanamsa_deg(jd_tt, ayanamsa)
    moon_nira = _norm360(moon_trop - ay)
    idx0 = int(math.floor(moon_nira / _NAK_WIDTH))  # 0..26
    start = idx0 * _NAK_WIDTH
    offset = moon_nira - start
    frac_left = max(0.0, min(1.0, ( _NAK_WIDTH - offset ) / _NAK_WIDTH))
    lord = _nak_lord_from_index(idx0)
    return NakshatraInfo(
        index=idx0 + 1, lord=lord, start_deg=start, offset_deg=offset,
        fraction_left=frac_left, moon_nirayana_deg=moon_nira
    )

# ─────────────────────────────────────────────────────────────────────────────
# Core partitioning
# ─────────────────────────────────────────────────────────────────────────────
def _partition_by_weights(start: float, end: float, order: List[str]) -> List[Tuple[str, float, float]]:
    """
    Partition [start,end) into 9 children according to Vim years / 120.
    The last child snaps to 'end' to ensure exact coverage.
    """
    span = float(end - start)
    weights = [VIM_YEARS[l] / 120.0 for l in order]
    # raw durations
    durs = [span * w for w in weights]
    # adjust last
    tail = span - _kahan_sum(durs[:-1])
    durs[-1] = tail
    out = []
    t = start
    for lord, d in zip(order, durs):
        a = t
        b = t + d
        out.append((lord, a, b))
        t = b
    # numerical guard
    out[-1] = (out[-1][0], out[-1][1], end)
    return out

def _build_children(parent_lord: str, start: float, end: float, level: int, max_level: int) -> List[Dict[str, Any]]:
    """
    Recursively build 9-way subperiods down to max_level (<=5).
    """
    if level >= max_level:
        return []
    order = _rotate_cycle(parent_lord)
    parts = _partition_by_weights(start, end, order)
    children: List[Dict[str, Any]] = []
    for lord, a, b in parts:
        node = {"level": level + 1, "lord": lord, "start_jd_tt": a, "end_jd_tt": b, "children": []}
        node["children"] = _build_children(lord, a, b, level + 1, max_level)
        children.append(node)
    return children

# ─────────────────────────────────────────────────────────────────────────────
# Top-level generation (Mahādaśā stream with clipping)
# ─────────────────────────────────────────────────────────────────────────────
def _maha_stream(start_maha_lord: str, maha0_start: float, jd_end: float, *, year_days: float) -> List[Tuple[str, float, float]]:
    """
    Generate consecutive Mahādaśās from maha0_start (which may begin before birth),
    until jd_end (exclusive). Returns list of (lord, start, end) covering the span.
    """
    seq = []
    t = float(maha0_start)
    lord_idx = VIM_ORDER.index(start_maha_lord)
    # Generate up to two full cycles worst-case
    cap = 9 * 2 + 2
    for _ in range(cap):
        lord = VIM_ORDER[(lord_idx) % 9]
        dur_days = VIM_YEARS[lord] * year_days
        a = t
        b = a + dur_days
        seq.append((lord, a, b))
        t = b
        lord_idx += 1
        if a > jd_end + 365.0:  # guard break
            break
    return seq

def _clip_interval(a: float, b: float, w0: float, w1: float) -> Optional[Tuple[float, float]]:
    aa = max(a, w0)
    bb = min(b, w1)
    if bb <= aa + _EPS:
        return None
    return (aa, bb)

def _clip_tree(node: Dict[str, Any], w0: float, w1: float) -> Optional[Dict[str, Any]]:
    c = _clip_interval(node["start_jd_tt"], node["end_jd_tt"], w0, w1)
    if c is None:
        return None
    a, b = c
    out = dict(node)
    out["start_jd_tt"] = a
    out["end_jd_tt"] = b
    out["children"] = []
    for ch in node.get("children", []):
        cc = _clip_tree(ch, a, b)  # child window within clipped parent
        if cc is not None:
            out["children"].append(cc)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def generate_vimshottari_tree(
    *,
    birth_jd_tt: float | None = None,
    date: str | None = None,
    time: str | None = None,
    tz: str | None = None,
    ayanamsa: Optional[Any] = "lahiri",
    levels: int = 5,
    span_years: float | None = 120.0,
    end_jd_tt: float | None = None,
    year_days: float | None = None,
) -> Dict[str, Any]:
    """
    Build Vimśottarī daśā tree from birth onward, clipped to the requested window.

    Window selection:
      - If end_jd_tt is provided, window = [birth_jd_tt, end_jd_tt)
      - Else if span_years is provided, window = [birth_jd_tt, birth_jd_tt + span_years*year_days)
      - Else default span_years=120.

    Returns:
      {
        "ok": True,
        "birth_jd_tt": ...,
        "moon_nirayana_deg": ...,
        "nakshatra": {"index":..,"lord":..,"fraction_left":..},
        "year_days": ...,
        "levels": 5,
        "periods": [ {Mahā node with nested children (to 'levels')} ... ]
      }
    """
    if levels < 1 or levels > 5:
        raise ValueError("levels must be between 1 and 5")

    # Resolve birth time
    if birth_jd_tt is None:
        if not (isinstance(date, str) and isinstance(time, str) and isinstance(tz, str)):
            raise ValueError("Provide birth_jd_tt or (date, time, tz)")
        _, birth_jd_tt, _ = _timescales_from_civil(date, time, tz)

    jd0 = float(birth_jd_tt)
    yd = float(year_days if isinstance(year_days, (int, float)) else VIM_YEAR_DAYS)

    # Moon & nakṣatra at birth
    info = nakshatra_info_from_jd(jd0, ayanamsa=ayanamsa)
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

    # Assemble tree with clipping to [jd0, jd1)
    out_periods: List[Dict[str, Any]] = []
    for lord, a, b in maha_seq:
        # skip if no overlap
        clipped = _clip_interval(a, b, jd0, jd1)
        if clipped is None:
            continue
        # build full nested tree for the COMPLETE parent, then clip tree to window
        root = {"level": 1, "lord": lord, "start_jd_tt": a, "end_jd_tt": b, "children": []}
        root["children"] = _build_children(lord, a, b, level=1, max_level=levels)
        clipped_root = _clip_tree(root, jd0, jd1)
        if clipped_root:
            out_periods.append(clipped_root)

    return {
        "ok": True,
        "birth_jd_tt": jd0,
        "moon_nirayana_deg": info.moon_nirayana_deg,
        "nakshatra": {
            "index": info.index, "lord": info.lord, "fraction_left": info.fraction_left,
            "offset_deg": info.offset_deg, "width_deg": _NAK_WIDTH
        },
        "year_days": yd,
        "levels": int(levels),
        "periods": out_periods,
    }

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

    If a prebuilt 'tree' (from generate_vimshottari_tree) is not provided, we’ll
    build a minimal window spanning the needed time.
    """
    if query_jd_tt is None:
        if not (isinstance(date, str) and isinstance(time, str) and isinstance(tz, str)):
            raise ValueError("Provide query_jd_tt or (date,time,tz)")
        _, query_jd_tt, _ = _timescales_from_civil(date, time, tz)
    t = float(query_jd_tt)

    if tree is None:
        if birth_jd_tt is None:
            raise ValueError("Provide tree or birth_jd_tt")
        # minimal window around query: ±1 day; we’ll extend if it lands outside
        yd = float(year_days if isinstance(year_days, (int, float)) else VIM_YEAR_DAYS)
        # Build 130 years to be safe
        tree = generate_vimshottari_tree(birth_jd_tt=float(birth_jd_tt), ayanamsa=ayanamsa, levels=int(levels),
                                         span_years=130.0, year_days=yd)

    out: Dict[str, Any] = {"ok": False, "levels": int(levels)}
    path: List[str] = []

    def find_level(nodes: List[Dict[str, Any]], when: float, depth: int) -> Optional[Dict[str, Any]]:
        for n in nodes:
            if n["start_jd_tt"] - _EPS <= when < n["end_jd_tt"] - _EPS:
                return n
        return None

    periods = tree.get("periods", []) if isinstance(tree, dict) else []
    n1 = find_level(periods, t, 1)
    if not n1:
        return {"ok": False, "error": "query_outside_window"}
    path.append(n1["lord"])
    n = n1
    for _lvl in range(2, int(levels) + 1):
        kids = n.get("children", [])
        n2 = find_level(kids, t, _lvl)
        if not n2:
            break
        path.append(n2["lord"])
        n = n2

    out.update({"ok": True, "path": path, "start_jd_tt": n["start_jd_tt"], "end_jd_tt": n["end_jd_tt"]})
    # Also expose each level explicitly if present
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

    Payload keys (any subset):
        birth_jd_tt | (date,time,tz)
        ayanamsa: float|str     default "lahiri"
        levels: 1..5            default 5
        span_years: float       default 120
        end_jd_tt: float        optional
        year_days: float        default env VIM_YEAR_DAYS or 365.25
        query_jd_tt | (q_date,q_time,q_tz)   optional → return 'current' in response
        flatten_level: 1..5     optional → return flattened list too
    """
    birth_jd_tt = payload.get("birth_jd_tt")
    date = payload.get("date"); time = payload.get("time"); tz = payload.get("tz")
    ay = payload.get("ayanamsa", "lahiri")
    levels = int(payload.get("levels", 5))
    span_years = payload.get("span_years", 120.0)
    end_jd_tt = payload.get("end_jd_tt")
    year_days = payload.get("year_days")

    tree = generate_vimshottari_tree(
        birth_jd_tt=birth_jd_tt,
        date=date, time=time, tz=tz,
        ayanamsa=ay, levels=levels, span_years=span_years, end_jd_tt=end_jd_tt, year_days=year_days
    )
    out: Dict[str, Any] = {"ok": True, "tree": tree}

    # Current at query (optional)
    q_jd = payload.get("query_jd_tt")
    if q_jd is None and all(k in payload for k in ("q_date","q_time","q_tz")):
        _, q_jd, _ = _timescales_from_civil(payload["q_date"], payload["q_time"], payload["q_tz"])
    if q_jd is not None:
        out["current"] = current_dasha_at(
            tree=tree, query_jd_tt=float(q_jd), levels=levels, ayanamsa=ay, year_days=year_days
        )

    # Flatten (optional)
    flat_level = payload.get("flatten_level")
    if isinstance(flat_level, int) and 1 <= flat_level <= levels:
        out["flat"] = flatten_periods(tree.get("periods", []), level=int(flat_level))

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Self-checks
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick numeric sanity (no I/O): synthetic birth at J2000 TT
    jd0 = 2451545.0
    t = generate_vimshottari_tree(birth_jd_tt=jd0, ayanamsa="lahiri", levels=5, span_years=120.0)
    assert t["ok"] and t["periods"], "Tree generation failed"
    # Flatten one level and check coverage continuity
    L1 = flatten_periods(t["periods"], level=1)
    for i in range(1, len(L1)):
        assert abs(L1[i-1]["end_jd_tt"] - L1[i]["start_jd_tt"]) < 1e-6
    print("Vimśottarī core probes OK.")
