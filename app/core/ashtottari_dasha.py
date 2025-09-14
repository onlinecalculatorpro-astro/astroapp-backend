# app/core/ashtottari_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Aṣṭottarī Daśā — 108-year cycle with five nested levels
(Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa)

Optimizations:
- Fixed-point time arithmetic in ticks (1 tick = 1e-7 day) → no Decimal in hot path.
- Largest-remainder partition for sub-spans → stable, drift-free, children sum to parent.
- Linear-time tree construction (per-level two-pointer) → avoids O(N^2) filtering.
- Spans are generated in order → no final sorting pass.

Public surface (unchanged):
  compute_dasha(jd_tt_birth, ayanamsa_key="lahiri", ayanamsa_deg=None, depth=None, options=None) -> dict
  compute_ashtottari(payload: dict) -> dict
  ashtottari_schedule(...)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

__all__ = [
    "ASHTOTTARI_ORDER",
    "ASHTOTTARI_YEARS",
    "ASHTOTTARI_TOTAL_YEARS",
    "compute_dasha",
    "compute_ashtottari",
    "ashtottari_schedule",
]

# ───────────────────────── deps (no adapter Config kwargs!) ─────────────────────────
try:
    # Use module-level helpers to avoid wrong Config kwargs
    from app.core.ephemeris_adapter import ecliptic_longitudes  # type: ignore
    _EPH_OK = True
except Exception:
    ecliptic_longitudes = None  # type: ignore
    _EPH_OK = False

try:
    from app.core import time_kernel as _tk  # type: ignore
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts  # type: ignore
except Exception:
    _ts = None

from app.core.ayanamsa import get_ayanamsa_deg
from app.core.constants_vedic import NAKSHATRAS_27

# ───────────────────────── numerics / helpers ─────────────────────────
_JD_QUANT = 1e-7  # ~0.00864 s
def _q(x: float) -> float:
    return round(float(x) / _JD_QUANT) * _JD_QUANT

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _num(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

_NAK_WIDTH = 360.0 / 27.0  # 13°20′

ASHTOTTARI_ORDER: Tuple[str, ...] = (
    "Sun", "Moon", "Mars", "Mercury", "Saturn", "Jupiter", "Rahu", "Venus"
)
ASHTOTTARI_YEARS: Dict[str, int] = {
    "Sun": 6, "Moon": 15, "Mars": 8, "Mercury": 17,
    "Saturn": 10, "Jupiter": 19, "Rahu": 12, "Venus": 21
}
ASHTOTTARI_TOTAL_YEARS = 108

# Kṛttikādi nakṣatra→start-lord (nak index 1..27)
_KRITTIKADI_RANGES = {
    "Sun":     [3, 4, 5],
    "Moon":    [6, 7, 8, 9],
    "Mars":    [10, 11, 12],
    "Mercury": [13, 14, 15, 16],
    "Saturn":  [17, 18, 19],
    "Jupiter": [20, 21, 22],
    "Rahu":    [23, 24, 25],
    "Venus":   [26, 27, 1, 2],
}
_KRITTIKADI_MAP: Dict[int, str] = {}
for _lord, _arr in _KRITTIKADI_RANGES.items():
    for _i in _arr:
        _KRITTIKADI_MAP[int(_i)] = _lord

def _nak_index(nirayana_lon: float) -> int:
    """1..27 (Aśvinī=1)."""
    return int(math.floor(_norm360(nirayana_lon) / _NAK_WIDTH)) + 1

def _nak_offset_in_deg(nirayana_lon: float) -> float:
    base = (_nak_index(nirayana_lon) - 1) * _NAK_WIDTH
    return _norm360(nirayana_lon) - base

# ───────────────────────── fixed-point time helpers ─────────────────────────
# 1 tick = 1e-7 day (matches _JD_QUANT)
_TICK_PER_DAY = int(round(1.0 / _JD_QUANT))  # 10_000_000
_FRACTION = {k: ASHTOTTARI_YEARS[k] / ASHTOTTARI_TOTAL_YEARS for k in ASHTOTTARI_ORDER}

def _years_to_days(years: float, year_days: float) -> float:
    return years * year_days

def _to_ticks(days: float) -> int:
    return int(round(days * _TICK_PER_DAY))

def _from_ticks(ticks: int) -> float:
    return ticks / _TICK_PER_DAY

def _append_span_ticks(
    spans: List["DashaSpan"],
    *,
    level: int,
    lord: str,
    base_jd: float,
    t_start: int,
    dur_ticks: int,
    limit_ticks: Optional[int],
) -> int:
    """
    Append a span [t_start, t_start+dur_ticks) (intersected with limit).
    Returns t_next in ticks (t_start + dur_ticks).
    """
    t_end = t_start + dur_ticks
    if limit_ticks is not None and t_start >= limit_ticks:
        return t_end
    end_eff = min(t_end, limit_ticks) if limit_ticks is not None else t_end
    s_jd = _q(base_jd + _from_ticks(t_start))
    e_jd = _q(base_jd + _from_ticks(end_eff))
    if abs(e_jd - s_jd) < 1e-12:  # closed-open safety
        e_jd = _q(s_jd + 1e-9)
    spans.append(DashaSpan(level, lord, s_jd, e_jd))
    return t_end

def _partition_ticks(parent_ticks: int, seq: List[str]) -> List[int]:
    """
    Largest-remainder partition: proportional to ASHTOTTARI_YEARS.
    Guarantees sum(child_ticks) == parent_ticks and stable boundaries.
    """
    if parent_ticks <= 0:
        return [0] * len(seq)
    mults = [_FRACTION[l] for l in seq]
    raw = [parent_ticks * m for m in mults]
    base = [int(math.floor(x)) for x in raw]
    rems = [x - b for x, b in zip(raw, base)]
    need = parent_ticks - sum(base)
    if need > 0:
        order = sorted(range(len(rems)), key=lambda i: rems[i], reverse=True)
        for i in range(need):
            base[order[i]] += 1
    return base

def _lord_after(lord: str) -> str:
    i = ASHTOTTARI_ORDER.index(lord)
    return ASHTOTTARI_ORDER[(i + 1) % len(ASHTOTTARI_ORDER)]

def _cycle_from(after_lord: str, *, start_mode: str) -> List[str]:
    """
    start_mode="after": sequence begins from the planet AFTER the parent (default).
    start_mode="same":  sequence begins from the parent planet itself.
    """
    base = list(ASHTOTTARI_ORDER)
    start = after_lord if str(start_mode).lower().strip() == "same" else _lord_after(after_lord)
    i = base.index(start)
    return base[i:] + base[:i]

# ───────────────────────── data types ─────────────────────────
@dataclass
class DashaSpan:
    __slots__ = ("level", "lord", "start_jd_tt", "end_jd_tt")
    level: int
    lord: str
    start_jd_tt: float
    end_jd_tt: float

# ───────────────────────── schedule core ─────────────────────────
def ashtottari_schedule(
    *,
    jd_start_tt: float,
    moon_nirayana_deg: float,
    start_mode: str = "after",
    levels: int = 5,
    year_days: float = 365.24219,
    limit_jd_tt: float | None = None
) -> Dict[str, Any]:
    """
    Build flat spans + nested tree. Public signature unchanged.
    """
    levels = max(1, min(5, int(levels)))
    start_mode = "after" if str(start_mode).lower().strip() != "same" else "same"

    # Start lord from birth nakṣatra
    nak_idx = _nak_index(moon_nirayana_deg)
    nak_off = _nak_offset_in_deg(moon_nirayana_deg)
    start_lord = _KRITTIKADI_MAP[nak_idx]

    # Mahā sequence beginning at start_lord
    order = list(ASHTOTTARI_ORDER)
    i0 = order.index(start_lord)
    maha_cycle = order[i0:] + order[:i0]

    # Fixed-point timeline anchored at jd_start_tt
    base_jd = float(jd_start_tt)
    t0 = 0  # ticks from base_jd
    limit_ticks = None if limit_jd_tt is None else _to_ticks(float(limit_jd_tt) - base_jd)

    spans: List[DashaSpan] = []

    # First (partial) Mahā balance
    rem_frac = (_NAK_WIDTH - nak_off) / _NAK_WIDTH
    first_years = ASHTOTTARI_YEARS[start_lord]
    first_days = _years_to_days(first_years, year_days) * rem_frac
    t1 = _append_span_ticks(
        spans, level=1, lord=start_lord, base_jd=base_jd,
        t_start=t0, dur_ticks=_to_ticks(first_days), limit_ticks=limit_ticks
    )
    t_cur = t1

    # Remaining Mahā in this 108-year cycle
    for lord in maha_cycle[1:]:
        dur_days = _years_to_days(ASHTOTTARI_YEARS[lord], year_days)
        t_nxt = _append_span_ticks(
            spans, level=1, lord=lord, base_jd=base_jd,
            t_start=t_cur, dur_ticks=_to_ticks(dur_days), limit_ticks=limit_ticks
        )
        t_cur = t_nxt
        if limit_ticks is not None and t_cur >= limit_ticks:
            break

    # Sublevels (levels >= 2)
    if levels >= 2 and spans:
        _expand_sublevels_ticks(
            spans_level1=spans,
            base_jd=base_jd,
            start_mode=start_mode,
            levels=levels,
            limit_ticks=limit_ticks,
        )

    # Nested tree (linear)
    tree = _to_nested_linear(spans, max_level=levels)

    return {
        "ok": True,
        "scheme": "ashtottari",
        "order": list(ASHTOTTARI_ORDER),
        "years": dict(ASHTOTTARI_YEARS),
        "start": {
            "nakshatra_index": nak_idx,
            "nakshatra_name": NAKSHATRAS_27[nak_idx - 1],
            "start_lord": start_lord,
            "nakshatra_offset_deg": float(nak_off),
        },
        "year_days": float(year_days),
        "levels": int(levels),
        "spans": [
            {
                "level": s.level,
                "lord": s.lord,
                "start_jd_tt": float(s.start_jd_tt),
                "end_jd_tt": float(s.end_jd_tt),
            }
            for s in spans
        ],
        "nested": tree,
    }

def _expand_sublevels_ticks(
    *,
    spans_level1: List[DashaSpan],
    base_jd: float,
    start_mode: str,
    levels: int,
    limit_ticks: Optional[int],
) -> None:
    """
    Expand sublevels in-place using fixed-point ticks.
    Generates spans in (level, start) order so no final sorting is needed.
    """
    # We'll append to this master list in increasing level order.
    all_spans: List[DashaSpan] = list(spans_level1)

    # Convenience: convert a JD boundary back to ticks relative to base
    def jd_to_ticks(jd: float) -> int:
        return _to_ticks(jd - base_jd)

    current: List[DashaSpan] = [s for s in all_spans if s.level == 1]
    for lvl in range(2, levels + 1):
        next_level: List[DashaSpan] = []
        for parent in current:
            p_start = jd_to_ticks(parent.start_jd_tt)
            p_end   = jd_to_ticks(parent.end_jd_tt)
            p_ticks = max(0, p_end - p_start)
            if p_ticks == 0:
                continue
            seq = _cycle_from(parent.lord, start_mode=start_mode)
            parts = _partition_ticks(p_ticks, seq)

            t = p_start
            for lord, dur in zip(seq, parts):
                if dur <= 0:
                    continue
                t_next = _append_span_ticks(
                    next_level, level=lvl, lord=lord, base_jd=base_jd,
                    t_start=t, dur_ticks=dur, limit_ticks=limit_ticks
                )
                t = t_next
        # Append in order; current becomes next_level for deeper expansion
        all_spans.extend(next_level)
        current = next_level

    # Replace incoming level-1 buffer with the full ordered set
    spans_level1.clear()
    spans_level1.extend(all_spans)

def _to_nested_linear(spans: List[DashaSpan], *, max_level: int) -> List[Dict[str, Any]]:
    """
    Linear-time nesting:
    - Group spans by level (they are already in level order).
    - For each level L>1, sweep once with a pointer over parents (L-1) to attach children.
    """
    if not spans:
        return []

    by_level: Dict[int, List[DashaSpan]] = {}
    for s in spans:
        if s.level > max_level:
            continue
        by_level.setdefault(s.level, []).append(s)

    # Build node objects mirroring DashaSpan
    def mk_node(s: DashaSpan) -> Dict[str, Any]:
        return {
            "level": s.level,
            "lord": s.lord,
            "start_jd_tt": float(s.start_jd_tt),
            "end_jd_tt": float(s.end_jd_tt),
            "children": [] if s.level < max_level else None,
        }

    # Level 1 nodes (roots)
    level1_spans = by_level.get(1, [])
    if not level1_spans:
        return []
    nodes_by_level: Dict[int, List[Dict[str, Any]]] = {1: [mk_node(s) for s in level1_spans]}

    # Attach deeper levels
    for lvl in range(2, max_level + 1):
        parents = by_level.get(lvl - 1, [])
        kids    = by_level.get(lvl, [])
        if not parents or not kids:
            continue
        parent_nodes = nodes_by_level[lvl - 1]
        # two-pointer sweep through time
        p = 0
        cur_parent = parents[p] if parents else None
        for child in kids:
            while cur_parent and child.start_jd_tt >= cur_parent.end_jd_tt - 1e-12 and p + 1 < len(parents):
                p += 1
                cur_parent = parents[p]
            if not cur_parent:
                break
            if (child.start_jd_tt + 1e-12) >= cur_parent.start_jd_tt and (child.end_jd_tt - 1e-12) <= cur_parent.end_jd_tt:
                node = mk_node(child)
                parent_nodes[p]["children"].append(node)
                nodes_by_level.setdefault(lvl, []).append(node)

    # Strip empty children lists at leaves for cleanliness
    def strip_none_children(n: Dict[str, Any]) -> Dict[str, Any]:
        if "children" in n:
            if n["children"] is None or len(n["children"]) == 0 or n["level"] >= max_level:
                n.pop("children", None)
            else:
                n["children"] = [strip_none_children(k) for k in n["children"]]
        return n

    return [strip_none_children(n) for n in nodes_by_level[1]]

# ───────────────────────── moon longitude + timescales ─────────────────────────
def _moon_nirayana_deg_at(jd_tt: float, *, ayanamsa_key: str) -> float:
    if not _EPH_OK or ecliptic_longitudes is None:
        raise RuntimeError("EphemerisAdapter unavailable; cannot compute Moon longitude")
    rows = (ecliptic_longitudes(float(jd_tt), names=["Moon"]) or {}).get("results", [])
    if not rows:
        raise RuntimeError("ephemeris returned no Moon longitude")
    moon_trop = float(rows[0]["longitude"])
    ay = float(get_ayanamsa_deg(float(jd_tt), ayanamsa_key))
    return _norm360(moon_trop - ay)

def _timescales_from_civil(date: str, time: str, tz: str) -> Tuple[float, float, float]:
    """Return (jd_ut, jd_tt, jd_ut1). Prefer time_kernel (with dut1=0.0), else fallback."""
    if _tk is not None:
        for fname in ("timescales_from_civil", "compute_timescales", "build_timescales", "to_timescales", "from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(date=date, time=time, tz=tz, dut1=0.0)
                except TypeError:
                    out = fn(date, time, tz, 0.0)
                if isinstance(out, dict):
                    return (
                        float(out.get("jd_ut") or out.get("jd_utc")),
                        float(out.get("jd_tt") or out["jd_tt"]),
                        float(out.get("jd_ut1") or out.get("jd_ut") or out.get("jd_utc")),
                    )
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    ju, jt, j1 = out[:3]
                    return float(ju), float(jt), float(j1)
    if _ts is None:
        raise ValueError("timescales module not available")
    jd_ut = float(_ts.julian_day_utc(date, time, tz))
    try:
        y, m = map(int, date.split("-")[:2])
        jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m))
    except Exception:
        jd_tt = jd_ut + 69.0 / 86400.0  # fallback ΔT
    return jd_ut, jd_tt, jd_ut  # jd_ut1≈jd_ut when DUT1 unknown

# ───────────────────────── wrappers ─────────────────────────
_LEVEL_NAMES_5 = ("maha", "antara", "pratyantara", "sookshma", "prana")

def _wrap_root(nested_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not nested_nodes:
        return {"level": 0, "lord": "Ashtottari", "label": "Ashtottari",
                "start_jd_tt": 0.0, "end_jd_tt": 0.0, "children": []}
    start = min(n["start_jd_tt"] for n in nested_nodes)
    end   = max(n["end_jd_tt"] for n in nested_nodes)
    return {
        "level": 0,
        "lord": "Ashtottari",
        "label": "Ashtottari",
        "start_jd_tt": float(start),
        "end_jd_tt": float(end),
        "children": nested_nodes,
    }

def compute_dasha(
    jd_tt_birth: float,
    ayanamsa_key: str = "lahiri",
    ayanamsa_deg: float | None = None,   # accepted for parity; not used
    depth: int | None = None,
    options: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Registry-compatible entrypoint. (Signature unchanged)

    options:
      - start_mode: "after" | "same"
      - levels: 1..5 (default 5)
      - year_days: float
      - limit_jd_tt: float | None
      - moon_nirayana_deg: float | str  (override; bypass ephemeris)
    """
    opts = dict(options or {})
    start_mode = str(opts.get("start_mode", "after")).lower()
    levels = max(1, min(5, int(opts.get("levels", depth or 5))))
    year_days = float(opts.get("year_days", 365.24219))
    limit = _num(opts.get("limit_jd_tt"))

    # Accept numeric *or string* override
    moon_nira = _num(opts.get("moon_nirayana_deg"))
    if moon_nira is None:
        moon_nira = _moon_nirayana_deg_at(float(jd_tt_birth), ayanamsa_key=ayanamsa_key)
    else:
        moon_nira = _norm360(moon_nira)

    sched = ashtottari_schedule(
        jd_start_tt=float(jd_tt_birth),
        moon_nirayana_deg=float(moon_nira),
        start_mode=start_mode,
        levels=levels,
        year_days=year_days,
        limit_jd_tt=(float(limit) if limit is not None else None),
    )
    tree = _wrap_root(sched["nested"])
    return {"tree": tree, "levels": _LEVEL_NAMES_5[:levels]}

def compute_ashtottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route-friendly wrapper. (Signature & behavior unchanged)

    Input:
      - jd_tt OR {date,time,tz}
      - ayanamsa: str (default "lahiri")
      - start_mode: "after" | "same"
      - levels: 1..5 (default 5)
      - year_days: float (default 365.24219)
      - limit_jd_tt: float
      - moon_nirayana_deg: float | str (override; bypass ephemeris)
    """
    try:
        jd_tt = payload.get("jd_tt")
        if not isinstance(jd_tt, (int, float)):
            date = str(payload.get("date"))
            time = str(payload.get("time", "12:00:00"))
            tz   = str(payload.get("tz") or payload.get("place_tz") or "UTC")
            _ju, jd_tt, _j1 = _timescales_from_civil(date, time, tz)

        ay_key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
        start_mode = str(payload.get("start_mode", "after")).lower()
        levels = max(1, min(5, int(payload.get("levels", 5))))
        year_days = float(payload.get("year_days", 365.24219))
        limit = _num(payload.get("limit_jd_tt"))

        moon_nira = _num(payload.get("moon_nirayana_deg"))
        if moon_nira is None:
            moon_nira = _moon_nirayana_deg_at(float(jd_tt), ayanamsa_key=ay_key)
        else:
            moon_nira = _norm360(moon_nira)

        sched = ashtottari_schedule(
            jd_start_tt=float(jd_tt),
            moon_nirayana_deg=float(moon_nira),
            start_mode=start_mode,
            levels=levels,
            year_days=year_days,
            limit_jd_tt=(float(limit) if limit is not None else None),
        )
        return sched
    except Exception as e:
        return {"ok": False, "error": f"ashtottari_failed:{e}"}
