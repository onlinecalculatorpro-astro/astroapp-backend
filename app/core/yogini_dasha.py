# app/core/yogini_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Yoginī Daśā — 36-year cycle with five nested levels
(Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa)

Optimizations:
- Fixed-point time arithmetic in ticks (1 tick = 1e-7 day) → no Decimal in hot path.
- Largest-remainder partition for sub-spans → stable, drift-free, children sum to parent.
- Linear-time tree construction (per-level two-pointer) → avoids O(N^2) filtering.
- Spans are generated in order → no final sorting pass.

Public API (unchanged):
  yogini_schedule(...)
  compute_yogini(payload: dict) -> dict
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

__all__ = [
    "YOGINI_ORDER", "YOGINI_YEARS", "YOGINI_TOTAL_YEARS",
    "yogini_schedule", "compute_yogini",
]

# ───────────────────────── deps (module-level adapter; no Config kwargs) ─────────────────────────
try:
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

YOGINI_ORDER: Tuple[str, ...] = (
    "Mangala","Pingala","Dhanya","Bhramari","Bhadrika","Ulka","Siddha","Sankata"
)
YOGINI_YEARS: Dict[str, int] = {
    "Mangala": 1, "Pingala": 2, "Dhanya": 3, "Bhramari": 4,
    "Bhadrika": 5, "Ulka": 6, "Siddha": 7, "Sankata": 8,
}
YOGINI_TOTAL_YEARS = 36

# Nakṣatra→start yoginī mapping (index 1..27): idx % 8 (1..7), 0→Sankata
def _start_lord_from_nak_index(idx: int) -> str:
    m = idx % 8
    return YOGINI_ORDER[(m - 1) % 8] if m != 0 else "Sankata"

def _nak_index(nirayana_lon: float) -> int:
    """1..27 (Aśvinī=1)."""
    return int(math.floor(_norm360(nirayana_lon) / _NAK_WIDTH)) + 1

def _nak_offset_in_deg(nirayana_lon: float) -> float:
    base = (_nak_index(nirayana_lon) - 1) * _NAK_WIDTH
    return _norm360(nirayana_lon) - base

# ───────────────────────── fixed-point time (ticks) ─────────────────────────
_TICK_PER_DAY = int(round(1.0 / _JD_QUANT))  # 10_000_000
_FRACTION = {k: YOGINI_YEARS[k] / YOGINI_TOTAL_YEARS for k in YOGINI_ORDER}

def _years_to_days(years: float, year_days: float) -> float:
    return years * year_days

def _to_ticks(days: float) -> int:
    return int(round(days * _TICK_PER_DAY))

def _from_ticks(ticks: int) -> float:
    return ticks / _TICK_PER_DAY

def _partition_ticks(parent_ticks: int, seq: List[str]) -> List[int]:
    """
    Largest-remainder partition proportional to YOGINI_YEARS.
    Ensures sum(child) == parent with stable boundaries.
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

def _after(lord: str) -> str:
    i = YOGINI_ORDER.index(lord)
    return YOGINI_ORDER[(i + 1) % 8]

def _cycle_from(parent_lord: str, *, start_mode: str) -> List[str]:
    start = parent_lord if str(start_mode).lower().strip() == "same" else _after(parent_lord)
    i = YOGINI_ORDER.index(start)
    return list(YOGINI_ORDER[i:] + YOGINI_ORDER[:i])

# ───────────────────────── data model ─────────────────────────
@dataclass
class DashaSpan:
    __slots__ = ("level", "lord", "start_jd_tt", "end_jd_tt")
    level: int
    lord: str
    start_jd_tt: float
    end_jd_tt: float

def _append_span_ticks(
    spans: List[DashaSpan],
    *, level: int, lord: str, base_jd: float,
    t_start: int, dur_ticks: int, limit_ticks: Optional[int],
) -> int:
    t_end = t_start + dur_ticks
    if limit_ticks is not None and t_start >= limit_ticks:
        return t_end
    end_eff = min(t_end, limit_ticks) if limit_ticks is not None else t_end
    s_jd = _q(base_jd + _from_ticks(t_start))
    e_jd = _q(base_jd + _from_ticks(end_eff))
    if abs(e_jd - s_jd) < 1e-12:
        e_jd = _q(s_jd + 1e-9)
    spans.append(DashaSpan(level, lord, s_jd, e_jd))
    return t_end

# ───────────────────────── schedule builder ─────────────────────────
def yogini_schedule(
    *,
    jd_start_tt: float,
    moon_nirayana_deg: float,
    start_mode: str = "after",          # "after" (default) or "same"
    levels: int = 5,
    year_days: float = 365.24219,
    limit_jd_tt: float | None = None,
    start_lord: str | None = None,
    compact: bool = False,              # optional compact nested encoding
    include_spans: bool = True,         # include flat spans array
) -> Dict[str, Any]:
    """
    Build a full Yoginī daśā timeline from birth epoch & Moon’s nirayana longitude.
    """
    levels = max(1, min(5, int(levels)))
    start_mode = "after" if str(start_mode).lower().strip() != "same" else "same"

    # Determine start yoginī
    nak_idx = _nak_index(moon_nirayana_deg)
    nak_off = _nak_offset_in_deg(moon_nirayana_deg)
    start_yogini = (start_lord.strip().title() if isinstance(start_lord, str) and start_lord.strip() else _start_lord_from_nak_index(nak_idx))
    if start_yogini not in YOGINI_YEARS:
        raise ValueError(f"Invalid start_lord '{start_yogini}' for Yoginī daśā")

    # Mahā order beginning at start_yogini
    i0 = YOGINI_ORDER.index(start_yogini)
    maha_cycle = list(YOGINI_ORDER[i0:] + YOGINI_ORDER[:i0])

    # Fixed-point timeline
    base_jd = float(jd_start_tt)
    limit_ticks = None if limit_jd_tt is None else _to_ticks(float(limit_jd_tt) - base_jd)

    spans: List[DashaSpan] = []

    # First (partial) Mahā balance
    rem_frac = (_NAK_WIDTH - nak_off) / _NAK_WIDTH
    first_days = _years_to_days(YOGINI_YEARS[start_yogini] * rem_frac, year_days)
    t_cur = _append_span_ticks(
        spans, level=1, lord=start_yogini, base_jd=base_jd,
        t_start=0, dur_ticks=_to_ticks(first_days), limit_ticks=limit_ticks
    )

    # Remaining Mahā in this 36-year cycle
    for lord in maha_cycle[1:]:
        dur_days = _years_to_days(YOGINI_YEARS[lord], year_days)
        t_cur = _append_span_ticks(
            spans, level=1, lord=lord, base_jd=base_jd,
            t_start=t_cur, dur_ticks=_to_ticks(dur_days), limit_ticks=limit_ticks
        )
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

    # Nested tree
    tree = _to_nested_linear(spans, max_level=levels, compact=compact)

    out: Dict[str, Any] = {
        "ok": True,
        "scheme": "yogini",
        "order": list(YOGINI_ORDER) if not compact else None,
        "years": dict(YOGINI_YEARS) if not compact else None,
        "start": {
            "nakshatra_index": nak_idx,
            "nakshatra_name": NAKSHATRAS_27[nak_idx - 1],
            "start_lord": start_yogini,
            "nakshatra_offset_deg": float(nak_off),
        },
        "year_days": float(year_days),
        "levels": int(levels),
        "nested": tree,
        "meta": {"compact": bool(compact), "include_spans": bool(include_spans), "encoding": "v1"},
    }
    if include_spans:
        out["spans"] = [
            {"level": s.level, "lord": s.lord, "start_jd_tt": float(s.start_jd_tt), "end_jd_tt": float(s.end_jd_tt)}
            for s in spans
        ]
    return {k: v for k, v in out.items() if v is not None}

def _expand_sublevels_ticks(
    *,
    spans_level1: List[DashaSpan],
    base_jd: float,
    start_mode: str,
    levels: int,
    limit_ticks: Optional[int],
) -> None:
    """
    Expand Antar..Prāṇa (levels 2..levels) in-place with fixed-point ticks.
    """
    all_spans: List[DashaSpan] = list(spans_level1)

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
                t = _append_span_ticks(
                    next_level, level=lvl, lord=lord, base_jd=base_jd,
                    t_start=t, dur_ticks=dur, limit_ticks=limit_ticks
                )
        all_spans.extend(next_level)
        current = next_level

    spans_level1.clear()
    spans_level1.extend(all_spans)

def _to_nested_linear(spans: List[DashaSpan], *, max_level: int, compact: bool = False) -> List[Dict[str, Any]]:
    """
    Group by level and attach children via a single sweep (two-pointer).
    In compact mode, nodes use short keys: l(level), p(planet index), s(start), e(end), c(children).
    """
    if not spans:
        return []

    by_level: Dict[int, List[DashaSpan]] = {}
    for s in spans:
        if s.level <= max_level:
            by_level.setdefault(s.level, []).append(s)

    def mk_node(s: DashaSpan) -> Dict[str, Any]:
        if not compact:
            node = {"level": s.level, "lord": s.lord, "start_jd_tt": float(s.start_jd_tt), "end_jd_tt": float(s.end_jd_tt)}
            if s.level < max_level: node["children"] = []
            return node
        idx = int(YOGINI_ORDER.index(s.lord))
        node = {"l": s.level, "p": idx, "s": float(s.start_jd_tt), "e": float(s.end_jd_tt)}
        if s.level < max_level: node["c"] = []
        return node

    level1_spans = by_level.get(1, [])
    if not level1_spans:
        return []
    nodes_by_level: Dict[int, List[Dict[str, Any]]] = {1: [mk_node(s) for s in level1_spans]}

    for lvl in range(2, max_level + 1):
        parents = by_level.get(lvl - 1, [])
        kids    = by_level.get(lvl, [])
        if not parents or not kids:
            continue
        parent_nodes = nodes_by_level[lvl - 1]
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
                if compact:
                    parent_nodes[p]["c"].append(node)
                else:
                    parent_nodes[p]["children"].append(node)
                nodes_by_level.setdefault(lvl, []).append(node)

    def strip(n: Dict[str, Any]) -> Dict[str, Any]:
        key = "c" if compact else "children"
        if key in n:
            if not n[key] or (compact and n.get("l", 0) >= max_level) or ((not compact) and n.get("level", 0) >= max_level):
                n.pop(key, None)
            else:
                n[key] = [strip(k) for k in n[key]]
        return n

    return [strip(n) for n in nodes_by_level[1]]

# ───────────────────────── Moon nirayana & timescales ─────────────────────────
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
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
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
        jd_tt = jd_ut + 69.0/86400.0  # fallback ΔT
    return jd_ut, jd_tt, jd_ut

# ───────────────────────── Orchestrator ─────────────────────────
def compute_yogini(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route-friendly wrapper.

    Inputs:
      - jd_tt (preferred) or date/time/tz
      - ayanamsa: str (default "lahiri")
      - start_mode: "after" (default) or "same"
      - levels: 1..5 (default 5)
      - year_days: float (default 365.24219)
      - limit_jd_tt: optional float
      - start_lord: optional explicit yoginī name
      - moon_nirayana_deg: optional float or str (deg)
      - compact: bool (optional; default False)
      - include_spans: bool (optional; default True)
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
        start_lord = payload.get("start_lord")
        compact = bool(payload.get("compact", False))
        include_spans = bool(payload.get("include_spans", True))

        moon_nira = _num(payload.get("moon_nirayana_deg"))
        if moon_nira is None:
            moon_nira = _moon_nirayana_deg_at(float(jd_tt), ayanamsa_key=ay_key)
        else:
            moon_nira = _norm360(moon_nira)

        sched = yogini_schedule(
            jd_start_tt=float(jd_tt),
            moon_nirayana_deg=float(moon_nira),
            start_mode=start_mode,
            levels=levels,
            year_days=year_days,
            limit_jd_tt=(float(limit) if limit is not None else None),
            start_lord=(str(start_lord).title() if isinstance(start_lord, str) and start_lord.strip() else None),
            compact=compact,
            include_spans=include_spans,
        )
        return sched
    except Exception as e:
        return {"ok": False, "error": f"yogini_failed:{e}"}
