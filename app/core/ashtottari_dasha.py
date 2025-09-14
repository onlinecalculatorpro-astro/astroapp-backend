# app/core/ashtottari_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Aṣṭottarī Daśā — 108-year cycle with five nested levels
(Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa)

- Canonical order (years): Sun(6), Moon(15), Mars(8), Mercury(17), Saturn(10), Jupiter(19), Rahu(12), Venus(21).
- Start lord from Moon’s *birth* nakṣatra via Kṛttikādi mapping.
- Birth balance = remaining fraction in current nakṣatra × mahādaśā years of start lord.
- Sub-level durations scale by years(sub-lord)/108 within each parent span.
- start_mode="after" (default) begins sublevels with the planet AFTER the parent;
  start_mode="same" begins with the parent itself.

Public surface:
  compute_dasha(jd_tt_birth, ayanamsa_key="lahiri", ayanamsa_deg=None, depth=None, options=None) -> dict
  compute_ashtottari(payload: dict) -> dict
  ashtottari_schedule(...)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal, getcontext
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
getcontext().prec = 34  # deep fractional stability

_JD_QUANT = 1e-7  # ~0.009 s
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

def _years_to_days(years: Decimal, year_days: Decimal) -> Decimal:
    return years * year_days

def _lord_after(lord: str) -> str:
    i = ASHTOTTARI_ORDER.index(lord)
    return ASHTOTTARI_ORDER[(i + 1) % len(ASHTOTTARI_ORDER)]

def _cycle_from(after_lord: str, *, start_mode: str) -> List[str]:
    """
    start_mode="after": sequence begins from the planet AFTER parent.
    start_mode="same":  sequence begins from the parent planet itself.
    """
    base = list(ASHTOTTARI_ORDER)
    start = after_lord if start_mode == "same" else _lord_after(after_lord)
    i = base.index(start)
    return base[i:] + base[:i]

# ───────────────────────── data types ─────────────────────────
@dataclass
class DashaSpan:
    level: int
    lord: str
    start_jd_tt: float
    end_jd_tt: float

# ───────────────────────── schedule core ─────────────────────────
def _birth_balance_days_for_lord(nak_offset_deg: float, lord: str, *, year_days: Decimal) -> Decimal:
    rem_frac = Decimal((_NAK_WIDTH - nak_offset_deg) / _NAK_WIDTH)
    years = Decimal(ASHTOTTARI_YEARS[lord])
    return _years_to_days(rem_frac * years, year_days)

def _span_days_for_lord(parent_days: Decimal, lord: str) -> Decimal:
    return parent_days * (Decimal(ASHTOTTARI_YEARS[lord]) / Decimal(ASHTOTTARI_TOTAL_YEARS))

def _append_span(spans: List[DashaSpan], level: int, lord: str, t0: Decimal, dur_days: Decimal, *, limit: Optional[Decimal]) -> Decimal:
    t1 = t0 + dur_days
    if limit is not None and t0 >= limit:
        return t1
    end = min(t1, limit) if limit is not None else t1
    s = _q(float(t0)); e = _q(float(end))
    if abs(e - s) < 1e-12:  # closed-open safety
        e = _q(s + 1e-9)
    spans.append(DashaSpan(level, lord, s, e))
    return t1

def ashtottari_schedule(
    *,
    jd_start_tt: float,
    moon_nirayana_deg: float,
    start_mode: str = "after",
    levels: int = 5,
    year_days: float = 365.24219,
    limit_jd_tt: float | None = None
) -> Dict[str, Any]:
    levels = max(1, min(5, int(levels)))
    start_mode = "after" if str(start_mode).lower().strip() != "same" else "same"
    year_days_D = Decimal(str(year_days))
    t0 = Decimal(str(jd_start_tt))
    t_limit = Decimal(str(limit_jd_tt)) if isinstance(limit_jd_tt, (int, float)) else None

    # Start lord
    nak_idx = _nak_index(moon_nirayana_deg)
    nak_off = _nak_offset_in_deg(moon_nirayana_deg)
    start_lord = _KRITTIKADI_MAP[nak_idx]

    # Mahā sequence beginning at start_lord
    order = list(ASHTOTTARI_ORDER)
    i0 = order.index(start_lord)
    maha_cycle = order[i0:] + order[:i0]

    # Birth balance for first Mahā
    balance_days = _birth_balance_days_for_lord(nak_off, start_lord, year_days=year_days_D)

    spans: List[DashaSpan] = []

    # First (partial) Mahā
    _ = _years_to_days(Decimal(ASHTOTTARI_YEARS[start_lord]), year_days_D)
    t1 = _append_span(spans, 1, start_lord, t0, balance_days, limit=t_limit)
    t_cur = t1

    # Remaining Mahā in this 108-year cycle
    for lord in maha_cycle[1:]:
        dur = _years_to_days(Decimal(ASHTOTTARI_YEARS[lord]), year_days_D)
        t_next = _append_span(spans, 1, lord, t_cur, dur, limit=t_limit)
        t_cur = t_next
        if t_limit is not None and t_cur >= t_limit:
            break

    # Sublevels
    if levels >= 2:
        _expand_sublevels(spans, start_mode=start_mode, levels=levels, t_limit=t_limit)

    tree = _to_nested(spans, max_level=levels)
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
        "spans": [s.__dict__ for s in spans],
        "nested": tree,
    }

def _expand_sublevels(
    spans_level1: List[DashaSpan],
    *,
    start_mode: str,
    levels: int,
    t_limit: Optional[Decimal]
) -> None:
    all_spans: List[DashaSpan] = list(spans_level1)

    def _span_children(parent: DashaSpan, level: int) -> List[DashaSpan]:
        p_dur = Decimal(str(parent.end_jd_tt)) - Decimal(str(parent.start_jd_tt))
        seq = _cycle_from(parent.lord, start_mode=start_mode)
        t = Decimal(str(parent.start_jd_tt))
        kids: List[DashaSpan] = []
        for lord in seq:
            dur = _span_days_for_lord(p_dur, lord)
            t_next = t + dur
            if t_limit is not None and t >= t_limit:
                t = t_next
                continue
            s = _q(float(t)); e = _q(float(min(t_next, t_limit) if t_limit is not None else t_next))
            if abs(e - s) < 1e-12:
                e = _q(s + 1e-9)
            kids.append(DashaSpan(level, lord, s, e))
            t = t_next
        return kids

    current = [s for s in all_spans if s.level == 1]
    for level in range(2, levels + 1):
        next_level: List[DashaSpan] = []
        for parent in current:
            next_level.extend(_span_children(parent, level))
        all_spans.extend(next_level)
        current = next_level

    spans_level1.clear()
    spans_level1.extend(sorted(all_spans, key=lambda s: (s.level, s.start_jd_tt, ASHTOTTARI_ORDER.index(s.lord))))

def _to_nested(spans: List[DashaSpan], *, max_level: int) -> List[Dict[str, Any]]:
    level1 = [s for s in spans if s.level == 1]

    def children_of(parent: DashaSpan, level: int) -> List[DashaSpan]:
        return [
            s for s in spans
            if s.level == level
            and parent.start_jd_tt <= s.start_jd_tt + 1e-12
            and s.end_jd_tt <= parent.end_jd_tt + 1e-12
        ]

    def node_for(span: DashaSpan, level: int) -> Dict[str, Any]:
        node = {"level": level, "lord": span.lord, "start_jd_tt": float(span.start_jd_tt), "end_jd_tt": float(span.end_jd_tt)}
        if level < max_level:
            kids = children_of(span, level + 1)
            node["children"] = [node_for(k, level + 1) for k in kids]
        return node

    return [node_for(s, 1) for s in level1]

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
    Registry-compatible entrypoint.

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
    Route-friendly wrapper.

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
