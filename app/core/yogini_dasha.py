# app/core/yogini_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Yoginī Daśā — 36-year cycle with five nested levels (Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa)

Scope & numerics
- Deterministic, research-grade schedule builder with precise time handling.
- 8-yoginī sequence totaling 36 years:
    Order (cyclic):
        Mangala(1), Pingala(2), Dhanya(3), Bhramari(4),
        Bhadrika(5), Ulka(6), Siddha(7), Sankata(8)
- Start lord chosen from Moon’s birth nakṣatra index using a clean modulo mapping:
      start = index % 8 with 1→Mangala, …, 7→Siddha, 0→Sankata.
  (You can override via payload['start_lord'] if needed.)
- Balance at birth = remaining fraction of the *current nakṣatra* × mahādaśā years of start lord.
- Sub-periods: each lower level scales by (years(sub_lord) / 36) and follows the same 8-yoginī order.
  Default Antar sequence begins with the yoginī *after* the parent; set start_mode="same" to begin with parent.

Public API
    compute_yogini(payload: dict) -> dict
      Inputs (either jd_tt directly OR civil date/time/tz):
        - jd_tt (float); optionally date="YYYY-MM-DD", time="HH:MM[:SS]", tz="Area/City"
        - ayanamsa (str) default "lahiri"
        - start_mode: "after" (default) or "same"
        - levels: 1..5 (default 5)
        - year_days: float (days per year, default 365.24219)
        - limit_jd_tt: optional end JD_TT to stop schedule
        - start_lord: optional explicit start yoginī name (overrides nakṣatra mapping)
        - moon_nirayana_deg: optional precomputed Moon nirayana longitude (deg)

    yogini_schedule(jd_start_tt, moon_nirayana_deg, *, start_mode="after",
                    levels=5, year_days=365.24219, limit_jd_tt=None,
                    start_lord: str | None = None) -> dict
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal, getcontext
import math

# Optional ephemeris / timescales used for Moon longitude & time conversion
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    from app.core.ephem_singleton import TS, PLANETS
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    TS = None; PLANETS = None
    _EPH_OK = False

try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

from app.core.ayanamsa import get_ayanamsa_deg
from app.core.constants_vedic import NAKSHATRAS_27

# ───────────────────────── constants / helpers ─────────────────────────

getcontext().prec = 34  # high precision for nested Decimal products

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

_NAK_WIDTH = 360.0 / 27.0  # 13°20′

YOGINI_ORDER: Tuple[str, ...] = (
    "Mangala","Pingala","Dhanya","Bhramari","Bhadrika","Ulka","Siddha","Sankata"
)
YOGINI_YEARS: Dict[str, int] = {
    "Mangala": 1, "Pingala": 2, "Dhanya": 3, "Bhramari": 4,
    "Bhadrika": 5, "Ulka": 6, "Siddha": 7, "Sankata": 8,
}
YOGINI_TOTAL_YEARS = 36

# Nakṣatra→start yoginī mapping (index 1..27). Default: modulo pattern.
# idx % 8: 1 Mangala, 2 Pingala, 3 Dhanya, 4 Bhramari, 5 Bhadrika, 6 Ulka, 7 Siddha, 0 Sankata
def _start_lord_from_nak_index(idx: int) -> str:
    m = idx % 8
    return YOGINI_ORDER[(m - 1) % 8] if m != 0 else "Sankata"

def _nak_index(nirayana_lon: float) -> int:
    """1..27 (Aśvinī=1)."""
    return int(math.floor(_norm360(nirayana_lon) / _NAK_WIDTH)) + 1

def _nak_offset_in_deg(nirayana_lon: float) -> float:
    base = (_nak_index(nirayana_lon) - 1) * _NAK_WIDTH
    return _norm360(nirayana_lon) - base

def _years_to_days(years: Decimal, year_days: Decimal) -> Decimal:
    return years * year_days

def _after(lord: str) -> str:
    i = YOGINI_ORDER.index(lord)
    return YOGINI_ORDER[(i + 1) % 8]

def _cycle_from(parent_lord: str, *, start_mode: str) -> List[str]:
    base = list(YOGINI_ORDER)
    start = parent_lord if start_mode == "same" else _after(parent_lord)
    i = base.index(start)
    return base[i:] + base[:i]

# ───────────────────────── data model ─────────────────────────

@dataclass
class DashaSpan:
    level: int                # 1..5
    lord: str
    start_jd_tt: float
    end_jd_tt: float

# ───────────────────────── core math ─────────────────────────

def _birth_balance_days(nak_offset_deg: float, lord: str, *, year_days: Decimal) -> Decimal:
    """
    Remaining portion of current nakṣatra × years(lord).
    """
    rem_frac = Decimal((_NAK_WIDTH - nak_offset_deg) / _NAK_WIDTH)
    years = Decimal(YOGINI_YEARS[lord])
    return _years_to_days(rem_frac * years, year_days)

def _sub_duration(parent_days: Decimal, lord: str) -> Decimal:
    """
    Sub-period duration = parent_duration × (years(lord)/36).
    """
    return parent_days * (Decimal(YOGINI_YEARS[lord]) / Decimal(YOGINI_TOTAL_YEARS))

def _append(spans: List[DashaSpan], level: int, lord: str, t0: Decimal, dur_days: Decimal, *, limit: Optional[Decimal]) -> Decimal:
    t1 = t0 + dur_days
    if limit is not None and t0 >= limit:
        return t1
    end = min(t1, limit) if limit is not None else t1
    spans.append(DashaSpan(level, lord, float(t0), float(end)))
    return t1

# ───────────────────────── schedule builder ─────────────────────────

def yogini_schedule(
    *,
    jd_start_tt: float,
    moon_nirayana_deg: float,
    start_mode: str = "after",          # "after" (default) or "same"
    levels: int = 5,
    year_days: float = 365.24219,
    limit_jd_tt: float | None = None,
    start_lord: str | None = None
) -> Dict[str, Any]:
    """
    Build a full Yoginī daśā timeline from starting epoch & Moon’s nirayana longitude.
    """
    levels = max(1, min(5, int(levels)))
    start_mode = "after" if str(start_mode).lower().strip() != "same" else "same"

    year_days_D = Decimal(str(year_days))
    t0 = Decimal(str(jd_start_tt))
    t_limit = Decimal(str(limit_jd_tt)) if isinstance(limit_jd_tt, (int,float)) else None

    # Determine start yoginī
    nak_idx = _nak_index(moon_nirayana_deg)
    nak_off = _nak_offset_in_deg(moon_nirayana_deg)
    start_yogini = (start_lord or _start_lord_from_nak_index(nak_idx)).strip().title()
    if start_yogini not in YOGINI_YEARS:
        raise ValueError(f"Invalid start_lord '{start_yogini}' for Yoginī daśā")

    # Build Mahā cycle starting at start_yogini
    order = list(YOGINI_ORDER)
    i0 = order.index(start_yogini)
    maha_cycle = order[i0:] + order[:i0]

    spans: List[DashaSpan] = []

    # First Mahā (partial balance)
    balance_days = _birth_balance_days(nak_off, start_yogini, year_days=year_days_D)
    t1 = _append(spans, 1, start_yogini, t0, balance_days, limit=t_limit)
    t_cur = t1

    # Remaining Mahā in current cycle (full durations)
    for lord in maha_cycle[1:]:
        dur = _years_to_days(Decimal(YOGINI_YEARS[lord]), year_days_D)
        t_next = _append(spans, 1, lord, t_cur, dur, limit=t_limit)
        t_cur = t_next
        if t_limit is not None and t_cur >= t_limit:
            break

    # Expand sublevels up to requested depth
    if levels >= 2:
        _expand_sublevels(spans, start_mode=start_mode, levels=levels, t_limit=t_limit)

    tree = _to_nested(spans, max_level=levels)
    return {
        "ok": True,
        "scheme": "yogini",
        "order": list(YOGINI_ORDER),
        "years": dict(YOGINI_YEARS),
        "start": {
            "nakshatra_index": nak_idx,
            "nakshatra_name": NAKSHATRAS_27[nak_idx - 1],
            "start_lord": start_yogini,
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
    t_limit: Optional[Decimal],
) -> None:
    """
    Populate Antar..Prāṇa (levels 2..levels) under each Mahā span in-place.
    """
    all_spans = list(spans_level1)

    def add_children(parent: DashaSpan, level: int) -> List[DashaSpan]:
        parent_days = Decimal(str(parent.end_jd_tt)) - Decimal(str(parent.start_jd_tt))
        seq = _cycle_from(parent.lord, start_mode=start_mode)
        t = Decimal(str(parent.start_jd_tt))
        kids: List[DashaSpan] = []
        for lord in seq:
            dur = _sub_duration(parent_days, lord)
            t_next = t + dur
            if t_limit is not None and t >= t_limit:
                t = t_next
                continue
            end = min(t_next, t_limit) if t_limit is not None else t_next
            kids.append(DashaSpan(level, lord, float(t), float(end)))
            t = t_next
        return kids

    current = [s for s in all_spans if s.level == 1]
    for level in range(2, levels + 1):
        nxt: List[DashaSpan] = []
        for p in current:
            nxt.extend(add_children(p, level))
        all_spans.extend(nxt)
        current = nxt

    spans_level1.clear()
    spans_level1.extend(sorted(all_spans, key=lambda s: (s.level, s.start_jd_tt, YOGINI_ORDER.index(s.lord))))

def _to_nested(spans: List[DashaSpan], *, max_level: int) -> List[Dict[str, Any]]:
    def children_of(parent: DashaSpan, level: int) -> List[DashaSpan]:
        # containment with tiny numerical cushion
        return [s for s in spans if s.level == level and parent.start_jd_tt <= s.start_jd_tt + 1e-12 and s.end_jd_tt <= parent.end_jd_tt + 1e-12]

    def node_for(span: DashaSpan, level: int) -> Dict[str, Any]:
        node = {"level": level, "lord": span.lord, "start_jd_tt": span.start_jd_tt, "end_jd_tt": span.end_jd_tt}
        if level < max_level:
            node["children"] = [node_for(k, level + 1) for k in children_of(span, level + 1)]
        return node

    return [node_for(s, 1) for s in spans if s.level == 1]

# ───────────────────────── Moon nirayana & timescales helpers ─────────────────────────

def _moon_nirayana_deg_at(jd_tt: float, *, ayanamsa_key: str) -> float:
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")
    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore
    rows = ep.ecliptic_longitudes(float(jd_tt), ["Moon"]).get("results", [])
    if not rows:
        raise RuntimeError("ephemeris returned no Moon longitude")
    moon_trop = float(rows[0]["longitude"])
    ay = float(get_ayanamsa_deg(float(jd_tt), ayanamsa_key))
    return _norm360(moon_trop - ay)

def _timescales_from_civil(date: str, time: str, tz: str) -> Tuple[float, float, float]:
    """
    Returns (jd_ut, jd_tt, jd_ut1). Uses time_kernel if present, else timescales.
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
                    return (float(out.get("jd_ut") or out.get("jd_utc")), float(out["jd_tt"]), float(out.get("jd_ut1") or out["jd_tt"]))
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    return (float(out[0]), float(out[1]), float(out[2]))
    if _ts is None:
        raise ValueError("timescales module not available")
    jd_ut = float(_ts.julian_day_utc(date, time, tz))
    y, m = map(int, date.split("-")[:2])
    jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m)) if hasattr(_ts, "jd_tt_from_utc_jd") else jd_ut + 69.0/86400.0
    jd_ut1 = jd_ut
    return jd_ut, jd_tt, jd_ut1

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
      - start_lord: optional explicit string
      - moon_nirayana_deg: optional float (deg)

    Returns:
      { ok, scheme, order, years, start:{...}, levels, year_days, spans:[...], nested:[...] }
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
        levels = int(payload.get("levels", 5))
        year_days = float(payload.get("year_days", 365.24219))
        limit_jd_tt = payload.get("limit_jd_tt")
        start_lord = payload.get("start_lord")
        limit = float(limit_jd_tt) if isinstance(limit_jd_tt, (int,float)) else None

        moon_nira = float(payload.get("moon_nirayana_deg")) if isinstance(payload.get("moon_nirayana_deg"), (int,float)) \
                    else _moon_nirayana_deg_at(float(jd_tt), ayanamsa_key=ay_key)

        sched = yogini_schedule(
            jd_start_tt=float(jd_tt),
            moon_nirayana_deg=moon_nira,
            start_mode=start_mode,
            levels=levels,
            year_days=year_days,
            limit_jd_tt=limit,
            start_lord=(str(start_lord).title() if isinstance(start_lord, str) and start_lord.strip() else None),
        )
        return sched
    except Exception as e:
        return {"ok": False, "error": f"yogini_failed:{e}"}
