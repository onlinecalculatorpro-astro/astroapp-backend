# app/core/chara_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Chara Daśā (Jaimini) — sign-based daśā with up to five nested levels
Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa

Integration & numerics:
- Sidereal-first pipeline (ayanāṁśa via app.core.ayanamsa.get_ayanamsa_deg).
- Timescales from time_kernel if available, else app.core.timescales.
- Ascendant from app.core.houses_advanced (GAST + true obliquity).
- Planet longitudes via app.core.ephemeris_adapter.ecliptic_longitudes (ecliptic-of-date),
  then siderealized.
- Deterministic nested scaling using Decimal to minimize floating drift.
- Single-pass tree/spans construction with caching for speed.

Rules (configurable):
- Start sign: Lagna (default) or Karakamśa (Ātmakāraka’s sign).
- Direction per sign (default "rashi_nature"):
    movable (Ar,Cn,Li,Cp) → forward; fixed (Ta,Le,Sc,Aq) → reverse; dual (Ge,Vi,Sg,Pi) → forward
  Alternatives: "odd_forward_even_reverse", "uniform_forward".
- Mahādaśā duration for sign S: inclusive count, following S’s own direction, from S to the
  sign occupied by S’s traditional lord (Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn).
  Same-sign ⇒ 12 years.
- Subperiod scaling (levels 2..5): proportional partition by local weights:
      child_days = parent_days × years(child) / Σ years(child_seq)
  (guarantees Σchildren = parent at every nesting; last child snaps to parent end.)
- Optional balance for the very first mahādaśā: `balance_years` or `balance_fraction` (0..1).

Public API
----------
compute_chara_dasha(payload: dict) -> dict (route-friendly)
Returns:
  {
    ok, scheme, start: {...}, rules: {...}, years_by_sign: {...},
    spans: [{level, sign_index, sign_name, start_jd_tt, end_jd_tt}, ...],
    nested: [{level,...,children:[...]}...],
    tree: {level:0,label:"chara",start_jd_tt,end_jd_tt,children:[...]},
    meta: {...}
  }
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal, getcontext
from functools import lru_cache
import math

# High-precision internal math for nested scaling
getcontext().prec = 34

# ───────────────────────── optional/guarded imports ──────────────────────────
try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None  # type: ignore

try:
    from app.core import timescales as _ts
except Exception:
    _ts = None  # type: ignore

try:
    from app.core.houses_advanced import PreciseHouseCalculator
except Exception:
    PreciseHouseCalculator = None  # type: ignore

# SAFE ephemeris wrapper (fast path; avoids constructing adapters here)
try:
    from app.core.ephemeris_adapter import ecliptic_longitudes  # type: ignore
    _EPH_OK = True
except Exception:
    ecliptic_longitudes = None  # type: ignore
    _EPH_OK = False

from app.core.ayanamsa import get_ayanamsa_deg

# ───────────────────────────── constants / helpers ───────────────────────────
SIGN_NAMES = (
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
)

# Traditional lords (Jaimini uses Parāśara rashi lords; nodes are not lords)
SIGN_LORD: Tuple[str, ...] = (
    "Mars","Venus","Mercury","Moon","Sun","Mercury",
    "Venus","Mars","Jupiter","Saturn","Saturn","Jupiter"
)

MOVABLE = {1,4,7,10}
FIXED   = {2,5,8,11}
DUAL    = {3,6,9,12}

_PLANETS_FOR_KARAKAS_7 = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn")
_PLANETS_FOR_KARAKAS_8 = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu")

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _sign_index(lon_deg: float) -> int:
    """1..12 (Aries=1)."""
    return int(math.floor(_norm360(lon_deg) / 30.0)) + 1

def _deg_within_sign(lon_deg: float) -> float:
    return _norm360(lon_deg) % 30.0

# ───────────────────────────── time helpers ─────────────────────────────
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
                    return (float(out.get("jd_ut") or out.get("jd_utc")),
                            float(out["jd_tt"]),
                            float(out.get("jd_ut1") or out["jd_tt"]))
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    return (float(out[0]), float(out[1]), float(out[2]))
    if _ts is None:
        raise ValueError("timescales module not available")
    jd_ut = float(_ts.julian_day_utc(date, time, tz))
    y, m = map(int, date.split("-")[:2])
    jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m)) if hasattr(_ts, "jd_tt_from_utc_jd") else jd_ut + 69.0/86400.0
    jd_ut1 = jd_ut
    return jd_ut, jd_tt, jd_ut1

# ───────────────────────── ephemeris / ayanāṁśa ─────────────────────────
def _planet_lons_sidereal(
    jd_tt: float, *, ay_key: str, preload: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Returns sidereal longitudes (deg) for required planets.
    Accepts optional preload {name: deg} (already sidereal).
    """
    want = {"Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu"}
    out: Dict[str, float] = {}
    if isinstance(preload, dict):
        for k, v in preload.items():
            if k in want:
                try:
                    x = float(v)
                except Exception:
                    continue
                if math.isfinite(x):
                    out[k] = x
    need = [p for p in want if p not in out]
    if need:
        if not _EPH_OK or ecliptic_longitudes is None:
            raise RuntimeError("ephemeris_unavailable")
        rows = (ecliptic_longitudes(float(jd_tt), names=list(need)) or {}).get("results", [])
        got = {str(r.get("name")): float(r.get("longitude")) for r in rows if "name" in r and "longitude" in r}
        ay = float(get_ayanamsa_deg(float(jd_tt), ay_key))
        for k in need:
            if k in got:
                out[k] = _norm360(got[k] - ay)
    # sanity check
    for lord in set(SIGN_LORD):
        if lord not in out:
            raise RuntimeError(f"missing sidereal longitude for {lord}")
    return out

# ────────────────────────── Ascendant (sidereal) ─────────────────────────
def _asc_sidereal_deg(payload: Dict[str, Any], *, ay_key: str) -> float:
    """
    Resolve sidereal Ascendant longitude in degrees from payload.
    Priority:
      1) asc_sidereal_deg
      2) asc_tropical_deg - ayanamsa
      3) compute via houses_advanced from site + date/time/tz or jd_tt/jd_ut1
    """
    if isinstance(payload.get("asc_sidereal_deg"), (int, float)):
        return _norm360(float(payload["asc_sidereal_deg"]))

    if isinstance(payload.get("asc_tropical_deg"), (int, float)):
        ay = float(get_ayanamsa_deg(float(payload.get("jd_tt") or 2451545.0), ay_key))
        return _norm360(float(payload["asc_tropical_deg"]) - ay)

    if PreciseHouseCalculator is None:
        raise RuntimeError("houses_advanced unavailable to compute Ascendant")

    jd_tt = payload.get("jd_tt")
    jd_ut1 = payload.get("jd_ut1")
    lat = payload.get("latitude")
    lon = payload.get("longitude")

    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
        # derive timescales from civil if present
        if not (isinstance(payload.get("date"), str)
                and isinstance(payload.get("time"), str)
                and (payload.get("tz") or payload.get("place_tz"))):
            raise ValueError("To compute Ascendant, provide either asc_* directly OR (date,time,tz,latitude,longitude)")
        tz = str(payload.get("tz") or payload.get("place_tz"))
        jd_ut, jd_tt, jd_ut1 = _timescales_from_civil(str(payload["date"]), str(payload["time"]), tz)
        lat = float(payload["latitude"]); lon = float(payload["longitude"])
    else:
        if not (isinstance(jd_tt, (int, float)) and isinstance(jd_ut1, (int, float))):
            if isinstance(payload.get("date"), str):
                tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
                _ju, jd_tt, jd_ut1 = _timescales_from_civil(str(payload["date"]), str(payload.get("time", "12:00:00")), tz)
            else:
                raise ValueError("Provide jd_tt and jd_ut1 when supplying latitude/longitude directly")

    calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=False, enable_validation=False)
    hd = calc.calculate_houses(latitude=float(lat),
                               longitude=float(lon),
                               jd_ut=float(jd_ut1),
                               jd_tt=float(jd_tt),
                               jd_ut1=float(jd_ut1))
    ay = float(get_ayanamsa_deg(float(jd_tt), ay_key))
    return _norm360(hd.ascendant - ay)

# ─────────────────────────── Karakas (Ātmakāraka) ──────────────────────────
def _atmakaraka_sign_index(planet_lons_sid: Dict[str, float], *, include_rahu: bool = False) -> int:
    """
    Ātmakāraka = planet with max longitude within its sign (0..30), ties by full precision.
    Default 7-karaka (exclude Rahu/Ketu). Optionally include Rahu (8-karaka).
    """
    cand_names = _PLANETS_FOR_KARAKAS_8 if include_rahu else _PLANETS_FOR_KARAKAS_7
    best_name = None
    best_within = -1.0
    for nm in cand_names:
        lon = planet_lons_sid.get(nm)
        if lon is None:
            continue
        w = _deg_within_sign(float(lon))
        if w > best_within:
            best_within = w
            best_name = nm
    if best_name is None:
        raise RuntimeError("Unable to determine Ātmakāraka (no planet longitudes)")
    return _sign_index(float(planet_lons_sid[best_name]))

# ─────────────────────── direction / sequencing / years ──────────────────────
def _dir_forward(sign_idx: int, *, mode: str) -> bool:
    mode = (mode or "rashi_nature").lower().strip()
    if mode == "rashi_nature":
        return sign_idx not in FIXED  # movable & dual forward
    if mode == "odd_forward_even_reverse":
        return (sign_idx % 2) == 1
    if mode == "uniform_forward":
        return True
    raise ValueError(f"Unknown direction_mode: {mode}")

def _next_sign(sign_idx: int, *, forward: bool) -> int:
    return 1 if (forward and sign_idx == 12) else (12 if (not forward and sign_idx == 1) else (sign_idx + 1 if forward else sign_idx - 1))

@lru_cache(maxsize=512)
def _sign_sequence_from(start_sign: int, direction_mode: str) -> Tuple[int, ...]:
    """Sequence of 12 signs stepping by the direction of each *current* sign."""
    order: List[int] = []
    seen = set()
    s = int(start_sign)
    for _ in range(24):  # safety
        if s in seen:
            break
        order.append(s); seen.add(s)
        fwd = _dir_forward(s, mode=direction_mode)
        s = _next_sign(s, forward=fwd)
    if len(order) != 12:
        for i in range(1, 13):
            if i not in seen:
                order.append(i)
        order = order[:12]
    return tuple(order)

# Mahā = same stepping rule as child sequences
_maha_sequence_cached = _sign_sequence_from

def _count_to_target_in_its_direction(start_sign: int, target_sign: int, *, forward_from_start: bool) -> int:
    """Inclusive count (1..12). Same sign => 12."""
    if start_sign == target_sign:
        return 12
    count = 1
    s = start_sign
    while True:
        s = _next_sign(s, forward=forward_from_start)
        count += 1
        if s == target_sign or count > 12:
            break
    return min(count, 12)

def _lord_pos_tuple(planets_sid: Dict[str, float]) -> Tuple[int, ...]:
    """Tuple of length 12: for each sign 1..12 → sign index of its lord."""
    try:
        return tuple(_sign_index(float(planets_sid[SIGN_LORD[s-1]])) for s in range(1, 13))
    except KeyError as ke:
        raise RuntimeError(f"missing sidereal longitude for {ke.args[0]}") from ke

@lru_cache(maxsize=256)
def _years_by_sign_cached(lord_pos_tuple: Tuple[int, ...], direction_mode: str) -> Tuple[int, ...]:
    """Cached years_by_sign; returns a tuple of 12 ints (index 0→sign 1)."""
    yrs = [0] * 12
    for s in range(1, 13):
        fwd = _dir_forward(s, mode=direction_mode)
        tgt = lord_pos_tuple[s - 1]
        yrs[s - 1] = _count_to_target_in_its_direction(s, tgt, forward_from_start=fwd)
    return tuple(yrs)

def _years_by_sign(planets_sid: Dict[str, float], *, direction_mode: str) -> Dict[int, int]:
    tup = _lord_pos_tuple(planets_sid)
    yrs_tuple = _years_by_sign_cached(tup, direction_mode)
    return {i: yrs_tuple[i-1] for i in range(1, 13)}

# ───────────────────────────── data model ─────────────────────────────
@dataclass
class DashaSpan:
    level: int                 # 1..5
    sign: int                  # 1..12
    start_jd_tt: float
    end_jd_tt: float

# ─────────────────────── core: single-pass tree + spans ──────────────────────
def _years_to_days(years: Decimal, year_days: Decimal) -> Decimal:
    return years * year_days

def _build_tree_and_spans(
    *,
    jd_start_tt: float,
    start_sign_index: int,
    years_by_sign: Dict[int, int],
    direction_mode: str,
    levels: int,
    year_days: float,
    limit_jd_tt: float | None,
    balance_years: float | None,
    include_spans: bool,
    include_nested: bool,
) -> Tuple[List[Dict[str, Any]], List[DashaSpan]]:
    """Build nested nodes (level 1..L) and flat spans in a single pass."""
    year_days_D = Decimal(str(year_days))
    t_limit = Decimal(str(limit_jd_tt)) if isinstance(limit_jd_tt, (int, float)) else None

    def clip_end(t_next: Decimal) -> Decimal:
        return min(t_next, t_limit) if t_limit is not None else t_next

    all_spans: List[DashaSpan] = []
    nested_lvl1: List[Dict[str, Any]] = []

    def append_span(level: int, sign: int, t0: Decimal, t1: Decimal):
        if include_spans and t1 > t0:
            all_spans.append(DashaSpan(level, sign, float(t0), float(t1)))

    def mk_node(level: int, sign: int, t0: Decimal, t1: Decimal) -> Dict[str, Any]:
        node = {
            "level": level,
            "sign_index": sign,
            "sign_name": SIGN_NAMES[sign - 1],
            "start_jd_tt": float(t0),
            "end_jd_tt": float(t1),
        }
        if include_nested and level < levels:
            node["children"] = []
        return node

    def _expand_children(node: Dict[str, Any], *, parent_sign: int, level_next: int, parent_start_nom: Decimal, parent_end_nom: Decimal):
        """Recursive children expansion (local-sum proportional split; last child snaps to effective end)."""
        if (not include_nested and not include_spans) or level_next > levels:
            return

        # Effective window (respect global limit); children must sum to this effective duration
        eff_start = parent_start_nom
        eff_end = min(parent_end_nom, t_limit) if t_limit is not None else parent_end_nom
        eff_days = eff_end - eff_start
        if eff_days <= Decimal(0):
            return

        seq = _sign_sequence_from(parent_sign, direction_mode)
        weights = [int(years_by_sign[s]) for s in seq]
        sum_w = Decimal(sum(weights))
        equal_part = eff_days / Decimal(len(seq)) if sum_w == 0 else None

        t0 = eff_start
        for idx, child_sign in enumerate(seq):
            # last child takes all remaining to absorb rounding
            if idx == len(seq) - 1:
                t1_nom = eff_end
            else:
                if sum_w == 0:
                    part = equal_part
                else:
                    part = eff_days * (Decimal(weights[idx]) / sum_w)
                t1_nom = t0 + part
            t1_eff = clip_end(t1_nom)
            append_span(level_next, child_sign, t0, t1_eff)

            if include_nested:
                child_node = mk_node(level_next, child_sign, t0, t1_eff)
                node.setdefault("children", []).append(child_node)
                # Recurse using nominal sub-window; deeper levels will again clip to limit
                if level_next + 1 <= levels and t1_nom > t0:
                    _expand_children(child_node,
                                     parent_sign=child_sign,
                                     level_next=level_next + 1,
                                     parent_start_nom=t0,
                                     parent_end_nom=t1_nom)

            t0 = t1_nom
            if t0 >= eff_end:
                break

    # Level-1 order
    order = _maha_sequence_cached(int(start_sign_index), direction_mode)
    t = Decimal(str(jd_start_tt))

    # Build L1 and recurse
    for idx, s in enumerate(order):
        years = Decimal(years_by_sign[s])
        dur_days = _years_to_days(years, year_days_D)
        if idx == 0 and isinstance(balance_years, (int, float)):
            dur_days = max(Decimal(0), dur_days - _years_to_days(Decimal(str(balance_years)), year_days_D))

        t_next_nom = t + dur_days
        if t_limit is not None and t >= t_limit:
            t = t_next_nom
            break

        end_eff = clip_end(t_next_nom)
        node = mk_node(1, s, t, end_eff)
        append_span(1, s, t, end_eff)
        nested_lvl1.append(node)

        if levels >= 2:
            _expand_children(node,
                             parent_sign=s,
                             level_next=2,
                             parent_start_nom=t,
                             parent_end_nom=t_next_nom)

        t = t_next_nom

    return nested_lvl1, all_spans

# ─────────────────────────── schedule facade ───────────────────────────
def chara_schedule(
    *_,
    jd_start_tt: float,
    start_sign_index: int,
    planets_sidereal: Dict[str, float],
    direction_mode: str = "rashi_nature",
    levels: int = 5,
    year_days: float = 365.24219,
    limit_jd_tt: float | None = None,
    balance_years: float | None = None,
) -> Dict[str, Any]:
    """Build the full Chara Daśā schedule."""
    levels = max(1, min(5, int(levels)))
    yrs = _years_by_sign(planets_sidereal, direction_mode=direction_mode)

    nested, spans = _build_tree_and_spans(
        jd_start_tt=float(jd_start_tt),
        start_sign_index=int(start_sign_index),
        years_by_sign=yrs,
        direction_mode=direction_mode,
        levels=int(levels),
        year_days=float(year_days),
        limit_jd_tt=limit_jd_tt,
        balance_years=balance_years,
        include_spans=True,
        include_nested=True,
    )

    if nested:
        s0 = min(float(n.get("start_jd_tt", 0.0)) for n in nested)
        e1 = max(float(n.get("end_jd_tt", 0.0)) for n in nested)
    else:
        s0 = float(jd_start_tt)
        e1 = float(jd_start_tt)

    return {
        "ok": True,
        "scheme": "chara",
        "years_by_sign": {i: int(yrs[i]) for i in range(1, 13)},
        "mahadasa_order": list(_maha_sequence_cached(int(start_sign_index), direction_mode)),
        "levels": int(levels),
        "year_days": float(year_days),
        "spans": [
            {
                "level": s.level,
                "sign_index": s.sign,
                "sign_name": SIGN_NAMES[s.sign - 1],
                "start_jd_tt": s.start_jd_tt,
                "end_jd_tt": s.end_jd_tt,
            } for s in spans
        ],
        "nested": nested,
        "tree": {
            "level": 0,
            "lord": None,
            "label": "chara",
            "start_jd_tt": s0,
            "end_jd_tt": e1,
            "children": nested,
        },
    }

# ───────────────────────── orchestrator (route-friendly) ─────────────────────
def compute_chara_dasha(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build Chara Daśā from a route-style payload.

    Inputs (any one path):
      A) asc_sidereal_deg (or asc_tropical_deg + ayanamsa) and jd_tt
      B) jd_tt & jd_ut1 & latitude & longitude
      C) date/time/tz & latitude & longitude

    Options:
      - ayanamsa: str = "lahiri"
      - start_from: "lagna" | "ak"
      - include_rahu_in_karakas: bool = False
      - direction_mode: "rashi_nature" | "odd_forward_even_reverse" | "uniform_forward"
      - levels: 1..5 (default 5)
      - year_days: 365.24219
      - limit_jd_tt: float | None
      - balance_years: float | None
      - balance_fraction: float | None (applied to first mahā)
      - override_start_sign_index: int 1..12
      - planet_longitudes_sidereal: optional dict {name: deg} (precomputed)
    """
    try:
        ay_key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
        direction_mode = str(payload.get("direction_mode", "rashi_nature")).strip().lower()
        levels = int(payload.get("levels", 5))
        year_days = float(payload.get("year_days", 365.24219))
        limit = payload.get("limit_jd_tt")
        limit_jd_tt = float(limit) if isinstance(limit, (int, float)) else None

        # Resolve jd_tt (and possibly jd_ut1 for asc calc path C)
        jd_tt = payload.get("jd_tt")
        jd_ut1 = payload.get("jd_ut1")
        if not isinstance(jd_tt, (int, float)):
            if isinstance(payload.get("date"), str):
                tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
                _ju, jd_tt, jd_ut1 = _timescales_from_civil(str(payload["date"]), str(payload.get("time", "12:00:00")), tz)
            else:
                raise ValueError("jd_tt or (date,time,tz) required")

        # Planet sidereal longitudes (for lords and optionally ĀK)
        preload = payload.get("planet_longitudes_sidereal")
        planets_sid = _planet_lons_sidereal(
            float(jd_tt), ay_key=ay_key,
            preload=preload if isinstance(preload, dict) else None
        )

        # Starting sign
        start_override = payload.get("override_start_sign_index")
        if isinstance(start_override, int) and 1 <= int(start_override) <= 12:
            start_sign = int(start_override)
            start_basis = "override"
        else:
            start_from = str(payload.get("start_from", "lagna")).strip().lower()
            if start_from == "ak":
                include_rahu = bool(payload.get("include_rahu_in_karakas", False))
                ak_sign = _atmakaraka_sign_index(planets_sid, include_rahu=include_rahu)
                start_sign = ak_sign
                start_basis = "karakamsha"
            else:
                asc_sid = _asc_sidereal_deg(payload, ay_key=ay_key)
                start_sign = _sign_index(asc_sid)
                start_basis = "lagna"

        # Balance for first mahādaśā (optional)
        balance_years: Optional[float] = None
        if isinstance(payload.get("balance_years"), (int, float)):
            balance_years = float(payload["balance_years"])
        elif isinstance(payload.get("balance_fraction"), (int, float)):
            frac = max(0.0, min(0.999999, float(payload["balance_fraction"])))
            yrs_tmp = _years_by_sign(planets_sid, direction_mode=direction_mode)
            first_years = float(yrs_tmp[start_sign])
            balance_years = first_years * frac

        # Build schedule
        sched = chara_schedule(
            jd_start_tt=float(jd_tt),
            start_sign_index=int(start_sign),
            planets_sidereal=planets_sid,
            direction_mode=direction_mode,
            levels=int(levels),
            year_days=float(year_days),
            limit_jd_tt=limit_jd_tt,
            balance_years=balance_years,
        )

        # Metadata
        sched["start"] = {
            "basis": start_basis,
            "sign_index": int(start_sign),
            "sign_name": SIGN_NAMES[start_sign - 1],
        }
        sched["rules"] = {
            "direction_mode": direction_mode,
            "duration_rule": "count_to_lord_in_sign_direction (inclusive; same-sign=12)",
            "sublevel_scaling": "proportional (local years / local sum; last child snaps)",
        }
        sched["meta"] = {
            "ayanamsa": ay_key,
            "asc_sidereal_deg": (float(payload.get("asc_sidereal_deg"))
                                 if isinstance(payload.get("asc_sidereal_deg"), (int, float)) else None),
        }
        return sched

    except Exception as e:
        return {"ok": False, "error": f"chara_failed:{e}"}
