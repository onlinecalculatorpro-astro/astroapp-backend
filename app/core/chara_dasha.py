# app/core/chara_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Chara Daśā (Jaimini) — sign-based daśā with up to five nested levels
Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa

Performance-oriented design:
- O(N) tree construction (no O(N²) scans).
- Early exit at limit_jd_tt on every level.
- Minimal ephemeris queries (only required planets; Ketu never requested).
- Optional trimming of large payload pieces (spans / nested) via flags.
- Decimal used internally for stable proportional splits.

Public entrypoint:
    compute_chara_dasha(payload: dict) -> dict

Payload (route-style):
  Required for start_from="lagna" (unless asc_* provided):
    {date,time,tz,latitude,longitude} OR {jd_tt,jd_ut1,latitude,longitude}
    OR asc_sidereal_deg (or asc_tropical_deg + ayanamsa)

  Options:
    ayanamsa: str = "lahiri"
    start_from: "lagna" | "ak"
    include_rahu_in_karakas: bool = False
    direction_mode: "rashi_nature" | "odd_forward_even_reverse" | "uniform_forward"
    levels: int in [1..5] (default 5)
    year_days: float = 365.24219
    limit_jd_tt: float | None
    balance_years: float | None
    balance_fraction: float | None (0..1)
    override_start_sign_index: int in [1..12]
    planet_longitudes_sidereal: Optional[dict[str,float]] (precomputed)

    # Performance/payload knobs (all default to True for back-compat):
    include_spans: bool = True          # include flat "spans" array
    include_nested: bool = True         # include hierarchical "nested"
    include_tree: bool = True           # include "tree" envelope

Return (route-friendly):
  {
    ok, scheme: "chara",
    years_by_sign: {1:int,...,12:int},
    mahadasa_order: [12 ints],
    levels, year_days,
    spans?: [...],              # if include_spans
    nested?: [...],             # if include_nested
    tree?: {...},               # if include_tree
    start: {...},
    rules: {...},
    meta: {...}
  }
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple
from decimal import Decimal, getcontext
import math

# High-precision internal math for nested scaling
getcontext().prec = 34

# ─────────────────────────────────── guarded imports ───────────────────────────────────
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

# SAFE ephemeris wrapper (avoids direct adapter construction)
try:
    from app.core.ephemeris_adapter import ecliptic_longitudes  # type: ignore
    _EPH_OK = True
except Exception:
    ecliptic_longitudes = None  # type: ignore
    _EPH_OK = False

from app.core.ayanamsa import get_ayanamsa_deg

# ─────────────────────────────────── constants / helpers ───────────────────────────────────
SIGN_NAMES = (
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
)

# Traditional rashi lords (Parāśara), used by Jaimini for sign lordship
SIGN_LORD: Tuple[str, ...] = (
    "Mars","Venus","Mercury","Moon","Sun","Mercury",
    "Venus","Mars","Jupiter","Saturn","Saturn","Jupiter"
)

MOVABLE = {1,4,7,10}
FIXED   = {2,5,8,11}
DUAL    = {3,6,9,12}

_KARAKAS_7 = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn")
_KARAKAS_8 = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu")

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _sign_index(lon_deg: float) -> int:
    """1..12 (Aries=1)"""
    return int(math.floor(_norm360(lon_deg) / 30.0)) + 1

def _deg_within_sign(lon_deg: float) -> float:
    return _norm360(lon_deg) % 30.0

# ─────────────────────────────────── time helpers ───────────────────────────────────
def _timescales_from_civil(date: str, time: str, tz: str) -> Tuple[float, float, float]:
    """
    Returns (jd_ut, jd_tt, jd_ut1). Uses time_kernel if present, else app.core.timescales.
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

# ─────────────────────────────────── ephemeris wrappers ───────────────────────────────────
def _canon_planet(n: str) -> str:
    n = str(n).strip().lower()
    return n.capitalize()  # 'sun' -> 'Sun', 'rahu' -> 'Rahu'

def _planet_lons_sidereal(
    jd_tt: float,
    *,
    ay_key: str,
    preload: Optional[Dict[str, float]] = None,
    need: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """
    Return sidereal longitudes (deg) for requested planets.
    - 'need' limits ephemeris calls (defaults to Sun..Saturn + Moon).
    - 'preload' can include any case; values are assumed sidereal.
    """
    default_need = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn")
    want = { _canon_planet(x) for x in (need or default_need) }
    out: Dict[str, float] = {}

    # Case-insensitive preload
    if preload:
        for k, v in preload.items():
            ck = _canon_planet(k)
            if ck in want:
                try:
                    x = float(v)
                    if math.isfinite(x):
                        out[ck] = x
                except Exception:
                    pass

    missing = [p for p in want if p not in out]
    if missing:
        if not _EPH_OK or ecliptic_longitudes is None:
            raise RuntimeError("ephemeris_unavailable")
        rows = (ecliptic_longitudes(float(jd_tt), names=missing) or {}).get("results", [])
        got = {str(r.get("name")): float(r.get("longitude")) for r in rows if "name" in r and "longitude" in r}
        ay = float(get_ayanamsa_deg(float(jd_tt), ay_key))
        for k in missing:
            if k in got:
                out[k] = _norm360(got[k] - ay)

    return out

# ─────────────────────────────────── Ascendant helpers ───────────────────────────────────
def _asc_sidereal_deg(payload: Dict[str, Any], *, ay_key: str) -> float:
    """
    Resolve sidereal Ascendant longitude in degrees.
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

    # From civil if needed
    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
        if not (isinstance(payload.get("date"), str)
                and isinstance(payload.get("time"), str)
                and (payload.get("tz") or payload.get("place_tz"))):
            raise ValueError("To compute Ascendant, provide asc_* OR (date,time,tz,latitude,longitude)")
        tz = str(payload.get("tz") or payload.get("place_tz"))
        jd_ut, jd_tt, jd_ut1 = _timescales_from_civil(str(payload["date"]), str(payload["time"]), tz)
        lat = float(payload["latitude"]); lon = float(payload["longitude"])
    else:
        if not (isinstance(jd_tt, (int, float)) and isinstance(jd_ut1, (int, float))):
            if isinstance(payload.get("date"), str):
                tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
                _ju, jd_tt, jd_ut1 = _timescales_from_civil(str(payload["date"]), str(payload.get("time","12:00:00")), tz)
            else:
                raise ValueError("Provide jd_tt and jd_ut1 with latitude/longitude")

    calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=False, enable_validation=False)
    hd = calc.calculate_houses(latitude=float(lat),
                               longitude=float(lon),
                               jd_ut=float(jd_ut1),
                               jd_tt=float(jd_tt),
                               jd_ut1=float(jd_ut1))
    ay = float(get_ayanamsa_deg(float(jd_tt), ay_key))
    return _norm360(hd.ascendant - ay)

# ─────────────────────────────────── Karakas (Ātmakāraka) ───────────────────────────────────
def _atmakaraka_sign_index(planet_lons_sid: Dict[str, float], *, include_rahu: bool = False) -> int:
    """
    Ātmakāraka = planet with max longitude within its sign (0..30). 7-karaka by default.
    """
    cand = _KARAKAS_8 if include_rahu else _KARAKAS_7
    best_nm = None
    best_within = -1.0
    for nm in cand:
        lon = planet_lons_sid.get(nm)
        if lon is None:
            continue
        w = _deg_within_sign(float(lon))
        if w > best_within:
            best_within = w
            best_nm = nm
    if best_nm is None:
        raise RuntimeError("Unable to determine Ātmakāraka (no planet longitudes)")
    return _sign_index(float(planet_lons_sid[best_nm]))

# ─────────────────────────────────── direction & duration rules ───────────────────────────────────
def _dir_forward(sign_idx: int, *, mode: str) -> bool:
    mode = (mode or "rashi_nature").lower().strip()
    if mode == "rashi_nature":
        return False if sign_idx in FIXED else True  # movable & dual forward
    if mode == "odd_forward_even_reverse":
        return (sign_idx % 2) == 1
    if mode == "uniform_forward":
        return True
    raise ValueError(f"Unknown direction_mode: {mode}")

def _next_sign(sign_idx: int, *, forward: bool) -> int:
    return (1 if sign_idx == 12 else (sign_idx + 1)) if forward else (12 if sign_idx == 1 else (sign_idx - 1))

def _lord_sign_index_map(planets_sid: Dict[str, float]) -> Dict[int, int]:
    m: Dict[int, int] = {}
    for s in range(1, 13):
        lord = SIGN_LORD[s - 1]
        lon = planets_sid.get(lord)
        if lon is None:
            raise RuntimeError(f"Missing sidereal longitude for {lord}")
        m[s] = _sign_index(float(lon))
    return m

def _count_to_target_in_its_direction(start_sign: int, target_sign: int, *, forward_from_start: bool) -> int:
    """Inclusive count (1..12). Same sign → 12."""
    if start_sign == target_sign:
        return 12
    count = 1
    s = start_sign
    for _ in range(12):
        s = _next_sign(s, forward=forward_from_start)
        count += 1
        if s == target_sign:
            break
    return min(count, 12)

def _years_by_sign(planets_sid: Dict[str, float], *, direction_mode: str) -> Dict[int, int]:
    """For each sign S, years[S] = inclusive count from S to sign(lord(S)) along S's direction."""
    lord_pos = _lord_sign_index_map(planets_sid)
    years: Dict[int, int] = {}
    for s in range(1, 13):
        fwd = _dir_forward(s, mode=direction_mode)
        tgt = lord_pos[s]
        years[s] = _count_to_target_in_its_direction(s, tgt, forward_from_start=fwd)
    return years

@lru_cache(maxsize=64)
def _maha_sequence_cached(start_sign: int, direction_mode: str) -> Tuple[int, ...]:
    order: List[int] = []
    seen = set()
    s = int(start_sign)
    for _ in range(24):
        if s in seen:
            break
        order.append(s); seen.add(s)
        fwd = _dir_forward(s, mode=direction_mode)
        s = _next_sign(s, forward=fwd)
    if len(order) != 12:
        remaining = [i for i in range(1, 13) if i not in seen]
        order.extend(remaining)
        order = order[:12]
    return tuple(order)

# ─────────────────────────────────── data model ───────────────────────────────────
@dataclass(slots=True, frozen=True)
class DashaSpan:
    level: int                 # 1..5
    sign: int                  # 1..12
    start_jd_tt: float
    end_jd_tt: float

# ─────────────────────────────────── schedule core (O(N)) ───────────────────────────────────
def _years_to_days(years: Decimal, year_days: Decimal) -> Decimal:
    return years * year_days

def _child_duration(parent_days: Decimal, years_for_sign: int) -> Decimal:
    # Proportional partition; children sum to parent exactly
    return parent_days * (Decimal(years_for_sign) / Decimal(36))

def _sign_sequence_from(sign_idx: int, *, direction_mode: str) -> Tuple[int, ...]:
    """12-sign sequence starting at sign_idx; each step uses current sign's direction."""
    seq: List[int] = []
    seen = set()
    s = sign_idx
    for _ in range(24):
        if s in seen:
            break
        seq.append(s); seen.add(s)
        fwd = _dir_forward(s, mode=direction_mode)
        s = _next_sign(s, forward=fwd)
    if len(seq) != 12:
        for i in range(1, 13):
            if i not in seen:
                seq.append(i)
    return tuple(seq[:12])

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
    """
    Build nested nodes (level 1..L) and optional flat spans in a single pass.
    Returns (nested_nodes, spans).
    """
    year_days_D = Decimal(str(year_days))
    t_limit = Decimal(str(limit_jd_tt)) if isinstance(limit_jd_tt, (int, float)) else None

    def clip_end(t_next: Decimal) -> Decimal:
        return min(t_next, t_limit) if t_limit is not None else t_next

    all_spans: List[DashaSpan] = []
    nested_lvl1: List[Dict[str, Any]] = []

    def append_span(level: int, sign: int, t0: Decimal, t1: Decimal):
        if include_spans:
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

    # Level-1 order
    order = _maha_sequence_cached(int(start_sign_index), direction_mode)
    t = Decimal(str(jd_start_tt))

    # Build L1 and recurse
    for idx, s in enumerate(order):
        years = Decimal(years_by_sign[s])
        dur_days = _years_to_days(years, year_days_D)
        if idx == 0 and isinstance(balance_years, (int, float)):
            dur_days = max(Decimal(0), dur_days - _years_to_days(Decimal(str(balance_years)), year_days_D))

        t_next = t + dur_days
        if t_limit is not None and t >= t_limit:
            t = t_next
            break

        end = clip_end(t_next)
        node = mk_node(1, s, t, end)
        append_span(1, s, t, end)
        nested_lvl1.append(node)

        if levels >= 2:
            _expand_children(node, parent_sign=s, level_next=2, parent_start=t, parent_end=t_next)

        t = t_next

    # Recursive children expansion (proportional split; child sequence from parent sign)
    def _expand_children(node: Dict[str, Any], *, parent_sign: int, level_next: int, parent_start: Decimal, parent_end: Decimal):
        if not include_nested and not include_spans:
            return
        if level_next > levels:
            return
        parent_days = parent_end - parent_start
        if parent_days <= 0:
            return
        seq = _sign_sequence_from(parent_sign, direction_mode=direction_mode)
        t0 = Decimal(parent_start)
        for child_sign in seq:
            dur = _child_duration(parent_days, years_by_sign[child_sign])
            t1 = t0 + dur
            if t_limit is not None and t0 >= t_limit:
                t0 = t1
                continue
            end = clip_end(t1)
            append_span(level_next, child_sign, t0, end)
            if include_nested:
                child_node = mk_node(level_next, child_sign, t0, end)
                node.setdefault("children", []).append(child_node)
                _expand_children(child_node, parent_sign=child_sign, level_next=level_next+1, parent_start=t0, parent_end=t1)
            t0 = t1

    return nested_lvl1, all_spans

# ─────────────────────────────────── top-level schedule ───────────────────────────────────
def chara_schedule(
    *,
    jd_start_tt: float,
    start_sign_index: int,
    planets_sidereal: Dict[str, float],
    direction_mode: str = "rashi_nature",
    levels: int = 5,
    year_days: float = 365.24219,
    limit_jd_tt: float | None = None,
    balance_years: float | None = None,
    return_spans: bool = True,
    return_nested: bool = True,
    return_tree: bool = True,
) -> Dict[str, Any]:
    """
    Build the full Chara Daśā schedule.
    Flags:
      - return_spans / return_nested / return_tree allow trimming payload for speed.
    """
    levels = max(1, min(5, int(levels)))
    years = _years_by_sign(planets_sidereal, direction_mode=direction_mode)

    nested, spans = _build_tree_and_spans(
        jd_start_tt=float(jd_start_tt),
        start_sign_index=int(start_sign_index),
        years_by_sign=years,
        direction_mode=direction_mode,
        levels=levels,
        year_days=float(year_days),
        limit_jd_tt=limit_jd_tt,
        balance_years=balance_years,
        include_spans=bool(return_spans),
        include_nested=bool(return_nested),
    )

    # Envelope bounds
    if nested:
        s0 = min(float(n.get("start_jd_tt", 0.0)) for n in nested)
        e1 = max(float(n.get("end_jd_tt", 0.0)) for n in nested)
    elif spans:
        s0 = min(s.start_jd_tt for s in spans); e1 = max(s.end_jd_tt for s in spans)
    else:
        s0 = float(jd_start_tt); e1 = float(jd_start_tt)

    out: Dict[str, Any] = {
        "ok": True,
        "scheme": "chara",
        "years_by_sign": {i: int(years[i]) for i in range(1, 13)},
        "mahadasa_order": list(_maha_sequence_cached(int(start_sign_index), direction_mode)),
        "levels": int(levels),
        "year_days": float(year_days),
    }

    if return_spans:
        out["spans"] = [
            {
                "level": s.level,
                "sign_index": s.sign,
                "sign_name": SIGN_NAMES[s.sign - 1],
                "start_jd_tt": s.start_jd_tt,
                "end_jd_tt": s.end_jd_tt,
            } for s in spans
        ]
    if return_nested:
        out["nested"] = nested
    if return_tree:
        out["tree"] = {
            "level": 0,
            "lord": None,
            "label": "chara",
            "start_jd_tt": s0,
            "end_jd_tt": e1,
            "children": nested if return_nested else [],
        }
    return out

# ─────────────────────────────────── Orchestrator (route-friendly) ───────────────────────────────────
def compute_chara_dasha(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build Chara Daśā from a route-style payload (see module docstring).
    """
    try:
        ay_key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
        direction_mode = str(payload.get("direction_mode", "rashi_nature")).strip().lower()
        levels = int(payload.get("levels", 5))
        year_days = float(payload.get("year_days", 365.24219))

        # Payload knobs (default True for back-compat; set to False client-side for speed)
        include_spans = bool(payload.get("include_spans", True))
        include_nested = bool(payload.get("include_nested", True))
        include_tree = bool(payload.get("include_tree", True))

        # Limit
        lim = payload.get("limit_jd_tt")
        limit_jd_tt = float(lim) if isinstance(lim, (int, float)) else None

        # Resolve jd_tt; only compute timescales if needed
        jd_tt = payload.get("jd_tt")
        jd_ut1 = payload.get("jd_ut1")
        if not isinstance(jd_tt, (int, float)):
            if isinstance(payload.get("date"), str):
                tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
                _ju, jd_tt, jd_ut1 = _timescales_from_civil(str(payload["date"]), str(payload.get("time","12:00:00")), tz)
            else:
                raise ValueError("jd_tt or (date,time,tz) required")

        # Determine which planets we actually need from ephemeris
        need_planets: List[str] = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
        start_from = str(payload.get("start_from", "lagna")).strip().lower()
        include_rahu = bool(payload.get("include_rahu_in_karakas", False))
        if start_from == "ak":
            if include_rahu:
                need_planets.append("Rahu")  # only AK logic needs Rahu

        # Planet longitudes (sidereal); use preload if available
        preload = payload.get("planet_longitudes_sidereal")
        planets_sid = _planet_lons_sidereal(
            float(jd_tt),
            ay_key=ay_key,
            preload=preload if isinstance(preload, dict) else None,
            need=need_planets,
        )

        # Starting sign
        start_override = payload.get("override_start_sign_index")
        if isinstance(start_override, int) and 1 <= int(start_override) <= 12:
            start_sign = int(start_override)
            start_basis = "override"
        else:
            if start_from == "ak":
                ak_sign = _atmakaraka_sign_index(planets_sid, include_rahu=include_rahu)
                start_sign = ak_sign
                start_basis = "karakamsha"
            else:
                # Only compute Ascendant if needed
                asc_sid = _asc_sidereal_deg(payload, ay_key=ay_key)
                start_sign = _sign_index(asc_sid)
                start_basis = "lagna"

        # First-mahā balance (optional)
        balance_years: Optional[float] = None
        if isinstance(payload.get("balance_years"), (int, float)):
            balance_years = float(payload["balance_years"])
        elif isinstance(payload.get("balance_fraction"), (int, float)):
            frac = max(0.0, min(0.999999, float(payload["balance_fraction"])))
            # need years_by_sign for the first sign
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
            return_spans=include_spans,
            return_nested=include_nested,
            return_tree=include_tree,
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
            "sublevel_scaling": "proportional (years/36)",
        }
        sched["meta"] = {
            "ayanamsa": ay_key,
            "asc_sidereal_deg": (float(payload.get("asc_sidereal_deg"))
                                 if isinstance(payload.get("asc_sidereal_deg"), (int, float)) else None),
        }
        return sched

    except Exception as e:
        return {"ok": False, "error": f"chara_failed:{e}"}
