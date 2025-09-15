# app/core/chara_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Chara Daśā (Jaimini) — sign-based daśā with five nested levels
Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa

Numerical / integration policy (aligned with your gold standard):
- Sidereal-first pipeline (ayanāṁśa subtraction via app.core.ayanamsa.get_ayanamsa_deg).
- High-precision timescales: time_kernel if present, else app.core.timescales.
- Ascendant from app.core.houses_advanced (PyERFA GAST + true obliquity).
- Planet longitudes via safe wrapper app.core.ephemeris_adapter.ecliptic_longitudes (ecliptic-of-date),
  then siderealized using chosen ayanāṁśa.
- Deterministic schedule using Decimal internally for nested scaling (minimizes drift).
- Full 5-level expansion; configurable direction and duration rules.

Default rules implemented (configurable):
- Start sign: Lagna sign (sidereal). Optionally start from Karakamśa (Ātmakāraka’s sign).
- Direction per sign (“rashi_nature”):
    movable (Ar, Cn, Li, Cp)   → forward (+1)
    fixed   (Ta, Le, Sc, Aq)   → reverse (−1)
    dual    (Ge, Vi, Sg, Pi)   → forward (+1)
  (Alternate modes: "odd_forward_even_reverse", "uniform_forward".)
- Mahādaśā duration for sign S: count, in S’s own direction, from S to the sign
  occupied by S’s traditional lord (Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn).
  Inclusive count with same-sign → 12 years.
- Subperiod scaling (levels 2..5): proportional partition
  child_duration = parent_duration × (years(child_sign) / 36).
  (Keeps each level summing exactly to its parent; avoids calendar mismatch.)
- Antar start within each parent: begins from the parent’s sign and proceeds by that child-sign’s direction at each nesting.
- Optional balance at birth for the very first mahādaśā: supply `balance_years` or `balance_fraction` (0..1)
  to reduce the first span accordingly (default: no balance).

Public API
----------
compute_chara_dasha(payload: dict) -> dict
  Inputs (provide either jd_tt directly OR civil date/time/tz plus site):
    Required for Lagna start unless asc provided:
      - date="YYYY-MM-DD", time="HH:MM[:SS]", tz="Area/City", latitude, longitude
    OR
      - jd_tt, jd_ut1, latitude, longitude
    OR
      - asc_sidereal_deg (or asc_tropical_deg + ayanamsa)
    For Karakamśa start:
      - jd_tt (or date/time/tz), ephemeris available.

  Options:
    - ayanamsa: str, default "lahiri"
    - start_from: "lagna" (default) | "ak"  (Karakamśa)
    - include_rahu_in_karakas: bool = False  (7-karaka by default)
    - direction_mode: "rashi_nature" (default) | "odd_forward_even_reverse" | "uniform_forward"
    - levels: int 1..5 (default 5)
    - year_days: float = 365.24219
    - limit_jd_tt: float | None   (truncate schedule)
    - balance_years: float | None  (only applied to first mahādaśā)
    - balance_fraction: float | None in [0,1)
    - override_start_sign_index: int 1..12 (overrides start_from logic)
    - planet_longitudes_sidereal: optional precomputed dict {name: deg} to avoid fresh ephemeris

Return (route-friendly):
  {
    ok, scheme, start: {...}, rules: {...}, years_by_sign: {...},
    spans: [ {level, sign_index, sign_name, start_jd_tt, end_jd_tt} ... ],
    nested: [ {level, sign_index, sign_name, start_jd_tt, end_jd_tt, children:[...]} ... ],
    tree: {level:0,label:"chara",start_jd_tt,end_jd_tt,children:[...]},
    meta: {...}
  }
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal, getcontext
import math

# High-precision internal math for nested scaling
getcontext().prec = 34

# ─────────────────────────────────── imports (optional/guarded) ───────────────────────────────────
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

# SAFE ephemeris wrapper (no direct adapter construction)
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

# Traditional lords (Jaimini = Parāśara rashi lords; nodes not lords)
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

# ─────────────────────────────────── time helpers ───────────────────────────────────
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

# ─────────────────────────────────── ephemeris wrappers ───────────────────────────────────
def _planet_lons_sidereal(
    jd_tt: float,
    *,
    ay_key: str,
    preload: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Returns sidereal longitudes (deg) for required planets.
    Accepts an optional preload {name: deg} (already sidereal).
    """
    want = {"Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu"}
    out: Dict[str, float] = {}
    if preload:
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

    return out

# ─────────────────────────────────── Ascendant helpers ───────────────────────────────────
def _asc_sidereal_deg(payload: Dict[str, Any], *, ay_key: str) -> float:
    """
    Resolve sidereal Ascendant longitude in degrees from payload ingredients.
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

# ─────────────────────────────────── Karakas (Ātmakāraka) ───────────────────────────────────
def _atmakaraka_sign_index(planet_lons_sid: Dict[str, float], *, include_rahu: bool = False) -> int:
    """
    Ātmakāraka = planet with max longitude within its sign (0..30), ties broken by full precision.
    Default: 7-karaka (exclude Rahu/Ketu). Optionally include Rahu (8-karaka).
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

# ─────────────────────────────────── direction & duration rules ───────────────────────────────────
def _dir_forward(sign_idx: int, *, mode: str) -> bool:
    mode = (mode or "rashi_nature").lower().strip()
    if mode == "rashi_nature":
        if sign_idx in FIXED:
            return False
        return True  # movable & dual forward
    if mode == "odd_forward_even_reverse":
        return (sign_idx % 2) == 1
    if mode == "uniform_forward":
        return True
    raise ValueError(f"Unknown direction_mode: {mode}")

def _next_sign(sign_idx: int, *, forward: bool) -> int:
    if forward:
        return 1 if sign_idx == 12 else (sign_idx + 1)
    else:
        return 12 if sign_idx == 1 else (sign_idx - 1)

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
    """
    Inclusive count (1..12). Same sign => 12.
    """
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

def _years_by_sign(planets_sid: Dict[str, float], *, direction_mode: str) -> Dict[int, int]:
    """
    For each sign S, duration_years[S] = inclusive sign count from S to sign(lord(S)),
    counting in the direction determined by S per direction_mode.
    """
    lord_pos = _lord_sign_index_map(planets_sid)
    years: Dict[int, int] = {}
    for s in range(1, 13):
        fwd = _dir_forward(s, mode=direction_mode)
        tgt = lord_pos[s]
        years[s] = _count_to_target_in_its_direction(s, tgt, forward_from_start=fwd)
    return years

def _maha_sequence(start_sign: int, *, direction_mode: str) -> List[int]:
    """
    Build the 12-sign mahādaśā sequence by stepping 1 sign each time,
    using the direction dictated by the CURRENT sign.
    """
    order: List[int] = []
    seen = set()
    s = start_sign
    for _ in range(24):  # safety
        if s in seen:
            break
        order.append(s); seen.add(s)
        fwd = _dir_forward(s, mode=direction_mode)
        s = _next_sign(s, forward=fwd)
    if len(order) != 12:
        remaining = [i for i in range(1, 13) if i not in seen]
        order.extend(remaining)
        order = order[:12]
    return order

# ─────────────────────────────────── data model ───────────────────────────────────
@dataclass
class DashaSpan:
    level: int                 # 1..5
    sign: int                  # 1..12
    start_jd_tt: float
    end_jd_tt: float

# ─────────────────────────────────── schedule core ───────────────────────────────────
def _append_span(
    spans: List[DashaSpan],
    level: int,
    sign: int,
    t0: Decimal,
    dur_days: Decimal,
    *,
    limit: Optional[Decimal]
) -> Decimal:
    t1 = t0 + dur_days
    if limit is not None and t0 >= limit:
        return t1
    end = min(t1, limit) if limit is not None else t1
    spans.append(DashaSpan(level, sign, float(t0), float(end)))
    return t1

def _years_to_days(years: Decimal, year_days: Decimal) -> Decimal:
    return years * year_days

def _sub_duration(parent_days: Decimal, years_for_sign: int) -> Decimal:
    # Proportional partition (keeps children summing to parent)
    return parent_days * (Decimal(years_for_sign) / Decimal(36))

def _expand_children_for_parent(
    parent: DashaSpan,
    *,
    level_next: int,
    direction_mode: str,
    years_by_sign: Dict[int, int],
    t_limit: Optional[Decimal]
) -> List[DashaSpan]:
    parent_days = Decimal(str(parent.end_jd_tt)) - Decimal(str(parent.start_jd_tt))
    # Child sequence starts at the parent's sign; each child advances by ITS OWN direction
    seq: List[int] = []
    s = parent.sign
    seen = set()
    for _ in range(12):
        if s in seen:
            break
        seq.append(s); seen.add(s)
        fwd = _dir_forward(s, mode=direction_mode)
        s = _next_sign(s, forward=fwd)
    if len(seq) != 12:
        for i in range(1, 13):
            if i not in seen:
                seq.append(i)
    t = Decimal(str(parent.start_jd_tt))
    kids: List[DashaSpan] = []
    for child_sign in seq[:12]:
        dur = _sub_duration(parent_days, years_by_sign[child_sign])
        t_next = t + dur
        if t_limit is not None and t >= t_limit:
            t = t_next
            continue
        end = min(t_next, t_limit) if t_limit is not None else t_next
        kids.append(DashaSpan(level_next, child_sign, float(t), float(end)))
        t = t_next
    return kids

def _to_nested(spans: List[DashaSpan], *, max_level: int) -> List[Dict[str, Any]]:
    def children_of(p: DashaSpan, lvl: int) -> List[DashaSpan]:
        eps = 1e-12
        return [
            s for s in spans
            if s.level == lvl
            and p.start_jd_tt - eps <= s.start_jd_tt <= p.end_jd_tt + eps
            and s.end_jd_tt <= p.end_jd_tt + eps
        ]

    def node_for(s: DashaSpan, lvl: int) -> Dict[str, Any]:
        node = {
            "level": lvl,
            "sign_index": s.sign,
            "sign_name": SIGN_NAMES[s.sign - 1],
            "start_jd_tt": s.start_jd_tt,
            "end_jd_tt": s.end_jd_tt,
        }
        if lvl < max_level:
            node["children"] = [node_for(k, lvl + 1) for k in children_of(s, lvl + 1)]
        return node

    return [node_for(s, 1) for s in spans if s.level == 1]

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
) -> Dict[str, Any]:
    """
    Build the full Chara Daśā schedule.
    """
    levels = max(1, min(5, int(levels)))
    year_days_D = Decimal(str(year_days))
    t0 = Decimal(str(jd_start_tt))
    t_limit = Decimal(str(limit_jd_tt)) if isinstance(limit_jd_tt, (int, float)) else None

    # Years per sign and mahā sequence
    yrs = _years_by_sign(planets_sidereal, direction_mode=direction_mode)
    order = _maha_sequence(start_sign_index, direction_mode=direction_mode)

    # Compose mahā spans
    spans: List[DashaSpan] = []
    t = t0
    for idx, s in enumerate(order):
        years = Decimal(yrs[s])
        dur_days = _years_to_days(years, year_days_D)
        if idx == 0 and isinstance(balance_years, (int, float)):
            reduce_days = _years_to_days(Decimal(str(balance_years)), year_days_D)
            dur_days = max(Decimal(0), dur_days - reduce_days)
        t = _append_span(spans, 1, s, t, dur_days, limit=t_limit)

    # Expand sublevels
    current = [sp for sp in spans if sp.level == 1]
    all_spans = list(spans)
    for level in range(2, levels + 1):
        nxt: List[DashaSpan] = []
        for p in current:
            nxt.extend(_expand_children_for_parent(
                p, level_next=level, direction_mode=direction_mode,
                years_by_sign=yrs, t_limit=t_limit
            ))
        all_spans.extend(nxt)
        current = nxt

    all_spans.sort(key=lambda s: (s.level, s.start_jd_tt, s.sign))
    nested = _to_nested(all_spans, max_level=levels)

    # Tree envelope for route-friendliness
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
        "mahadasa_order": order,
        "levels": int(levels),
        "year_days": float(year_days),
        "spans": [
            {
                "level": s.level,
                "sign_index": s.sign,
                "sign_name": SIGN_NAMES[s.sign - 1],
                "start_jd_tt": s.start_jd_tt,
                "end_jd_tt": s.end_jd_tt
            } for s in all_spans
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

# ─────────────────────────────────── Orchestrator (route-friendly) ───────────────────────────────────
def compute_chara_dasha(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build Chara Daśā from a route-style payload.

    Accepted inputs (any one path):
      A) Provide asc_sidereal_deg (or asc_tropical_deg + ayanamsa) and jd_tt
      B) Provide jd_tt & jd_ut1 & latitude & longitude
      C) Provide date/time/tz & latitude & longitude

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
            float(jd_tt),
            ay_key=ay_key,
            preload=preload if isinstance(preload, dict) else None
        )

        # Starting sign:
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

        # Add metadata
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
