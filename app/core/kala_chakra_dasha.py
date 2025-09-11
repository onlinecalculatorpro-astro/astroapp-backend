# app/core/kala_chakra_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Kālachakra Daśā — pluggable, research-grade, five-level scheduler
Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa

Why pluggable?
- Kālachakra has multiple lineage variants (BPHS editions, K.N. Rao school, etc.).
- Two things vary across sources: (1) pada→rāśi sequence (12-sign order), (2) per-rāśi year counts.
- To maintain a gold-standard implementation, we separate the *engine* from the *tables*.
  You must supply the mapping you certify against (see `kcd_table`).

Engine features:
- Sidereal-first numerics (ayanāṁśa subtraction via app.core.ayanamsa.get_ayanamsa_deg).
- Timescales: time_kernel preferred; fallback to app.core.timescales (TT/UT1 handled).
- Nakṣatra/pada from Moon’s sidereal longitude (high-precision EphemerisAdapter).
- Deterministic Decimal math for nested partitions (minimizes drift).
- Full 5-level nesting. Children start from the parent’s rāśi and follow that scheme’s 12-sign sequence.
- Child duration = parent_duration × (years(child_rāśi) / total_years_per_cycle).
  (Keeps exact closure to the parent span.)
- Optional first-daśā balance by years or fraction.

Inputs (route-friendly `compute_kalachakra_dasha`):
- Provide either:
  A) jd_tt (± jd_ut1) + latitude/longitude, or
  B) date, time, tz + latitude/longitude, or
  C) Precomputed moon sidereal longitude (rare; see `planet_longitudes_sidereal`).
- ayanamsa: str key (default "lahiri").
- kcd_table: dict with two keys:
    "pada_to_sequence": dict[int(1..108)] -> list[int length 12, each 1..12]  # full rāśi order for that pada
    "sign_years": dict[int(1..12)] -> int/float  # years per rāśi, as per lineage
  Optionally, "name": str for provenance.
- Optional:
    levels (1..5, default 5), year_days (default 365.24219),
    limit_jd_tt (truncate schedule),
    balance_years or balance_fraction (applies to first Mahā),
    override_start_sign_index (force start rāśi),
    planet_longitudes_sidereal (pre-load to avoid fresh ephemeris call).

Return:
{
  ok: bool,
  scheme: "kalachakra",
  kcd_table_name: str|None,
  start: { nakshatra: {...}, pada: 1..4, sign_index: 1..12, sign_name: str, basis: "pada_table"|... },
  rules: {...},
  sign_years: {1:years,...,12:years},
  spans: [ {level, sign_index, sign_name, start_jd_tt, end_jd_tt}, ... ],
  nested: [ tree with children ... ],
  meta: {...}
}

NOTE
- This module raises a clear error if `kcd_table` is not supplied. To activate production mode,
  pass your curated tables (or implement `load_kcd_table_preset` with your canonical mapping).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal, getcontext
import math

# High-precision nested math
getcontext().prec = 34

# ───────────────────────── optional deps (guarded) ─────────────────────────

try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None

try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    from app.core.ephem_singleton import TS, PLANETS
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    TS = None; PLANETS = None
    _EPH_OK = False

from app.core.ayanamsa import get_ayanamsa_deg

SIGN_NAMES = (
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
)

# ───────────────────────── small math helpers ─────────────────────────

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

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

# ───────────────────────── ephemeris helpers ─────────────────────────

def _moon_sidereal_longitude(jd_tt: float, *, ay_key: str, preload_sidereal: Optional[Dict[str, float]] = None) -> float:
    """
    Return Moon's sidereal longitude in degrees (0..360).
    If preload_sidereal contains "Moon" (already sidereal), it is used.
    """
    if preload_sidereal and isinstance(preload_sidereal.get("Moon"), (int,float)):
        return _norm360(float(preload_sidereal["Moon"]))
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")
    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore
    rows = ep.ecliptic_longitudes(float(jd_tt), ["Moon"]).get("results", [])
    if not rows:
        raise RuntimeError("ephemeris returned no Moon longitude")
    trop = float(rows[0]["longitude"])
    ay = float(get_ayanamsa_deg(float(jd_tt), ay_key))
    return _norm360(trop - ay)

def _nak_pada_from_moon_sidereal(moon_lon_sid: float) -> Tuple[int, int, float]:
    """
    Returns (nak_index 1..27, pada 1..4, offset_within_pada_fraction 0..1).
    """
    width = 360.0 / 27.0                      # 13°20'
    idx0 = int(math.floor(_norm360(moon_lon_sid) / width))   # 0..26
    nak = idx0 + 1
    pos_in = _norm360(moon_lon_sid) - idx0 * width
    pada_float = (pos_in / width) * 4.0
    pada = int(math.floor(pada_float)) + 1     # 1..4
    frac_in_pada = max(0.0, min(1.0, pada_float - (pada - 1)))
    return nak, pada, frac_in_pada

def _pada_linear_index(nak: int, pada: int) -> int:
    """
    Map (nak 1..27, pada 1..4) -> 1..108
    """
    return (nak - 1) * 4 + pada

# ───────────────────────── data model ─────────────────────────

@dataclass
class DashaSpan:
    level: int          # 1..5
    sign: int           # 1..12
    start_jd_tt: float
    end_jd_tt: float

# ───────────────────────── table loader / guardrails ─────────────────────────

def load_kcd_table_preset(name: str) -> Dict[str, Any]:
    """
    Hook to return a canonical KCD table by name, if you want to hardwire one.
    The table must be:
      {
        "name": "label",
        "pada_to_sequence": { 1:[...12 ints...], 2:[...], ..., 108:[...] },
        "sign_years": {1:int,2:int,...,12:int}
      }
    By default we do NOT ship a preset to avoid embedding a contested variant.
    """
    raise NotImplementedError(f"KCD preset '{name}' not bundled. Supply kcd_table explicitly.")

def _validate_kcd_table(tbl: Dict[str, Any]) -> None:
    if not isinstance(tbl, dict):
        raise ValueError("kcd_table must be a dict")
    seq = tbl.get("pada_to_sequence")
    yrs = tbl.get("sign_years")
    if not isinstance(seq, dict) or not isinstance(yrs, dict):
        raise ValueError("kcd_table must contain 'pada_to_sequence' and 'sign_years'")
    if len(yrs) != 12 or any(s not in yrs for s in range(1,13)):
        raise ValueError("kcd_table['sign_years'] must define 12 entries keyed 1..12")
    for k, v in seq.items():
        if not (1 <= int(k) <= 108):
            raise ValueError("kcd_table['pada_to_sequence'] keys must be 1..108")
        if not (isinstance(v, (list, tuple)) and len(v) == 12 and all(1 <= int(x) <= 12 for x in v)):
            raise ValueError("each 'pada_to_sequence'[k] must be a 12-length list of 1..12")

# ───────────────────────── schedule builder ─────────────────────────

def _years_total(yrs: Dict[int, float|int]) -> Decimal:
    s = Decimal(0)
    for i in range(1,13):
        s += Decimal(str(yrs[i]))
    return s

def _append_span(spans: List[DashaSpan], level: int, sign: int, t0: Decimal, dur_days: Decimal,
                 *, limit: Optional[Decimal]) -> Decimal:
    t1 = t0 + dur_days
    if limit is not None and t0 >= limit:
        return t1
    end = min(t1, limit) if limit is not None else t1
    spans.append(DashaSpan(level, sign, float(t0), float(end)))
    return t1

def _expand_children(parent: DashaSpan, *, level_next: int, seq12: List[int],
                     yrs: Dict[int, float|int], total_years: Decimal, t_limit: Optional[Decimal]) -> List[DashaSpan]:
    parent_days = Decimal(str(parent.end_jd_tt)) - Decimal(str(parent.start_jd_tt))
    # Child sequence rotates so it begins with parent's sign, then follows seq12 order cyclically
    if parent.sign not in seq12:
        # Defensive: build a safe rotation even if seq12 malformed
        base = list(range(1,13))
    else:
        idx = seq12.index(parent.sign)
        base = seq12[idx:] + seq12[:idx]
    t = Decimal(str(parent.start_jd_tt))
    kids: List[DashaSpan] = []
    for s in base:
        share = Decimal(str(yrs[s])) / total_years
        dur = parent_days * share
        t_next = t + dur
        if t_limit is not None and t >= t_limit:
            t = t_next
            continue
        end = min(t_next, t_limit) if t_limit is not None else t_next
        kids.append(DashaSpan(level_next, int(s), float(t), float(end)))
        t = t_next
    return kids

def kalachakra_schedule(
    *,
    jd_start_tt: float,
    sequence12: List[int],
    sign_years: Dict[int, float|int],
    levels: int = 5,
    year_days: float = 365.24219,
    limit_jd_tt: float | None = None,
    balance_years: float | None = None,
) -> Dict[str, Any]:
    """
    Build KCD schedule from a 12-sign order and a per-sign year table.
    """
    levels = max(1, min(5, int(levels)))
    year_days_D = Decimal(str(year_days))
    t0 = Decimal(str(jd_start_tt))
    t_limit = Decimal(str(limit_jd_tt)) if isinstance(limit_jd_tt, (int,float)) else None

    # Sanity
    if not (isinstance(sequence12, (list, tuple)) and len(sequence12) == 12 and all(1 <= int(x) <= 12 for x in sequence12)):
        raise ValueError("sequence12 must be a 12-length list of sign indices 1..12")
    _ = _validate_kcd_table({"pada_to_sequence": {1: sequence12}, "sign_years": sign_years})  # reuse checks

    total_years = _years_total(sign_years)

    # Level-1 (Mahā)
    spans: List[DashaSpan] = []
    t = t0
    for idx, s in enumerate(sequence12):
        y = Decimal(str(sign_years[int(s)]))
        dur_days = y * year_days_D
        if idx == 0 and isinstance(balance_years, (int,float)):
            reduce_days = Decimal(str(balance_years)) * year_days_D
            dur_days = max(Decimal(0), dur_days - reduce_days)
        t = _append_span(spans, 1, int(s), t, dur_days, limit=t_limit)

    # Expand sub-levels
    all_spans = list(spans)
    current = [sp for sp in spans if sp.level == 1]
    for lvl in range(2, levels + 1):
        nxt: List[DashaSpan] = []
        for p in current:
            nxt.extend(_expand_children(
                p, level_next=lvl, seq12=sequence12,
                yrs=sign_years, total_years=total_years, t_limit=t_limit
            ))
        all_spans.extend(nxt)
        current = nxt

    all_spans.sort(key=lambda s: (s.level, s.start_jd_tt, s.sign))

    # Nested tree
    def children_of(par: DashaSpan, lvl: int) -> List[DashaSpan]:
        eps = 1e-12
        return [s for s in all_spans if s.level == lvl and par.start_jd_tt - eps <= s.start_jd_tt <= par.end_jd_tt + eps and s.end_jd_tt <= par.end_jd_tt + eps]

    def node_for(s: DashaSpan, lvl: int) -> Dict[str, Any]:
        node = {
            "level": lvl,
            "sign_index": s.sign,
            "sign_name": SIGN_NAMES[s.sign-1],
            "start_jd_tt": s.start_jd_tt,
            "end_jd_tt": s.end_jd_tt,
        }
        if lvl < levels:
            node["children"] = [node_for(k, lvl+1) for k in children_of(s, lvl+1)]
        return node

    tree = [node_for(s, 1) for s in all_spans if s.level == 1]

    return {
        "ok": True,
        "scheme": "kalachakra",
        "sign_years": {i: float(sign_years[i]) for i in range(1,13)},
        "spans": [
            {"level": s.level, "sign_index": s.sign, "sign_name": SIGN_NAMES[s.sign-1],
             "start_jd_tt": s.start_jd_tt, "end_jd_tt": s.end_jd_tt}
            for s in all_spans
        ],
        "nested": tree,
    }

# ───────────────────────── orchestrator ─────────────────────────

def compute_kalachakra_dasha(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route-friendly builder.

    Required: a KCD table (either directly in payload['kcd_table'] or via preset name in payload['kcd_preset']).

    Steps:
      1) Resolve jd_tt (from args or civil).
      2) Get Moon sidereal longitude (from ephemeris or preload).
      3) Compute nakshatra & pada (1..27, 1..4) → pada_idx (1..108).
      4) From supplied kcd_table:
             sequence12 = table['pada_to_sequence'][pada_idx]
             sign_years = table['sign_years']
         (Optionally, override starting sign with 'override_start_sign_index'.)
      5) Build schedule with full nesting and optional balance.

    Payload keys:
      - date, time, tz, latitude, longitude  (or jd_tt (+ jd_ut1))
      - ayanamsa: str = "lahiri"
      - kcd_table: dict (see header)  OR kcd_preset: str (if you implement load_kcd_table_preset)
      - levels: 1..5 (default 5)
      - year_days: float (default 365.24219)
      - limit_jd_tt: float | None
      - balance_years: float | None
      - balance_fraction: float | None  (fraction of first Mahā years to shave)
      - override_start_sign_index: int 1..12
      - planet_longitudes_sidereal: optional dict { "Moon": deg, ... } (already sidereal)

    Returns schedule dict (see header). On configuration issues, returns {ok: False, error: "..."}.
    """
    try:
        ay_key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
        levels = int(payload.get("levels", 5))
        year_days = float(payload.get("year_days", 365.24219))
        limit = payload.get("limit_jd_tt")
        limit_jd_tt = float(limit) if isinstance(limit, (int,float)) else None

        # Resolve jd_tt
        jd_tt = payload.get("jd_tt")
        jd_ut1 = payload.get("jd_ut1")
        if not isinstance(jd_tt, (int, float)):
            if isinstance(payload.get("date"), str):
                tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
                _ju, jd_tt, jd_ut1 = _timescales_from_civil(str(payload["date"]), str(payload.get("time","12:00:00")), tz)
            else:
                return {"ok": False, "error": "jd_tt_or_civil_required"}

        # KCD table
        table = payload.get("kcd_table")
        if table is None and isinstance(payload.get("kcd_preset"), str):
            table = load_kcd_table_preset(str(payload["kcd_preset"]))
        if table is None:
            return {"ok": False, "error": "kcd_table_required"}
        _validate_kcd_table(table)
        table_name = str(table.get("name") or "")

        # Moon lon sidereal → nak/pada
        preload = payload.get("planet_longitudes_sidereal")
        moon_sid = _moon_sidereal_longitude(float(jd_tt), ay_key=ay_key, preload_sidereal=preload if isinstance(preload, dict) else None)
        nak, pada, frac_in_pada = _nak_pada_from_moon_sidereal(moon_sid)
        pada_idx = _pada_linear_index(nak, pada)

        # Sequence & per-sign years
        seq12 = table["pada_to_sequence"].get(int(pada_idx))
        if not seq12:
            return {"ok": False, "error": f"no_sequence_for_pada_{pada_idx}"}
        sign_years = {int(k): float(v) for k, v in table["sign_years"].items()}

        # Override start sign (rare)
        start_override = payload.get("override_start_sign_index")
        if isinstance(start_override, int) and 1 <= int(start_override) <= 12:
            s = int(start_override)
            if s in seq12:
                # rotate sequence to start at the override sign
                i = seq12.index(s)
                seq12 = seq12[i:] + seq12[:i]
                start_basis = "override"
            else:
                # force by placing it first then the rest in original order without dup
                seq12 = [s] + [x for x in seq12 if x != s]
                start_basis = "override"
        else:
            start_basis = "pada_table"

        # Balance for first mahā
        balance_years = None
        if isinstance(payload.get("balance_years"), (int,float)):
            balance_years = float(payload["balance_years"])
        elif isinstance(payload.get("balance_fraction"), (int,float)):
            frac = max(0.0, min(0.999999, float(payload["balance_fraction"])))
            first_years = float(sign_years[seq12[0]])
            balance_years = first_years * frac

        sched = kalachakra_schedule(
            jd_start_tt=float(jd_tt),
            sequence12=list(map(int, seq12)),
            sign_years=sign_years,
            levels=int(levels),
            year_days=float(year_days),
            limit_jd_tt=limit_jd_tt,
            balance_years=balance_years,
        )

        # Attach start & rules & meta
        sched["kcd_table_name"] = table_name or None
        sched["start"] = {
            "nakshatra_index": int(nak),
            "pada": int(pada),
            "pada_linear_index": int(pada_idx),
            "start_sign_index": int(seq12[0]),
            "start_sign_name": SIGN_NAMES[seq12[0]-1],
            "basis": start_basis,
        }
        sched["rules"] = {
            "child_order": "rotated: start at parent sign, follow scheme’s 12-sign sequence",
            "child_duration": "proportional to sign_years / total_years_per_cycle",
            "levels": int(levels),
        }
        sched["meta"] = {
            "ayanamsa": ay_key,
            "year_days": float(year_days),
            "limit_jd_tt": limit_jd_tt,
        }
        return sched

    except NotImplementedError as e:
        return {"ok": False, "error": f"not_implemented:{e}"}
    except Exception as e:
        return {"ok": False, "error": f"kalachakra_failed:{e}"}
