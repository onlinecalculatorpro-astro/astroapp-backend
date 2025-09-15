# app/core/kala_chakra_dasha.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Kālachakra Daśā — pluggable, research-grade, five-level scheduler
Mahā → Antar → Pratyantar → Sūkṣma → Prāṇa

Why pluggable?
- Kālachakra has lineage variants (BPHS editions, K.N. Rao school, etc.).
- Two knobs vary across sources: (1) pada→rāśi sequence (12-sign order), (2) per-rāśi year counts.
- To stay canonical, the *engine* is separate from the *tables* — you supply `kcd_table`.

Engine highlights
- Sidereal-first numerics (ayanāṁśa subtraction via app.core.ayanamsa.get_ayanamsa_deg).
- Timescales: time_kernel preferred; fallback to app.core.timescales.
- Moon sidereal longitude → nakṣatra/pada via guarded EphemerisAdapter.
- Deterministic Decimal math for partitions; exact child-sum closure to parent (float output only at the edges).
- Full 5-level nesting. Children start at parent’s rāśi and follow the scheme’s 12-sign sequence.
- Child duration = parent_days × (years(child_rāśi) / total_years_per_cycle).
- Optional balance for first mahā by absolute years or fraction.

Public API (route-friendly)
compute_kalachakra_dasha(payload: dict) -> dict:
  Returns:
    {
      ok, scheme: "kalachakra",
      kcd_table_name,
      start: {nakshatra_index, pada, pada_linear_index, start_sign_index, start_sign_name, basis},
      rules: {...},
      sign_years: {1:years,...,12:years},
      mahadasa_order: [12 ints],
      spans: [{level, sign_index, sign_name, start_jd_tt, end_jd_tt}, ...],
      nested: [ ... level-1 nodes with children ... ],
      tree: {level:0,label:"kalachakra", start_jd_tt, end_jd_tt, children:[...]},
      meta: {...}
    }

NOTE
- This module requires `kcd_table` at runtime (or implement `load_kcd_table_preset`).
- For quick demos, set `use_demo_kcd_table: true` in the payload to auto-supply a simple, uniform mapping.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal, getcontext
import math

__all__ = [
    "compute_kalachakra_dasha",
    "kalachakra_schedule",
    "load_kcd_table_preset",
]

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
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig  # type: ignore
    from app.core.ephem_singleton import TS, PLANETS  # type: ignore
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

# ───────────────────────── small math / time helpers ─────────────────────────
def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _timescales_from_civil(date: str, time: str, tz: str) -> Tuple[float, float, float]:
    """Returns (jd_ut, jd_tt, jd_ut1). Uses time_kernel if present, else timescales."""
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
                        float(out["jd_tt"]),
                        float(out.get("jd_ut1") or out["jd_tt"]),
                    )
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
_EPHEM_CACHED: Optional[Any] = None

def _make_ephem_adapter() -> Any:
    """Robust adapter constructor across versions; cached for reuse."""
    global _EPHEM_CACHED
    if _EPHEM_CACHED is not None:
        return _EPHEM_CACHED
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")
    try:
        cfg = EphemConfig(frame="ecliptic-of-date", planets=PLANETS)  # type: ignore[arg-type]
    except TypeError:
        cfg = EphemConfig(frame="ecliptic-of-date")  # type: ignore
    try:
        ep = EphemerisAdapter(cfg, timescale=TS)  # type: ignore[arg-type]
    except TypeError:
        try:
            ep = EphemerisAdapter(cfg, TS)  # type: ignore[misc]
        except TypeError:
            try:
                cfg2 = EphemConfig(frame="ecliptic-of-date", planets=PLANETS, timescale=TS)  # type: ignore
                ep = EphemerisAdapter(cfg2)  # type: ignore
            except TypeError:
                ep = EphemerisAdapter(cfg)  # type: ignore
    _EPHEM_CACHED = ep
    return ep

def _moon_sidereal_longitude(
    jd_tt: float, *, ay_key: str, preload_sidereal: Optional[Dict[str, float]] = None
) -> float:
    """
    Return Moon's sidereal longitude in degrees (0..360).
    If preload_sidereal contains "Moon" (already sidereal), it is used.
    """
    if preload_sidereal and isinstance(preload_sidereal.get("Moon"), (int, float)):
        return _norm360(float(preload_sidereal["Moon"]))
    ep = _make_ephem_adapter()
    rows = (ep.ecliptic_longitudes(float(jd_tt), ["Moon"]) or {}).get("results", [])
    if not rows:
        raise RuntimeError("ephemeris returned no Moon longitude")
    trop = float(rows[0]["longitude"])
    ay = float(get_ayanamsa_deg(float(jd_tt), ay_key))
    return _norm360(trop - ay)

def _nak_pada_from_moon_sidereal(moon_lon_sid: float) -> Tuple[int, int, float]:
    """Returns (nak_index 1..27, pada 1..4, offset_within_pada_fraction 0..1)."""
    width = 360.0 / 27.0  # 13°20'
    idx0 = int(math.floor(_norm360(moon_lon_sid) / width))   # 0..26
    nak = idx0 + 1
    pos_in = _norm360(moon_lon_sid) - idx0 * width
    pada_float = (pos_in / width) * 4.0
    pada = int(math.floor(pada_float)) + 1  # 1..4
    frac_in_pada = max(0.0, min(1.0, pada_float - (pada - 1)))
    return nak, pada, frac_in_pada

def _pada_linear_index(nak: int, pada: int) -> int:
    """Map (nak 1..27, pada 1..4) -> 1..108."""
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
    Must return:
      {
        "name": "label",
        "pada_to_sequence": { 1:[12 ints], ..., 108:[...] },
        "sign_years": {1:int/float, ..., 12:int/float}
      }
    """
    raise NotImplementedError(f"KCD preset '{name}' not bundled. Supply kcd_table explicitly.")

def _normalize_kcd_table(tbl: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize JSON-ish tables:
      - Coerce 'pada_to_sequence' keys to int and values to 12-int lists
      - Accept 'sign_years' as dict with string/int keys or as a 12-length list
    Returns a new dict (does not mutate input).
    """
    if not isinstance(tbl, dict):
        raise ValueError("kcd_table must be a dict")

    # pada_to_sequence
    seq_raw = tbl.get("pada_to_sequence")
    if not isinstance(seq_raw, dict):
        raise ValueError("kcd_table must contain 'pada_to_sequence'")
    p2s: Dict[int, List[int]] = {}
    for k, v in seq_raw.items():
        k_i = int(k)
        if not (1 <= k_i <= 108):
            raise ValueError("kcd_table['pada_to_sequence'] keys must be 1..108")
        if not (isinstance(v, (list, tuple)) and len(v) == 12 and all(1 <= int(x) <= 12 for x in v)):
            raise ValueError("each 'pada_to_sequence'[k] must be a 12-length list of 1..12")
        p2s[k_i] = [int(x) for x in v]

    # sign_years
    yrs_raw = tbl.get("sign_years")
    if isinstance(yrs_raw, dict):
        yrs = {int(k): float(v) for k, v in yrs_raw.items()}
    elif isinstance(yrs_raw, (list, tuple)) and len(yrs_raw) == 12:
        yrs = {i + 1: float(yrs_raw[i]) for i in range(12)}
    else:
        raise ValueError("kcd_table['sign_years'] must be a dict with keys 1..12 or a 12-length list")
    if len(yrs) != 12 or any(i not in yrs for i in range(1, 13)):
        raise ValueError("kcd_table['sign_years'] must define 12 entries keyed 1..12")

    out = dict(tbl)
    out["pada_to_sequence"] = p2s
    out["sign_years"] = yrs
    return out

def _validate_kcd_table(tbl: Dict[str, Any]) -> None:
    # Once normalized, checks are trivial
    if not isinstance(tbl.get("pada_to_sequence"), dict) or not isinstance(tbl.get("sign_years"), dict):
        raise ValueError("kcd_table must contain 'pada_to_sequence' and 'sign_years'")
    if len(tbl["sign_years"]) != 12 or any(i not in tbl["sign_years"] for i in range(1, 13)):
        raise ValueError("kcd_table['sign_years'] must define 12 entries keyed 1..12")
    for k, v in tbl["pada_to_sequence"].items():
        if not (1 <= int(k) <= 108):
            raise ValueError("kcd_table['pada_to_sequence'] keys must be 1..108")
        if not (isinstance(v, list) and len(v) == 12 and all(1 <= int(x) <= 12 for x in v)):
            raise ValueError("each 'pada_to_sequence'[k] must be a 12-length list of 1..12")

def _demo_kcd_table() -> Dict[str, Any]:
    """
    Demo mapping:
      - Every pada maps to Aries..Pisces (1..12) order
      - Per-sign years are uniform (10 each → total 120)
    This is ONLY for dev console tests; do not use for research.
    """
    order = list(range(1, 13))
    return {
        "name": "demo_uniform",
        "pada_to_sequence": {i: order[:] for i in range(1, 109)},
        "sign_years": {i: 10 for i in range(1, 13)},
    }

# ───────────────────────── schedule builder ─────────────────────────
def _years_total(yrs: Dict[int, float | int]) -> Decimal:
    s = Decimal(0)
    for i in range(1, 13):
        s += Decimal(str(yrs[i]))
    return s

def _append_span(
    spans: List[DashaSpan],
    level: int,
    sign: int,
    t0: Decimal,
    dur_days: Decimal,
    *,
    limit: Optional[Decimal]
) -> Tuple[Decimal, Optional[DashaSpan]]:
    t1 = t0 + dur_days
    if limit is not None and t0 >= limit:
        return t1, None
    end = min(t1, limit) if limit is not None else t1
    sp = DashaSpan(level, sign, float(t0), float(end))
    spans.append(sp)
    return t1, sp

def _expand_children(
    parent: DashaSpan,
    *,
    level_next: int,
    seq12: List[int],
    yrs: Dict[int, float | int],
    total_years: Decimal,
    t_limit: Optional[Decimal],
) -> List[DashaSpan]:
    """Proportional split of parent into 12 children, rotating sequence to parent.sign."""
    parent_days = Decimal(str(parent.end_jd_tt)) - Decimal(str(parent.start_jd_tt))
    # rotate sequence so it begins at parent's sign
    if parent.sign in seq12:
        idx = seq12.index(parent.sign)
        order = seq12[idx:] + seq12[:idx]
    else:  # defensive
        order = list(range(1, 13))
    t = Decimal(str(parent.start_jd_tt))
    kids: List[DashaSpan] = []
    for s in order:
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
    sign_years: Dict[int, float | int],
    levels: int = 5,
    year_days: float = 365.24219,
    limit_jd_tt: float | None = None,
    balance_years: float | None = None,
) -> Dict[str, Any]:
    """Build KCD schedule from a 12-sign order and a per-sign year table."""
    levels = max(1, min(5, int(levels)))
    year_days_D = Decimal(str(year_days))
    t0 = Decimal(str(jd_start_tt))
    t_limit = Decimal(str(limit_jd_tt)) if isinstance(limit_jd_tt, (int, float)) else None

    # sanity
    if not (isinstance(sequence12, (list, tuple)) and len(sequence12) == 12 and all(1 <= int(x) <= 12 for x in sequence12)):
        raise ValueError("sequence12 must be a 12-length list of sign indices 1..12")
    _validate_kcd_table({"pada_to_sequence": {1: sequence12}, "sign_years": sign_years})

    total_years = _years_total(sign_years)

    # Level-1
    spans_L1: List[DashaSpan] = []
    t = t0
    for idx, s in enumerate(sequence12):
        y = Decimal(str(sign_years[int(s)]))
        dur_days = y * year_days_D
        if idx == 0 and isinstance(balance_years, (int, float)):
            reduce_days = Decimal(str(balance_years)) * year_days_D
            dur_days = max(Decimal(0), dur_days - reduce_days)
        t, _sp = _append_span(spans_L1, 1, int(s), t, dur_days, limit=t_limit)

    # Expand sub-levels
    all_spans: List[DashaSpan] = list(spans_L1)
    current = spans_L1[:]
    for lvl in range(2, levels + 1):
        next_level: List[DashaSpan] = []
        for p in current:
            next_level.extend(
                _expand_children(
                    p,
                    level_next=lvl,
                    seq12=sequence12,
                    yrs=sign_years,
                    total_years=total_years,
                    t_limit=t_limit,
                )
            )
        all_spans.extend(next_level)
        current = next_level

    all_spans.sort(key=lambda s: (s.level, s.start_jd_tt, s.sign))

    # Nested tree (level-1 roots only; children embedded)
    def children_of(par: DashaSpan, lvl: int) -> List[DashaSpan]:
        eps = 1e-9
        return [
            s for s in all_spans
            if s.level == lvl
            and par.start_jd_tt - eps <= s.start_jd_tt <= par.end_jd_tt + eps
            and s.end_jd_tt <= par.end_jd_tt + eps
        ]

    def node_for(s: DashaSpan, lvl: int) -> Dict[str, Any]:
        node = {
            "level": lvl,
            "sign_index": s.sign,
            "sign_name": SIGN_NAMES[s.sign - 1],
            "start_jd_tt": s.start_jd_tt,
            "end_jd_tt": s.end_jd_tt,
        }
        if lvl < levels:
            node["children"] = [node_for(k, lvl + 1) for k in children_of(s, lvl + 1)]
        return node

    nested = [node_for(s, 1) for s in all_spans if s.level == 1]
    if nested:
        s0 = min(float(n.get("start_jd_tt", 0.0)) for n in nested)
        e1 = max(float(n.get("end_jd_tt", 0.0)) for n in nested)
    else:
        s0 = float(jd_start_tt)
        e1 = float(jd_start_tt)

    return {
        "ok": True,
        "scheme": "kalachakra",
        "sign_years": {i: float(sign_years[i]) for i in range(1, 13)},
        "mahadasa_order": list(map(int, sequence12)),
        "spans": [
            {
                "level": s.level,
                "sign_index": s.sign,
                "sign_name": SIGN_NAMES[s.sign - 1],
                "start_jd_tt": s.start_jd_tt,
                "end_jd_tt": s.end_jd_tt,
            }
            for s in all_spans
        ],
        "nested": nested,
        "tree": {
            "level": 0,
            "label": "kalachakra",
            "start_jd_tt": s0,
            "end_jd_tt": e1,
            "children": nested,
        },
    }

# ───────────────────────── orchestrator (route-friendly) ─────────────────────
def compute_kalachakra_dasha(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Required: payload['kcd_table'] OR payload['kcd_preset'] (if you implement load_kcd_table_preset).
    For quick dev tests, you may pass { use_demo_kcd_table: true } to auto-supply a uniform mapping.

    Steps:
      1) Resolve jd_tt (from args or civil).
      2) Moon sidereal longitude (from ephemeris or preload).
      3) nak/pada (1..27, 1..4) → pada_idx (1..108).
      4) From kcd_table:
           sequence12 = table['pada_to_sequence'][pada_idx]
           sign_years  = table['sign_years']
         Optionally rotate/force start with 'override_start_sign_index'.
      5) Build schedule with full nesting and optional balance.

    Optional payload keys:
      - ayanamsa: str = "lahiri"
      - levels: 1..5 (default 5)
      - year_days: float (default 365.24219)
      - limit_jd_tt: float | None
      - balance_years: float | None
      - balance_fraction: float | None  (fraction of first Mahā years to shave)
      - override_start_sign_index: int 1..12
      - planet_longitudes_sidereal: optional dict {"Moon": deg, ...} (already sidereal)
    """
    try:
        ay_key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
        levels = int(payload.get("levels", 5))
        year_days = float(payload.get("year_days", 365.24219))
        limit = payload.get("limit_jd_tt")
        limit_jd_tt = float(limit) if isinstance(limit, (int, float)) else None

        # Resolve jd_tt
        jd_tt = payload.get("jd_tt")
        jd_ut1 = payload.get("jd_ut1")
        if not isinstance(jd_tt, (int, float)):
            if isinstance(payload.get("date"), str):
                tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
                _ju, jd_tt, jd_ut1 = _timescales_from_civil(str(payload["date"]), str(payload.get("time", "12:00:00")), tz)
            else:
                return {"ok": False, "error": "jd_tt_or_civil_required"}

        # KCD table
        table = payload.get("kcd_table")
        if table is None and bool(payload.get("use_demo_kcd_table")):
            table = _demo_kcd_table()
        if table is None and isinstance(payload.get("kcd_preset"), str):
            table = load_kcd_table_preset(str(payload["kcd_preset"]))
        if table is None:
            return {"ok": False, "error": "kcd_table_required"}

        # Normalize → Validate
        table = _normalize_kcd_table(table)
        _validate_kcd_table(table)
        table_name = str(table.get("name") or "")

        # Moon → nak/pada
        preload = payload.get("planet_longitudes_sidereal")
        moon_sid = _moon_sidereal_longitude(float(jd_tt), ay_key=ay_key, preload_sidereal=preload if isinstance(preload, dict) else None)
        nak, pada, frac_in_pada = _nak_pada_from_moon_sidereal(moon_sid)
        pada_idx = _pada_linear_index(nak, pada)

        # Sequence & per-sign years
        seq12 = list(map(int, table["pada_to_sequence"].get(int(pada_idx), [])))
        if not seq12:
            return {"ok": False, "error": f"no_sequence_for_pada_{pada_idx}"}
        sign_years = {int(k): float(v) for k, v in table["sign_years"].items()}

        # Override start sign
        start_basis = "pada_table"
        start_override = payload.get("override_start_sign_index")
        if isinstance(start_override, int) and 1 <= int(start_override) <= 12:
            s = int(start_override)
            if s in seq12:
                i = seq12.index(s)
                seq12 = seq12[i:] + seq12[:i]
            else:
                seq12 = [s] + [x for x in seq12 if x != s]
            start_basis = "override"

        # Balance (first mahā)
        balance_years_val: Optional[float] = None
        if isinstance(payload.get("balance_years"), (int, float)):
            balance_years_val = float(payload["balance_years"])
        elif isinstance(payload.get("balance_fraction"), (int, float)):
            frac = max(0.0, min(0.999999, float(payload["balance_fraction"])))
            first_years = float(sign_years[seq12[0]])
            balance_years_val = first_years * frac

        sched = kalachakra_schedule(
            jd_start_tt=float(jd_tt),
            sequence12=seq12,
            sign_years=sign_years,
            levels=int(levels),
            year_days=float(year_days),
            limit_jd_tt=limit_jd_tt,
            balance_years=balance_years_val,
        )

        # Attach start/rules/meta (plus root tree bounds)
        if sched["nested"] and isinstance(sched["nested"], list):
            s0 = min(n["start_jd_tt"] for n in sched["nested"]) if sched["nested"] else float(jd_tt)
            e1 = max(n["end_jd_tt"] for n in sched["nested"]) if sched["nested"] else float(jd_tt)
        else:
            s0 = float(jd_tt); e1 = float(jd_tt)

        sched["kcd_table_name"] = table_name or None
        sched["start"] = {
            "nakshatra_index": int(nak),
            "pada": int(pada),
            "pada_linear_index": int(pada_idx),
            "start_sign_index": int(seq12[0]),
            "start_sign_name": SIGN_NAMES[seq12[0] - 1],
            "basis": start_basis,
        }
        sched["rules"] = {
            "child_order": "rotate to parent sign, then follow scheme’s 12-sign sequence",
            "child_duration": "proportional: years(sign) / total_years_per_cycle",
            "levels": int(levels),
        }
        sched["meta"] = {
            "ayanamsa": ay_key,
            "year_days": float(year_days),
            "limit_jd_tt": limit_jd_tt,
        }
        # Ensure root tree is present (harmonize with other engines)
        sched["tree"] = {
            "level": 0,
            "label": "kalachakra",
            "start_jd_tt": s0,
            "end_jd_tt": e1,
            "children": sched["nested"],
        }
        return sched

    except NotImplementedError as e:
        return {"ok": False, "error": f"not_implemented:{e}"}
    except Exception as e:
        return {"ok": False, "error": f"kalachakra_failed:{e}"}  # clear, route-friendly
