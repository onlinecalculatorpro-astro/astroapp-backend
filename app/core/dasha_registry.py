# app/core/dasha_registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Unified Dasha Registry — research-grade orchestration (gold-standard timescales)

Purpose
-------
Single entry point for all supported Vedic dasha systems (Vimshottari, Ashtottari,
Yogini, Chara, Kalachakra). This module:

- Normalizes civil inputs -> strict timescales (jd_tt, jd_ut, jd_ut1).
- Resolves ayanāṁśa (key + degrees at birth).
- Dispatches to the specific engine module with a uniform call wrapper.
- Returns a standardized Dasha Tree (same schema for all systems).
- Provides utilities:
  * window slicing (start/end jd_tt)
  * dasha_at() point query (active path across levels)
  * flatten_dasha() (tree -> rows)
  * next/prev boundary finders

Gold-standard rules
-------------------
- No UT≈TT shortcuts. Uses time_kernel/timescales; respects DUT1 if available.
- Deterministic boundary handling with closed-open intervals [start, end).
- 1-second JD bucketing for de-dupe in downstream consumers.
- Ayanāṁśa from app.core.ayanamsa (or explicit degrees if provided).

Public API
----------
    list_supported_dashas() -> list[dict]
    compute_dasha(system: str, payload: dict, *, depth: int|None=None) -> dict
    dasha_at(system: str, payload: dict, *, jd_tt: float) -> dict
    dasha_window(system: str, payload: dict, *, start_jd_tt: float, end_jd_tt: float, depth: int|None=None) -> dict
    flatten_dasha(tree: dict) -> list[dict]
    next_boundary(system: str, payload: dict, *, jd_tt: float, level: int|None=None) -> dict
    prev_boundary(system: str, payload: dict, *, jd_tt: float, level: int|None=None) -> dict
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from functools import lru_cache
import importlib
import inspect
import math
import os

# ─────────────────────────── Timescales & ayanāṁśa ───────────────────────────

try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

try:
    from app.core.ayanamsa import get_ayanamsa_deg as _get_ayanamsa_deg
except Exception:
    _get_ayanamsa_deg = None  # guarded

# JD quantization (align with astronomy.py defaults when present)
_JD_QUANT = float(os.getenv("OCP_ASTRO_JD_QUANT", "1e-7"))  # ~0.009 s


def _q_jd(x: float) -> float:
    return round(float(x) / _JD_QUANT) * _JD_QUANT


@dataclass(frozen=True)
class _TSOut:
    jd_ut: float
    jd_tt: float
    jd_ut1: float
    warnings: List[str]


def _timescales_from_payload(payload: Dict[str, Any]) -> _TSOut:
    """
    Accept:
      - direct {jd_tt[, jd_ut, jd_ut1]}  OR
      - civil {date, time, tz}
    """
    jd_tt = payload.get("jd_tt")
    jd_ut = payload.get("jd_ut")
    jd_ut1 = payload.get("jd_ut1")
    warns: List[str] = []

    if all(isinstance(v, (int, float)) for v in (jd_tt, jd_ut, jd_ut1)):
        return _TSOut(_q_jd(float(jd_ut)), _q_jd(float(jd_tt)), _q_jd(float(jd_ut1)), warns)

    # Accept nested birth dict or flat keys
    birth = payload.get("birth") or {}
    d = payload.get("date") or birth.get("date")
    t = payload.get("time") or birth.get("time") or "12:00:00"
    tz = payload.get("tz") or payload.get("place_tz") or birth.get("tz") or "UTC"

    # Preferred: time_kernel dynamic helpers
    if _tk is not None:
        for fname in ("timescales_from_civil", "compute_timescales", "build_timescales", "to_timescales", "from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(date=d, time=t, tz=tz)
                except TypeError:
                    out = fn(d, t, tz)
                if isinstance(out, dict):
                    ju = float(out.get("jd_ut") or out.get("jd_utc"))
                    jt = float(out["jd_tt"])
                    j1 = float(out["jd_ut1"])
                    return _TSOut(_q_jd(ju), _q_jd(jt), _q_jd(j1), warns)
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    ju, jt, j1 = map(float, out[:3])
                    return _TSOut(_q_jd(ju), _q_jd(jt), _q_jd(j1), warns)

    if _ts is None:
        raise ValueError("timescales module unavailable and time_kernel fell through")

    # Fallback chain via timescales
    try:
        jd_ut = float(_ts.julian_day_utc(d, t, tz))
    except Exception as e:
        raise ValueError(f"Failed to compute JD_UTC from {d} {t} {tz}: {e}")
    try:
        y, m = map(int, str(d).split("-")[:2])
    except Exception:
        y, m = 2000, 1
    try:
        jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m))
    except Exception:
        jd_tt = jd_ut + 69.0 / 86400.0  # constant ΔT fallback
        warns.append("deltaT_fallback_69s")
    # If DUT1 is known in payload, respect it
    dut1 = payload.get("dut1") or payload.get("dut1_seconds") or 0.0
    try:
        jd_ut1 = jd_ut + float(dut1) / 86400.0
    except Exception:
        jd_ut1 = jd_ut

    return _TSOut(_q_jd(jd_ut), _q_jd(jd_tt), _q_jd(jd_ut1), warns)


def _resolve_ayanamsa(jd_tt: float, payload: Dict[str, Any]) -> Tuple[str, float, List[str]]:
    """Return (key, degrees, warnings). Supports explicit numeric override."""
    warns: List[str] = []
    if isinstance(payload.get("ayanamsa"), (int, float)):
        return "explicit", float(payload["ayanamsa"]), warns
    key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
    if _get_ayanamsa_deg is None:
        # Conservative fallback if module is missing
        # Lahiri-like linearized fallback (only if unavoidable)
        AY_J2000_DEG = (23 + 51 / 60 + 26.26 / 3600)  # 23°51'26.26"
        RATE_AS_PER_YR = 50.290966
        Tcent = (float(jd_tt) - 2451545.0) / 36525.0
        years = Tcent * 100.0
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
        warns.append("ayanamsa_fallback_lahiri_linearized")
        return key, ay, warns
    try:
        ay = float(_get_ayanamsa_deg(float(jd_tt), key))
        return key, ay, warns
    except Exception as e:
        warns.append(f"ayanamsa_resolve_error:{e}")
        # last-resort linearized
        AY_J2000_DEG = (23 + 51 / 60 + 26.26 / 3600)
        RATE_AS_PER_YR = 50.290966
        Tcent = (float(jd_tt) - 2451545.0) / 36525.0
        years = Tcent * 100.0
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
        warns.append("ayanamsa_fallback_lahiri_linearized")
        return key, ay, warns


# ───────────────────────────── Engine registry ────────────────────────────────

@dataclass(frozen=True)
class _DashaSpec:
    key: str
    title: str
    module: str
    fn_candidates: Tuple[str, ...]
    levels: Tuple[str, ...]
    anchor: str  # informational: "moon_nakshatra", "nakshatra_pada", "charakaraka", etc.


_ENGINE_SPECS: Tuple[_DashaSpec, ...] = (
    _DashaSpec(
        key="vimshottari",
        title="Vimśottarī (120y) — 5 levels",
        module="app.core.dasha_vimshottari",
        fn_candidates=("compute_dasha", "compute_vimshottari", "build_dasha_tree", "generate_dasha"),
        levels=("maha", "antara", "pratyantara", "sookshma", "prana"),
        anchor="moon_nakshatra",
    ),
    _DashaSpec(
        key="ashtottari",
        title="Aṣṭottarī (108y)",
        module="app.core.dasha_ashtottari",
        fn_candidates=("compute_dasha", "compute_ashtottari", "build_dasha_tree", "generate_dasha"),
        levels=("maha", "antara", "pratyantara", "sookshma"),
        anchor="moon_nakshatra",
    ),
    _DashaSpec(
        key="yogini",
        title="Yoginī (36y)",
        module="app.core.dasha_yogini",
        fn_candidates=("compute_dasha", "compute_yogini", "build_dasha_tree", "generate_dasha"),
        levels=("maha", "antara", "pratyantara"),
        anchor="moon_nakshatra",
    ),
    _DashaSpec(
        key="chara",
        title="Cāra (Jaimini)",
        module="app.core.dasha_chara",
        fn_candidates=("compute_dasha", "compute_chara", "build_dasha_tree", "generate_dasha"),
        levels=("maha", "antara", "pratyantara"),
        anchor="charakaraka",
    ),
    _DashaSpec(
        key="kalachakra",
        title="Kālacakra",
        module="app.core.dasha_kalachakra",
        fn_candidates=("compute_dasha", "compute_kalachakra", "build_dasha_tree", "generate_dasha"),
        levels=("maha", "antara", "pratyantara", "sookshma"),
        anchor="nakshatra_pada",
    ),
)


def list_supported_dashas() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in _ENGINE_SPECS:
        out.append({
            "key": s.key,
            "title": s.title,
            "levels": list(s.levels),
            "anchor": s.anchor,
            "module": s.module,
        })
    return out


# ───────────────────────────── Tree schema helpers ────────────────────────────

# Standard node keys: level(int, 1..N), lord(str), label(str), start_jd_tt, end_jd_tt, children(list)

def _norm_node(n: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce engine-specific node shapes into the standard schema."""
    # Common aliases
    lord = n.get("lord") or n.get("graha") or n.get("ruler") or n.get("planet") or n.get("deity") or n.get("name")
    label = n.get("label") or n.get("title") or str(lord or "Dasha")
    level = int(n.get("level") or n.get("depth") or 1)
    start = float(n.get("start_jd_tt") or n.get("start") or n.get("start_tt") or n.get("startJD") or 0.0)
    end = float(n.get("end_jd_tt") or n.get("end") or n.get("end_tt") or n.get("endJD") or 0.0)
    children = n.get("children") or n.get("subs") or n.get("items") or []
    # Enforce closed-open: if end == start due to rounding, nudge end by 1e-9 d
    if not math.isfinite(start) or not math.isfinite(end):
        start = float(n.get("start", 0.0))
        end = float(n.get("end", 0.0))
    if abs(end - start) < 1e-12:
        end = start + 1e-9
    node = {
        "level": int(level),
        "lord": (str(lord) if lord is not None else None),
        "label": str(label),
        "start_jd_tt": _q_jd(start),
        "end_jd_tt": _q_jd(end),
        "children": [],
    }
    if isinstance(children, list):
        node["children"] = [_norm_node(c) for c in children]
    return node


def _norm_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tree, dict):
        raise ValueError("engine returned non-dict dasha tree")
    root = _norm_node(tree)
    return root


# ───────────────────────────── Engine dispatch ────────────────────────────────

def _call_engine(spec: _DashaSpec, jd_tt_birth: float, ay_key: str, ay_deg: float, *, depth: Optional[int], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attempt a few natural signatures in order, to maximize compatibility:

    Preferred engine signatures:
        compute_dasha(jd_tt_birth, ayanamsa_key="lahiri", ayanamsa_deg=None, depth=None, options=None) -> dict
        compute_dasha(jd_tt_birth, ayanamsa_key, depth=None, options=None) -> dict
        compute_dasha(jd_tt_birth, depth=None, **options) -> dict

    Engine should return dict with keys at least: {'levels': [...], 'tree': {...}}
    If engine returns only 'tree', we will attach levels from spec.
    """
    mod = importlib.import_module(spec.module)
    fn = None
    for name in spec.fn_candidates:
        f = getattr(mod, name, None)
        if callable(f):
            fn = f
            break
    if fn is None:
        raise NotImplementedError(f"Engine function not found in {spec.module} (tried {spec.fn_candidates})")

    # Try keyword-first call patterns
    kwargs_variants: List[Dict[str, Any]] = [
        {"jd_tt_birth": jd_tt_birth, "ayanamsa_key": ay_key, "ayanamsa_deg": ay_deg, "depth": depth, "options": options},
        {"jd_tt_birth": jd_tt_birth, "ayanamsa_key": ay_key, "depth": depth, "options": options},
        {"jd_tt_birth": jd_tt_birth, "depth": depth, **options},
        {"birth_jd_tt": jd_tt_birth, "ayanamsa_key": ay_key, "ayanamsa_deg": ay_deg, "depth": depth, **options},
    ]
    res = None
    last_err = None
    for kw in kwargs_variants:
        # Strip Nones the engine may not accept
        k2 = {k: v for k, v in kw.items() if v is not None}
        try:
            res = fn(**k2)
            break
        except TypeError:
            try:
                # Positional fallback: (jd_tt_birth, ayanamsa_key, ay_deg?, depth?)
                sig = inspect.signature(fn)
                pos_count = sum(1 for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
                if pos_count >= 4:
                    res = fn(jd_tt_birth, ay_key, ay_deg, depth)
                elif pos_count == 3:
                    res = fn(jd_tt_birth, ay_key, depth)
                elif pos_count == 2:
                    res = fn(jd_tt_birth, ay_key)
                else:
                    res = fn(jd_tt_birth)
                break
            except Exception as e:
                last_err = e
                continue
        except Exception as e:
            last_err = e
            continue

    if res is None:
        raise RuntimeError(f"Engine call failed for {spec.key}: {last_err}")

    # Normalize engine output
    if isinstance(res, dict) and "tree" in res:
        out = dict(res)
        out["tree"] = _norm_tree(res["tree"])
        if "levels" not in out or not out["levels"]:
            out["levels"] = list(spec.levels)
        return out

    # If engine directly returned a node (tree), wrap
    if isinstance(res, dict) and all(k in res for k in ("start_jd_tt", "end_jd_tt")):
        return {"levels": list(spec.levels), "tree": _norm_tree(res)}

    raise ValueError(f"Engine returned unsupported shape for {spec.key}")


def _find_spec(system: str) -> _DashaSpec:
    key = str(system).strip().lower()
    for s in _ENGINE_SPECS:
        if s.key == key:
            return s
    raise ValueError(f"Unsupported dasha system: {system!r}")


# ───────────────────────────── Windowing / slicing ────────────────────────────

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of overlap between [a0,a1) and [b0,b1)."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _slice_node(node: Dict[str, Any], w0: float, w1: float) -> Optional[Dict[str, Any]]:
    s = float(node["start_jd_tt"]); e = float(node["end_jd_tt"])
    if _overlap(s, e, w0, w1) <= 0.0:
        return None
    ns = max(s, w0); ne = min(e, w1)
    out = {
        "level": int(node["level"]),
        "lord": node.get("lord"),
        "label": node.get("label"),
        "start_jd_tt": _q_jd(ns),
        "end_jd_tt": _q_jd(ne),
        "children": [],
    }
    children = node.get("children") or []
    if children:
        sliced_children: List[Dict[str, Any]] = []
        for ch in children:
            sc = _slice_node(ch, w0, w1)
            if sc is not None:
                sliced_children.append(sc)
        out["children"] = sliced_children
    return out


def _slice_tree(tree: Dict[str, Any], w0: float, w1: float) -> Dict[str, Any]:
    res = _slice_node(tree, float(w0), float(w1))
    if res is None:
        # Return empty envelope matching root level
        return {
            "level": int(tree.get("level", 1)),
            "lord": tree.get("lord"),
            "label": tree.get("label"),
            "start_jd_tt": _q_jd(w0),
            "end_jd_tt": _q_jd(w1),
            "children": [],
        }
    return res


# ───────────────────────────── Flatten / queries ─────────────────────────────

def flatten_dasha(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def rec(n: Dict[str, Any], path: List[str], ipath: List[int], idx_at_level: Dict[int, int]) -> None:
        lv = int(n["level"])
        idx = idx_at_level.get(lv, 0)
        idx_at_level[lv] = idx + 1
        row = {
            "level": lv,
            "lord": n.get("lord"),
            "label": n.get("label"),
            "start_jd_tt": float(n["start_jd_tt"]),
            "end_jd_tt": float(n["end_jd_tt"]),
            "path": path + [str(n.get("lord") or n.get("label"))],
            "index_path": ipath + [idx],
        }
        rows.append(row)
        for i, ch in enumerate(n.get("children") or []):
            rec(ch, row["path"], row["index_path"], dict(idx_at_level))

    rec(tree, [], [], {})
    rows.sort(key=lambda r: (r["start_jd_tt"], r["level"], r["index_path"]))
    return rows


def _contains(t: float, s: float, e: float) -> bool:
    # Closed-open [s,e)
    return (s <= t) and (t < e - 1e-12)


def _active_path(tree: Dict[str, Any], t: float) -> List[Dict[str, Any]]:
    path: List[Dict[str, Any]] = []
    def rec(n: Dict[str, Any]) -> bool:
        s = float(n["start_jd_tt"]); e = float(n["end_jd_tt"])
        if not _contains(t, s, e):
            return False
        path.append(n)
        for ch in n.get("children") or []:
            if rec(ch):
                break
        return True
    rec(tree)
    return path


def _scan_boundaries(tree: Dict[str, Any], *, level: Optional[int] = None) -> List[Tuple[float, str, int, List[str]]]:
    """
    Collect (jd_tt, kind, level, path_names)
    kind in {"start","end"}
    """
    out: List[Tuple[float, str, int, List[str]]] = []
    def rec(n: Dict[str, Any], path: List[str]):
        lv = int(n["level"])
        if (level is None) or (lv == level):
            out.append((float(n["start_jd_tt"]), "start", lv, path + [str(n.get("lord") or n.get("label"))]))
            out.append((float(n["end_jd_tt"]), "end", lv, path + [str(n.get("lord") or n.get("label"))]))
        for ch in n.get("children") or []:
            rec(ch, path + [str(n.get("lord") or n.get("label"))])
    rec(tree, [])
    out.sort(key=lambda x: (x[0], x[1] == "start"))  # start before end at same instant
    return out


# ───────────────────────────── Orchestrators ─────────────────────────────────

@lru_cache(maxsize=256)
def _cached_engine_result(system_key: str, birth_jd_tt_q: float, ay_key: str, ay_deg_q: float, depth_key: int, options_key: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    """Cache full-tree engine result only; windowing/point queries operate on it."""
    spec = _find_spec(system_key)
    options = dict(options_key)
    res = _call_engine(spec, birth_jd_tt_q, ay_key, ay_deg_q, depth=(depth_key or None), options=options)
    # Normalize once
    tree = _norm_tree(res["tree"])
    levels = list(res.get("levels") or spec.levels)
    return {"levels": levels, "tree": tree}


def _options_key(options: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    if not options:
        return tuple()
    # Sort keys for stable cache key; coerce floats/ints/strs
    items = []
    for k in sorted(options.keys()):
        v = options[k]
        if isinstance(v, (float, int, str, bool)) or v is None:
            items.append((str(k), v))
        else:
            items.append((str(k), str(v)))
    return tuple(items)


def compute_dasha(system: str, payload: Dict[str, Any], *, depth: int | None = None) -> Dict[str, Any]:
    """
    Normalize inputs, call engine, and return standardized tree with meta.

    payload accepts:
      - birth: {date,time,tz} or {jd_tt[, jd_ut, jd_ut1]}
      - OR flat {date,time,tz} or {jd_tt[, jd_ut, jd_ut1]}
      - ayanamsa: "lahiri" (default) or explicit degrees
      - options: dict passed-through to engines (system-specific knobs)
    """
    ts = _timescales_from_payload(payload)
    ay_key, ay_deg, ay_warns = _resolve_ayanamsa(ts.jd_tt, payload)
    opts = payload.get("options") or {}
    res = _cached_engine_result(
        str(system).strip().lower(),
        _q_jd(ts.jd_tt),
        ay_key,
        float(ay_deg),
        int(depth or 0),
        _options_key(opts),
    )
    out = {
        "ok": True,
        "system": str(system).strip().lower(),
        "levels": list(res["levels"]),
        "tree": res["tree"],
        "meta": {
            "ayanamsa_key": ay_key,
            "ayanamsa_deg": float(ay_deg),
            "birth": {"jd_tt": float(ts.jd_tt), "jd_ut": float(ts.jd_ut), "jd_ut1": float(ts.jd_ut1)},
            "timescale_engine": ("time_kernel" if _tk is not None else "timescales"),
            "options": dict(opts),
        },
        "warnings": ay_warns + ts.warnings,
    }
    return out


def dasha_window(system: str, payload: Dict[str, Any], *, start_jd_tt: float, end_jd_tt: float, depth: int | None = None) -> Dict[str, Any]:
    core = compute_dasha(system, payload, depth=depth)
    tree = core["tree"]
    w0 = _q_jd(float(start_jd_tt)); w1 = _q_jd(float(end_jd_tt))
    sliced = _slice_tree(tree, w0, w1)
    return {
        "ok": True,
        "system": core["system"],
        "levels": core["levels"],
        "tree": sliced,
        "meta": {**core["meta"], "window_jd_tt": [w0, w1]},
        "warnings": core.get("warnings", []),
    }


def dasha_at(system: str, payload: Dict[str, Any], *, jd_tt: float) -> Dict[str, Any]:
    core = compute_dasha(system, payload)
    t = _q_jd(float(jd_tt))
    path = _active_path(core["tree"], t)
    # time to end for the deepest active node
    tte = None
    if path:
        deepest = path[-1]
        tte = max(0.0, float(deepest["end_jd_tt"]) - t)
    return {
        "ok": True,
        "system": core["system"],
        "levels": core["levels"],
        "active_path": path,  # list of nodes from level 1..k
        "query_jd_tt": t,
        "time_to_end_days": tte,
        "meta": core["meta"],
        "warnings": core.get("warnings", []),
    }


def next_boundary(system: str, payload: Dict[str, Any], *, jd_tt: float, level: int | None = None) -> Dict[str, Any]:
    core = compute_dasha(system, payload)
    t = _q_jd(float(jd_tt))
    events = _scan_boundaries(core["tree"], level=level)
    for jd, kind, lv, path in events:
        if jd > t:
            return {
                "ok": True,
                "system": core["system"],
                "level": lv,
                "kind": kind,
                "jd_tt": _q_jd(jd),
                "path": path,
                "delta_days": float(jd - t),
                "meta": core["meta"],
            }
    return {"ok": True, "system": core["system"], "message": "no_future_boundary", "meta": core["meta"]}


def prev_boundary(system: str, payload: Dict[str, Any], *, jd_tt: float, level: int | None = None) -> Dict[str, Any]:
    core = compute_dasha(system, payload)
    t = _q_jd(float(jd_tt))
    events = _scan_boundaries(core["tree"], level=level)
    prev = None
    for jd, kind, lv, path in events:
        if jd >= t:
            break
        prev = (jd, kind, lv, path)
    if prev is None:
        return {"ok": True, "system": core["system"], "message": "no_past_boundary", "meta": core["meta"]}
    jd, kind, lv, path = prev
    return {
        "ok": True,
        "system": core["system"],
        "level": lv,
        "kind": kind,
        "jd_tt": _q_jd(jd),
        "path": path,
        "delta_days": float(t - jd),
        "meta": core["meta"],
    }
