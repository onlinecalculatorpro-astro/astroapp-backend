# app/core/dasha_registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Unified Dasha Registry — research-grade orchestration (gold-standard timescales)

Works with engines that expose either:
  A) keyword/positional args: compute_dasha(jd_tt_birth, ayanamsa_key=..., ayanamsa_deg=..., depth=..., options=...)
  B) single-argument payload: compute_xxx(payload: Dict[str, Any]) -> Dict

This registry normalizes inputs (timescales, ayanāṁśa), dispatches to the
engine, and returns a standardized tree.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
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

# JD quantization (~0.009 s by default)
_JD_QUANT = float(os.getenv("OCP_ASTRO_JD_QUANT", "1e-7"))


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

    # DUT1 (seconds)
    try:
        dut1_sec = float(payload.get("dut1_seconds", payload.get("dut1", 0.0)) or 0.0)
    except Exception:
        dut1_sec = 0.0

    # Preferred: time_kernel helpers (support 4-arg and 3-arg forms)
    if _tk is not None:
        for fname in ("timescales_from_civil", "compute_timescales", "build_timescales", "to_timescales", "from_civil"):
            fn = getattr(_tk, fname, None)
            if not callable(fn):
                continue
            attempts = (
                lambda: fn(date=d, time=t, tz=tz, dut1_seconds=dut1_sec),
                lambda: fn(date=d, time=t, tz=tz, dut1=dut1_sec),
                lambda: fn(d, t, tz, dut1_sec),   # legacy positional
                lambda: fn(d, t, tz),             # legacy 3-arg
                lambda: fn(date=d, time=t, tz=tz)
            )
            for call in attempts:
                try:
                    out = call()
                except TypeError:
                    continue
                except Exception:
                    continue
                if isinstance(out, dict):
                    ju = float(out.get("jd_ut") or out.get("jd_utc"))
                    jt = float(out.get("jd_tt") or out.get("tt") or out.get("jdtt"))
                    j1 = float(out.get("jd_ut1") or (ju + dut1_sec / 86400.0))
                    return _TSOut(_q_jd(ju), _q_jd(jt), _q_jd(j1), warns)
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    ju, jt, j1 = map(float, out[:3])
                    return _TSOut(_q_jd(ju), _q_jd(jt), _q_jd(j1), warns)

    if _ts is None:
        raise ValueError("timescales module unavailable and time_kernel fell through")

    # Fallback via timescales
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

    jd_ut1 = jd_ut + dut1_sec / 86400.0
    return _TSOut(_q_jd(jd_ut), _q_jd(jd_tt), _q_jd(jd_ut1), warns)


def _resolve_ayanamsa(jd_tt: float, payload: Dict[str, Any]) -> Tuple[str, float, List[str]]:
    """Return (key, degrees, warnings). Supports explicit numeric override."""
    warns: List[str] = []
    if isinstance(payload.get("ayanamsa"), (int, float)):
        return "explicit", float(payload["ayanamsa"]), warns
    key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
    if _get_ayanamsa_deg is None:
        # Linearized Lahiri fallback
        AY_J2000_DEG = (23 + 51 / 60 + 26.26 / 3600)
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
    anchor: str  # info only


_ENGINE_SPECS: Tuple[_DashaSpec, ...] = (
    _DashaSpec(
        key="vimshottari",
        title="Vimśottarī (120y) — 5 levels",
        module="app.core.vimshottari_dasha",  # FIXED module path
        fn_candidates=("compute_dasha", "compute_vimshottari", "build_dasha_tree", "generate_dasha"),
        levels=("maha", "antara", "pratyantara", "sookshma", "prana"),
        anchor="moon_nakshatra",
    ),
    _DashaSpec(
        key="ashtottari",
        title="Aṣṭottarī (108y) — 5 levels",
        module="app.core.ashtottari_dasha",
        fn_candidates=("compute_dasha", "compute_ashtottari", "build_dasha_tree", "generate_dasha"),
        levels=("maha", "antara", "pratyantara", "sookshma", "prana"),
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
    return [{
        "key": s.key, "title": s.title, "levels": list(s.levels), "anchor": s.anchor, "module": s.module
    } for s in _ENGINE_SPECS]


# ───────────────────────────── Tree schema helpers ────────────────────────────

def _norm_node(n: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce engine-specific node shapes into the standard schema."""
    lord = n.get("lord") or n.get("graha") or n.get("ruler") or n.get("planet") or n.get("deity") or n.get("name")
    label = n.get("label") or n.get("title") or str(lord or "Dasha")
    level = int(n.get("level") or n.get("depth") or 1)
    start = float(n.get("start_jd_tt") or n.get("start") or n.get("start_tt") or n.get("startJD") or 0.0)
    end = float(n.get("end_jd_tt") or n.get("end") or n.get("end_tt") or n.get("endJD") or 0.0)
    children = n.get("children") or n.get("subs") or n.get("items") or []
    if not math.isfinite(start) or not math.isfinite(end):
        start = float(n.get("start", 0.0)); end = float(n.get("end", 0.0))
    if abs(end - start) < 1e-12:
        end = start + 1e-9  # enforce [start, end)
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
    return _norm_node(tree)


def _envelope_from_forest(nodes: List[Dict[str, Any]], *, label: str = "Dasha", level_hint: int | None = None) -> Dict[str, Any]:
    if not nodes:
        return {"level": 1, "lord": None, "label": label, "start_jd_tt": 0.0, "end_jd_tt": 0.0, "children": []}
    s0 = min(float(n.get("start_jd_tt") or n.get("start") or 0.0) for n in nodes)
    e1 = max(float(n.get("end_jd_tt") or n.get("end") or 0.0) for n in nodes)
    child_levels = [int(n.get("level") or 1) for n in nodes]
    env_level = (min(child_levels) - 1) if min(child_levels) > 0 else 1
    if isinstance(level_hint, int):
        env_level = level_hint
    return {"level": int(env_level), "lord": None, "label": str(label),
            "start_jd_tt": _q_jd(s0), "end_jd_tt": _q_jd(e1), "children": nodes}


# ───────────────────────────── Engine dispatch ────────────────────────────────

def _payload_for_single_arg_engine(
    spec: _DashaSpec,
    original_payload: Dict[str, Any],
    jd_tt_birth: float,
    ay_key: str,
    ay_deg: float,
    depth: Optional[int],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a payload compatible with engines that want a single dict arg.
    Covers both Vimśottarī + Aṣṭottarī module styles.
    """
    p: Dict[str, Any] = {}

    # Birth epoch (provide both common keys)
    p["jd_tt"] = float(jd_tt_birth)
    p["birth_jd_tt"] = float(jd_tt_birth)

    # Ayanāṁśa: preserve caller's type if provided; else give both key + degrees
    if "ayanamsa" in original_payload:
        p["ayanamsa"] = original_payload["ayanamsa"]
    else:
        # engines vary: some want key, some degrees; include both
        p["ayanamsa"] = ay_key
        p["ayanamsa_deg"] = float(ay_deg)

    # Depth/levels
    if isinstance(depth, int) and depth > 0:
        p["levels"] = int(depth)
    elif isinstance(original_payload.get("levels"), int):
        p["levels"] = int(original_payload["levels"])

    # Common knobs (pass through if present)
    for k in ("start_mode", "year_days", "limit_jd_tt", "moon_nirayana_deg"):
        if k in original_payload:
            p[k] = original_payload[k]
        elif k in options:
            p[k] = options[k]

    # Also pass original civil triplet if present (engines may prefer civil)
    for k in ("date", "time", "tz", "place_tz"):
        if k in original_payload:
            p[k] = original_payload[k]

    # Options bag (if engine reads it)
    if options:
        p["options"] = dict(options)

    return p


def _coerce_engine_tree(res: Dict[str, Any], spec: _DashaSpec) -> Dict[str, Any]:
    """
    Accept multiple engine shapes and return a normalized single 'tree' node:
      - {'tree': {...}} → normalize
      - {'nested': [...]} → wrap into an envelope and normalize
      - direct node {'start_jd_tt', 'end_jd_tt', ...} → normalize
    """
    if isinstance(res, dict) and "tree" in res:
        return _norm_tree(res["tree"])
    if isinstance(res, dict) and "nested" in res and isinstance(res["nested"], list):
        env = _envelope_from_forest(res["nested"], label=(res.get("scheme") or spec.key or "Dasha"))
        return _norm_tree(env)
    if isinstance(res, dict) and all(k in res for k in ("start_jd_tt", "end_jd_tt")):
        return _norm_tree(res)
    raise ValueError(f"Engine returned unsupported shape for {spec.key}")


def _call_engine(
    spec: _DashaSpec,
    jd_tt_birth: float,
    ay_key: str,
    ay_deg: float,
    *,
    depth: Optional[int],
    options: Dict[str, Any],
    original_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Try single-arg payload engines first (compute_xxx(payload)), then
    the keyword/positional variants.
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

    # Detect single-argument payload function
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
    except Exception:
        params = []

    # Case 1: single positional/pos-or-kw param → call with built payload
    if len(params) == 1 and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        engine_payload = _payload_for_single_arg_engine(spec, original_payload, jd_tt_birth, ay_key, ay_deg, depth, options)
        res = fn(engine_payload)  # type: ignore[misc]
        tree = _coerce_engine_tree(res, spec)
        levels = list(res.get("levels")) if isinstance(res, dict) and isinstance(res.get("levels"), (list, tuple)) else list(spec.levels)
        return {"levels": levels, "tree": tree}

    # Case 2: keyword/positional variants
    kwargs_variants: List[Dict[str, Any]] = [
        {"jd_tt_birth": jd_tt_birth, "ayanamsa_key": ay_key, "ayanamsa_deg": ay_deg, "depth": depth, "options": options},
        {"jd_tt_birth": jd_tt_birth, "ayanamsa_key": ay_key, "depth": depth, "options": options},
        {"jd_tt_birth": jd_tt_birth, "depth": depth, **options},
        {"birth_jd_tt": jd_tt_birth, "ayanamsa_key": ay_key, "ayanamsa_deg": ay_deg, "depth": depth, **options},
    ]
    res = None
    last_err = None
    for kw in kwargs_variants:
        k2 = {k: v for k, v in kw.items() if v is not None}
        try:
            res = fn(**k2)
            break
        except TypeError:
            # Positional fallbacks based on arity
            try:
                pos_count = sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)) if params else 0
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

    tree = _coerce_engine_tree(res, spec)
    levels = list(res.get("levels")) if isinstance(res, dict) and isinstance(res.get("levels"), (list, tuple)) else list(spec.levels)
    return {"levels": levels, "tree": tree}


def _find_spec(system: str) -> _DashaSpec:
    key = str(system).strip().lower()
    for s in _ENGINE_SPECS:
        if s.key == key:
            return s
    raise ValueError(f"Unsupported dasha system: {system!r}")


# ───────────────────────────── Windowing / slicing ────────────────────────────

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
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
        for ch in n.get("children") or []:
            rec(ch, row["path"], row["index_path"], dict(idx_at_level))

    rec(tree, [], [], {})
    rows.sort(key=lambda r: (r["start_jd_tt"], r["level"], r["index_path"]))
    return rows


def _contains(t: float, s: float, e: float) -> bool:
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
    out: List[Tuple[float, str, int, List[str]]] = []
    def rec(n: Dict[str, Any], path: List[str]):
        lv = int(n["level"])
        if (level is None) or (lv == level):
            out.append((float(n["start_jd_tt"]), "start", lv, path + [str(n.get("lord") or n.get("label"))]))
            out.append((float(n["end_jd_tt"]), "end", lv, path + [str(n.get("lord") or n.get("label"))]))
        for ch in n.get("children") or []:
            rec(ch, path + [str(n.get("lord") or n.get("label"))])
    rec(tree, [])
    out.sort(key=lambda x: (x[0], x[1] == "start"))
    return out


# ───────────────────────────── Orchestrators ─────────────────────────────────

@lru_cache(maxsize=256)
def _cached_engine_result(system_key: str, birth_jd_tt_q: float, ay_key: str, ay_deg_q: float, depth_key: int, options_key: Tuple[Tuple[str, Any], ...], payload_key: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    """Cache full-tree engine result only; windowing/point queries operate on it."""
    spec = _find_spec(system_key)
    options = dict(options_key)
    original_payload = dict(payload_key)
    res = _call_engine(spec, birth_jd_tt_q, ay_key, ay_deg_q, depth=(depth_key or None), options=options, original_payload=original_payload)
    tree = _norm_tree(res["tree"])
    levels = list(res.get("levels") or spec.levels)
    return {"levels": levels, "tree": tree}


def _options_key(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    if not d:
        return tuple()
    items: List[Tuple[str, Any]] = []
    for k in sorted(d.keys()):
        v = d[k]
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
      - start_mode/year_days/limit_jd_tt/moon_nirayana_deg (forwarded to single-arg engines)
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
        _options_key(payload),   # include original payload (for single-arg engines)
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
    tte = None
    if path:
        deepest = path[-1]
        tte = max(0.0, float(deepest["end_jd_tt"]) - t)
    return {
        "ok": True,
        "system": core["system"],
        "levels": core["levels"],
        "active_path": path,
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
