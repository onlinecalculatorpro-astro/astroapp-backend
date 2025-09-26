# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Ashtakavarga engine — precision-first (gold-standard ready)

Public API:
    compute_ashtakavarga(payload: dict) -> dict
    clear_caches() -> None
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import json

# ---------- Astronomy acquisition (primary & fallbacks) ----------
# Primary import (if your build exposes a pure function)
try:
    from app.core.astronomy import compute_chart  # type: ignore
except Exception:
    compute_chart = None  # type: ignore

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _maybe_chart_shape(obj: Any) -> Optional[Dict[str, Any]]:
    """Try to coerce various 'ops/calculate' result shapes into a chart dict."""
    if not isinstance(obj, dict):
        return None
    # Already a chart?
    if "planets" in obj and (isinstance(obj["planets"], dict) or obj["planets"] is None):
        return obj

    # Common wrappers: {"chart": {...}} or {"result": {"chart": {...}}}
    for key in ("chart", "vedic_chart", "sidereal_chart", "tropical_chart"):
        c = obj.get(key)
        if isinstance(c, dict) and "planets" in c:
            return c

    res = obj.get("result")
    if isinstance(res, dict):
        for key in ("chart", "vedic_chart", "sidereal_chart", "tropical_chart"):
            c = res.get(key)
            if isinstance(c, dict) and "planets" in c:
                return c

    # Bundle: {"charts": {"vedic": {...}}}
    charts = obj.get("charts")
    if isinstance(charts, dict):
        for k in ("vedic", "primary", "sidereal", "tropical"):
            c = charts.get(k)
            if isinstance(c, dict) and "planets" in c:
                return c

    # Flat top-level planets/angles
    if {"Sun", "Moon"} & set(obj.keys()):
        # normalize to {planets:{...}, angles:{...}}
        planets = {k: v for k, v in obj.items() if k in ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn")}
        angles = {k: v for k, v in obj.items() if k.lower() in ("asc","mc","ascendant")}
        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        return {"planets": planets, "angles": angles, "meta": meta}

    return None

def _try_import_fallback_compute() -> Optional[Any]:
    """
    Try a few alternate import paths some codebases use for the same idea.
    Returns a callable(payload) -> chart | result | None, or None if not found.
    """
    candidates = (
        # (module, attribute)
        ("app.ops.astronomy", "compute_chart"),
        ("app.api.astronomy", "compute_chart"),
        ("app.core.ops_calculate", "compute"),         # sometimes export compute(payload)
        ("app.api.ops_calculate", "compute"),
        ("ops_api", "compute_chart"),
    )
    for mod, attr in candidates:
        try:
            m = __import__(mod, fromlist=[attr])
            fn = getattr(m, attr, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

def _call_ops_calculate_http(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    POST to /ops/calculate on the same service.
    Uses OPS_BASE_URL or http://127.0.0.1:{PORT} as a base.
    Returns a normalized chart dict or None.
    """
    try:
        import requests  # type: ignore
    except Exception:
        return None

    base = os.environ.get("OPS_BASE_URL")
    if not base:
        port = os.environ.get("PORT", "5000")
        base = f"http://127.0.0.1:{port}"
    url = base.rstrip("/") + "/ops/calculate"

    try:
        resp = requests.post(url, json=payload, timeout=15)
    except Exception:
        return None
    if resp.status_code >= 400:
        return None

    try:
        data = resp.json()
    except Exception:
        try:
            data = json.loads(resp.text or "{}")
        except Exception:
            return None

    chart = _maybe_chart_shape(data)
    return chart

def _obtain_chart(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Try multiple ways to obtain an astronomy chart in the canonical shape.
    Returns (chart_dict_or_None, warnings)
    """
    warnings: List[str] = []

    # 1) Primary
    if compute_chart is not None:
        try:
            raw = compute_chart(payload)
            chart = _maybe_chart_shape(raw) or _maybe_chart_shape(raw.get("chart") if isinstance(raw, dict) else None)
            if chart:
                return chart, warnings
        except Exception as e:
            warnings.append(f"astronomy_compute_failed:{type(e).__name__}")

    # 2) Importable fallbacks
    fallback = _try_import_fallback_compute()
    if callable(fallback):
        try:
            raw = fallback(payload)
            chart = _maybe_chart_shape(raw) or (isinstance(raw, dict) and _maybe_chart_shape(raw.get("chart")))
            if chart:
                return chart, warnings
        except Exception as e:
            warnings.append(f"astronomy_fallback_failed:{type(e).__name__}")

    # 3) HTTP to /ops/calculate
    chart = _call_ops_calculate_http(payload)
    if chart:
        return chart, warnings

    warnings.append("astronomy_router_unavailable")
    return None, warnings


# ---------- Engine constants ----------
PLANETS: List[str] = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]

# Canonical Parāśari / BPHS bindu-offset tables (giver -> receiver)
_BPHS_RULES: Dict[str, Dict[str, List[int]]] = {
    "Sun":     {"Sun":[1,2,4,7,8,9,10,11],"Moon":[3,6,10,11],"Mars":[1,2,4,7,8,9,10,11],
                "Mercury":[3,5,6,9,10,11,12],"Jupiter":[5,6,9,11],"Venus":[6,7,12],
                "Saturn":[1,2,4,7,8,9,10,11]},
    "Moon":    {"Sun":[3,6,7,8,10,11],"Moon":[1,3,6,7,10,11],"Mars":[2,3,5,6,9,10,11],
                "Mercury":[1,3,4,5,7,8,10,11],"Jupiter":[1,4,7,8,10,11,12],
                "Venus":[3,4,5,7,9,10,11],"Saturn":[3,5,6,11]},
    "Mars":    {"Sun":[3,5,6,10,11],"Moon":[3,6,11],"Mars":[1,2,4,7,8,10,11],
                "Mercury":[3,5,6,11],"Jupiter":[6,10,11,12],"Venus":[6,8,11,12],
                "Saturn":[1,4,7,8,9,10,11]},
    "Mercury": {"Sun":[1,3,5,6,9,10,11,12],"Moon":[2,4,6,8,10,11],
                "Mars":[1,2,4,7,8,9,10,11],"Mercury":[1,3,5,6,9,10,11,12],
                "Jupiter":[6,8,11,12],"Venus":[1,2,3,4,5,9,10,11],
                "Saturn":[1,2,4,7,8,9,10,11]},
    "Jupiter": {"Sun":[1,2,3,4,7,8,9,10,11],"Moon":[2,5,7,9,11],
                "Mars":[1,2,4,7,8,10,11],"Mercury":[1,2,4,5,6,9,10,11],
                "Jupiter":[1,2,3,4,7,8,10,11],"Venus":[2,5,6,9,10,11],
                "Saturn":[3,5,6,12]},
    "Venus":   {"Sun":[8,11,12],"Moon":[1,2,3,4,5,8,9,11,12],
                "Mars":[3,5,6,9,11,12],"Mercury":[3,5,6,9,11],
                "Jupiter":[5,8,9,10,11],"Venus":[1,2,3,4,5,8,9,10,11],
                "Saturn":[3,4,5,8,9,10,11]},
    "Saturn":  {"Sun":[1,2,4,7,8,10,11],"Moon":[3,6,11],
                "Mars":[3,5,6,10,11,12],"Mercury":[6,8,9,10,11,12],
                "Jupiter":[5,6,11,12],"Venus":[6,11,12],"Saturn":[3,5,6,11]},
}

# Lagna as receiver
_BPHS_RULES_LAGNA: Dict[str, List[int]] = {
    "Sun":[3,4,6,10,11,12], "Moon":[3,6,10,11], "Mars":[1,3,6,10,11],
    "Mercury":[1,2,4,6,8,10,11], "Jupiter":[1,2,4,5,6,9,10,11],
    "Venus":[1,2,3,4,5,8,9,11], "Saturn":[1,3,4,6,10,11],
}

# Cache of compiled rules
_COMPILED_RULES_CACHE: Dict[str, Dict[str, Dict[str, List[int]]]] = {}

# ========================= Utilities =========================
def _wrap360(x: float) -> float:
    v = x % 360.0
    return v if v >= 0 else v + 360.0

def _sign_index(lon_deg: float) -> int:
    return int(_wrap360(lon_deg) // 30.0)  # 0=Aries … 11=Pisces

def _bav_for_planet(
    giver: str,
    signs: Dict[str, int],
    rules: Dict[str, Dict[str, List[int]]],
    warnings: List[str],
) -> List[int]:
    vec = [0] * 12
    grules = rules.get(giver, {})
    for receiver, offsets in grules.items():
        if receiver not in signs:
            continue
        base = signs[receiver]
        for off in offsets:
            if not isinstance(off, int) or not (1 <= off <= 12):
                warnings.append(f"invalid_offset:{giver}->{receiver}:{off}")
                continue
            idx = (base + (off - 1)) % 12
            vec[idx] += 1
    return vec

def _validate_rules_map(
    rules_map: Dict[str, Dict[str, List[int]]],
    include_lagna: bool = True
) -> Tuple[bool, List[str]]:
    w: List[str] = []
    ok = True
    for giver, rmap in rules_map.items():
        if giver not in PLANETS:
            ok = False; w.append(f"unknown_giver:{giver}")
        for receiver, offs in rmap.items():
            if receiver not in PLANETS and not (include_lagna and receiver in ("Lagna","Asc")):
                ok = False; w.append(f"unknown_receiver:{giver}->{receiver}")
            if not isinstance(offs, (list,tuple)) or not all(isinstance(o,int) for o in offs):
                ok = False; w.append(f"invalid_offsets_type:{giver}->{receiver}")
            for o in offs:
                if o < 1 or o > 12:
                    ok = False; w.append(f"invalid_offset_range:{giver}->{receiver}:{o}")
    return ok, w

def _compile_rules_parashari() -> Dict[str, Dict[str, List[int]]]:
    compiled: Dict[str, Dict[str, List[int]]] = {}
    for giver in PLANETS:
        gr = _BPHS_RULES[giver]
        compiled[giver] = {}
        for receiver in PLANETS:
            compiled[giver][receiver] = list(gr[receiver])
        compiled[giver]["Lagna"] = list(_BPHS_RULES_LAGNA[giver])
    return compiled

def _load_ruleset(name: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, List[int]]], List[str], str]:
    warnings: List[str] = []
    tag = str(name or "parashari-bphs").lower()
    if tag == "custom":
        custom = payload.get("ruleset_map")
        if not isinstance(custom, dict):
            warnings.append("ruleset_custom_missing")
            tag = "parashari-bphs"
        else:
            ok, w = _validate_rules_map(custom, include_lagna=True)
            warnings.extend(w)
            if not ok:
                warnings.append("ruleset_custom_invalid")
                tag = "parashari-bphs"
            else:
                compiled: Dict[str, Dict[str, List[int]]] = {}
                for giver, rmap in custom.items():
                    compiled[giver] = {}
                    for receiver, offs in rmap.items():
                        key = "Lagna" if receiver in ("Lagna","Asc") else receiver
                        compiled[giver][key] = list(offs)
                return compiled, warnings, "custom"
    cache_key = "parashari-bphs"
    if cache_key in _COMPILED_RULES_CACHE:
        return _COMPILED_RULES_CACHE[cache_key], warnings, "parashari-bphs"
    compiled = _compile_rules_parashari()
    _COMPILED_RULES_CACHE[cache_key] = compiled
    return compiled, warnings, "parashari-bphs"

def clear_caches() -> None:
    _COMPILED_RULES_CACHE.clear()

# ========================= Core Engine =========================
def compute_ashtakavarga(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute BAV (per planet) and SAV (per sign) using Parāśari Ashtakavarga rules.
    Astronomy comes from app.core.astronomy.compute_chart(payload).
    """
    warnings: List[str] = []
    meta: Dict[str, Any] = {
        "module": "ashtakavarga(core)",
        "version": 1,
        "mode": None,
        "ayanamsa_deg": None,
        "ruleset": None,
    }

    # ---- Normalize payload for astronomy/OPS (compat shim) ----
    ap = dict(payload or {})
    # OPS expects "mode" not "zodiac_mode"
    if "mode" not in ap and ap.get("zodiac_mode"):
        ap["mode"] = ap["zodiac_mode"]
    # tz fallbacks
    if "tz" not in ap:
        ap["tz"] = ap.get("place_tz") or ap.get("tz_name") or "UTC"
    # coordinate coercion (avoid strings)
    for k in ("latitude", "longitude", "elevation_m"):
        if k in ap:
            try:
                ap[k] = float(ap[k])
            except Exception:
                pass
    # keep house system explicitly if provided
    if ap.get("house_system") is None and payload.get("house_system"):
        ap["house_system"] = payload["house_system"]

    if compute_chart is None:
        return {
            "ok": False, "bav": {}, "bav_totals": {},
            "sav": {"by_sign": [0]*12, "total": 0},
            "meta": meta,
            "warnings": ["astronomy_router_unavailable"],
        }

    # ---- Pull authoritative chart ----
    try:
        chart = compute_chart(ap)
    except Exception as e:
        return {
            "ok": False, "bav": {}, "bav_totals": {},
            "sav": {"by_sign": [0]*12, "total": 0},
            "meta": meta,
            "warnings": [f"astronomy_compute_failed:{type(e).__name__}"],
        }

    # Mode & ayanamsa reporting
    meta["mode"] = (chart.get("meta", {}).get("mode")
                    or ap.get("mode")
                    or ap.get("zodiac_mode")
                    or "sidereal")
    a_deg = chart.get("meta", {}).get("ayanamsa_deg")
    if a_deg is None and str(meta["mode"]).lower() == "sidereal":
        try:
            a_deg = float(ap.get("ayanamsa")) if ap.get("ayanamsa") is not None else None
        except Exception:
            a_deg = None
    meta["ayanamsa_deg"] = a_deg

    # Extract longitudes (7 planets)
    planets_block = chart.get("planets", {}) if isinstance(chart, dict) else {}
    longitudes: Dict[str, Optional[float]] = {}
    missing: List[str] = []
    for p in PLANETS:
        v = planets_block.get(p, {}) or {}
        lon = v.get("lon", v.get("longitude"))
        try:
            fv = float(lon) if lon is not None else None
        except Exception:
            fv = None
        if fv is None:
            missing.append(p)
        longitudes[p] = fv

    # Ascendant (Lagna)
    asc = None
    angles_block = chart.get("angles", chart.get("meta", {})) if isinstance(chart, dict) else {}
    if isinstance(angles_block, dict):
        for key in ("asc", "ASC", "Ascendant", "asc_deg"):
            if key in angles_block:
                try:
                    asc = float(angles_block[key])
                    break
                except Exception:
                    pass
    if asc is None and isinstance(payload.get("angles"), dict):
        try:
            asc = float(payload["angles"].get("asc"))
        except Exception:
            asc = None

    if missing:
        warnings.append("missing_longitudes:" + ",".join(missing))
    if asc is None:
        warnings.append("missing_lagna:asc")

    if all(longitudes[p] is None for p in PLANETS):
        return {
            "ok": False, "bav": {}, "bav_totals": {},
            "sav": {"by_sign": [0]*12, "total": 0},
            "meta": meta,
            "warnings": warnings + ["no_planet_longitudes_available"],
        }

    # Sign indices
    def _wrap360(x: float) -> float:
        v = x % 360.0
        return v if v >= 0 else v + 360.0
    def _sign_index(lon_deg: float) -> int:
        return int(_wrap360(lon_deg) // 30.0)

    signs: Dict[str, int] = {}
    for p, lon in longitudes.items():
        if lon is not None:
            signs[p] = _sign_index(lon)
    if asc is not None:
        signs["Lagna"] = _sign_index(asc)

    # Ruleset
    requested_ruleset = str(payload.get("ruleset", "parashari-bphs")).lower()
    rules, rs_warnings, ruleset_name = _load_ruleset(requested_ruleset, payload)
    warnings.extend(rs_warnings)
    meta["ruleset"] = ruleset_name

    # BAV per planet
    def _bav_for_planet(giver: str,
                        signs: Dict[str, int],
                        rules: Dict[str, Dict[str, List[int]]],
                        warnings: List[str]) -> List[int]:
        vec = [0]*12
        for receiver, offsets in rules.get(giver, {}).items():
            if receiver not in signs:
                continue
            base = signs[receiver]
            for off in offsets:
                if not isinstance(off, int) or not (1 <= off <= 12):
                    warnings.append(f"invalid_offset:{giver}->{receiver}:{off}")
                    continue
                idx = (base + (off - 1)) % 12
                vec[idx] += 1
        return vec

    bav: Dict[str, List[int]] = {g: _bav_for_planet(g, signs, rules, warnings) for g in PLANETS}
    bav_totals: Dict[str, int] = {k: int(sum(v)) for k, v in bav.items()}
    sav_by_sign = [sum(bav[g][i] for g in PLANETS) for i in range(12)]
    sav_total = int(sum(sav_by_sign))
    if sav_total != sum(bav_totals.values()):
        warnings.append("consistency_mismatch:sav_total!=sum(bav_totals)")

    return {
        "ok": True,
        "bav": bav,
        "bav_totals": bav_totals,
        "sav": {"by_sign": sav_by_sign, "total": sav_total},
        "meta": meta,
        "warnings": warnings,
    }

__all__ = ["compute_ashtakavarga", "clear_caches"]
