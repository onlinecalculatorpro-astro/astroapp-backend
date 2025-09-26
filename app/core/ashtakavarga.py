# app/core/ashtakavarga.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Ashtakavarga engine — precision-first (gold-standard ready)

Public API:
    compute_ashtakavarga(payload: dict) -> dict
    clear_caches() -> None

Design notes
------------
• Single source of truth for astronomy, same pattern as shadbala.py:
    from app.core.astronomy import compute_chart as _compute_chart
• No HTTP fallbacks here. If astronomy is unavailable, we fail fast with a clear error.
• We normalize the astronomy chart into the AV-specific minimal shape locally.
"""

from typing import Dict, List, Tuple, Optional, Any

# ────────────────────────────── Required core (single source of truth) ───────
try:
    from app.core.astronomy import compute_chart as _compute_chart
except Exception as e:
    raise RuntimeError(f"ashtakavarga: astronomy.compute_chart import failed: {e}")

# ========================= Small Utilities =========================
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _wrap360(x: float) -> float:
    v = x % 360.0
    return v if v >= 0 else v + 360.0


def _sign_index(lon_deg: float) -> int:
    return int(_wrap360(lon_deg) // 30.0)  # 0=Aries … 11=Pisces


# ========================= Chart Normalization ======================
def _chart_to_av_shape(chart: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert astronomy.compute_chart() output into Ashtakavarga's canonical minimal shape:
      {
        "planets": { "Sun": {"lon": deg}, ..., "Saturn": {"lon": deg} },
        "angles":  { "asc": deg }   # optional but preferred (for Lagna rules)
        "meta":    { "mode": "sidereal|tropical", "ayanamsa_deg": float|None }
      }

    Accepts the chart format used by shadbala.py (bodies/points/meta/angles).
    """
    planets_out: Dict[str, Dict[str, float]] = {}

    # Bodies block: prefer longitude_deg; fallback to lon
    for row in (chart.get("bodies") or []):
        try:
            name = str(row.get("name"))
            if name in ("Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"):
                lon = float(row.get("longitude_deg", row.get("lon")))
                planets_out[name] = {"lon": lon}
        except Exception:
            continue

    # Fallback: some adapters may expose classic "planets" dict already
    if not planets_out and isinstance(chart.get("planets"), dict):
        for k, v in (chart["planets"] or {}).items():
            if k in ("Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"):
                if isinstance(v, dict):
                    lon = v.get("lon", v.get("longitude"))
                else:
                    lon = v
                f = _safe_float(lon)
                if f is not None:
                    planets_out[k] = {"lon": f}

    # Angles: prefer chart.angles.asc / asc_deg; fallback to meta.asc_deg
    angles_out: Dict[str, float] = {}
    angles_in = chart.get("angles") or {}
    asc = angles_in.get("asc") or angles_in.get("asc_deg")
    if asc is None:
        asc = (chart.get("meta") or {}).get("asc_deg")
    f_asc = _safe_float(asc)
    if f_asc is not None:
        angles_out["asc"] = f_asc

    # Meta: mode and ayanamsa_deg if available
    meta_in = chart.get("meta") or {}
    meta_out = {
        "mode": meta_in.get("mode") or chart.get("mode"),
        "ayanamsa_deg": meta_in.get("ayanamsa_deg"),
    }

    return {"planets": planets_out, "angles": angles_out, "meta": meta_out}


def _extract_asc_deg(chart_like: Dict[str, Any]) -> Optional[float]:
    """ASC in angles or meta, tolerant to a few common keys."""
    angles = chart_like.get("angles")
    if isinstance(angles, dict):
        for k in ("asc", "ASC", "Ascendant", "ascendant", "asc_deg"):
            v = angles.get(k)
            f = _safe_float(v)
            if f is not None:
                return f
        v = angles.get("asc")
        if isinstance(v, dict):
            f = _safe_float(v.get("deg"))
            if f is not None:
                return f

    meta = chart_like.get("meta", {})
    if isinstance(meta, dict):
        f = _safe_float(meta.get("asc_deg"))
        if f is not None:
            return f

    return None


# ========================= Engine constants =========================
PLANETS: List[str] = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]

# Canonical Parāśari / BPHS bindu-offset tables (giver -> receiver)
_BPHS_RULES: Dict[str, Dict[str, List[int]]] = {
    "Sun": {"Sun": [1, 2, 4, 7, 8, 9, 10, 11], "Moon": [3, 6, 10, 11], "Mars": [1, 2, 4, 7, 8, 9, 10, 11],
            "Mercury": [3, 5, 6, 9, 10, 11, 12], "Jupiter": [5, 6, 9, 11], "Venus": [6, 7, 12],
            "Saturn": [1, 2, 4, 7, 8, 9, 10, 11]},
    "Moon": {"Sun": [3, 6, 7, 8, 10, 11], "Moon": [1, 3, 6, 7, 10, 11],
             "Mars": [2, 3, 5, 6, 9, 10, 11], "Mercury": [1, 3, 4, 5, 7, 8, 10, 11],
             "Jupiter": [1, 4, 7, 8, 10, 11, 12], "Venus": [3, 4, 5, 7, 9, 10, 11],
             "Saturn": [3, 5, 6, 11]},
    "Mars": {"Sun": [3, 5, 6, 10, 11], "Moon": [3, 6, 11], "Mars": [1, 2, 4, 7, 8, 10, 11],
             "Mercury": [3, 5, 6, 11], "Jupiter": [6, 10, 11, 12], "Venus": [6, 8, 11, 12],
             "Saturn": [1, 4, 7, 8, 9, 10, 11]},
    "Mercury": {"Sun": [1, 3, 5, 6, 9, 10, 11, 12], "Moon": [2, 4, 6, 8, 10, 11],
                "Mars": [1, 2, 4, 7, 8, 9, 10, 11], "Mercury": [1, 3, 5, 6, 9, 10, 11, 12],
                "Jupiter": [6, 8, 11, 12], "Venus": [1, 2, 3, 4, 5, 9, 10, 11],
                "Saturn": [1, 2, 4, 7, 8, 9, 10, 11]},
    "Jupiter": {"Sun": [1, 2, 3, 4, 7, 8, 9, 10, 11], "Moon": [2, 5, 7, 9, 11],
                "Mars": [1, 2, 4, 7, 8, 10, 11], "Mercury": [1, 2, 4, 5, 6, 9, 10, 11],
                "Jupiter": [1, 2, 3, 4, 7, 8, 10, 11], "Venus": [2, 5, 6, 9, 10, 11],
                "Saturn": [3, 5, 6, 12]},
    "Venus": {"Sun": [8, 11, 12], "Moon": [1, 2, 3, 4, 5, 8, 9, 11, 12],
              "Mars": [3, 5, 6, 9, 11, 12], "Mercury": [3, 5, 6, 9, 11],
              "Jupiter": [5, 8, 9, 10, 11], "Venus": [1, 2, 3, 4, 5, 8, 9, 10, 11],
              "Saturn": [3, 4, 5, 8, 9, 10, 11]},
    "Saturn": {"Sun": [1, 2, 4, 7, 8, 10, 11], "Moon": [3, 6, 11],
               "Mars": [3, 5, 6, 10, 11, 12], "Mercury": [6, 8, 9, 10, 11, 12],
               "Jupiter": [5, 6, 11, 12], "Venus": [6, 11, 12], "Saturn": [3, 5, 6, 11]},
}

# Lagna as receiver
_BPHS_RULES_LAGNA: Dict[str, List[int]] = {
    "Sun": [3, 4, 6, 10, 11, 12], "Moon": [3, 6, 10, 11], "Mars": [1, 3, 6, 10, 11],
    "Mercury": [1, 2, 4, 6, 8, 10, 11], "Jupiter": [1, 2, 4, 5, 6, 9, 10, 11],
    "Venus": [1, 2, 3, 4, 5, 8, 9, 11], "Saturn": [1, 3, 4, 6, 10, 11],
}

# Cache of compiled rules
_COMPILED_RULES_CACHE: Dict[str, Dict[str, Dict[str, List[int]]]] = {}


# ========================= Rules Utilities =========================
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
            if receiver not in PLANETS and not (include_lagna and receiver in ("Lagna", "Asc")):
                ok = False; w.append(f"unknown_receiver:{giver}->{receiver}")
            if not isinstance(offs, (list, tuple)) or not all(isinstance(o, int) for o in offs):
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
                        key = "Lagna" if receiver in ("Lagna", "Asc") else receiver
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


def compute_ashtakavarga(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute BAV (per planet) and SAV (per sign) using Parāśari Ashtakavarga rules.
    Astronomy is obtained from app.core.astronomy.compute_chart (single source of truth).
    """
    warnings: List[str] = []
    meta: Dict[str, Any] = {
        "module": "ashtakavarga(core)",
        "version": 2,
        "mode": None,
        "ayanamsa_deg": None,
        "ruleset": None,
    }

    # Respect caller's requested zodiac mode (forward it to astronomy as "mode")
    req_mode = (str(payload.get("zodiac_mode") or payload.get("mode") or "").strip().lower() or None)
    if req_mode:
        payload = {**payload, "mode": req_mode}
    if "tz" not in payload:
        tz_guess = payload.get("place_tz") or payload.get("tz_name") or "UTC"
        payload = {**payload, "tz": tz_guess}

    # 1) Authoritative chart
    chart = _compute_chart(payload)  # may raise; propagate clear error upstream
    avc = _chart_to_av_shape(chart)

    # Mode & ayanamsa reporting
    meta["mode"] = (avc.get("meta") or {}).get("mode") or req_mode or "sidereal"
    a_deg = (avc.get("meta") or {}).get("ayanamsa_deg")
    if a_deg is None and str(meta["mode"]).lower() == "sidereal":
        a_deg = _safe_float(payload.get("ayanamsa"))
    meta["ayanamsa_deg"] = a_deg

    # Extract longitudes (7 planets)
    planets_block = avc.get("planets", {}) if isinstance(avc, dict) else {}
    longitudes: Dict[str, Optional[float]] = {}
    missing: List[str] = []
    for p in PLANETS:
        v = planets_block.get(p, {}) or {}
        lon = v.get("lon", v.get("longitude"))
        fv = _safe_float(lon)
        if fv is None:
            missing.append(p)
        longitudes[p] = fv

    # Ascendant (Lagna)
    asc = _extract_asc_deg(avc)
    if asc is None and isinstance(payload.get("angles"), dict):
        asc = _safe_float(payload["angles"].get("asc"))

    if missing:
        warnings.append("missing_longitudes:" + ",".join(missing))
    if asc is None:
        warnings.append("missing_lagna:asc")

    if all(longitudes[p] is None for p in PLANETS):
        return {
            "ok": False,
            "bav": {},
            "bav_totals": {},
            "sav": {"by_sign": [0] * 12, "total": 0},
            "meta": meta,
            "warnings": warnings + ["no_planet_longitudes_available"],
        }

    # Sign indices
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
    bav: Dict[str, List[int]] = {g: _bav_for_planet(g, signs, rules, warnings) for g in PLANETS}
    bav_totals: Dict[str, int] = {k: int(sum(v)) for k, v in bav.items()}

    # SAV
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
