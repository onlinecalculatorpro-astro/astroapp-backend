# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Ashtakavarga engine — precision-first (gold-standard ready)

Public API:
    compute_ashtakavarga(payload: dict) -> dict
    clear_caches() -> None

Design principles (mirrors shadbala.py):
1) Authoritative astronomy only — all longitudes/angles come from
   `app.core.astronomy.compute_chart(payload)`. No ephemeris duplication here.
2) Respect `zodiac_mode` and `ayanamsa` exactly as astronomy provides. Longitudes
   we receive are already in the requested frame (sidereal/tropical).
3) Transparent `meta` + `warnings`: ruleset used, ayanāṁśa, mode, fallbacks, version, etc.
4) Deterministic, audited math. If something is missing, compute what you can,
   and emit explicit warnings. No approximations.
5) Minimal payload shape (same spirit as shadbala):
   {
     "date":"YYYY-MM-DD", "time":"HH:MM[:SS]", "tz":"IANA/Zone",
     "latitude": float, "longitude": float,
     "zodiac_mode":"sidereal"|"tropical", "ayanamsa": name|number,
     "house_system": str, "angles": {"asc": deg, "mc": deg}
   }

Output shape:
{
  "ok": true,
  "bav": { "Sun":[..12 ints..], ..., "Saturn":[..12..] },  # entries are integer bindu counts (0..8)
  "bav_totals": {"Sun": int, ..., "Saturn": int},
  "sav": {"by_sign":[..12..], "total": int},
  "meta": {"module":"ashtakavarga(core)","mode":"sidereal|tropical",
           "ayanamsa_deg":float|None,"ruleset":"parashari-bphs|custom",
           "version":1},
  "warnings":[...]
}
"""

from typing import Dict, List, Tuple, Optional, Any

# ---------- External dependency: astronomy router (authoritative longitudes) ----------
try:
    from app.core.astronomy import compute_chart  # type: ignore
except Exception:
    compute_chart = None  # type: ignore

PLANETS: List[str] = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]

# ------------------------------------------------------------------------------
# Canonical Parāśari / BPHS bindu-offset tables (giver -> receiver -> offsets 1..12)
# Offsets are counted from the receiver’s sign; add 1 bindu to that relative sign.
# Receivers are the seven planets; Lagna rows are provided separately below.
# Sources (consolidated/lined up with BPHS practice): see project docs.
# ------------------------------------------------------------------------------

_BPHS_RULES: Dict[str, Dict[str, List[int]]] = {
    # ---------------- Sun (giver) ----------------
    "Sun": {
        "Sun":     [1, 2, 4, 7, 8, 9, 10, 11],
        "Moon":    [3, 6, 10, 11],
        "Mars":    [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [3, 5, 6, 9, 10, 11, 12],
        "Jupiter": [5, 6, 9, 11],
        "Venus":   [6, 7, 12],
        "Saturn":  [1, 2, 4, 7, 8, 9, 10, 11],
    },

    # ---------------- Moon (giver) ----------------
    "Moon": {
        "Sun":     [3, 6, 7, 8, 10, 11],
        "Moon":    [1, 3, 6, 7, 10, 11],
        "Mars":    [2, 3, 5, 6, 9, 10, 11],
        "Mercury": [1, 3, 4, 5, 7, 8, 10, 11],
        "Jupiter": [1, 4, 7, 8, 10, 11, 12],
        "Venus":   [3, 4, 5, 7, 9, 10, 11],
        "Saturn":  [3, 5, 6, 11],
    },

    # ---------------- Mars (giver) ----------------
    "Mars": {
        "Sun":     [3, 5, 6, 10, 11],
        "Moon":    [3, 6, 11],
        "Mars":    [1, 2, 4, 7, 8, 10, 11],
        "Mercury": [3, 5, 6, 11],
        "Jupiter": [6, 10, 11, 12],
        "Venus":   [6, 8, 11, 12],
        "Saturn":  [1, 4, 7, 8, 9, 10, 11],
    },

    # ---------------- Mercury (giver) ----------------
    "Mercury": {
        "Sun":     [1, 3, 5, 6, 9, 10, 11, 12],
        "Moon":    [2, 4, 6, 8, 10, 11],
        "Mars":    [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [1, 3, 5, 6, 9, 10, 11, 12],
        "Jupiter": [6, 8, 11, 12],
        "Venus":   [1, 2, 3, 4, 5, 9, 10, 11],
        "Saturn":  [1, 2, 4, 7, 8, 9, 10, 11],
    },

    # ---------------- Jupiter (giver) ----------------
    "Jupiter": {
        "Sun":     [1, 2, 3, 4, 7, 8, 9, 10, 11],
        "Moon":    [2, 5, 7, 9, 11],
        "Mars":    [1, 2, 4, 7, 8, 10, 11],
        "Mercury": [1, 2, 4, 5, 6, 9, 10, 11],
        "Jupiter": [1, 2, 3, 4, 7, 8, 10, 11],
        "Venus":   [2, 5, 6, 9, 10, 11],
        "Saturn":  [3, 5, 6, 12],
    },

    # ---------------- Venus (giver) ----------------
    "Venus": {
        "Sun":     [8, 11, 12],
        "Moon":    [1, 2, 3, 4, 5, 8, 9, 11, 12],
        "Mars":    [3, 5, 6, 9, 11, 12],
        "Mercury": [3, 5, 6, 9, 11],
        "Jupiter": [5, 8, 9, 10, 11],
        "Venus":   [1, 2, 3, 4, 5, 8, 9, 10, 11],
        "Saturn":  [3, 4, 5, 8, 9, 10, 11],
    },

    # ---------------- Saturn (giver) ----------------
    "Saturn": {
        "Sun":     [1, 2, 4, 7, 8, 10, 11],
        "Moon":    [3, 6, 11],
        "Mars":    [3, 5, 6, 10, 11, 12],
        "Mercury": [6, 8, 9, 10, 11, 12],
        "Jupiter": [5, 6, 11, 12],
        "Venus":   [6, 11, 12],
        "Saturn":  [3, 5, 6, 11],
    },
}

# Lagna as receiver (offsets counted from Lagna’s sign) for each giver:
_BPHS_RULES_LAGNA: Dict[str, List[int]] = {
    "Sun":     [3, 4, 6, 10, 11, 12],
    "Moon":    [3, 6, 10, 11],
    "Mars":    [1, 3, 6, 10, 11],
    "Mercury": [1, 2, 4, 6, 8, 10, 11],
    "Jupiter": [1, 2, 4, 5, 6, 9, 10, 11],
    "Venus":   [1, 2, 3, 4, 5, 8, 9, 11],
    "Saturn":  [1, 3, 4, 6, 10, 11],
}

# ---------------- Internal caches (ruleset compilation) ----------------
_COMPILED_RULES_CACHE: Dict[str, Dict[str, Dict[str, List[int]]]] = {}


# ========================= Utilities =========================

def _wrap360(x: float) -> float:
    """Wrap angle into [0, 360)."""
    v = x % 360.0
    return v if v >= 0 else v + 360.0


def _sign_index(lon_deg: float) -> int:
    """Return 0..11 sign index from an ecliptic longitude in degrees (0° = Aries)."""
    return int(_wrap360(lon_deg) // 30.0)  # 0=Aries, ... 11=Pisces


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _bav_for_planet(
    giver: str,
    signs: Dict[str, int],
    rules: Dict[str, Dict[str, List[int]]],
    warnings: List[str],
) -> List[int]:
    """
    Compute the 12-sign BAV vector for `giver` using the supplied `rules`.

    `signs` maps body -> sign_index (0..11). Bodies include "Lagna" (if available) and the 7 planets.
    `rules[giver][receiver] = [offsets 1..12]`.

    For each receiver present in `signs` and in rules, each offset marks
    (signs[receiver] + offset - 1) % 12 as +1 in the giver's BAV.
    """
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
    """Validate a custom ruleset map coming from payload."""
    w: List[str] = []
    ok = True
    for giver, rmap in rules_map.items():
        if giver not in PLANETS:
            ok = False
            w.append(f"unknown_giver:{giver}")
        for receiver, offs in rmap.items():
            if receiver not in PLANETS and not (include_lagna and receiver in ("Lagna", "Asc")):
                ok = False
                w.append(f"unknown_receiver:{giver}->{receiver}")
            if not isinstance(offs, (list, tuple)) or not all(isinstance(o, int) for o in offs):
                ok = False
                w.append(f"invalid_offsets_type:{giver}->{receiver}")
            for o in offs:
                if o < 1 or o > 12:
                    ok = False
                    w.append(f"invalid_offset_range:{giver}->{receiver}:{o}")
    return ok, w


def _compile_rules_parashari() -> Dict[str, Dict[str, List[int]]]:
    """Compile embedded BPHS rules (+ Lagna receiver rows)."""
    compiled: Dict[str, Dict[str, List[int]]] = {}
    for giver in PLANETS:
        gr = _BPHS_RULES[giver]
        compiled[giver] = {}
        for receiver in PLANETS:
            compiled[giver][receiver] = list(gr[receiver])
        # Lagna offsets:
        compiled[giver]["Lagna"] = list(_BPHS_RULES_LAGNA[giver])
    return compiled


def _load_ruleset(name: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, List[int]]], List[str], str]:
    """
    Load/compile a ruleset.
    - name "parashari-bphs": uses embedded constants in this module.
    - name "custom": uses `payload['ruleset_map']` (validated).
    Returns: (rules, warnings, ruleset_name_effective)
    """
    warnings: List[str] = []
    tag = str(name or "parashari-bphs").lower()

    if tag == "custom":
        custom = payload.get("ruleset_map")
        if not isinstance(custom, dict):
            warnings.append("ruleset_custom_missing")
            # Fallback to BPHS
            tag = "parashari-bphs"
        else:
            ok, w = _validate_rules_map(custom, include_lagna=True)
            warnings.extend(w)
            if not ok:
                warnings.append("ruleset_custom_invalid")
                tag = "parashari-bphs"
            else:
                # normalize "Asc" -> "Lagna"
                compiled: Dict[str, Dict[str, List[int]]] = {}
                for giver, rmap in custom.items():
                    compiled[giver] = {}
                    for receiver, offs in rmap.items():
                        key = "Lagna" if receiver in ("Lagna", "Asc") else receiver
                        compiled[giver][key] = list(offs)
                return compiled, warnings, "custom"

    # Default / fallback: parashari-bphs
    cache_key = "parashari-bphs"
    if cache_key in _COMPILED_RULES_CACHE:
        return _COMPILED_RULES_CACHE[cache_key], warnings, "parashari-bphs"

    compiled = _compile_rules_parashari()
    _COMPILED_RULES_CACHE[cache_key] = compiled
    return compiled, warnings, "parashari-bphs"


def clear_caches() -> None:
    """Clear ruleset compilation caches."""
    _COMPILED_RULES_CACHE.clear()


# ========================= Core Engine =========================

def compute_ashtakavarga(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute BAV (per planet) and SAV (per sign) using Parāśari Ashtakavarga rules.

    `payload` — minimal input; all astronomy comes from `compute_chart(payload)`.

    Returns a dict with keys: ok, bav, bav_totals, sav{by_sign,total}, meta, warnings.
    """
    warnings: List[str] = []
    meta: Dict[str, Any] = {
        "module": "ashtakavarga(core)",
        "version": 1,
        "mode": None,
        "ayanamsa_deg": None,
        "ruleset": None,
    }

    if compute_chart is None:
        return {
            "ok": False,
            "bav": {},
            "bav_totals": {},
            "sav": {"by_sign": [0] * 12, "total": 0},
            "meta": meta,
            "warnings": ["astronomy_router_unavailable"],
        }

    # Pull authoritative chart from astronomy adapter
    try:
        chart = compute_chart(payload)
    except Exception as e:
        return {
            "ok": False,
            "bav": {},
            "bav_totals": {},
            "sav": {"by_sign": [0] * 12, "total": 0},
            "meta": meta,
            "warnings": [f"astronomy_compute_failed:{type(e).__name__}"],
        }

    # Mode & ayanamsa reporting
    meta["mode"] = (chart.get("meta", {}).get("mode")
                    or payload.get("zodiac_mode")
                    or "sidereal")
    a_deg = chart.get("meta", {}).get("ayanamsa_deg")
    if a_deg is None and str(meta["mode"]).lower() == "sidereal":
        a_deg = _safe_float(payload.get("ayanamsa"))
    meta["ayanamsa_deg"] = a_deg

    # Extract longitudes for the 7 planets
    planets_block = chart.get("planets", {})
    longitudes: Dict[str, Optional[float]] = {}
    missing: List[str] = []
    for p in PLANETS:
        v = planets_block.get(p, {})
        lon = v.get("lon", v.get("longitude"))
        fv = _safe_float(lon)
        if fv is None:
            missing.append(p)
        longitudes[p] = fv

    # Ascendant (Lagna)
    asc = None
    angles_block = chart.get("angles", chart.get("meta", {}))
    if isinstance(angles_block, dict):
        asc = _safe_float(angles_block.get("asc") or angles_block.get("ASC") or angles_block.get("Ascendant"))
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

    # Build sign indices
    signs: Dict[str, int] = {}
    for p, lon in longitudes.items():
        if lon is not None:
            signs[p] = _sign_index(lon)
    if asc is not None:
        signs["Lagna"] = _sign_index(asc)

    # Load ruleset
    requested_ruleset = str(payload.get("ruleset", "parashari-bphs")).lower()
    rules, rs_warnings, ruleset_name = _load_ruleset(requested_ruleset, payload)
    warnings.extend(rs_warnings)
    meta["ruleset"] = ruleset_name

    # BAV per planet
    bav: Dict[str, List[int]] = {}
    for giver in PLANETS:
        vec = _bav_for_planet(giver, signs, rules, warnings)
        bav[giver] = vec

    # BAV totals per planet
    bav_totals: Dict[str, int] = {k: int(sum(v)) for k, v in bav.items()}

    # SAV: column-wise sum
    sav_by_sign = [sum(bav[g][i] for g in PLANETS) for i in range(12)]
    sav_total = int(sum(sav_by_sign))

    # Consistency check
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
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Ashtakavarga engine — precision-first (gold-standard ready)

Public API:
    compute_ashtakavarga(payload: dict) -> dict
    clear_caches() -> None

Design principles (mirrors shadbala.py):
1) Authoritative astronomy only — all longitudes/angles come from
   `app.core.astronomy.compute_chart(payload)`. No ephemeris duplication here.
2) Respect `zodiac_mode` and `ayanamsa` exactly as astronomy provides. Longitudes
   we receive are already in the requested frame (sidereal/tropical).
3) Transparent `meta` + `warnings`: ruleset used, ayanāṁśa, mode, fallbacks, version, etc.
4) Deterministic, audited math. If something is missing, compute what you can,
   and emit explicit warnings. No approximations.
5) Minimal payload shape (same spirit as shadbala):
   {
     "date":"YYYY-MM-DD", "time":"HH:MM[:SS]", "tz":"IANA/Zone",
     "latitude": float, "longitude": float,
     "zodiac_mode":"sidereal"|"tropical", "ayanamsa": name|number,
     "house_system": str, "angles": {"asc": deg, "mc": deg}
   }

Output shape:
{
  "ok": true,
  "bav": { "Sun":[..12 ints..], ..., "Saturn":[..12..] },  # entries are integer bindu counts (0..8)
  "bav_totals": {"Sun": int, ..., "Saturn": int},
  "sav": {"by_sign":[..12..], "total": int},
  "meta": {"module":"ashtakavarga(core)","mode":"sidereal|tropical",
           "ayanamsa_deg":float|None,"ruleset":"parashari-bphs|custom",
           "version":1},
  "warnings":[...]
}
"""

from typing import Dict, List, Tuple, Optional, Any

# ---------- External dependency: astronomy router (authoritative longitudes) ----------
try:
    from app.core.astronomy import compute_chart  # type: ignore
except Exception:
    compute_chart = None  # type: ignore

PLANETS: List[str] = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]

# ------------------------------------------------------------------------------
# Canonical Parāśari / BPHS bindu-offset tables (giver -> receiver -> offsets 1..12)
# Offsets are counted from the receiver’s sign; add 1 bindu to that relative sign.
# Receivers are the seven planets; Lagna rows are provided separately below.
# Sources (consolidated/lined up with BPHS practice): see project docs.
# ------------------------------------------------------------------------------

_BPHS_RULES: Dict[str, Dict[str, List[int]]] = {
    # ---------------- Sun (giver) ----------------
    "Sun": {
        "Sun":     [1, 2, 4, 7, 8, 9, 10, 11],
        "Moon":    [3, 6, 10, 11],
        "Mars":    [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [3, 5, 6, 9, 10, 11, 12],
        "Jupiter": [5, 6, 9, 11],
        "Venus":   [6, 7, 12],
        "Saturn":  [1, 2, 4, 7, 8, 9, 10, 11],
    },

    # ---------------- Moon (giver) ----------------
    "Moon": {
        "Sun":     [3, 6, 7, 8, 10, 11],
        "Moon":    [1, 3, 6, 7, 10, 11],
        "Mars":    [2, 3, 5, 6, 9, 10, 11],
        "Mercury": [1, 3, 4, 5, 7, 8, 10, 11],
        "Jupiter": [1, 4, 7, 8, 10, 11, 12],
        "Venus":   [3, 4, 5, 7, 9, 10, 11],
        "Saturn":  [3, 5, 6, 11],
    },

    # ---------------- Mars (giver) ----------------
    "Mars": {
        "Sun":     [3, 5, 6, 10, 11],
        "Moon":    [3, 6, 11],
        "Mars":    [1, 2, 4, 7, 8, 10, 11],
        "Mercury": [3, 5, 6, 11],
        "Jupiter": [6, 10, 11, 12],
        "Venus":   [6, 8, 11, 12],
        "Saturn":  [1, 4, 7, 8, 9, 10, 11],
    },

    # ---------------- Mercury (giver) ----------------
    "Mercury": {
        "Sun":     [1, 3, 5, 6, 9, 10, 11, 12],
        "Moon":    [2, 4, 6, 8, 10, 11],
        "Mars":    [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [1, 3, 5, 6, 9, 10, 11, 12],
        "Jupiter": [6, 8, 11, 12],
        "Venus":   [1, 2, 3, 4, 5, 9, 10, 11],
        "Saturn":  [1, 2, 4, 7, 8, 9, 10, 11],
    },

    # ---------------- Jupiter (giver) ----------------
    "Jupiter": {
        "Sun":     [1, 2, 3, 4, 7, 8, 9, 10, 11],
        "Moon":    [2, 5, 7, 9, 11],
        "Mars":    [1, 2, 4, 7, 8, 10, 11],
        "Mercury": [1, 2, 4, 5, 6, 9, 10, 11],
        "Jupiter": [1, 2, 3, 4, 7, 8, 10, 11],
        "Venus":   [2, 5, 6, 9, 10, 11],
        "Saturn":  [3, 5, 6, 12],
    },

    # ---------------- Venus (giver) ----------------
    "Venus": {
        "Sun":     [8, 11, 12],
        "Moon":    [1, 2, 3, 4, 5, 8, 9, 11, 12],
        "Mars":    [3, 5, 6, 9, 11, 12],
        "Mercury": [3, 5, 6, 9, 11],
        "Jupiter": [5, 8, 9, 10, 11],
        "Venus":   [1, 2, 3, 4, 5, 8, 9, 10, 11],
        "Saturn":  [3, 4, 5, 8, 9, 10, 11],
    },

    # ---------------- Saturn (giver) ----------------
    "Saturn": {
        "Sun":     [1, 2, 4, 7, 8, 10, 11],
        "Moon":    [3, 6, 11],
        "Mars":    [3, 5, 6, 10, 11, 12],
        "Mercury": [6, 8, 9, 10, 11, 12],
        "Jupiter": [5, 6, 11, 12],
        "Venus":   [6, 11, 12],
        "Saturn":  [3, 5, 6, 11],
    },
}

# Lagna as receiver (offsets counted from Lagna’s sign) for each giver:
_BPHS_RULES_LAGNA: Dict[str, List[int]] = {
    "Sun":     [3, 4, 6, 10, 11, 12],
    "Moon":    [3, 6, 10, 11],
    "Mars":    [1, 3, 6, 10, 11],
    "Mercury": [1, 2, 4, 6, 8, 10, 11],
    "Jupiter": [1, 2, 4, 5, 6, 9, 10, 11],
    "Venus":   [1, 2, 3, 4, 5, 8, 9, 11],
    "Saturn":  [1, 3, 4, 6, 10, 11],
}

# ---------------- Internal caches (ruleset compilation) ----------------
_COMPILED_RULES_CACHE: Dict[str, Dict[str, Dict[str, List[int]]]] = {}


# ========================= Utilities =========================

def _wrap360(x: float) -> float:
    """Wrap angle into [0, 360)."""
    v = x % 360.0
    return v if v >= 0 else v + 360.0


def _sign_index(lon_deg: float) -> int:
    """Return 0..11 sign index from an ecliptic longitude in degrees (0° = Aries)."""
    return int(_wrap360(lon_deg) // 30.0)  # 0=Aries, ... 11=Pisces


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _bav_for_planet(
    giver: str,
    signs: Dict[str, int],
    rules: Dict[str, Dict[str, List[int]]],
    warnings: List[str],
) -> List[int]:
    """
    Compute the 12-sign BAV vector for `giver` using the supplied `rules`.

    `signs` maps body -> sign_index (0..11). Bodies include "Lagna" (if available) and the 7 planets.
    `rules[giver][receiver] = [offsets 1..12]`.

    For each receiver present in `signs` and in rules, each offset marks
    (signs[receiver] + offset - 1) % 12 as +1 in the giver's BAV.
    """
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
    """Validate a custom ruleset map coming from payload."""
    w: List[str] = []
    ok = True
    for giver, rmap in rules_map.items():
        if giver not in PLANETS:
            ok = False
            w.append(f"unknown_giver:{giver}")
        for receiver, offs in rmap.items():
            if receiver not in PLANETS and not (include_lagna and receiver in ("Lagna", "Asc")):
                ok = False
                w.append(f"unknown_receiver:{giver}->{receiver}")
            if not isinstance(offs, (list, tuple)) or not all(isinstance(o, int) for o in offs):
                ok = False
                w.append(f"invalid_offsets_type:{giver}->{receiver}")
            for o in offs:
                if o < 1 or o > 12:
                    ok = False
                    w.append(f"invalid_offset_range:{giver}->{receiver}:{o}")
    return ok, w


def _compile_rules_parashari() -> Dict[str, Dict[str, List[int]]]:
    """Compile embedded BPHS rules (+ Lagna receiver rows)."""
    compiled: Dict[str, Dict[str, List[int]]] = {}
    for giver in PLANETS:
        gr = _BPHS_RULES[giver]
        compiled[giver] = {}
        for receiver in PLANETS:
            compiled[giver][receiver] = list(gr[receiver])
        # Lagna offsets:
        compiled[giver]["Lagna"] = list(_BPHS_RULES_LAGNA[giver])
    return compiled


def _load_ruleset(name: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, List[int]]], List[str], str]:
    """
    Load/compile a ruleset.
    - name "parashari-bphs": uses embedded constants in this module.
    - name "custom": uses `payload['ruleset_map']` (validated).
    Returns: (rules, warnings, ruleset_name_effective)
    """
    warnings: List[str] = []
    tag = str(name or "parashari-bphs").lower()

    if tag == "custom":
        custom = payload.get("ruleset_map")
        if not isinstance(custom, dict):
            warnings.append("ruleset_custom_missing")
            # Fallback to BPHS
            tag = "parashari-bphs"
        else:
            ok, w = _validate_rules_map(custom, include_lagna=True)
            warnings.extend(w)
            if not ok:
                warnings.append("ruleset_custom_invalid")
                tag = "parashari-bphs"
            else:
                # normalize "Asc" -> "Lagna"
                compiled: Dict[str, Dict[str, List[int]]] = {}
                for giver, rmap in custom.items():
                    compiled[giver] = {}
                    for receiver, offs in rmap.items():
                        key = "Lagna" if receiver in ("Lagna", "Asc") else receiver
                        compiled[giver][key] = list(offs)
                return compiled, warnings, "custom"

    # Default / fallback: parashari-bphs
    cache_key = "parashari-bphs"
    if cache_key in _COMPILED_RULES_CACHE:
        return _COMPILED_RULES_CACHE[cache_key], warnings, "parashari-bphs"

    compiled = _compile_rules_parashari()
    _COMPILED_RULES_CACHE[cache_key] = compiled
    return compiled, warnings, "parashari-bphs"


def clear_caches() -> None:
    """Clear ruleset compilation caches."""
    _COMPILED_RULES_CACHE.clear()


# ========================= Core Engine =========================

def compute_ashtakavarga(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute BAV (per planet) and SAV (per sign) using Parāśari Ashtakavarga rules.

    `payload` — minimal input; all astronomy comes from `compute_chart(payload)`.

    Returns a dict with keys: ok, bav, bav_totals, sav{by_sign,total}, meta, warnings.
    """
    warnings: List[str] = []
    meta: Dict[str, Any] = {
        "module": "ashtakavarga(core)",
        "version": 1,
        "mode": None,
        "ayanamsa_deg": None,
        "ruleset": None,
    }

    if compute_chart is None:
        return {
            "ok": False,
            "bav": {},
            "bav_totals": {},
            "sav": {"by_sign": [0] * 12, "total": 0},
            "meta": meta,
            "warnings": ["astronomy_router_unavailable"],
        }

    # Pull authoritative chart from astronomy adapter
    try:
        chart = compute_chart(payload)
    except Exception as e:
        return {
            "ok": False,
            "bav": {},
            "bav_totals": {},
            "sav": {"by_sign": [0] * 12, "total": 0},
            "meta": meta,
            "warnings": [f"astronomy_compute_failed:{type(e).__name__}"],
        }

    # Mode & ayanamsa reporting
    meta["mode"] = (chart.get("meta", {}).get("mode")
                    or payload.get("zodiac_mode")
                    or "sidereal")
    a_deg = chart.get("meta", {}).get("ayanamsa_deg")
    if a_deg is None and str(meta["mode"]).lower() == "sidereal":
        a_deg = _safe_float(payload.get("ayanamsa"))
    meta["ayanamsa_deg"] = a_deg

    # Extract longitudes for the 7 planets
    planets_block = chart.get("planets", {})
    longitudes: Dict[str, Optional[float]] = {}
    missing: List[str] = []
    for p in PLANETS:
        v = planets_block.get(p, {})
        lon = v.get("lon", v.get("longitude"))
        fv = _safe_float(lon)
        if fv is None:
            missing.append(p)
        longitudes[p] = fv

    # Ascendant (Lagna)
    asc = None
    angles_block = chart.get("angles", chart.get("meta", {}))
    if isinstance(angles_block, dict):
        asc = _safe_float(angles_block.get("asc") or angles_block.get("ASC") or angles_block.get("Ascendant"))
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

    # Build sign indices
    signs: Dict[str, int] = {}
    for p, lon in longitudes.items():
        if lon is not None:
            signs[p] = _sign_index(lon)
    if asc is not None:
        signs["Lagna"] = _sign_index(asc)

    # Load ruleset
    requested_ruleset = str(payload.get("ruleset", "parashari-bphs")).lower()
    rules, rs_warnings, ruleset_name = _load_ruleset(requested_ruleset, payload)
    warnings.extend(rs_warnings)
    meta["ruleset"] = ruleset_name

    # BAV per planet
    bav: Dict[str, List[int]] = {}
    for giver in PLANETS:
        vec = _bav_for_planet(giver, signs, rules, warnings)
        bav[giver] = vec

    # BAV totals per planet
    bav_totals: Dict[str, int] = {k: int(sum(v)) for k, v in bav.items()}

    # SAV: column-wise sum
    sav_by_sign = [sum(bav[g][i] for g in PLANETS) for i in range(12)]
    sav_total = int(sum(sav_by_sign))

    # Consistency check
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
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Ashtakavarga engine — precision-first (gold-standard ready)

Public API:
    compute_ashtakavarga(payload: dict) -> dict
    clear_caches() -> None

Design principles (mirrors shadbala.py):
1) Authoritative astronomy only — all longitudes/angles come from
   `app.core.astronomy.compute_chart(payload)`. No ephemeris duplication here.
2) Respect `zodiac_mode` and `ayanamsa` exactly as astronomy provides. Longitudes
   we receive are already in the requested frame (sidereal/tropical).
3) Transparent `meta` + `warnings`: ruleset used, ayanāṁśa, mode, fallbacks, version, etc.
4) Deterministic, audited math. If something is missing, compute what you can,
   and emit explicit warnings. No approximations.
5) Minimal payload shape (same spirit as shadbala):
   {
     "date":"YYYY-MM-DD", "time":"HH:MM[:SS]", "tz":"IANA/Zone",
     "latitude": float, "longitude": float,
     "zodiac_mode":"sidereal"|"tropical", "ayanamsa": name|number,
     "house_system": str, "angles": {"asc": deg, "mc": deg}
   }

Output shape:
{
  "ok": true,
  "bav": { "Sun":[..12 ints..], ..., "Saturn":[..12..] },  # entries are integer bindu counts (0..8)
  "bav_totals": {"Sun": int, ..., "Saturn": int},
  "sav": {"by_sign":[..12..], "total": int},
  "meta": {"module":"ashtakavarga(core)","mode":"sidereal|tropical",
           "ayanamsa_deg":float|None,"ruleset":"parashari-bphs|custom",
           "version":1},
  "warnings":[...]
}
"""

from typing import Dict, List, Tuple, Optional, Any

# ---------- External dependency: astronomy router (authoritative longitudes) ----------
try:
    from app.core.astronomy import compute_chart  # type: ignore
except Exception:
    compute_chart = None  # type: ignore

PLANETS: List[str] = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]

# ------------------------------------------------------------------------------
# Canonical Parāśari / BPHS bindu-offset tables (giver -> receiver -> offsets 1..12)
# Offsets are counted from the receiver’s sign; add 1 bindu to that relative sign.
# Receivers are the seven planets; Lagna rows are provided separately below.
# Sources (consolidated/lined up with BPHS practice): see project docs.
# ------------------------------------------------------------------------------

_BPHS_RULES: Dict[str, Dict[str, List[int]]] = {
    # ---------------- Sun (giver) ----------------
    "Sun": {
        "Sun":     [1, 2, 4, 7, 8, 9, 10, 11],
        "Moon":    [3, 6, 10, 11],
        "Mars":    [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [3, 5, 6, 9, 10, 11, 12],
        "Jupiter": [5, 6, 9, 11],
        "Venus":   [6, 7, 12],
        "Saturn":  [1, 2, 4, 7, 8, 9, 10, 11],
    },

    # ---------------- Moon (giver) ----------------
    "Moon": {
        "Sun":     [3, 6, 7, 8, 10, 11],
        "Moon":    [1, 3, 6, 7, 10, 11],
        "Mars":    [2, 3, 5, 6, 9, 10, 11],
        "Mercury": [1, 3, 4, 5, 7, 8, 10, 11],
        "Jupiter": [1, 4, 7, 8, 10, 11, 12],
        "Venus":   [3, 4, 5, 7, 9, 10, 11],
        "Saturn":  [3, 5, 6, 11],
    },

    # ---------------- Mars (giver) ----------------
    "Mars": {
        "Sun":     [3, 5, 6, 10, 11],
        "Moon":    [3, 6, 11],
        "Mars":    [1, 2, 4, 7, 8, 10, 11],
        "Mercury": [3, 5, 6, 11],
        "Jupiter": [6, 10, 11, 12],
        "Venus":   [6, 8, 11, 12],
        "Saturn":  [1, 4, 7, 8, 9, 10, 11],
    },

    # ---------------- Mercury (giver) ----------------
    "Mercury": {
        "Sun":     [1, 3, 5, 6, 9, 10, 11, 12],
        "Moon":    [2, 4, 6, 8, 10, 11],
        "Mars":    [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [1, 3, 5, 6, 9, 10, 11, 12],
        "Jupiter": [6, 8, 11, 12],
        "Venus":   [1, 2, 3, 4, 5, 9, 10, 11],
        "Saturn":  [1, 2, 4, 7, 8, 9, 10, 11],
    },

    # ---------------- Jupiter (giver) ----------------
    "Jupiter": {
        "Sun":     [1, 2, 3, 4, 7, 8, 9, 10, 11],
        "Moon":    [2, 5, 7, 9, 11],
        "Mars":    [1, 2, 4, 7, 8, 10, 11],
        "Mercury": [1, 2, 4, 5, 6, 9, 10, 11],
        "Jupiter": [1, 2, 3, 4, 7, 8, 10, 11],
        "Venus":   [2, 5, 6, 9, 10, 11],
        "Saturn":  [3, 5, 6, 12],
    },

    # ---------------- Venus (giver) ----------------
    "Venus": {
        "Sun":     [8, 11, 12],
        "Moon":    [1, 2, 3, 4, 5, 8, 9, 11, 12],
        "Mars":    [3, 5, 6, 9, 11, 12],
        "Mercury": [3, 5, 6, 9, 11],
        "Jupiter": [5, 8, 9, 10, 11],
        "Venus":   [1, 2, 3, 4, 5, 8, 9, 10, 11],
        "Saturn":  [3, 4, 5, 8, 9, 10, 11],
    },

    # ---------------- Saturn (giver) ----------------
    "Saturn": {
        "Sun":     [1, 2, 4, 7, 8, 10, 11],
        "Moon":    [3, 6, 11],
        "Mars":    [3, 5, 6, 10, 11, 12],
        "Mercury": [6, 8, 9, 10, 11, 12],
        "Jupiter": [5, 6, 11, 12],
        "Venus":   [6, 11, 12],
        "Saturn":  [3, 5, 6, 11],
    },
}

# Lagna as receiver (offsets counted from Lagna’s sign) for each giver:
_BPHS_RULES_LAGNA: Dict[str, List[int]] = {
    "Sun":     [3, 4, 6, 10, 11, 12],
    "Moon":    [3, 6, 10, 11],
    "Mars":    [1, 3, 6, 10, 11],
    "Mercury": [1, 2, 4, 6, 8, 10, 11],
    "Jupiter": [1, 2, 4, 5, 6, 9, 10, 11],
    "Venus":   [1, 2, 3, 4, 5, 8, 9, 11],
    "Saturn":  [1, 3, 4, 6, 10, 11],
}

# ---------------- Internal caches (ruleset compilation) ----------------
_COMPILED_RULES_CACHE: Dict[str, Dict[str, Dict[str, List[int]]]] = {}


# ========================= Utilities =========================

def _wrap360(x: float) -> float:
    """Wrap angle into [0, 360)."""
    v = x % 360.0
    return v if v >= 0 else v + 360.0


def _sign_index(lon_deg: float) -> int:
    """Return 0..11 sign index from an ecliptic longitude in degrees (0° = Aries)."""
    return int(_wrap360(lon_deg) // 30.0)  # 0=Aries, ... 11=Pisces


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _bav_for_planet(
    giver: str,
    signs: Dict[str, int],
    rules: Dict[str, Dict[str, List[int]]],
    warnings: List[str],
) -> List[int]:
    """
    Compute the 12-sign BAV vector for `giver` using the supplied `rules`.

    `signs` maps body -> sign_index (0..11). Bodies include "Lagna" (if available) and the 7 planets.
    `rules[giver][receiver] = [offsets 1..12]`.

    For each receiver present in `signs` and in rules, each offset marks
    (signs[receiver] + offset - 1) % 12 as +1 in the giver's BAV.
    """
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
    """Validate a custom ruleset map coming from payload."""
    w: List[str] = []
    ok = True
    for giver, rmap in rules_map.items():
        if giver not in PLANETS:
            ok = False
            w.append(f"unknown_giver:{giver}")
        for receiver, offs in rmap.items():
            if receiver not in PLANETS and not (include_lagna and receiver in ("Lagna", "Asc")):
                ok = False
                w.append(f"unknown_receiver:{giver}->{receiver}")
            if not isinstance(offs, (list, tuple)) or not all(isinstance(o, int) for o in offs):
                ok = False
                w.append(f"invalid_offsets_type:{giver}->{receiver}")
            for o in offs:
                if o < 1 or o > 12:
                    ok = False
                    w.append(f"invalid_offset_range:{giver}->{receiver}:{o}")
    return ok, w


def _compile_rules_parashari() -> Dict[str, Dict[str, List[int]]]:
    """Compile embedded BPHS rules (+ Lagna receiver rows)."""
    compiled: Dict[str, Dict[str, List[int]]] = {}
    for giver in PLANETS:
        gr = _BPHS_RULES[giver]
        compiled[giver] = {}
        for receiver in PLANETS:
            compiled[giver][receiver] = list(gr[receiver])
        # Lagna offsets:
        compiled[giver]["Lagna"] = list(_BPHS_RULES_LAGNA[giver])
    return compiled


def _load_ruleset(name: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, List[int]]], List[str], str]:
    """
    Load/compile a ruleset.
    - name "parashari-bphs": uses embedded constants in this module.
    - name "custom": uses `payload['ruleset_map']` (validated).
    Returns: (rules, warnings, ruleset_name_effective)
    """
    warnings: List[str] = []
    tag = str(name or "parashari-bphs").lower()

    if tag == "custom":
        custom = payload.get("ruleset_map")
        if not isinstance(custom, dict):
            warnings.append("ruleset_custom_missing")
            # Fallback to BPHS
            tag = "parashari-bphs"
        else:
            ok, w = _validate_rules_map(custom, include_lagna=True)
            warnings.extend(w)
            if not ok:
                warnings.append("ruleset_custom_invalid")
                tag = "parashari-bphs"
            else:
                # normalize "Asc" -> "Lagna"
                compiled: Dict[str, Dict[str, List[int]]] = {}
                for giver, rmap in custom.items():
                    compiled[giver] = {}
                    for receiver, offs in rmap.items():
                        key = "Lagna" if receiver in ("Lagna", "Asc") else receiver
                        compiled[giver][key] = list(offs)
                return compiled, warnings, "custom"

    # Default / fallback: parashari-bphs
    cache_key = "parashari-bphs"
    if cache_key in _COMPILED_RULES_CACHE:
        return _COMPILED_RULES_CACHE[cache_key], warnings, "parashari-bphs"

    compiled = _compile_rules_parashari()
    _COMPILED_RULES_CACHE[cache_key] = compiled
    return compiled, warnings, "parashari-bphs"


def clear_caches() -> None:
    """Clear ruleset compilation caches."""
    _COMPILED_RULES_CACHE.clear()


# ========================= Core Engine =========================

def compute_ashtakavarga(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute BAV (per planet) and SAV (per sign) using Parāśari Ashtakavarga rules.

    `payload` — minimal input; all astronomy comes from `compute_chart(payload)`.

    Returns a dict with keys: ok, bav, bav_totals, sav{by_sign,total}, meta, warnings.
    """
    warnings: List[str] = []
    meta: Dict[str, Any] = {
        "module": "ashtakavarga(core)",
        "version": 1,
        "mode": None,
        "ayanamsa_deg": None,
        "ruleset": None,
    }

    if compute_chart is None:
        return {
            "ok": False,
            "bav": {},
            "bav_totals": {},
            "sav": {"by_sign": [0] * 12, "total": 0},
            "meta": meta,
            "warnings": ["astronomy_router_unavailable"],
        }

    # Pull authoritative chart from astronomy adapter
    try:
        chart = compute_chart(payload)
    except Exception as e:
        return {
            "ok": False,
            "bav": {},
            "bav_totals": {},
            "sav": {"by_sign": [0] * 12, "total": 0},
            "meta": meta,
            "warnings": [f"astronomy_compute_failed:{type(e).__name__}"],
        }

    # Mode & ayanamsa reporting
    meta["mode"] = (chart.get("meta", {}).get("mode")
                    or payload.get("zodiac_mode")
                    or "sidereal")
    a_deg = chart.get("meta", {}).get("ayanamsa_deg")
    if a_deg is None and str(meta["mode"]).lower() == "sidereal":
        a_deg = _safe_float(payload.get("ayanamsa"))
    meta["ayanamsa_deg"] = a_deg

    # Extract longitudes for the 7 planets
    planets_block = chart.get("planets", {})
    longitudes: Dict[str, Optional[float]] = {}
    missing: List[str] = []
    for p in PLANETS:
        v = planets_block.get(p, {})
        lon = v.get("lon", v.get("longitude"))
        fv = _safe_float(lon)
        if fv is None:
            missing.append(p)
        longitudes[p] = fv

    # Ascendant (Lagna)
    asc = None
    angles_block = chart.get("angles", chart.get("meta", {}))
    if isinstance(angles_block, dict):
        asc = _safe_float(angles_block.get("asc") or angles_block.get("ASC") or angles_block.get("Ascendant"))
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

    # Build sign indices
    signs: Dict[str, int] = {}
    for p, lon in longitudes.items():
        if lon is not None:
            signs[p] = _sign_index(lon)
    if asc is not None:
        signs["Lagna"] = _sign_index(asc)

    # Load ruleset
    requested_ruleset = str(payload.get("ruleset", "parashari-bphs")).lower()
    rules, rs_warnings, ruleset_name = _load_ruleset(requested_ruleset, payload)
    warnings.extend(rs_warnings)
    meta["ruleset"] = ruleset_name

    # BAV per planet
    bav: Dict[str, List[int]] = {}
    for giver in PLANETS:
        vec = _bav_for_planet(giver, signs, rules, warnings)
        bav[giver] = vec

    # BAV totals per planet
    bav_totals: Dict[str, int] = {k: int(sum(v)) for k, v in bav.items()}

    # SAV: column-wise sum
    sav_by_sign = [sum(bav[g][i] for g in PLANETS) for i in range(12)]
    sav_total = int(sum(sav_by_sign))

    # Consistency check
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
