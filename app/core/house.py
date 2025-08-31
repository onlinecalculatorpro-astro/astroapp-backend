# app/core/house.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Final

# ───────────────────────── Public ↔ Engine names ─────────────────────────

# Public names we accept from the API/client. Include common aliases.
SUPPORTED_HOUSE_SYSTEMS: Final[Tuple[str, ...]] = (
    "placidus",
    "equal",
    "whole",        # alias family for whole-sign
    "whole_sign",
    "whole-sign",
    "whole sign",
    "koch",
    "regiomontanus",
    "campanus",
    "porphyry",
    "alcabitius",
    "morinus",
    "topocentric",
)

# Map public/alias → engine canonical
_ALIAS_TO_ENGINE: Final[Dict[str, str]] = {
    "whole": "whole_sign",
    "whole_sign": "whole_sign",
    "whole-sign": "whole_sign",
    "whole sign": "whole_sign",
}

# Map engine canonical → public preferred label
_ENGINE_TO_PUBLIC: Final[Dict[str, str]] = {
    "whole_sign": "whole",
}

# ───────────────────────── Polar policy knobs ─────────────────────────

POLAR_SOFT_LIMIT_DEG: Final[float] = float(os.getenv("ASTRO_POLAR_SOFT_LAT", "66.0"))
POLAR_HARD_LIMIT_DEG: Final[float] = float(os.getenv("ASTRO_POLAR_HARD_LAT", "80.0"))
POLAR_ABSOLUTE_LIMIT_DEG: Final[float] = 89.999999  # hard guard near poles
POLAR_POLICY: Final[str] = os.getenv("ASTRO_POLAR_POLICY", "fallback_to_equal_above_66deg")

# Systems known to be problematic near the poles
_HARD_REJECT_AT_POLAR: Final[set[str]] = {"placidus", "koch", "topocentric", "alcabitius"}
_RISKY_AT_POLAR: Final[set[str]] = {
    "placidus", "koch", "regiomontanus", "campanus", "topocentric", "alcabitius", "morinus",
}

# Numeric fallback enabled?
NUMERIC_FALLBACK_ENABLED: Final[bool] = os.getenv("ASTRO_HOUSES_NUMERIC_FALLBACK", "1").lower() in (
    "1", "true", "yes", "on"
)

# Optional JSON to override chains:
#   ASTRO_HOUSES_FALLBACK_JSON='{"placidus":["koch","regiomontanus","campanus","porphyry","equal","whole"]}'
_FALLBACK_JSON: Optional[str] = os.getenv("ASTRO_HOUSES_FALLBACK_JSON")

# ───────────────────────── Engine import ─────────────────────────

try:
    from app.core.houses_advanced import PreciseHouseCalculator, HouseData
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "houses_advanced backend unavailable. Install numpy>=1.26 and pyerfa>=2.0.1. "
        f"Original import error: {type(_e).__name__}: {_e}"
    ) from _e


# ───────────────────────── Metrics (lazy, no import cycles) ─────────────────────────

def _met_warn(kind: str) -> None:
    """Increment astro_warning_total{kind=...} if metrics are available."""
    try:
        from app.main import MET_WARNINGS  # type: ignore
        MET_WARNINGS.labels(kind=kind).inc()
    except Exception:
        pass  # metrics are optional; never break domain logic


def _met_fallback(requested: str, fallback: str) -> None:
    """Increment astro_house_fallback_total{requested=...,fallback=...} if available."""
    try:
        from app.main import MET_FALLBACKS  # type: ignore
        MET_FALLBACKS.labels(requested=requested, fallback=fallback).inc()
    except Exception:
        pass


# ───────────────────────── Utilities ─────────────────────────

def list_supported_house_systems() -> List[str]:
    """Public list (includes aliases we accept)."""
    return list(SUPPORTED_HOUSE_SYSTEMS)


def _normalize_system(s: Optional[str]) -> str:
    """Normalize a user-facing house-system name to our public label (still public, not engine)."""
    if not s:
        return "placidus"
    key = s.strip().lower()
    if key not in SUPPORTED_HOUSE_SYSTEMS:
        raise ValueError(f"unsupported house system: {s}")
    # Collapse aliases to a single public label where appropriate
    if key in {"whole_sign", "whole-sign", "whole sign"}:
        return "whole"
    return key


def _public_to_engine(system: str) -> str:
    """Convert a normalized public system name to the engine canonical name."""
    return _ALIAS_TO_ENGINE.get(system, system)


def _engine_to_public(system: str) -> str:
    """Convert engine canonical name back to a preferred public label."""
    return _ENGINE_TO_PUBLIC.get(system, system)


def _needs_polar_fallback(system: str, latitude: float, limit: float) -> bool:
    return abs(latitude) > limit and system in _RISKY_AT_POLAR


def _allowed_at_lat(public_system: str, lat: float, hard_lim: float) -> bool:
    """True if we should attempt this system at the given latitude."""
    if abs(lat) >= hard_lim and public_system in _HARD_REJECT_AT_POLAR:
        return False
    return True


def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _default_fallback_chain_for(requested_public: str) -> List[str]:
    """
    A robust, opinionated fallback chain per system.
    Order tries similar time-division systems first, then robust angular ones.
    """
    chains: Dict[str, List[str]] = {
        "placidus":      ["koch", "regiomontanus", "campanus", "porphyry", "equal", "whole"],
        "koch":          ["placidus", "regiomontanus", "campanus", "porphyry", "equal", "whole"],
        "topocentric":   ["placidus", "koch", "regiomontanus", "campanus", "porphyry", "equal", "whole"],
        "alcabitius":    ["regiomontanus", "campanus", "porphyry", "equal", "whole"],
        "morinus":       ["regiomontanus", "campanus", "porphyry", "equal", "whole"],
        "regiomontanus": ["campanus", "porphyry", "equal", "whole"],
        "campanus":      ["regiomontanus", "porphyry", "equal", "whole"],
        "porphyry":      ["equal", "whole"],
        "equal":         ["porphyry", "whole"],
        "whole":         ["equal", "porphyry"],
    }
    return chains.get(requested_public, ["equal", "whole"])


def _fallback_chain(requested_public: str) -> List[str]:
    """
    Merge env-provided chain (if any) with our defaults; always end with equal/whole for robustness.
    """
    requested_public = _normalize_system(requested_public)
    default_chain = _default_fallback_chain_for(requested_public)

    env_chain: List[str] = []
    if _FALLBACK_JSON:
        try:
            mapping = json.loads(_FALLBACK_JSON)
            if isinstance(mapping, dict):
                maybe = mapping.get(requested_public)
                if isinstance(maybe, list):
                    env_chain = [str(x).lower() for x in maybe if str(x).lower() in SUPPORTED_HOUSE_SYSTEMS]
        except Exception:
            pass

    chain = [requested_public] + env_chain + default_chain + ["equal", "whole"]
    # Normalize/alias collapse and dedupe
    chain = [_normalize_system(x) for x in chain]
    chain = _dedupe(chain)
    return chain


# ───────────────────────── Policy façade ─────────────────────────

def compute_houses_with_policy(
    *,
    # primary
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    system: Optional[str] = None,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    jd_ut: Optional[float] = None,           # ignored in strict mode
    diagnostics: Optional[bool] = None,

    # legacy aliases (be lenient for callers)
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    requested_house_system: Optional[str] = None,
    enable_diagnostics: Optional[bool] = None,

    # policy overrides
    polar_policy: Optional[str] = None,
    polar_soft_limit: Optional[float] = None,
    polar_hard_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """
    House computation façade with:
      • strict timescales (requires jd_tt & jd_ut1),
      • polar safety (fallback/reject per policy),
      • multi-step numeric fallback chain,
      • stable, API-friendly payload (includes `cusps_deg` for compatibility).
    """
    # ---- inputs & normalization
    latitude = lat if lat is not None else latitude
    longitude = lon if lon is not None else longitude
    if latitude is None or longitude is None:
        raise ValueError("lat/lon are required")

    lat_f = float(latitude)
    lon_f = float(longitude)

    if not (-90.0 < lat_f < 90.0):
        raise ValueError("latitude must be strictly between -90 and 90 degrees")
    if abs(lat_f) >= POLAR_ABSOLUTE_LIMIT_DEG:
        raise ValueError(f"latitude {lat_f:.8f}° is at/near the pole; house systems are undefined")

    requested_public = system if system is not None else requested_house_system
    requested_public = _normalize_system(requested_public)

    polar_policy = (polar_policy or POLAR_POLICY).strip()
    soft_lim = float(polar_soft_limit if polar_soft_limit is not None else POLAR_SOFT_LIMIT_DEG)
    hard_lim = float(polar_hard_limit if polar_hard_limit is not None else POLAR_HARD_LIMIT_DEG)

    warnings: List[str] = []

    # ---- strict timescales
    if jd_tt is None or jd_ut1 is None:
        raise ValueError(
            "houses: strict mode requires jd_tt and jd_ut1 (no UT≈UTC shortcuts). "
            "Provide both timescales derived from UTC via leap seconds & ΔT."
        )

    # diagnostics flag
    if diagnostics is None:
        diagnostics = enable_diagnostics
    if diagnostics is None:
        diagnostics = os.getenv("ASTRO_HOUSES_DEBUG", "").lower() in ("1", "true", "yes", "on")

    # ---- polar rules
    # Hard rejection only blocks specific systems; we can still try safer ones from the chain.
    if abs(lat_f) >= hard_lim and requested_public in _HARD_REJECT_AT_POLAR:
        warnings.append(
            f"Requested '{requested_public}' is undefined/unstable at latitude {lat_f:.2f}° (≥ hard limit {hard_lim}°). "
            "Will try fallbacks."
        )
        _met_warn("polar_hard_limit")

    if polar_policy == "reject_above_66deg" and _needs_polar_fallback(requested_public, lat_f, soft_lim):
        # In strict-reject policy, don't attempt chain for risky systems above soft limit.
        _met_warn("polar_reject_strict")
        raise ValueError(f"house_system '{requested_public}' is unstable above |lat|>{soft_lim}°")

    # Build the chain and filter out candidates not allowed at this latitude
    chain = [s for s in _fallback_chain(requested_public) if _allowed_at_lat(s, lat_f, hard_lim)]
    if not chain:
        raise ValueError("no allowed house systems at this latitude under current policy")

    # If soft-policy and requested is risky, prefer robust systems first by reordering:
    if polar_policy == "fallback_to_equal_above_66deg" and _needs_polar_fallback(chain[0], lat_f, soft_lim):
        # Move robust angular systems to the front while keeping relative order
        def robust_rank(s: str) -> int:
            if s in ("equal", "whole"): return 0
            if s == "porphyry": return 1
            return 2
        chain.sort(key=robust_rank)
        warnings.append(
            f"Soft polar policy: '{requested_public}' is risky at |lat|>{soft_lim}°. "
            f"Reordered fallback priority to robust systems first: {chain[:3]}..."
        )
        _met_warn("polar_soft_fallback")

    # ---- compute with numeric safety net and multi-step fallback
    calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=bool(diagnostics))

    last_err: Optional[Exception] = None
    used_public: Optional[str] = None
    hd: Optional[HouseData] = None

    def _compute(sys_public: str) -> HouseData:
        return calc.calculate_houses(
            latitude=lat_f,
            longitude=lon_f,
            jd_ut=float(jd_ut) if jd_ut is not None else 0.0,  # ignored in strict mode
            house_system=_public_to_engine(sys_public),
            jd_tt=float(jd_tt),
            jd_ut1=float(jd_ut1),
        )

    for sys_public in chain:
        # If numeric fallback is disabled and this is not the very first attempt, stop.
        if sys_public != chain[0] and not NUMERIC_FALLBACK_ENABLED:
            break
        try:
            hd = _compute(sys_public)
            used_public = _engine_to_public(hd.system)
            if sys_public != requested_public:
                warnings.append(f"Fallback to '{sys_public}' applied.")
                _met_fallback(requested=requested_public, fallback=sys_public)
            break
        except Exception as e:
            last_err = e
            continue

    if hd is None or used_public is None:
        # Everything failed
        if last_err:
            raise RuntimeError(f"houses engine failed for all fallbacks (last: {type(last_err).__name__}: {last_err})")
        raise RuntimeError("houses engine failed and no fallback succeeded")

    # ---- build payload (compat: include asc/mc and asc_deg/mc_deg and cusps_deg)
    all_warnings = list(warnings) + list(hd.warnings or [])

    payload: Dict[str, Any] = {
        "requested_system": requested_public,         # public label requested
        "system": used_public,                        # public label used after policy/fallback
        "engine_system": hd.system,                   # engine canonical actually used
        "cusps": hd.cusps,                            # list[12] degrees
        "cusps_deg": hd.cusps,                        # ← compatibility with existing clients/tests
        "asc": hd.ascendant,
        "mc": hd.midheaven,
        "asc_deg": hd.ascendant,                      # compatibility helpers
        "mc_deg": hd.midheaven,
        "angles": {                                   # optional rich angles
            "ASC": {"deg": hd.ascendant},
            "MC":  {"deg": hd.midheaven},
        },
        "vertex": hd.vertex,
        "eastpoint": hd.eastpoint,
        "warnings": all_warnings,
        "policy": {
            "polar_policy": polar_policy,
            "polar_soft_limit_deg": soft_lim,
            "polar_hard_limit_deg": hard_lim,
            "absolute_pole_guard_deg": POLAR_ABSOLUTE_LIMIT_DEG,
            "hard_reject_systems_at_polar": sorted(_HARD_REJECT_AT_POLAR),
            "risky_at_polar": sorted(_RISKY_AT_POLAR),
            "numeric_fallback_enabled": NUMERIC_FALLBACK_ENABLED,
            "fallback_chain_tried": chain,
        },
    }
    if getattr(hd, "solver_stats", None):
        payload["solver_stats"] = hd.solver_stats
    return payload


__all__ = [
    "compute_houses_with_policy",
    "list_supported_house_systems",
    "SUPPORTED_HOUSE_SYSTEMS",
    "POLAR_POLICY",
    "POLAR_SOFT_LIMIT_DEG",
    "POLAR_HARD_LIMIT_DEG",
    "POLAR_ABSOLUTE_LIMIT_DEG",
]
