# app/core/house.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Final

# ───────────────────────── Public ↔ Engine names ─────────────────────────

# Public names accepted by the API (include common aliases).
# We list all 18 so the API never 400's for a declared name.
SUPPORTED_HOUSE_SYSTEMS: Final[Tuple[str, ...]] = (
    # classical/modern (implemented)
    "placidus", "koch", "regiomontanus", "campanus",
    "porphyry", "alcabitius", "morinus", "topocentric",
    "equal",
    # whole-sign & aliases (implemented)
    "whole", "whole_sign", "whole-sign", "whole sign",
    # additional implemented
    "vehlow_equal", "sripati",
    # vendor-variant / research (gated → 501 Not Implemented)
    "meridian", "horizon", "carter_pe", "sunshine", "krusinski", "pullen_sd",
)

# Public/alias → engine canonical
_ALIAS_TO_ENGINE: Final[Dict[str, str]] = {
    "whole": "whole_sign",
    "whole_sign": "whole_sign",
    "whole-sign": "whole_sign",
    "whole sign": "whole_sign",
}

# Engine canonical → preferred public label
_ENGINE_TO_PUBLIC: Final[Dict[str, str]] = {
    "whole_sign": "whole",
}

# Systems intentionally not implemented yet (HTTP layer maps to 501)
GATED_NOT_IMPLEMENTED: Final[frozenset[str]] = frozenset({
    # vendor-variant (ambiguous specs)
    "horizon", "carter_pe", "sunshine", "pullen_sd",
    # research-claimed (awaiting gold vectors/spec)
    "meridian", "krusinski",
})

# ───────────────────────── Env / Policy knobs ─────────────────────────

POLAR_SOFT_LIMIT_DEG: Final[float] = float(os.getenv("ASTRO_POLAR_SOFT_LAT", "66.0"))
POLAR_HARD_LIMIT_DEG: Final[float] = float(os.getenv("ASTRO_POLAR_HARD_LAT", "80.0"))
POLAR_ABSOLUTE_LIMIT_DEG: Final[float] = 89.999999  # guard near poles
POLAR_POLICY: Final[str] = os.getenv("ASTRO_POLAR_POLICY", "fallback_to_equal_above_66deg")

# Systems that misbehave near the poles (time-division/numeric issues)
_HARD_REJECT_AT_POLAR: Final[frozenset[str]] = frozenset({"placidus", "koch", "topocentric", "alcabitius"})
_RISKY_AT_POLAR: Final[frozenset[str]] = frozenset({
    "placidus", "koch", "regiomontanus", "campanus", "topocentric", "alcabitius", "morinus",
    # robust angular/segment systems intentionally omitted: equal/whole/porphyry/vehlow_equal/sripati
})

# Numeric multi-fallback enabled?
NUMERIC_FALLBACK_ENABLED: Final[bool] = os.getenv("ASTRO_HOUSES_NUMERIC_FALLBACK", "1").lower() in ("1", "true", "yes", "on")

# Optional JSON mapping of custom fallback chains:
#   ASTRO_HOUSES_FALLBACK_JSON='{"placidus":["koch","regiomontanus","campanus","porphyry","equal","whole"]}'
_FALLBACK_JSON: Optional[str] = os.getenv("ASTRO_HOUSES_FALLBACK_JSON")

# Defaults for diagnostics/validation passthrough to the engine
DEFAULT_DIAGNOSTICS: Final[bool] = os.getenv("ASTRO_HOUSES_DEBUG", "").lower() in ("1", "true", "yes", "on")
DEFAULT_VALIDATION: Final[bool] = os.getenv("ASTRO_HOUSES_VALIDATE", "").lower() in ("1", "true", "yes", "on")

# ───────────────────────── Engine import ─────────────────────────

try:
    from app.core.houses_advanced import PreciseHouseCalculator, HouseData
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "houses_advanced backend unavailable. Install numpy>=1.26 and pyerfa>=2.0.1. "
        f"Original import error: {type(_e).__name__}: {_e}"
    ) from _e

# ───────────────────────── Metrics (lazy; avoid cycles) ─────────────────────────

def _met_warn(kind: str) -> None:
    try:
        from app.main import MET_WARNINGS  # type: ignore
        MET_WARNINGS.labels(kind=kind).inc()
    except Exception:
        pass

def _met_fallback(requested: str, fallback: str) -> None:
    try:
        from app.main import MET_FALLBACKS  # type: ignore
        MET_FALLBACKS.labels(requested=requested, fallback=fallback).inc()
    except Exception:
        pass

# ───────────────────────── Utilities ─────────────────────────

def list_supported_house_systems() -> List[str]:
    """Return the public list (includes accepted aliases)."""
    return list(SUPPORTED_HOUSE_SYSTEMS)

def _normalize_public_name(name: Optional[str]) -> str:
    """
    Normalize a user-facing house-system to our *public* label.
    (Still public, not engine; aliases collapse to a single label.)
    """
    if not name:
        return "placidus"
    key = name.strip().lower()
    if key not in SUPPORTED_HOUSE_SYSTEMS:
        raise ValueError(f"unsupported house system: {name}")
    return "whole" if key in {"whole_sign", "whole-sign", "whole sign"} else key

def _public_to_engine(system_public: str) -> str:
    """Public normalized name → engine canonical."""
    return _ALIAS_TO_ENGINE.get(system_public, system_public)

def _engine_to_public(system_engine: str) -> str:
    """Engine canonical → preferred public label."""
    return _ENGINE_TO_PUBLIC.get(system_engine, system_engine)

def _needs_polar_fallback(public_system: str, latitude: float, limit: float) -> bool:
    return abs(latitude) > limit and public_system in _RISKY_AT_POLAR

def _allowed_at_lat(public_system: str, lat: float, hard_lim: float) -> bool:
    """True if we should attempt this system at the given latitude."""
    return not (abs(lat) >= hard_lim and public_system in _HARD_REJECT_AT_POLAR)

def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def _default_fallback_chain_for(requested_public: str) -> List[str]:
    """
    Opinionated fallback chain per system.
    Prioritize “similar” time-division engines first, then robust angular ones.
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
        "vehlow_equal":  ["equal", "whole"],
        "sripati":       ["porphyry", "equal", "whole"],
    }
    return chains.get(requested_public, ["equal", "whole"])

def _fallback_chain(requested_public: str) -> List[str]:
    """
    Merge env-provided chain (if any) with defaults; always end with equal/whole.
    The chain is returned as *public* labels (aliases normalized).
    """
    requested_public = _normalize_public_name(requested_public)
    default_chain = _default_fallback_chain_for(requested_public)

    env_chain: List[str] = []
    if _FALLBACK_JSON:
        try:
            mapping = json.loads(_FALLBACK_JSON)
            maybe = mapping.get(requested_public)
            if isinstance(maybe, list):
                env_chain = [
                    _normalize_public_name(str(x))
                    for x in maybe
                    if str(x).lower() in SUPPORTED_HOUSE_SYSTEMS
                ]
        except Exception:
            pass

    chain = [requested_public] + env_chain + default_chain + ["equal", "whole"]
    return _dedupe([_normalize_public_name(x) for x in chain])

# ───────────────────────── Policy façade ─────────────────────────

def compute_houses_with_policy(
    *,
    # primary
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    system: Optional[str] = None,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    jd_ut: Optional[float] = None,              # ignored in strict mode
    diagnostics: Optional[bool] = None,
    validation: Optional[bool] = None,

    # legacy aliases (lenient)
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
    House computation façade that enforces:
      • strict time scales (jd_tt + jd_ut1 required),
      • polar safety (fallback/reject per policy),
      • multi-step numeric fallback chain,
      • stable, API-friendly payload (keeps cusps_deg for compatibility),
      • optional passthrough of diagnostics & self-validation.

    Raises:
      ValueError           — bad inputs or policy rejection
      NotImplementedError  — declared but gated systems (HTTP 501)
      RuntimeError         — all fallbacks failed
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
    requested_public = _normalize_public_name(requested_public)

    # Gate vendor-variant / research systems for clear 501
    if requested_public in GATED_NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"house_system '{requested_public}' is declared but intentionally not implemented yet "
            f"(awaiting gold vectors/specs)."
        )

    polar_policy = (polar_policy or POLAR_POLICY).strip()
    soft_lim = float(polar_soft_limit if polar_soft_limit is not None else POLAR_SOFT_LIMIT_DEG)
    hard_lim = float(polar_hard_limit if polar_hard_limit is not None else POLAR_HARD_LIMIT_DEG)

    # strict time scales
    if jd_tt is None or jd_ut1 is None:
        raise ValueError(
            "houses: strict mode requires jd_tt and jd_ut1 (no UT≈UTC shortcuts). "
            "Provide both timescales derived from UTC via leap seconds & ΔT."
        )

    # flags (CLI/env/param)
    if diagnostics is None:
        diagnostics = enable_diagnostics
    if diagnostics is None:
        diagnostics = DEFAULT_DIAGNOSTICS
    if validation is None:
        validation = DEFAULT_VALIDATION

    warnings: List[str] = []

    # ---- polar rules
    if abs(lat_f) >= hard_lim and requested_public in _HARD_REJECT_AT_POLAR:
        warnings.append(
            f"Requested '{requested_public}' is undefined/unstable at latitude {lat_f:.2f}° "
            f"(≥ hard limit {hard_lim}°). Will try fallbacks."
        )
        _met_warn("polar_hard_limit")

    if polar_policy == "reject_above_66deg" and _needs_polar_fallback(requested_public, lat_f, soft_lim):
        _met_warn("polar_reject_strict")
        raise ValueError(f"house_system '{requested_public}' is unstable above |lat|>{soft_lim}°")

    # Build the chain and filter by hard-limit allowance
    chain_public = [s for s in _fallback_chain(requested_public) if _allowed_at_lat(s, lat_f, hard_lim)]
    if not chain_public:
        raise ValueError("no allowed house systems at this latitude under current policy")

    # If soft policy and requested is risky, prefer robust systems first
    if polar_policy == "fallback_to_equal_above_66deg" and _needs_polar_fallback(chain_public[0], lat_f, soft_lim):
        def robust_rank(s: str) -> int:
            if s in ("equal", "whole"): return 0
            if s == "porphyry": return 1
            return 2
        chain_public.sort(key=robust_rank)
        warnings.append(
            f"Soft polar policy: '{requested_public}' is risky at |lat|>{soft_lim}°. "
            f"Prioritizing robust systems first: {chain_public[:3]}..."
        )
        _met_warn("polar_soft_fallback")

    # ---- compute with numeric safety net & fallbacks
    calc = PreciseHouseCalculator(
        require_strict_timescales=True,
        enable_diagnostics=bool(diagnostics),
        enable_validation=bool(validation),
    )

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

    for sys_public in chain_public:
        # obey the toggle: only the first item if fallback disabled
        if sys_public != chain_public[0] and not NUMERIC_FALLBACK_ENABLED:
            break
        try:
            hd = _compute(sys_public)
            used_public = _engine_to_public(hd.system)
            if sys_public != requested_public:
                warnings.append(f"Fallback applied: '{requested_public}' → '{sys_public}'.")
                _met_fallback(requested=requested_public, fallback=sys_public)
            break
        except Exception as e:
            last_err = e
            continue

    if hd is None or used_public is None:
        if last_err:
            raise RuntimeError(f"houses engine failed for all fallbacks (last: {type(last_err).__name__}: {last_err})")
        raise RuntimeError("houses engine failed and no fallback succeeded")

    # ---- payload (stable shape; keep legacy keys)
    all_warnings = list(warnings) + list(hd.warnings or [])

    payload: Dict[str, Any] = {
        # what was asked vs. used
        "requested_system": requested_public,       # public label requested
        "system": used_public,                      # public label used after policy/fallback
        "engine_system": hd.system,                 # engine canonical actually used
        # angles & cusps (deg)
        "asc": hd.ascendant, "mc": hd.midheaven,
        "asc_deg": hd.ascendant, "mc_deg": hd.midheaven,  # legacy
        "cusps": hd.cusps, "cusps_deg": hd.cusps,         # legacy alias
        "angles": {
            "ASC": {"deg": hd.ascendant},
            "MC":  {"deg": hd.midheaven},
        },
        # auxiliary points
        "vertex": hd.vertex,
        "eastpoint": hd.eastpoint,
        # policy echo
        "policy": {
            "polar_policy": polar_policy,
            "polar_soft_limit_deg": soft_lim,
            "polar_hard_limit_deg": hard_lim,
            "absolute_pole_guard_deg": POLAR_ABSOLUTE_LIMIT_DEG,
            "hard_reject_systems_at_polar": sorted(_HARD_REJECT_AT_POLAR),
            "risky_at_polar": sorted(_RISKY_AT_POLAR),
            "numeric_fallback_enabled": NUMERIC_FALLBACK_ENABLED,
            "fallback_chain_tried": chain_public,
        },
        # timescale echo (useful for client sanity checks)
        "meta": {
            "timescales": {
                "jd_tt": float(jd_tt),
                "jd_ut1": float(jd_ut1),
                "jd_ut_legacy": float(jd_ut) if jd_ut is not None else None,
            },
            "diagnostics_enabled": bool(diagnostics),
            "validation_enabled": bool(validation),
        },
        # warnings (policy + engine)
        "warnings": all_warnings,
    }

    # Optional attachments
    if getattr(hd, "solver_stats", None):
        payload["solver_stats"] = hd.solver_stats
    if getattr(hd, "validation_results", None):
        # keep it compact; clients can opt to display summary
        payload["validation_results"] = [r._asdict() for r in hd.validation_results]
    if getattr(hd, "error_budget", None):
        eb = hd.error_budget
        payload["error_budget"] = {
            "coordinate_precision": eb.coordinate_precision,
            "algorithm_truncation": eb.algorithm_truncation,
            "time_scale_uncertainty": eb.time_scale_uncertainty,
            "reference_comparison": eb.reference_comparison,
            "total_rss": eb.total_rss,
            "certified": eb.certify_accuracy(),
        }

    return payload

# ───────────────────────── Public API ─────────────────────────

__all__ = [
    "compute_houses_with_policy",
    "list_supported_house_systems",
    "SUPPORTED_HOUSE_SYSTEMS",
    "POLAR_POLICY",
    "POLAR_SOFT_LIMIT_DEG",
    "POLAR_HARD_LIMIT_DEG",
    "POLAR_ABSOLUTE_LIMIT_DEG",
    "GATED_NOT_IMPLEMENTED",
]
