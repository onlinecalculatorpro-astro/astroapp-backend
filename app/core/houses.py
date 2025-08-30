# app/core/house.py
from __future__ import annotations
import os
from typing import Dict, Any, List, Tuple, Optional

# Core math engine (all numerical work lives there)
from app.core.houses_advanced import PreciseHouseCalculator, HouseData

# -----------------------------------------------------------------------------
# Public API (external) house-system names
# -----------------------------------------------------------------------------
SUPPORTED_HOUSE_SYSTEMS: Tuple[str, ...] = (
    "placidus",
    "whole",        # public enum; engine uses "whole_sign"
    "equal",
    "koch",
    "regiomontanus",
    "campanus",
    "porphyry",
    "alcabitius",
    "morinus",
    "topocentric",
)

# Map public -> engine-internal names
_ALIAS_TO_ENGINE = {
    "whole": "whole_sign",
}
# And the reverse (for clean output)
_ENGINE_TO_PUBLIC = {v: k for k, v in _ALIAS_TO_ENGINE.items()}

# -----------------------------------------------------------------------------
# Polar policy & safety rails
# -----------------------------------------------------------------------------
# Soft warning/fallback threshold (systems may distort above this)
POLAR_SOFT_LIMIT_DEG: float = 66.0
# Hard reject threshold (time-division systems are undefined/unstable here)
POLAR_HARD_LIMIT_DEG: float = 80.0
# Absolute pole guard (no house system defined at the poles)
POLAR_ABSOLUTE_LIMIT_DEG: float = 89.999999

# Systems that become undefined/unstable as |lat| → poles (hard reject ≥ 80°)
_HARD_REJECT_AT_POLAR = {
    "placidus", "koch", "topocentric", "alcabitius",
}

# Systems that are risky (we soft-fallback above 66° if policy says so)
_RISKY_AT_POLAR = {
    "placidus", "koch", "regiomontanus", "campanus",
    "topocentric", "alcabitius", "morinus",
    # Note: porphyry/equal/whole are generally robust
}

# Polar policy modes
# - "fallback_to_equal_above_66deg": soft fallback for risky systems
# - "reject_above_66deg": hard reject already at soft threshold
POLAR_POLICY: str = "fallback_to_equal_above_66deg"


def list_supported_house_systems() -> List[str]:
    """Return the public list used by validators and /api/config."""
    return list(SUPPORTED_HOUSE_SYSTEMS)


def _normalize_requested(system: str | None) -> str:
    """Validate & normalize the requested public enum."""
    if not system:
        return "placidus"
    s = system.strip().lower()
    if s not in SUPPORTED_HOUSE_SYSTEMS:
        raise ValueError(f"unsupported house system: {system}")
    return s


def _public_to_engine(system: str) -> str:
    """Translate public enum -> engine-internal name."""
    return _ALIAS_TO_ENGINE.get(system, system)


def _engine_to_public(system: str) -> str:
    """Translate engine-internal name -> public enum."""
    return _ENGINE_TO_PUBLIC.get(system, system)


def _needs_polar_fallback(system: str, latitude: float, limit: float) -> bool:
    return abs(latitude) > limit and system in _RISKY_AT_POLAR


def compute_houses_with_policy(
    *,
    latitude: float,
    longitude: float,
    # Legacy param retained for signature compatibility; not used in strict mode by the engine
    jd_ut: float,
    requested_house_system: str | None = None,
    # Policy knobs (override in tests if needed)
    polar_policy: str = POLAR_POLICY,
    polar_soft_limit: float = POLAR_SOFT_LIMIT_DEG,
    polar_hard_limit: float = POLAR_HARD_LIMIT_DEG,
    # Strict timescales required by the v4 engine (NO shortcuts)
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    # Diagnostics toggle (default: read env ASTRO_HOUSES_DEBUG)
    enable_diagnostics: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Policy façade for house computation.

    Responsibilities:
      - Validate latitude safety rails (absolute pole guard).
      - Validate/normalize the requested system (public enums).
      - Enforce polar policies:
          * hard reject at >= polar_hard_limit for time-division systems
          * optional soft fallback to 'equal' above polar_soft_limit
      - Enforce strict timescales (jd_tt & jd_ut1 required; no UT≈UTC approximations).
      - Delegate to PreciseHouseCalculator (engine-internal names).
      - Return normalized dict with requested/effective system, angles, cusps, and warnings.
    """

    # -------- Absolute latitude guard (never compute at/near poles) --------
    if not (-90.0 < latitude < 90.0):
        # Match validators: poles are undefined for house systems
        raise ValueError(
            "Latitude must be strictly between -90 and 90 degrees; poles are undefined for house systems."
        )
    if abs(latitude) >= POLAR_ABSOLUTE_LIMIT_DEG:
        raise ValueError(
            f"latitude {latitude:.8f}° is at/near the pole; house systems are undefined. "
            "Provide latitude strictly between -90° and 90°."
        )

    # -------- Validate requested system (public) --------
    requested_public = _normalize_requested(requested_house_system)
    effective_public = requested_public
    warnings: List[str] = []

    # -------- Hard-reject near poles (time-division systems) --------
    if abs(latitude) >= polar_hard_limit and requested_public in _HARD_REJECT_AT_POLAR:
        allowed = ["equal", "whole", "porphyry", "regiomontanus", "campanus", "morinus"]
        raise ValueError(
            f"house_system '{requested_public}' is undefined/unstable at latitude {latitude:.2f}°. "
            f"Allowed at this latitude: {', '.join(allowed)}"
        )

    # -------- Soft polar policy (above soft limit) --------
    if polar_policy == "reject_above_66deg" and _needs_polar_fallback(
        requested_public, latitude, polar_soft_limit
    ):
        raise ValueError(
            f"house_system '{requested_public}' is unstable above |lat|>{polar_soft_limit}°"
        )

    if polar_policy == "fallback_to_equal_above_66deg" and _needs_polar_fallback(
        requested_public, latitude, polar_soft_limit
    ):
        warnings.append(
            f"Requested house_system '{requested_public}' is unstable above |lat|>{polar_soft_limit}°. "
            "Fallback to 'equal' applied."
        )
        effective_public = "equal"

    # -------- Strict timescales (no shortcuts) --------
    if jd_tt is None or jd_ut1 is None:
        raise ValueError(
            "houses: strict mode requires jd_tt and jd_ut1 (no UT≈UTC shortcuts). "
            "Provide both timescales derived from UTC via leap seconds & ΔT."
        )

    # -------- Diagnostics toggle --------
    if enable_diagnostics is None:
        enable_diagnostics = os.getenv("ASTRO_HOUSES_DEBUG", "").lower() in ("1", "true", "yes", "on")

    # -------- Translate to engine-internal system name --------
    engine_system = _public_to_engine(effective_public)

    # -------- Delegate to math engine (strict mode ON) --------
    calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=enable_diagnostics)
    hd: HouseData = calc.calculate_houses(
        latitude=latitude,
        longitude=longitude,
        jd_ut=jd_ut,           # not used in strict mode; kept for signature stability
        house_system=engine_system,
        jd_tt=jd_tt,
        jd_ut1=jd_ut1,
    )

    # -------- Merge warnings (policy + engine) --------
    all_warnings = list(warnings) + list(hd.warnings or [])

    # -------- Normalize system back to public enum in the output --------
    effective_public_out = _engine_to_public(hd.system)

    # -------- Build response --------
    payload: Dict[str, Any] = {
        "requested_house_system": requested_public,
        "house_system": effective_public_out,   # effective system actually used
        "asc_deg": hd.ascendant,
        "mc_deg": hd.midheaven,
        "cusps_deg": hd.cusps,
        "vertex": hd.vertex,
        "eastpoint": hd.eastpoint,
        "warnings": all_warnings,
        "policy": {
            "polar_policy": polar_policy,
            "polar_soft_limit_deg": polar_soft_limit,
            "polar_hard_limit_deg": polar_hard_limit,
            "absolute_pole_guard_deg": POLAR_ABSOLUTE_LIMIT_DEG,
            "hard_reject_systems_at_polar": sorted(_HARD_REJECT_AT_POLAR),
            "risky_at_polar": sorted(_RISKY_AT_POLAR),
        },
    }
    # Include solver statistics only when diagnostics are enabled
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
