# app/core/house.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

# Core math engine (keep all numerical work here)
from app.core.houses_advanced import PreciseHouseCalculator, HouseData

# ---------------------------------------------
# Public API (external) house-system names
# ---------------------------------------------
SUPPORTED_HOUSE_SYSTEMS: Tuple[str, ...] = (
    "placidus",
    "whole",        # public name; engine uses "whole_sign"
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

# Systems known to be unstable near the poles (|lat| ≳ 66°)
_RISKY_AT_POLAR = {
    "placidus",
    "koch",
    "regiomontanus",
    "campanus",
    "topocentric",
    "alcabitius",
    "morinus",
    # Note: "porphyry" and "equal"/"whole" are typically robust
}

# Polar policy knobs
POLAR_LIMIT_DEG: float = 66.0
POLAR_POLICY: str = "fallback_to_equal_above_66deg"  # or "reject_above_66deg"


def list_supported_house_systems() -> List[str]:
    """Return the public list used by validators and /api/config."""
    return list(SUPPORTED_HOUSE_SYSTEMS)


def _normalize_requested(system: str | None) -> str:
    if not system:
        return "placidus"
    s = system.strip().lower()
    if s not in SUPPORTED_HOUSE_SYSTEMS:
        raise ValueError(f"unsupported house system: {system}")
    return s


def _public_to_engine(system: str) -> str:
    """Translate a public/system enum to the engine-internal name."""
    return _ALIAS_TO_ENGINE.get(system, system)


def _engine_to_public(system: str) -> str:
    """Translate engine-internal name back to the public enum."""
    return _ENGINE_TO_PUBLIC.get(system, system)


def _needs_polar_fallback(system: str, latitude: float, limit: float) -> bool:
    return abs(latitude) > limit and system in _RISKY_AT_POLAR


def compute_houses_with_policy(
    *,
    latitude: float,
    longitude: float,
    jd_ut: float,
    requested_house_system: str | None = None,
    polar_policy: str = POLAR_POLICY,
    polar_limit: float = POLAR_LIMIT_DEG,
) -> Dict[str, Any]:
    """
    Policy façade for house computation.

    - Validates and normalizes the requested system (public names).
    - Applies polar policy (fallback to 'equal' above |lat|>limit for risky systems).
    - Delegates computation to PreciseHouseCalculator (engine-internal names).
    - Returns a normalized dict with requested/effective system and warnings.
    """
    requested_public = _normalize_requested(requested_house_system)
    effective_public = requested_public
    warnings: List[str] = []

    if polar_policy == "reject_above_66deg" and _needs_polar_fallback(requested_public, latitude, polar_limit):
        # Let the caller convert this to your uniform validation error shape if desired.
        raise ValueError(
            f"house_system '{requested_public}' is unstable above |lat|>{polar_limit}°"
        )

    if polar_policy == "fallback_to_equal_above_66deg" and _needs_polar_fallback(
        requested_public, latitude, polar_limit
    ):
        warnings.append(
            f"Requested house_system '{requested_public}' is unstable above |lat|>{polar_limit}°. "
            f"Fallback to 'equal' applied."
        )
        effective_public = "equal"

    # Translate to engine-internal name (e.g., whole -> whole_sign)
    engine_system = _public_to_engine(effective_public)

    # Delegate to the math engine
    calc = PreciseHouseCalculator()
    hd: HouseData = calc.calculate_houses(
        latitude=latitude,
        longitude=longitude,
        jd_ut=jd_ut,
        house_system=engine_system,
    )

    # Merge engine warnings with policy warnings
    all_warnings = list(warnings) + list(hd.warnings or [])

    # Normalize system back to public enum in the output
    effective_public_out = _engine_to_public(hd.system)

    return {
        "requested_house_system": requested_public,
        "house_system": effective_public_out,  # effective system actually used
        "asc_deg": hd.ascendant,
        "mc_deg": hd.midheaven,
        "cusps_deg": hd.cusps,
        "vertex": hd.vertex,
        "eastpoint": hd.eastpoint,
        "warnings": all_warnings,
        # Helpful meta if your routes want to surface config:
        "policy": {
            "polar_policy": polar_policy,
            "polar_limit_deg": polar_limit,
            "risky_at_polar": sorted(_RISKY_AT_POLAR),
        },
    }


__all__ = [
    "compute_houses_with_policy",
    "list_supported_house_systems",
    "SUPPORTED_HOUSE_SYSTEMS",
    "POLAR_POLICY",
    "POLAR_LIMIT_DEG",
]
# app/core/house.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

# Core math engine (keep all numerical work here)
from app.core.houses_advanced import PreciseHouseCalculator, HouseData

# ---------------------------------------------
# Public API (external) house-system names
# ---------------------------------------------
SUPPORTED_HOUSE_SYSTEMS: Tuple[str, ...] = (
    "placidus",
    "whole",        # public name; engine uses "whole_sign"
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

# Systems known to be unstable near the poles (|lat| ≳ 66°)
_RISKY_AT_POLAR = {
    "placidus",
    "koch",
    "regiomontanus",
    "campanus",
    "topocentric",
    "alcabitius",
    "morinus",
    # Note: "porphyry" and "equal"/"whole" are typically robust
}

# Polar policy knobs
POLAR_LIMIT_DEG: float = 66.0
POLAR_POLICY: str = "fallback_to_equal_above_66deg"  # or "reject_above_66deg"


def list_supported_house_systems() -> List[str]:
    """Return the public list used by validators and /api/config."""
    return list(SUPPORTED_HOUSE_SYSTEMS)


def _normalize_requested(system: str | None) -> str:
    if not system:
        return "placidus"
    s = system.strip().lower()
    if s not in SUPPORTED_HOUSE_SYSTEMS:
        raise ValueError(f"unsupported house system: {system}")
    return s


def _public_to_engine(system: str) -> str:
    """Translate a public/system enum to the engine-internal name."""
    return _ALIAS_TO_ENGINE.get(system, system)


def _engine_to_public(system: str) -> str:
    """Translate engine-internal name back to the public enum."""
    return _ENGINE_TO_PUBLIC.get(system, system)


def _needs_polar_fallback(system: str, latitude: float, limit: float) -> bool:
    return abs(latitude) > limit and system in _RISKY_AT_POLAR


def compute_houses_with_policy(
    *,
    latitude: float,
    longitude: float,
    jd_ut: float,
    requested_house_system: str | None = None,
    polar_policy: str = POLAR_POLICY,
    polar_limit: float = POLAR_LIMIT_DEG,
) -> Dict[str, Any]:
    """
    Policy façade for house computation.

    - Validates and normalizes the requested system (public names).
    - Applies polar policy (fallback to 'equal' above |lat|>limit for risky systems).
    - Delegates computation to PreciseHouseCalculator (engine-internal names).
    - Returns a normalized dict with requested/effective system and warnings.
    """
    requested_public = _normalize_requested(requested_house_system)
    effective_public = requested_public
    warnings: List[str] = []

    if polar_policy == "reject_above_66deg" and _needs_polar_fallback(requested_public, latitude, polar_limit):
        # Let the caller convert this to your uniform validation error shape if desired.
        raise ValueError(
            f"house_system '{requested_public}' is unstable above |lat|>{polar_limit}°"
        )

    if polar_policy == "fallback_to_equal_above_66deg" and _needs_polar_fallback(
        requested_public, latitude, polar_limit
    ):
        warnings.append(
            f"Requested house_system '{requested_public}' is unstable above |lat|>{polar_limit}°. "
            f"Fallback to 'equal' applied."
        )
        effective_public = "equal"

    # Translate to engine-internal name (e.g., whole -> whole_sign)
    engine_system = _public_to_engine(effective_public)

    # Delegate to the math engine
    calc = PreciseHouseCalculator()
    hd: HouseData = calc.calculate_houses(
        latitude=latitude,
        longitude=longitude,
        jd_ut=jd_ut,
        house_system=engine_system,
    )

    # Merge engine warnings with policy warnings
    all_warnings = list(warnings) + list(hd.warnings or [])

    # Normalize system back to public enum in the output
    effective_public_out = _engine_to_public(hd.system)

    return {
        "requested_house_system": requested_public,
        "house_system": effective_public_out,  # effective system actually used
        "asc_deg": hd.ascendant,
        "mc_deg": hd.midheaven,
        "cusps_deg": hd.cusps,
        "vertex": hd.vertex,
        "eastpoint": hd.eastpoint,
        "warnings": all_warnings,
        # Helpful meta if your routes want to surface config:
        "policy": {
            "polar_policy": polar_policy,
            "polar_limit_deg": polar_limit,
            "risky_at_polar": sorted(_RISKY_AT_POLAR),
        },
    }


__all__ = [
    "compute_houses_with_policy",
    "list_supported_house_systems",
    "SUPPORTED_HOUSE_SYSTEMS",
    "POLAR_POLICY",
    "POLAR_LIMIT_DEG",
]
