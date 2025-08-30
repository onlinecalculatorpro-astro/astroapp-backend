# app/core/house.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Final

# ── Supported public systems (keep in sync with engine) ───────────────────────
SUPPORTED_HOUSE_SYSTEMS: Final[Tuple[str, ...]] = (
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

# Public → engine-internal aliases
_ALIAS_TO_ENGINE: Final[Dict[str, str]] = {
    "whole": "whole_sign",
}
_ENGINE_TO_PUBLIC: Final[Dict[str, str]] = {v: k for k, v in _ALIAS_TO_ENGINE.items()}

# ── Polar policy knobs (env-tunable) ──────────────────────────────────────────
POLAR_SOFT_LIMIT_DEG: Final[float] = float(os.getenv("ASTRO_POLAR_SOFT_LAT", 66.0))
POLAR_HARD_LIMIT_DEG: Final[float] = float(os.getenv("ASTRO_POLAR_HARD_LAT", 80.0))
POLAR_ABSOLUTE_LIMIT_DEG: Final[float] = 89.999999  # absolute pole guard
# Policy: "fallback_to_equal_above_66deg" | "reject_above_66deg"
POLAR_POLICY: Final[str] = os.getenv("ASTRO_POLAR_POLICY", "fallback_to_equal_above_66deg")

# Time-division systems that blow up near the poles (hard-reject ≥ HARD limit)
_HARD_REJECT_AT_POLAR: Final[set[str]] = {
    "placidus", "koch", "topocentric", "alcabitius",
}
# Systems that are risky above SOFT limit (soft-fallback or early reject)
_RISKY_AT_POLAR: Final[set[str]] = {
    "placidus", "koch", "regiomontanus", "campanus",
    "topocentric", "alcabitius", "morinus",
    # porphyry/equal/whole are generally robust
}

# ── Core math engine (requires NumPy + PyERFA) ────────────────────────────────
try:
    from app.core.houses_advanced import PreciseHouseCalculator, HouseData
except Exception as _e:
    raise RuntimeError(
        "houses_advanced backend unavailable. Ensure NumPy and PyERFA are installed "
        "(e.g., numpy>=1.26 and pyerfa>=2.0.1). "
        f"Original import error: {type(_e).__name__}: {_e}"
    ) from _e


# ╭──────────────────────────── helpers ─────────────────────────────╮
def list_supported_house_systems() -> List[str]:
    return list(SUPPORTED_HOUSE_SYSTEMS)


def _normalize_system(s: Optional[str]) -> str:
    if not s:
        return "placidus"
    s2 = s.strip().lower()
    if s2 not in SUPPORTED_HOUSE_SYSTEMS:
        raise ValueError(f"unsupported house system: {s}")
    return s2


def _public_to_engine(system: str) -> str:
    return _ALIAS_TO_ENGINE.get(system, system)


def _engine_to_public(system: str) -> str:
    return _ENGINE_TO_PUBLIC.get(system, system)


def _needs_polar_fallback(system: str, latitude: float, limit: float) -> bool:
    return abs(latitude) > limit and system in _RISKY_AT_POLAR
# ╰──────────────────────────────────────────────────────────────────╯


def compute_houses_with_policy(
    *,
    # Preferred modern names
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    system: Optional[str] = None,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    jd_ut: Optional[float] = None,           # optional; engine ignores in strict mode
    diagnostics: Optional[bool] = None,
    # Legacy aliases kept for compatibility
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    requested_house_system: Optional[str] = None,
    enable_diagnostics: Optional[bool] = None,
    # Policy overrides (mainly for tests)
    polar_policy: Optional[str] = None,
    polar_soft_limit: Optional[float] = None,
    polar_hard_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Policy façade for house computation.

    - Validates latitude (absolute pole guard).
    - Normalizes the requested system (public enum ⮕ engine name).
    - Enforces polar policy:
        * hard reject time-division systems at |lat| >= HARD limit
        * soft fallback to 'equal' above SOFT limit when policy demands
    - Requires strict timescales: jd_tt & jd_ut1.
    - Delegates to PreciseHouseCalculator(require_strict_timescales=True).
    - Returns a compat-friendly payload (angles, houses, *deg fields, warnings, policy).
    """

    # Resolve args from modern or legacy names
    latitude = lat if lat is not None else latitude
    longitude = lon if lon is not None else longitude
    requested_public = system if system is not None else requested_house_system

    if latitude is None or longitude is None:
        raise ValueError("lat/lon are required")
    lat_f = float(latitude)
    lon_f = float(longitude)

    # Absolute latitude guards
    if not (-90.0 < lat_f < 90.0):
        raise ValueError("Latitude must be strictly between -90 and 90 degrees.")
    if abs(lat_f) >= POLAR_ABSOLUTE_LIMIT_DEG:
        raise ValueError(
            f"latitude {lat_f:.8f}° is at/near the pole; house systems are undefined."
        )

    # Normalize requested system (public)
    requested_public = _normalize_system(requested_public)
    effective_public = requested_public
    warnings: List[str] = []

    # Policy effective values
    polar_policy = (polar_policy or POLAR_POLICY).strip()
    soft_lim = float(polar_soft_limit if polar_soft_limit is not None else POLAR_SOFT_LIMIT_DEG)
    hard_lim = float(polar_hard_limit if polar_hard_limit is not None else POLAR_HARD_LIMIT_DEG)

    # Hard-reject near poles for time-division systems
    if abs(lat_f) >= hard_lim and requested_public in _HARD_REJECT_AT_POLAR:
        allowed = ["equal", "whole", "porphyry", "regiomontanus", "campanus", "morinus"]
        raise ValueError(
            f"house_system '{requested_public}' is undefined/unstable at latitude {lat_f:.2f}°. "
            f"Allowed at this latitude: {', '.join(allowed)}"
        )

    # Soft policy above soft limit
    if polar_policy == "reject_above_66deg" and _needs_polar_fallback(requested_public, lat_f, soft_lim):
        raise ValueError(
            f"house_system '{requested_public}' is unstable above |lat|>{soft_lim}°"
        )

    if polar_policy == "fallback_to_equal_above_66deg" and _needs_polar_fallback(requested_public, lat_f, soft_lim):
        warnings.append(
            f"Requested house_system '{requested_public}' is unstable above |lat|>{soft_lim}°. "
            "Fallback to 'equal' applied."
        )
        effective_public = "equal"

    # Strict timescales requirement
    if jd_tt is None or jd_ut1 is None:
        raise ValueError(
            "houses: strict mode requires jd_tt and jd_ut1 (no UT≈UTC shortcuts). "
            "Provide both timescales derived from UTC via leap seconds & ΔT."
        )

    # Diagnostics toggle (env default)
    if diagnostics is None:
        diagnostics = enable_diagnostics
    if diagnostics is None:
        diagnostics = os.getenv("ASTRO_HOUSES_DEBUG", "").lower() in ("1", "true", "yes", "on")

    # Translate system to engine-internal name
    engine_system = _public_to_engine(effective_public)

    # Delegate to math engine (strict mode ON)
    calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=bool(diagnostics))
    hd: HouseData = calc.calculate_houses(
        latitude=lat_f,
        longitude=lon_f,
        jd_ut=jd_ut if jd_ut is not None else 0.0,  # kept for signature stability; ignored in strict mode
        house_system=engine_system,
        jd_tt=float(jd_tt),
        jd_ut1=float(jd_ut1),
    )

    # Merge warnings (policy + engine)
    all_warnings = list(warnings) + list(hd.warnings or [])

    # Normalize system back to public enum for output
    effective_public_out = _engine_to_public(hd.system)

    # Build a compat-friendly response (keep both modern and legacy shapes)
    payload: Dict[str, Any] = {
        "requested_house_system": requested_public,
        "house_system": effective_public_out,    # effective system actually used (public name)
        "asc_deg": hd.ascendant,
        "mc_deg": hd.midheaven,
        "angles": {
            "ASC": {"lon": hd.ascendant, "deg": hd.ascendant},
            "MC":  {"lon": hd.midheaven, "deg": hd.midheaven},
        },
        "houses": hd.cusps,          # legacy key
        "cusps_deg": hd.cusps,       # explicit key
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
