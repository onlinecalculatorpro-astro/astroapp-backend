# app/core/house.py
from __future__ import annotations

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

# Sets of systems that are known to be problematic near the poles
_HARD_REJECT_AT_POLAR: Final[set[str]] = {"placidus", "koch", "topocentric", "alcabitius"}
_RISKY_AT_POLAR: Final[set[str]] = {
    "placidus", "koch", "regiomontanus", "campanus", "topocentric", "alcabitius", "morinus",
}

# If a numeric solver fails (e.g., placidus non-convergence), should we fall back?
NUMERIC_FALLBACK_ENABLED: Final[bool] = os.getenv("ASTRO_HOUSES_NUMERIC_FALLBACK", "1").lower() in ("1", "true", "yes", "on")

# ───────────────────────── Engine import ─────────────────────────

try:
    from app.core.houses_advanced import PreciseHouseCalculator, HouseData
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "houses_advanced backend unavailable. Install numpy>=1.26 and pyerfa>=2.0.1. "
        f"Original import error: {type(_e).__name__}: {_e}"
    ) from _e


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
      • numeric safety net (fallback to 'equal' on non-convergence),
      • stable, API-friendly payload.
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

    # ---- polar policy
    if abs(lat_f) >= hard_lim and requested_public in _HARD_REJECT_AT_POLAR:
        allowed = ["equal", "whole", "porphyry", "regiomontanus", "campanus", "morinus"]
        raise ValueError(
            f"house_system '{requested_public}' is undefined/unstable at latitude {lat_f:.2f}°. "
            f"Allowed at this latitude: {', '.join(allowed)}"
        )

    effective_public = requested_public
    if polar_policy == "reject_above_66deg" and _needs_polar_fallback(requested_public, lat_f, soft_lim):
        raise ValueError(f"house_system '{requested_public}' is unstable above |lat|>{soft_lim}°")

    if polar_policy == "fallback_to_equal_above_66deg" and _needs_polar_fallback(requested_public, lat_f, soft_lim):
        warnings.append(
            f"Requested house_system '{requested_public}' is unstable above |lat|>{soft_lim}°. "
            "Fallback to 'equal' applied."
        )
        effective_public = "equal"

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

    engine_system = _public_to_engine(effective_public)

    # ---- compute with numeric safety net
    calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=bool(diagnostics))

    def _compute(sys_name: str) -> HouseData:
        return calc.calculate_houses(
            latitude=lat_f,
            longitude=lon_f,
            jd_ut=float(jd_ut) if jd_ut is not None else 0.0,  # ignored in strict mode
            house_system=sys_name,
            jd_tt=float(jd_tt),
            jd_ut1=float(jd_ut1),
        )

    try:
        hd: HouseData = _compute(engine_system)
        used_public = _engine_to_public(hd.system)
    except Exception as e:
        if NUMERIC_FALLBACK_ENABLED and effective_public in _RISKY_AT_POLAR:
            # graceful numeric fallback → equal
            try:
                hd = _compute("equal")
                used_public = "equal"
                warnings.append(f"{effective_public} numeric failure → fallback to 'equal' ({type(e).__name__}: {e})")
            except Exception as e2:
                # If even equal fails, bubble up original error context
                raise RuntimeError(f"houses engine failed and fallback failed: {type(e2).__name__}: {e2}") from e
        else:
            raise

    # ---- build payload (compat: include asc/mc and asc_deg/mc_deg)
    all_warnings = list(warnings) + list(hd.warnings or [])

    payload: Dict[str, Any] = {
        "requested_system": requested_public,         # public label requested
        "system": used_public,                        # public label used after policy/fallback
        "engine_system": hd.system,                   # engine canonical actually used
        "cusps": hd.cusps,                            # list[12] degrees
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
