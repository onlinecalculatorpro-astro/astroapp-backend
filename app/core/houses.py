# app/core/houses.py
from __future__ import annotations
"""
Lightweight legacy façade used by astronomy.py and older call-sites.

Public helpers (return ASC, MC, cusps in degrees):
  • asc_mc_equal_houses
  • asc_mc_placidus_houses      (numeric fallback → Equal when enabled)
  • asc_mc_houses               (minimal aliasing for a few common labels)

STRICT path (preferred):
  If jd_tt & jd_ut1 are provided, we delegate to
  app.core.house.compute_houses_with_policy to keep alias normalization,
  polar rules, and robust fallback behavior **exactly** in line with the main API.

LEGACY path:
  If strict timescales are absent but jd_ut is provided, we call
  PreciseHouseCalculator with require_strict_timescales=False.
  For "placidus" only, this wrapper keeps the historic numeric fallback → "equal"
  when ASTRO_HOUSES_NUMERIC_FALLBACK is enabled.

Deployment note:
  Set ASTRO_REQUIRE_POLICY_FOR_STRICT=1 to **fail fast** when strict timescales
  are used but the policy façade cannot be imported (prevents silent degradation).
"""

from typing import List, Tuple, Optional
import difflib
import logging
import os

logger = logging.getLogger(__name__)

# ------------------------- Config / Env toggles -------------------------

NUMERIC_FALLBACK_ENABLED: bool = os.getenv(
    "ASTRO_HOUSES_NUMERIC_FALLBACK", "1"
).lower() in ("1", "true", "yes", "on")

REQUIRE_POLICY_FOR_STRICT: bool = os.getenv(
    "ASTRO_REQUIRE_POLICY_FOR_STRICT", "0"
).lower() in ("1", "true", "yes", "on")

# ------------------------- Prefer the policy façade ----------------------

try:
    from app.core.house import compute_houses_with_policy as _policy_compute  # aligned behavior
    _HAS_POLICY = True
except Exception:
    _HAS_POLICY = False

# emit a one-time import-time log (info/warn) so ops can spot config issues
if not _HAS_POLICY:
    logger.warning(
        "house policy façade unavailable; strict calls may fall back to direct engine path. "
        "Set ASTRO_REQUIRE_POLICY_FOR_STRICT=1 to fail fast in prod."
    )

# ------------------------- Precise backend (legacy path) -----------------

try:
    from app.core.houses_advanced import PreciseHouseCalculator, HouseData
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    PreciseHouseCalculator = None  # type: ignore
    HouseData = None  # type: ignore
    _IMPORT_ERR = _e

def _require_backend() -> None:
    if PreciseHouseCalculator is None:
        raise RuntimeError(
            "houses: precise backend unavailable (numpy/pyerfa missing)."
        ) from _IMPORT_ERR  # type: ignore


# ------------------------- Minimal aliasing for this wrapper --------------

# Keep this surface intentionally small; for a full alias surface & policy,
# call app.core.house.compute_houses_with_policy directly.
_WRAPPER_SUPPORTED_PUBLIC = {
    "equal",
    "placidus",
    "whole",             # maps to engine "whole_sign" on the legacy (direct) path
    # conveniences (supported when strict path uses policy façade; direct path passes through)
    "equal_from_mc",
    "natural_houses",
}

def _slug(s: str) -> str:
    """Lowercase + strip non-alphanumerics → compact, stable alias token."""
    return "".join(ch for ch in s.lower() if ch.isalnum())

_CANON_FROM_SLUG = {
    "equal": "equal",
    "placidus": "placidus",
    "whole": "whole",
    "wholesign": "whole",             # common alias
    "equalfrommc": "equal_from_mc",
    "naturalhouses": "natural_houses",
}

def _suggest_wrapper_labels(name: str, n: int = 5) -> List[str]:
    """Suggest closest wrapper-supported labels for friendlier error messages."""
    candidates = sorted(_WRAPPER_SUPPORTED_PUBLIC)
    hits = difflib.get_close_matches(name.lower(), candidates, n=n, cutoff=0.5)
    return hits[:n]

def _normalize_public_for_wrapper(system: Optional[str]) -> str:
    """Return a canonical public label for this legacy wrapper (small surface)."""
    if not system:
        return "equal"
    slug = _slug(system)
    canon = _CANON_FROM_SLUG.get(slug)
    if not canon or canon not in _WRAPPER_SUPPORTED_PUBLIC:
        sugg = _suggest_wrapper_labels(system)
        hint = f" Try one of: {', '.join(sugg)}." if sugg else ""
        raise ValueError(
            f"Unsupported house system for this legacy wrapper: '{system}' (slug='{slug}').{hint}"
        )
    return canon

def _public_to_engine_legacy(public: str) -> str:
    """Public→engine name mapping for the legacy (direct engine) path."""
    return "whole_sign" if public == "whole" else public


# ------------------------- Core executor ----------------------------------

_warned_no_policy_strict_once = False

def _calc_core(
    *,
    lat: float,
    lon: float,
    system: str,
    jd_tt: Optional[float],
    jd_ut1: Optional[float],
    jd_ut: Optional[float],
    allow_fallback: bool,
) -> Tuple[float, float, List[float]]:
    """
    STRICT path (preferred):
      If jd_tt & jd_ut1 are provided and the policy façade is available, call
      compute_houses_with_policy for fully aligned behavior (aliases, polar rules,
      robust fallbacks). Read ASC/MC/cusps from its payload.

      If the façade is NOT importable:
        - If ASTRO_REQUIRE_POLICY_FOR_STRICT=1 → raise RuntimeError (fail fast).
        - Else log a one-time warning and continue via direct engine strict mode.

    LEGACY path:
      Otherwise call the engine directly with require_strict_timescales=False.
      If allow_fallback and NUMERIC_FALLBACK_ENABLED and system == "placidus",
      fall back to "equal" on engine failure (historic behavior).
    """
    global _warned_no_policy_strict_once

    strict = (jd_tt is not None and jd_ut1 is not None)

    # STRICT path via policy façade (aligned with house.py)
    if strict:
        if _HAS_POLICY:
            payload = _policy_compute(
                lat=float(lat),
                lon=float(lon),
                system=system,                 # façade handles aliasing & engine mapping
                jd_tt=float(jd_tt),            # type: ignore[arg-type]
                jd_ut1=float(jd_ut1),          # type: ignore[arg-type]
                jd_ut=float(jd_ut or 0.0),     # echoed only; ignored by policy
                diagnostics=False,
                validation=False,
            )
            asc = float(payload["asc"])
            mc = float(payload["mc"])
            cusps = [float(x) for x in payload["cusps"]]
            return asc, mc, cusps

        # façade missing: respect env gate
        msg = (
            "Strict timescales provided but policy façade is not importable. "
            "Falling back to direct engine path (aliases/polar/fallbacks handled only by this wrapper)."
        )
        if REQUIRE_POLICY_FOR_STRICT:
            raise RuntimeError(msg + " Set ASTRO_REQUIRE_POLICY_FOR_STRICT=0 to allow fallback.")
        if not _warned_no_policy_strict_once:
            logger.warning(msg)
            _warned_no_policy_strict_once = True

    # DIRECT ENGINE path (strict or legacy depending on args)
    _require_backend()
    calc = PreciseHouseCalculator(require_strict_timescales=strict, enable_diagnostics=False)

    def _do(engine_name: str) -> HouseData:
        if strict:
            return calc.calculate_houses(
                latitude=float(lat),
                longitude=float(lon),
                jd_ut=float(jd_ut or 0.0),   # ignored in strict mode
                house_system=engine_name,
                jd_tt=float(jd_tt),          # type: ignore[arg-type]
                jd_ut1=float(jd_ut1),        # type: ignore[arg-type]
            )
        # legacy (non-strict) path
        if jd_ut is None:
            raise ValueError("Provide jd_tt & jd_ut1 (strict) or legacy jd_ut.")
        return calc.calculate_houses(
            latitude=float(lat),
            longitude=float(lon),
            jd_ut=float(jd_ut),
            house_system=engine_name,
            jd_tt=None,
            jd_ut1=None,
        )

    engine_name = _public_to_engine_legacy(system)

    try:
        hd: HouseData = _do(engine_name)
    except Exception as e:
        # Keep the historic numeric fallback for Placidus only (legacy behavior)
        if not strict and allow_fallback and NUMERIC_FALLBACK_ENABLED and system == "placidus":
            hd = _do("equal")
        else:
            # Re-raise with clearer context including original system label
            raise RuntimeError(f"houses solver failed for '{system}': {type(e).__name__}: {e}") from e

    return float(hd.ascendant), float(hd.midheaven), list(hd.cusps)


# ------------------------- Public helpers ---------------------------------

def asc_mc_equal_houses(
    lat: float,
    lon: float,
    *,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    jd_ut: Optional[float] = None,
) -> Tuple[float, float, List[float]]:
    """
    Legacy helper expected by astronomy.py.
    Returns (ASC, MC, cusps) for Equal houses, in degrees.
    Uses the policy façade when strict timescales are provided.
    """
    return _calc_core(
        lat=lat,
        lon=lon,
        system="equal",
        jd_tt=jd_tt,
        jd_ut1=jd_ut1,
        jd_ut=jd_ut,
        allow_fallback=False,  # Equal is robust; no numeric fallback needed
    )

def asc_mc_placidus_houses(
    lat: float,
    lon: float,
    *,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    jd_ut: Optional[float] = None,
) -> Tuple[float, float, List[float]]:
    """
    Placidus houses with a numeric safety net (legacy behavior):
    If the direct engine fails to converge on the **legacy** path (no jd_tt/jd_ut1),
    we fall back to Equal when ASTRO_HOUSES_NUMERIC_FALLBACK is enabled (default: on).

    On the strict path, we delegate to compute_houses_with_policy, which provides
    robust fallbacks and polar safety consistent with the main API.
    """
    return _calc_core(
        lat=lat,
        lon=lon,
        system="placidus",
        jd_tt=jd_tt,
        jd_ut1=jd_ut1,
        jd_ut=jd_ut,
        allow_fallback=True,
    )

def asc_mc_houses(
    system: str,
    lat: float,
    lon: float,
    *,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    jd_ut: Optional[float] = None,
) -> Tuple[float, float, List[float]]:
    """
    General wrapper for a few common systems (legacy surface):
      • "equal"                 → Equal
      • "placidus"              → Placidus (numeric fallback → Equal on legacy path)
      • "whole"/"whole_sign"    → Whole-sign
      • "equal_from_mc"         → Equal-from-MC  (strict path uses policy façade)
      • "natural_houses"        → Natural Houses (strict path uses policy façade)

    For a richer alias surface and policy/fallback control, call
    app.core.house.compute_houses_with_policy directly.
    """
    canon = _normalize_public_for_wrapper(system)

    if canon == "equal":
        return asc_mc_equal_houses(lat, lon, jd_tt=jd_tt, jd_ut1=jd_ut1, jd_ut=jd_ut)
    if canon == "placidus":
        return asc_mc_placidus_houses(lat, lon, jd_tt=jd_tt, jd_ut1=jd_ut1, jd_ut=jd_ut)
    if canon == "whole":
        # policy façade accepts "whole"; legacy engine needs "whole_sign"
        return _calc_core(
            lat=lat,
            lon=lon,
            system="whole",
            jd_tt=jd_tt,
            jd_ut1=jd_ut1,
            jd_ut=jd_ut,
            allow_fallback=False,
        )
    if canon in ("equal_from_mc", "natural_houses"):
        return _calc_core(
            lat=lat,
            lon=lon,
            system=canon,
            jd_tt=jd_tt,
            jd_ut1=jd_ut1,
            jd_ut=jd_ut,
            allow_fallback=False,
        )

    # Shouldn't reach here because _normalize_public_for_wrapper guards inputs
    raise ValueError(f"Unsupported house system for this legacy wrapper: {system!r}")


__all__ = [
    "PreciseHouseCalculator",
    "HouseData",
    "asc_mc_equal_houses",
    "asc_mc_placidus_houses",
    "asc_mc_houses",
]
