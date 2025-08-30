# app/core/houses.py
from __future__ import annotations

from typing import List, Tuple, Optional

import os

# Try the precise backend
try:
    from app.core.houses_advanced import PreciseHouseCalculator, HouseData
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    PreciseHouseCalculator = None  # type: ignore
    HouseData = None  # type: ignore
    _IMPORT_ERR = _e

# Enable/disable numeric fallback (e.g., Placidus non-convergence) via env
NUMERIC_FALLBACK_ENABLED = os.getenv("ASTRO_HOUSES_NUMERIC_FALLBACK", "1").lower() in ("1", "true", "yes", "on")


def _require_backend() -> None:
    if PreciseHouseCalculator is None:
        raise RuntimeError(
            "houses: precise backend unavailable (numpy/pyerfa missing)."
        ) from _IMPORT_ERR  # type: ignore


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
    Internal helper that runs PreciseHouseCalculator with strict-timescale preference.
    If allow_fallback is True and the solver fails (e.g., Placidus non-convergence),
    we transparently fall back to 'equal' and return those cusps instead.
    """
    _require_backend()

    strict = (jd_tt is not None and jd_ut1 is not None)
    calc = PreciseHouseCalculator(require_strict_timescales=strict, enable_diagnostics=False)

    def _do(system_name: str) -> HouseData:
        if strict:
            return calc.calculate_houses(
                latitude=float(lat),
                longitude=float(lon),
                jd_ut=float(jd_ut or 0.0),   # ignored in strict mode
                house_system=system_name,
                jd_tt=float(jd_tt),
                jd_ut1=float(jd_ut1),
            )
        # legacy (non-strict) path
        if jd_ut is None:
            raise ValueError("Provide jd_tt & jd_ut1 (strict) or legacy jd_ut.")
        return calc.calculate_houses(
            latitude=float(lat),
            longitude=float(lon),
            jd_ut=float(jd_ut),
            house_system=system_name,
            jd_tt=None,
            jd_ut1=None,
        )

    try:
        hd: HouseData = _do(system)
    except Exception as e:
        if allow_fallback and NUMERIC_FALLBACK_ENABLED and system == "placidus":
            # Graceful numeric fallback → equal
            hd = _do("equal")
        else:
            # Re-raise with clearer context
            raise RuntimeError(f"houses solver failed for '{system}': {type(e).__name__}: {e}") from e

    return float(hd.ascendant), float(hd.midheaven), list(hd.cusps)


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
    """
    return _calc_core(
        lat=lat,
        lon=lon,
        system="equal",
        jd_tt=jd_tt,
        jd_ut1=jd_ut1,
        jd_ut=jd_ut,
        allow_fallback=False,  # no need; equal is robust
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
    Placidus houses with a **numeric safety net**:
    If the solver fails to converge, we fall back to Equal and return those cusps
    (when ASTRO_HOUSES_NUMERIC_FALLBACK is enabled; default = on).
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
    General wrapper for a few common systems:
      • "equal"      → Equal
      • "placidus"   → Placidus with numeric fallback → Equal
      • "whole"/"whole_sign"/"whole-sign"/"whole sign" → Whole-sign
    Other systems can be added here if needed by legacy code.
    """
    # normalize a couple of common aliases (keep minimal spread here)
    key = (system or "equal").strip().lower()
    if key in {"whole_sign", "whole-sign", "whole sign"}:
        key = "whole"

    if key == "equal":
        return asc_mc_equal_houses(lat, lon, jd_tt=jd_tt, jd_ut1=jd_ut1, jd_ut=jd_ut)
    if key == "placidus":
        return asc_mc_placidus_houses(lat, lon, jd_tt=jd_tt, jd_ut1=jd_ut1, jd_ut=jd_ut)
    if key == "whole":
        return _calc_core(
            lat=lat,
            lon=lon,
            system="whole_sign",
            jd_tt=jd_tt,
            jd_ut1=jd_ut1,
            jd_ut=jd_ut,
            allow_fallback=False,
        )

    raise ValueError(f"Unsupported house system for this legacy wrapper: {system!r}")


__all__ = [
    "PreciseHouseCalculator",
    "HouseData",
    "asc_mc_equal_houses",
    "asc_mc_placidus_houses",
    "asc_mc_houses",
]
