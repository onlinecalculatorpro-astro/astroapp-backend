# app/core/houses.py
from __future__ import annotations
from typing import List, Tuple, Optional

try:
    from .houses_advanced import PreciseHouseCalculator, HouseData
    _IMPORT_ERR = None
except Exception as _e:
    PreciseHouseCalculator = None  # type: ignore
    HouseData = None  # type: ignore
    _IMPORT_ERR = _e

def asc_mc_equal_houses(
    lat: float,
    lon: float,
    *,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    jd_ut: Optional[float] = None,
) -> Tuple[float, float, List[float]]:
    """Legacy helper expected by astronomy.py. Returns (ASC, MC, cusps) in degrees for Equal houses."""
    if PreciseHouseCalculator is None:
        raise RuntimeError(
            "asc_mc_equal_houses unavailable: houses_advanced/pyerfa not importable."
        ) from _IMPORT_ERR  # type: ignore

    strict = (jd_tt is not None and jd_ut1 is not None)
    calc = PreciseHouseCalculator(require_strict_timescales=strict, enable_diagnostics=False)

    if strict:
        hd: HouseData = calc.calculate_houses(
            latitude=float(lat), longitude=float(lon),
            jd_ut=float(jd_ut or 0.0),  # ignored in strict mode
            house_system="equal", jd_tt=float(jd_tt), jd_ut1=float(jd_ut1),
        )
    else:
        if jd_ut is None:
            raise ValueError("Provide jd_tt & jd_ut1 (strict) or legacy jd_ut.")
        hd: HouseData = calc.calculate_houses(
            latitude=float(lat), longitude=float(lon),
            jd_ut=float(jd_ut), house_system="equal", jd_tt=None, jd_ut1=None,
        )

    return float(hd.ascendant), float(hd.midheaven), list(hd.cusps)

__all__ = ["PreciseHouseCalculator", "HouseData", "asc_mc_equal_houses"]
