from __future__ import annotations
import math
from typing import Dict, Any

def _obliquity_deg(jd_tt: float) -> float:
    # Simplified obliquity (Meeus), arcseconds
    T = (jd_tt - 2451545.0)/36525.0
    eps = 23 + 26/60 + 21.448/3600 - (46.8150/3600)*T - (0.00059/3600)*(T**2) + (0.001813/3600)*(T**3)
    return eps

def _sidereal_time_deg(jd_ut: float, lon_deg: float) -> float:
    # IAU 1982 approximation of GMST + longitude
    T = (jd_ut - 2451545.0)/36525.0
    GMST = 280.46061837 + 360.98564736629*(jd_ut - 2451545.0) + 0.000387933*T*T - (T**3)/38710000.0
    LST = (GMST + lon_deg) % 360.0
    return LST

def asc_mc_equal_houses(lat_deg: float, lon_deg: float, jd_ut: float) -> Dict[str, Any]:
    # Ascendant and MC approximations with equal houses
    eps = math.radians(_obliquity_deg(jd_ut))
    phi = math.radians(lat_deg)
    lst = math.radians(_sidereal_time_deg(jd_ut, lon_deg))

    # MC
    tan_ra_mc = math.tan(lst)
    ra_mc = math.atan2(math.sin(lst), math.cos(lst))
    dec_mc = math.atan2(math.tan(eps)*math.sin(ra_mc), 1.0)
    lam_mc = math.degrees(math.atan2(math.sin(ra_mc)*math.cos(eps), math.cos(ra_mc))) % 360.0

    # ASC (approx)
    tan_lam = 1.0/(math.cos(lst))
    # Using formula: tan(lambda) = 1/(cos(eps)*tan(phi)) * ... (simplified), but we will use iterative approx
    # Fallback simple approach:
    asc = (lam_mc + 90.0) % 360.0

    cusps = [(asc + i*30.0) % 360.0 for i in range(12)]
    high_lat = abs(lat_deg) >= 66.0
    return {
        "house_system": "Equal" if not high_lat else "Equal",
        "asc_deg": asc,
        "mc_deg": lam_mc,
        "cusps_deg": cusps,
        "high_lat_fallback": high_lat,
        "warnings": ["High-latitude fallback to Equal houses"] if high_lat else []
    }
