from __future__ import annotations
from typing import Dict, Any
from .timescales import julian_day_utc, jd_tt_from_utc_jd
from .ephemeris_adapter import ecliptic_longitudes
from .houses import asc_mc_equal_houses

def compute_chart(date: str, time_s: str, lat: float, lon: float, mode: str, tz: str = "UTC") -> Dict[str, Any]:
    jd_ut = julian_day_utc(date, time_s, tz)
    jd_tt = jd_tt_from_utc_jd(jd_ut, int(date.split('-')[0]), int(date.split('-')[1]))
    bodies = ecliptic_longitudes(jd_tt, lat, lon)
    # Ayanamsa (Lahiri) simple offset ~ 24 deg for now; real ayanamsa tables can be plugged later
    ayanamsa = 24.0 if mode == "sidereal" else 0.0
    # Apply mode
    out_bodies = []
    for b in bodies:
        lon_adj = (b["lon"] - ayanamsa) % 360.0 if mode=="sidereal" else b["lon"]
        out_bodies.append({
            "name": b["name"],
            "longitude_deg": lon_adj,
            "latitude_deg": b["lat"],
            "speed_deg_per_day": b["speed"],
            "retrograde": bool(b["retro"]),
        })
    return {
        "mode": mode,
        "ayanamsa_deg": ayanamsa,
        "jd_ut": jd_ut,
        "jd_tt": jd_tt,
        "bodies": out_bodies
    }

def compute_houses(lat: float, lon: float, mode: str, jd_ut: float) -> Dict[str, Any]:
    return asc_mc_equal_houses(lat, lon, jd_ut)
