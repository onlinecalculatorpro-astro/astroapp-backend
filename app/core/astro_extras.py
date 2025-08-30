# app/core/astro_extras.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from math import fmod

from skyfield.api import load, wgs84, Star

def _wrap360(x: float) -> float:
    return fmod(fmod(x, 360.0) + 360.0, 360.0)

# ---------- Harmonic charts ----------
def harmonic_longitudes(lons_deg: Dict[str, float], n: int) -> Dict[str, float]:
    return {name: _wrap360(n * lon) for name, lon in lons_deg.items()}

# ---------- Arabic parts ----------
def part_of_fortune(asc: float, sun: float, moon: float, is_day: bool) -> float:
    return _wrap360(asc + (moon - sun) if is_day else asc + (sun - moon))

def part_of_spirit(asc: float, sun: float, moon: float, is_day: bool) -> float:
    return _wrap360(asc + (sun - moon) if is_day else asc + (moon - sun))

# ---------- Aspect engine ----------
ASPECTS = {
    "conjunction": 0.0, "sextile": 60.0, "square": 90.0, "trine": 120.0, "opposition": 180.0,
}

def _angle_sep(a: float, b: float) -> float:
    d = abs(((a - b + 180.0) % 360.0) - 180.0)
    return d

def find_aspects(
    lons_a: Dict[str, float],
    lons_b: Dict[str, float],
    orb_deg: float = 2.0,
    aspects: Dict[str, float] = ASPECTS,
) -> List[Dict[str, Any]]:
    out = []
    for na, la in lons_a.items():
        for nb, lb in lons_b.items():
            if (lons_a is lons_b) and (na >= nb):
                continue
            sep = _angle_sep(la, lb)
            for name, exact in aspects.items():
                delta = min(abs(sep - exact), abs(360.0 - sep - exact))
                if delta <= orb_deg:
                    out.append({"a": na, "b": nb, "aspect": name, "sep": sep, "orb": delta})
    return out

# ---------- Secondary progressions (1 day = 1 year) ----------
def secondary_progressed_time(jd_tt_birth: float, jd_tt_target: float) -> float:
    years = (jd_tt_target - jd_tt_birth) / 365.2422
    return jd_tt_birth + years

# ---------- Solar arc directions ----------
def solar_arc_offset(sun_natal_lon: float, sun_progressed_lon: float) -> float:
    return _wrap360(sun_progressed_lon - sun_natal_lon)

def apply_solar_arc(natal_lons: Dict[str, float], arc: float) -> Dict[str, float]:
    return {k: _wrap360(v + arc) for k, v in natal_lons.items()}

# ---------- Fixed stars (small bright set) ----------
BRIGHT_STARS = {
    "Regulus":  (10 + 8/60 + 22/3600,  11.97),
    "Spica":    (13 + 25/60 + 12/3600, -11.16),
    "Aldebaran":( 4 + 35/60 + 55/3600,  16.51),
    "Antares":  (16 + 29/60 + 24/3600, -26.43),
    "Sirius":   ( 6 + 45/60 +  9/3600, -16.72),
}

def fixed_star_ecliptics(jd_tt: float) -> Dict[str, Dict[str, float]]:
    ts = load.timescale()
    eph = load("de421.bsp")
    t  = ts.tdb(jd=jd_tt)
    out = {}
    for name, (rah, decd) in BRIGHT_STARS.items():
        star = Star(ra_hours=rah, dec_degrees=decd)
        ast  = eph["earth"].at(t).observe(star).apparent()
        try:
            lon, lat, _ = ast.ecliptic_latlon(epoch="date")
        except TypeError:
            lon, lat, _ = ast.ecliptic_latlon()
        out[name] = {"lon": _wrap360(lon.degrees), "lat": float(lat.degrees)}
    return out

def star_conjunctions(planet_lons: Dict[str, float],
                      stars: Dict[str, Dict[str, float]],
                      orb_deg: float = 1.0) -> List[Dict[str, Any]]:
    out = []
    for p, lp in planet_lons.items():
        for s, coords in stars.items():
            sep = _angle_sep(lp, coords["lon"])
            if sep <= orb_deg:
                out.append({"planet": p, "star": s, "sep": sep})
    return out
