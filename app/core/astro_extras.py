from __future__ import annotations
from typing import Dict, Any, List, Tuple
from math import fmod, isnan
from skyfield.api import load, wgs84, Star

def _wrap360(x: float) -> float:
    return fmod(fmod(x, 360.0) + 360.0, 360.0)

# ---------- Harmonic charts ----------
def harmonic_longitudes(lons_deg: Dict[str, float], n: int) -> Dict[str, float]:
    """n-th harmonic: multiply longitudes by n, wrap 0..360."""
    return {name: _wrap360(n * lon) for name, lon in lons_deg.items()}

# ---------- Arabic parts (basic set) ----------
def part_of_fortune(asc: float, sun: float, moon: float, is_day: bool) -> float:
    """POF = Asc + Moon − Sun (day) ; Asc + Sun − Moon (night)."""
    return _wrap360(asc + (moon - sun) if is_day else asc + (sun - moon))

def part_of_spirit(asc: float, sun: float, moon: float, is_day: bool) -> float:
    """POS = Asc + Sun − Moon (day) ; Asc + Moon − Sun (night)."""
    return _wrap360(asc + (sun - moon) if is_day else asc + (moon - sun))

# ---------- Aspect engine ----------
ASPECTS = {
    "conjunction": 0.0,
    "sextile": 60.0,
    "square": 90.0,
    "trine": 120.0,
    "opposition": 180.0,
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
    """All aspects within orb between two sets of longitudes (can be same set)."""
    results = []
    for na, la in lons_a.items():
        for nb, lb in lons_b.items():
            if (lons_a is lons_b) and (na >= nb):
                continue
            sep = _angle_sep(la, lb)
            for name, exact in aspects.items():
                delta = min(abs(sep - exact), abs(360.0 - sep - exact))
                if delta <= orb_deg:
                    results.append({
                        "a": na, "b": nb, "aspect": name,
                        "sep": sep, "orb": delta
                    })
    return results

# ---------- Secondary progressions (1 day = 1 year) ----------
def secondary_progressed_time(jd_tt_birth: float, jd_tt_target: float) -> float:
    """Return progressed TT Julian Date using Naibod key."""
    years = (jd_tt_target - jd_tt_birth) / 365.2422
    return jd_tt_birth + years  # add N days

# ---------- Solar arc directions ----------
def solar_arc_offset(sun_natal_lon: float, sun_progressed_lon: float) -> float:
    """Solar arc = progressed Sun − natal Sun (add to all natal points)."""
    return _wrap360(sun_progressed_lon - sun_natal_lon)

def apply_solar_arc(natal_lons: Dict[str, float], arc: float) -> Dict[str, float]:
    return {k: _wrap360(v + arc) for k, v in natal_lons.items()}

# ---------- Solar/Lunar returns (one return near a guess) ----------
def find_return_jd(
    body_key: str,
    target_lon: float,
    jd_guess: float,
    lat: float,
    lon: float,
    window_days: float = 2.0,
    tol_arcmin: float = 1.0,
) -> float:
    """
    Find JD(TT) near guess where body ecliptic longitude equals target_lon.
    Simple 1D bisection on longitude difference.
    """
    ts = load.timescale()
    eph = load("de421.bsp")
    topo = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon, elevation_m=0.0)
    def lon_at(jd):
        t = ts.tdb(jd=jd)
        obs = (eph["earth"] + topo).at(t)
        ast = eph[body_key].at(t).observe_from(obs).apparent()
        try:
            lon, _, _ = ast.ecliptic_latlon(epoch="date")
        except TypeError:
            lon, _, _ = ast.ecliptic_latlon()
        return _wrap360(lon.degrees)

    a = jd_guess - window_days/2
    b = jd_guess + window_days/2
    for _ in range(40):  # ~arcminute precision
        mid = (a + b) / 2
        la = _angle_sep(lon_at(a), target_lon)
        lm = _angle_sep(lon_at(mid), target_lon)
        lb = _angle_sep(lon_at(b), target_lon)
        if lm <= tol_arcmin / 60.0:
            return mid
        # choose half-interval that reduces separation
        if la < lb:
            b = mid
        else:
            a = mid
    return mid

# ---------- Fixed stars (sample bright set) ----------
BRIGHT_STARS = {
    # name: (RA hours, Dec degrees) — J2000 approx
    "Regulus":  (10 + 8/60 + 22/3600,  11.97),
    "Spica":    (13 + 25/60 + 12/3600, -11.16),
    "Aldebaran":( 4 + 35/60 + 55/3600,  16.51),
    "Antares":  (16 + 29/60 + 24/3600, -26.43),
    "Sirius":   ( 6 + 45/60 +  9/3600, -16.72),
}

def fixed_star_ecliptics(jd_tt: float) -> Dict[str, Dict[str, float]]:
    """Return ecliptic lon/lat of a small bright-star set, precessed to date."""
    ts = load.timescale()
    eph = load("de421.bsp")
    t  = ts.tdb(jd=jd_tt)
    out = {}
    for name, (rah, decd) in BRIGHT_STARS.items():
        star = Star(ra_hours=rah, dec_degrees=decd)
        ast = eph["earth"].at(t).observe(star).apparent()
        try:
            lon, lat, _ = ast.ecliptic_latlon(epoch="date")
        except TypeError:
            lon, lat, _ = ast.ecliptic_latlon()
        out[name] = {"lon": _wrap360(lon.degrees), "lat": float(lat.degrees)}
    return out

def star_conjunctions(
    planet_lons: Dict[str, float],
    stars: Dict[str, Dict[str, float]],
    orb_deg: float = 1.0,
) -> List[Dict[str, Any]]:
    """Find planet–fixed-star conjunctions within orb."""
    out = []
    for p, lp in planet_lons.items():
        for s, coords in stars.items():
            sep = _angle_sep(lp, coords["lon"])
            if sep <= orb_deg:
                out.append({"planet": p, "star": s, "sep": sep})
    return out
