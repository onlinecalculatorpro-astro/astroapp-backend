# --- Enhanced, version-safe ecliptic calculations -------------------
from typing import Dict, Any, List, Optional
from math import fmod
from skyfield.api import load, wgs84

def _wrap360(x: float) -> float:
    return fmod(fmod(x, 360.0) + 360.0, 360.0)

def enhanced_ecliptic_longitudes(
    jd_tt: float,
    lat: float,
    lon: float,
    temperature_C: Optional[float] = None,
    pressure_mbar: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns 'mean' (J2000) vs 'true' (of-date, includes nutation) ecliptic coords,
    and 'apparent' (includes light-time + annual aberration). Topocentric.

    Refraction: provided in Alt/Az only (if temperature/pressure given).
    """
    ts = load.timescale()
    eph = load("de421.bsp")
    t  = ts.tdb(jd=jd_tt)

    topo = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon, elevation_m=0.0)
    obs_t = (eph["earth"] + topo).at(t)

    keys = {
        "Sun": "sun",
        "Moon": "moon",
        "Mercury": "mercury",
        "Venus": "venus",
        "Mars": "mars",
        "Jupiter": "jupiter barycenter",
        "Saturn": "saturn barycenter",
    }

    out: List[Dict[str, Any]] = []

    for name, key in keys.items():
        body = eph[key]

        # Astrometric (geometric) topocentric
        ast = body.at(t).observe_from(obs_t)

        # True ecliptic of date (nutation)
        try:
            lon_true, lat_true, _ = ast.apparent().ecliptic_latlon(epoch="date")
        except TypeError:
            lon_true, lat_true, _ = ast.apparent().ecliptic_latlon()

        # Mean ecliptic (J2000) — no nutation
        lon_mean, lat_mean, _ = ast.apparent().ecliptic_latlon()

        # Apparent = apparent true-of-date in ecliptic. Already includes aberration/light-time.
        lon_app = lon_true
        lat_app = lat_true

        entry: Dict[str, Any] = {
            "name": name,
            "mean": {"lon": _wrap360(lon_mean.degrees), "lat": float(lat_mean.degrees)},
            "true": {"lon": _wrap360(lon_true.degrees), "lat": float(lat_true.degrees)},
            "apparent": {"lon": _wrap360(lon_app.degrees), "lat": float(lat_app.degrees)},
        }

        # Optional: refraction in Alt/Az (cannot directly refract ecliptic).
        if temperature_C is not None and pressure_mbar is not None:
            alt, az, _ = ast.apparent().altaz(
                temperature_C=temperature_C, pressure_mbar=pressure_mbar
            )
            entry["topocentric_altaz_refracted"] = {
                "alt_deg": float(alt.degrees),
                "az_deg": float(az.degrees),
            }

        out.append(entry)

    return {"bodies": out, "kernel": "de421", "jd_tt": jd_tt}
