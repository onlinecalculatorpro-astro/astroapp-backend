from __future__ import annotations
import math
from typing import Dict, Any, List

def _skyfield_available():
    try:
        import skyfield
        return True
    except Exception:
        return False

def load_kernel(kernel_name: str = "de421"):
    if not _skyfield_available():
        return None, None
    from skyfield.api import load
    ts = load.timescale()
    if kernel_name.lower() == "de421":
        eph = load("de421.bsp")  # will download if not present
    else:
        eph = load("de421.bsp")
    return ts, eph

def ecliptic_longitudes(jd_tt: float, lat: float, lon: float, kernel: str = "de421") -> List[Dict[str, Any]]:
    """Return list of bodies with ecliptic longitudes/latitudes/speeds.
    If Skyfield not available, return approximate/fake deterministic values.
    """
    names = ["sun","moon","mercury","venus","mars","jupiter","saturn"]
    if not _skyfield_available():
        # Fallback: simple modular angles based on jd
        seed = int((jd_tt % 1) * 1e6)
        out = []
        for i,n in enumerate(["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"]):
            ang = (seed * (i+3) % 36000) / 100.0
            out.append({"name": n, "lon": ang, "lat": 0.0, "speed": 1.0 + 0.1*i, "retro": False})
        out += [{"name":"Rahu","lon": (out[1]["lon"]+180.0)%360, "lat":0.0, "speed":0.0, "retro": True},
                {"name":"Ketu","lon": out[1]["lon"], "lat":0.0, "speed":0.0, "retro": True}]
        return out
    # Skyfield path
    from skyfield.api import load
    from skyfield.positionlib import ICRF
    from skyfield import almanac
    from skyfield.api import N, W, wgs84
    from skyfield.units import Angle
    from skyfield.data import mpc
    ts = load.timescale()
    eph = load("de421.bsp")
    t = ts.tdb(jd=jd_tt)
    # observer at lat/lon
    obs = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
    bodies = {
        "Sun": eph["sun"],
        "Moon": eph["moon"],
        "Mercury": eph["mercury"],
        "Venus": eph["venus"],
        "Mars": eph["mars"],
        "Jupiter": eph["jupiter barycenter"],
        "Saturn": eph["saturn barycenter"],
    }
    res = []
    for name, body in bodies.items():
        astrometric = obs.at(t).observe(body).apparent()
        lon, lat, distance = astrometric.ecliptic_latlon()
        lon_deg = (lon.degrees % 360.0)
        lat_deg = lat.degrees
        # speed approx: finite diff 1 day
        t2 = ts.tdb(jd=jd_tt+1.0)
        ast2 = obs.at(t2).observe(body).apparent()
        lon2, _, _ = ast2.ecliptic_latlon()
        speed = ((lon2.degrees - lon.degrees + 540) % 360) - 180
        res.append({"name": name, "lon": lon_deg, "lat": lat_deg, "speed": speed, "retro": speed<0})
    # Nodes: opposite points of Moon's path approx (true node computation omitted)
    moon_lon = [b for b in res if b["name"]=="Moon"][0]["lon"]
    res.append({"name":"Rahu","lon": (moon_lon+180.0)%360, "lat":0.0, "speed":0.0, "retro": True})
    res.append({"name":"Ketu","lon": moon_lon, "lat":0.0, "speed":0.0, "retro": True})
    return res
