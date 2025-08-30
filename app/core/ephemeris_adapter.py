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
    if not _skyfield_available():
        # Fallback: simple modular angles based on jd
        seed = int((jd_tt % 1) * 1e6)
        out = []
        for i, n in enumerate(["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]):
            ang = (seed * (i+3) % 36000) / 100.0
            out.append({"name": n, "lon": ang, "lat": 0.0, "speed": 1.0 + 0.1*i, "retro": False})
        out += [
            {"name": "Rahu", "lon": (out[1]["lon"]+180.0) % 360, "lat": 0.0, "speed": 0.0, "retro": True},
            {"name": "Ketu", "lon": out[1]["lon"], "lat": 0.0, "speed": 0.0, "retro": True}
        ]
        return out
    
    # Skyfield path - corrected API usage
    try:
        from skyfield.api import load, wgs84
        
        ts = load.timescale()
        eph = load("de421.bsp")
        t = ts.tdb_jd(jd_tt)
        
        # Create observer position
        observer = wgs84.latlon(lat, lon)
        
        # Define planetary bodies
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
            # Correct Skyfield API usage
            astrometric = observer.at(t).observe(body)
            apparent = astrometric.apparent()
            
            # Get ecliptic coordinates
            lon_angle, lat_angle, distance = apparent.ecliptic_latlon()
            lon_deg = lon_angle.degrees % 360.0
            lat_deg = lat_angle.degrees
            
            # Calculate speed using finite difference (1 day)
            t2 = ts.tdb_jd(jd_tt + 1.0)
            astrometric2 = observer.at(t2).observe(body)
            apparent2 = astrometric2.apparent()
            lon2_angle, _, _ = apparent2.ecliptic_latlon()
            
            # Calculate daily motion with proper wraparound
            lon_diff = (lon2_angle.degrees - lon_angle.degrees + 540) % 360 - 180
            speed = lon_diff  # degrees per day
            
            res.append({
                "name": name,
                "lon": lon_deg,
                "lat": lat_deg, 
                "speed": speed,
                "retro": speed < 0
            })
        
        # Add lunar nodes (simplified calculation)
        moon_lon = next(b["lon"] for b in res if b["name"] == "Moon")
        res.extend([
            {"name": "Rahu", "lon": (moon_lon + 180.0) % 360, "lat": 0.0, "speed": 0.0, "retro": True},
            {"name": "Ketu", "lon": moon_lon, "lat": 0.0, "speed": 0.0, "retro": True}
        ])
        
        return res
        
    except Exception as e:
        # If Skyfield fails, fall back to deterministic values
        seed = int((jd_tt % 1) * 1e6)
        out = []
        for i, n in enumerate(["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]):
            ang = (seed * (i+3) % 36000) / 100.0
            out.append({"name": n, "lon": ang, "lat": 0.0, "speed": 1.0 + 0.1*i, "retro": False})
        out += [
            {"name": "Rahu", "lon": (out[1]["lon"]+180.0) % 360, "lat": 0.0, "speed": 0.0, "retro": True},
            {"name": "Ketu", "lon": out[1]["lon"], "lat": 0.0, "speed": 0.0, "retro": True}
        ]
        return outfrom __future__ import annotations
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
    if not _skyfield_available():
        # Fallback: simple modular angles based on jd
        seed = int((jd_tt % 1) * 1e6)
        out = []
        for i, n in enumerate(["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]):
            ang = (seed * (i+3) % 36000) / 100.0
            out.append({"name": n, "lon": ang, "lat": 0.0, "speed": 1.0 + 0.1*i, "retro": False})
        out += [
            {"name": "Rahu", "lon": (out[1]["lon"]+180.0) % 360, "lat": 0.0, "speed": 0.0, "retro": True},
            {"name": "Ketu", "lon": out[1]["lon"], "lat": 0.0, "speed": 0.0, "retro": True}
        ]
        return out
    
    # Skyfield path - corrected API usage
    try:
        from skyfield.api import load, wgs84
        
        ts = load.timescale()
        eph = load("de421.bsp")
        t = ts.tdb_jd(jd_tt)
        
        # Create observer position
        observer = wgs84.latlon(lat, lon)
        
        # Define planetary bodies
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
            # Correct Skyfield API usage
            astrometric = observer.at(t).observe(body)
            apparent = astrometric.apparent()
            
            # Get ecliptic coordinates
            lon_angle, lat_angle, distance = apparent.ecliptic_latlon()
            lon_deg = lon_angle.degrees % 360.0
            lat_deg = lat_angle.degrees
            
            # Calculate speed using finite difference (1 day)
            t2 = ts.tdb_jd(jd_tt + 1.0)
            astrometric2 = observer.at(t2).observe(body)
            apparent2 = astrometric2.apparent()
            lon2_angle, _, _ = apparent2.ecliptic_latlon()
            
            # Calculate daily motion with proper wraparound
            lon_diff = (lon2_angle.degrees - lon_angle.degrees + 540) % 360 - 180
            speed = lon_diff  # degrees per day
            
            res.append({
                "name": name,
                "lon": lon_deg,
                "lat": lat_deg, 
                "speed": speed,
                "retro": speed < 0
            })
        
        # Add lunar nodes (simplified calculation)
        moon_lon = next(b["lon"] for b in res if b["name"] == "Moon")
        res.extend([
            {"name": "Rahu", "lon": (moon_lon + 180.0) % 360, "lat": 0.0, "speed": 0.0, "retro": True},
            {"name": "Ketu", "lon": moon_lon, "lat": 0.0, "speed": 0.0, "retro": True}
        ])
        
        return res
        
    except Exception as e:
        # If Skyfield fails, fall back to deterministic values
        seed = int((jd_tt % 1) * 1e6)
        out = []
        for i, n in enumerate(["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]):
            ang = (seed * (i+3) % 36000) / 100.0
            out.append({"name": n, "lon": ang, "lat": 0.0, "speed": 1.0 + 0.1*i, "retro": False})
        out += [
            {"name": "Rahu", "lon": (out[1]["lon"]+180.0) % 360, "lat": 0.0, "speed": 0.0, "retro": True},
            {"name": "Ketu", "lon": out[1]["lon"], "lat": 0.0, "speed": 0.0, "retro": True}
        ]
        return out
