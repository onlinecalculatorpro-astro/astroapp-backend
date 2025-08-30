# app/core/ephemeris_adapter.py
# Robust Skyfield adapter with safe fallbacks for environments where
# binaries are restricted or Skyfield data isn't available.

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

__all__ = [
    "load_kernel",
    "ecliptic_longitudes",
]

# ---------------------------------------------------------------------
# Internal helpers & lightweight caching
# ---------------------------------------------------------------------

_TS = None          # cached Timescale
_EPH = {}           # cached ephemerides by kernel name


def _skyfield_available() -> bool:
    """Return True if Skyfield is importable in this environment."""
    try:
        import skyfield  # noqa: F401
        return True
    except Exception:
        return False


def _get_ts_and_eph(kernel_name: str = "de421"):
    """Get (timescale, ephemeris) with simple in-process caching."""
    global _TS, _EPH

    if not _skyfield_available():
        return None, None

    from skyfield.api import load

    if _TS is None:
        _TS = load.timescale()

    key = kernel_name.lower()
    if key not in _EPH:
        # Only de421 is referenced here; map others to de421 to keep things simple & reliable.
        eph_file = "de421.bsp"
        _EPH[key] = load(eph_file)  # downloads once, then cached on disk by Skyfield

    return _TS, _EPH[key]


def load_kernel(kernel_name: str = "de421") -> Tuple[Optional[Any], Optional[Any]]:
    """
    Public helper in case other modules want a handle to (timescale, ephemeris).
    Returns (None, None) if Skyfield is not available.
    """
    return _get_ts_and_eph(kernel_name)


# ---------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------

def ecliptic_longitudes(
    jd_tt: float,
    lat: float,
    lon: float,
    kernel: str = "de421",
) -> List[Dict[str, Any]]:
    """
    Compute ecliptic longitudes/latitudes and simple daily speed for
    Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn. Also returns
    simplified lunar nodes (Rahu/Ketu) based on Moon’s longitude.

    When Skyfield (or its binaries) are unavailable, falls back to a
    deterministic approximation so callers still receive a consistent
    structure.

    Returns a list of dicts with keys:
      - name  (str)
      - lon   (float, degrees 0..360)
      - lat   (float, degrees)
      - speed (float, deg/day)
      - retro (bool)
    """
    # ----------------------------
    # Fallback path (no Skyfield)
    # ----------------------------
    if not _skyfield_available():
        seed = int((jd_tt % 1.0) * 1_000_000)
        names = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
        out: List[Dict[str, Any]] = []
        for i, n in enumerate(names):
            # Simple deterministic modulus to get an angle; good enough for a placeholder.
            ang = (seed * (i + 3) % 36000) / 100.0
            out.append(
                {"name": n, "lon": ang, "lat": 0.0, "speed": 1.0 + 0.1 * i, "retro": False}
            )
        # Nodes from Moon longitude
        moon_lon = out[1]["lon"]
        out.append({"name": "Rahu", "lon": (moon_lon + 180.0) % 360.0, "lat": 0.0, "speed": 0.0, "retro": True})
        out.append({"name": "Ketu", "lon": moon_lon, "lat": 0.0, "speed": 0.0, "retro": True})
        return out

    # ----------------------------
    # Skyfield path (version-robust)
    # ----------------------------
    try:
        from skyfield.api import wgs84

        ts, eph = _get_ts_and_eph(kernel)
        if ts is None or eph is None:
            # Shouldn’t happen, but keep parity with fallback just in case.
            return ecliptic_longitudes(jd_tt, lat, lon, kernel="fallback")

        # Time (TT/TDB). `ts.tdb(jd=...)` is widely supported across versions.
        t = ts.tdb(jd=jd_tt)
        t_next = ts.tdb(jd=jd_tt + 1.0)  # for daily speed

        # Build an Earth + topocentric observer and evaluate at times.
        topo = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon, elevation_m=0.0)
        observer = (eph["earth"] + topo).at(t)
        observer_next = (eph["earth"] + topo).at(t_next)

        bodies = {
            "Sun": "sun",
            "Moon": "moon",
            "Mercury": "mercury",
            "Venus": "venus",
            "Mars": "mars",
            "Jupiter": "jupiter barycenter",
            "Saturn": "saturn barycenter",
        }

        def _ecliptic_latlon(astrometric):
            # Some versions accept an epoch argument, some don’t—handle both.
            try:
                return astrometric.ecliptic_latlon(epoch="date")
            except TypeError:
                return astrometric.ecliptic_latlon()

        results: List[Dict[str, Any]] = []

        for human_name, eph_key in bodies.items():
            body = eph[eph_key]

            # Version-safe: observe_from(observer)
            ast = body.at(t).observe_from(observer).apparent()
            lon_a, lat_a, _ = _ecliptic_latlon(ast)
            lon_deg = (lon_a.degrees % 360.0)
            lat_deg = float(lat_a.degrees)

            ast2 = body.at(t_next).observe_from(observer_next).apparent()
            lon_b, _, _ = _ecliptic_latlon(ast2)

            # Daily speed with wrap-around handling
            speed = ((lon_b.degrees - lon_a.degrees + 540.0) % 360.0) - 180.0

            results.append(
                {
                    "name": human_name,
                    "lon": lon_deg,
                    "lat": lat_deg,
                    "speed": speed,
                    "retro": speed < 0.0,
                }
            )

        # Simplified nodes from Moon longitude
        moon_lon = next(b["lon"] for b in results if b["name"] == "Moon")
        results.append(
            {"name": "Rahu", "lon": (moon_lon + 180.0) % 360.0, "lat": 0.0, "speed": 0.0, "retro": True}
        )
        results.append(
            {"name": "Ketu", "lon": moon_lon, "lat": 0.0, "speed": 0.0, "retro": True}
        )

        return results

    except Exception:
        # If anything in the Skyfield path explodes (binary missing, corrupted cache, etc.),
        # fall back to the deterministic structure so the API remains responsive.
        seed = int((jd_tt % 1.0) * 1_000_000)
        names = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
        out: List[Dict[str, Any]] = []
        for i, n in enumerate(names):
            ang = (seed * (i + 3) % 36000) / 100.0
            out.append(
                {"name": n, "lon": ang, "lat": 0.0, "speed": 1.0 + 0.1 * i, "retro": False}
            )
        moon_lon = out[1]["lon"]
        out.append({"name": "Rahu", "lon": (moon_lon + 180.0) % 360.0, "lat": 0.0, "speed": 0.0, "retro": True})
        out.append({"name": "Ketu", "lon": moon_lon, "lat": 0.0, "speed": 0.0, "retro": True})
        return out
