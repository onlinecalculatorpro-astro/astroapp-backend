import os
import time
from typing import Optional, Dict

import requests

# --- Config ---
NOMINATIM_ENDPOINT   = os.getenv("NOMINATIM_ENDPOINT", "https://nominatim.openstreetmap.org")
NOMINATIM_USER_AGENT = os.environ.get("NOMINATIM_USER_AGENT")  # REQUIRED by Nominatim policy
ACCEPT_LANGUAGE      = os.getenv("GEO_ACCEPT_LANGUAGE", "en")
MIN_DELAY            = float(os.getenv("GEOCODER_MIN_DELAY_SECONDS", "1.2"))  # be nice to OSM
TIMEOUT_SEC          = float(os.getenv("GEOCODER_TIMEOUT", "10"))

_last_call = 0.0
def _rate_limit():
    """Simple spacing between Nominatim calls (client-side politeness)."""
    global _last_call
    now = time.time()
    wait = max(0.0, MIN_DELAY - (now - _last_call))
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()

def geocode_place(q: str) -> Optional[Dict]:
    """
    Free geocoding (Nominatim): place name -> {label, latitude, longitude}
    Returns None if nothing found.
    """
    if not q:
        return None
    if not NOMINATIM_USER_AGENT:
        # Fail closed with a clear hint
        raise RuntimeError("NOMINATIM_USER_AGENT is required for geocoding")

    _rate_limit()
    r = requests.get(
        f"{NOMINATIM_ENDPOINT}/search",
        params={
            "q": q,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 1,
            "accept-language": ACCEPT_LANGUAGE,
        },
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=TIMEOUT_SEC,
    )
    r.raise_for_status()
    hits = r.json() or []
    if not hits:
        return None
    hit = hits[0]
    try:
        lat = float(hit["lat"])
        lon = float(hit["lon"])
    except Exception:
        return None
    return {
        "label": hit.get("display_name"),
        "latitude": lat,
        "longitude": lon,
    }

def tz_from_coords(lat: float, lon: float) -> Optional[str]:
    """
    Free timezone lookup (no key) using Open-Meteo forecast API:
    Pass timezone=auto and read the 'timezone' field from the response.
    """
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            # we must request *something*; pick a tiny field, ignore the data:
            "current": "temperature_2m",
            "timezone": "auto",
        },
        timeout=TIMEOUT_SEC,
    )
    r.raise_for_status()
    js = r.json() or {}
    tz = js.get("timezone")
    # sanity: Open-Meteo also returns utc_offset_seconds; not needed, but nice to have
    return tz if isinstance(tz, str) and tz else None

def resolve_place(q: str) -> Optional[Dict]:
    """
    Convenience: name -> {label, latitude, longitude, tz}
    tz may be None if the timezone endpoint can’t resolve.
    """
    base = geocode_place(q)
    if not base:
        return None
    tz = None
    try:
        tz = tz_from_coords(base["latitude"], base["longitude"])
    except Exception:
        # keep geocoding result even if tz service hiccups
        tz = None
    base["tz"] = tz
    return base
