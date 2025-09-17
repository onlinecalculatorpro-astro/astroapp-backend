import os, time, requests
from typing import Optional, Dict

NOMINATIM_ENDPOINT   = os.getenv("NOMINATIM_ENDPOINT", "https://nominatim.openstreetmap.org")
NOMINATIM_USER_AGENT = os.environ["NOMINATIM_USER_AGENT"]  # required
GEONAMES_USERNAME    = os.getenv("GEONAMES_USERNAME")      # required for timezone lookup
ACCEPT_LANGUAGE      = os.getenv("GEO_ACCEPT_LANGUAGE", "en")
MIN_DELAY            = float(os.getenv("GEOCODER_MIN_DELAY_SECONDS", "1.1"))
TIMEOUT_SEC          = float(os.getenv("GEOCODER_TIMEOUT", "10"))

_last_call = 0.0
def _rate_limit():
    global _last_call
    now = time.time()
    wait = max(0.0, MIN_DELAY - (now - _last_call))
    if wait > 0: time.sleep(wait)
    _last_call = time.time()

def geocode_place(q: str) -> Optional[Dict]:
    """Nominatim free geocoding: place name -> lat/lon/label"""
    if not q: return None
    _rate_limit()
    r = requests.get(
        f"{NOMINATIM_ENDPOINT}/search",
        params={"q": q, "format": "jsonv2", "addressdetails": 1, "limit": 1, "accept-language": ACCEPT_LANGUAGE},
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=TIMEOUT_SEC,
    )
    r.raise_for_status()
    hits = r.json()
    if not hits: return None
    hit = hits[0]
    return {
        "label": hit.get("display_name"),
        "latitude": float(hit["lat"]),
        "longitude": float(hit["lon"]),
    }

def tz_from_coords(lat: float, lon: float) -> Optional[str]:
    """GeoNames timezone lookup: lat/lon -> IANA tz (e.g., Asia/Kolkata)"""
    if not GEONAMES_USERNAME:
        return None
    r = requests.get(
        "http://api.geonames.org/timezoneJSON",
        params={"lat": lat, "lng": lon, "username": GEONAMES_USERNAME},
        timeout=TIMEOUT_SEC,
    )
    r.raise_for_status()
    js = r.json()
    return js.get("timezoneId")

def resolve_place(q: str) -> Optional[Dict]:
    """Convenience: name -> {label, latitude, longitude, tz}"""
    base = geocode_place(q)
    if not base: return None
    tz = tz_from_coords(base["latitude"], base["longitude"])
    base["tz"] = tz
    return base
