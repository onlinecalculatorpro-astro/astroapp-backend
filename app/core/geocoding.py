# app/core/geocoding.py
import os, time, requests
from typing import Optional, Dict, Tuple
from functools import lru_cache

NOMINATIM_ENDPOINT   = os.getenv("NOMINATIM_ENDPOINT", "https://nominatim.openstreetmap.org")
NOMINATIM_USER_AGENT = os.environ.get("NOMINATIM_USER_AGENT")  # REQUIRED
GEONAMES_USERNAME    = os.getenv("GEONAMES_USERNAME")          # REQUIRED for timezone
ACCEPT_LANGUAGE      = os.getenv("GEO_ACCEPT_LANGUAGE", "en")
COUNTRY_CODES        = os.getenv("GEO_COUNTRY_CODES")          # e.g. "in,us" (optional)
MIN_DELAY            = float(os.getenv("GEOCODER_MIN_DELAY_SECONDS", "1.1"))
TIMEOUT_SEC          = float(os.getenv("GEOCODER_TIMEOUT", "10"))
MAX_RETRIES          = 3

_last_call = 0.0

def _rate_limit():
    global _last_call
    now = time.time()
    wait = max(0.0, MIN_DELAY - (now - _last_call))
    if wait > 0: time.sleep(wait)
    _last_call = time.time()

def _get_json_with_backoff(url: str, *, params: Dict, headers: Dict) -> Dict:
    delay = 0.6
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limit()
        r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_SEC)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            wait = float(ra) if ra and ra.isdigit() else delay
            time.sleep(wait)
            delay = min(delay * 2, 5.0)
            continue
        r.raise_for_status()
        return r.json()
    # last try (raise if still bad)
    r.raise_for_status()  # type: ignore[name-defined]

@lru_cache(maxsize=2048)
def geocode_place(q: str) -> Optional[Dict]:
    """Nominatim: place -> {label, latitude, longitude}"""
    if not q or not q.strip():
        return None
    if not NOMINATIM_USER_AGENT:
        raise RuntimeError("NOMINATIM_USER_AGENT not set (OSM policy requires a contact UA).")
    params = {
        "q": q, "format": "jsonv2", "addressdetails": 1, "limit": 1,
        "accept-language": ACCEPT_LANGUAGE
    }
    if COUNTRY_CODES:
        params["countrycodes"] = COUNTRY_CODES
    data = _get_json_with_backoff(
        f"{NOMINATIM_ENDPOINT}/search",
        params=params,
        headers={"User-Agent": NOMINATIM_USER_AGENT}
    )
    if not data:
        return None
    hit = data[0]
    try:
        return {
            "label": hit.get("display_name"),
            "latitude": float(hit["lat"]),
            "longitude": float(hit["lon"]),
            "attribution": "© OpenStreetMap contributors",
        }
    except Exception:
        return None

@lru_cache(maxsize=4096)
def tz_from_coords(lat: float, lon: float) -> Optional[str]:
    """GeoNames: lat/lon -> IANA tz (e.g., 'Asia/Kolkata')"""
    if not GEONAMES_USERNAME:
        return None
    params = {"lat": float(lat), "lng": float(lon), "username": GEONAMES_USERNAME}
    data = _get_json_with_backoff("http://api.geonames.org/timezoneJSON", params=params, headers={})
    # GeoNames returns {"status":{"message":"...","value":xx}} on error
    if isinstance(data, dict) and "timezoneId" in data:
        return data["timezoneId"]
    return None

def resolve_place(q: str) -> Optional[Dict]:
    """name -> {label, latitude, longitude, tz, attribution}"""
    base = geocode_place(q)
    if not base:
        return None
    tz = tz_from_coords(base["latitude"], base["longitude"])
    base["tz"] = tz
    return base
