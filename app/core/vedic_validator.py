# app/core/vedic_validator.py
"""
Vedic API — Payload normalization & validation (Vimśottarī only)

Public API:
    normalize_vim_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], list[str], str]

What it does for Vimśottarī:
- Accepts your frontend shape:
    {
      "method": "sidereal|tropical",
      "observer": "geocentric|topocentric",
      "ayanamsa": "lahiri|fagan_bradley|krishnamurti|raman|yukteswar|devore|...",
      "date": "YYYY-MM-DD",
      "time": "HH:MM[:SS]",
      "Place of Birth": "City, State, Country"
    }
- Resolves Place of Birth → latitude, longitude, elevation_m, tz (if resolver available).
- Normalizes to a canonical dict for the core:
    method, ayanamsa, coordinate_mode, topocentric, tz, lat/lon/elevation_m, levels, jd_tt, jd_ut1, etc.
- NEVER returns jd_utc (ERFA-safe).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re

# ── Optional timescales (ERFA-aligned); NO jd_utc here ──
try:
    # Expected signature: build_timescales(date_str, time_str, tz_name, dut1_seconds)
    from app.core.timescales import build_timescales  # type: ignore
    _TIMESCALES_OK = True
except Exception:
    build_timescales = None  # type: ignore
    _TIMESCALES_OK = False

# ── Optional place resolver (geocoding + tz + elevation) ──
# Your project can expose any of these; we try a few names gracefully.
_resolve_place = None
for _modname, _fname in (
    ("app.core.place", "resolve_place"),
    ("app.core.place_resolver", "resolve_place"),
    ("app.core.geo", "resolve_place"),
):
    try:
        _m = __import__(_modname, fromlist=[_fname])  # type: ignore
        _resolve_place = getattr(_m, _fname, None)
        if callable(_resolve_place):
            break
    except Exception:
        _resolve_place = None

# ── Timezone normalization table (lightweight) ──
_TZ_ALIAS = {
    "asia/patna": "Asia/Kolkata",
    "asia/calcutta": "Asia/Kolkata",
    "ist": "Asia/Kolkata",
    "indian standard time": "Asia/Kolkata",
}
_TZ_FALLBACK = "UTC"

# ── Simple helpers ──
_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")

def _coerce_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return default
    return str(x)

def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str) and _NUM_RE.match(x.strip()):
        try:
            return float(x.strip())
        except Exception:
            return None
    return None

def _normalize_tz(tz: Any) -> str:
    if not isinstance(tz, str):
        return _TZ_FALLBACK
    key = tz.strip()
    if not key:
        return _TZ_FALLBACK
    return _TZ_ALIAS.get(key.lower(), key)

def _pad_hms(t: str) -> str:
    """Ensure HH:MM:SS (append :00 if only HH:MM)."""
    t = t.strip()
    if not t:
        return "12:00:00"
    parts = t.split(":")
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}:00"
    if len(parts) == 1:
        hh = parts[0] or "12"
        return f"{hh}:00:00"
    return t

def _clamp_levels(v: Any, default: int = 5) -> int:
    try:
        depth = int(v)
    except Exception:
        if isinstance(v, list):
            depth = len(v)
        else:
            depth = default
    if depth < 1: depth = 1
    if depth > 5: depth = 5
    return depth

def _norm_method(v: Any, default: str = "sidereal") -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("sidereal","nirayana","nirāyaṇa","sid","s"): return "sidereal"
        if s in ("tropical","sayana","sāyana","trop","t"):    return "tropical"
    return default

def _norm_observer(v: Any, default: str = "geocentric") -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("geocentric","geo","center"):    return "geocentric"
        if s in ("topocentric","apparent","obs"): return "topocentric"
    return default

def _norm_ayanamsa(v: Any) -> str:
    # Default to LAHIRI
    if v is None: return "lahiri"
    s = str(v).strip().lower()
    return s or "lahiri"

# ────────────────────────────────────────────────────────────────────────────────

def normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Vimśottarī with your new basis & observer switches.

    Returns:
        (norm, warns, tz_norm)
    """
    warns: List[str] = []

    # ── Extract civic primitives from frontend shape ──
    date = _coerce_str(payload.get("date") or payload.get("birth_date") or "")
    time_in = _coerce_str(payload.get("time") or payload.get("birth_time") or "12:00")
    time_str = _pad_hms(time_in)

    # Place of Birth: mandatory in your product spec (but we avoid raising here to keep route flow).
    pob = (
        payload.get("Place of Birth")
        or payload.get("place")
        or payload.get("birth_place")
        or payload.get("birthPlace")
        or ""
    )
    pob_str = _coerce_str(pob).strip()

    # Observer & method & ayanamsa (defaults as requested)
    method = _norm_method(payload.get("method", payload.get("mode", "sidereal")), default="sidereal")
    observer = _norm_observer(payload.get("observer", "geocentric"))
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    # Optional pre-supplied site & tz (will be overridden by resolver if present)
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _normalize_tz(payload.get("tz") or payload.get("place_tz") or _TZ_FALLBACK)

    # Levels / depth
    depth = _clamp_levels(payload.get("levels", payload.get("depth", payload.get("max_levels", 5))), default=5)

    # ── Resolve place name to site + tz if we can ──
    if not pob_str:
        warns.append("missing_place_of_birth")
    else:
        if callable(_resolve_place):
            try:
                pr = _resolve_place(pob_str)  # expected to return dict-like
                # Try common keys
                _lat = pr.get("lat") if isinstance(pr, dict) else getattr(pr, "lat", None)
                _lon = pr.get("lon") if isinstance(pr, dict) else getattr(pr, "lon", None)
                _tz  = pr.get("tz")  if isinstance(pr, dict) else getattr(pr, "tz", None)
                _elev= pr.get("elevation_m") if isinstance(pr, dict) else getattr(pr, "elevation_m", None)
                if _lat is not None and _lon is not None:
                    lat = _as_float(_lat)
                    lon = _as_float(_lon)
                if _elev is not None:
                    elevation_m = _as_float(_elev)
                if _tz:
                    tz_norm = _normalize_tz(_tz)
                else:
                    warns.append("place_resolved_without_tz:fallback_tz_applied")
            except Exception as e:
                warns.append(f"place_resolution_failed:{e!s}")
        else:
            warns.append("place_resolver_unavailable")

    # ── Observer normalization → coordinate_mode/topocentric flag ──
    coordinate_mode = observer  # "geocentric" | "topocentric"
    topocentric = (coordinate_mode == "topocentric")

    # If topocentric requested but lat/lon missing → fallback to geocentric
    if topocentric and (lat is None or lon is None):
        coordinate_mode = "geocentric"
        topocentric = False
        warns.append("topocentric_requires_coordinates:fallback_geocentric")

    # ── Timescales → jd_tt / jd_ut1 (NO jd_utc) ──
    jd_tt: Optional[float] = None
    jd_ut1: Optional[float] = None

    # Respect caller-provided values if present
    if payload.get("jd_tt") is not None:
        jd_tt = _as_float(payload.get("jd_tt"))
    if payload.get("jd_ut1") is not None:
        jd_ut1 = _as_float(payload.get("jd_ut1"))

    if _TIMESCALES_OK and (jd_tt is None or jd_ut1 is None) and date:
        try:
            ts = build_timescales(date, time_str, tz_norm, 0.0)  # type: ignore[call-arg]
            if isinstance(ts, dict):
                if jd_tt is None and ts.get("jd_tt") is not None:
                    jd_tt = float(ts["jd_tt"])
                if jd_ut1 is None and ts.get("jd_ut1") is not None:
                    jd_ut1 = float(ts["jd_ut1"])
            else:
                if jd_tt is None:
                    jd_tt = float(getattr(ts, "jd_tt"))
                if jd_ut1 is None:
                    jd_ut1 = float(getattr(ts, "jd_ut1"))
        except Exception as e:
            warns.append(f"timescales_failed:{e!s}")

    # ── Canonical normalized dict for the core ──
    norm: Dict[str, Any] = {
        "system": "vimshottari",
        "date": date,
        "time": time_str,
        "tz": tz_norm,
        "method": method,                  # "sidereal" | "tropical"
        "ayanamsa": ayanamsa,              # default "lahiri"
        "levels": depth,
        "max_levels": depth,
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation_m,
        "coordinate_mode": coordinate_mode,  # "geocentric" | "topocentric"
        "topocentric": bool(topocentric),
        "place_name": pob_str or None,
        "jd_tt": jd_tt,
        "jd_ut1": jd_ut1,
        "raw": payload,
    }

    # Registry/common aliases (harmless if unused)
    norm.update({
        "tz_name": tz_norm,
        "ayanamsa_key": ayanamsa,
        "birth_date": date,
        "birth_time": time_str,
        "place_tz": tz_norm,
        "depth": depth,
    })

    # ── Final light sanity notes ──
    if not date:
        warns.append("missing_date")
    if not time_str:
        warns.append("missing_time")
    if tz_norm == _TZ_FALLBACK and payload.get("tz"):
        warns.append("tz_normalization_fallback")

    return norm, warns, tz_norm
