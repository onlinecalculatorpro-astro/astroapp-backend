# app/core/vedic_validator.py
"""
Vedic API — Payload normalization & validation (Vimśottarī + optional Varga request)

Public API:
    normalize_vim_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], list[str], str]

What it does
------------
- Accepts your frontend shape (preferred):
    {
      "method": "sidereal|tropical",
      "observer": "geocentric|topocentric",
      "ayanamsa": "lahiri|fagan_bradley|krishnamurti|raman|yukteswar|devore|...",
      "date": "YYYY-MM-DD",
      "time": "HH:MM[:SS]",
      "place_city": "City",
      "place_state": "State/Region",
      "place_country": "Country",

      // NEW — optional Varga request (no computation here):
      // Any of these keys may be used by the frontend; we normalize them.
      "vargas": ["D9","D10"],
      "varga": "D9, D10",
      "divisional": ["navamsa","dashamsa"],
      "divisional_charts": "d2, d9, trimsamsa",
      "include_vargas": true,
      "varga_zodiac_mode": "sidereal|tropical",   // default → "sidereal"
      "varga_ayanamsa": "lahiri" | 23.856         // default → ayanamsa from payload
    }

- Resolves place → latitude, longitude, elevation_m, tz via astronomy.resolve_place(...) if available.
- Computes time scales (jd_tt/jd_ut1). NEVER returns jd_utc (ERFA-safe).
- Returns a canonical dict for the core, plus an optional **varga_request** block that downstream engines
  (e.g., varga_charts) can use to compute divisional charts.
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

# ── Optional place resolver (geocoding + tz + elevation) via astronomy.py ──
_resolve_place = None
try:
    import app.core.astronomy as _astro  # type: ignore
    for _fname in (
        "resolve_place",          # preferred
        "geocode_place",
        "place_to_site",
        "lookup_place",
        "city_to_coords",
        "resolve_site",
    ):
        _resolve_place = getattr(_astro, _fname, None)
        if callable(_resolve_place):
            break
except Exception:
    _astro = None  # type: ignore
    _resolve_place = None

# ── Optional varga module (for key normalization only; no computation here) ──
try:
    from app.core import varga_charts as _varga  # type: ignore
    _VARGA_OK = True
except Exception:
    _varga = None  # type: ignore
    _VARGA_OK = False

# ── Simple helpers ──
_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


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
    if depth < 1:
        depth = 1
    if depth > 5:
        depth = 5
    return depth


def _norm_method(v: Any, default: str = "sidereal") -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("sidereal", "nirayana", "nirāyaṇa", "sid", "s"):
            return "sidereal"
        if s in ("tropical", "sayana", "sāyana", "trop", "t"):
            return "tropical"
    return default


def _norm_observer(v: Any, default: str = "geocentric") -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("geocentric", "geo", "center"):
            return "geocentric"
        if s in ("topocentric", "apparent", "obs"):
            return "topocentric"
    return default


def _norm_ayanamsa(v: Any) -> str:
    # Default to LAHIRI
    if v is None:
        return "lahiri"
    s = str(v).strip().lower()
    return s or "lahiri"


def _join_place(city: str, state: str, country: str) -> str:
    parts = [p.strip() for p in (city, state, country) if _coerce_str(p).strip()]
    return ", ".join(parts)


# ── Varga key collection & normalization (no computation here) ──
_VARGA_ALIAS_MIN = {
    # minimal alias map to decouple validator from varga internals
    "rasi": "D1", "d1": "D1",
    "hora": "D2", "d2": "D2",
    "drekkana": "D3", "d3": "D3",
    "chaturthamsa": "D4", "d4": "D4",
    "saptamsa": "D7", "d7": "D7",
    "navamsa": "D9", "navamsha": "D9", "d9": "D9",
    "dasamsa": "D10", "dashamsa": "D10", "d10": "D10",
    "dvadasamsa": "D12", "dvadashamsa": "D12", "d12": "D12",
    "shodasamsa": "D16", "sodasamsa": "D16", "d16": "D16",
    "vimsamsa": "D20", "d20": "D20",
    "chaturvimshamsa": "D24", "siddhamsa": "D24", "d24": "D24",
    "bhamsa": "D27", "nakshatramsa": "D27", "d27": "D27",
    "trimsamsa": "D30", "trimsamsha": "D30", "d30": "D30",
    "khavedamsa": "D40", "d40": "D40",
    "akshavedamsa": "D45", "d45": "D45",
    "shashtiamsa": "D60", "shastiamsa": "D60", "d60": "D60",
}


def _canon_varga_key(x: str) -> Optional[str]:
    if not x:
        return None
    k = x.strip()
    if not k:
        return None
    # Prefer project’s varga module resolver if available
    if _VARGA_OK:
        try:
            res = getattr(_varga, "_resolve_key", None)
            if callable(res):
                return str(res(k))
        except Exception:
            pass
    kl = k.lower()
    if kl in _VARGA_ALIAS_MIN:
        return _VARGA_ALIAS_MIN[kl]
    if kl.startswith("d") and kl[1:].isdigit():
        return "D" + kl[1:]
    return None


def _collect_vargas(payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (canon_vargas, warnings). Accept list or comma strings across multiple keys."""
    warns: List[str] = []
    raw_keys: List[str] = []
    for key in ("vargas", "varga", "divisional", "divisional_charts"):
        v = payload.get(key)
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            raw_keys.extend([_coerce_str(x) for x in v])
        else:
            txt = _coerce_str(v)
            raw_keys.extend([p for p in re.split(r"[,\s]+", txt) if p])
    canon: List[str] = []
    bad: List[str] = []
    for x in raw_keys:
        ck = _canon_varga_key(x)
        if ck:
            if ck not in canon:
                canon.append(ck)
        else:
            bad.append(x)
    if bad:
        warns.append("unknown_varga_keys:" + ",".join(bad))
    return canon, warns


# ─────────────────────────────────────────────────────────────────────────────
# Main entry — Vimśottarī + optional Varga payload
# ─────────────────────────────────────────────────────────────────────────────

def normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Vimśottarī with basis (method) & observer switches,
    and optionally include a varga_request block (no varga computation here).

    Returns: (norm, warns, tz_norm)
    """
    warns: List[str] = []

    # ── Extract civic primitives from frontend shape ──
    date = _coerce_str(payload.get("date") or payload.get("birth_date") or "")
    time_in = _coerce_str(payload.get("time") or payload.get("birth_time") or "12:00")
    time_str = _pad_hms(time_in)

    # Place of Birth: accept combined or split fields
    pob_str = _coerce_str(
        payload.get("Place of Birth")
        or payload.get("place")
        or payload.get("birth_place")
        or payload.get("birthPlace")
        or ""
    ).strip()
    if not pob_str:
        pob_str = _join_place(
            _coerce_str(payload.get("place_city")),
            _coerce_str(payload.get("place_state")),
            _coerce_str(payload.get("place_country")),
        )

    # Observer & method & ayanamsa (defaults as requested)
    method = _norm_method(payload.get("method", payload.get("mode", "sidereal")), default="sidereal")
    observer = _norm_observer(payload.get("observer", "geocentric"))
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    # Optional pre-supplied site & tz (will be overridden by resolver if present)
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz") or "UTC").strip() or "UTC"

    # Levels / depth
    depth = _clamp_levels(payload.get("levels", payload.get("depth", payload.get("max_levels", 5))), default=5)

    # ── Resolve place name to site + tz if we can ──
    if not pob_str:
        warns.append("missing_place_of_birth")
    else:
        if callable(_resolve_place):
            try:
                city = _coerce_str(payload.get("place_city")).strip() or None
                state = _coerce_str(payload.get("place_state")).strip() or None
                country = _coerce_str(payload.get("place_country")).strip() or None
                try:
                    pr = _resolve_place(pob_str, place_city=city, place_state=state, place_country=country)  # type: ignore[misc]
                except TypeError:
                    pr = _resolve_place(pob_str)  # type: ignore[misc]

                # Dict or object access
                _lat = pr.get("lat") if isinstance(pr, dict) else getattr(pr, "lat", None)
                _lon = pr.get("lon") if isinstance(pr, dict) else getattr(pr, "lon", None)
                _tz = pr.get("tz") if isinstance(pr, dict) else getattr(pr, "tz", None)
                _elev = pr.get("elevation_m") if isinstance(pr, dict) else getattr(pr, "elevation_m", None)

                if _lat is not None and _lon is not None:
                    lat = _as_float(_lat)
                    lon = _as_float(_lon)
                if _elev is not None:
                    elevation_m = _as_float(_elev)
                if _tz:
                    tz_norm = _coerce_str(_tz).strip() or tz_norm
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

    # ── Varga request normalization (no computation) ──
    vargas, varga_warns = _collect_vargas(payload)
    warns.extend(varga_warns)
    include_vargas = bool(payload.get("include_vargas") or vargas)
    varga_mode = _norm_method(payload.get("varga_zodiac_mode"), default="sidereal")
    varga_ayan = payload.get("varga_ayanamsa", ayanamsa)

    varga_request = None
    if include_vargas and vargas:
        varga_request = {
            "enabled": True,
            "vargas": vargas,                 # canonical keys like D9, D10, D30
            "zodiac_mode": varga_mode,        # "sidereal" (default) or "tropical"
            "ayanamsa": varga_ayan,           # string key or numeric degrees; no JD used here
        }

    # ── Canonical normalized dict for the core ──
    norm: Dict[str, Any] = {
        "system": "vimshottari",
        "date": date,
        "time": time_str,
        "tz": tz_norm,

        "method": method,                    # "sidereal" | "tropical"
        "ayanamsa": ayanamsa,                # default "lahiri"

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

    if varga_request:
        norm["varga_request"] = varga_request

    # ── Final light sanity notes ──
    if not date:
        warns.append("missing_date")
    if not time_str:
        warns.append("missing_time")

    return norm, warns, tz_norm
