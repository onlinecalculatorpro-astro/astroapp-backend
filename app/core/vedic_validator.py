# app/core/vedic_validator.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Vedic API — Payload normalization & validation (Vimśottarī + Yogas)

Public API:
    normalize_vim_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], list[str], str]
    normalize_yoga_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], list[str], str]

Key points
----------
• Sidereal-first defaults: method/zodiac_mode="sidereal", ayanamsa="lahiri".
• For *yoga* normalization there are *no modes* (A/B/C). We only accept civil birth details:
  date, time, tz, latitude, longitude (+ optional place string for auto resolve).
• No precomputed points/cusps are used; if provided, they are ignored with a warning.
• We do NOT compute timescales here for yogas; the yoga core computes those itself.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import re
import inspect

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

# ── Optional timescales (ERFA-aligned) for Vimśottarī only; NO jd_utc here ──
try:
    # Usually: build_timescales(date_str, time_str, tz_name, dut1_seconds)
    from app.core.timescales import build_timescales  # type: ignore
    _TIMESCALES_OK = True
except Exception:
    build_timescales = None  # type: ignore
    _TIMESCALES_OK = False

def _env_dut1_seconds() -> float:
    """Read DUT1 seconds from env with safe defaults."""
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST",
                                    os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0

def _call_build_timescales(date: str, time_str: str, tz_name: str):
    """Compat shim for build_timescales with/without dut1_seconds."""
    if build_timescales is None:
        raise RuntimeError("build_timescales_unavailable")
    try:
        sig = inspect.signature(build_timescales)  # type: ignore
        if len(sig.parameters) >= 4:
            return build_timescales(date, time_str, tz_name, _env_dut1_seconds())  # type: ignore[misc]
        return build_timescales(date, time_str, tz_name)  # type: ignore[misc]
    except Exception as e:
        raise

# ── Optional varga module (for key normalization only; no computation here) ──
try:
    from app.core import varga_charts as _varga  # type: ignore
    _VARGA_OK = True
except Exception:
    _varga = None  # type: ignore
    _VARGA_OK = False


# ── Helpers ──
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
    t = (t or "").strip()
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

def _norm_ayanamsa(v: Any) -> str | float:
    # Default to LAHIRI; allow numeric degrees as pass-through
    if v is None:
        return "lahiri"
    if isinstance(v, (int, float)) or (isinstance(v, str) and _NUM_RE.match(v.strip())):
        try:
            return float(v)  # explicit degrees
        except Exception:
            pass
    s = str(v).strip().lower()
    return s or "lahiri"

def _join_place(city: str, state: str, country: str) -> str:
    parts = [p.strip() for p in (city, state, country) if _coerce_str(p).strip()]
    return ", ".join(parts)


# ── Varga key collection & normalization (no computation here) ──
_VARGA_ALIAS_MIN = {
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
    for key in ("vargas", "varga", "divisional", "divisional_charts", "varga_boost", "varga_keys_for_boost"):
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
# Vimśottarī normalization (unchanged behavior, with DUT1 compat)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Vimśottarī with basis (method) & observer switches,
    and optionally include a varga_request block (no varga computation here).

    Returns: (norm, warns, tz_norm)
    """
    warns: List[str] = []

    # Extract civic primitives
    date = _coerce_str(payload.get("date") or payload.get("birth_date") or "")
    time_in = _coerce_str(payload.get("time") or payload.get("birth_time") or "12:00")
    time_str = _pad_hms(time_in)

    # Place of Birth
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

    # Observer & method & ayanamsa
    method = _norm_method(payload.get("method", payload.get("mode", "sidereal")), default="sidereal")
    observer = _norm_observer(payload.get("observer", "geocentric"))
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    # Optional pre-supplied site & tz
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz") or "UTC").strip() or "UTC"

    # Levels / depth
    depth = _clamp_levels(payload.get("levels", payload.get("depth", payload.get("max_levels", 5))), default=5)

    # Resolve place
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

    # Observer normalization
    coordinate_mode = observer
    topocentric = (coordinate_mode == "topocentric")
    if topocentric and (lat is None or lon is None):
        coordinate_mode = "geocentric"
        topocentric = False
        warns.append("topocentric_requires_coordinates:fallback_geocentric")

    # Timescales → jd_tt / jd_ut1
    jd_tt: Optional[float] = None
    jd_ut1: Optional[float] = None
    if payload.get("jd_tt") is not None:
        jd_tt = _as_float(payload.get("jd_tt"))
    if payload.get("jd_ut1") is not None:
        jd_ut1 = _as_float(payload.get("jd_ut1"))

    if _TIMESCALES_OK and (jd_tt is None or jd_ut1 is None) and date:
        try:
            ts = _call_build_timescales(date, time_str, tz_norm)  # <— compat shim
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

    # Varga request (for Vimśottarī consumers that want it)
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
            "ayanamsa": varga_ayan,           # string key or numeric degrees
        }

    norm: Dict[str, Any] = {
        "system": "vimshottari",
        "date": date,
        "time": time_str,
        "tz": tz_norm,

        "method": method,
        "ayanamsa": ayanamsa,

        "levels": depth,
        "max_levels": depth,

        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation_m,

        "coordinate_mode": coordinate_mode,
        "topocentric": bool(topocentric),

        "place_name": pob_str or None,

        "jd_tt": jd_tt,
        "jd_ut1": jd_ut1,

        "raw": payload,
    }

    # Registry/common aliases
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

    # Final light sanity notes
    if not date:
        warns.append("missing_date")
    if not time_str:
        warns.append("missing_time")

    return norm, warns, tz_norm


# ─────────────────────────────────────────────────────────────────────────────
# Yoga normalization — Mode-C only (birth details → core computes everything)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_yoga_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Yoga detection (Mode-C only).

    Returns: (norm, warns, tz_norm)
    - norm.system = "yoga"
    - Requires: date, time, tz, latitude, longitude (place string optional)
    - No precomputed points/cusps; any such fields are ignored with warnings.
    """
    warns: List[str] = []

    # Global switches (sidereal-first)
    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))
    house_system = _coerce_str(payload.get("house_system") or "placidus").strip() or "placidus"

    # Civil/site inputs
    date = _coerce_str(payload.get("date") or payload.get("birth_date") or "")
    time_str = _pad_hms(_coerce_str(payload.get("time") or payload.get("birth_time") or "12:00"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz") or "UTC").strip() or "UTC"

    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))

    # Optional place resolution (if a place string is given)
    pob_str = _coerce_str(
        payload.get("place")
        or payload.get("birth_place")
        or _join_place(
            _coerce_str(payload.get("place_city")),
            _coerce_str(payload.get("place_state")),
            _coerce_str(payload.get("place_country")),
        )
    ).strip()

    if pob_str:
        if callable(_resolve_place):
            try:
                city = _coerce_str(payload.get("place_city")).strip() or None
                state = _coerce_str(payload.get("place_state")).strip() or None
                country = _coerce_str(payload.get("place_country")).strip() or None
                try:
                    pr = _resolve_place(pob_str, place_city=city, place_state=state, place_country=country)  # type: ignore[misc]
                except TypeError:
                    pr = _resolve_place(pob_str)  # type: ignore[misc]
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
    else:
        warns.append("missing_place_of_birth")

    # Reject/ignore any legacy or precomputed fields
    if any(k in payload for k in ("points_deg", "points_sidereal_deg", "points", "cusps_deg", "cusps_sidereal_deg", "cusps")):
        warns.append("precomputed_points_cusps_ignored")

    if any(k in payload for k in ("jd_tt", "jd_ut1")):
        warns.append("timescales_ignored:core_computes_internally")

    # Include/exclude controls & varga boost hints (for scoring only)
    include: List[str] = []
    inc = payload.get("include") or payload.get("include_yogas") or payload.get("yogas")
    if inc:
        if isinstance(inc, (list, tuple)):
            include = [(_coerce_str(x)).strip() for x in inc if _coerce_str(x).strip()]
        else:
            text = _coerce_str(inc)
            include = [p for p in re.split(r"[,\s]+", text) if p]

    enable_tags = payload.get("enable_tags") or payload.get("enable_catalog_tags") or ()
    if isinstance(enable_tags, str):
        enable_tags = [t for t in re.split(r"[,\s]+", enable_tags) if t]
    disable_tags = payload.get("disable_tags") or payload.get("disable_catalog_tags") or ()
    if isinstance(disable_tags, str):
        disable_tags = [t for t in re.split(r"[,\s]+", disable_tags) if t]

    # Optional varga boost keys for yoga scoring (D9/D10 default in core)
    varga_keys_for_boost, varga_warns = _collect_vargas(payload)
    warns.extend(varga_warns)
    use_vargas_for_scoring = bool(payload.get("use_vargas_for_scoring", True))

    # Build canonical dict for the core yoga engine
    norm: Dict[str, Any] = {
        "system": "yoga",

        # Global switches
        "zodiac_mode": zodiac_mode,     # "sidereal" | "tropical"
        "method": zodiac_mode,          # mirror for parity with other cores
        "ayanamsa": ayanamsa,           # key or degrees
        "house_system": house_system,

        # Civil/site (REQUIRED for Mode-C)
        "date": date or None,
        "time": time_str if date else None,
        "tz": tz_norm,
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation_m,

        # Request-scope filters
        "include": include,
        "enable_catalog_tags": list(enable_tags) if enable_tags else [],
        "disable_catalog_tags": list(disable_tags) if disable_tags else [],

        # Scoring hints
        "use_vargas_for_scoring": bool(use_vargas_for_scoring),
        "varga_keys_for_boost": varga_keys_for_boost or None,

        # Meta & passthrough
        "place_tz": tz_norm,
        "tz_name": tz_norm,
        "place_name": pob_str or None,
        "raw": payload,
    }

    # Basic sanity notes for routes layer
    if not date:
        warns.append("missing_date")
    if not time_str:
        warns.append("missing_time")
    if lat is None or lon is None:
        warns.append("missing_coordinates")

    return norm, warns, tz_norm
