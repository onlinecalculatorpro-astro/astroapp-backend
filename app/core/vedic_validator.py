# app/core/vedic_validator.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Vedic API — Payload normalization & validation (strict geocoding)

Public API
----------
    normalize_vim_payload(payload)      -> (norm, warns, tz_norm)
    normalize_yoga_payload(payload)     -> (norm, warns, tz_norm)

    # Gochar / Ingress / Stations:
    normalize_gochar_payload(payload)   -> (norm, warns, tz_norm)
    normalize_ingress_payload(payload)  -> (norm, warns, tz_norm)
    normalize_stations_payload(payload) -> (norm, warns, tz_norm)

Highlights
----------
• **DOB/TOB/POB aliases** are accepted across validators.
• **Mandatory geocoding** when any place string/parts are provided (incl. POB).
• Sidereal-first defaults (zodiac_mode="sidereal", ayanamsa="lahiri").
• Gochar defaults now include angles in natal targets (Asc, Dsc, MC, IC).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import re
import inspect

# ── astronomy.resolve_place is our REQUIRED geocoder when place is present ──
_RESOLVE_PLACE = None
try:
    import app.core.astronomy as _astro  # type: ignore
    _RESOLVE_PLACE = getattr(_astro, "resolve_place", None)
except Exception:
    _astro = None  # type: ignore
    _RESOLVE_PLACE = None

# ── Optional timescales for Vimśottarī only (ERFA-aligned; no jd_utc here) ──
try:
    from app.core.timescales import build_timescales  # type: ignore
    _TIMESCALES_OK = True
except Exception:
    build_timescales = None  # type: ignore
    _TIMESCALES_OK = False

def _env_dut1_seconds() -> float:
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST",
                                    os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0

def _call_build_timescales(date: str, time_str: str, tz_name: str):
    if build_timescales is None:
        raise RuntimeError("build_timescales_unavailable")
    sig = inspect.signature(build_timescales)  # type: ignore
    if len(sig.parameters) >= 4:
        return build_timescales(date, time_str, tz_name, _env_dut1_seconds())  # type: ignore[misc]
    return build_timescales(date, time_str, tz_name)  # type: ignore[misc]

# ── Optional varga key normalizer (no computation here) ──
try:
    from app.core import varga_charts as _varga  # type: ignore
    _VARGA_OK = True
except Exception:
    _varga = None  # type: ignore
    _VARGA_OK = False


# ── Basic helpers ──
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
        depth = len(v) if isinstance(v, list) else default
    return max(1, min(5, depth))

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

def _norm_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1","true","yes","y","on"): return True
        if s in ("0","false","no","n","off"): return False
    return default

def _norm_ayanamsa(v: Any) -> str | float:
    if v is None:
        return "lahiri"
    if isinstance(v, (int, float)) or (isinstance(v, str) and _NUM_RE.match(v.strip())):
        try:
            return float(v)
        except Exception:
            pass
    s = str(v).strip().lower()
    return s or "lahiri"

def _join_place(city: str, state: str, country: str) -> str:
    parts = [p.strip() for p in (city, state, country) if _coerce_str(p).strip()]
    return ", ".join(parts)

def _split_csv_or_list(v: Any) -> List[str]:
    out: List[str] = []
    if not v:
        return out
    if isinstance(v, (list, tuple)):
        for x in v:
            s = _coerce_str(x).strip()
            if s:
                out.append(s)
        return out
    for p in re.split(r"[,\s]+", _coerce_str(v)):
        if p:
            out.append(p.strip())
    return out

def _collect_bodies(payload: Dict[str, Any], *, key_order: Tuple[str, ...], default: List[str]) -> List[str]:
    for k in key_order:
        v = payload.get(k)
        if v:
            items = _split_csv_or_list(v)
            return items or list(default)
    return list(default)

def _collect_targets(payload: Dict[str, Any], *, key_order: Tuple[str, ...], default: List[str]) -> List[str]:
    return _collect_bodies(payload, key_order=key_order, default=default)

def _collect_orb_map(payload: Dict[str, Any], *, base_orb: float) -> Tuple[float, Dict[str, float]]:
    orb_deg = _as_float(payload.get("orb") or payload.get("orb_deg") or payload.get("max_orb"))
    if orb_deg is None:
        orb_deg = float(base_orb)
    om: Dict[str, float] = {}
    v = payload.get("orbs") or payload.get("orb_map") or {}
    if isinstance(v, dict):
        for k, val in v.items():
            f = _as_float(val)
            if f is not None:
                om[_coerce_str(k)] = f
    return float(orb_deg), om

def _collect_time_window(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Parse date window; return (from, to, warns)."""
    warns: List[str] = []
    date_from = None
    date_to = None

    win = payload.get("window") or payload.get("time_window") or payload.get("range") or {}
    if isinstance(win, dict):
        date_from = _coerce_str(win.get("from") or win.get("start"))
        date_to   = _coerce_str(win.get("to")   or win.get("end"))

    tr = payload.get("time_range") or payload.get("timerange")
    if (not date_from or not date_to) and isinstance(tr, (list, tuple)) and len(tr) == 2:
        date_from = date_from or _coerce_str(tr[0])
        date_to   = date_to   or _coerce_str(tr[1])

    date_from = date_from or _coerce_str(payload.get("date_from") or payload.get("from"))
    date_to   = date_to   or _coerce_str(payload.get("date_to")   or payload.get("to"))

    if not (date_from and date_to):
        warns.append("missing_time_range")
    return (date_from or None, date_to or None, warns)


# ── Varga key normalization (no computation here) ──
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


# ────────────────────────── Strict place resolution ──────────────────────────
def _must_resolve_place_if_provided(p: Dict[str, Any],
                                    *,
                                    place_prefix: str = "",
                                    warns: List[str]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    If any place string/parts are present, we MUST resolve via astronomy.resolve_place().
    Returns (lat, lon, elev_m, tz_norm, fatal_reason).
    """
    # Gather any place hints (now includes POB alias)
    pob_str = _coerce_str(
        p.get(f"{place_prefix}place") or
        p.get(f"{place_prefix}birth_place") or
        p.get(f"{place_prefix}POB") or
        _join_place(
            _coerce_str(p.get(f"{place_prefix}place_city")),
            _coerce_str(p.get(f"{place_prefix}place_state")),
            _coerce_str(p.get(f"{place_prefix}place_country")),
        )
    ).strip()

    has_any_place = bool(
        pob_str or
        _coerce_str(p.get(f"{place_prefix}place_city")).strip() or
        _coerce_str(p.get(f"{place_prefix}place_state")).strip() or
        _coerce_str(p.get(f"{place_prefix}place_country")).strip() or
        _coerce_str(p.get(f"{place_prefix}POB")).strip()
    )

    if not has_any_place:
        # No place input provided → do nothing; caller may still pass explicit lat/lon/tz.
        return None, None, None, None, None

    # Resolver is required when place is present
    if not callable(_RESOLVE_PLACE):
        warns.append("place_resolver_unavailable")
        return None, None, None, None, "place_resolver_unavailable"

    city = _coerce_str(p.get(f"{place_prefix}place_city")).strip() or None
    state = _coerce_str(p.get(f"{place_prefix}place_state")).strip() or None
    country = _coerce_str(p.get(f"{place_prefix}place_country")).strip() or None

    try:
        try:
            pr = _RESOLVE_PLACE(pob_str, place_city=city, place_state=state, place_country=country)  # type: ignore[misc]
        except TypeError:
            pr = _RESOLVE_PLACE(pob_str)  # type: ignore[misc]
    except Exception as e:
        warns.append(f"place_resolution_failed:{e!s}")
        return None, None, None, None, f"place_resolution_failed:{e!s}"

    if not isinstance(pr, dict):
        warns.append("place_resolution_failed:bad_shape")
        return None, None, None, None, "place_resolution_failed:bad_shape"

    lat = pr.get("lat"); lon = pr.get("lon")
    tz  = pr.get("tz") or pr.get("timezone")
    elev = pr.get("elevation_m", pr.get("elevation"))

    if lat is None or lon is None or not tz:
        warns.append("place_resolution_failed:incomplete_result")
        return None, None, None, None, "place_resolution_failed:incomplete_result"

    latf = _as_float(lat); lonf = _as_float(lon)
    elevf = _as_float(elev) if elev not in (None, "") else None
    tz_norm = _coerce_str(tz).strip() or None

    return latf, lonf, elevf, tz_norm, None


# ───────────────────────────── Vimśottarī ─────────────────────────────
def normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []

    # Civil primitives (DOB/TOB accepted)
    date = _coerce_str(payload.get("DOB") or payload.get("date") or payload.get("birth_date") or "")
    time_in = _coerce_str(payload.get("TOB") or payload.get("time") or payload.get("birth_time") or "12:00")
    time_str = _pad_hms(time_in)

    # Method & ayanamsa
    method = _norm_method(payload.get("method", payload.get("mode", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    # Explicit site fields (accepted), but if place is present we must geocode
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    # Also accept POB
    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload)
        payload["place"] = payload.get("POB")

    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        norm = {
            "system": "vimshottari",
            "date": date, "time": time_str, "tz": tz_norm,
            "method": method, "ayanamsa": ayanamsa,
            "latitude": None, "longitude": None, "elevation_m": None,
            "coordinate_mode": "geocentric", "topocentric": False,
            "jd_tt": None, "jd_ut1": None,
            "fatal": fatal,
            "raw": payload,
            "tz_name": tz_norm, "ayanamsa_key": ayanamsa,
            "birth_date": date, "birth_time": time_str, "place_tz": tz_norm,
        }
        return norm, warns + ["fatal"], tz_norm or "UTC"

    # Adopt resolved values when available; otherwise keep explicit ones
    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    coordinate_mode = _norm_observer(payload.get("observer"), "geocentric")
    topocentric = (coordinate_mode == "topocentric")
    if topocentric and (lat is None or lon is None):
        coordinate_mode = "geocentric"
        topocentric = False
        warns.append("topocentric_requires_coordinates:fallback_geocentric")

    # Timescales → jd_tt / jd_ut1 (only for Vim)
    jd_tt: Optional[float] = _as_float(payload.get("jd_tt"))
    jd_ut1: Optional[float] = _as_float(payload.get("jd_ut1"))
    if _TIMESCALES_OK and (jd_tt is None or jd_ut1 is None) and date and tz_norm:
        try:
            ts = _call_build_timescales(date, time_str, tz_norm)
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

    # Varga request hints (optional)
    vargas, varga_warns = _collect_vargas(payload)
    warns.extend(varga_warns)
    include_vargas = bool(payload.get("include_vargas") or vargas)
    varga_mode = _norm_method(payload.get("varga_zodiac_mode"), default="sidereal")
    varga_ayan = payload.get("varga_ayanamsa", ayanamsa)
    varga_request = None
    if include_vargas and vargas:
        varga_request = {
            "enabled": True,
            "vargas": vargas,
            "zodiac_mode": varga_mode,
            "ayanamsa": varga_ayan,
        }

    norm: Dict[str, Any] = {
        "system": "vimshottari",
        "date": date or None,
        "time": time_str if date else None,
        "tz": tz_norm,
        "method": method,
        "ayanamsa": ayanamsa,
        "levels": _clamp_levels(payload.get("levels", payload.get("depth", payload.get("max_levels", 5))), default=5),
        "latitude": lat, "longitude": lon, "elevation_m": elevation_m,
        "coordinate_mode": coordinate_mode, "topocentric": bool(topocentric),
        "jd_tt": jd_tt, "jd_ut1": jd_ut1,
        "raw": payload,
        "tz_name": tz_norm, "ayanamsa_key": ayanamsa,
        "birth_date": date, "birth_time": time_str, "place_tz": tz_norm,
        "place_name": _coerce_str(payload.get("place") or payload.get("birth_place") or payload.get("POB") or "").strip() or None,
    }
    if varga_request:
        norm["varga_request"] = varga_request

    if not date: warns.append("missing_date")
    if not time_str: warns.append("missing_time")
    return norm, warns, tz_norm or "UTC"


# ───────────────────────────── Yoga (Mode-C) ──────────────────────────
def normalize_yoga_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []

    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))
    house_system = _coerce_str(payload.get("house_system") or "placidus").strip() or "placidus"

    # DOB/TOB accepted
    date = _coerce_str(payload.get("DOB") or payload.get("date") or payload.get("birth_date") or "")
    time_str = _pad_hms(_coerce_str(payload.get("TOB") or payload.get("time") or payload.get("birth_time") or "12:00"))

    # Accept explicit coords/tz, but enforce geocoding if a place was given
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    # Also accept POB
    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload)
        payload["place"] = payload.get("POB")

    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        norm = {
            "system": "yoga",
            "zodiac_mode": zodiac_mode, "method": zodiac_mode,
            "ayanamsa": ayanamsa, "house_system": house_system,
            "date": date or None, "time": time_str if date else None,
            "tz": tz_norm, "latitude": None, "longitude": None, "elevation_m": None,
            "use_vargas_for_scoring": bool(payload.get("use_vargas_for_scoring", True)),
            "enable_catalog_tags": [], "disable_catalog_tags": [],
            "fatal": fatal,
            "raw": payload,
        }
        return norm, warns + ["fatal"], tz_norm or "UTC"

    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    include: List[str] = []
    inc = payload.get("include") or payload.get("include_yogas") or payload.get("yogas")
    if inc:
        include = _split_csv_or_list(inc)

    enable_tags = payload.get("enable_tags") or payload.get("enable_catalog_tags") or ()
    if isinstance(enable_tags, str):
        enable_tags = [t for t in re.split(r"[,\s]+", enable_tags) if t]
    disable_tags = payload.get("disable_tags") or payload.get("disable_catalog_tags") or ()
    if isinstance(disable_tags, str):
        disable_tags = [t for t in re.split(r"[,\s]+", disable_tags) if t]

    varga_keys_for_boost, varga_warns = _collect_vargas(payload)
    warns.extend(varga_warns)
    use_vargas_for_scoring = bool(payload.get("use_vargas_for_scoring", True))

    norm: Dict[str, Any] = {
        "system": "yoga",
        "zodiac_mode": zodiac_mode, "method": zodiac_mode,
        "ayanamsa": ayanamsa, "house_system": house_system,
        "date": date or None, "time": time_str if date else None,
        "tz": tz_norm,
        "latitude": lat, "longitude": lon, "elevation_m": elevation_m,
        "include": include,
        "enable_catalog_tags": list(enable_tags) if enable_tags else [],
        "disable_catalog_tags": list(disable_tags) if disable_tags else [],
        "use_vargas_for_scoring": bool(use_vargas_for_scoring),
        "varga_keys_for_boost": varga_keys_for_boost or None,
        "place_tz": tz_norm, "tz_name": tz_norm,
        "place_name": _coerce_str(payload.get("place") or payload.get("birth_place") or payload.get("POB") or "").strip() or None,
        "raw": payload,
    }

    if not date: warns.append("missing_date")
    if not time_str: warns.append("missing_time")
    if lat is None or lon is None: warns.append("missing_coordinates")
    return norm, warns, tz_norm or "UTC"


# ───────────────────────────── Gochar bundle ──────────────────────────
_DEFAULT_MOVERS  = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"]
# Expanded targets include angles; nodes are still opt-in via include_nodes
_DEFAULT_TARGETS = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn",
                    "Ascendant","Descendant","MC","IC"]

def normalize_gochar_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []

    frame = _coerce_str(payload.get("frame") or "ecliptic-of-date").strip() or "ecliptic-of-date"
    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))
    include_nodes = _norm_bool(payload.get("include_nodes"), False)
    treat_nodes_like_saturn = _norm_bool(payload.get("treat_nodes_like_saturn"), False)
    step_minutes: Union[str,float,int] = payload.get("step_minutes", "auto")

    # Selectors are optional; defaults used when omitted
    movers = _collect_bodies(payload, key_order=("movers","bodies","transiting_bodies"), default=_DEFAULT_MOVERS)
    if include_nodes:
        if "Rahu" not in movers: movers.append("Rahu")
        if "Ketu" not in movers: movers.append("Ketu")
    natal_targets = _collect_targets(payload, key_order=("targets","natal_targets"), default=_DEFAULT_TARGETS)
    orb_deg, orb_map = _collect_orb_map(payload, base_orb=12.0)

    # Natal block (either nested "natal" or flat)
    natal_in = payload.get("natal") or payload

    # Civil primitives for natal binding (DOB/TOB accepted; optional if longitudes/jd supplied)
    date = _coerce_str(natal_in.get("DOB") or natal_in.get("date") or natal_in.get("birth_date") or "")
    time_str = _pad_hms(_coerce_str(natal_in.get("TOB") or natal_in.get("time") or natal_in.get("birth_time") or "12:00"))
    tz_norm = _coerce_str(natal_in.get("tz") or natal_in.get("place_tz")).strip() or None

    # Accept explicit coords; but enforce geocoding if a place/POB string/parts are present
    lat = _as_float(natal_in.get("latitude") or natal_in.get("lat"))
    lon = _as_float(natal_in.get("longitude") or natal_in.get("lon"))
    elevation_m = _as_float(natal_in.get("elevation_m") or natal_in.get("elevation"))

    # Accept POB in natal block
    if not any([natal_in.get("place"), natal_in.get("birth_place"), natal_in.get("place_city"),
                natal_in.get("place_state"), natal_in.get("place_country")]) and natal_in.get("POB"):
        natal_in = dict(natal_in)
        natal_in["place"] = natal_in.get("POB")

    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(natal_in, warns=warns)
    if fatal:
        norm = {
            "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa,
            "include_nodes": bool(include_nodes), "treat_nodes_like_saturn": bool(treat_nodes_like_saturn),
            "step_minutes": step_minutes,
            "movers": movers, "natal_targets": natal_targets,
            "orb_deg": float(orb_deg), "orb_map": orb_map or {},
            "natal_chart": {
                "date": date or None, "time": time_str if date else None,
                "tz": tz_norm, "place_tz": tz_norm,
                "place_name": _coerce_str(natal_in.get("place") or natal_in.get("birth_place") or natal_in.get("POB") or "").strip() or None,
                "latitude": None, "longitude": None, "elevation_m": None,
                "longitudes": natal_in.get("longitudes") or natal_in.get("ecliptic_longitudes") or None,
                "natal_jd_tt": _as_float(natal_in.get("natal_jd_tt") or natal_in.get("jd_tt")),
                "jd_utc": _as_float(natal_in.get("jd_utc")),
            },
            "time_range": None,  # filled below if present
            "fatal": fatal,
            "raw": payload,
            "transiting_bodies": movers, "ayanamsa_key": ayanamsa,
            "tz_name": tz_norm,
        }
        d0, d1, _ = _collect_time_window(payload)
        if d0 and d1:
            norm["time_range"] = [d0, d1]
        return norm, warns + ["fatal"], tz_norm or "UTC"

    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    # Time window (civil strings)
    date_from, date_to, win_warns = _collect_time_window(payload)
    warns.extend(win_warns)

    longitudes = natal_in.get("longitudes") or natal_in.get("ecliptic_longitudes") or None
    natal_jd_tt = _as_float(natal_in.get("natal_jd_tt") or natal_in.get("jd_tt"))
    natal_jd_utc = _as_float(natal_in.get("jd_utc"))

    norm: Dict[str, Any] = {
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa": ayanamsa,
        "include_nodes": bool(include_nodes),
        "treat_nodes_like_saturn": bool(treat_nodes_like_saturn),
        "step_minutes": step_minutes,
        "movers": movers,
        "natal_targets": natal_targets,
        "orb_deg": float(orb_deg),
        "orb_map": orb_map or {},
        "natal_chart": {
            "date": date or None,
            "time": time_str if date else None,
            "tz": tz_norm,
            "place_tz": tz_norm,
            "place_name": _coerce_str(natal_in.get("place") or natal_in.get("birth_place") or natal_in.get("POB") or "").strip() or None,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elevation_m,
            "longitudes": longitudes or None,
            "natal_jd_tt": natal_jd_tt,
            "jd_utc": natal_jd_utc,
        },
        "time_range": [date_from, date_to] if (date_from and date_to) else None,
        "raw": payload,
        "transiting_bodies": movers,
        "ayanamsa_key": ayanamsa,
        "tz_name": tz_norm,
    }

    if not (date_from and date_to): warns.append("gochar_missing_window")
    if not movers: warns.append("gochar_missing_movers")
    if not natal_targets: warns.append("gochar_missing_targets")
    if longitudes is None and not (natal_jd_tt or natal_jd_utc or (date and tz_norm)):
        warns.append("gochar_missing_natal_binding")

    return norm, warns, tz_norm or "UTC"


# ───────────────────────────── Ingress scans ──────────────────────────
def normalize_ingress_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []
    frame = _coerce_str(payload.get("frame") or "ecliptic-of-date").strip() or "ecliptic-of-date"
    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    # Optional topocentric (requires coords). If place is given (incl. POB), we must resolve it.
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload)
        payload["place"] = payload.get("POB")

    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        norm = {
            "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa,
            "movers": [], "time_range": None, "step_minutes": payload.get("step_minutes", "auto"),
            "topocentric": False, "latitude": None, "longitude": None, "elevation_m": None,
            "tz_name": tz_norm, "fatal": fatal, "raw": payload,
        }
        d0, d1, _ = _collect_time_window(payload)
        if d0 and d1:
            norm["time_range"] = [d0, d1]
        return norm, warns + ["fatal"], tz_norm or "UTC"

    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    movers = _collect_bodies(payload, key_order=("movers","bodies","transiting_bodies"), default=["Sun","Mercury","Venus","Mars","Jupiter","Saturn"])
    if _norm_bool(payload.get("include_moon"), True) and "Moon" not in movers:
        movers.insert(0, "Moon")

    date_from, date_to, win_warns = _collect_time_window(payload)
    warns.extend(win_warns)

    topocentric = _norm_observer(payload.get("observer"), "geocentric") == "topocentric"
    if topocentric and (lat is None or lon is None):
        topocentric = False
        warns.append("topocentric_requires_coordinates:fallback_geocentric")

    norm = {
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa": ayanamsa,
        "movers": movers,
        "time_range": [date_from, date_to] if (date_from and date_to) else None,
        "step_minutes": payload.get("step_minutes", "auto"),
        "topocentric": bool(topocentric),
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation_m,
        "tz_name": tz_norm,
        "raw": payload,
    }
    if not (date_from and date_to):
        warns.append("ingress_missing_window")

    return norm, warns, tz_norm or "UTC"


# ───────────────────────────── Stations scans ─────────────────────────
def normalize_stations_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []
    frame = _coerce_str(payload.get("frame") or "ecliptic-of-date").strip() or "ecliptic-of-date"
    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    # Optional topocentric (requires coords). If place is given (incl. POB), we must resolve it.
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload)
        payload["place"] = payload.get("POB")

    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        norm = {
            "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa,
            "movers": [], "time_range": None, "step_minutes": payload.get("step_minutes", "auto"),
            "topocentric": False, "latitude": None, "longitude": None, "elevation_m": None,
            "tz_name": tz_norm, "fatal": fatal, "raw": payload,
        }
        d0, d1, _ = _collect_time_window(payload)
        if d0 and d1:
            norm["time_range"] = [d0, d1]
        return norm, warns + ["fatal"], tz_norm or "UTC"

    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    movers = _collect_bodies(payload, key_order=("movers","bodies","transiting_bodies"), default=["Mercury","Venus","Mars","Jupiter","Saturn"])
    date_from, date_to, win_warns = _collect_time_window(payload)
    warns.extend(win_warns)

    topocentric = _norm_observer(payload.get("observer"), "geocentric") == "topocentric"
    if topocentric and (lat is None or lon is None):
        topocentric = False
        warns.append("topocentric_requires_coordinates:fallback_geocentric")

    norm = {
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa": ayanamsa,
        "movers": movers,
        "time_range": [date_from, date_to] if (date_from and date_to) else None,
        "step_minutes": payload.get("step_minutes", "auto"),
        "topocentric": bool(topocentric),
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation_m,
        "tz_name": tz_norm,
        "raw": payload,
    }
    if not (date_from and date_to):
        warns.append("stations_missing_window")

    return norm, warns, tz_norm or "UTC"
