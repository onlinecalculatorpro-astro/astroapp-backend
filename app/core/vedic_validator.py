# app/core/vedic_validator.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Vedic API — Payload normalization & validation (strict geocoding; core-wired)

Public API
----------
    normalize_vim_payload(payload)          -> (norm, warns, tz_norm)
    normalize_yoga_payload(payload)         -> (norm, warns, tz_norm)

    # Gochar / Ingress / Stations:
    normalize_gochar_payload(payload)       -> (norm, warns, tz_norm)
    normalize_ingress_payload(payload)      -> (norm, warns, tz_norm)
    normalize_stations_payload(payload)     -> (norm, warns, tz_norm)

    # Engines backed by core chart:
    normalize_shadbala_payload(payload)     -> (norm, warns, tz_norm)
    normalize_ashtakavarga_payload(payload) -> (norm, warns, tz_norm)

    # Horary / Prasna (Parāśari + KP + Hybrid):
    normalize_horary_payload(payload)       -> (norm, warns, tz_norm)

Key policies
------------
• Sidereal-first defaults (zodiac_mode="sidereal", ayanamsa="lahiri").
• If ANY place/POB string/parts are present, we MUST resolve via a resolver
  (preferred: app.core.geocoding.resolve_place; fallback: app.core.astronomy.resolve_place)
  — even if lat/lon/tz were also provided. On failure, we return a fatal reason.
• Timescales helper used where appropriate (no jd leak to other systems).
• Ashtakavarga: passes through `ruleset` and validated `ruleset_map` to core;
  accepts optional `angles` {asc, mc}.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Set
from enum import Enum
from datetime import datetime, timezone
import os
import re
import inspect

# ── (Optional) link to horary primitives for enums (no heavy imports at import time) ──
try:
    from app.core.horary import QuestionType as _QuestionType  # for Enum mapping if available
    _HORARY_ENUM_OK = True
except Exception:
    _QuestionType = None  # type: ignore
    _HORARY_ENUM_OK = False

# ── Required geocoder when any place/POB is present (prefer dedicated module) ──
_RESOLVE_PLACE = None
try:
    from app.core.geocoding import resolve_place as _RESOLVE_PLACE  # type: ignore
except Exception:
    try:
        # Fallback to astronomy.resolve_place (exported in your astronomy core)
        from app.core.astronomy import resolve_place as _RESOLVE_PLACE  # type: ignore
    except Exception:
        _RESOLVE_PLACE = None  # type: ignore

# ── Optional timescales where used ──
try:
    from app.core.timescales import build_timescales  # type: ignore
    _TIMESCALES_OK = True
except Exception:
    build_timescales = None  # type: ignore
    _TIMESCALES_OK = False

# ── Optional varga key normalizer (no computation here) ──
try:
    from app.core import varga_charts as _varga  # type: ignore
    _VARGA_OK = True
except Exception:
    _varga = None  # type: ignore
    _VARGA_OK = False


# ============================ Basic helpers ============================

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
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
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

def _coerce_step_minutes(x: Any, default: Union[str, float, int] = "auto") -> Union[str, float, int]:
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            try:
                return float(s)
            except Exception:
                return default
        return s or default
    return default


# ============================ Geocoding (strict) ============================

def _must_resolve_place_if_provided(
    p: Dict[str, Any], *, place_prefix: str = "", warns: List[str]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    STRICT: If any place string/parts/POB present, we MUST call resolve_place(),
    even if lat/lon/tz were already provided. On failure → fatal.
    Returns (lat, lon, elev_m, tz_norm, fatal_reason).
    """
    pob_str = _coerce_str(
        p.get(f"{place_prefix}place")
        or p.get(f"{place_prefix}birth_place")
        or p.get(f"{place_prefix}POB")
        or _join_place(
            _coerce_str(p.get(f"{place_prefix}place_city")),
            _coerce_str(p.get(f"{place_prefix}place_state")),
            _coerce_str(p.get(f"{place_prefix}place_country")),
        )
    ).strip()

    has_any_place = bool(
        pob_str
        or _coerce_str(p.get(f"{place_prefix}place_city")).strip()
        or _coerce_str(p.get(f"{place_prefix}place_state")).strip()
        or _coerce_str(p.get(f"{place_prefix}place_country")).strip()
        or _coerce_str(p.get(f"{place_prefix}POB")).strip()
    )

    if not has_any_place:
        return None, None, None, None, None

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

    lat = pr.get("latitude") if "latitude" in pr else pr.get("lat")
    lon = pr.get("longitude") if "longitude" in pr else pr.get("lon")
    tz = pr.get("tz") or pr.get("timezone")
    elev = pr.get("elevation_m", pr.get("elevation"))

    if lat is None or lon is None or not tz:
        warns.append("place_resolution_failed:incomplete_result")
        return None, None, None, None, "place_resolution_failed:incomplete_result"

    latf = _as_float(lat)
    lonf = _as_float(lon)
    elevf = _as_float(elev) if elev not in (None, "") else None
    tz_norm = _coerce_str(tz).strip() or None
    return latf, lonf, elevf, tz_norm, None


# ============================ Timescales helpers ============================

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


# ============================ Varga key helpers ============================

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


# ============================ Ruleset helpers (Ashtakavarga) ============================

_AV_PLANETS = {"Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"}

def _norm_ruleset_name(v: Any) -> str:
    s = _coerce_str(v).strip().lower()
    if s in ("custom", "user", "override"):
        return "custom"
    return "parashari-bphs"

def _validate_ruleset_map(m: Any) -> Tuple[Optional[Dict[str, Dict[str, List[int]]]], List[str]]:
    """
    Very light validator so the core doesn’t get garbage. Exact enforcement happens in core.
    Expect: {giver->{receiver->[offsets 1..12]}} receivers ∈ planets ∪ {"Lagna","Asc"}.
    """
    warns: List[str] = []
    if not isinstance(m, dict):
        return None, ["ruleset_map_invalid:shape"]
    out: Dict[str, Dict[str, List[int]]] = {}
    for giver, rmap in m.items():
        if giver not in _AV_PLANETS:
            warns.append(f"ruleset_map_invalid:unknown_giver:{giver}")
            continue
        if not isinstance(rmap, dict):
            warns.append(f"ruleset_map_invalid:receiver_map:{giver}")
            continue
        gout: Dict[str, List[int]] = {}
        for receiver, offs in rmap.items():
            if receiver not in _AV_PLANETS and receiver not in ("Lagna", "Asc"):
                warns.append(f"ruleset_map_invalid:unknown_receiver:{giver}->{receiver}")
                continue
            if not isinstance(offs, (list, tuple)) or not all(isinstance(o, int) for o in offs):
                warns.append(f"ruleset_map_invalid:offsets_type:{giver}->{receiver}")
                continue
            offs_ok: List[int] = []
            for o in offs:
                try:
                    oi = int(o)
                except Exception:
                    warns.append(f"ruleset_map_invalid:offset_type:{giver}->{receiver}:{o}")
                    continue
                if 1 <= oi <= 12:
                    offs_ok.append(oi)
                else:
                    warns.append(f"ruleset_map_invalid:offset_range:{giver}->{receiver}:{o}")
            if offs_ok:
                gout["Lagna" if receiver in ("Lagna", "Asc") else receiver] = offs_ok
        if gout:
            out[giver] = gout
    if not out:
        return None, warns or ["ruleset_map_invalid:empty"]
    return out, warns


# ============================ Collectors ============================

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
    warns: List[str] = []
    date_from = None
    date_to = None
    win = payload.get("window") or payload.get("time_window") or payload.get("range") or {}
    if isinstance(win, dict):
        date_from = _coerce_str(win.get("from") or win.get("start"))
        date_to = _coerce_str(win.get("to") or win.get("end"))
    tr = payload.get("time_range") or payload.get("timerange")
    if (not date_from and not date_to) and isinstance(tr, (list, tuple)) and len(tr) == 2:
        date_from = date_from or _coerce_str(tr[0])
        date_to = date_to or _coerce_str(tr[1])
    date_from = date_from or _coerce_str(payload.get("date_from") or payload.get("from"))
    date_to = date_to or _coerce_str(payload.get("date_to") or payload.get("to"))
    if not (date_from and date_to):
        warns.append("missing_time_range")
    return (date_from or None, date_to or None, warns)


# ============================ Normalizers: Vimshottari ============================

def normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []

    date = _coerce_str(payload.get("DOB") or payload.get("date") or payload.get("birth_date") or "")
    time_in = _coerce_str(payload.get("TOB") or payload.get("time") or payload.get("birth_time") or "12:00")
    time_str = _pad_hms(time_in)

    method = _norm_method(payload.get("method", payload.get("mode", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    # Accept POB alias
    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload)
        payload["place"] = payload.get("POB")

    # STRICT geocoding if any place/POB
    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        norm = {
            "system": "vimshottari",
            "date": date, "time": time_str, "tz": tz_norm,
            "method": method, "ayanamsa": ayanamsa,
            "latitude": None, "longitude": None, "elevation_m": None,
            "coordinate_mode": "geocentric", "topocentric": False,
            "jd_tt": None, "jd_ut1": None,
            "fatal": fatal, "raw": payload,
            "tz_name": tz_norm, "ayanamsa_key": ayanamsa,
            "birth_date": date, "birth_time": time_str, "place_tz": tz_norm,
        }
        return norm, warns + ["fatal"], tz_norm or "UTC"

    # adopt resolved values when available
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

    # Timescales (Vim only)
    jd_tt: Optional[float] = _as_float(payload.get("jd_tt"))
    jd_ut1: Optional[float] = _as_float(payload.get("jd_ut1"))
    if _TIMESCALES_OK and (jd_tt is None or jd_ut1 is None) and date and tz_norm:
        try:
            ts = _call_build_timescales(date, time_str, tz_norm)
            if isinstance(ts, dict):
                if jd_tt is None and ts.get("jd_tt") is not None: jd_tt = float(ts["jd_tt"])
                if jd_ut1 is None and ts.get("jd_ut1") is not None: jd_ut1 = float(ts["jd_ut1"])
            else:
                if jd_tt is None: jd_tt = float(getattr(ts, "jd_tt"))
                if jd_ut1 is None: jd_ut1 = float(getattr(ts, "jd_ut1"))
        except Exception as e:
            warns.append(f"timescales_failed:{e!s}")

    vargas, varga_warns = _collect_vargas(payload)
    warns.extend(varga_warns)
    include_vargas = bool(payload.get("include_vargas") or vargas)
    varga_mode = _norm_method(payload.get("varga_zodiac_mode"), default="sidereal")
    varga_ayan = payload.get("varga_ayanamsa", ayanamsa)
    varga_request = None
    if include_vargas and vargas:
        varga_request = {"enabled": True, "vargas": vargas, "zodiac_mode": varga_mode, "ayanamsa": varga_ayan}

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


# ============================ Normalizers: Yoga ============================

def normalize_yoga_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []

    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))
    house_system = _coerce_str(payload.get("house_system") or "placidus").strip() or "placidus"

    date = _coerce_str(payload.get("DOB") or payload.get("date") or payload.get("birth_date") or "")
    time_str = _pad_hms(_coerce_str(payload.get("TOB") or payload.get("time") or payload.get("birth_time") or "12:00"))

    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    # POB alias
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


# ============================ Normalizers: Gochar / Ingress / Stations ============================

_DEFAULT_MOVERS = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
_DEFAULT_TARGETS = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Ascendant", "Descendant", "MC", "IC"]

def normalize_gochar_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []

    frame = _coerce_str(payload.get("frame") or "ecliptic-of-date").strip() or "ecliptic-of-date"
    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))
    include_nodes = _norm_bool(payload.get("include_nodes"), False)
    treat_nodes_like_saturn = _norm_bool(payload.get("treat_nodes_like_saturn"), False)
    step_minutes: Union[str, float, int] = _coerce_step_minutes(payload.get("step_minutes", "auto"))

    movers = _collect_bodies(payload, key_order=("movers", "bodies", "transiting_bodies"), default=_DEFAULT_MOVERS)
    if include_nodes:
        if "Rahu" not in movers: movers.append("Rahu")
        if "Ketu" not in movers: movers.append("Ketu")
    natal_targets = _collect_targets(payload, key_order=("targets", "natal_targets"), default=_DEFAULT_TARGETS)
    orb_deg, orb_map = _collect_orb_map(payload, base_orb=12.0)

    natal_in = payload.get("natal") or payload

    date = _coerce_str(natal_in.get("DOB") or natal_in.get("date") or natal_in.get("birth_date") or "")
    time_str = _pad_hms(_coerce_str(natal_in.get("TOB") or natal_in.get("time") or natal_in.get("birth_time") or "12:00"))
    tz_norm = _coerce_str(natal_in.get("tz") or natal_in.get("place_tz")).strip() or None

    lat = _as_float(natal_in.get("latitude") or natal_in.get("lat"))
    lon = _as_float(natal_in.get("longitude") or natal_in.get("lon"))
    elevation_m = _as_float(natal_in.get("elevation_m") or natal_in.get("elevation"))

    if not any([natal_in.get("place"), natal_in.get("birth_place"), natal_in.get("place_city"),
                natal_in.get("place_state"), natal_in.get("place_country")]) and natal_in.get("POB"):
        natal_in = dict(natal_in); natal_in["place"] = natal_in.get("POB")

    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(natal_in, warns=warns)
    if fatal:
        norm = {
            "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa,
            "include_nodes": bool(include_nodes), "treat_nodes_like_saturn": bool(treat_nodes_like_saturn),
            "step_minutes": step_minutes,
            "observer": "geocentric", "topocentric": False,
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
            "time_range": None,
            "fatal": fatal,
            "raw": payload,
            "transiting_bodies": movers, "ayanamsa_key": ayanamsa,
            "tz_name": tz_norm,
        }
        d0, d1, _ = _collect_time_window(payload)
        if d0 and d1: norm["time_range"] = [d0, d1]
        return norm, warns + ["fatal"], tz_norm or "UTC"

    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    date_from, date_to, win_warns = _collect_time_window(payload); warns.extend(win_warns)

    longitudes = natal_in.get("longitudes") or natal_in.get("ecliptic_longitudes") or None
    natal_jd_tt = _as_float(natal_in.get("natal_jd_tt") or natal_in.get("jd_tt"))
    natal_jd_utc = _as_float(natal_in.get("jd_utc"))

    # Observer handling (consistent with other normalizers)
    observer = _norm_observer(payload.get("observer"), "geocentric")
    topocentric = (observer == "topocentric")
    if topocentric and (lat is None or lon is None):
        observer = "geocentric"
        topocentric = False
        warns.append("topocentric_requires_coordinates:fallback_geocentric")

    # angle targets auto-prune if we cannot bind angles
    if any(t in ("Ascendant", "Descendant", "MC", "IC") for t in natal_targets):
        have_angles_map = isinstance(longitudes, dict) and any(k in longitudes for k in ("Ascendant", "Descendant", "MC", "IC"))
        have_natal_jd = isinstance(natal_jd_tt, (int, float)) or isinstance(natal_jd_utc, (int, float))
        if not (have_angles_map or have_natal_jd):
            natal_targets = [t for t in natal_targets if t not in ("Ascendant", "Descendant", "MC", "IC")]
            warns.append("gochar_angles_pruned_unbound")

    norm: Dict[str, Any] = {
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa": ayanamsa,
        "observer": observer,
        "topocentric": bool(topocentric),
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


def normalize_ingress_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []
    frame = _coerce_str(payload.get("frame") or "ecliptic-of-date").strip() or "ecliptic-of-date"
    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload); payload["place"] = payload.get("POB")

    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        norm = {
            "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa,
            "movers": [], "time_range": None, "step_minutes": _coerce_step_minutes(payload.get("step_minutes", "auto")),
            "topocentric": False, "latitude": None, "longitude": None, "elevation_m": None,
            "tz": tz_norm, "tz_name": tz_norm, "fatal": fatal, "raw": payload,
        }
        d0, d1, _ = _collect_time_window(payload)
        if d0 and d1: norm["time_range"] = [d0, d1]
        return norm, warns + ["fatal"], tz_norm or "UTC"

    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    movers = _collect_bodies(payload, key_order=("movers", "bodies", "transiting_bodies"),
                             default=["Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"])
    if _norm_bool(payload.get("include_moon"), True) and "Moon" not in movers:
        movers.insert(0, "Moon")

    date_from, date_to, win_warns = _collect_time_window(payload); warns.extend(win_warns)

    topocentric = _norm_observer(payload.get("observer"), "geocentric") == "topocentric"
    if topocentric and (lat is None or lon is None):
        topocentric = False; warns.append("topocentric_requires_coordinates:fallback_geocentric")

    norm = {
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa": ayanamsa,
        "movers": movers,
        "time_range": [date_from, date_to] if (date_from and date_to) else None,
        "step_minutes": _coerce_step_minutes(payload.get("step_minutes", "auto")),
        "topocentric": bool(topocentric),
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation_m,
        "tz": tz_norm,
        "tz_name": tz_norm,
        "raw": payload,
    }
    if not (date_from and date_to): warns.append("ingress_missing_window")
    return norm, warns, tz_norm or "UTC"


def normalize_stations_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []
    frame = _coerce_str(payload.get("frame") or "ecliptic-of-date").strip() or "ecliptic-of-date"
    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload); payload["place"] = payload.get("POB")

    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        norm = {
            "frame": frame, "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa,
            "movers": [], "time_range": None, "step_minutes": _coerce_step_minutes(payload.get("step_minutes", "auto")),
            "topocentric": False, "latitude": None, "longitude": None, "elevation_m": None,
            "tz": tz_norm, "tz_name": tz_norm, "fatal": fatal, "raw": payload,
        }
        d0, d1, _ = _collect_time_window(payload)
        if d0 and d1: norm["time_range"] = [d0, d1]
        return norm, warns + ["fatal"], tz_norm or "UTC"

    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    movers = _collect_bodies(payload, key_order=("movers", "bodies", "transiting_bodies"),
                             default=["Mercury", "Venus", "Mars", "Jupiter", "Saturn"])
    date_from, date_to, win_warns = _collect_time_window(payload); warns.extend(win_warns)

    topocentric = _norm_observer(payload.get("observer"), "geocentric") == "topocentric"
    if topocentric and (lat is None or lon is None):
        topocentric = False; warns.append("topocentric_requires_coordinates:fallback_geocentric")

    norm = {
        "frame": frame,
        "zodiac_mode": zodiac_mode,
        "ayanamsa": ayanamsa,
        "movers": movers,
        "time_range": [date_from, date_to] if (date_from and date_to) else None,
        "step_minutes": _coerce_step_minutes(payload.get("step_minutes", "auto")),
        "topocentric": bool(topocentric),
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation_m,
        "tz": tz_norm,
        "tz_name": tz_norm,
        "raw": payload,
    }
    if not (date_from and date_to): warns.append("stations_missing_window")
    return norm, warns, tz_norm or "UTC"


# ============================ Normalizers: Śaḍbala ============================

def normalize_shadbala_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Śaḍbala computation (core chart).
    """
    warns: List[str] = []

    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))
    house_system = _coerce_str(payload.get("house_system") or "placidus").strip() or "placidus"

    date = _coerce_str(payload.get("DOB") or payload.get("date") or payload.get("birth_date") or "")
    time_str = _pad_hms(_coerce_str(payload.get("TOB") or payload.get("time") or payload.get("birth_time") or "12:00"))

    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload); payload["place"] = payload.get("POB")

    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        norm = {
            "system": "shadbala",
            "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa, "house_system": house_system,
            "date": date or None, "time": time_str if date else None, "tz": tz_norm,
            "latitude": None, "longitude": None, "elevation_m": None,
            "coordinate_mode": "geocentric", "topocentric": False,
            "prefer_houses_advanced": bool(payload.get("prefer_houses_advanced", True)),
            "include_velocity": bool(payload.get("include_velocity", True)),
            "fatal": fatal, "raw": payload,
            "place_tz": tz_norm, "tz_name": tz_norm,
            "place_name": _coerce_str(payload.get("place") or payload.get("birth_place") or payload.get("POB") or "").strip() or None,
        }
        return norm, warns + ["fatal"], tz_norm or "UTC"

    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    coordinate_mode = _norm_observer(payload.get("observer"), "geocentric")
    topocentric = (coordinate_mode == "topocentric")
    if topocentric and (lat is None or lon is None):
        coordinate_mode = "geocentric"; topocentric = False
        warns.append("topocentric_requires_coordinates:fallback_geocentric")

    prefer_houses_advanced = bool(payload.get("prefer_houses_advanced", True))
    if prefer_houses_advanced and (lat is None or lon is None or not date or not tz_norm):
        warns.append("shadbala_kala_dig_degraded:missing_time_or_coordinates")

    norm: Dict[str, Any] = {
        "system": "shadbala",
        "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa, "house_system": house_system,
        "date": date or None, "time": time_str if date else None, "tz": tz_norm,
        "latitude": lat, "longitude": lon, "elevation_m": elevation_m,
        "coordinate_mode": coordinate_mode, "topocentric": bool(topocentric),
        "include_velocity": bool(payload.get("include_velocity", True)),
        "prefer_houses_advanced": prefer_houses_advanced,
        "place_tz": tz_norm, "tz_name": tz_norm,
        "place_name": _coerce_str(payload.get("place") or payload.get("birth_place") or payload.get("POB") or "").strip() or None,
        "raw": payload,
    }

    if not date: warns.append("missing_date")
    if not time_str: warns.append("missing_time")
    return norm, warns, tz_norm or "UTC"


# ============================ Normalizers: Aṣṭakavarga ============================

def normalize_ashtakavarga_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Aṣṭakavarga (core chart):
      • Outputs minimal core payload + pass-through of ruleset and ruleset_map.
      • Accepts optional `angles` {asc, mc}.
    """
    warns: List[str] = []

    zodiac_mode = _norm_method(payload.get("zodiac_mode", payload.get("method", "sidereal")), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))
    house_system = _coerce_str(payload.get("house_system") or "placidus").strip() or "placidus"

    date = _coerce_str(payload.get("DOB") or payload.get("date") or payload.get("birth_date") or "")
    time_str = _pad_hms(_coerce_str(payload.get("TOB") or payload.get("time") or payload.get("birth_time") or "12:00"))

    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    # Allow angles pass-through if provided
    angles_in = payload.get("angles")
    angles: Optional[Dict[str, Any]] = None
    if isinstance(angles_in, dict):
        asc = _as_float(angles_in.get("asc") or angles_in.get("ASC") or angles_in.get("Ascendant"))
        mc = _as_float(angles_in.get("mc") or angles_in.get("MC") or angles_in.get("Midheaven"))
        angles = {}
        if asc is not None: angles["asc"] = asc
        if mc is not None: angles["mc"] = mc
        if not angles: angles = None

    # POB alias
    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload); payload["place"] = payload.get("POB")

    # STRICT geocoding if any place/POB present
    lat_r, lon_r, elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        norm = {
            "system": "ashtakavarga",
            "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa, "house_system": house_system,
            "date": date or None, "time": time_str if date else None, "tz": tz_norm,
            "latitude": None, "longitude": None, "elevation_m": None,
            "coordinate_mode": "geocentric", "topocentric": False,
            "fatal": fatal, "raw": payload,
            "place_tz": tz_norm, "tz_name": tz_norm,
            "place_name": _coerce_str(payload.get("place") or payload.get("birth_place") or payload.get("POB") or "").strip() or None,
            "ruleset": _norm_ruleset_name(payload.get("ruleset")),
            "ruleset_map": None,
            "angles": angles,
            "spec_path": _coerce_str(payload.get("spec") or payload.get("spec_path") or ""),
        }
        return norm, warns + ["fatal"], tz_norm or "UTC"

    # adopt resolved when present
    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if elevation_m is None and elev_r is not None:
        elevation_m = elev_r
    if tz_r:
        tz_norm = tz_r

    # Ashtakavarga needs Lagna — warn if coords absent
    if lat is None or lon is None:
        warns.append("ashtakavarga_requires_coordinates_for_lagna")

    ruleset = _norm_ruleset_name(payload.get("ruleset"))
    ruleset_map_norm: Optional[Dict[str, Dict[str, List[int]]]] = None
    if ruleset == "custom":
        ruleset_map_norm, w = _validate_ruleset_map(payload.get("ruleset_map"))
        warns.extend(w)
        if ruleset_map_norm is None:
            # fall back to canonical rules
            ruleset = "parashari-bphs"
            warns.append("ruleset_custom_invalid:fallback_to_parashari_bphs")

    norm: Dict[str, Any] = {
        "system": "ashtakavarga",
        "zodiac_mode": zodiac_mode, "ayanamsa": ayanamsa, "house_system": house_system,
        "date": date or None, "time": time_str if date else None, "tz": tz_norm,
        "latitude": lat, "longitude": lon, "elevation_m": elevation_m,
        "coordinate_mode": "geocentric", "topocentric": False,
        "place_tz": tz_norm, "tz_name": tz_norm,
        "place_name": _coerce_str(payload.get("place") or payload.get("birth_place") or payload.get("POB") or "").strip() or None,
        "angles": angles,
        "ruleset": ruleset,
        "ruleset_map": ruleset_map_norm,
        "spec_path": _coerce_str(payload.get("spec") or payload.get("spec_path") or ""),
        "raw": payload,
    }

    if not date: warns.append("missing_date")
    if not time_str: warns.append("missing_time")
    if not tz_norm: warns.append("missing_timezone")
    return norm, warns, tz_norm or "UTC"


# ============================ Normalizers: Horary / Prasna ============================

class _LocalQuestionType(Enum):
    JOB = "job"
    MARRIAGE = "marriage"
    LITIGATION = "litigation"
    HEALTH = "health"
    LOST_ITEM = "lost_item"
    PROPERTY = "property"
    FOREIGN = "foreign"
    EDUCATION = "education"
    CHILDREN = "children"
    BUSINESS = "business"

_QSTR_ALIASES: Dict[str, _LocalQuestionType] = {
    "job": _LocalQuestionType.JOB,
    "career": _LocalQuestionType.JOB,
    "offer": _LocalQuestionType.JOB,
    "promotion": _LocalQuestionType.JOB,
    "marriage": _LocalQuestionType.MARRIAGE,
    "relationship": _LocalQuestionType.MARRIAGE,
    "dating": _LocalQuestionType.MARRIAGE,
    "litigation": _LocalQuestionType.LITIGATION,
    "lawsuit": _LocalQuestionType.LITIGATION,
    "court": _LocalQuestionType.LITIGATION,
    "health": _LocalQuestionType.HEALTH,
    "illness": _LocalQuestionType.HEALTH,
    "lost_item": _LocalQuestionType.LOST_ITEM,
    "lost": _LocalQuestionType.LOST_ITEM,
    "missing": _LocalQuestionType.LOST_ITEM,
    "property": _LocalQuestionType.PROPERTY,
    "real_estate": _LocalQuestionType.PROPERTY,
    "foreign": _LocalQuestionType.FOREIGN,
    "travel": _LocalQuestionType.FOREIGN,
    "abroad": _LocalQuestionType.FOREIGN,
    "education": _LocalQuestionType.EDUCATION,
    "study": _LocalQuestionType.EDUCATION,
    "children": _LocalQuestionType.CHILDREN,
    "child": _LocalQuestionType.CHILDREN,
    "business": _LocalQuestionType.BUSINESS,
    "partnership": _LocalQuestionType.BUSINESS,
}

def _norm_question_type(v: Any) -> str:
    """
    Normalize question_type string to match app.core.horary.QuestionType value.
    Returns the lowercase value string.
    """
    if isinstance(v, _LocalQuestionType):
        return v.value
    s = _coerce_str(v).strip().lower()
    if not s:
        return _LocalQuestionType.JOB.value
    if s in _QSTR_ALIASES:
        return _QSTR_ALIASES[s].value
    # Already one of the canonical names?
    for qt in _LocalQuestionType:
        if s == qt.value:
            return qt.value
    return _LocalQuestionType.JOB.value

def _as_core_qtype(value_str: str):
    """
    Try to convert normalized question-type string into the real Enum
    app.core.horary.QuestionType if available; otherwise return None.
    """
    if not _HORARY_ENUM_OK:
        return None
    try:
        # Enum members are defined with .value matching our normalized string
        for m in _QuestionType:  # type: ignore
            if str(m.value).lower() == value_str:
                return m
    except Exception:
        pass
    return None

def _norm_horary_method(v: Any) -> str:
    s = _coerce_str(v).strip().lower()
    if s in ("kp", "krishnamurti", "kp_horary", "kph"):
        return "kp"
    if s in ("parashari", "parasari", "parāśari", "parashari_prasna", "prasna", "prashna"):
        return "parashari"
    if s in ("hybrid", "mix", "blended", "combined", "prasna_hybrid", "horary_hybrid"):
        return "hybrid"
    # default: parashari (classical)
    return "parashari"

def normalize_horary_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Horary / Prasna endpoints.
    Returns a dict that maps directly to:
      - app.core.horary.HoraryInput fields when method="parashari" | "kp"
      - app.core.horary.HybridPrasnaInput fields when method="hybrid"

    Accepted keys (superset):
      method|mode: "parashari" | "kp" | "hybrid"

      # Question moment (parashari/kp use date/time/tz + place/lat/lon; hybrid uses question_*):
      date, time, tz, place / (place_city/state/country) / POB, latitude, longitude
      question_date, question_time, question_tz, question_place / *_city/state/country,
      question_latitude, question_longitude

      # Birth data (only used for hybrid; aliases DOB/TOB/POB):
      birth_date|DOB, birth_time|TOB, birth_tz|tz_name, birth_place|POB,
      birth_latitude|latitude, birth_longitude|longitude,
      birth_zodiac_mode, birth_ayanamsa

      # Shared astro settings:
      zodiac_mode (default sidereal), ayanamsa (default lahiri), house_system (default sripati)

      # KP options (for method="kp"):
      kp_house_system (default placidus), kp_ayanamsa (default krishnamurti),
      kp_number (1..249), kp_number_mode ("anchor_asc" | "advisory")

      # Question typing:
      question | question_type | topic (free text & type)
    """
    warns: List[str] = []

    method = _norm_horary_method(payload.get("method") or payload.get("mode") or "parashari")

    # Common chart basics
    zodiac_mode = _norm_method(payload.get("zodiac_mode", "sidereal"), default="sidereal")
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key") or "lahiri")
    house_system = _coerce_str(payload.get("house_system") or "sripati").strip() or "sripati"

    # Question type / text
    qtype_val = _norm_question_type(payload.get("question_type") or payload.get("question") or payload.get("topic"))
    qtype_enum = _as_core_qtype(qtype_val)
    qtext = _coerce_str(payload.get("question_text") or payload.get("query") or payload.get("topic") or "").strip() or None

    # ---------- HYBRID BRANCH ----------
    if method == "hybrid":
        # Question moment
        q_date = _coerce_str(payload.get("question_date") or payload.get("date") or "").strip()
        q_time = _pad_hms(_coerce_str(payload.get("question_time") or payload.get("time") or "").strip())
        q_tz   = _coerce_str(payload.get("question_tz") or payload.get("tz") or payload.get("place_tz")).strip() or None

        # Question location (strict geocoding if any place string present)
        # Allow: question_place / place / question_place_city|state|country
        if not any([payload.get("question_place"), payload.get("question_place_city"),
                    payload.get("question_place_state"), payload.get("question_place_country")]) and payload.get("place"):
            payload = dict(payload)
            payload["question_place"] = payload.get("place")

        q_lat = _as_float(payload.get("question_latitude") or payload.get("latitude") or payload.get("lat"))
        q_lon = _as_float(payload.get("question_longitude") or payload.get("longitude") or payload.get("lon"))

        q_lat_r, q_lon_r, _q_elev_r, q_tz_r, q_fatal = _must_resolve_place_if_provided(payload, place_prefix="question_", warns=warns)
        if q_fatal:
            norm = {
                "system": "horary",
                "method": "hybrid",
                "question_date": q_date or None,
                "question_time": q_time or None,
                "question_tz": q_tz,
                "zodiac_mode": zodiac_mode,
                "ayanamsa": ayanamsa,
                "house_system": house_system,
                "question_type": qtype_val,
                "question_type_enum": qtype_enum,
                "question_text": qtext,
                "question_latitude": None, "question_longitude": None,
                "fatal": q_fatal,
                "raw": payload,
            }
            return norm, warns + ["fatal"], q_tz or "UTC"

        if q_lat_r is not None and q_lon_r is not None:
            q_lat, q_lon = q_lat_r, q_lon_r
        if q_tz_r:
            q_tz = q_tz_r

        if q_lat is None or q_lon is None:
            warns.append("missing_question_coordinates")
            norm = {
                "system": "horary",
                "method": "hybrid",
                "question_date": q_date or None,
                "question_time": q_time or None,
                "question_tz": q_tz,
                "zodiac_mode": zodiac_mode,
                "ayanamsa": ayanamsa,
                "house_system": house_system,
                "question_type": qtype_val,
                "question_type_enum": qtype_enum,
                "question_text": qtext,
                "question_latitude": None, "question_longitude": None,
                "fatal": "missing_location",
                "raw": payload,
            }
            return norm, warns + ["fatal"], q_tz or "UTC"

        # Querent birth data (with common aliases)
        b_date = _coerce_str(payload.get("birth_date") or payload.get("DOB") or "").strip()
        b_time = _pad_hms(_coerce_str(payload.get("birth_time") or payload.get("TOB") or "").strip())
        b_tz   = _coerce_str(payload.get("birth_tz") or payload.get("tz_name") or payload.get("birth_place_tz")).strip() or None

        # Use separate birth_* coords/place if provided; else allow aliases
        if not any([payload.get("birth_place"), payload.get("birth_place_city"),
                    payload.get("birth_place_state"), payload.get("birth_place_country")]) and payload.get("POB"):
            payload = dict(payload)
            payload["birth_place"] = payload.get("POB")

        b_place_to_resolve = any([
            payload.get("birth_place"), payload.get("birth_place_city"),
            payload.get("birth_place_state"), payload.get("birth_place_country"), payload.get("POB")
        ])

        b_lat = _as_float(payload.get("birth_latitude"))
        b_lon = _as_float(payload.get("birth_longitude"))

        if b_place_to_resolve:
            b_lat_r, b_lon_r, _b_elev_r, b_tz_r, b_fatal = _must_resolve_place_if_provided(payload, place_prefix="birth_", warns=warns)
            if b_fatal:
                # Hybrid can still continue if birth cannot be resolved, but mark fatal to prevent core call
                norm = {
                    "system": "horary",
                    "method": "hybrid",
                    "question_date": q_date or None,
                    "question_time": q_time or None,
                    "question_tz": q_tz,
                    "zodiac_mode": zodiac_mode,
                    "ayanamsa": ayanamsa,
                    "house_system": house_system,
                    "question_type": qtype_val,
                    "question_type_enum": qtype_enum,
                    "question_text": qtext,
                    "question_latitude": q_lat, "question_longitude": q_lon,
                    "fatal": b_fatal,
                    "raw": payload,
                }
                return norm, warns + ["fatal"], q_tz or "UTC"
            if b_lat_r is not None and b_lon_r is not None:
                b_lat, b_lon = b_lat_r, b_lon_r
            if b_tz_r:
                b_tz = b_tz_r

        # Birth zodiac/ayana (allow specific overrides)
        b_zodiac = _norm_method(payload.get("birth_zodiac_mode", payload.get("zodiac_mode")), default="sidereal")
        b_ayan   = _norm_ayanamsa(payload.get("birth_ayanamsa", payload.get("ayanamsa")) or "lahiri")

        norm = {
            "system": "horary",
            "method": "hybrid",
            # question moment as HybridPrasnaInput expects:
            "question_date": q_date or None,
            "question_time": q_time or None,
            "question_tz": q_tz,
            "question_place": _coerce_str(payload.get("question_place") or payload.get("place") or "").strip() or None,
            "question_latitude": q_lat,
            "question_longitude": q_lon,
            # shared settings:
            "zodiac_mode": zodiac_mode,
            "ayanamsa": ayanamsa,
            "house_system": house_system,
            # question typing:
            "question_type": qtype_val,
            "question_type_enum": qtype_enum,
            "question_text": qtext,
            # nested querent birth object (as Horary.HybridPrasnaInput expects):
            "querent_birth": {
                "date": b_date or None,
                "time": b_time or None,
                "tz_name": b_tz,
                "place": _coerce_str(payload.get("birth_place") or payload.get("POB") or "").strip() or None,
                "latitude": b_lat,
                "longitude": b_lon,
                "zodiac_mode": b_zodiac,
                "ayanamsa": b_ayan,
            },
            "raw": payload,
        }

        if not q_date: warns.append("missing_question_date")
        if not q_time: warns.append("missing_question_time")
        if not q_tz: warns.append("missing_question_timezone")
        # Birth data is optional but recommended
        if not b_date: warns.append("missing_birth_date")
        if not b_time: warns.append("missing_birth_time")
        if not b_tz: warns.append("missing_birth_timezone")

        return norm, warns, q_tz or "UTC"

    # ---------- PARASHARI / KP BRANCH ----------
    # KP options
    kp_house_system = _coerce_str(payload.get("kp_house_system") or "placidus").strip() or "placidus"
    kp_ayanamsa = _coerce_str(payload.get("kp_ayanamsa") or "krishnamurti").strip() or "krishnamurti"
    kp_number = None
    if payload.get("kp_number") is not None:
        try:
            n = int(payload.get("kp_number"))
            if 1 <= n <= 249:
                kp_number = n
            else:
                warns.append("kp_number_out_of_range")
        except Exception:
            warns.append("kp_number_invalid")
    kp_number_mode = _coerce_str(payload.get("kp_number_mode") or "anchor_asc").strip().lower()
    if kp_number_mode not in ("anchor_asc", "advisory"):
        kp_number_mode = "anchor_asc"

    # Time & TZ (question moment)
    date = _coerce_str(payload.get("date")).strip() or _coerce_str(payload.get("DOB")).strip()
    time_str = _pad_hms(_coerce_str(payload.get("time") or payload.get("TOB") or ""))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz")).strip() or None

    # Coordinates / place resolution for question moment
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))

    # POB alias for question place
    if not any([payload.get("place"), payload.get("birth_place"), payload.get("place_city"),
                payload.get("place_state"), payload.get("place_country")]) and payload.get("POB"):
        payload = dict(payload); payload["place"] = payload.get("POB")

    lat_r, lon_r, _elev_r, tz_r, fatal = _must_resolve_place_if_provided(payload, warns=warns)
    if fatal:
        # For horary, missing location is fatal — need tz & coordinates for houses.
        norm = {
            "system": "horary",
            "method": method,
            "date": date or None,
            "time": time_str or None,
            "tz": tz_norm,
            "zodiac_mode": zodiac_mode,
            "ayanamsa": ayanamsa,
            "house_system": house_system,
            "kp_house_system": kp_house_system,
            "kp_ayanamsa": kp_ayanamsa,
            "kp_number": kp_number,
            "kp_number_mode": kp_number_mode,
            "question_type": qtype_val,
            "question_type_enum": qtype_enum,
            "question_text": qtext,
            "latitude": None, "longitude": None,
            "fatal": fatal,
            "raw": payload,
        }
        return norm, warns + ["fatal"], tz_norm or "UTC"

    if lat_r is not None and lon_r is not None:
        lat, lon = lat_r, lon_r
    if tz_r:
        tz_norm = tz_r

    # Without any place/coords at all, flag fatal — chart/house calc won't be reliable.
    if lat is None or lon is None:
        warns.append("missing_coordinates")
        norm = {
            "system": "horary",
            "method": method,
            "date": date or None,
            "time": time_str or None,
            "tz": tz_norm,
            "zodiac_mode": zodiac_mode,
            "ayanamsa": ayanamsa,
            "house_system": house_system,
            "kp_house_system": kp_house_system,
            "kp_ayanamsa": kp_ayanamsa,
            "kp_number": kp_number,
            "kp_number_mode": kp_number_mode,
            "question_type": qtype_val,
            "question_type_enum": qtype_enum,
            "question_text": qtext,
            "latitude": None, "longitude": None,
            "fatal": "missing_location",
            "raw": payload,
        }
        return norm, warns + ["fatal"], tz_norm or "UTC"

    norm: Dict[str, Any] = {
        "system": "horary",
        "method": method,  # "parashari" | "kp"
        "date": date or None,
        "time": time_str or None,
        "tz": tz_norm,
        "place": _coerce_str(payload.get("place") or payload.get("birth_place") or "").strip() or None,
        "latitude": lat,
        "longitude": lon,
        "zodiac_mode": zodiac_mode,
        "ayanamsa": ayanamsa,
        "house_system": house_system,
        "kp_house_system": kp_house_system,
        "kp_ayanamsa": kp_ayanamsa,
        "kp_number": kp_number,
        "kp_number_mode": kp_number_mode,
        "question_type": qtype_val,         # string value (stable for APIs)
        "question_type_enum": qtype_enum,   # real Enum instance if available
        "question_text": qtext,
        "querent_house": 1,
        "quesited_house": _as_float(payload.get("quesited_house")) and int(float(payload.get("quesited_house"))) or None,
        "raw": payload,
    }

    # Warnings for missing time/tz — the core will fallback to now/UTC if absent
    if not date: warns.append("missing_date")
    if not time_str: warns.append("missing_time")
    if not tz_norm: warns.append("missing_timezone")

    return norm, warns, tz_norm or "UTC"


__all__ = [
    "normalize_vim_payload",
    "normalize_yoga_payload",
    "normalize_gochar_payload",
    "normalize_ingress_payload",
    "normalize_stations_payload",
    "normalize_shadbala_payload",
    "normalize_ashtakavarga_payload",
    "normalize_horary_payload",
]
