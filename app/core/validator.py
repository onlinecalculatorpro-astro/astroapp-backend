# app/core/validator.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Central validator wiring for AstroApp.

Responsibilities
- Common normalization used by ALL routes (central, Vedic, Western).
- Preferred timescale forwarder to time_kernel.build_timescales (dict),
  with a safe fallback to timescales.build_timescales (dataclass).
- Optional, pluggable domain-specific normalizers:
    • app.core.vedic_validator
    • app.core.western_validator
  If present, their exported normalizer functions are used; otherwise
  we fall back to the common base normalization in this module.

For astronomy.compute_chart (FINAL engine):
- normalize_chart_payload(payload, ...) -> (normalized: dict, warnings: list[str], tz_normalized: str)

For houses (policy façade):
- normalize_house_system(name) -> (canonical_public_label|None, warnings)
- normalize_houses_payload(payload, ...) -> (normalized: dict, warnings: list[str], tz_normalized: str)

Stable Exports
- normalize_tz(tz, default="UTC") -> str
- env_dut1_seconds() -> float
- clamp_levels(val, default=5, min_v=1, max_v=5) -> int
- parse_latlon(payload) -> (lat, lon)
- sanitize_timescales(ts, include_jd_utc=False) -> dict
- normalize_common_payload(payload, *, default_time="12:00:00",
                           compute_timescales=True, include_jd_utc=False,
                           dut1_seconds=None, max_levels=5)
    -> (normalized: dict, warnings: list[str], tz_normalized: str)
- normalize_timescales_input(...) -> alias of normalize_common_payload with defaults
- normalize_body(payload|name) -> (canonical_body_or_None, warnings)
- normalize_for_vedic(payload, **opts) -> (dict, warnings, tz)
- normalize_for_western(payload, **opts) -> (dict, warnings, tz)
- normalize_for_domain(domain, payload, **opts) -> (dict, warnings, tz)
- normalize_house_system(name) -> (canonical|None, warnings)
- normalize_houses_payload(payload, **opts) -> (dict, warnings, tz)
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import os

# ───────────────────────── optional domain normalizers ─────────────────────────
def _discover_normalizer(mod_name: str) -> Optional[Callable[..., Tuple[Dict[str, Any], List[str], str]]]:
    try:
        mod = __import__(mod_name, fromlist=["*"])
    except Exception:
        return None
    for attr in ("normalize_payload", "normalize", "normalize_common_payload"):
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn  # type: ignore[return-value]
    return None

_VEDIC_NORMALIZER = _discover_normalizer("app.core.vedic_validator")
_WESTERN_NORMALIZER = _discover_normalizer("app.core.western_validator")

# ───────────────────────── prefer the time_kernel forwarder ─────────────────────────
try:
    from app.core.time_kernel import build_timescales as tk_build_timescales  # type: ignore
    _TK_AVAILABLE = True
except Exception:
    tk_build_timescales = None  # type: ignore
    _TK_AVAILABLE = False

# canonical engine fallback (dataclass TimeScales)
try:
    from app.core.timescales import build_timescales as ts_build_timescales, TimeScales  # type: ignore
    _TS_AVAILABLE = True
except Exception:
    ts_build_timescales = None  # type: ignore
    TimeScales = None           # type: ignore
    _TS_AVAILABLE = False

# Houses façade discovery (for normalization only)
try:
    from app.core.house import canonicalize_system as _house_canonicalize  # type: ignore
    from app.core.house import list_supported_house_systems as _house_list  # type: ignore
    _HOUSE_AVAILABLE = True
except Exception:
    _house_canonicalize = None  # type: ignore
    _house_list = None          # type: ignore
    _HOUSE_AVAILABLE = False

__all__ = [
    "normalize_tz",
    "env_dut1_seconds",
    "clamp_levels",
    "parse_latlon",
    "sanitize_timescales",
    "normalize_common_payload",
    "normalize_timescales_input",
    "normalize_body",
    "normalize_for_vedic",
    "normalize_for_western",
    "normalize_for_domain",
    # FINAL astronomy wiring
    "normalize_chart_payload",
    # Houses wiring
    "normalize_house_system",
    "normalize_houses_payload",
]

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

_TZ_ALIAS = {
    "utc": "UTC",
    "gmt": "UTC",
    "ist": "Asia/Kolkata",
    "asia/calcutta": "Asia/Kolkata",
}

def normalize_tz(tz: Any, default: str = "UTC") -> str:
    if not isinstance(tz, str) or not tz.strip():
        return default
    key = tz.strip()
    alias = _TZ_ALIAS.get(key.lower())
    return alias or key

def _coerce_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
    return default

# ──────────────────────────────────────────────────────────────────────────────
# Generic coercions
# ──────────────────────────────────────────────────────────────────────────────

def clamp_levels(val: Any, *, default: int = 5, min_v: int = 1, max_v: int = 5) -> int:
    if isinstance(val, list):
        n = len(val)
        return max(min_v, min(max_v, n))
    try:
        i = int(val)
        return max(min_v, min(max_v, i))
    except Exception:
        return default

def parse_latlon(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat = payload.get("latitude")
    lon = payload.get("longitude")
    try:
        lat_f = float(lat) if isinstance(lat, (int, float, str)) and str(lat).strip() != "" else None
    except Exception:
        lat_f = None
    try:
        lon_f = float(lon) if isinstance(lon, (int, float, str)) and str(lon).strip() != "" else None
    except Exception:
        lon_f = None
    return lat_f, lon_f

# ──────────────────────────────────────────────────────────────────────────────
# Body helper (for ephemeris single-body ops)
# ──────────────────────────────────────────────────────────────────────────────

_BODY_ALIAS_MAP = {
    "sun": "sun", "sol": "sun",
    "moon": "moon", "luna": "moon",
    "mercury": "mercury",
    "venus": "venus",
    "earth": "earth",
    "mars": "mars",
    "jupiter": "jupiter",
    "saturn": "saturn",
    "uranus": "uranus",
    "neptune": "neptune",
    "pluto": "pluto",
}

def normalize_body(payload_or_name: Dict[str, Any] | str) -> Tuple[Optional[str], List[str]]:
    warnings: List[str] = []
    if isinstance(payload_or_name, dict):
        raw = payload_or_name.get("body")
    else:
        raw = payload_or_name
    if not isinstance(raw, str) or not raw.strip():
        return None, ["missing_body"]
    k = raw.strip().lower()
    if k in _BODY_ALIAS_MAP:
        return _BODY_ALIAS_MAP[k], warnings
    warnings.append(f"unknown_body:{raw}")
    return None, warnings

# ──────────────────────────────────────────────────────────────────────────────
# TimeScales helpers
# ──────────────────────────────────────────────────────────────────────────────

def env_dut1_seconds() -> float:
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0

def sanitize_timescales(ts: "TimeScales | Dict[str, Any]", *, include_jd_utc: bool = False) -> Dict[str, Any]:
    if isinstance(ts, dict):
        d = dict(ts)
    else:
        try:
            d = ts.to_dict()  # dataclass
        except Exception:
            d = getattr(ts, "__dict__", {}) or {}

    out = {
        "jd_tt": d.get("jd_tt"),
        "jd_ut1": d.get("jd_ut1"),
        "delta_t": d.get("delta_t"),
        "dut1": d.get("dut1"),
        "dat": d.get("dat"),
        "tz_offset_seconds": d.get("tz_offset_seconds"),
        "timezone": d.get("timezone"),
        "warnings": d.get("warnings", []),
        "precision": d.get("precision"),
    }
    if include_jd_utc:
        out["jd_utc"] = d.get("jd_utc")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Common payload normalization (civil + optional TimeScales)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_civil(payload: Dict[str, Any], *, default_time: str = "12:00:00") -> Tuple[str, str, str, List[str]]:
    warns: List[str] = []
    date = str(payload.get("date") or payload.get("birth_date") or "").strip()
    if not date:
        warns.append("missing_date")

    time_str = str(payload.get("time") or payload.get("birth_time") or default_time).strip()
    tz_name = normalize_tz(
        payload.get("tz")
        or payload.get("timezone")
        or payload.get("place_tz")
        or "UTC"
    )

    return date, time_str, tz_name, warns

def _compute_timescales_dict(
    date: str,
    time_str: str,
    tz_name: str,
    dut1: float,
    *,
    include_jd_utc: bool = False,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warns: List[str] = []
    if _TK_AVAILABLE and tk_build_timescales is not None:
        ts = tk_build_timescales(date, time_str, tz_name, dut1)  # type: ignore[misc]
        ts_dict = sanitize_timescales(ts, include_jd_utc=include_jd_utc)
        warns.extend(ts_dict.get("warnings", []) or [])
        return ts_dict, warns

    if _TS_AVAILABLE and ts_build_timescales is not None and date:
        ts = ts_build_timescales(date, time_str, tz_name, dut1)  # type: ignore[misc]
        ts_dict = sanitize_timescales(ts, include_jd_utc=include_jd_utc)
        warns.extend(ts_dict.get("warnings", []) or [])
        return ts_dict, warns

    warns.append("timescales_engine_unavailable")
    return None, warns

def normalize_common_payload(
    payload: Dict[str, Any],
    *,
    default_time: str = "12:00:00",
    compute_timescales: bool = True,
    include_jd_utc: bool = False,
    dut1_seconds: Optional[float] = None,
    max_levels: int = 5,
) -> Tuple[Dict[str, Any], List[str], str]:
    warns: List[str] = []
    date, time_str, tz_name, wz = _extract_civil(payload, default_time=default_time)
    warns.extend(wz)

    levels_in = payload.get("levels", payload.get("depth", payload.get("max_levels", max_levels)))
    levels = clamp_levels(levels_in, default=max_levels, min_v=1, max_v=max_levels)

    ayanamsa = str(payload.get("ayanamsa") or "lahiri").lower()

    lat_f, lon_f = parse_latlon(payload)

    norm: Dict[str, Any] = {
        "date": date,
        "time": time_str,
        "tz": tz_name,
        "tz_name": tz_name,
        "ayanamsa": ayanamsa,
        "ayanamsa_key": ayanamsa,
        "levels": levels,
        "depth": levels,
        "max_levels": levels,
        "latitude": lat_f,
        "longitude": lon_f,
        "raw": payload,
    }

    norm["birth_date"] = date
    norm["birth_time"] = time_str
    norm["place_tz"] = tz_name

    dut1 = env_dut1_seconds() if dut1_seconds is None else float(dut1_seconds)
    norm["dut1_seconds"] = dut1
    norm["dut1"] = dut1  # also expose as 'dut1' for consumers that expect this key

    if compute_timescales and date:
        try:
            ts_dict, wz_ts = _compute_timescales_dict(
                date, time_str, tz_name, dut1, include_jd_utc=include_jd_utc
            )
            warns.extend(wz_ts)
            if ts_dict:
                norm.update({
                    "jd_tt": ts_dict.get("jd_tt"),
                    "jd_ut1": ts_dict.get("jd_ut1"),
                    "timescales": ts_dict,
                })
                if include_jd_utc:
                    norm["jd_utc"] = ts_dict.get("jd_utc")
        except Exception as e:
            warns.append(f"timescales_failed:{e!s}")

    return norm, warns, tz_name

def normalize_timescales_input(
    payload: Dict[str, Any],
    *,
    default_time: str = "12:00:00",
    compute_timescales: bool = False,
    include_jd_utc: bool = False,
    dut1_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str], str]:
    return normalize_common_payload(
        payload,
        default_time=default_time,
        compute_timescales=compute_timescales,
        include_jd_utc=include_jd_utc,
        dut1_seconds=dut1_seconds,
        max_levels=5,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Astronomy chart normalization (for astronomy.compute_chart)
# ──────────────────────────────────────────────────────────────────────────────

_NODE_KEYS = {
    "north node", "south node", "node", "true node", "mean node",
    "truenode", "meannode", "ascending node", "descending node",
    "northnode", "southnode", "rahu", "ketu", "lunar node", "moon node", "☊", "☋",
}

def _coerce_elevation_m(payload: Dict[str, Any]) -> Optional[float]:
    for k in ("elevation_m", "elev_m", "elevation"):
        v = payload.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None

def _coerce_list_of_str(v: Any) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    # Allow single string "Sun,Moon" or "Sun Moon"
    if isinstance(v, str) and v.strip():
        if "," in v:
            return [s.strip() for s in v.split(",") if s.strip()]
        if " " in v:
            return [s for s in v.split() if s.strip()]
        return [v.strip()]
    return None

def normalize_chart_payload(
    payload: Dict[str, Any],
    *,
    default_time: str = "12:00:00",
    compute_timescales: bool = True,
    include_jd_utc: bool = True,  # make jd_utc available for astronomy.compute_chart
    dut1_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Prepare a request for app.core.astronomy.compute_chart (FINAL).

    - Normalizes civil fields + (optionally) precomputes timescales.
      (include_jd_utc=True by default so astronomy receives jd_ut/jd_utc + jd_tt + jd_ut1.)
    - Coerces mode ('tropical'|'sidereal') and frame (default 'ecliptic-of-date').
    - Coerces topocentric flag from 'topocentric' or 'center'=='topocentric'.
    - Extracts latitude/longitude and unifies elevation to 'elevation_m'.
    - Bodies/points coerced to list[str] if present (left as-is otherwise).
    - Always exposes both 'dut1_seconds' and 'dut1'.
    """
    base, warns, tz_name = normalize_common_payload(
        payload,
        default_time=default_time,
        compute_timescales=compute_timescales,
        include_jd_utc=include_jd_utc,
        dut1_seconds=dut1_seconds,
        max_levels=5,
    )

    # Mode / frame
    mode_raw = payload.get("mode", "tropical")
    mode = str(mode_raw).strip().lower()
    if mode not in ("tropical", "sidereal"):
        warns.append("invalid_mode")
        mode = "tropical"

    frame_raw = payload.get("frame")
    frame = (str(frame_raw).strip() if isinstance(frame_raw, str) and frame_raw.strip() else "ecliptic-of-date")

    # Center / topocentric
    center_raw = payload.get("center")
    topocentric = _coerce_bool(payload.get("topocentric"), False)
    if isinstance(center_raw, str) and center_raw.strip():
        c = center_raw.strip().lower()
        if c in ("topocentric", "geo", "geocentric"):
            topocentric = (c == "topocentric")

    # Observer
    lat, lon = base.get("latitude"), base.get("longitude")
    elev_m = _coerce_elevation_m(payload)

    # Normalize lists
    bodies = _coerce_list_of_str(payload.get("bodies"))
    points = _coerce_list_of_str(payload.get("points"))

    out: Dict[str, Any] = dict(base)
    out.update({
        "mode": mode,
        "frame": frame,
        "center": ("topocentric" if topocentric else "geocentric"),
        "topocentric": bool(topocentric),
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elev_m,
        "elev_m": elev_m,  # mirror, astronomy accepts several variants
        "bodies": bodies if bodies is not None else payload.get("bodies"),
        "points": points if points is not None else payload.get("points"),
        # expose DUT1 under both keys
        "dut1_seconds": base.get("dut1_seconds"),
        "dut1": base.get("dut1_seconds"),
    })

    # Observer convenience envelope if topo + coords present (adapter likes this)
    if topocentric and lat is not None and lon is not None:
        obs = {"latitude": float(lat), "longitude": float(lon)}
        if isinstance(elev_m, (int, float)):
            obs["elevation_m"] = float(elev_m)
        out["observer"] = obs

    return out, warns, tz_name

# ──────────────────────────────────────────────────────────────────────────────
# Houses normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_house_system(name: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """
    Canonicalize a user-facing house-system name to the public label used by `house.py`.

    Returns:
        (canonical_public_label|None, warnings)

    Behavior:
      - If `house.py` is available, uses its `canonicalize_system` (raises on unsupported).
      - If unavailable, returns a lowercased guess and warns.
      - On unsupported input, returns (None, ["unsupported_house_system:<name>"]).
      - If name is missing/blank, defaults to 'placidus'.
    """
    warns: List[str] = []
    if not isinstance(name, str) or not name.strip():
        return "placidus", warns

    if not _HOUSE_AVAILABLE or not callable(_house_canonicalize):
        # Best-effort fallback
        warns.append("house_module_unavailable")
        return name.strip().lower(), warns

    try:
        canon = _house_canonicalize(name)  # may raise ValueError with suggestions
        return canon, warns
    except Exception:
        warns.append(f"unsupported_house_system:{name}")
        return None, warns

def normalize_houses_payload(
    payload: Dict[str, Any],
    *,
    default_time: str = "12:00:00",
    compute_timescales: bool = True,   # houses façade requires jd_tt & jd_ut1 strictly
    include_jd_utc: bool = False,
    dut1_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Prepare a request for app.core.house.compute_houses_with_policy.

    - Reuses normalize_common_payload (so tz/date/time + timescales are normalized).
    - Ensures 'latitude' and 'longitude' are present in the normalized dict.
    - Resolves 'house_system' (or 'system') → canonical public label via house.py.
    - Returns (normalized_dict, warnings, tz_normalized).
    """
    base, warns, tz_name = normalize_common_payload(
        payload,
        default_time=default_time,
        compute_timescales=compute_timescales,
        include_jd_utc=include_jd_utc,
        dut1_seconds=dut1_seconds,
        max_levels=5,
    )

    # Geography (required by façade; route will enforce and 400 if missing)
    lat, lon = base.get("latitude"), base.get("longitude")
    if lat is None or lon is None:
        warns.append("houses_missing_geography")

    # System
    raw_system = payload.get("house_system", payload.get("system"))
    canon, wsys = normalize_house_system(raw_system)
    warns.extend(wsys)

    out = dict(base)
    out.update({
        "house_system": canon,          # preferred key for routes → façade
        "system": canon,                # mirror for convenience
        "requested_house_system": raw_system,  # for echo/diag if needed
    })

    return out, warns, tz_name

# ──────────────────────────────────────────────────────────────────────────────
# Domain-aware wrappers (wire-in Vedic/Western normalizers if present)
# ──────────────────────────────────────────────────────────────────────────────

def _invoke_or_fallback(
    fn: Optional[Callable[..., Tuple[Dict[str, Any], List[str], str]]],
    payload: Dict[str, Any],
    **opts: Any
) -> Tuple[Dict[str, Any], List[str], str]:
    if callable(fn):
        try:
            return fn(payload, **opts)  # type: ignore[misc]
        except Exception as e:
            base, w, tz = normalize_common_payload(payload, **opts)
            w = list(w) + [f"domain_normalizer_failed:{type(e).__name__}"]
            return base, w, tz
    return normalize_common_payload(payload, **opts)

def normalize_for_vedic(
    payload: Dict[str, Any],
    *,
    default_time: str = "12:00:00",
    compute_timescales: bool = True,
    include_jd_utc: bool = False,
    dut1_seconds: Optional[float] = None,
    max_levels: int = 5,
) -> Tuple[Dict[str, Any], List[str], str]:
    return _invoke_or_fallback(
        _VEDIC_NORMALIZER,
        payload,
        default_time=default_time,
        compute_timescales=compute_timescales,
        include_jd_utc=include_jd_utc,
        dut1_seconds=dut1_seconds,
        max_levels=max_levels,
    )

def normalize_for_western(
    payload: Dict[str, Any],
    *,
    default_time: str = "12:00:00",
    compute_timescales: bool = True,
    include_jd_utc: bool = False,
    dut1_seconds: Optional[float] = None,
    max_levels: int = 5,
) -> Tuple[Dict[str, Any], List[str], str]:
    return _invoke_or_fallback(
        _WESTERN_NORMALIZER,
        payload,
        default_time=default_time,
        compute_timescales=compute_timescales,
        include_jd_utc=include_jd_utc,
        dut1_seconds=dut1_seconds,
        max_levels=max_levels,
    )

def normalize_for_domain(
    domain: str,
    payload: Dict[str, Any],
    **opts: Any
) -> Tuple[Dict[str, Any], List[str], str]:
    d = (domain or "").strip().lower()
    if d == "vedic":
        return normalize_for_vedic(payload, **opts)
    if d == "western":
        return normalize_for_western(payload, **opts)
    return normalize_common_payload(payload, **opts)
