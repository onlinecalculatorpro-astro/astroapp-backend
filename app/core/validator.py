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

Exports (stable)
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

Domain-aware helpers
- normalize_for_vedic(payload, **opts) -> (dict, warnings, tz)
- normalize_for_western(payload, **opts) -> (dict, warnings, tz)
- normalize_for_domain(domain, payload, **opts) -> (dict, warnings, tz)
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import os

# ───────────────────────── optional domain normalizers ─────────────────────────
# Discover the best available function signature at import time.
# Accepted names inside each module, in order of preference:
#   normalize_payload, normalize, normalize_common_payload
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
# time_kernel.build_timescales → always returns a plain dict
try:
    from app.core.time_kernel import build_timescales as tk_build_timescales  # type: ignore
    _TK_AVAILABLE = True
except Exception:
    tk_build_timescales = None  # type: ignore
    _TK_AVAILABLE = False

# canonical engine (dataclass TimeScales); kept as fallback/compat
try:
    from app.core.timescales import build_timescales as ts_build_timescales, TimeScales  # type: ignore
    _TS_AVAILABLE = True
except Exception:
    ts_build_timescales = None  # type: ignore
    TimeScales = None           # type: ignore
    _TS_AVAILABLE = False

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
]

# ──────────────────────────────────────────────────────────────────────────────
# TZ helpers
# ──────────────────────────────────────────────────────────────────────────────

_TZ_ALIAS = {
    # small, conservative alias set
    "utc": "UTC",
    "gmt": "UTC",
    "ist": "Asia/Kolkata",
    "asia/calcutta": "Asia/Kolkata",
}

def normalize_tz(tz: Any, default: str = "UTC") -> str:
    """
    Normalize a timezone to a canonical IANA name (best-effort).
    Unknown/invalid types fall back to `default`.
    """
    if not isinstance(tz, str) or not tz.strip():
        return default
    key = tz.strip()
    alias = _TZ_ALIAS.get(key.lower())
    return alias or key

# ──────────────────────────────────────────────────────────────────────────────
# Generic coercions
# ──────────────────────────────────────────────────────────────────────────────

def clamp_levels(val: Any, *, default: int = 5, min_v: int = 1, max_v: int = 5) -> int:
    """
    Coerce an integer-like 'levels/depth/max_levels' into [min_v..max_v].
    Lists map to their length (bounded). Non-coercible → default.
    """
    if isinstance(val, list):
        n = len(val)
        return max(min_v, min(max_v, n))
    try:
        i = int(val)
        return max(min_v, min(max_v, i))
    except Exception:
        return default

def parse_latlon(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract latitude/longitude if present and numeric; else (None, None).
    Accepts 'latitude'/'longitude' keys.
    """
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
# Body helper
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
    """
    Normalize an ephemeris 'body' name. Accepts either the payload dict (with key 'body')
    or a raw string body name. Returns (canonical_or_None, warnings[]).
    """
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
    """
    Pull DUT1 from env (broadcast) with a safe default. Enforced again by ERFA.
    """
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0

def sanitize_timescales(ts: "TimeScales | Dict[str, Any]", *, include_jd_utc: bool = False) -> Dict[str, Any]:
    """
    Return a safe dict view of TimeScales. By default, **omits jd_utc** to avoid downstream
    reliance. Western flows can set include_jd_utc=True if needed.
    Accepts either:
      • dict (as returned by time_kernel.build_timescales)
      • TimeScales dataclass (from timescales engine)
    """
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
        or payload.get("timezone")      # alias
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
    """
    Compute timescales using the preferred forwarder. Returns (ts_dict|None, warnings[]).
    - Prefers time_kernel.build_timescales (dict)
    - Falls back to timescales.build_timescales (dataclass → dict via sanitize)
    """
    warns: List[str] = []
    # Preferred: time_kernel forwarder (dict)
    if _TK_AVAILABLE and tk_build_timescales is not None:
        ts = tk_build_timescales(date, time_str, tz_name, dut1)  # type: ignore[misc]
        ts_dict = sanitize_timescales(ts, include_jd_utc=include_jd_utc)
        warns.extend(ts_dict.get("warnings", []) or [])
        return ts_dict, warns

    # Fallback: canonical engine (dataclass)
    if _TS_AVAILABLE and ts_build_timescales is not None and date:
        ts = ts_build_timescales(date, time_str, tz_name, dut1)  # type: ignore[misc]
        ts_dict = sanitize_timescales(ts, include_jd_utc=include_jd_utc)
        warns.extend(ts_dict.get("warnings", []) or [])
        return ts_dict, warns

    # Neither available
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
    """
    Normalize cross-cutting inputs used by both Western & Vedic routes.

    Returns: (normalized_payload, warnings[], tz_normalized)

    - Extracts civil: date, time, tz (aliases resolved).
    - Injects friendly aliases (tz_name, ayanamsa_key, birth_date, birth_time, depth).
    - Coerces 'levels/depth/max_levels' into [1..max_levels].
    - Adds latitude/longitude if numeric.
    - If compute_timescales=True, computes jd_tt/jd_ut1 (and jd_utc only if include_jd_utc=True).
    - Always sets 'dut1_seconds' (env or provided).
    """
    warns: List[str] = []
    date, time_str, tz_name, wz = _extract_civil(payload, default_time=default_time)
    warns.extend(wz)

    # levels/depth
    levels_in = payload.get("levels", payload.get("depth", payload.get("max_levels", max_levels)))
    levels = clamp_levels(levels_in, default=max_levels, min_v=1, max_v=max_levels)

    # ayanamsa (neutral default 'lahiri' is safe for Vedic, harmless for Western)
    ayanamsa = str(payload.get("ayanamsa") or "lahiri").lower()

    # lat/lon (optional)
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

    # friendly civil aliases
    norm["birth_date"] = date
    norm["birth_time"] = time_str
    norm["place_tz"] = tz_name

    # DUT1
    dut1 = env_dut1_seconds() if dut1_seconds is None else float(dut1_seconds)
    norm["dut1_seconds"] = dut1

    # Compute TimeScales (optional)
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
        except Exception as e:
            warns.append(f"timescales_failed:{e!s}")

    return norm, warns, tz_name

# Thin alias for ops dispatcher (keeps its name stable)
def normalize_timescales_input(
    payload: Dict[str, Any],
    *,
    default_time: str = "12:00:00",
    compute_timescales: bool = False,  # routes may call computing later
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
            # If the domain module blows up, fall back to common but preserve context
            base, w, tz = normalize_common_payload(payload, **opts)
            w = list(w) + [f"domain_normalizer_failed:{type(e).__name__}"]
            return base, w, tz
    # no domain module → fallback
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
    """
    Use app.core.vedic_validator if available; else fall back to common normalization.
    The options mirror normalize_common_payload.
    """
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
    """
    Use app.core.western_validator if available; else fall back to common normalization.
    """
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
    """
    Generic dispatcher: domain in {"vedic", "western"} (case-insensitive).
    Unknown domain falls back to common normalization.
    """
    d = (domain or "").strip().lower()
    if d == "vedic":
        return normalize_for_vedic(payload, **opts)
    if d == "western":
        return normalize_for_western(payload, **opts)
    return normalize_common_payload(payload, **opts)
