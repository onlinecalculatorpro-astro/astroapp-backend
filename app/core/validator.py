# app/core/validator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os

# Only light dependency: your ERFA-aligned timescales
try:
    from app.core.timescales import build_timescales, TimeScales  # type: ignore
    _TS_AVAILABLE = True
except Exception:
    build_timescales = None  # type: ignore
    TimeScales = None        # type: ignore
    _TS_AVAILABLE = False

__all__ = [
    "normalize_tz",
    "env_dut1_seconds",
    "clamp_levels",
    "parse_latlon",
    "sanitize_timescales",
    "normalize_common_payload",
]

# ──────────────────────────────────────────────────────────────────────────────
# TZ helpers
# ──────────────────────────────────────────────────────────────────────────────

_TZ_ALIAS = {
    # Common aliases (keep small + conservative)
    "utc": "UTC",
    "gmt": "UTC",
    "ist": "Asia/Kolkata",
    "asia/patna": "Asia/Kolkata",
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
# TimeScales helpers
# ──────────────────────────────────────────────────────────────────────────────

def env_dut1_seconds() -> float:
    """
    Pull DUT1 from env (broadcast) with a safe default. Enforced again by build_timescales.
    """
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0

def sanitize_timescales(ts: TimeScales | Dict[str, Any], *, include_jd_utc: bool = False) -> Dict[str, Any]:
    """
    Return a safe dict view of TimeScales. By default, **omits jd_utc** to avoid downstream
    reliance (fits Vedic requirements). Western flows can set include_jd_utc=True if needed.
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
    """
    Extract civil inputs (date, time, tz) using relaxed keys and provide warnings.
    """
    warns: List[str] = []
    date = str(payload.get("date") or payload.get("birth_date") or "").strip()
    if not date:
        warns.append("missing_date")

    time_str = str(payload.get("time") or payload.get("birth_time") or default_time).strip()
    tz_name = normalize_tz(payload.get("tz") or payload.get("place_tz") or "UTC")

    return date, time_str, tz_name, warns

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
    - If compute_timescales=True and build_timescales is available, computes jd_tt/jd_ut1,
      **not** jd_utc (unless include_jd_utc=True).
    - Always sets 'dut1_seconds' (env or provided).
    """
    warns: List[str] = []
    date, time_str, tz_name, wz = _extract_civil(payload, default_time=default_time)
    warns.extend(wz)

    # levels/depth
    levels_in = payload.get("levels", payload.get("depth", payload.get("max_levels", max_levels)))
    levels = clamp_levels(levels_in, default=max_levels, min_v=1, max_v=max_levels)

    # ayanamsa (neutral default 'lahiri' is harmless on Western flows)
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
    if compute_timescales and _TS_AVAILABLE and date:
        try:
            ts = build_timescales(date, time_str, tz_name, dut1)  # type: ignore[misc]
            ts_dict = sanitize_timescales(ts, include_jd_utc=include_jd_utc)
            # Attach only jd_tt / jd_ut1 by default (no jd_utc for Vedic)
            norm.update({
                "jd_tt": ts_dict.get("jd_tt"),
                "jd_ut1": ts_dict.get("jd_ut1"),
                # keep full sanitized view for advanced consumers
                "timescales": ts_dict,
            })
            warns.extend(ts_dict.get("warnings", []) or [])
        except Exception as e:
            warns.append(f"timescales_failed:{e!s}")

    return norm, warns, tz_name
