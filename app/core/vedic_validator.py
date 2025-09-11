# app/core/vedic_validator.py
"""
Vedic API — Payload normalization & validation (Vimśottari)

Public API:
    normalize_vim_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], list[str], str]

What it does:
- Normalizes client payload into a canonical dict ("norm") suitable for Vimśottari engines.
- Adds common alias keys used by various registries (tz_name, ayanamsa_key, depth, etc).
- Optionally computes jd_tt and jd_ut1 via build_timescales(...) if available.
- NEVER includes jd_utc (ERFA-safe as requested).
- Returns (norm, warns, tz_norm) where:
    • norm   : normalized dict
    • warns  : human-readable warnings
    • tz_norm: normalized IANA zone string (e.g., "Asia/Kolkata")

Notes:
- This module is intentionally lightweight; it does not validate content beyond basic coercions.
- Depth/levels are clamped to [1..5].
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

# ── Timezone normalization table ──
_TZ_ALIAS = {
    "asia/patna": "Asia/Kolkata",
    "asia/calcutta": "Asia/Kolkata",
    "ist": "Asia/Kolkata",
    "indian standard time": "Asia/Kolkata",
}
_TZ_FALLBACK = "UTC"

# ── Simple helpers ──
_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")

def _normalize_tz(tz: Any) -> str:
    if not isinstance(tz, str):
        return _TZ_FALLBACK
    key = tz.strip()
    if not key:
        return _TZ_FALLBACK
    canon = _TZ_ALIAS.get(key.lower(), key)
    return canon

def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str) and _NUM_RE.match(x.strip()):
        try:
            return float(x.strip())
        except Exception:
            return None
    return None

def _clamp_depth(v: Any, default: int = 5) -> int:
    try:
        depth = int(v)
    except Exception:
        # If list passed (e.g., wanted levels [1,2,3,...]), use its length
        if isinstance(v, list):
            depth = len(v)
        else:
            depth = default
    if depth < 1: depth = 1
    if depth > 5: depth = 5
    return depth

def _coerce_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return default
    return str(x)

# ────────────────────────────────────────────────────────────────────────────────

def normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Vimśottari:

    Args:
        payload: raw request payload

    Returns:
        (norm, warns, tz_norm)
    """
    warns: List[str] = []

    # Civil primitives
    date = _coerce_str(payload.get("date") or payload.get("birth_date") or "")
    time_str = _coerce_str(payload.get("time") or payload.get("birth_time") or "12:00:00")
    tz_norm = _normalize_tz(payload.get("tz") or payload.get("place_tz") or _TZ_FALLBACK)

    # Optional geography
    lat = _as_float(payload.get("latitude"))
    lon = _as_float(payload.get("longitude"))

    # Levels / depth (clamped 1..5)
    levels_raw = payload.get("levels", payload.get("depth", payload.get("max_levels", 5)))
    depth = _clamp_depth(levels_raw, default=5)

    # Ayanāṃśa
    ayanamsa = _coerce_str(payload.get("ayanamsa") or "lahiri").strip().lower()
    if not ayanamsa:
        ayanamsa = "lahiri"

    # Optional timescales → jd_tt / jd_ut1 (NO jd_utc)
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
            # Accept dict-like or attr-like objects
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

    # Canonical normalized dict
    norm: Dict[str, Any] = {
        "system": "vimshottari",
        "date": date,
        "time": time_str,
        "tz": tz_norm,
        "ayanamsa": ayanamsa,
        "levels": depth,
        "max_levels": depth,
        "latitude": lat,
        "longitude": lon,
        "jd_tt": jd_tt,
        "jd_ut1": jd_ut1,
        "raw": payload,
    }

    # Common aliases some registries expect (harmless if unused)
    norm.update({
        "tz_name": tz_norm,
        "ayanamsa_key": ayanamsa,
        "birth_date": date,
        "birth_time": time_str,
        "place_tz": tz_norm,
        "depth": depth,
    })

    # Final light sanity notes
    if not date:
        warns.append("missing_date")
    if not time_str:
        warns.append("missing_time")
    if tz_norm == _TZ_FALLBACK and payload.get("tz"):
        warns.append("tz_normalization_fallback")

    return norm, warns, tz_norm
