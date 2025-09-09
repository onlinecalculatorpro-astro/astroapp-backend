# app/core/time_kernel.py
from __future__ import annotations
"""
time_kernel.py — thin compatibility shim over the canonical timescales engine.

- Authoritative implementation lives in app.core.timescales (LOCKED).
- This module forwards the preferred API and preserves a few deprecated helpers.
- No POSIX timestamp math; ERFA two-part JD API only.
- Leap seconds (ss==60) supported exactly via ERFA.
- UT1 handled via ERFA utcut1 with DUT1 policy applied.

Preferred API (forwarder):
  build_timescales(date_str, time_str, tz_name, dut1_seconds) -> dict

Deprecated helpers (kept for back-compat):
  julian_day_utc(date_str, time_str, tz_name)                 -> float (JD_UTC)
  jd_tt_from_utc_jd(jd_utc)                                   -> float (JD_TT)
  jd_ut1_from_utc_jd(jd_utc, dut1_seconds)                    -> float (JD_UT1)

Return shape mirrors app.core.timescales:
{
  "jd_utc": float, "jd_tt": float, "jd_ut1": float,
  "delta_t": float, "dat": float, "dut1": float,
  "tz_offset_seconds": int, "timezone": str,
  "warnings": list[str], "precision": dict,
}
"""

from typing import Dict, Any, Tuple, Mapping, Optional
import math
import warnings

import erfa  # pyERFA exposes the ERFA namespace as 'erfa'
from app.core.timescales import (
    build_timescales as _engine_build_timescales,
    TimeScales as _TimeScales,
)

__all__ = [
    "build_timescales",
    "julian_day_utc",
    "jd_tt_from_utc_jd",
    "jd_ut1_from_utc_jd",
]

TIMEKERNEL_VERSION = "3.2.0"  # bumped for DST-nonexistent handling & dict normalization


# ──────────────────────────────────────────────────────────────────────────────
# Internal shape helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_dict(ts: Any) -> Dict[str, Any]:
    """
    Normalize TimeScales/dataclass or mapping to a plain dict.
    """
    if isinstance(ts, _TimeScales):
        return ts.to_dict()
    if isinstance(ts, Mapping):
        # Create a shallow copy to avoid leaking custom mapping types
        return dict(ts)
    raise TypeError(
        "Unsupported timescales payload (expected TimeScales or Mapping), "
        f"got {type(ts)!r}"
    )

def _get(d: Mapping[str, Any], key: str) -> Any:
    try:
        return d[key]
    except Exception:
        raise KeyError(f"Missing key '{key}' in timescales result")


# ──────────────────────────────────────────────────────────────────────────────
# Preferred API — direct forwarder to the locked engine (returns dict)
# ──────────────────────────────────────────────────────────────────────────────

def build_timescales(
    date_str: str,
    time_str: str,
    tz_name: str,
    dut1_seconds: float,
) -> Dict[str, Any]:
    """
    Forward to the canonical engine and normalize the return to a dict.

    This ensures stable shape for callers regardless of the engine's internal
    representation (dataclass vs mapping).
    """
    ts = _engine_build_timescales(date_str, time_str, tz_name, dut1_seconds)
    return _to_dict(ts)


# ──────────────────────────────────────────────────────────────────────────────
# Deprecated helpers (precision-safe; ERFA-only)
# ──────────────────────────────────────────────────────────────────────────────

def julian_day_utc(date_str: str, time_str: str, tz_name: str) -> float:
    """
    DEPRECATED — Use build_timescales(...)[\"jd_utc\"] instead.

    Convert civil local time + IANA zone to UTC JD via the canonical engine.
    DUT1 does not affect UTC itself; we pass 0.0 for stable UTC JD.
    """
    warnings.warn(
        "julian_day_utc() is deprecated; use build_timescales(...)[\"jd_utc\"]",
        DeprecationWarning,
        stacklevel=2,
    )
    ts = build_timescales(date_str, time_str, tz_name, dut1_seconds=0.0)
    return float(_get(ts, "jd_utc"))


def jd_tt_from_utc_jd(jd_utc: float) -> float:
    """
    DEPRECATED — Prefer build_timescales(...)[\"jd_tt\"].

    Convert UTC JD -> TT JD using the exact ERFA chain:
      UTC -> TAI -> TT
    Two-part JD arithmetic preserved; sums via math.fsum for stability.
    """
    warnings.warn(
        "jd_tt_from_utc_jd() is deprecated; prefer build_timescales(...)[\"jd_tt\"]",
        DeprecationWarning,
        stacklevel=2,
    )
    utc1, utc2 = _split_jd(jd_utc)
    tai1, tai2 = erfa.utctai(utc1, utc2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    return _fsum2(tt1, tt2)


def jd_ut1_from_utc_jd(jd_utc: float, dut1_seconds: float) -> float:
    """
    DEPRECATED — Prefer build_timescales(...)[\"jd_ut1\"].

    Convert UTC JD -> UT1 JD using ERFA:
      UT1 = UTC + DUT1 (seconds) via erfa.utcut1 in JD domain.

    Enforces DUT1 policy: |DUT1| ≤ 0.9 s (IERS), with tiny epsilon.
    """
    warnings.warn(
        "jd_ut1_from_utc_jd() is deprecated; prefer build_timescales(...)[\"jd_ut1\"]",
        DeprecationWarning,
        stacklevel=2,
    )
    _validate_dut1(dut1_seconds)
    utc1, utc2 = _split_jd(jd_utc)
    ut11, ut12 = erfa.utcut1(utc1, utc2, float(dut1_seconds))
    return _fsum2(ut11, ut12)


# ──────────────────────────────────────────────────────────────────────────────
# Internal precision-preserving helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_jd(jd: float) -> Tuple[float, float]:
    """
    Split a JD into ERFA two-part form as (integer_day, fractional_day).
    This preserves precision vs. passing (jd, 0.0).
    """
    d1 = math.floor(jd)
    d2 = jd - d1
    # Guard against rare FP rounding pushing d2 to 1.0
    if d2 >= 1.0:
        d1 += 1.0
        d2 -= 1.0
    return float(d1), float(d2)


def _fsum2(a: float, b: float) -> float:
    """Stable sum of two floats (mirrors canonical engine behavior)."""
    return math.fsum((float(a), float(b)))


def _validate_dut1(dut1_seconds: float) -> None:
    """Enforce IERS policy |DUT1| ≤ 0.9 s with a tiny epsilon."""
    if abs(float(dut1_seconds)) > 0.9 + 1e-12:
        raise ValueError(f"dut1_seconds out of range (|DUT1| ≤ 0.9 s): {dut1_seconds}")
