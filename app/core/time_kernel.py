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
from typing import Dict, Any, Tuple
import math
import warnings

import erfa  # pyERFA exposes the ERFA namespace as 'erfa'
from app.core.timescales import build_timescales as _build_timescales

__all__ = [
    "build_timescales",
    "julian_day_utc",
    "jd_tt_from_utc_jd",
    "jd_ut1_from_utc_jd",
]

TIMEKERNEL_VERSION = "3.1.0"  # bumped for precision fixes


# ──────────────────────────────────────────────────────────────────────────────
# Preferred API — direct forwarder to the locked engine
# ──────────────────────────────────────────────────────────────────────────────
def build_timescales(
    date_str: str,
    time_str: str,
    tz_name: str,
    dut1_seconds: float,
) -> Dict[str, Any]:
    """Forward to the canonical engine."""
    return _build_timescales(date_str, time_str, tz_name, dut1_seconds)


# ──────────────────────────────────────────────────────────────────────────────
# Deprecated helpers (precision-safe)
# ──────────────────────────────────────────────────────────────────────────────
def julian_day_utc(date_str: str, time_str: str, tz_name: str) -> float:
    """
    DEPRECATED — Use build_timescales(...)[\"jd_utc\"] instead.

    Convert civil local time + IANA zone to UTC JD via the canonical engine.
    """
    warnings.warn(
        "julian_day_utc() is deprecated; use build_timescales(...)[\"jd_utc\"]",
        DeprecationWarning,
        stacklevel=2,
    )
    # DUT1 does not affect UTC itself; pass 0.0 for a stable UTC JD.
    ts = _build_timescales(date_str, time_str, tz_name, dut1_seconds=0.0)
    return float(ts["jd_utc"])


def jd_tt_from_utc_jd(jd_utc: float) -> float:
    """
    DEPRECATED — Prefer build_timescales(...)[\"jd_tt\"].

    Convert UTC JD -> TT JD using exact ERFA chain:
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
      UT1 = UTC + DUT1 (seconds) applied via utcut1 in JD domain.

    Enforces DUT1 policy: |DUT1| ≤ 0.9 s.
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
# Internal helpers (precision-preserving)
# ──────────────────────────────────────────────────────────────────────────────
def _split_jd(jd: float) -> Tuple[float, float]:
    """
    Split a JD into ERFA two-part form as (integer_day, fractional_day).

    This preserves precision vs. passing (jd, 0.0).
    """
    # Integer part (floor) + fractional remainder in [0, 1)
    d1 = math.floor(jd)
    d2 = jd - d1
    # Guard against pathological rounding pushing d2 to 1.0 due to FP
    if d2 >= 1.0:
        d1 += 1.0
        d2 -= 1.0
    return float(d1), float(d2)


def _fsum2(a: float, b: float) -> float:
    """Stable sum of two floats (mirrors math used in timescales.py)."""
    return math.fsum((float(a), float(b)))


def _validate_dut1(dut1_seconds: float) -> None:
    """Enforce IERS policy |DUT1| ≤ 0.9 s with a tiny epsilon."""
    if abs(float(dut1_seconds)) > 0.9 + 1e-12:
        raise ValueError(
            f"dut1_seconds out of range (|DUT1| ≤ 0.9 s): {dut1_seconds}"
        )
