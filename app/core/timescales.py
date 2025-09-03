# app/core/timescales.py
"""
Timescales (UTC↔TAI↔TT + UT1) — Research-grade core for AstroApp
-----------------------------------------------------------------
Author/Owner: AstroApp Core Team (research-grade implementation)

What this module does
- Converts local civil time (date/time + IANA timezone) to precise UTC, TT, and UT1
  Julian Days using ERFA (IAU SOFA) without POSIX timestamp round-trips.
- Computes ΔT = TT − UT1, returns ΔAT (TAI − UTC) actually used by ERFA, and applies
  DUT1 (UT1 − UTC) with IERS limits by default.
- Preserves the original local timezone offset (DST-aware) alongside UTC results.
- Provides precision metadata and warnings for edge policies (pre-1960 UTC, far-future ΔAT).

Key logic & design choices
- Exact chain via ERFA:
    UTC(two-part JD) → TAI (utctai) → TT (taitt)
  This guarantees leap-second correctness using ERFA’s internal tables.
- Two-part Julian Day representations everywhere (no float timestamp conversion),
  avoiding microsecond-level precision loss and leap-second ambiguities.
- UT1 is derived deterministically from UTC JD + DUT1/86400; DUT1 validated against
  IERS ±0.9 s (override via env if needed).
- Local time is parsed with zoneinfo (IANA tz database) so DST and historical offsets
  are faithfully captured and exported as tz_offset_seconds.

Outputs
- jd_utc, jd_tt, jd_ut1        (Julian Days)
- delta_t (TT−UT1, seconds), dat (TAI−UTC, seconds used by ERFA), dut1 (seconds)
- tz_offset_seconds (local offset), warnings (policy/precision notes)

Compatibility
- Back-compat helpers (julian_day_utc, jd_tt_from_utc_jd, jd_ut1_from_utc_jd) are kept
  for existing callers; new code should prefer build_timescales(...) for full fidelity.

References
- IAU SOFA / PyERFA time scale conversions: UTC↔TAI↔TT
- IERS conventions for DUT1 bounds (±0.9 s)
"""

# Research-grade time scale construction using PyERFA.
# - No POSIX timestamp conversions
# - Two-part JD everywhere
# - Exact UTC→TAI→TT via ERFA (uses internal leap-second table)
# - UT1 from provided DUT1 (strictly ±0.9s unless overridden)
# - Preserves local-timezone offset (DST-aware)

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, List

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    # Python <3.9 fallback if needed; better to vendor backports.zoneinfo instead.
    from backports.zoneinfo import ZoneInfo  # type: ignore

import erfa  # PyERFA (BSD) — IAU SOFA

# -----------------------------------------------------------------------------
# Policy toggles (env)
# -----------------------------------------------------------------------------
ALLOW_OUT_OF_RANGE_DUT1 = os.getenv("ASTRO_ALLOW_OUT_OF_RANGE_DUT1", "0").lower() in ("1", "true", "yes", "on")
ALLOW_PRE1960_UTC       = os.getenv("ASTRO_ALLOW_PRE1960_UTC", "0").lower() in ("1", "true", "yes", "on")
STRICT_DAT_RANGE        = os.getenv("ASTRO_TIMESCALES_STRICT_DAT_RANGE", "0").lower() in ("1","true","yes","on")
CROSSCHECK_DAT          = os.getenv("ASTRO_TIMESCALES_CROSSCHECK_DAT", "0").lower() in ("1","true","yes","on")

IERS_DUT1_LIMIT = 0.9  # seconds
LAST_KNOWN_LEAP_YEAR = 2016  # update when new leap second is announced

# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeScales:
    jd_utc: float           # Julian Day (UTC)
    jd_tt: float            # Julian Day (TT)
    jd_ut1: float           # Julian Day (UT1)
    delta_t: float          # TT-UT1 in seconds (a.k.a. ΔT)
    dat: float              # TAI-UTC (ΔAT) in seconds actually used by ERFA conversion
    dut1: float             # UT1-UTC (seconds) provided/used
    tz_offset_seconds: int  # local offset from UTC (seconds, incl. DST)
    warnings: List[str]     # any policy/precision notes

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_hms_to_ints(time_str: str) -> Tuple[int, int, float]:
    """
    Accept 'HH:MM' | 'HH:MM:SS' | 'HH:MM:SS.sss' and return (H, M, S.float).
    """
    s = str(time_str).strip()
    parts = s.split(":")
    if len(parts) == 2:
        hh, mm = int(parts[0]), int(parts[1])
        return hh, mm, 0.0
    if len(parts) == 3:
        hh, mm = int(parts[0]), int(parts[1])
        sec = float(parts[2])
        return hh, mm, sec
    raise ValueError("time must be 'HH:MM' or 'HH:MM:SS[.fff]'")

def _parse_local_to_utc(date_str: str, time_str: str, tz_name: str) -> Tuple[datetime, int, Tuple[int,int,float,int,int,float]]:
    """
    Parse local civil date/time + IANA TZ → UTC datetime (aware) + local tz offset (s)
    + components (Y,M,D,h,m,sec) for ERFA.
    """
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        raise ValueError("place_tz must be a valid IANA zone like 'Asia/Kolkata'")

    Y, M, D = [int(x) for x in str(date_str).split("-")]
    h, m, s = _parse_hms_to_ints(time_str)
    # Build local aware datetime (DST-aware)
    dt_local = datetime(Y, M, D, h, m, int(math.floor(s)), tzinfo=tz)
    # add fractional seconds to microsecond part if present
    frac = float(s) - math.floor(s)
    dt_local = dt_local.replace(microsecond=int(round(frac * 1e6)))
    # Offset in seconds (preserves DST)
    off = dt_local.utcoffset()
    tz_offset_seconds = int(off.total_seconds()) if off else 0
    # Convert to UTC
    dt_utc = dt_local.astimezone(timezone.utc)

    # ERFA components in UTC
    h_utc = dt_utc.hour
    m_utc = dt_utc.minute
    s_utc = dt_utc.second + dt_utc.microsecond / 1e6

    return dt_utc, tz_offset_seconds, (dt_utc.year, dt_utc.month, dt_utc.day, h_utc, m_utc, s_utc)

def _enforce_epoch_policy(dt_utc: datetime, warnings_list: List[str]) -> None:
    if dt_utc.year < 1960 and not ALLOW_PRE1960_UTC:
        raise ValueError("UTC before 1960-01-01 is outside strict research-grade support (set ASTRO_ALLOW_PRE1960_UTC=1 to override).")
    # Optional conservative guard on very-far future ΔAT usage
    if STRICT_DAT_RANGE and dt_utc.year > LAST_KNOWN_LEAP_YEAR + 50:
        warnings_list.append(f"dat_range_uncertain_for_year_{dt_utc.year}")

def _utc_two_part_jd(y: int, m: int, d: int, hh: int, mm: int, ss: float) -> Tuple[float, float]:
    """
    Build a two-part JD (UTC) via ERFA. Uses ERFA's leap-second logic.
    """
    # dtf2d(scale, iy, im, id, ihr, imn, sec) -> (djm0, djm)
    # Use 'UTC' scale to ensure leap-second correctness for the given civil time.
    utc1, utc2 = erfa.dtf2d("UTC", y, m, d, hh, mm, ss)
    return utc1, utc2

# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------
def build_timescales(date_str: str, time_str: str, tz_name: str, dut1_seconds: float) -> TimeScales:
    """
    Research-grade construction of UTC/TT/UT1 Julian Days from local civil time.
    - Strict: relies on ERFA for leap seconds & UTC→TAI→TT chain.
    - UT1 from DUT1 (UT1−UTC) in seconds. Enforced to ±0.9s unless ASTRO_ALLOW_OUT_OF_RANGE_DUT1=1.

    Returns TimeScales with jd_utc, jd_tt, jd_ut1, delta_t (TT−UT1, s), dat (TAI−UTC, s), dut1, tz_offset_seconds, warnings.
    """
    warnings_list: List[str] = []

    # Parse local → UTC with exact TZ offset preserved
    dt_utc, tz_offset_seconds, (y, mo, d, hh, mm, ss) = _parse_local_to_utc(date_str, time_str, tz_name)

    # Policy checks (epoch range, future ΔAT range)
    _enforce_epoch_policy(dt_utc, warnings_list)

    # DUT1 validation
    try:
        d = float(dut1_seconds)
    except Exception:
        raise ValueError(f"DUT1 must be a float seconds value; got {dut1_seconds!r}")
    if not ALLOW_OUT_OF_RANGE_DUT1 and abs(d) > IERS_DUT1_LIMIT:
        raise ValueError(f"DUT1 exceeds IERS limit ±{IERS_DUT1_LIMIT:.1f}s: {d:.6f} s")
    dut1 = float(d)

    # UTC two-part JD (leap-safe)
    utc1, utc2 = _utc_two_part_jd(y, mo, d, hh, mm, ss)
    jd_utc = float(utc1 + utc2)

    # UTC → TAI → TT via ERFA (exact; uses internal leap-second table)
    tai1, tai2 = erfa.utctai(utc1, utc2)    # UTC → TAI
    tt1, tt2   = erfa.taitt(tai1, tai2)     # TAI → TT
    jd_tt = float(tt1 + tt2)

    # ΔAT actually used (TAI−UTC)
    dat_used = float((tai1 + tai2 - utc1 - utc2) * 86400.0)  # seconds

    # Optional cross-check against erfa.dat()
    if CROSSCHECK_DAT:
        try:
            fd = (hh + mm/60.0 + ss/3600.0) / 24.0
            dat_probe = float(erfa.dat(y, mo, d, fd))
            if abs(dat_probe - dat_used) > 5e-10:  # ~0.5 ns
                warnings_list.append(f"dat_crosscheck_mismatch(use={dat_used:.12f}s vs probe={dat_probe:.12f}s)")
        except Exception as e:  # pragma: no cover
            warnings_list.append(f"dat_crosscheck_error:{type(e).__name__}")

    # UT1 = UTC + DUT1
    jd_ut1 = float(jd_utc + dut1 / 86400.0)

    # ΔT = TT − UT1  (seconds)
    delta_t = float((jd_tt - jd_ut1) * 86400.0)

    return TimeScales(
        jd_utc=jd_utc,
        jd_tt=jd_tt,
        jd_ut1=jd_ut1,
        delta_t=delta_t,
        dat=dat_used,
        dut1=dut1,
        tz_offset_seconds=tz_offset_seconds,
        warnings=warnings_list,
    )

# -----------------------------------------------------------------------------
# Back-compat/utility functions (kept to avoid breaking existing imports)
# -----------------------------------------------------------------------------
def julian_day_utc(date_str: str, time_str: str, tz_name: str) -> float:
    """
    Return UTC JD for the given local civil time (uses ERFA; leap-safe).
    """
    dt_utc, _off, (y, mo, d, hh, mm, ss) = _parse_local_to_utc(date_str, time_str, tz_name)
    _enforce_epoch_policy(dt_utc, [])
    utc1, utc2 = _utc_two_part_jd(y, mo, d, hh, mm, ss)
    return float(utc1 + utc2)

def jd_ut1_from_utc_jd(jd_utc: float, dut1_seconds: float) -> float:
    """
    Convert UTC JD → UT1 JD using DUT1. Research-grade note:
    - Assumes jd_utc is a true UTC JD (not an approximate civil->timestamp conversion).
    """
    try:
        d = float(dut1_seconds)
    except Exception:
        raise ValueError(f"DUT1 must be float seconds; got {dut1_seconds!r}")
    if not ALLOW_OUT_OF_RANGE_DUT1 and abs(d) > IERS_DUT1_LIMIT:
        raise ValueError(f"DUT1 exceeds IERS limit ±{IERS_DUT1_LIMIT:.1f}s: {d:.6f} s")
    return float(jd_utc) + d / 86400.0

def jd_tt_from_utc_jd(jd_utc: float) -> float:
    """
    Convert UTC JD → TT JD via ERFA: UTC→TAI→TT (exact; leap-safe).
    NOTE: Prefer build_timescales() when you also need UT1 and local offset.
    """
    # Split deterministically into two-part JD
    u1, u2 = erfa.dj2d(float(jd_utc))
    # UTC→TAI→TT
    t1, t2 = erfa.utctai(u1, u2)
    q1, q2 = erfa.taitt(t1, t2)
    return float(q1 + q2)

def delta_t_seconds(year: int, month: int) -> float:
    """
    DEPRECATED: Approximate ΔT (TT−UT1) lookup/fit (Espenak–Meeus-like).
    - Kept only for legacy callers. The main API should use build_timescales().
    - Returns a smooth approximation in seconds; not for research-grade.
    """
    warnings.warn(
        "delta_t_seconds(year, month) is deprecated; use build_timescales(...) for exact ΔT.",
        DeprecationWarning, stacklevel=2
    )
    # Very conservative polynomial/segments; not used by core.
    y = float(year) + (max(1, min(12, int(month))) - 0.5) / 12.0
    # Piecewise coarse model
    if y < 1600:
        t = (y - 1600.0) / 100.0
        dt = 120.0 + (-0.9808)*t + (-0.01532)*(t*t)
    elif y < 1860:
        t = y - 1800.0
        dt = 13.72 - 0.332447*t + 0.0068612*(t*t) + 0.0041116*(t**3) - 0.00037436*(t**4) + 0.0000121272*(t**5) - 0.0000001699*(t**6) + 0.000000000875*(t**7)
    elif y < 1900:
        t = y - 1860.0
        dt = 7.62 + 0.5737*t - 0.251754*(t*t) + 0.01680668*(t**3) - 0.0004473624*(t**4) + (1.0/233174)*(t**5)
    elif y < 2000:
        t = y - 1900.0
        dt = -2.79 + 1.494119*t - 0.0598939*(t*t) + 0.0061966*(t**3) - 0.000197*(t**4)
    elif y < 2100:
        t = y - 2000.0
        dt = 64.0 + 0.2930*t
    else:
        t = (y - 1820.0) / 100.0
        dt = -20.0 + 32.0*(t*t)
    return float(dt)

__all__ = [
    "TimeScales",
    "build_timescales",
    "julian_day_utc",
    "jd_ut1_from_utc_jd",
    "jd_tt_from_utc_jd",
    "delta_t_seconds",
]
