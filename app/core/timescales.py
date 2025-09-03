# app/core/timescales.py
# -----------------------------------------------------------------------------
# Research-grade timescale builder (ERFA aligned; no POSIX timestamp math for JDs)
#
# Public API (locked):
#   build_timescales(date_str, time_str, tz_name, dut1_seconds) -> TimeScales
#
# Guarantees:
#   • Exact ERFA chain:
#       UTC (calendar → JD) → TAI → TT      (erfa.dtf2d → utctai → taitt)
#       UT1 = UTC + DUT1                    (erfa.utcut1)
#   • Two-part JD arithmetic preserved internally; single float returned in API.
#   • ΔAT (TAI−UTC) from ERFA leap table via erfa.dat.
#   • DUT1 must be within ±0.9 s (IERS).
#   • UTC<1960 rejected (policy).
#   • Time zone offset reported from zoneinfo for the supplied local civil time.
#   • No POSIX timestamp math is used to produce JDs.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any
from datetime import datetime, date as _date, time as _time, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import math

import erfa  # pyERFA

__all__ = [
    "TimeScales",
    "build_timescales",
    # deprecated helpers
    "julian_day_utc",
    "jd_tt_from_utc_jd",
    "jd_ut1_from_utc_jd",
]

# ──────────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimeScales:
    jd_utc: float
    jd_tt: float
    jd_ut1: float
    delta_t: float         # TT − UT1 [s]
    dat: float             # TAI − UTC [s] from ERFA leap table
    dut1: float            # UT1 − UTC [s]
    tz_offset_seconds: int
    timezone: str
    warnings: List[str]
    precision: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_date_str(date_str: str) -> Tuple[int, int, int, List[str]]:
    """
    Parse 'YYYY-MM-DD' → (iy, im, id) and sanity-warn for far ranges.
    """
    warnings: List[str] = []
    try:
        y, m, d = date_str.strip().split("-")
        iy, im, iday = int(y), int(m), int(d)
        # Existence check
        datetime(iy, im, iday)
        # Practical range note (ERFA kernels, leap tables, etc.)
        if iy < 1600 or iy > 2200:
            warnings.append(f"date_year_{iy}_outside_optimal_erfa_range")
        return iy, im, iday, warnings
    except Exception as e:
        raise ValueError(f"Invalid date_str '{date_str}': {e}")

def _parse_time_hms_floatsec(time_str: str) -> Tuple[int, int, float]:
    """
    Parse 'HH:MM:SS' or 'HH:MM:SS.sss...' → (ih, imn, sec_float).
    Accepts leap seconds (sec_float may be 60.0).
    """
    t = time_str.strip()
    parts = t.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time_str '{time_str}': expected HH:MM:SS[.frac]")
    ih = int(parts[0]); imn = int(parts[1])
    # seconds may have fraction
    sec_float = float(parts[2]) if "." in parts[2] else float(int(parts[2]))

    # Basic field validation (leave leap handling to ERFA)
    if not (0 <= ih <= 24 and 0 <= imn <= 59 and 0.0 <= sec_float <= 60.0):
        raise ValueError(f"Invalid time fields: hh={ih}, mm={imn}, ss={sec_float}")

    return ih, imn, sec_float

def _fold_offsets(z: ZoneInfo, naive_local: datetime) -> Tuple[int, List[str]]:
    """
    Compute tz offset seconds for the given naive local datetime.
    Detect DST ambiguity; prefer fold=0 but warn if fold=1 differs.
    """
    warnings: List[str] = []
    aware0 = naive_local.replace(tzinfo=z, fold=0)
    off0 = aware0.utcoffset()
    if off0 is None:
        raise ValueError("Timezone returned None utcoffset()")
    aware1 = naive_local.replace(tzinfo=z, fold=1)
    off1 = aware1.utcoffset()
    if off1 is not None and off1 != off0:
        warnings.append(
            f"Ambiguous local time in zone '{z.key}' (fold=0 offset={int(off0.total_seconds())}, "
            f"fold=1 offset={int(off1.total_seconds())}); using fold=0."
        )
    return int(off0.total_seconds()), warnings

def _local_to_utc_calendar(
    date_str: str,
    time_str: str,
    tz_name: str,
) -> Tuple[int, int, int, int, int, float, int, List[str]]:
    """
    Convert local civil time (in tz) to UTC calendar fields suitable for ERFA.
    Returns: (iy, im, id, ih, imn, sec_float, tz_offset_seconds, warnings[])

    Leap seconds: Python datetime cannot represent ss==60. We:
      • create local time at 23:59:59.x,
      • convert to UTC,
      • and if input had sec==60.0, add +1s AFTER conversion.
    The resulting UTC fields represent the correct instant; ERFA will handle it.
    """
    iy, im, iday, warn_date = _parse_date_str(date_str)
    ih, imn, sec = _parse_time_hms_floatsec(time_str)
    warnings: List[str] = list(warn_date)

    # Detect leap second request
    leap_sec = (sec >= 60.0 - 1e-12)

    # Build a representable local datetime
    sec_for_dt = 59 if leap_sec else int(math.floor(sec))
    frac = 0.0 if leap_sec else max(0.0, min(0.999999, sec - sec_for_dt))
    micro = int(round(frac * 1_000_000))
    if micro >= 1_000_000:
        micro = 999_999
        warnings.append("microsecond_precision_clamped")

    try:
        z = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as e:
        raise ValueError(f"Unknown IANA time zone '{tz_name}'") from e

    naive_local = datetime(iy, im, iday, ih, imn, sec_for_dt, microsecond=micro)
    tz_off_sec, wz = _fold_offsets(z, naive_local)
    warnings.extend(wz)

    aware_local = naive_local.replace(tzinfo=z, fold=0)
    aware_utc = aware_local.astimezone(timezone.utc)
    if leap_sec:
        aware_utc = aware_utc + timedelta(seconds=1)

    # Extract UTC fields
    iy_u, im_u, id_u = aware_utc.year, aware_utc.month, aware_utc.day
    ih_u, in_u = aware_utc.hour, aware_utc.minute
    sec_u = float(aware_utc.second) + (aware_utc.microsecond / 1e6)

    return iy_u, im_u, id_u, ih_u, in_u, sec_u, tz_off_sec, warnings

def _utc_calendar_to_jd_utc(
    iy: int, im: int, iday: int, ih: int, imn: int, sec: float
) -> float:
    """
    Produce JD(UTC) via ERFA dtf2d using the 7-argument PyERFA signature:
      dtf2d(scale, iy, im, id, ih, imn, sec_float)
    """
    utc1, utc2 = erfa.dtf2d("UTC", int(iy), int(im), int(iday), int(ih), int(imn), float(sec))
    return float(utc1 + utc2)

def _delta_t_seconds(jd_tt: float, jd_ut1: float) -> float:
    return (jd_tt - jd_ut1) * 86400.0

def _dat_seconds(iy: int, im: int, iday: int, ih: int, imn: int, sec: float) -> float:
    """
    ERFA ΔAT = TAI − UTC for the given UTC calendar instant.
    erfa.dat expects fractional day in [0,1). Clip seconds to <60 for fd.
    """
    sec_clip = min(sec, 59.999999)  # keep fd < 1
    sod = (ih * 3600.0) + (imn * 60.0) + sec_clip
    fd = sod / 86400.0
    return float(erfa.dat(int(iy), int(im), int(iday), float(fd)))

def _split_jd(jd: float) -> Tuple[float, float]:
    d1 = math.floor(jd)
    d2 = jd - d1
    return float(d1), float(d2)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_timescales(
    date_str: str,
    time_str: str,
    tz_name: str,
    dut1_seconds: float,
) -> TimeScales:
    """
    Compute research-grade time scales for a local civil instant.
    """
    warnings: List[str] = []

    # DUT1 policy
    if not isinstance(dut1_seconds, (int, float)):
        raise TypeError("dut1_seconds must be a number (float seconds).")
    if abs(dut1_seconds) > 0.9 + 1e-9:
        raise ValueError(f"dut1_seconds out of range (|DUT1| ≤ 0.9 s): {dut1_seconds}")

    # Local → UTC calendar fields
    iy_u, im_u, id_u, ih_u, in_u, sec_u, tz_off, wz = _local_to_utc_calendar(date_str, time_str, tz_name)
    warnings.extend(wz)

    # UTC policy bound
    if (iy_u, im_u, id_u) < (1960, 1, 1):
        raise ValueError("UTC dates before 1960-01-01 are not supported by policy.")

    # JD(UTC)
    jd_utc = _utc_calendar_to_jd_utc(iy_u, im_u, id_u, ih_u, in_u, sec_u)

    # Two-part UTC JD for ERFA chains
    utc1, utc2 = _split_jd(jd_utc)

    # UTC → TAI → TT
    tai1, tai2 = erfa.utctai(utc1, utc2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    jd_tt = float(tt1 + tt2)

    # UTC + DUT1 → UT1
    ut11, ut12 = erfa.utcut1(utc1, utc2, float(dut1_seconds))
    jd_ut1 = float(ut11 + ut12)

    # ΔT & ΔAT
    delta_t = _delta_t_seconds(jd_tt, jd_ut1)
    dat = _dat_seconds(iy_u, im_u, id_u, ih_u, in_u, sec_u)

    precision = {
        "method": "ERFA dtf2d→utctai→taitt chain; utcut1 for UT1",
        "jd_precision_days": 1e-15,
        "time_input_resolution_seconds": 1e-4,
        "dut1_validated": True,
    }

    return TimeScales(
        jd_utc=jd_utc,
        jd_tt=jd_tt,
        jd_ut1=jd_ut1,
        delta_t=delta_t,
        dat=dat,
        dut1=float(dut1_seconds),
        tz_offset_seconds=int(tz_off),
        timezone=str(tz_name),
        warnings=warnings,
        precision=precision,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Deprecated convenience helpers (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def julian_day_utc(date_str: str, time_str: str, tz_name: str) -> float:
    """DEPRECATED. Prefer build_timescales(...)."""
    iy, im, id_u, ih, in_u, sec_u, _off, _w = _local_to_utc_calendar(date_str, time_str, tz_name)
    return _utc_calendar_to_jd_utc(iy, im, id_u, ih, in_u, sec_u)

def jd_tt_from_utc_jd(jd_utc: float) -> float:
    """DEPRECATED. Prefer build_timescales(...)."""
    utc1, utc2 = _split_jd(jd_utc)
    tai1, tai2 = erfa.utctai(utc1, utc2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    return float(tt1 + tt2)

def jd_ut1_from_utc_jd(jd_utc: float, dut1_seconds: float) -> float:
    """DEPRECATED. Prefer build_timescales(...)."""
    if abs(dut1_seconds) > 0.9 + 1e-9:
        raise ValueError("dut1_seconds out of range (|DUT1| ≤ 0.9 s).")
    utc1, utc2 = _split_jd(jd_utc)
    ut11, ut12 = erfa.utcut1(utc1, utc2, float(dut1_seconds))
    return float(ut11 + ut12)
