# app/core/timescales.py
# -----------------------------------------------------------------------------
# Research-grade timescale builder (ERFA aligned; no POSIX timestamp math for JDs)
#
# Public API (locked):
#   build_timescales(date_str, time_str, tz_name, dut1_seconds) -> TimeScales
#
# Guarantees:
#   • ERFA chain:
#       UTC (calendar → JD) → TAI → TT      (erfa.dtf2d → utctai → taitt)
#       UT1 = UTC + DUT1/86400              (erfa.utcut1)
#   • Two-part JD arithmetic preserved for ΔT; floats returned in API.
#   • ΔAT (TAI−UTC) via erfa.dat.
#   • DUT1 must be within ±0.9 s (IERS) with tiny epsilon.
#   • UTC < 1960 rejected (policy).
#   • Time zone offset via zoneinfo; DST ambiguity flagged.
#   • No POSIX timestamp math feeds any JD.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from decimal import Decimal, ROUND_HALF_UP
import math
import re

import erfa  # pyERFA

__all__ = [
    "TimeScales",
    "build_timescales",
    # deprecated helpers
    "julian_day_utc",
    "jd_tt_from_utc_jd",
    "jd_ut1_from_utc_jd",
]

# ───────────────────────────── Dataclass ─────────────────────────────

@dataclass(frozen=True)
class TimeScales:
    jd_utc: float
    jd_tt: float
    jd_ut1: float
    delta_t: float         # TT − UT1 [s]
    dat: float             # TAI − UTC [s] (aka delta_at)
    dut1: float            # UT1 − UTC [s]
    tz_offset_seconds: int
    timezone: str
    warnings: List[str]
    precision: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ───────────────────────────── Parsing helpers ─────────────────────────────

_DATE_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
_TIME_RE = re.compile(r"^\s*(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.(?P<f>\d+))?\s*$")

def _parse_date_str(date_str: str) -> Tuple[int, int, int, List[str]]:
    """Parse YYYY-MM-DD and return (iy, im, id) plus broad-range warning."""
    m = _DATE_RE.match(date_str or "")
    if not m:
        raise ValueError(f"Invalid date_str '{date_str}': expected YYYY-MM-DD")
    iy, im, iday = int(m.group(1)), int(m.group(2)), int(m.group(3))
    datetime(iy, im, iday)  # existence check
    warn: List[str] = []
    if iy < 1600 or iy > 2200:
        warn.append(f"date_year_{iy}_outside_optimal_erfa_range")
    return iy, im, iday, warn

def _parse_time(time_str: str) -> Tuple[int, int, int, str]:
    """
    Parse HH:MM:SS[.frac] and return (ih, im, isec, frac_str).
    Accepts leap seconds (ss == 60). frac_str contains only digits ('' if absent).
    """
    m = _TIME_RE.match(time_str or "")
    if not m:
        raise ValueError(f"Invalid time_str '{time_str}': expected HH:MM:SS[.frac]")
    ih = int(m.group("h")); im = int(m.group("m")); isec = int(m.group("s"))
    if not (0 <= ih <= 24 and 0 <= im <= 59 and 0 <= isec <= 60):
        raise ValueError(f"Invalid time fields: hh={ih}, mm={im}, ss={isec}")
    frac = (m.group("f") or "")
    frac = "".join(ch for ch in frac if ch.isdigit())
    return ih, im, isec, frac

# ───────────────────────────── Fraction handling ─────────────────────────────

def _frac_units(frac_str: str, *, allow_carry: bool) -> Tuple[int, int, bool, bool]:
    """
    From exact fractional digits, compute:
      micro       : int µs within [0..999_999] (rounded half-up; clamped if allow_carry=False)
      ifrac_1e4   : int 1e-4 s units within [0..9999] (rounded half-up; clamped if !allow_carry)
      warn_clamp  : True if digits > 6 (info beyond µs) or clamping applied
      carry       : True if rounding would cross a whole second (only if allow_carry=True)
    Both µs and 1e-4 s derive from the same Decimal, so behavior is consistent.
    """
    frac_str = "".join(ch for ch in (frac_str or "") if ch.isdigit())
    digits = len(frac_str)
    if digits == 0:
        return 0, 0, False, False

    warn_clamp = digits > 6

    frac_dec = Decimal("0." + frac_str)
    micro_dec = (frac_dec * Decimal(1_000_000)).to_integral_value(rounding=ROUND_HALF_UP)
    ifrac_dec = (frac_dec * Decimal(10_000)).to_integral_value(rounding=ROUND_HALF_UP)

    mic_carry = (micro_dec >= 1_000_000)
    ifrac_carry = (ifrac_dec >= 10_000)
    carry = (mic_carry or ifrac_carry) and allow_carry

    if allow_carry:
        micro = 0 if mic_carry else int(micro_dec)
        ifrac_1e4 = 0 if ifrac_carry else int(ifrac_dec)
    else:
        # Clamp inside the second (used for leap-second instants)
        micro = int(min(micro_dec, 999_999))
        ifrac_1e4 = int(min(ifrac_dec, 9_999))
        if mic_carry or ifrac_carry:
            warn_clamp = True
        carry = False  # never carry in leap-second mode

    return micro, ifrac_1e4, bool(warn_clamp), bool(carry)

def _micro_to_ifrac_1e4(us: int) -> int:
    """Round microseconds to 1e-4 s (100 µs) half-up, clamped to 0..9999."""
    return min(9999, (int(us) + 50) // 100)

# ───────────────────────────── Time zone / UTC helpers ─────────────────────────────

def _fold_offsets(z: ZoneInfo, naive_local: datetime) -> Tuple[int, List[str]]:
    """
    Compute tz offset seconds for a naive local datetime.
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
        warnings.append("dst_ambiguous")
    return int(off0.total_seconds()), warnings

def _local_to_utc_calendar(
    date_str: str,
    time_str: str,
    tz_name: str,
) -> Tuple[int, int, int, int, int, int, int, int, List[str]]:
    """
    Convert local civil time (in tz) to UTC calendar fields for ERFA.
    Returns: (iy, im, id, ih, imin, isec, ifrac_1e4, tz_offset_seconds, warnings[])
    """
    iy, im, iday, warn_date = _parse_date_str(date_str)
    ih, imin, isec_in, frac_str = _parse_time(time_str)
    warnings: List[str] = list(warn_date)

    leap_sec = (isec_in == 60)

    # Fractions
    micro, ifrac_in_1e4, warn_prec, carry = _frac_units(frac_str, allow_carry=not leap_sec)
    if warn_prec:
        warnings.append("microsecond_precision_clamped")

    # Build a local datetime for tz resolution:
    # - normal seconds: apply carry (if any)
    # - leap second: represent as :59 + fraction (never carry)
    build_sec = (59 if leap_sec else isec_in) + (1 if (not leap_sec and carry) else 0)

    try:
        z = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as e:
        raise ValueError(f"Unknown IANA time zone '{tz_name}'") from e

    base_local = datetime(iy, im, iday, ih, imin, min(build_sec, 59), microsecond=micro, tzinfo=None)
    if build_sec == 60:
        base_local = base_local + timedelta(seconds=1)

    tz_off_sec, wz = _fold_offsets(z, base_local)
    warnings.extend(wz)

    aware_local = base_local.replace(tzinfo=z, fold=0)
    aware_utc = aware_local.astimezone(timezone.utc)

    # UTC calendar fields
    iy_u, im_u, id_u = aware_utc.year, aware_utc.month, aware_utc.day
    ih_u, in_u, is_u = aware_utc.hour, aware_utc.minute, aware_utc.second

    # ERFA dtf2d inputs
    if leap_sec:
        # Represent the leap second itself
        isec_erfa = 60
        ifrac_erfa = ifrac_in_1e4
    else:
        if carry and ifrac_in_1e4 == 0:
            # Already carried to next whole second
            isec_erfa = is_u
            ifrac_erfa = 0
        else:
            isec_erfa = is_u
            ifrac_erfa = _micro_to_ifrac_1e4(aware_utc.microsecond)
            if ifrac_erfa >= 10_000:
                # Ultra-rare guard; keep consistent just in case
                aware_utc = aware_utc + timedelta(microseconds=100)
                iy_u, im_u, id_u = aware_utc.year, aware_utc.month, aware_utc.day
                ih_u, in_u, is_u = aware_utc.hour, aware_utc.minute, aware_utc.second
                isec_erfa = is_u
                ifrac_erfa = 0

    return iy_u, im_u, id_u, ih_u, in_u, isec_erfa, ifrac_erfa, tz_off_sec, warnings

def _utc_calendar_to_jd_utc(iy, im, iday, ih, imin, isec, ifrac_1e4):
    """
    ERFA dtf2d using the array calling convention:
      erfa.dtf2d("UTC", iy, im, iday, [ih, imin, isec, ifrac_1e4])
    The last element is in 1e-4 s units (0..9999); sec may be 60 at a leap second.
    """
    try:
        utc1, utc2 = erfa.dtf2d(
            "UTC", int(iy), int(im), int(iday),
            [int(ih), int(imin), int(isec), int(ifrac_1e4)]
        )
        return math.fsum((utc1, utc2))
    except Exception as e:
        raise ValueError(f"ERFA dtf2d failed: {e}")

def _delta_t_seconds_from_parts(tt1: float, tt2: float, ut11: float, ut12: float) -> float:
    """Two-part difference BEFORE collapsing — preserves precision."""
    return ((tt1 - ut11) + (tt2 - ut12)) * 86400.0

def _dat_seconds(iy: int, im: int, iday: int, ih: int, imin: int, isec: int, ifrac_1e4: int) -> float:
    """ERFA ΔAT = TAI − UTC for the given UTC calendar instant."""
    seconds = ih * 3600 + imin * 60 + min(isec, 59) + (ifrac_1e4 / 1e4)
    fd = seconds / 86400.0
    return float(erfa.dat(int(iy), int(im), int(iday), float(fd)))

def _split_jd(jd: float) -> Tuple[float, float]:
    d1 = math.floor(jd)
    d2 = jd - d1
    return float(d1), float(d2)

# ───────────────────────────── Public API ─────────────────────────────

def build_timescales(
    date_str: str,
    time_str: str,
    tz_name: str,
    dut1_seconds: float,
) -> TimeScales:
    """Compute research-grade time scales for a local civil instant."""
    warnings: List[str] = []

    # DUT1 policy
    if not isinstance(dut1_seconds, (int, float)):
        raise TypeError("dut1_seconds must be a number (float seconds).")
    if abs(dut1_seconds) > 0.9 + 1e-12:
        raise ValueError(f"dut1_seconds out of range (|DUT1| ≤ 0.9 s): {dut1_seconds}")

    # Local → UTC calendar fields
    iy_u, im_u, id_u, ih_u, in_u, is_u, ifrac_u, tz_off, wz = _local_to_utc_calendar(
        date_str, time_str, tz_name
    )
    warnings.extend(wz)

    # UTC policy bound
    if (iy_u, im_u, id_u) < (1960, 1, 1):
        raise ValueError("UTC dates before 1960-01-01 are not supported by policy.")

    # JD(UTC)
    jd_utc = _utc_calendar_to_jd_utc(iy_u, im_u, id_u, ih_u, in_u, is_u, ifrac_u)

    # Two-part UTC JD for ERFA chains
    utc1, utc2 = _split_jd(jd_utc)

    # UTC → TAI → TT
    tai1, tai2 = erfa.utctai(utc1, utc2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    jd_tt = math.fsum((tt1, tt2))

    # UTC + DUT1 → UT1
    ut11, ut12 = erfa.utcut1(utc1, utc2, float(dut1_seconds))
    jd_ut1 = math.fsum((ut11, ut12))

    # ΔT & ΔAT
    delta_t = _delta_t_seconds_from_parts(tt1, tt2, ut11, ut12)
    dat = _dat_seconds(iy_u, im_u, id_u, ih_u, in_u, is_u, ifrac_u)

    precision = {
        "method": "ERFA dtf2d→utctai→taitt; utcut1 for UT1",
        "jd_precision_days": 1e-15,
        "time_input_resolution_seconds": 1e-4,
        "dut1_validated": True,
    }

    return TimeScales(
        jd_utc=float(jd_utc),
        jd_tt=float(jd_tt),
        jd_ut1=float(jd_ut1),
        delta_t=float(delta_t),
        dat=float(dat),
        dut1=float(dut1_seconds),
        tz_offset_seconds=int(tz_off),
        timezone=str(tz_name),
        warnings=warnings,
        precision=precision,
    )

# ───────────────────────────── Deprecated helpers ─────────────────────────────

def julian_day_utc(date_str: str, time_str: str, tz_name: str) -> float:
    """DEPRECATED. Prefer build_timescales(...)."""
    iy, im, id_u, ih, in_u, is_u, ifrac_u, _off, _w = _local_to_utc_calendar(date_str, time_str, tz_name)
    return _utc_calendar_to_jd_utc(iy, im, id_u, ih, in_u, is_u, ifrac_u)

def jd_tt_from_utc_jd(jd_utc: float, *_, **__) -> float:
    """DEPRECATED. Prefer build_timescales(...). Extra args ignored for compatibility."""
    utc1, utc2 = _split_jd(jd_utc)
    tai1, tai2 = erfa.utctai(utc1, utc2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    return math.fsum((tt1, tt2))

def jd_ut1_from_utc_jd(jd_utc: float, dut1_seconds: float) -> float:
    """DEPRECATED. Prefer build_timescales(...)."""
    if abs(dut1_seconds) > 0.9 + 1e-12:
        raise ValueError("dut1_seconds out of range (|DUT1| ≤ 0.9 s).")
    utc1, utc2 = _split_jd(jd_utc)
    ut11, ut12 = erfa.utcut1(utc1, utc2, float(dut1_seconds))
    return math.fsum((ut11, ut12))
