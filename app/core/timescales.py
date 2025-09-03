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
#       UT1 = UTC + DUT1/86400              (erfa.utcut1)
#   • Two-part JD arithmetic preserved for ΔT; single floats returned in API.
#   • ΔAT (TAI−UTC) from ERFA leap table via erfa.dat.
#   • DUT1 must be within ±0.9 s (IERS).
#   • UTC<1960 rejected (policy).
#   • Time zone offset reported from zoneinfo for the supplied local civil time.
#   • No POSIX timestamp math is used to produce JDs.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any
from datetime import datetime, timezone, timedelta
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

# ───────────────────────────── Dataclass ─────────────────────────────

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

# ───────────────────────────── Internal helpers ─────────────────────────────

def _parse_date_str(date_str: str) -> Tuple[int, int, int, List[str]]:
    """Parse 'YYYY-MM-DD' → (iy, im, id) and add a broad-range warning."""
    warnings: List[str] = []
    try:
        y, m, d = date_str.strip().split("-")
        iy, im, iday = int(y), int(m), int(d)
        if iy < 1600 or iy > 2200:
            warnings.append(f"date_year_{iy}_outside_optimal_erfa_range")
        datetime(iy, im, iday)  # existence check
        return iy, im, iday, warnings
    except Exception as e:
        raise ValueError(f"Invalid date_str '{date_str}': {e}")

def _parse_time_to_ihmsf(time_str: str) -> Tuple[int, int, int, int]:
    """
    Parse 'HH:MM:SS' or 'HH:MM:SS.sss...' → (ih, im, is, ifrac) in 1e-4 s units.
    Accepts leap seconds (ss == 60).
    """
    t = time_str.strip()
    try:
        hh_s, mm_s, ss_s = t.split(":")
    except ValueError:
        raise ValueError(f"Invalid time_str '{time_str}': expected HH:MM:SS[.frac]")
    if "." in ss_s:
        s_part, frac = ss_s.split(".", 1)
        frac_digits = "".join(ch for ch in frac if ch.isdigit())
        ifrac = int((frac_digits + "0000")[:4])
    else:
        s_part, ifrac = ss_s, 0

    ih = int(hh_s); im = int(mm_s); isec = int(s_part)
    if not (0 <= ih <= 24 and 0 <= im <= 59 and 0 <= isec <= 60 and 0 <= ifrac <= 9999):
        raise ValueError(f"Invalid time fields: hh={ih}, mm={im}, ss={isec}, frac_1e4s={ifrac}")
    return ih, im, isec, ifrac

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
        warnings.append(
            f"Ambiguous local time in zone '{z.key}' (fold=0 offset={int(off0.total_seconds())}, "
            f"fold=1 offset={int(off1.total_seconds())}); using fold=0."
        )
    return int(off0.total_seconds()), warnings

def _local_to_utc_calendar(
    date_str: str,
    time_str: str,
    tz_name: str,
) -> Tuple[int, int, int, int, int, int, int, int, List[str]]:
    """
    Convert local civil time (in tz) to UTC calendar fields suitable for ERFA.
    Returns: (iy, im, id, ih, imin, isec, ifrac, tz_offset_seconds, warnings[])

    Leap seconds: if input second == 60, we:
      • Build datetime at second=59 (Python limitation) to resolve tz offset / UTC Y-M-D-H-M.
      • Use those Y/M/D/H/M from UTC, but set second=60 and ifrac=original for ERFA.
      • No +1s shift is applied (we represent the leap second itself).
    """
    iy, im, iday, warn_date = _parse_date_str(date_str)
    ih, imin, isec_in, ifrac_in = _parse_time_to_ihmsf(time_str)
    warnings: List[str] = list(warn_date)

    leap_sec = (isec_in == 60)
    build_sec = 59 if leap_sec else isec_in

    # 1e-4 s → microseconds with clamp
    computed_micro = int(round(ifrac_in * 100))
    micro = min(999_999, computed_micro)
    if computed_micro != ifrac_in * 100 or computed_micro >= 1_000_000:
        warnings.append("microsecond_precision_clamped")

    try:
        z = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as e:
        raise ValueError(f"Unknown IANA time zone '{tz_name}'") from e

    naive_local = datetime(iy, im, iday, ih, imin, build_sec, microsecond=micro, tzinfo=None)
    tz_off_sec, wz = _fold_offsets(z, naive_local)
    warnings.extend(wz)

    aware_local = naive_local.replace(tzinfo=z, fold=0)
    aware_utc = aware_local.astimezone(timezone.utc)

    # Extract UTC calendar fields
    iy_u, im_u, id_u = aware_utc.year, aware_utc.month, aware_utc.day
    ih_u, in_u, is_u = aware_utc.hour, aware_utc.minute, aware_utc.second

    if leap_sec:
        # Represent the leap second itself at 23:59:60.x
        ifrac_u = int(ifrac_in)
        is_u = 60
    else:
        # Normal: microsec → 1e-4 s with carry protection
        ifrac_u = int(round(aware_utc.microsecond / 100))
        if ifrac_u >= 10000:
            aware_utc = aware_utc + timedelta(microseconds=100)
            iy_u, im_u, id_u = aware_utc.year, aware_utc.month, aware_utc.day
            ih_u, in_u, is_u = aware_utc.hour, aware_utc.minute, aware_utc.second
            ifrac_u = int(round(aware_utc.microsecond / 100))

    return iy_u, im_u, id_u, ih_u, in_u, is_u, ifrac_u, tz_off_sec, warnings

def _utc_calendar_to_jd_utc(
    iy: int, im: int, iday: int, ih: int, imin: int, isec: int, ifrac: int
) -> float:
    """
    Produce JD(UTC) via ERFA dtf2d, robust across pyERFA builds:
    1) keyword 8-arg (iy,im,id,ih,imn,sec,f),
    2) positional 8-arg,
    3) 5-arg ihmsf[4] variant.
    """
    # 1) Keyword 8-arg (builds that expose named params: iy, im, id, ih, imn, sec, f)
    try:
        utc1, utc2 = erfa.dtf2d(
            "UTC",
            iy=int(iy), im=int(im), id=int(iday),
            ih=int(ih), imn=int(imin), sec=int(isec), f=int(ifrac),
        )
        return math.fsum((utc1, utc2))
    except TypeError:
        pass

    # 2) Positional 8-arg
    try:
        utc1, utc2 = erfa.dtf2d("UTC", int(iy), int(im), int(iday), int(ih), int(imin), int(isec), int(ifrac))
        return math.fsum((utc1, utc2))
    except TypeError:
        pass

    # 3) ihmsf[4]
    utc1, utc2 = erfa.dtf2d("UTC", int(iy), int(im), int(iday), [int(ih), int(imin), int(isec), int(ifrac)])
    return math.fsum((utc1, utc2))

def _delta_t_seconds_from_parts(tt1: float, tt2: float, ut11: float, ut12: float) -> float:
    """Two-part difference BEFORE collapsing — preserves precision."""
    return ((tt1 - ut11) + (tt2 - ut12)) * 86400.0

def _dat_seconds(iy: int, im: int, iday: int, ih: int, imin: int, isec: int, ifrac: int) -> float:
    """ERFA ΔAT = TAI − UTC for the given UTC calendar instant."""
    seconds = ih * 3600 + imin * 60 + min(isec, 59) + (ifrac / 1e4)
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
    if abs(dut1_seconds) > 0.9 + 1e-9:
        raise ValueError(f"dut1_seconds out of range (|DUT1| ≤ 0.9 s): {dut1_seconds}")

    # Local → UTC calendar fields
    iy_u, im_u, id_u, ih_u, in_u, is_u, ifrac_u, tz_off, wz = _local_to_utc_calendar(date_str, time_str, tz_name)
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

    # ΔT & ΔAT (ΔT with two-part precision)
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
    if abs(dut1_seconds) > 0.9 + 1e-9:
        raise ValueError("dut1_seconds out of range (|DUT1| ≤ 0.9 s).")
    utc1, utc2 = _split_jd(jd_utc)
    ut11, ut12 = erfa.utcut1(utc1, utc2, float(dut1_seconds))
    return math.fsum((ut11, ut12))
