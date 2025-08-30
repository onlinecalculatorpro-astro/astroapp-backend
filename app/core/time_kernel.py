# app/core/time_kernel.py
"""
Time Kernel — Offline, research-grade timescale conversions for AstroApp

Inputs (from validated payload):
  - Local civil time: date "YYYY-MM-DD", time "HH:MM[:SS]", IANA tz (e.g., "Asia/Kolkata")

Outputs:
  - JD_UTC : Julian Date in UTC
  - JD_TT  : Terrestrial Time (for ERFA/angles)
  - JD_UT1 : UT1 (for apparent sidereal time/GAST)
  - Meta   : ΔT (TT−UT1), ΔAT (TAI−UTC), TT−UTC seconds, DUT1 used, decimal year, MJD_UTC
             Leap source/status/last-known-MJD and warnings

Leap-seconds strategy (maintenance-light):
  1) Ask PyERFA at runtime (erfa.dat) → uses system ERFA table if fresh.
  2) If unavailable or "dubious", fall back to built-in steps (through 2017-01-01).
  3) Optional ops overrides via env/JSON (see app/core/leapseconds.py).
  4) Freshness policy via env ASTRO_LEAP_POLICY = "warn" (default) | "error".

ΔT model:
  - Espenak–Meeus polynomials (NASA) with small "c" correction outside 1955–2005.
  - Valid & smooth for 1900–2650 (and beyond by official pieces).

DUT1 (UT1−UTC):
  - Default from env ASTRO_DUT1_BROADCAST (seconds), clamped to [-0.9, +0.9]; else 0.0.
  - This keeps errors within ~0.004° if unset.

Pre-1972 handling:
  - No official TAI−UTC; we infer TT−UTC = ΔT + DUT1 (as standard practice).
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, List
import math
import os

from app.core.leapseconds import delta_at as resolve_delta_at, LeapInfo


# ---------------------------
# Datamodel
# ---------------------------
@dataclass
class Timescales:
    jd_utc: float
    jd_tt: float
    jd_ut1: float
    delta_t_sec: float            # TT - UT1 (Espenak–Meeus)
    delta_at_sec: Optional[float] # TAI - UTC (None when pre-1961/1972 path used)
    tt_minus_utc_sec: float       # TT - UTC seconds actually applied
    dut1_applied_sec: float       # UT1 - UTC (env override or 0.0), clamped to [-0.9, 0.9]
    y_decimal: float              # decimal year used for ΔT
    mjd_utc: float
    leap_source: Optional[str] = None
    leap_status: Optional[str] = None       # ok | stale | overridden | erfa-dubious
    leap_last_known_mjd: Optional[float] = None
    warnings: Optional[List[str]] = None


# ---------------------------
# Calendar & JD helpers
# ---------------------------
def _julian_date(dt_utc: datetime) -> float:
    """UTC-aware datetime -> Julian Date (days)."""
    if dt_utc.tzinfo is None or dt_utc.tzinfo != timezone.utc:
        raise ValueError("dt_utc must be timezone-aware in UTC")
    Y, M, D = dt_utc.year, dt_utc.month, dt_utc.day
    h, m = dt_utc.hour, dt_utc.minute
    s = dt_utc.second + dt_utc.microsecond / 1_000_000.0

    a = (14 - M) // 12
    y = Y + 4800 - a
    m_ = M + 12 * a - 3

    jdn = D + (153 * m_ + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    dayfrac = (h - 12) / 24.0 + m / 1440.0 + s / 86400.0
    return jdn + dayfrac


def _mjd_from_jd(jd: float) -> float:
    return jd - 2400000.5


def _decimal_year(dt_utc: datetime) -> float:
    """Espenak uses y = year + (month - 0.5)/12 (mid-month)."""
    return dt_utc.year + (dt_utc.month - 0.5) / 12.0


# ---------------------------
# ΔT (TT−UT1) — Espenak–Meeus
# ---------------------------
def _delta_t_seconds_espenak(y_decimal: float) -> float:
    """
    ΔT in seconds for decimal year y (NASA/Espenak–Meeus).
    Includes the small 'c' correction outside 1955–2005 per NASA note.
    """
    y = y_decimal

    if y < -500:
        u = (y - 1820.0) / 100.0
        delta = -20.0 + 32.0 * u * u
    elif -500 <= y < 500:
        u = y / 100.0
        delta = (10583.6 - 1014.41 * u + 33.78311 * u**2 - 5.952053 * u**3
                 - 0.1798452 * u**4 + 0.022174192 * u**5 + 0.0090316521 * u**6)
    elif 500 <= y < 1600:
        u = (y - 1000.0) / 100.0
        delta = (1574.2 - 556.01 * u + 71.23472 * u**2 + 0.319781 * u**3
                 - 0.8503463 * u**4 - 0.005050998 * u**5 + 0.0083572073 * u**6)
    elif 1600 <= y < 1700:
        t = y - 1600.0
        delta = 120.0 - 0.9808 * t - 0.01532 * t**2 + t**3 / 7129.0
    elif 1700 <= y < 1800:
        t = y - 1700.0
        delta = 8.83 + 0.1603 * t - 0.0059285 * t**2 + 0.00013336 * t**3 - t**4 / 1174000.0
    elif 1800 <= y < 1860:
        t = y - 1800.0
        delta = (13.72 - 0.332447 * t + 0.0068612 * t**2 + 0.0041116 * t**3
                 - 0.00037436 * t**4 + 0.0000121272 * t**5 - 0.0000001699 * t**6 + 0.000000000875 * t**7)
    elif 1860 <= y < 1900:
        t = y - 1860.0
        delta = 7.62 + 0.5737 * t - 0.251754 * t**2 + 0.01680668 * t**3 - 0.0004473624 * t**4 + t**5 / 233174.0
    elif 1900 <= y < 1920:
        t = y - 1900.0
        delta = -2.79 + 1.494119 * t - 0.0598939 * t**2 + 0.0061966 * t**3 - 0.000197 * t**4
    elif 1920 <= y < 1941:
        t = y - 1920.0
        delta = 21.20 + 0.84493 * t - 0.076100 * t**2 + 0.0020936 * t**3
    elif 1941 <= y < 1961:
        t = y - 1950.0
        delta = 29.07 + 0.407 * t - (t**2) / 233.0 + (t**3) / 2547.0
    elif 1961 <= y < 1986:
        t = y - 1975.0
        delta = 45.45 + 1.067 * t - (t**2) / 260.0 - (t**3) / 718.0
    elif 1986 <= y < 2005:
        t = y - 2000.0
        delta = (63.86 + 0.3345 * t - 0.060374 * t**2 + 0.0017275 * t**3
                 + 0.000651814 * t**4 + 0.00002373599 * t**5)
    elif 2005 <= y < 2050:
        t = y - 2000.0
        delta = 62.92 + 0.32217 * t + 0.005589 * t**2
    elif 2050 <= y < 2150:
        delta = -20.0 + 32.0 * ((y - 1820.0) / 100.0) ** 2 - 0.5628 * (2150.0 - y)
    else:
        u = (y - 1820.0) / 100.0
        delta = -20.0 + 32.0 * u * u

    # small "c" correction outside 1955–2005
    if not (1955.0 <= y <= 2005.0):
        c = -0.000012932 * (y - 1955.0) ** 2
        delta += c

    return float(delta)


# ---------------------------
# Public entrypoints
# ---------------------------
def utc_from_local(date_str: str, time_str: str, tz_name: str) -> datetime:
    """
    Build a timezone-aware UTC datetime from local components and IANA tz.
    date_str: 'YYYY-MM-DD'; time_str: 'HH:MM' or 'HH:MM:SS'; tz_name: 'Area/City'
    """
    parts = time_str.split(":")
    if len(parts) == 2:
        hh, mm = parts
        ss = "00"
    elif len(parts) == 3:
        hh, mm, ss = parts
    else:
        raise ValueError("time must be 'HH:MM' or 'HH:MM:SS'")

    try:
        tz = ZoneInfo(tz_name)
    except Exception as e:
        raise ValueError(f"invalid IANA timezone: {tz_name}") from e

    try:
        dt_local = datetime(
            int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]),
            int(hh), int(mm), int(ss), tzinfo=tz
        )
    except Exception as e:
        raise ValueError("invalid date/time components") from e

    return dt_local.astimezone(timezone.utc)


def compute_timescales_from_utc(
    dt_utc: datetime,
    *,
    dut1_override_sec: Optional[float] = None,
    leap_policy: Optional[str] = None,  # "warn" (default) | "error"
) -> Timescales:
    """
    Core conversion:
      - JD_UTC from UTC datetime
      - ΔT via Espenak–Meeus (TT−UT1)
      - ΔAT (TAI−UTC) via ERFA/builtin/override strategy
      - TT: if ΔAT known → TT−UTC = (TAI−UTC)+32.184; else (pre-1972) TT−UTC = ΔT + DUT1
      - UT1: UTC + DUT1 (env override ASTRO_DUT1_BROADCAST, clamped to [-0.9, 0.9])

    Freshness policy (leap seconds):
      - leap_policy = "warn" (default) → continue, attach warnings if status != ok
      - leap_policy = "error"         → raise ValueError if status != ok
    """
    if dt_utc.tzinfo != timezone.utc:
        raise ValueError("compute_timescales_from_utc expects a UTC-aware datetime")

    jd_utc = _julian_date(dt_utc)
    mjd_utc = _mjd_from_jd(jd_utc)
    y_dec = _decimal_year(dt_utc)
    warnings: List[str] = []

    # ΔT (TT − UT1)
    delta_t = _delta_t_seconds_espenak(y_dec)

    # DUT1 selection (UT1 − UTC)
    if dut1_override_sec is None:
        env = os.getenv("ASTRO_DUT1_BROADCAST", "")
        try:
            dut1_override_sec = float(env) if env else 0.0
        except ValueError:
            dut1_override_sec = 0.0
    dut1 = max(-0.9, min(0.9, float(dut1_override_sec)))

    # Leap policy
    if leap_policy is None:
        leap_policy = os.getenv("ASTRO_LEAP_POLICY", "warn").lower()
    if leap_policy not in ("warn", "error"):
        leap_policy = "warn"

    # ΔAT resolution (TAI − UTC)
    li: LeapInfo = resolve_delta_at(mjd_utc)
    if li.status != "ok":
        msg = f"leap second table status: {li.status} (source={li.source}; last_mjd={li.last_known_mjd})"
        if leap_policy == "error":
            raise ValueError(msg)
        warnings.append(msg)

    # Pre-1972 handling: no official TAI−UTC; infer TT−UTC = ΔT + DUT1
    PRE1972_MJD = 41317.0  # 1972-01-01
    if mjd_utc < PRE1972_MJD:
        delta_at = None
        tt_minus_utc = delta_t + dut1
        warnings.append("pre-1972 era: TT-UTC inferred as ΔT + DUT1 (no official TAI−UTC)")
    else:
        delta_at = float(li.delta_at)  # seconds
        tt_minus_utc = delta_at + 32.184

    jd_tt = jd_utc + tt_minus_utc / 86400.0
    jd_ut1 = jd_utc + dut1 / 86400.0

    return Timescales(
        jd_utc=jd_utc,
        jd_tt=jd_tt,
        jd_ut1=jd_ut1,
        delta_t_sec=delta_t,
        delta_at_sec=delta_at,
        tt_minus_utc_sec=tt_minus_utc,
        dut1_applied_sec=dut1,
        y_decimal=y_dec,
        mjd_utc=mjd_utc,
        leap_source=li.source,
        leap_status=li.status,
        leap_last_known_mjd=li.last_known_mjd,
        warnings=warnings or None,
    )


def compute_timescales_from_payload(
    *,
    date: str,
    time: str,
    place_tz: str,
    dut1_override_sec: Optional[float] = None,
    leap_policy: Optional[str] = None,
) -> Timescales:
    """
    Convenience wrapper for your validators’ payload shape.
    """
    dt_utc = utc_from_local(date, time, place_tz)
    return compute_timescales_from_utc(
        dt_utc,
        dut1_override_sec=dut1_override_sec,
        leap_policy=leap_policy,
    )


__all__ = [
    "Timescales",
    "utc_from_local",
    "compute_timescales_from_utc",
    "compute_timescales_from_payload",
]
