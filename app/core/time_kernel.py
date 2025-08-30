# app/core/time_kernel.py
"""
Time Kernel — research-grade timescale conversions for AstroApp.

Inputs:
  • Local civil time: date "YYYY-MM-DD", time "HH:MM[:SS]", IANA tz (e.g., "Asia/Kolkata")
  • Or JD(UTC) via the jd_utc_to_timescales adapter

Outputs:
  • jd_utc  : Julian Date (UTC)
  • jd_tt   : Terrestrial Time (TT)
  • jd_ut1  : UT1
  • meta    : ΔT (TT−UT1), ΔAT (TAI−UTC), TT−UTC, DUT1 used, decimal year, MJD_UTC,
              leap table source/status/last-known, warnings

Policies & knobs:
  • ΔT: Espenak–Meeus polynomials (+ small 'c' correction outside 1955–2005)
  • ΔAT: try PyERFA (erfa.dat); fallback to internal helper (leapseconds.py)
  • DUT1: env ASTRO_DUT1_BROADCAST (clamped to [-0.9, 0.9]) or 0.0
  • Pre-1972: TT−UTC := ΔT + DUT1 (no official TAI−UTC then)
  • Freshness: ASTRO_LEAP_POLICY = "warn" (default) | "error"
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any

import os

from app.core.leapseconds import delta_at as resolve_delta_at, LeapInfo


# ─────────────────────────── Data model ────────────────────────────
@dataclass
class Timescales:
    jd_utc: float
    jd_tt: float
    jd_ut1: float
    delta_t_sec: float              # TT - UT1 (Espenak–Meeus)
    delta_at_sec: Optional[float]   # TAI - UTC (None when pre-1972 path used)
    tt_minus_utc_sec: float         # TT - UTC actually applied
    dut1_applied_sec: float         # UT1 - UTC applied (env override / 0.0)
    y_decimal: float                # decimal year used for ΔT
    mjd_utc: float
    leap_source: Optional[str] = None
    leap_status: Optional[str] = None     # ok | stale | overridden | unavailable
    leap_last_known_mjd: Optional[float] = None
    warnings: Optional[List[str]] = None


# ─────────────────────── JD & calendar helpers ─────────────────────
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


def _datetime_from_jd_utc(jd_utc: float) -> datetime:
    """
    Convert JD(UTC) -> timezone-aware datetime(UTC).
    Prefer PyERFA jd2cal; fall back to a pure-python conversion.
    """
    try:
        import erfa  # type: ignore
        iy, im, id_, fd = erfa.jd2cal(jd_utc, 0.0)  # fd in days [0,1)
        return datetime(iy, im, id_, tzinfo=timezone.utc) + timedelta(days=float(fd))
    except Exception:
        # Meeus-style inverse algorithm
        jd = jd_utc + 0.5
        Z = int(jd)
        F = jd - Z
        if Z < 2299161:
            A = Z
        else:
            alpha = int((Z - 1867216.25) / 36524.25)
            A = Z + 1 + alpha - int(alpha / 4)
        B = A + 1524
        C = int((B - 122.1) / 365.25)
        D = int(365.25 * C)
        E = int((B - D) / 30.6001)
        day = B - D - int(30.6001 * E) + F
        month = E - 1 if E < 14 else E - 13
        year = C - 4716 if month > 2 else C - 4715

        day_int = int(day)
        frac = day - day_int
        seconds = frac * 86400.0
        hh = int(seconds // 3600); seconds -= hh * 3600
        mm = int(seconds // 60);   seconds -= mm * 60
        ss = int(seconds)
        us = int(round((seconds - ss) * 1e6))
        if us == 1_000_000:
            ss += 1; us = 0
        return datetime(year, month, day_int, hh, mm, ss, us, tzinfo=timezone.utc)


# ─────────────────────── ΔT (Espenak–Meeus) ───────────────────────
def _delta_t_seconds_espenak(y_decimal: float) -> float:
    """NASA/Espenak–Meeus ΔT, with small 'c' correction outside 1955–2005."""
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
        delta = 8.83 + 0.1603 * t - 0.0059285 * t**2 + 0.00013336 * t**3 - t**4 / 1_174_000.0
    elif 1800 <= y < 1860:
        t = y - 1800.0
        delta = (13.72 - 0.332447 * t + 0.0068612 * t**2 + 0.0041116 * t**3
                 - 0.00037436 * t**4 + 0.0000121272 * t**5
                 - 0.0000001699 * t**6 + 0.000000000875 * t**7)
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

    if not (1955.0 <= y <= 2005.0):  # small 'c' correction
        delta += -0.000012932 * (y - 1955.0) ** 2
    return float(delta)


# ───────────────────────────── Public API ──────────────────────────
def utc_from_local(date_str: str, time_str: str, tz_name: str) -> datetime:
    """
    Build an aware UTC datetime from local components and IANA tz.
    date: 'YYYY-MM-DD'; time: 'HH:MM' or 'HH:MM:SS'; tz_name: 'Area/City'
    """
    parts = time_str.split(":")
    if len(parts) == 2:
        hh, mm = parts; ss = "00"
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
    leap_policy: Optional[str] = None,  # "warn" | "error"
) -> Timescales:
    """
    Core conversion:
      • JD_UTC from a UTC datetime
      • ΔT via Espenak–Meeus
      • ΔAT via PyERFA (preferred) or leapseconds helper
      • TT: ΔAT? → (TAI−UTC)+32.184 else (pre-1972 or unavailable) → ΔT + DUT1
      • UT1: UTC + DUT1 (env override clamped to [-0.9, 0.9])
    """
    if dt_utc.tzinfo != timezone.utc:
        raise ValueError("compute_timescales_from_utc expects a UTC-aware datetime")

    jd_utc = _julian_date(dt_utc)
    mjd_utc = _mjd_from_jd(jd_utc)
    y_dec = _decimal_year(dt_utc)
    warnings: List[str] = []

    # ΔT (TT − UT1)
    delta_t = _delta_t_seconds_espenak(y_dec)

    # DUT1 (UT1 − UTC)
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

    # ΔAT (TAI − UTC) — prefer ERFA, fallback to helper, final fallback to None
    delta_at: Optional[float] = None
    leap_source: Optional[str] = "erfa"
    leap_status: Optional[str] = "ok"
    leap_last_known: Optional[float] = None

    # Try ERFA first
    try:
        import erfa  # type: ignore
        fd = (dt_utc.hour + (dt_utc.minute + (dt_utc.second + dt_utc.microsecond / 1e6) / 60.0) / 60.0) / 24.0
        delta_at = float(erfa.dat(dt_utc.year, dt_utc.month, dt_utc.day, fd))
    except Exception:
        # Fallback to our helper
        try:
            li: LeapInfo = resolve_delta_at(mjd_utc)
            delta_at = float(li.delta_at)
            leap_source = li.source
            leap_status = li.status
            leap_last_known = li.last_known_mjd
            if li.status != "ok":
                msg = f"leap second table status: {li.status} (source={li.source}; last_mjd={li.last_known_mjd})"
                if leap_policy == "error":
                    raise ValueError(msg)
                warnings.append(msg)
        except Exception as e2:
            # Last resort: operate without ΔAT and warn (handled below)
            leap_source = "unavailable"
            leap_status = "unavailable"
            if leap_policy == "error":
                raise ValueError("leap seconds unavailable") from e2
            warnings.append("leap seconds unavailable; using TT-UTC = ΔT + DUT1 fallback")

    # Pre-1972 (or ΔAT unavailable): use ΔT + DUT1
    PRE1972_MJD = 41317.0  # 1972-01-01
    if delta_at is None or mjd_utc < PRE1972_MJD:
        if mjd_utc < PRE1972_MJD:
            warnings.append("pre-1972 era: TT-UTC inferred as ΔT + DUT1 (no official TAI−UTC)")
        tt_minus_utc = delta_t + dut1
    else:
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
        leap_source=leap_source,
        leap_status=leap_status,
        leap_last_known_mjd=leap_last_known,
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
    """Convenience wrapper for your validators’ payload shape."""
    dt_utc = utc_from_local(date, time, place_tz)
    return compute_timescales_from_utc(
        dt_utc,
        dut1_override_sec=dut1_override_sec,
        leap_policy=leap_policy,
    )


# ─────────────── Public adapter expected by routes.py ──────────────
def jd_utc_to_timescales(jd_utc: float) -> Dict[str, Any]:
    """
    Given JD(UTC), return a dict with jd_tt, jd_ut1, delta_t, delta_at, dut1, warnings, policy.
    """
    dt_utc = _datetime_from_jd_utc(jd_utc)
    ts = compute_timescales_from_utc(dt_utc)

    return {
        "jd_tt": float(ts.jd_tt),
        "jd_ut1": float(ts.jd_ut1),
        "delta_t": float(ts.delta_t_sec),
        "delta_at": 0.0 if ts.delta_at_sec is None else float(ts.delta_at_sec),
        "dut1": float(ts.dut1_applied_sec),
        "warnings": ts.warnings or [],
        "policy": {
            "leap_policy": os.getenv("ASTRO_LEAP_POLICY", "warn"),
            "dut1_source": "env:ASTRO_DUT1_BROADCAST",
            "leap_source": ts.leap_source,
            "leap_status": ts.leap_status,
            "leap_last_known_mjd": ts.leap_last_known_mjd,
        },
    }


# Alternate name accepted by routes
utc_jd_to_timescales = jd_utc_to_timescales


# Optional OO surface
class TimeKernel:
    def from_jd_utc(self, jd_utc: float) -> Dict[str, Any]:
        return jd_utc_to_timescales(jd_utc)

    def utc_jd_to_timescales(self, jd_utc: float) -> Dict[str, Any]:  # pragma: no cover
        return jd_utc_to_timescales(jd_utc)


__all__ = [
    "Timescales",
    "utc_from_local",
    "compute_timescales_from_utc",
    "compute_timescales_from_payload",
    "jd_utc_to_timescales",
    "utc_jd_to_timescales",
    "TimeKernel",
]
