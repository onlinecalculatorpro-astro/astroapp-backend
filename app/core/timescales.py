# app/core/timescales.py
from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, Tuple, Optional

# ──────────────────────────────── Internals ────────────────────────────────

def _normalize_hms(time_str: str) -> str:
    """
    Accept 'HH:MM' or 'HH:MM:SS' (24h). Return canonical 'HH:MM:SS'.
    """
    s = str(time_str).strip()
    parts = s.split(":")
    if len(parts) == 2:
        hh, mm = parts
        ss = "00"
    elif len(parts) == 3:
        hh, mm, ss = parts
    else:
        raise ValueError("time must be 'HH:MM' or 'HH:MM:SS'")
    h = int(hh); m = int(mm); sec = int(ss)
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59):
        raise ValueError("time must be within 00:00:00–23:59:59")
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _to_utc_datetime(date_str: str, time_str: str, tz_str: str) -> datetime:
    """
    Parse local civil date/time/tz and return an aware UTC datetime.
    """
    t_norm = _normalize_hms(time_str)
    try:
        tz = ZoneInfo(str(tz_str).strip())
    except Exception as e:
        raise ValueError(f"place_tz/timezone must be a valid IANA zone (got {tz_str!r})") from e
    dt_local = datetime.fromisoformat(f"{date_str}T{t_norm}").replace(tzinfo=tz)
    return dt_local.astimezone(timezone.utc)


def _jd_from_utc_datetime(dt_utc: datetime) -> float:
    """
    Meeus algorithm for JD from an aware UTC datetime.
    Uses proleptic Gregorian calendar, valid for modern era usage.
    """
    Y = dt_utc.year
    M = dt_utc.month
    D = dt_utc.day

    frac_day = (
        dt_utc.hour
        + dt_utc.minute / 60.0
        + dt_utc.second / 3600.0
        + dt_utc.microsecond / 3.6e9
    ) / 24.0

    if M <= 2:
        Y -= 1
        M += 12

    A = math.floor(Y / 100)
    B = 2 - A + math.floor(A / 4)

    JD0 = math.floor(365.25 * (Y + 4716)) + math.floor(30.6001 * (M + 1)) + D + B - 1524.5
    return float(JD0 + frac_day)


# ───────────────────────────── Leap seconds (ΔAT=TAI−UTC) ─────────────────────────────
# Table of (UTC effective date, ΔAT seconds) for dates >= 1972-01-01.
# Source: IERS bulletins (last leap second currently 2017-01-01 with ΔAT=37).
# We intentionally keep this small and modern; for pre-1972 dates we fall back to ΔT polynomials.
_LEAP_TABLE: Tuple[Tuple[datetime, int], ...] = tuple(
    (datetime(y, m, d, tzinfo=timezone.utc), dat) for (y, m, d, dat) in (
        (1972,  1,  1, 10), (1972,  7,  1, 11),
        (1973,  1,  1, 12), (1974,  1,  1, 13), (1975,  1,  1, 14),
        (1976,  1,  1, 15), (1977,  1,  1, 16), (1978,  1,  1, 17),
        (1979,  1,  1, 18), (1980,  1,  1, 19),
        (1981,  7,  1, 20), (1982,  7,  1, 21), (1983,  7,  1, 22),
        (1985,  7,  1, 23), (1988,  1,  1, 24),
        (1990,  1,  1, 25), (1991,  1,  1, 26), (1992,  7,  1, 27),
        (1993,  7,  1, 28), (1994,  7,  1, 29), (1996,  1,  1, 30),
        (1997,  7,  1, 31), (1999,  1,  1, 32),
        (2006,  1,  1, 33), (2009,  1,  1, 34),
        (2012,  7,  1, 35), (2015,  7,  1, 36),
        (2017,  1,  1, 37),
    )
)

def _delta_at_seconds(dt_utc: datetime) -> Optional[int]:
    """
    Return ΔAT (TAI−UTC) in seconds for dt_utc if >= 1972-01-01, else None.
    """
    at: Optional[int] = None
    for eff, dat in _LEAP_TABLE:
        if dt_utc >= eff:
            at = dat
        else:
            break
    return at


# ───────────────────────────── Public API ─────────────────────────────

def to_utc_iso(date_str: str, time_str: str, tz_str: str) -> str:
    """
    Convert local civil to 'YYYY-MM-DDTHH:MM:SSZ' UTC.
    """
    return _to_utc_datetime(date_str, time_str, tz_str).strftime("%Y-%m-%dT%H:%M:%SZ")


def julian_day_utc(date_str: str, time_str: str, tz_str: str) -> float:
    """
    JD(UTC) for the provided local civil instant.
    """
    return _jd_from_utc_datetime(_to_utc_datetime(date_str, time_str, tz_str))


# ---- ΔT (TT−UT1) model (Espenak–Meeus style) ---------------------------------

def delta_t_seconds(year: int, month: int) -> float:
    """
    ΔT ≡ TT − UT1, in seconds. Piecewise polynomial per Espenak–Meeus,
    suitable ~1800–2050 with gentle extrapolation beyond.
    """
    y = float(year) + (float(month) - 0.5) / 12.0

    if 2005.0 <= y <= 2050.0:
        t = y - 2000.0
        return 62.92 + 0.32217 * t + 0.005589 * (t ** 2)

    if 1986.0 <= y < 2005.0:
        t = y - 2000.0
        return (
            63.86
            + 0.3345 * t
            - 0.060374 * (t ** 2)
            + 0.0017275 * (t ** 3)
            + 0.000651814 * (t ** 4)
            + 0.00002373599 * (t ** 5)
        )

    if 1900.0 <= y < 1986.0:
        t = y - 1900.0
        return -2.79 + 1.494119 * t - 0.0598939 * (t ** 2) + 0.0061966 * (t ** 3) - 0.000197 * (t ** 4)

    if 1800.0 <= y < 1900.0:
        t = (y - 1860.0) / 100.0
        return 13.72 - 33.244 * t + 68.612 * (t ** 2) + 4111.6 * (t ** 4)

    if 2050.0 < y <= 2150.0:
        t = y - 2000.0
        return 62.92 + 0.32217 * t + 0.005589 * (t ** 2)

    # far outside main range: quadratic around 1820 baseline
    u = (y - 1820.0) / 100.0
    return -20.0 + 32.0 * (u ** 2)

# Back-compat alias
delta_T_seconds = delta_t_seconds


# ---- TT from UTC JD (modern: ΔAT+32.184; legacy: ΔT [+ DUT1]) ----------------

def jd_tt_from_utc_jd(
    jd_utc: float,
    year: int,
    month: int,
    *,
    dut1_seconds: Optional[float] = None,
) -> float:
    """
    Convert JD(UTC) → JD(TT).

    Preferred modern method (UTC ≥ 1972-01-01):
        TT = UTC + (ΔAT + 32.184 s)

    Legacy fallback (pre-1972, where ΔAT not standardized):
        TT = UTC + (ΔT + DUT1)  [since ΔT=TT−UT1 and UT1=UTC+DUT1]

    If DUT1 is not provided for pre-1972 dates, we assume DUT1≈0 for the fallback.
    """
    # We need the actual UTC datetime to check leap second availability.
    # Approximate reconstruction from jd_utc by using year/month arguments
    # (callers supply these from the original civil instant).
    try:
        # Use the 15th of the month at 00:00 as a stable representative instant.
        # Only used to pick ΔAT regime, not to compute JD.
        dt_probe = datetime(year, month, 15, tzinfo=timezone.utc)
    except Exception:
        dt_probe = datetime(1972, 1, 1, tzinfo=timezone.utc)

    dat = _delta_at_seconds(dt_probe)
    if dat is not None:
        # Modern: TT−UTC = ΔAT + 32.184
        return float(jd_utc) + (dat + 32.184) / 86400.0

    # Legacy/pre-1972 path: use ΔT polynomial, optionally add DUT1 if given
    DT = float(delta_t_seconds(year, month))
    if dut1_seconds is None:
        return float(jd_utc) + DT / 86400.0
    return float(jd_utc) + (DT + float(dut1_seconds)) / 86400.0


def jd_ut1_from_utc_jd(jd_utc: float, dut1_seconds: float) -> float:
    """
    JD(UT1) = JD(UTC) + DUT1/86400
    """
    return float(jd_utc) + float(dut1_seconds) / 86400.0


# ────────────────────── Convenience: full set from civil inputs ──────────────────────

def civil_timescales(
    date_str: str,
    time_str: str,
    tz_str: str,
    dut1_seconds: float = 0.0,
) -> Dict[str, float]:
    """
    Convenience helper returning:
      {
        "jd_utc", "jd_tt", "jd_ut1", "delta_t", "dut1"
      }

    Notes:
      • jd_tt uses the modern TT−UTC = ΔAT + 32.184 s method where available.
      • delta_t is the Espenak–Meeus ΔT (TT−UT1) polynomial, for diagnostics/UI.
      • jd_ut1 uses the provided DUT1 seconds (default 0.0 if unknown).
    """
    dt_utc = _to_utc_datetime(date_str, time_str, tz_str)
    jd_utc = _jd_from_utc_datetime(dt_utc)

    # Modern TT from leap seconds when possible; else ΔT (+DUT1)
    jd_tt = jd_tt_from_utc_jd(jd_utc, dt_utc.year, dt_utc.month, dut1_seconds=dut1_seconds)
    jd_ut1 = jd_ut1_from_utc_jd(jd_utc, dut1_seconds)

    return {
        "jd_utc": float(jd_utc),
        "jd_tt": float(jd_tt),
        "jd_ut1": float(jd_ut1),
        "delta_t": float(delta_t_seconds(dt_utc.year, dt_utc.month)),
        "dut1": float(dut1_seconds),
    }


# ───────────────────────────── Exports ─────────────────────────────

__all__ = [
    "to_utc_iso",
    "julian_day_utc",
    "delta_t_seconds",
    "delta_T_seconds",
    "jd_tt_from_utc_jd",
    "jd_ut1_from_utc_jd",
    "civil_timescales",
]
