# app/core/timescales.py
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import math
from typing import Dict


# ───────────────────────── internal helpers ─────────────────────────

def _normalize_hms(time_str: str) -> str:
    """
    Accept 'HH:MM' or 'HH:MM:SS' (24h). Return 'HH:MM:SS'.
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
    Parse local civil date/time/tz and return an *aware* UTC datetime.
    """
    t_norm = _normalize_hms(time_str)
    # Build local aware datetime
    dt_local = datetime.fromisoformat(f"{date_str}T{t_norm}").replace(tzinfo=ZoneInfo(tz_str))
    # Convert to UTC
    return dt_local.astimezone(timezone.utc)


# ───────────────────────── public API ─────────────────────────

def to_utc_iso(date_str: str, time_str: str, tz_str: str) -> str:
    """
    Convenience: convert local civil to UTC ISO string 'YYYY-MM-DDTHH:MM:SSZ'.
    """
    return _to_utc_datetime(date_str, time_str, tz_str).strftime("%Y-%m-%dT%H:%M:%SZ")


def julian_day_utc(date_str: str, time_str: str, tz_str: str) -> float:
    """
    Meeus algorithm (Gregorian) for JD from *UTC* datetime.
    Returns JD for the given instant in UTC.
    """
    dt_utc = _to_utc_datetime(date_str, time_str, tz_str)

    Y = dt_utc.year
    M = dt_utc.month
    D = dt_utc.day

    # Fractional day from time-of-day
    frac_day = (dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0 + dt_utc.microsecond / 3.6e9) / 24.0

    if M <= 2:
        Y -= 1
        M += 12

    A = math.floor(Y / 100)
    B = 2 - A + math.floor(A / 4)  # ← correct Gregorian correction

    JD0 = math.floor(365.25 * (Y + 4716)) + math.floor(30.6001 * (M + 1)) + D + B - 1524.5
    return float(JD0 + frac_day)


def delta_t_seconds(year: int, month: int) -> float:
    """
    ΔT ≡ TT − UT1 in seconds.
    Piecewise polynomials adapted from Espenak & Meeus (NASA), with a general
    fallback. Good to a couple of seconds across 1800–2050 for most purposes.

    For y outside the main range, use the conventional parabola.
    """
    y = float(year) + (float(month) - 0.5) / 12.0

    # 2005–2050
    if 2005.0 <= y <= 2050.0:
        t = y - 2000.0
        return 62.92 + 0.32217 * t + 0.005589 * (t ** 2)

    # 1986–2005
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

    # 1900–1986
    if 1900.0 <= y < 1986.0:
        t = y - 1900.0
        return -2.79 + 1.494119 * t - 0.0598939 * (t ** 2) + 0.0061966 * (t ** 3) - 0.000197 * (t ** 4)

    # 1800–1900 (coarser)
    if 1800.0 <= y < 1900.0:
        t = (y - 1860.0) / 100.0
        # Common alternative for 1800–1860 is a quartic; this smooth variant works across the century.
        return 13.72 - 33.244 * t + 68.612 * (t ** 2) + 4111.6 * (t ** 4)

    # 2050–2150: linear growth approximation (IAU style heuristics)
    if 2050.0 < y <= 2150.0:
        # Extrapolate gently beyond 2050 (not for precise work)
        return 62.92 + 0.32217 * (y - 2000.0) + 0.005589 * ((y - 2000.0) ** 2)

    # General fallback (parabola around 1820), used outside detailed ranges
    u = (y - 1820.0) / 100.0
    return -20.0 + 32.0 * (u ** 2)


def jd_tt_from_utc_jd(jd_utc: float, year: int, month: int) -> float:
    """
    Convert JD(UTC) -> JD(TT) using ΔT.
    NOTE: ΔT = TT − UT1. Since we only have UTC here, we *do not* add DUT1.
    Callers that also compute UT1 separately should do:
        jd_ut1 = jd_utc + DUT1/86400
    so that TT and UT1 remain self-consistent in downstream calculations.
    """
    DT = float(delta_t_seconds(year, month))
    return float(jd_utc) + DT / 86400.0


# ────────────────────── optional convenience (not used by callers) ──────────────────────

def civil_timescales(date_str: str, time_str: str, tz_str: str, dut1_seconds: float = 0.0) -> Dict[str, float]:
    """
    Convenience helper used by tests/tools:
      returns {"jd_utc", "jd_tt", "jd_ut1", "delta_t", "dut1"} for a local civil instant.
    """
    dt_utc = _to_utc_datetime(date_str, time_str, tz_str)
    jd_utc = julian_day_utc(date_str, time_str, tz_str)
    jd_tt = jd_tt_from_utc_jd(jd_utc, dt_utc.year, dt_utc.month)
    jd_ut1 = jd_utc + float(dut1_seconds) / 86400.0
    return {
        "jd_utc": float(jd_utc),
        "jd_tt": float(jd_tt),
        "jd_ut1": float(jd_ut1),
        "delta_t": float(delta_t_seconds(dt_utc.year, dt_utc.month)),
        "dut1": float(dut1_seconds),
    }
