from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import math

def to_utc_iso(date_str: str, time_str: str, tz_str: str) -> str:
    dt_local = datetime.fromisoformat(f"{date_str}T{time_str}:00").replace(tzinfo=ZoneInfo(tz_str))
    return dt_local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

def julian_day_utc(date_str: str, time_str: str, tz_str: str) -> float:
    # Meeus algorithm for JD from UTC datetime
    dt_utc = datetime.fromisoformat(f"{date_str}T{time_str}:00").replace(tzinfo=ZoneInfo(tz_str)).astimezone(ZoneInfo("UTC"))
    Y = dt_utc.year; M = dt_utc.month; D = dt_utc.day
    h = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600
    if M <= 2:
        Y -= 1; M += 12
    A = Y // 100
    B = 2 - A + A // 5
    JD0 = int(365.25*(Y + 4716)) + int(30.6001*(M + 1)) + D + B - 1524.5
    return JD0 + h/24.0

def delta_T_seconds(year: int, month: int) -> float:
    # Simplified polynomial approximations (Espenak & Meeus), valid ~1800-2100
    y = year + (month - 0.5)/12.0
    if 2005 <= y <= 2050:
        t = y - 2000
        return 62.92 + 0.32217*t + 0.005589*t*t
    if 1986 <= y < 2005:
        t = y - 2000
        return 63.86 + 0.3345*t - 0.060374*t*t + 0.0017275*t**3 + 0.000651814*t**4 + 0.00002373599*t**5
    if 1900 <= y < 1986:
        t = y - 1900
        return -2.79 + 1.494119*t - 0.0598939*t*t + 0.0061966*t**3 - 0.000197*t**4
    if 1800 <= y < 1900:
        t = (y - 1900)/100
        return 13.72 - 33.244*t + 68.612*t**2 + 4111.6*t**4
    # Fallback rough
    return 69.0

def jd_tt_from_utc_jd(jd_utc: float, year: int, month: int) -> float:
    DT = delta_T_seconds(year, month)
    return jd_utc + DT / 86400.0
