# app/core/panchanga_calendar.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Pañcāṅga Calendar / Almanac — daywise summaries (research-grade)

What this provides
- almanac_day(...): Panchāṅga for a single civil date at a site, sunrise→next sunrise
- almanac_for_range(...): multi-day generator/collector using almanac_day
- Helpers to present change times (Tithi/Nakṣatra/Yoga/Karaṇa) for that vday
- Optional Rahu Kālam / Yamaganda / Gulika segments from sunrise→sunset

Numerics
- Uses app.core.panchanga for elements + sunrise/sunset (ERFA-backed path)
- Uses app.core.panchanga_events for exact change times with Brent refinement
- Sidereal longitudes via EphemerisAdapter; ayanāṁśa via app.core.ayanamsa
- Times are returned as Julian Days; local ISO strings are added when conversion
  is available. For display, UT1≈UTC may be used with a warning if UTC isn’t resolved.

Public API
    almanac_day(date, tz, lat, lon, *, ayanamsa="lahiri", include_muhurta=False, include_rahukaalam=True) -> dict
    almanac_for_range(date_start, date_end, tz, lat, lon, *, ayanamsa="lahiri", include_muhurta=False, include_rahukaalam=True) -> dict
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
import math
import datetime as _dt

# Optional time helpers (consistent with other core modules)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

from app.core.panchanga import (
    panchanga_elements_at,
    sunrise_sunset_for_julian_day,
    muhurta_windows,
)
from app.core.panchanga_events import find_panchanga_changes
from app.core.ayanamsa import get_ayanamsa_deg


__all__ = ["almanac_day", "almanac_for_range"]


# ────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _wrap180(x: float) -> float:
    return ((float(x) + 180.0) % 360.0) - 180.0

def _iso_next(date: str) -> str:
    y, m, d = [int(x) for x in date.split("-")]
    return str((_dt.date(y, m, d) + _dt.timedelta(days=1)))

def _utc_from_tt_fallback(jd_tt: float, *, y: int, m: int) -> float:
    """
    Convert JD_TT→JD_UTC with monthly ΔT if available; otherwise use ~69 s fallback.
    JD_UTC = JD_TT - ΔT/86400.
    """
    if _ts and hasattr(_ts, "delta_t_seconds_for_month"):
        try:
            dt_sec = float(_ts.delta_t_seconds_for_month(y, m))
            return jd_tt - dt_sec / 86400.0
        except Exception:
            pass
    # crude fallback
    return jd_tt - 69.0 / 86400.0

def _ymd_from_date(date: str) -> Tuple[int, int, int]:
    y, m, d = [int(x) for x in date.split("-")]
    return y, m, d

def _jdutc_to_local_iso(jd_utc: float, tz: str) -> str | None:
    """
    Minimal JD(UTC) → local ISO string. If ZoneInfo missing, return None.
    """
    if ZoneInfo is None:
        return None
    # JD -> UTC datetime (algorithmic; valid for modern Gregorian)
    Z = int(jd_utc + 0.5)
    F = (jd_utc + 0.5) - Z
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
    frac = float(day - day_int)
    seconds = int(round(frac * 86400.0))
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    try:
        dt_utc = _dt.datetime(year, int(month), day_int, hh, mm, ss, tzinfo=_dt.timezone.utc)
        dt_loc = dt_utc.astimezone(ZoneInfo(tz))
        return dt_loc.isoformat(timespec="seconds")
    except Exception:
        return None

def _weekday_name(date: str, tz: str) -> str:
    if ZoneInfo is None:
        # Fallback: assume UTC
        wd = _dt.datetime.fromisoformat(f"{date}T12:00:00").weekday()  # Mon=0..Sun=6
    else:
        wd = _dt.datetime.fromisoformat(f"{date}T12:00:00").replace(tzinfo=ZoneInfo(tz)).weekday()
    return ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][wd]

def _weekday_index_sun0(date: str, tz: str) -> int:
    """Sunday=0 .. Saturday=6"""
    if ZoneInfo is None:
        wd = _dt.datetime.fromisoformat(f"{date}T12:00:00").weekday()  # Mon=0..Sun=6
    else:
        wd = _dt.datetime.fromisoformat(f"{date}T12:00:00").replace(tzinfo=ZoneInfo(tz)).weekday()
    return (wd + 1) % 7

# ────────────────────────────────────────────────────────────────────────
# Rahu Kālam / Yamaganda / Gulika (day parts)
# ────────────────────────────────────────────────────────────────────────

# Index (Sunday=0..Saturday=6) → 1-based part index (1..8) for daytime (sunrise→sunset)
_RAHU_PART =   [8, 2, 7, 5, 6, 4, 3]  # Sun, Mon, Tue, Wed, Thu, Fri, Sat
_YAMAG_PART =  [5, 3, 6, 2, 7, 1, 4]
_GULIKA_PART = [7, 6, 5, 4, 3, 2, 1]

def _segment_nth_of_eight(start_jd_ut1: float, end_jd_ut1: float, n: int) -> Tuple[float, float]:
    """Return the n-th (1..8) equal slice within [start,end)."""
    n = max(1, min(8, int(n)))
    span = float(end_jd_ut1 - start_jd_ut1)
    step = span / 8.0
    a = start_jd_ut1 + (n - 1) * step
    b = a + step
    return a, b

def _day_parts(sunrise_ut1: float, sunset_ut1: float, weekday_sun0: int) -> Dict[str, Tuple[float, float]]:
    if not (math.isfinite(sunrise_ut1) and math.isfinite(sunset_ut1)) or sunset_ut1 <= sunrise_ut1:
        return {}
    rah = _RAHU_PART[weekday_sun0]
    yam = _YAMAG_PART[weekday_sun0]
    gul = _GULIKA_PART[weekday_sun0]
    return {
        "rahukaalam": _segment_nth_of_eight(sunrise_ut1, sunset_ut1, rah),
        "yamaganda":  _segment_nth_of_eight(sunrise_ut1, sunset_ut1, yam),
        "gulika":     _segment_nth_of_eight(sunrise_ut1, sunset_ut1, gul),
    }

# ────────────────────────────────────────────────────────────────────────
# Core API
# ────────────────────────────────────────────────────────────────────────

def almanac_day(
    date: str,
    tz: str,
    lat: float,
    lon: float,
    *,
    elevation_m: float | None = None,
    ayanamsa: str = "lahiri",
    include_muhurta: bool = False,
    include_rahukaalam: bool = True,
) -> Dict[str, Any]:
    """
    Panchāṅga for one civil date at site. The 'vday' spans sunrise→next sunrise.
    Returns:
      {
        ok, date, tz, site, weekday, sunrise_sunset:{...},
        elements_at_sunrise:{...}, changes:{...}, timeline:{...},
        rahukaalam/yamaganda/gulika (if requested),
        warnings:[]
      }
    """
    site = {"latitude": float(lat), "longitude": float(lon), "elevation_m": elevation_m}
    sr = sunrise_sunset_for_julian_day(date=date, tz=tz, site=_SiteProxy(lat, lon, elevation_m), ayanamsa_key=ayanamsa)
    warnings = list(sr.get("warnings", []))
    sunrise_ut1 = sr.get("sunrise_jd_ut1")
    next_sunrise_ut1 = sr.get("next_sunrise_jd_ut1")
    sunset_ut1 = sr.get("sunset_jd_ut1")

    # Convert sunrise UT1 to an approximate UTC JD for display/labeling
    sunrise_utc = None
    next_sunrise_utc = None
    sunset_utc = None
    if sunrise_ut1 is not None:
        # Use TT reconstruction around date midnight, then adjust to UTC for display
        y, m, _ = _ymd_from_date(date)
        # We don't have TT at the exact sunrise here; for labeling, UT1≈UTC is fine
        sunrise_utc = float(sunrise_ut1)
    if next_sunrise_ut1 is not None:
        next_sunrise_utc = float(next_sunrise_ut1)
    if sunset_ut1 is not None:
        sunset_utc = float(sunset_ut1)

    # Elements at vday start (evaluate at sunrise TT; approximate from UTC if needed)
    # If we can derive TT from UTC JD, do so; otherwise use local noon TT as fallback.
    if _ts and isinstance(sunrise_utc, float):
        y, m, _ = _ymd_from_date(date)
        try:
            jd_tt_guess = float(_ts.jd_tt_from_utc_jd(sunrise_utc, y, m))
        except Exception:
            jd_tt_guess = sunrise_utc + 69.0/86400.0
    else:
        # fallback: local noon TT
        if _tk and hasattr(_tk, "timescales_from_civil"):
            ts = _tk.timescales_from_civil(date=date, time="12:00:00", tz=tz, dut1=0.0)  # type: ignore
            jd_tt_guess = float(ts["jd_tt"]) if isinstance(ts, dict) and "jd_tt" in ts else None
        else:
            jd_tt_guess = None

    elements_at_sunrise = {}
    if isinstance(jd_tt_guess, (int, float)):
        elements_at_sunrise = panchanga_elements_at(float(jd_tt_guess), ayanamsa_key=ayanamsa)

    # Panchāṅga changes within vday [sunrise, next_sunrise)
    changes = {}
    timeline = {}
    if isinstance(sunrise_ut1, float) and isinstance(next_sunrise_ut1, float):
        # For change times, use TT window; get TT via monthly ΔT approximation
        y0, m0, _ = _ymd_from_date(date)
        y1, m1, _ = _ymd_from_date(_iso_next(date))
        start_tt = float(_ts.jd_tt_from_utc_jd(sunrise_utc, y0, m0)) if (_ts and sunrise_utc is not None) else (sunrise_utc + 69.0/86400.0 if sunrise_utc else None)  # type: ignore
        end_tt   = float(_ts.jd_tt_from_utc_jd(next_sunrise_utc, y1, m1)) if (_ts and next_sunrise_utc is not None) else (next_sunrise_utc + 69.0/86400.0 if next_sunrise_utc else None)  # type: ignore
        if isinstance(start_tt, float) and isinstance(end_tt, float):
            changes = find_panchanga_changes(start_tt, end_tt, ayanamsa_key=ayanamsa, step_minutes="auto")
            # Timeline over the same window
            # We can derive timeline by folding changes; keep JD_TT outputs
            from app.core.panchanga_events import panchanga_timeline
            timeline = panchanga_timeline(start_tt, end_tt, ayanamsa_key=ayanamsa)

    # Rahu Kālam / Yamaganda / Gulika (daytime)
    extras = {}
    if include_rahukaalam and isinstance(sunrise_ut1, float) and isinstance(sunset_ut1, float):
        widx = _weekday_index_sun0(date, tz)
        parts = _day_parts(float(sunrise_ut1), float(sunset_ut1), widx)
        for k, (a, b) in parts.items():
            extras[k] = {
                "start_jd_ut1": float(a),
                "end_jd_ut1": float(b),
                "start_local": _jdutc_to_local_iso(a, tz),
                "end_local": _jdutc_to_local_iso(b, tz),
            }

    out = {
        "ok": bool(sunrise_ut1 is not None and next_sunrise_ut1 is not None),
        "date": date,
        "tz": tz,
        "site": site,
        "weekday": _weekday_name(date, tz),
        "sunrise_sunset": {
            "sunrise_jd_ut1": sunrise_ut1,
            "sunset_jd_ut1": sunset_ut1,
            "next_sunrise_jd_ut1": next_sunrise_ut1,
            "sunrise_local": _jdutc_to_local_iso(sunrise_utc, tz) if sunrise_utc else None,
            "sunset_local": _jdutc_to_local_iso(sunset_utc, tz) if sunset_utc else None,
            "next_sunrise_local": _jdutc_to_local_iso(next_sunrise_utc, tz) if next_sunrise_utc else None,
        },
        "elements_at_sunrise": elements_at_sunrise,
        "changes": changes,
        "timeline": timeline,
        **extras,
    }
    if warnings:
        out["warnings"] = warnings
    return out


def almanac_for_range(
    date_start: str,
    date_end: str,
    tz: str,
    lat: float,
    lon: float,
    *,
    elevation_m: float | None = None,
    ayanamsa: str = "lahiri",
    include_muhurta: bool = False,
    include_rahukaalam: bool = True,
) -> Dict[str, Any]:
    """
    Build an almanac for every civil date in [date_start, date_end] inclusive.
    """
    y0, m0, d0 = _ymd_from_date(date_start)
    y1, m1, d1 = _ymd_from_date(date_end)
    ds = _dt.date(y0, m0, d0)
    de = _dt.date(y1, m1, d1)
    if de < ds:
        ds, de = de, ds
    out_days: List[Dict[str, Any]] = []
    cur = ds
    while cur <= de:
        day = str(cur)
        rec = almanac_day(
            day, tz, lat, lon,
            elevation_m=elevation_m,
            ayanamsa=ayanamsa,
            include_muhurta=include_muhurta,
            include_rahukaalam=include_rahukaalam,
        )
        out_days.append(rec)
        cur = cur + _dt.timedelta(days=1)
    return {
        "ok": all(d.get("ok", False) for d in out_days),
        "ayanamsa": ayanamsa,
        "range": [date_start, date_end],
        "tz": tz,
        "site": {"latitude": float(lat), "longitude": float(lon), "elevation_m": elevation_m},
        "days": out_days,
    }


# ────────────────────────────────────────────────────────────────────────
# Internal light proxy for sunrise_sunset API (dataclass would be fine too)
# ────────────────────────────────────────────────────────────────────────

class _SiteProxy:
    def __init__(self, lat: float, lon: float, elevation_m: float | None):
        self.lat = float(lat)
        self.lon = float(lon)
        self.elev_m = elevation_m
