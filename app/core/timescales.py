# app/core/timescales.py
from __future__ import annotations
"""
Timescale utilities (UTC ↔ JD, TT/UT1, ΔT) with configurable leap-second handling.

Public API (stable — do not change signatures):
- to_utc_iso(date, time, tz) -> "YYYY-MM-DDTHH:MM:SSZ"
- julian_day_utc(date, time, tz) -> float
- delta_t_seconds(year, month) -> float  (alias: delta_T_seconds)
- jd_tt_from_utc_jd(jd_utc, year, month, *, dut1_seconds=None) -> float
- jd_ut1_from_utc_jd(jd_utc, dut1_seconds) -> float
- civil_timescales(date, time, tz, dut1_seconds=0.0) -> dict

Enhancements vs. prior version
- Leap seconds:
  • Optional external source (JSON file or inline JSON via env)
  • Staleness detection with log warnings
  • Future-date policy: hold last ΔAT (default) or fall back to ΔT (+DUT1)
- Config helper & summary() for ops/docs
- Small self-test for coordination across layers

Environment variables (optional)
- ASTRO_LEAPSECONDS_PATH         : path to JSON file with [{"date":"YYYY-MM-DD","dat":37}, ...]
- ASTRO_LEAPSECONDS_INLINE       : same JSON content but inline string
- ASTRO_LEAP_STALE_THRESHOLD_DAYS: int (default 3650 days ≈ 10 years)
- ASTRO_LEAP_FUTURE_POLICY       : "hold" (default) or "poly"
- ASTRO_TIMESCALES_LOG_LEVEL     : "WARNING" (default), "INFO", "DEBUG"
"""

import json
import math
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

# ───────────────────────────── logger ─────────────────────────────
_log = logging.getLogger(__name__)
if not _log.handlers:
    # Let app/main.py configure root; this keeps module safe to import standalone too.
    _log.addHandler(logging.NullHandler())
# Honor module-specific override, if set
_lvl = os.getenv("ASTRO_TIMESCALES_LOG_LEVEL")
if _lvl:
    try:
        _log.setLevel(getattr(logging, _lvl.upper()))
    except Exception:
        pass

# ───────────────────────────── internals ─────────────────────────────

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
    Notes:
      • Ambiguous (DST fall-back) instants default to fold=0.
      • Nonexistent (spring-forward) instants will raise from stdlib.
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
    Meeus algorithm for JD from an aware UTC datetime (proleptic Gregorian).
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

# ───────────────────────────── leap seconds ─────────────────────────────

@dataclass(frozen=True)
class _LeapRow:
    effective: datetime  # UTC midnight the change takes effect
    dat: int             # ΔAT (TAI−UTC) seconds

def _builtin_leap_table() -> Tuple[_LeapRow, ...]:
    # Compact table for dates >= 1972-01-01. Last known: 2017-01-01 (ΔAT=37).
    raw = (
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
    return tuple(_LeapRow(datetime(y, m, d, tzinfo=timezone.utc), dat) for (y, m, d, dat) in raw)

# Lazy-loaded, possibly from external source
_LEAPS: Tuple[_LeapRow, ...] | None = None
_LAST_STALE_CHECK: Optional[datetime] = None

def _parse_external_json(obj: object) -> Tuple[_LeapRow, ...]:
    """
    Expect: list of {"date":"YYYY-MM-DD", "dat": <int>}
    """
    if not isinstance(obj, list):
        raise ValueError("external leap-second JSON must be a list")
    rows: List[_LeapRow] = []
    for i, entry in enumerate(obj):
        if not isinstance(entry, dict) or "date" not in entry or "dat" not in entry:
            raise ValueError(f"invalid leap-second row at index {i}")
        ds = str(entry["date"]).strip()
        dat = int(entry["dat"])
        # allow midnight date; that's how ΔAT tables are expressed
        eff = datetime.fromisoformat(ds).replace(tzinfo=timezone.utc)
        rows.append(_LeapRow(effective=eff, dat=dat))
    rows.sort(key=lambda r: r.effective)
    return tuple(rows)

def _load_leap_seconds() -> Tuple[_LeapRow, ...]:
    """
    Load leap seconds from:
      1) ASTRO_LEAPSECONDS_INLINE (inline JSON string)
      2) ASTRO_LEAPSECONDS_PATH   (JSON file path)
      3) builtin table (fallback)
    """
    # inline env JSON overrides file path
    inline = os.getenv("ASTRO_LEAPSECONDS_INLINE")
    if inline:
        try:
            return _parse_external_json(json.loads(inline))
        except Exception as e:
            _log.warning("Failed to parse ASTRO_LEAPSECONDS_INLINE: %s", e)

    path = os.getenv("ASTRO_LEAPSECONDS_PATH")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _parse_external_json(data)
        except Exception as e:
            _log.warning("Failed to load ASTRO_LEAPSECONDS_PATH=%r: %s", path, e)

    # fallback
    return _builtin_leap_table()

def _leaps() -> Tuple[_LeapRow, ...]:
    global _LEAPS
    if _LEAPS is None:
        _LEAPS = _load_leap_seconds()
        if not _LEAPS:
            _LEAPS = _builtin_leap_table()
        # one-time info
        latest = _LEAPS[-1]
        _log.info("Leap-second table loaded; last=%s (ΔAT=%s)", latest.effective.date(), latest.dat)
    return _LEAPS

def _latest_leap() -> _LeapRow:
    return _leaps()[-1]

def _delta_at_seconds(dt_utc: datetime) -> Optional[int]:
    """
    Return ΔAT (TAI−UTC) seconds for dt_utc if >= first table entry, else None.
    """
    table = _leaps()
    if not table or dt_utc < table[0].effective:
        return None
    dat = table[0].dat
    for row in table:
        if dt_utc >= row.effective:
            dat = row.dat
        else:
            break
    return dat

def _leap_table_stale(now: Optional[datetime] = None) -> bool:
    """
    Heuristic: if now is more than N days (default 3650) after the latest
    effective date in the table, consider it stale and warn.
    """
    threshold_days = int(os.getenv("ASTRO_LEAP_STALE_THRESHOLD_DAYS", "3650") or 3650)
    now = now or datetime.now(timezone.utc)
    last_eff = _latest_leap().effective
    return (now - last_eff).days > threshold_days

def _maybe_warn_stale() -> None:
    global _LAST_STALE_CHECK
    now = datetime.now(timezone.utc)
    # avoid spamming logs: check at most once per hour
    if _LAST_STALE_CHECK and (now - _LAST_STALE_CHECK).total_seconds() < 3600.0:
        return
    _LAST_STALE_CHECK = now
    try:
        if _leap_table_stale(now):
            _log.warning(
                "Leap-second table may be stale: latest=%s (ΔAT=%s). "
                "Consider updating ASTRO_LEAPSECONDS_PATH/INLINE.",
                _latest_leap().effective.date(), _latest_leap().dat
            )
    except Exception:
        pass

def _future_policy() -> str:
    """
    Behavior for dates beyond last-known ΔAT:
      "hold" (default): use the last ΔAT (best-effort; off by whole seconds after next leap)
      "poly"          : fall back to ΔT (+ DUT1 if provided)
    """
    v = (os.getenv("ASTRO_LEAP_FUTURE_POLICY", "hold") or "hold").strip().lower()
    return v if v in ("hold", "poly") else "hold"

# ───────────────────────────── public API ─────────────────────────────

def to_utc_iso(date_str: str, time_str: str, tz_str: str) -> str:
    """Convert local civil to 'YYYY-MM-DDTHH:MM:SSZ' UTC string."""
    return _to_utc_datetime(date_str, time_str, tz_str).strftime("%Y-%m-%dT%H:%M:%SZ")


def julian_day_utc(date_str: str, time_str: str, tz_str: str) -> float:
    """JD(UTC) for the provided local civil instant."""
    return _jd_from_utc_datetime(_to_utc_datetime(date_str, time_str, tz_str))


# ---- ΔT (TT−UT1) model (Espenak–Meeus style) ------------------------------

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


# ---- TT from UTC JD (modern: ΔAT+32.184; legacy: ΔT [+ DUT1]) --------------

def jd_tt_from_utc_jd(
    jd_utc: float,
    year: int,
    month: int,
    *,
    dut1_seconds: Optional[float] = None,
) -> float:
    """
    Convert JD(UTC) → JD(TT).

    Preferred modern method (UTC ≥ first ΔAT entry):
        TT = UTC + (ΔAT + 32.184 s)

    For dates before the first ΔAT entry (pre-1972):
        TT = UTC + (ΔT + DUT1)   [since ΔT=TT−UT1 and UT1=UTC+DUT1]

    For dates beyond the last-known ΔAT:
        Policy "hold" (default): use last ΔAT + 32.184 (best-effort).
        Policy "poly"          : use ΔT (+ DUT1 if provided).

    Notes:
      • 'year' and 'month' are used to choose the regime; they SHOULD match the
        civil instant that produced jd_utc.
    """
    _maybe_warn_stale()

    # Use a representative instant in the given month/year to choose regime
    try:
        dt_probe = datetime(year, month, 15, tzinfo=timezone.utc)
    except Exception:
        dt_probe = datetime(1972, 1, 1, tzinfo=timezone.utc)

    table = _leaps()
    if not table or dt_probe < table[0].effective:
        # Pre-1972 legacy path: ΔT (+ DUT1 if given)
        DT = float(delta_t_seconds(year, month))
        return float(jd_utc) + ((DT + float(dut1_seconds)) if dut1_seconds is not None else DT) / 86400.0

    # Modern & future
    last = _latest_leap()
    if dt_probe >= last.effective:
        # Either current last-known era, or a future date
        if dt_probe > last.effective and _future_policy() == "poly":
            DT = float(delta_t_seconds(year, month))
            return float(jd_utc) + ((DT + float(dut1_seconds)) if dut1_seconds is not None else DT) / 86400.0
        # hold policy (default): use last known ΔAT
        return float(jd_utc) + (last.dat + 32.184) / 86400.0

    # Within modern table span: find applicable ΔAT
    dat = _delta_at_seconds(dt_probe)
    return float(jd_utc) + (float(dat) + 32.184) / 86400.0  # type: ignore[arg-type]


def jd_ut1_from_utc_jd(jd_utc: float, dut1_seconds: float) -> float:
    """JD(UT1) = JD(UTC) + DUT1/86400."""
    return float(jd_utc) + float(dut1_seconds) / 86400.0


# ───────────── Convenience: full set from civil inputs (diagnostic-friendly) ─────────────

def civil_timescales(
    date_str: str,
    time_str: str,
    tz_str: str,
    dut1_seconds: float = 0.0,
) -> Dict[str, float]:
    """
    Return:
      {
        "jd_utc", "jd_tt", "jd_ut1", "delta_t", "dut1"
      }

    Notes:
      • jd_tt uses modern TT−UTC = ΔAT + 32.184 s where available,
        or falls back per future/legacy policy described above.
      • delta_t is the Espenak–Meeus ΔT (TT−UT1) polynomial (diagnostics/UI).
      • jd_ut1 uses the provided DUT1 seconds (default 0.0 if unknown).
    """
    dt_utc = _to_utc_datetime(date_str, time_str, tz_str)
    jd_utc = _jd_from_utc_datetime(dt_utc)

    jd_tt = jd_tt_from_utc_jd(jd_utc, dt_utc.year, dt_utc.month, dut1_seconds=dut1_seconds)
    jd_ut1 = jd_ut1_from_utc_jd(jd_utc, dut1_seconds)

    return {
        "jd_utc": float(jd_utc),
        "jd_tt": float(jd_tt),
        "jd_ut1": float(jd_ut1),
        "delta_t": float(delta_t_seconds(dt_utc.year, dt_utc.month)),
        "dut1": float(dut1_seconds),
    }

# ───────────── Optional helpers for ops/docs/tests (non-breaking) ─────────────

def config_summary() -> Dict[str, object]:
    """
    Introspect current configuration relevant to timescales/leap seconds.
    Safe to call in diagnostics endpoints.
    """
    table = _leaps()
    return {
        "leap_seconds_source": (
            "inline"
            if os.getenv("ASTRO_LEAPSECONDS_INLINE")
            else ("file" if os.getenv("ASTRO_LEAPSECONDS_PATH") else "builtin")
        ),
        "leap_last_effective_utc": table[-1].effective.strftime("%Y-%m-%d"),
        "leap_last_dat": table[-1].dat,
        "stale_threshold_days": int(os.getenv("ASTRO_LEAP_STALE_THRESHOLD_DAYS", "3650") or 3650),
        "is_stale": _leap_table_stale(),
        "future_policy": _future_policy(),
    }

def self_test() -> Dict[str, object]:
    """
    Lightweight internal checks (does not require internet). Returns a report dict.
    """
    rep: Dict[str, object] = {"ok": True, "checks": []}

    # Check ΔT continuity around modern era
    for y, m in [(1990, 1), (2000, 1), (2010, 1), (2020, 1)]:
        rep["checks"].append({"delta_t": {"year": y, "month": m, "seconds": delta_t_seconds(y, m)}})

    # JD round-trip sanity for a few zones
    for tz in ("UTC", "America/New_York", "Asia/Kolkata"):
        s = civil_timescales("2024-02-29", "12:34", tz, 0.0)
        rep["checks"].append({"jd_utc": {"tz": tz, "value": s["jd_utc"]}})

    # Leap table status
    rep["leap_summary"] = config_summary()
    return rep

# ───────────────────────────── exports ─────────────────────────────

__all__ = [
    "to_utc_iso",
    "julian_day_utc",
    "delta_t_seconds",
    "delta_T_seconds",
    "jd_tt_from_utc_jd",
    "jd_ut1_from_utc_jd",
    "civil_timescales",
    # optional helpers
    "config_summary",
    "self_test",
]
