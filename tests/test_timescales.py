# tests/test_timescales.py
from __future__ import annotations

import math
import pytest

from datetime import date, timedelta
from app.core.timescales import build_timescales

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
TZS = [
    "UTC",
    "Asia/Kolkata",         # +05:30 no DST
    "Australia/Eucla",      # +08:45 quarter-hour
    "America/New_York",     # DST region
    "Europe/Berlin",        # DST Europe
    "America/St_Johns",     # -03:30
    "Pacific/Kiritimati",   # +14:00 extreme positive
    "Pacific/Pago_Pago",    # -11:00 extreme negative
    "Australia/Lord_Howe",  # +10:30/+11:00 odd DST
    "Africa/Cairo",         # DST-flip history
    "Europe/Moscow",        # no DST currently
]

def _keys_ok(ts: dict) -> None:
    # Canonical fields as per app/core/timescales.py (module return)
    # Endpoint maps dat→delta_at, but module uses "dat".
    for k in ["jd_utc", "jd_tt", "jd_ut1", "delta_t", "dat",
              "dut1", "tz_offset_seconds", "timezone", "warnings", "precision"]:
        assert k in ts, f"missing key: {k}"

    assert isinstance(ts["jd_utc"], float)
    assert isinstance(ts["jd_tt"], float)
    assert isinstance(ts["jd_ut1"], float)
    assert isinstance(ts["delta_t"], float)
    assert isinstance(ts["dat"], (float, int))
    assert isinstance(ts["dut1"], (float, int))
    assert isinstance(ts["tz_offset_seconds"], int)
    assert isinstance(ts["timezone"], str)
    assert isinstance(ts["warnings"], list)
    for w in ts["warnings"]:
        assert isinstance(w, str)
    assert isinstance(ts["precision"], dict)  # metadata bag


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_schema_and_types() -> None:
    ts = build_timescales("2020-06-01", "00:00:00", "UTC", 0.1)
    _keys_ok(ts)

def test_dut1_policy_accept_bounds() -> None:
    _ = build_timescales("2024-01-01", "12:00:00", "UTC", -0.9)
    _ = build_timescales("2024-01-01", "12:00:00", "UTC", 0.9)

def test_dut1_policy_reject_out_of_bounds() -> None:
    with pytest.raises(Exception):
        _ = build_timescales("2024-01-01", "12:00:00", "UTC", -0.9000001)
    with pytest.raises(Exception):
        _ = build_timescales("2024-01-01", "12:00:00", "UTC", 0.9000001)

def test_delta_t_monotonic_non_decreasing_daily() -> None:
    # Check ΔT(t+1d) >= ΔT(t) - tiny epsilon (ΔT ~ smooth over days)
    d0 = date(2020, 6, 1)
    d1 = d0 + timedelta(days=1)
    ts0 = build_timescales(d0.isoformat(), "00:00:00", "UTC", 0.0)
    ts1 = build_timescales(d1.isoformat(), "00:00:00", "UTC", 0.0)
    _keys_ok(ts0); _keys_ok(ts1)

    dt0 = ts0["delta_t"]  # seconds
    dt1 = ts1["delta_t"]
    # Allow minute numerical jitter
    assert dt1 + 1e-6 >= dt0

def test_repeatability_same_inputs() -> None:
    a = build_timescales("1999-12-31", "23:59:59.123456", "Asia/Kolkata", 0.05)
    b = build_timescales("1999-12-31", "23:59:59.123456", "Asia/Kolkata", 0.05)
    for k in ["jd_utc", "jd_tt", "jd_ut1", "delta_t", "dat", "dut1"]:
        assert abs(a[k] - b[k]) <= 1e-12, f"{k} not repeatable"
    assert a["tz_offset_seconds"] == b["tz_offset_seconds"]
    assert a["timezone"] == b["timezone"]
    assert a["warnings"] == b["warnings"]

def test_dst_ambiguity_warning_new_york_fall_back() -> None:
    # 2020-11-01 01:30 occurs twice; library should warn and choose fold=0.
    ts = build_timescales("2020-11-01", "01:30:00", "America/New_York", 0.0)
    _keys_ok(ts)
    assert any("ambiguous" in w.lower() or "fold" in w.lower() for w in ts["warnings"])

def test_leap_second_2016_12_31_23_59_60_utc() -> None:
    ts = build_timescales("2016-12-31", "23:59:60", "UTC", 0.0)
    _keys_ok(ts)
    # Known JD for the leap second instant (approx, ERFA exact chain)
    assert abs(ts["jd_utc"] - 2457754.499988426) < 1e-6
    assert ts["dat"] > 0.0

def test_microsecond_clamp_warning() -> None:
    # Provide >6 fractional digits; expect a clamp warning.
    ts = build_timescales("2024-03-10", "12:34:56.123456789", "UTC", 0.0)
    _keys_ok(ts)
    assert any("microsecond" in w.lower() for w in ts["warnings"])

def test_delta_at_positive_since_1960() -> None:
    ts = build_timescales("1975-07-01", "00:00:00", "UTC", 0.0)
    _keys_ok(ts)
    assert ts["dat"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Property / fuzz test (Hypothesis)
# ─────────────────────────────────────────────────────────────────────────────
from hypothesis import given, strategies as st

def _time_str(h: int, m: int, s: float) -> str:
    # Ensure we don't generate :60 here (reserved for leap-second unit test).
    # Keep <= 6 decimals to avoid clamp unless we want it.
    if s >= 60.0:
        s = math.nextafter(60.0, 0.0)
    return f"{h:02d}:{m:02d}:{s:09.6f}".rstrip("0").rstrip(".")

@given(
    y=st.integers(min_value=1960, max_value=2100),
    m=st.integers(min_value=1, max_value=12),
    d=st.integers(min_value=1, max_value=28),  # safe day to avoid month-end edge noise
    hh=st.integers(min_value=0, max_value=23),
    mm=st.integers(min_value=0, max_value=59),
    ss=st.floats(min_value=0.0, max_value=59.999999, allow_nan=False, allow_infinity=False),
    tz=st.sampled_from(TZS),
    dut1=st.floats(min_value=-0.9, max_value=0.9, allow_nan=False, allow_infinity=False),
)
def test_ut1_minus_utc_matches_dut1(y, m, d, hh, mm, ss, tz, dut1) -> None:
    datestr = f"{y:04d}-{m:02d}-{d:02d}"
    timestr = _time_str(hh, mm, ss)
    ts = build_timescales(datestr, timestr, tz, dut1)
    _keys_ok(ts)
    # Invariant: (UT1 - UTC) == DUT1  (in seconds), within 3e-5 tolerance
    ut1_minus_utc_sec = (ts["jd_ut1"] - ts["jd_utc"]) * 86400.0
    assert abs(ut1_minus_utc_sec - dut1) < 3e-5
