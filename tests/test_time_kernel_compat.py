# tests/test_time_kernel_compat.py
from __future__ import annotations
import math
import pytest

from app.core import time_kernel as tk
from app.core.timescales import build_timescales as ts_build

EPS = 1e-12  # tight JD tolerance

def _near(a: float, b: float, eps: float = EPS) -> bool:
    return abs(float(a) - float(b)) <= eps

def test_forwarder_parity_basic():
    a = tk.build_timescales("2020-06-01", "00:00:00", "UTC", 0.1)
    b = ts_build           ("2020-06-01", "00:00:00", "UTC", 0.1)
    assert a.keys() == b.keys()
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, float):
            assert _near(av, bv), f"{k} differs"
        else:
            assert av == bv, f"{k} differs"

def test_forwarder_parity_dst_fold():
    a = tk.build_timescales("2020-11-01", "01:30:00", "America/New_York", 0.0)
    b = ts_build           ("2020-11-01", "01:30:00", "America/New_York", 0.0)
    for k in ("jd_utc","jd_tt","jd_ut1","delta_t","dat","dut1","tz_offset_seconds","timezone"):
        assert _near(a[k], b[k]), f"{k} differs"
    assert a["warnings"] == b["warnings"]
    assert any("ambiguous" in w.lower() or "fold" in w.lower() for w in a["warnings"])

def test_helper_utc_jd_matches_engine():
    ts = ts_build("1999-12-31", "23:59:59.123456", "Asia/Kolkata", 0.05)
    jdu = tk.julian_day_utc("1999-12-31", "23:59:59.123456", "Asia/Kolkata")
    assert _near(jdu, ts["jd_utc"])

def test_helper_tt_from_utc_jd_matches_engine():
    ts = ts_build("2010-01-15", "06:45:12.5", "Europe/Berlin", 0.0)
    jdtt = tk.jd_tt_from_utc_jd(ts["jd_utc"])
    assert _near(jdtt, ts["jd_tt"])

def test_helper_ut1_from_utc_jd_matches_engine():
    ts = ts_build("2024-01-01", "12:00:00", "UTC", 0.08)
    jdut1 = tk.jd_ut1_from_utc_jd(ts["jd_utc"], 0.08)
    assert _near(jdut1, ts["jd_ut1"])
    assert abs((jdut1 - ts["jd_utc"]) * 86400.0 - 0.08) < 3e-5

def test_leap_second_helpers():
    ts = ts_build("2016-12-31", "23:59:60", "UTC", 0.0)
    jdu = tk.julian_day_utc("2016-12-31", "23:59:60", "UTC")
    assert _near(jdu, ts["jd_utc"])
    jdtt = tk.jd_tt_from_utc_jd(ts["jd_utc"])
    assert _near(jdtt, ts["jd_tt"])
    jdut1 = tk.jd_ut1_from_utc_jd(ts["jd_utc"], 0.0)
    assert _near(jdut1, ts["jd_ut1"])

def test_dut1_validation_raises():
    with pytest.raises(ValueError):
        _ = tk.jd_ut1_from_utc_jd(2459001.5, 0.9000001)
    with pytest.raises(ValueError):
        _ = tk.jd_ut1_from_utc_jd(2459001.5, -0.9000001)
