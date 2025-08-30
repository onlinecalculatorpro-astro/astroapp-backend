*** a/app/api/helpers.py
--- b/app/api/helpers.py
@@
 from __future__ import annotations
-from typing import Any, Dict, Tuple
+from typing import Any, Dict, Tuple, Optional
+from dataclasses import asdict
+from datetime import datetime
+from zoneinfo import ZoneInfo
 
-from app.core.validators import parse_date, parse_time, ValidationError
+from app.core.validators import parse_date, parse_time, ValidationError
+from app.core import time_kernel as _tk
 
+# ---- Timescale helpers ------------------------------------------------------
+# We keep a very small, stable surface here so routes don't depend on any
+# internals of time_kernel. If time_kernel exposes a class or different
+# function names, we detect them and adapt at runtime.
+
+def _find_kernel_fn():
+    """
+    Locate a JD_UTC -> timescales function on app.core.time_kernel.
+    Returns a callable taking (jd_utc: float) and returning either a dict-like
+    or dataclass with attributes: jd_tt, jd_ut1, delta_t, delta_at, dut1, warnings (opt).
+    """
+    candidate_names = (
+        "utc_jd_to_timescales",
+        "jd_utc_to_timescales",
+        "timescales_from_jd_utc",
+        "compute_from_jd_utc",
+        "derive_timescales",
+    )
+    for name in candidate_names:
+        fn = getattr(_tk, name, None)
+        if callable(fn):
+            return fn
+    # Class-based fallback: TimeKernel().from_jd_utc(...)
+    TK = getattr(_tk, "TimeKernel", None)
+    if TK is not None:
+        inst = TK()  # type: ignore
+        for name in ("from_jd_utc", "utc_jd_to_timescales"):
+            fn = getattr(inst, name, None)
+            if callable(fn):
+                return fn
+    raise RuntimeError("time_kernel: no JD_UTC→timescales function found")
+
+_JD_TO_TS = _find_kernel_fn()
+
+def _datetime_to_jd_utc(dt_utc: datetime) -> float:
+    """
+    Convert a UTC-aware datetime to Julian Day (UTC).
+    Implemented locally to avoid coupling routes to time_kernel internals.
+    Valid for Gregorian dates (which we always use).
+    """
+    if dt_utc.tzinfo is None:
+        raise ValueError("dt_utc must be timezone-aware (UTC)")
+    # Algorithm from Meeus, with microsecond precision.
+    y = dt_utc.year
+    m = dt_utc.month
+    d = dt_utc.day + (
+        dt_utc.hour + (dt_utc.minute + (dt_utc.second + dt_utc.microsecond/1e6)/60.0)/60.0
+    )/24.0
+    if m <= 2:
+        y -= 1
+        m += 12
+    A = y // 100
+    B = 2 - A + (A // 25)
+    jd = int(365.25*(y + 4716)) + int(30.6001*(m + 1)) + d + B - 1524.5
+    return float(jd)
+
+def compute_timescales_from_payload(data: Dict[str, Any]) -> Dict[str, Any]:
+    """
+    Given request payload containing 'date' (YYYY-MM-DD), 'time' (HH:MM[:SS]),
+    and 'timezone' (IANA), compute jd_utc, jd_tt, jd_ut1, deltas, etc.
+    Returns a plain dict for JSON friendliness.
+    """
+    # Parse strict date/time (raises ValidationError on failure)
+    d = parse_date(str(data.get("date", "")))
+    t = parse_time(str(data.get("time", "")))
+    tz = str(data.get("timezone", "UTC")).strip() or "UTC"
+
+    # Local -> UTC
+    try:
+        tzinfo = ZoneInfo(tz)
+    except Exception:
+        raise ValidationError("timezone must be an IANA zone like 'Asia/Kolkata'")
+    local_dt = datetime.combine(d, t).replace(tzinfo=tzinfo)
+    dt_utc = local_dt.astimezone(ZoneInfo("UTC"))
+
+    jd_utc = _datetime_to_jd_utc(dt_utc)
+    ts_obj = _JD_TO_TS(jd_utc)  # dict or dataclass
+
+    # Make a dict regardless of return type
+    if hasattr(ts_obj, "__dict__"):
+        ts = dict(ts_obj.__dict__)
+    elif "asdict" in dir(ts_obj):  # unlikely, but handle dataclass instance with method
+        ts = asdict(ts_obj)  # type: ignore
+    elif isinstance(ts_obj, dict):
+        ts = dict(ts_obj)
+    else:
+        # Try attribute access fallbacks
+        def _get(name, default=None):
+            return getattr(ts_obj, name, default)
+        ts = {
+            "jd_tt": _get("jd_tt"),
+            "jd_ut1": _get("jd_ut1"),
+            "delta_t": _get("delta_t"),
+            "delta_at": _get("delta_at"),
+            "dut1": _get("dut1"),
+            "warnings": _get("warnings", []),
+        }
+
+    tz_offset_seconds = int(local_dt.utcoffset().total_seconds()) if local_dt.utcoffset() else 0
+    ts_out = {
+        "jd_utc": jd_utc,
+        "jd_tt": float(ts.get("jd_tt")),
+        "jd_ut1": float(ts.get("jd_ut1")),
+        "delta_t": float(ts.get("delta_t")),
+        "delta_at": float(ts.get("delta_at")),
+        "dut1": float(ts.get("dut1")),
+        "timezone": tz,
+        "tz_offset_seconds": tz_offset_seconds,
+        "warnings": ts.get("warnings", []) or [],
+    }
+    # Optional policy echo if provided by kernel (e.g., ASTRO_LEAP_POLICY, DUT1 broadcast)
+    if "policy" in ts:
+        ts_out["policy"] = ts["policy"]
+    return ts_out
