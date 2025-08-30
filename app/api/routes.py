# app/api/routes.py
from __future__ import annotations

import json
import os
import inspect
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, Callable, Optional

from flask import Blueprint, jsonify, request

from app.version import VERSION
from app.utils.metrics import metrics
from app.utils.ratelimit import rate_limit
from app.utils.hc import flag_predictions
from app.utils.config import load_config

from app.core.validators import (
    ValidationError,
    parse_chart_payload,
    parse_prediction_payload,
    parse_rectification_payload,
)

# Prefer the policy façade if available
try:
    from app.core.house import compute_houses_with_policy as _houses_fn  # type: ignore
    _HOUSES_KIND = "policy"
except Exception:
    from app.core.astronomy import compute_houses as _houses_fn  # type: ignore
    _HOUSES_KIND = "legacy"

from app.core.astronomy import compute_chart  # still the canonical chart call

# Optional helpers (best-effort — code is resilient if absent)
try:
    from app.core import leapseconds as _leaps
except Exception:  # pragma: no cover
    _leaps = None  # type: ignore

# Time kernel is required (but we adapt to several surfaces)
from app.core import time_kernel as _tk  # raises if missing

api = Blueprint("api", __name__)

DEBUG_VERBOSE = bool(int(os.environ.get("ASTRO_DEBUG_VERBOSE", "0")))


# ------------------------------------------------------------------------------
# Timescale plumbing
# ------------------------------------------------------------------------------

def _datetime_to_jd_utc(dt_utc: datetime) -> float:
    """Meeus-style Gregorian JD from an aware UTC datetime."""
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware UTC")
    y = dt_utc.year
    m = dt_utc.month
    d = dt_utc.day + (
        dt_utc.hour + (dt_utc.minute + (dt_utc.second + dt_utc.microsecond / 1e6) / 60.0) / 60.0
    ) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + (A // 25)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return float(jd)


def _find_kernel_callable() -> Callable[[float], Dict[str, Any]]:
    """
    Locate a JD_UTC -> timescales callable on app.core.time_kernel.
    Returns a function taking (jd_utc: float) and returning dict-like with:
    jd_tt, jd_ut1, delta_t, delta_at, dut1, warnings?, policy?
    """
    candidates = (
        "utc_jd_to_timescales",
        "jd_utc_to_timescales",
        "timescales_from_jd_utc",
        "compute_from_jd_utc",
        "derive_timescales",
    )
    for name in candidates:
        fn = getattr(_tk, name, None)
        if callable(fn):
            def _wrap(jd_utc: float, fn=fn):
                out = fn(jd_utc)
                if is_dataclass(out):
                    return asdict(out)  # type: ignore
                if hasattr(out, "__dict__"):
                    return dict(out.__dict__)
                return dict(out)
            return _wrap

    # Class API fallback: TimeKernel().from_jd_utc(...)
    TK = getattr(_tk, "TimeKernel", None)
    if TK is not None:
        inst = TK()  # type: ignore
        for name in ("from_jd_utc", "utc_jd_to_timescales"):
            fn = getattr(inst, name, None)
            if callable(fn):
                def _wrap2(jd_utc: float, fn=fn):
                    out = fn(jd_utc)
                    if is_dataclass(out):
                        return asdict(out)  # type: ignore
                    if hasattr(out, "__dict__"):
                        return dict(out.__dict__)
                    return dict(out)
                return _wrap2

    raise RuntimeError("time_kernel: no JD_UTC→timescales function found")


_JD_TO_TS = _find_kernel_callable()


def _compute_timescales_from_local(date_str: str, time_str: str, tz_name: str) -> Dict[str, Any]:
    """
    Convert local date/time/tz to jd_utc, then expand to jd_tt/jd_ut1 with time_kernel.
    Returns a dict safe for JSON.
    """
    # Robust parsing: allow “HH:MM[:SS]”
    if len(time_str.split(":")) == 2:
        fmt = "%Y-%m-%d %H:%M"
    else:
        fmt = "%Y-%m-%d %H:%M:%S"

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        raise ValidationError({"timezone": "must be a valid IANA zone like 'Asia/Kolkata'"})

    dt_local = datetime.strptime(f"{date_str} {time_str}", fmt).replace(tzinfo=tz)
    dt_utc = dt_local.astimezone(timezone.utc)
    jd_utc = _datetime_to_jd_utc(dt_utc)

    ts = _JD_TO_TS(jd_utc)  # may add warnings/policy
    tz_offset_seconds = int(dt_local.utcoffset().total_seconds()) if dt_local.utcoffset() else 0

    out = {
        "jd_utc": float(jd_utc),
        "jd_tt": float(ts.get("jd_tt")),
        "jd_ut1": float(ts.get("jd_ut1")),
        "delta_t": float(ts.get("delta_t")),
        "delta_at": float(ts.get("delta_at")),
        "dut1": float(ts.get("dut1")),
        "timezone": tz_name,
        "tz_offset_seconds": tz_offset_seconds,
        "warnings": ts.get("warnings", []) or [],
    }
    if "policy" in ts:
        out["policy"] = ts["policy"]
    return out


# ------------------------------------------------------------------------------
# Engine call adapters (signature-safe)
# ------------------------------------------------------------------------------

def _sig_accepts(fn: Callable, *names: str) -> Dict[str, bool]:
    """Map of param -> accepted by fn."""
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return {n: False for n in names}
    return {n: (n in params) for n in names}


def _call_compute_chart(payload: Dict[str, Any], ts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call compute_chart with signature introspection.
    Supplies jd_ut / jd_tt / jd_ut1 when supported.
    Ensures returned dict has jd_ut.
    """
    accepts = _sig_accepts(
        compute_chart, "date", "time", "latitude", "longitude", "mode", "place_tz",
        "jd_ut", "jd_tt", "jd_ut1"
    )

    kwargs: Dict[str, Any] = {}
    for key in ("date", "time", "latitude", "longitude", "mode"):
        if accepts.get(key):
            kwargs[key] = payload[key]
    # tz field name may vary between payloads
    tz_key = "place_tz" if accepts.get("place_tz") else None
    if tz_key:
        kwargs[tz_key] = payload.get("timezone") or payload.get("place_tz")

    # Pass timescales if supported
    if accepts.get("jd_ut"):
        kwargs["jd_ut"] = ts["jd_utc"]
    if accepts.get("jd_tt"):
        kwargs["jd_tt"] = ts["jd_tt"]
    if accepts.get("jd_ut1"):
        kwargs["jd_ut1"] = ts["jd_ut1"]

    chart = compute_chart(**kwargs)

    # Guarantee jd_ut for downstream code
    if "jd_ut" not in chart:
        chart["jd_ut"] = ts["jd_utc"]

    # Make mode explicit if engine didn’t echo it
    if "mode" not in chart and "mode" in payload:
        chart["mode"] = payload["mode"]

    return chart


def _call_compute_houses(payload: Dict[str, Any], ts: Dict[str, Any]) -> Any:
    """
    Prefer policy façade if available; pass jd_tt/jd_ut1 when accepted.
    Fallback to legacy compute_houses(lat, lon, mode[, jd_ut]).
    """
    lat = float(payload["latitude"])
    lon = float(payload["longitude"])

    accepts = _sig_accepts(_houses_fn, "lat", "lon", "system", "mode", "jd_ut", "jd_tt", "jd_ut1", "diagnostics")

    kwargs: Dict[str, Any] = {}
    # Param naming differs (system vs mode)
    if accepts.get("system"):
        kwargs["system"] = payload["mode"]
    elif accepts.get("mode"):
        kwargs["mode"] = payload["mode"]

    kwargs["lat"] = lat
    kwargs["lon"] = lon

    # Timescales (prefer jd_tt/jd_ut1; legacy may accept jd_ut)
    if accepts.get("jd_tt"):
        kwargs["jd_tt"] = ts["jd_tt"]
    if accepts.get("jd_ut1"):
        kwargs["jd_ut1"] = ts["jd_ut1"]
    if accepts.get("jd_ut") and "jd_tt" not in kwargs and "jd_ut1" not in kwargs:
        kwargs["jd_ut"] = ts["jd_utc"]

    # Diagnostics passthrough if requested by client
    if accepts.get("diagnostics") and "diagnostics" in payload:
        kwargs["diagnostics"] = bool(payload["diagnostics"])

    return _houses_fn(**kwargs)


# ------------------------------------------------------------------------------
# Error helpers
# ------------------------------------------------------------------------------

def _json_error(code: str, details: Any = None, http: int = 400):
    out: Dict[str, Any] = {"ok": False, "error": code}
    if details is not None:
        out["details"] = details
    return jsonify(out), http


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@api.get("/api/health")
def health():
    return jsonify({"ok": True, "status": "up", "version": VERSION}), 200


@api.post("/api/calculate")
def calculate():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        if DEBUG_VERBOSE:
            return _json_error("bad_request", str(e), 400)
        return _json_error("bad_request", None, 400)

    # Build timescales from payload
    tz_name = payload.get("timezone") or payload.get("place_tz") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name)

    # Compute
    chart = _call_compute_chart(payload, ts)
    houses = _call_compute_houses(payload, ts)

    return jsonify({"ok": True, "chart": chart, "houses": houses, "meta": {"timescales": ts}}), 200


@api.post("/api/report")
def report():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    tz_name = payload.get("timezone") or payload.get("place_tz") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name)

    chart = _call_compute_chart(payload, ts)
    houses = _call_compute_houses(payload, ts)

    narrative = (
        "This is a placeholder narrative aligned to your mode and computed houses. "
        "Evidence chips will explain predictions in /predictions."
    )

    return jsonify(
        {"ok": True, "chart": chart, "houses": houses, "narrative": narrative, "meta": {"timescales": ts}}
    ), 200


@api.post("/predictions")  # keeping the original path for compatibility
def predictions():
    body = request.get_json(force=True) or {}
    try:
        payload, horizon = parse_prediction_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    tz_name = payload.get("timezone") or payload.get("place_tz") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name)

    chart = _call_compute_chart(payload, ts)
    houses = _call_compute_houses(payload, ts)

    preds_raw = predict(chart, houses, horizon)

    # Load HC thresholds (with sensible defaults)
    th_path = os.environ.get("ASTRO_HC_THRESHOLDS", "config/hc_thresholds.json")
    try:
        with open(th_path, "r", encoding="utf-8") as f:
            hc = json.load(f) or {}
    except Exception:
        hc = {}

    defaults = hc.get("defaults", {}) or {}
    tau = float(defaults.get("tau", 0.88))
    floor = float(defaults.get("floor", 0.60))

    # Optional one-off overrides for testing via request body
    overrides = body.get("hc_overrides") or {}
    if isinstance(overrides, dict):
        if "tau" in overrides:
            tau = float(overrides["tau"])
        if "floor" in overrides:
            floor = float(overrides["floor"])

    # (Optional) environment-level debug overrides
    env_over = os.environ.get("ASTRO_HC_DEBUG_OVERRIDES")
    if env_over:
        try:
            env_dict = json.loads(env_over)
            if isinstance(env_dict, dict):
                if "tau" in env_dict:
                    tau = float(env_dict["tau"])
                if "floor" in env_dict:
                    floor = float(env_dict["floor"])
        except Exception:
            pass

    preds = []
    for i, pr in enumerate(preds_raw):
        p = float(pr.get("probability", 0.0))
        abstained = p < floor
        hc_flag = (not abstained) and (p >= tau)
        preds.append(
            {
                "prediction_id": f"pred_{i}",
                "domain": pr.get("domain"),
                "horizon": horizon,
                "interval_start_utc": pr.get("interval", {}).get("start"),
                "interval_end_utc": pr.get("interval", {}).get("end"),
                "probability_calibrated": p,
                "hc_flag": hc_flag,
                "abstained": abstained,
                "evidence": pr.get("evidence"),
                "mode": chart.get("mode"),
                "ayanamsa_deg": chart.get("ayanamsa_deg"),
                "notes": "QIA+calibrated placeholder; subject to M3 tuning "
                         + ("abstained" if abstained else "accepted"),
            }
        )

    if not overrides and not os.environ.get("ASTRO_HC_DEBUG_OVERRIDES"):
        preds = flag_predictions(preds, horizon, th_path)

    return jsonify({"ok": True, "predictions": preds, "meta": {"timescales": ts}}), 200


@api.post("/rectification/quick")
def rect_quick():
    try:
        body = request.get_json(force=True) or {}
        payload, window_minutes = parse_rectification_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    result = rectification_candidates(payload, window_minutes)
    return jsonify({"ok": True, **result}), 200


@api.get("/api/openapi")
def openapi_spec():
    # Try local app/openapi.yaml first, then project root
    import yaml
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "..", "openapi.yaml"),
        os.path.join(base, "..", "..", "openapi.yaml"),
    ]
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f)
                return jsonify(spec), 200
        except Exception:
            continue
    return _json_error("openapi_not_found", None, 404)


@api.get("/system-validation")
def system_validation():
    cfg = load_config(os.environ.get("ASTRO_CONFIG", "config/defaults.yaml"))

    # Kernel & leap-second policy status (best-effort)
    leap_status: Optional[Dict[str, Any]] = None
    if _leaps:
        for name in ("get_status", "status", "summary"):
            fn = getattr(_leaps, name, None)
            if callable(fn):
                try:
                    s = fn()
                    if is_dataclass(s):
                        leap_status = asdict(s)  # type: ignore
                    elif hasattr(s, "__dict__"):
                        leap_status = dict(s.__dict__)
                    elif isinstance(s, dict):
                        leap_status = s
                    break
                except Exception:
                    pass

    policy = {
        "leap_policy": os.environ.get("ASTRO_LEAP_POLICY", "warn"),
        "dut1_broadcast": os.environ.get("ASTRO_DUT1_BROADCAST", "0") in ("1", "true", "True"),
        "houses_engine": _HOUSES_KIND,
        "polar_policy": {
            "soft_fallback_lat_gt": 66.0,  # façade default (docs)
            "hard_reject_lat_ge": 80.0,
        },
    }

    return jsonify(
        {
            "ok": True,
            "astronomy_accuracy": "ERFA/Skyfield-first, strict timescales (JD_TT/JD_UT1)",
            "performance_slo": {"calculate_p95_ms": 800, "rect_quick_p95_s": 20},
            "mode_consistency": {
                "sidereal_default": cfg.mode == "sidereal",
                "ayanamsa": getattr(cfg, "ayanamsa", None),
            },
            "policy": policy,
            "leap_seconds": leap_status,
            "version": VERSION,
        }
    ), 200


@api.get("/metrics")
def metrics_export():
    from flask import Response
    return Response(metrics.export_prometheus(), mimetype="text/plain")


@api.get("/api/config")
@metrics.middleware("config")
@rate_limit(1)
def config_info():
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    calib_path = os.environ.get("ASTRO_CALIBRATORS", "config/calibrators.json")
    th_path = os.environ.get("ASTRO_HC_THRESHOLDS", "config/hc_thresholds.json")

    cfg = load_config(cfg_path)
    calib_ver = None
    th_summary = None

    # Timescale sample for "now" (debug aid)
    try:
        now_utc = datetime.now(timezone.utc)
        jd_now = _datetime_to_jd_utc(now_utc)
        ts_now = _JD_TO_TS(jd_now)
        ts_sample = {
            "jd_utc": float(jd_now),
            "jd_tt": float(ts_now.get("jd_tt")),
            "jd_ut1": float(ts_now.get("jd_ut1")),
            "delta_t": float(ts_now.get("delta_t")),
            "delta_at": float(ts_now.get("delta_at")),
            "dut1": float(ts_now.get("dut1")),
        }
    except Exception:
        ts_sample = None

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_ver = (json.load(f) or {}).get("version")
    except Exception:
        pass

    try:
        with open(th_path, "r", encoding="utf-8") as f:
            th = json.load(f) or {}
            th_summary = {"entropy_H": th.get("entropy_H"), "defaults": th.get("defaults")}
    except Exception:
        pass

    return jsonify(
        {
            "ok": True,
            "mode": cfg.mode,
            "ayanamsa": getattr(cfg, "ayanamsa", None),
            "rate_limits_per_hour": getattr(cfg, "rate_limits_per_hour", None),
            "pro_features_enabled": getattr(cfg, "pro_features_enabled", None),
            "calibrators_version": calib_ver,
            "hc_thresholds_summary": th_summary,
            "timescale_sample": ts_sample,
            "leap_policy": os.environ.get("ASTRO_LEAP_POLICY", "warn"),
            "version": VERSION,
        }
    ), 200
