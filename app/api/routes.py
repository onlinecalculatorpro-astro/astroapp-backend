# app/api/routes.py
from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo

from flask import Blueprint, jsonify, request

from app.version import VERSION
from app.utils.config import load_config
from app.utils.hc import flag_predictions
from app.utils.metrics import metrics
from app.utils.ratelimit import rate_limit

from app.core.validators import (
    ValidationError,
    parse_chart_payload,
    parse_prediction_payload,
    parse_rectification_payload,
)

# canonical chart call (may error at runtime; we guard at call sites)
from app.core.astronomy import compute_chart  # type: ignore
from app.core.predict import predict
from app.core.rectify import rectification_candidates

# houses: prefer policy façade if present
try:
    from app.core.house import compute_houses_with_policy as _houses_fn  # type: ignore
    _HOUSES_KIND = "policy"
except Exception:
    from app.core.astronomy import compute_houses as _houses_fn  # type: ignore
    _HOUSES_KIND = "legacy"

# time kernel (required)
from app.core import time_kernel as _tk

# optional leap-seconds helper (best-effort)
try:
    from app.core import leapseconds as _leaps
except Exception:  # pragma: no cover
    _leaps = None  # type: ignore

api = Blueprint("api", __name__)
DEBUG_VERBOSE = os.getenv("ASTRO_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "on")


# ─────────────────────────────── timescales ────────────────────────────────
def _datetime_to_jd_utc(dt_utc: datetime) -> float:
    """Return Julian Date (UTC). Prefer ERFA; fallback to safe Meeus form."""
    if dt_utc.tzinfo is None or dt_utc.tzinfo is not timezone.utc:
        raise ValueError("dt_utc must be timezone-aware UTC")
    try:
        import erfa  # PyERFA
        iy, im, id_ = dt_utc.year, dt_utc.month, dt_utc.day
        ih, iv = dt_utc.hour, dt_utc.minute
        sf = dt_utc.second + dt_utc.microsecond / 1e6
        d1, d2 = erfa.dtf2d("UTC", iy, im, id_, ih, iv, sf)
        return float(d1 + d2)
    except Exception:
        # Meeus fallback
        Y, M, D = dt_utc.year, dt_utc.month, dt_utc.day
        h, m = dt_utc.hour, dt_utc.minute
        s = dt_utc.second + dt_utc.microsecond / 1_000_000.0
        a = (14 - M) // 12
        y = Y + 4800 - a
        m_ = M + 12 * a - 3
        jdn = D + (153 * m_ + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
        dayfrac = (h - 12) / 24.0 + m / 1440.0 + s / 86400.0
        return float(jdn + dayfrac)


def _find_kernel_callable() -> Callable[[float], Dict[str, Any]]:
    """
    Locate a JD_UTC -> timescales callable on app.core.time_kernel.
    Must return a dict-like with jd_tt, jd_ut1, delta_t, delta_at, dut1, warnings?, policy?
    """
    candidates = (
        "jd_utc_to_timescales",
        "utc_jd_to_timescales",
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

    # Class API fallback
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
    Convert local date/time/tz to JD(UTC), then expand to jd_tt/jd_ut1 with time_kernel.
    Returns a dict safe for JSON.
    """
    fmt = "%Y-%m-%d %H:%M" if time_str.count(":") == 1 else "%Y-%m-%d %H:%M:%S"

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        raise ValidationError({"place_tz": "must be a valid IANA zone like 'Asia/Kolkata'"})

    dt_local = datetime.strptime(f"{date_str} {time_str}", fmt).replace(tzinfo=tz)
    dt_utc = dt_local.astimezone(timezone.utc)
    jd_utc = _datetime_to_jd_utc(dt_utc)

    ts = _JD_TO_TS(jd_utc)
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


# ─────────────────────── engine call adapters ───────────────────────
def _sig_accepts(fn: Callable, *names: str) -> Dict[str, bool]:
    """Map of param -> accepted by fn."""
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return {n: False for n in names}
    return {n: (n in params) for n in names}


def _call_compute_chart(payload: Dict[str, Any], ts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call compute_chart with signature introspection. Supports alternate names like
    time_s/lat/lon/date_s. Supplies jd_ut/jd_tt/jd_ut1 when supported. Ensures 'jd_ut'.
    """
    params = inspect.signature(compute_chart).parameters

    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in params:
                return c
        return None

    # Map payload fields to the function's parameter names
    name_date = pick("date", "date_s", "date_str")
    name_time = pick("time", "time_s", "time_str")
    name_lat  = pick("latitude", "lat")
    name_lon  = pick("longitude", "lon")
    name_mode = pick("mode", "system")        # some codebases say 'system'
    name_tz   = pick("place_tz", "timezone", "tz_name")

    kwargs: Dict[str, Any] = {}
    if name_date: kwargs[name_date] = payload["date"]
    if name_time: kwargs[name_time] = payload["time"]
    if name_lat:  kwargs[name_lat]  = payload["latitude"]
    if name_lon:  kwargs[name_lon]  = payload["longitude"]
    if name_mode: kwargs[name_mode] = payload["mode"]
    if name_tz:   kwargs[name_tz]   = payload.get("timezone") or payload.get("place_tz")

    # Timescales passthrough (only if the function accepts them)
    if "jd_ut" in params:
        kwargs["jd_ut"] = ts["jd_utc"]
    if "jd_tt" in params:
        kwargs["jd_tt"] = ts["jd_tt"]
    if "jd_ut1" in params:
        kwargs["jd_ut1"] = ts["jd_ut1"]

    chart = compute_chart(**kwargs)  # may raise

    # Guarantee jd_ut for downstream code
    if "jd_ut" not in chart:
        chart["jd_ut"] = ts["jd_utc"]
    if "mode" not in chart and "mode" in payload:
        chart["mode"] = payload["mode"]

    return chart


def _call_compute_houses(payload: Dict[str, Any], ts: Dict[str, Any]) -> Any:
    """
    Prefer policy façade if available; pass jd_tt/jd_ut1 when accepted.
    Fallback to legacy compute_houses(lat, lon, mode[, jd_ut]).

    IMPORTANT: chart 'mode' is 'sidereal'/'tropical' (NOT a house system).
               We pass user's requested house system via payload['house_system'].
    """
    lat = float(payload["latitude"])
    lon = float(payload["longitude"])

    requested_system = payload.get("house_system") or None

    accepts = _sig_accepts(
        _houses_fn,
        "lat", "lon",
        "system", "requested_house_system", "house_system",
        "mode", "jd_ut", "jd_tt", "jd_ut1", "diagnostics"
    )

    kwargs: Dict[str, Any] = {"lat": lat, "lon": lon}

    # Map requested system to whatever the target supports
    if requested_system:
        rs = str(requested_system).lower()
        if accepts.get("system"):
            kwargs["system"] = rs
        elif accepts.get("requested_house_system"):
            kwargs["requested_house_system"] = rs
        elif accepts.get("house_system"):
            kwargs["house_system"] = rs

    # Legacy engines sometimes expect 'mode' (sidereal/tropical)
    if accepts.get("mode") and "mode" in payload:
        kwargs["mode"] = payload["mode"]

    # Timescales (prefer jd_tt/jd_ut1; legacy may accept jd_ut)
    if accepts.get("jd_tt"):
        kwargs["jd_tt"] = ts["jd_tt"]
    if accepts.get("jd_ut1"):
        kwargs["jd_ut1"] = ts["jd_ut1"]
    if accepts.get("jd_ut") and "jd_tt" not in kwargs and "jd_ut1" not in kwargs:
        kwargs["jd_ut"] = ts["jd_utc"]

    # Diagnostics passthrough
    if accepts.get("diagnostics") and "diagnostics" in payload:
        kwargs["diagnostics"] = bool(payload["diagnostics"])

    return _houses_fn(**kwargs)


# ───────────────────────────── error helpers ─────────────────────────────
def _json_error(code: str, details: Any = None, http: int = 400):
    out: Dict[str, Any] = {"ok": False, "error": code}
    if details is not None:
        out["details"] = details
    return jsonify(out), http


def _normalize_houses_payload(houses: Any) -> Any:
    """Ensure houses payload has both 'cusps' and 'cusps_deg' and consistent system fields."""
    if isinstance(houses, dict):
        # cusps/cusps_deg compatibility
        if "cusps" not in houses and "cusps_deg" in houses:
            houses["cusps"] = houses["cusps_deg"]
        if "cusps_deg" not in houses and "cusps" in houses:
            houses["cusps_deg"] = houses["cusps"]
        # system name compatibility
        if "house_system" not in houses and "system" in houses:
            houses["house_system"] = houses["system"]
        if "system" not in houses and "house_system" in houses:
            houses["system"] = houses["house_system"]
        # angles compatibility
        if "asc_deg" not in houses and "asc" in houses:
            houses["asc_deg"] = houses["asc"]
        if "mc_deg" not in houses and "mc" in houses:
            houses["mc_deg"] = houses["mc"]
    return houses


# ───────────────────────────── endpoints ─────────────────────────────
@api.get("/api/health")
def health():
    return jsonify({"ok": True, "status": "up", "version": VERSION}), 200


@api.post("/api/calculate")
def calculate():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)  # expects: date, time, place_tz, latitude, longitude, mode
        hs = str(body.get("house_system", "")).strip().lower()
        if hs:
            payload["house_system"] = hs
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name)

    # Compute chart — but never fail the endpoint if it errors
    chart: Dict[str, Any]
    chart_warnings: list[str] = []
    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        chart = {
            "mode": payload.get("mode"),
            "jd_ut": ts["jd_utc"],
        }
        chart_warnings.append("chart_failed")
        if DEBUG_VERBOSE:
            chart["error"] = str(e)

    try:
        houses = _call_compute_houses(payload, ts)
        houses = _normalize_houses_payload(houses)
    except ValueError as e:
        # user input issue for houses
        return _json_error("houses_error", str(e), 400)
    except Exception as e:
        # internal houses failure only → this endpoint fails
        return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

    if chart_warnings:
        chart["warnings"] = chart.get("warnings", []) + chart_warnings

    return jsonify({"ok": True, "chart": chart, "houses": houses, "meta": {"timescales": ts}}), 200


@api.post("/api/report")
def report():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)
        hs = str(body.get("house_system", "")).strip().lower()
        if hs:
            payload["house_system"] = hs
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name)

    chart: Dict[str, Any]
    chart_warnings: list[str] = []
    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        chart = {
            "mode": payload.get("mode"),
            "jd_ut": ts["jd_utc"],
        }
        chart_warnings.append("chart_failed")
        if DEBUG_VERBOSE:
            chart["error"] = str(e)

    try:
        houses = _call_compute_houses(payload, ts)
        houses = _normalize_houses_payload(houses)
    except ValueError as e:
        return _json_error("houses_error", str(e), 400)
    except Exception as e:
        return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

    if chart_warnings:
        chart["warnings"] = chart.get("warnings", []) + chart_warnings

    narrative = (
        "This is a placeholder narrative aligned to your mode and computed houses. "
        "Evidence chips will explain predictions in /predictions."
    )

    return jsonify(
        {"ok": True, "chart": chart, "houses": houses, "narrative": narrative, "meta": {"timescales": ts}}
    ), 200


@api.post("/predictions")
def predictions_route():
    body = request.get_json(force=True) or {}
    try:
        payload, horizon = parse_prediction_payload(body)
        hs = str(body.get("house_system", "")).strip().lower()
        if hs:
            payload["house_system"] = hs
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name)

    # Chart (required for predictions, but we still compute houses to include helpful context)
    chart: Optional[Dict[str, Any]]
    chart_error: Optional[str] = None
    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        chart = None
        chart_error = str(e) if DEBUG_VERBOSE else "chart_failed"

    try:
        houses = _call_compute_houses(payload, ts)
        houses = _normalize_houses_payload(houses)
    except Exception as e:
        return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

    if chart is None:
        # For predictions we fail clearly if chart is unavailable
        return _json_error("chart_internal", chart_error or "internal_error", 500)

    preds_raw = predict(chart, houses, horizon)

    # thresholds (with sensible defaults)
    th_path = os.environ.get("ASTRO_HC_THRESHOLDS", "config/hc_thresholds.json")
    try:
        with open(th_path, "r", encoding="utf-8") as f:
            hc = json.load(f) or {}
    except Exception:
        hc = {}

    defaults = hc.get("defaults", {}) or {}
    tau = float(defaults.get("tau", 0.88))
    floor = float(defaults.get("floor", 0.60))

    # request-body overrides (test only)
    overrides = body.get("hc_overrides") or {}
    if isinstance(overrides, dict):
        if "tau" in overrides:
            tau = float(overrides["tau"])
        if "floor" in overrides:
            floor = float(overrides["floor"])

    # env debug overrides
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

    # leap status (best-effort)
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
        "leap_policy": os.getenv("ASTRO_LEAP_POLICY", "warn"),
        "dut1_broadcast": os.getenv("ASTRO_DUT1_BROADCAST", "0") in ("1", "true", "True"),
        "houses_engine": _HOUSES_KIND,
        "polar_policy": {
            "soft_fallback_lat_gt": 66.0,
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

    # timescale sample for "now" (debug aid)
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
            "leap_policy": os.getenv("ASTRO_LEAP_POLICY", "warn"),
            "version": VERSION,
        }
    ), 200
