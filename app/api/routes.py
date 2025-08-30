# app/api/routes.py
from __future__ import annotations

import json
import os
import inspect
from datetime import datetime, timezone
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request
from zoneinfo import ZoneInfo

from app.version import VERSION
from app.utils.metrics import metrics
from app.utils.ratelimit import rate_limit
from app.utils.hc import flag_predictions
from app.utils.config import load_config

from app.core.validators import (
    parse_chart_payload,
    ValidationError,  # our custom error with .errors()
)

from app.core.astronomy import compute_chart, compute_houses
from app.core.predict import predict
from app.core.rectify import rectification_candidates

api = Blueprint("api", __name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _jd_ut_from_dt(dt_local: datetime) -> float:
    """Compute JD(UT) from a timezone-aware local datetime."""
    if dt_local.tzinfo is None:
        # Assume UTC if somehow naive sneaks in
        dt_local = dt_local.replace(tzinfo=timezone.utc)

    dt_utc = dt_local.astimezone(timezone.utc)
    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    day_frac = (
        (dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600 + dt_utc.microsecond / 3_600_000_000)
        / 24.0
    )

    # Gregorian JD at 0h UT
    a = (14 - m) // 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    jdn = d + ((153 * m2 + 2) // 5) + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045
    return jdn + day_frac - 0.5


def _call_compute_chart(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call compute_chart with whatever signature the current implementation expects.
    Ensures 'jd_ut' is present in the returned dict.
    Preferred order:
      1) (date, time, lat, lon, mode, place_tz)
      2) (date, time, lat, lon, mode)
      3) (date, time, lat, lon)
    """
    date_s = payload["date"]
    time_s = payload["time"]
    lat = payload["latitude"]
    lon = payload["longitude"]
    mode = payload["mode"]
    place_tz = payload["place_tz"]

    chart = None
    errors: List[str] = []

    try:
        chart = compute_chart(date_s, time_s, lat, lon, mode, place_tz)
    except TypeError as e:
        errors.append(str(e))
        try:
            chart = compute_chart(date_s, time_s, lat, lon, mode)
        except TypeError as e2:
            errors.append(str(e2))
            chart = compute_chart(date_s, time_s, lat, lon)

    # Ensure jd_ut
    if "jd_ut" not in chart:
        # Build aware local dt from normalized payload
        tzinfo = ZoneInfo(place_tz if place_tz != "UTC" else "UTC")
        dt_local = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M").replace(tzinfo=tzinfo)
        chart["jd_ut"] = _jd_ut_from_dt(dt_local)

    # Ensure mode/ayanamsa presence for callers that expect it
    chart.setdefault("mode", mode)
    chart.setdefault("ayanamsa_deg", chart.get("ayanamsa_deg"))

    return chart


def _call_compute_houses(lat: float, lon: float, mode: str, jd_ut: float | None) -> Any:
    """Call compute_houses; pass jd_ut if accepted by the implementation."""
    sig = inspect.signature(compute_houses)
    if "jd_ut" in sig.parameters and jd_ut is not None:
        return compute_houses(lat, lon, mode, jd_ut=jd_ut)
    return compute_houses(lat, lon, mode)


def _json_400(e: ValidationError):
    return jsonify({"error": "validation_error", "details": e.errors()}), 400


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@api.get("/api/health")
def health():
    return jsonify({"status": "ok", "version": VERSION}), 200


@api.post("/api/calculate")
def calculate():
    body = request.get_json(force=True) or {}
    try:
        payload = parse_chart_payload(body)
    except ValidationError as e:
        return _json_400(e)

    chart = _call_compute_chart(payload)
    houses = _call_compute_houses(payload["latitude"], payload["longitude"], payload["mode"], chart.get("jd_ut"))
    return jsonify({"chart": chart, "houses": houses}), 200


@api.post("/api/report")
def report():
    body = request.get_json(force=True) or {}
    try:
        payload = parse_chart_payload(body)
    except ValidationError as e:
        return _json_400(e)

    chart = _call_compute_chart(payload)
    houses = _call_compute_houses(payload["latitude"], payload["longitude"], payload["mode"], chart.get("jd_ut"))

    narrative = (
        "This is a placeholder narrative aligned to your mode and computed houses. "
        "Evidence chips will explain predictions in /predictions."
    )
    return jsonify({"chart": chart, "houses": houses, "narrative": narrative}), 200


@api.post("/predictions")  # compatibility path kept
def predictions():
    body = request.get_json(force=True) or {}

    # Base chart validation (strict IANA tz etc.)
    try:
        payload = parse_chart_payload(body)
    except ValidationError as e:
        return _json_400(e)

    # Horizon validation
    horizon = (body.get("horizon") or "short").lower()
    if horizon not in ("short", "medium", "long"):
        return _json_400(
            ValidationError([{"loc": ["horizon"], "msg": "must be one of: short, medium, long", "type": "value_error"}])
        )

    chart = _call_compute_chart(payload)
    houses = _call_compute_houses(payload["latitude"], payload["longitude"], payload["mode"], chart.get("jd_ut"))
    preds_raw = predict(chart, houses, horizon)

    # HC thresholds (file with safe defaults)
    th_path = os.environ.get("ASTRO_HC_THRESHOLDS", "config/hc_thresholds.json")
    try:
        with open(th_path, "r", encoding="utf-8") as f:
            hc = json.load(f) or {}
    except Exception:
        hc = {}

    defaults = hc.get("defaults", {}) or {}
    tau = float(defaults.get("tau", 0.88))
    floor = float(defaults.get("floor", 0.60))

    # Optional overrides for quick experiments
    overrides = body.get("hc_overrides") or {}
    if isinstance(overrides, dict):
        if "tau" in overrides:
            tau = float(overrides["tau"])
        if "floor" in overrides:
            floor = float(overrides["floor"])

    # Env-level debug overrides
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

    # Apply file-driven per-domain/horizon adjustments when not explicitly overridden
    if not overrides and not os.environ.get("ASTRO_HC_DEBUG_OVERRIDES"):
        preds = flag_predictions(preds, horizon, th_path)

    return jsonify({"predictions": preds}), 200


@api.post("/rectification/quick")
def rect_quick():
    body = request.get_json(force=True) or {}

    # Require a window for rectification (keeps "Minimal Format" as an expected 400)
    window = body.get("window_minutes")
    if window is None:
        return _json_400(
            ValidationError([{"loc": ["window_minutes"], "msg": "field required", "type": "missing"}])
        )
    try:
        window_i = int(window)
        if not (5 <= window_i <= 12 * 60):
            raise ValueError
    except Exception:
        return _json_400(
            ValidationError([{
                "loc": ["window_minutes"],
                "msg": "must be an integer between 5 and 720",
                "type": "value_error.number.not_in_range",
            }])
        )

    # Base chart fields
    try:
        payload = parse_chart_payload(body)
    except ValidationError as e:
        return _json_400(e)

    result = rectification_candidates(payload, window_i)
    return jsonify(result), 200


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
    return jsonify({"error": "openapi_not_found"}), 404


@api.get("/system-validation")
def system_validation():
    cfg = load_config(os.environ.get("ASTRO_CONFIG", "config/defaults.yaml"))
    return jsonify(
        {
            "astronomy_accuracy": "free-first (Skyfield if available, fallback approx)",
            "performance_slo": {"calculate_p95_ms": 800, "rect_quick_p95_s": 20},
            "mode_consistency": {
                "sidereal_default": getattr(cfg, "mode", "sidereal") == "sidereal",
                "ayanamsa": getattr(cfg, "ayanamsa", None),
            },
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

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_ver = (json.load(f) or {}).get("version")
    except Exception:
        pass

    try:
        with open(th_path, "r", encoding="utf-8") as f:
            th = json.load(f) or {}
            th_summary = {
                "entropy_H": th.get("entropy_H"),
                "defaults": th.get("defaults"),
            }
    except Exception:
        pass

    return jsonify(
        {
            "mode": getattr(cfg, "mode", "sidereal"),
            "ayanamsa": getattr(cfg, "ayanamsa", None),
            "rate_limits_per_hour": getattr(cfg, "rate_limits_per_hour", None),
            "pro_features_enabled": getattr(cfg, "pro_features_enabled", None),
            "calibrators_version": calib_ver,
            "hc_thresholds_summary": th_summary,
            "version": VERSION,
        }
    ), 200
