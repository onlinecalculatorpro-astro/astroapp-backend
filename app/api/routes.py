# app/api/routes.py
from __future__ import annotations

import json
import os
import inspect
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict

from flask import Blueprint, jsonify, request
from pydantic import ValidationError
from app.core.validators import parse_chart_payload, ValidationError

from app.version import VERSION
from app.utils.validation import (
    ChartRequest,
    PredictionRequest,
    RectificationRequest,
)
from app.utils.metrics import metrics
from app.utils.ratelimit import rate_limit
from app.utils.hc import flag_predictions
from app.utils.config import load_config

from app.core.astronomy import compute_chart, compute_houses
from app.core.predict import predict
from app.core.rectify import rectification_candidates

api = Blueprint("api", __name__)

# ---------- helpers -----------------------------------------------------------

def _jd_ut_from_local(date_str: str, time_str: str, tz_name: str) -> float:
    """Compute JD(UT) from local date/time and IANA timezone name."""
    dt_local = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(
        tzinfo=ZoneInfo(tz_name)
    )
    dt_utc = dt_local.astimezone(timezone.utc)

    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    day_frac = (dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600) / 24.0

    # Gregorian JD at 0h UT
    a = (14 - m) // 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    jdn = d + ((153 * m2 + 2) // 5) + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045
    return jdn + day_frac - 0.5


def _call_compute_chart(payload: ChartRequest | PredictionRequest) -> Dict[str, Any]:
    """
    Call compute_chart with whatever signature the current implementation expects.
    Ensures 'jd_ut' is present in the returned dict.
    """
    # Try (date, time, lat, lon, mode, place_tz)
    try:
        chart = compute_chart(
            payload.date,
            payload.time,
            payload.latitude,
            payload.longitude,
            payload.mode,
            getattr(payload, "place_tz", None),
        )
    except TypeError:
        # Fallback: (date, time, lat, lon, mode)
        chart = compute_chart(
            payload.date,
            payload.time,
            payload.latitude,
            payload.longitude,
            payload.mode,
        )

    if "jd_ut" not in chart:
        tz = getattr(payload, "place_tz", "UTC")
        chart["jd_ut"] = _jd_ut_from_local(payload.date, payload.time, tz)

    return chart


def _call_compute_houses(lat: float, lon: float, mode: str, jd_ut: float | None) -> Any:
    """
    Call compute_houses; pass jd_ut if the implementation accepts it.
    """
    sig = inspect.signature(compute_houses)
    if "jd_ut" in sig.parameters and jd_ut is not None:
        return compute_houses(lat, lon, mode, jd_ut=jd_ut)
    # Legacy 3-arg version
    return compute_houses(lat, lon, mode)


# ---------- endpoints ---------------------------------------------------------

@api.get("/api/health")
def health():
    return jsonify({"status": "ok", "version": VERSION}), 200


@api.post("/api/calculate")
def calculate():
    try:
        payload = ChartRequest(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    chart = _call_compute_chart(payload)
    houses = _call_compute_houses(payload.latitude, payload.longitude, payload.mode, chart.get("jd_ut"))

    return jsonify({"chart": chart, "houses": houses}), 200


@api.post("/api/report")
def report():
    try:
        payload = ChartRequest(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    chart = _call_compute_chart(payload)
    houses = _call_compute_houses(payload.latitude, payload.longitude, payload.mode, chart.get("jd_ut"))
    narrative = (
        "This is a placeholder narrative aligned to your mode and computed houses. "
        "Evidence chips will explain predictions in /predictions."
    )

    return jsonify({"chart": chart, "houses": houses, "narrative": narrative}), 200


@api.post("/predictions")  # keeping the original path for compatibility
def predictions():
    # Read body once so we can pick up optional overrides
    body = request.get_json(force=True) or {}
    try:
        payload = PredictionRequest(**body)
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    chart = _call_compute_chart(payload)
    houses = _call_compute_houses(payload.latitude, payload.longitude, payload.mode, chart.get("jd_ut"))

    preds_raw = predict(chart, houses, payload.horizon)

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
            pass  # ignore malformed env override

    preds = []
    for i, pr in enumerate(preds_raw):
        p = float(pr.get("probability", 0.0))
        abstained = p < floor
        hc_flag = (not abstained) and (p >= tau)
        preds.append(
            {
                "prediction_id": f"pred_{i}",
                "domain": pr.get("domain"),
                "horizon": payload.horizon,
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

    # Only run file-driven per-domain/horizon adjustments if no explicit overrides were used
    if not overrides and not os.environ.get("ASTRO_HC_DEBUG_OVERRIDES"):
        preds = flag_predictions(preds, payload.horizon, th_path)

    return jsonify({"predictions": preds}), 200


@api.post("/rectification/quick")
def rect_quick():
    try:
        payload = RectificationRequest(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    result = rectification_candidates(payload.dict(), payload.window_minutes)
    return jsonify(result), 200


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

    return jsonify({"error": "openapi_not_found"}), 404


@api.get("/system-validation")
def system_validation():
    cfg = load_config(os.environ.get("ASTRO_CONFIG", "config/defaults.yaml"))
    return jsonify(
        {
            "astronomy_accuracy": "free-first (Skyfield if available, fallback approx)",
            "performance_slo": {"calculate_p95_ms": 800, "rect_quick_p95_s": 20},
            "mode_consistency": {
                "sidereal_default": cfg.mode == "sidereal",
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
            th_summary = {"entropy_H": th.get("entropy_H"), "defaults": th.get("defaults")}
    except Exception:
        pass

    return jsonify(
        {
            "mode": cfg.mode,
            "ayanamsa": getattr(cfg, "ayanamsa", None),
            "rate_limits_per_hour": getattr(cfg, "rate_limits_per_hour", None),
            "pro_features_enabled": getattr(cfg, "pro_features_enabled", None),
            "calibrators_version": calib_ver,
            "hc_thresholds_summary": th_summary,
            "version": VERSION,
        }
    ), 200
