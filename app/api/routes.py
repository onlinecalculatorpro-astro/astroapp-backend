from flask import Blueprint, jsonify, request
from pydantic import ValidationError
from app.utils.validation import (
    ChartRequest, PredictionRequest, RectificationRequest,
    Chart, Houses, Prediction, RectificationResult, Evidence
)
from app.core.astronomy import compute_chart, compute_houses
from app.core.predict import predict
from app.core.rectify import rectification_candidates
from app.version import VERSION
from app.utils.hc import flag_predictions
from app.utils.metrics import metrics
from app.utils.ratelimit import rate_limit
import os

api = Blueprint("api", __name__)

@api.get("/api/health")
def health():
    return jsonify({"status":"ok","version":VERSION}), 200

@api.post("/api/calculate")
def calculate():
    try:
        payload = ChartRequest(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error":"validation_error","details":e.errors()}), 400
    chart = compute_chart(payload.date, payload.time, payload.latitude, payload.longitude, payload.mode)
    houses = compute_houses(payload.latitude, payload.longitude, payload.mode)
    return jsonify({"chart": chart, "houses": houses}), 200

@api.post("/api/report")
def report():
    try:
        payload = ChartRequest(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error":"validation_error","details":e.errors()}), 400
    chart = compute_chart(payload.date, payload.time, payload.latitude, payload.longitude, payload.mode)
    houses = compute_houses(payload.latitude, payload.longitude, payload.mode)
    narrative = "This is a placeholder narrative aligned to your mode and computed houses."                " Evidence chips will explain predictions in /predictions."
    return jsonify({"chart": chart, "houses": houses, "narrative": narrative}), 200

@api.post("/predictions")
def predictions():
    try:
        payload = PredictionRequest(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error":"validation_error","details":e.errors()}), 400
    chart = compute_chart(payload.date, payload.time, payload.latitude, payload.longitude, payload.mode, payload.place_tz)
    houses = compute_houses(payload.latitude, payload.longitude, payload.mode, chart["jd_ut"])
    preds_raw = predict(chart, houses, payload.horizon)
    # HC thresholds
    import json
    try:
        with open("config/hc_thresholds.json","r",encoding="utf-8") as f:
            hc = json.load(f)
    except Exception:
        hc = {"entropy_H":1.2, "defaults":{"tau":0.88, "delta":0.08, "floor":0.6}}
    tau = hc.get("defaults",{}).get("tau",0.88)
    # Build response
    preds = []
    for i, pr in enumerate(preds_raw):
        preds.append({
            "prediction_id": f"pred_{i}",
            "domain": pr["domain"],
            "horizon": payload.horizon,
            "interval_start_utc": pr["interval"]["start"],
            "interval_end_utc": pr["interval"]["end"],
            "probability_calibrated": pr["probability"],
            "hc_flag": pr["probability"] >= tau,
            "abstained": pr["probability"] < hc.get("defaults",{}).get("floor",0.6),
            "evidence": pr["evidence"],
            "mode": chart["mode"],
            "ayanamsa_deg": chart["ayanamsa_deg"],
            "notes": "QIA+calibrated placeholder; subject to M3 tuning"
        })
    preds = flag_predictions(preds, payload.horizon, os.environ.get("ASTRO_HC_THRESHOLDS","config/hc_thresholds.json"))
    return jsonify({"predictions": preds}), 200

@api.post("/rectification/quick")
def rect_quick():
    try:
        payload = RectificationRequest(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error":"validation_error","details":e.errors()}), 400
    result = rectification_candidates(payload.dict(), payload.window_minutes)
    return jsonify(result), 200

@api.get("/api/openapi")
def openapi_spec():
    import yaml, os
    # Try local app/openapi.yaml first, then project root
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "openapi.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "..", "openapi.yaml"),
    ]
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f)
                return jsonify(spec), 200
        except Exception:
            continue
    return jsonify({"error":"openapi_not_found"}), 404

@api.get("/system-validation")
def system_validation():
    from app.utils.config import load_config
    cfg = load_config("config/defaults.yaml")
    return jsonify({
        "astronomy_accuracy": "free-first (Skyfield if available, fallback approx)",
        "performance_slo": {"calculate_p95_ms": 800, "rect_quick_p95_s": 20},
        "mode_consistency": {"sidereal_default": cfg.mode == "sidereal", "ayanamsa": cfg.ayanamsa},
        "version": VERSION
    }), 200


@api.get("/metrics")
def metrics_export():
    from flask import Response
    return Response(metrics.export_prometheus(), mimetype="text/plain")


@api.get("/api/config")
@metrics.middleware("config")
@rate_limit(1)
def config_info():
    cfg_path = os.environ.get("ASTRO_CONFIG","config/defaults.yaml")
    cfg = load_config(cfg_path)
    calib_path = os.environ.get("ASTRO_CALIBRATORS","config/calibrators.json")
    th_path = os.environ.get("ASTRO_HC_THRESHOLDS","config/hc_thresholds.json")
    calib_ver = None
    th_summary = None
    try:
        import json
        with open(calib_path,"r",encoding="utf-8") as f:
            calib_ver = (json.load(f) or {}).get("version")
    except Exception:
        pass
    try:
        import json
        with open(th_path,"r",encoding="utf-8") as f:
            th = json.load(f) or {}
            th_summary = {
                "entropy_H": th.get("entropy_H"),
                "defaults": th.get("defaults")
            }
    except Exception:
        pass
    return jsonify({
        "mode": cfg.mode,
        "ayanamsa": cfg.ayanamsa,
        "rate_limits_per_hour": cfg.rate_limits_per_hour,
        "pro_features_enabled": cfg.pro_features_enabled,
        "calibrators_version": calib_ver,
        "hc_thresholds_summary": th_summary,
        "version": VERSION
    }), 200
