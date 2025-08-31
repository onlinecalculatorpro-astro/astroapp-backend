# app/api/predictions.py
from __future__ import annotations

import uuid, logging
from typing import Any, Dict, List

from flask import Blueprint, request, jsonify
from app.core.validate_prediction import validate_prediction_payload, ValidationError

# ───────────────────────────── metrics (optional) ─────────────────────────────
try:
    from prometheus_client import Counter
    _PRED_REQS = Counter("predictions_requests_total", "Prediction requests", ["outcome"])
except Exception:  # Prometheus not installed
    _PRED_REQS = None  # type: ignore

# ───────────────────────────── engine import ─────────────────────────────
try:
    from app.core.professional_astro import ProfessionalAstrologyEngine  # type: ignore
except Exception:
    ProfessionalAstrologyEngine = None  # type: ignore

log = logging.getLogger(__name__)
predictions_bp = Blueprint("predictions", __name__)

API_VERSION = "predictions_v1"
SCHEMA_VERSION = "1.0.0"

# ───────────────────────────── helpers ─────────────────────────────
def _label_for_conf(v: float) -> str:
    if v >= 0.75:
        return "high"
    if v >= 0.50:
        return "medium"
    return "low"

def _compute_confidence(item: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic confidence computation from score + signals."""
    signals = item.get("signals") or []
    try:
        base = float(item.get("score", 0.5))
    except Exception:
        base = 0.5

    n = len(signals)
    avg_strength = 0.5
    min_orb = 3.0
    if n:
        total_strength = 0.0
        for s in signals:
            try:
                total_strength += float(s.get("strength", 0.5))
                min_orb = min(min_orb, float(s.get("orb", 3.0)))
            except Exception:
                pass
        avg_strength = total_strength / max(n, 1)

    w_count = min(n / 4.0, 1.0) * 0.40
    w_strength = min(max(avg_strength, 0.0), 1.0) * 0.40
    w_orb = max(0.0, (3.0 - min_orb) / 3.0) * 0.20

    conf = min(1.0, max(0.0, 0.15 + 0.35 * base + w_count + w_strength + w_orb))
    return {"value": round(conf, 3), "label": _label_for_conf(conf)}

def _shape_prediction(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw engine prediction into Gold-Standard schema."""
    topic = (item.get("topic") or "general").lower()
    date = str(item.get("date") or item.get("timestamp") or "")
    narrative = item.get("narrative") or item.get("summary") or ""
    signals = item.get("signals") or []

    try:
        score = float(item.get("score", 0.5))
    except Exception:
        score = 0.5

    conf = item.get("confidence")
    if not isinstance(conf, dict) or "value" not in conf:
        conf = _compute_confidence(item)

    shaped_signals = []
    for s in signals:
        try:
            orb = float(s.get("orb", 0.0))
        except Exception:
            orb = 0.0
        try:
            strength = float(s.get("strength", 0.5))
        except Exception:
            strength = 0.5
        shaped_signals.append({
            "aspect": s.get("aspect", ""),
            "bodies": s.get("bodies", []),
            "orb": orb,
            "strength": strength,
            "direction": s.get("direction", "applying"),
        })

    return {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "date": date,
        "score": round(score, 3),
        "confidence": conf,
        "narrative": narrative,
        "signals": shaped_signals,
    }

def _summarize(predictions: List[Dict[str, Any]]) -> str:
    """Return a compact summary of top predictions."""
    if not predictions:
        return "No significant events detected in the requested window."
    top = sorted(predictions, key=lambda p: p["score"] * p["confidence"]["value"], reverse=True)[:3]
    return " | ".join(
        f"{p['date']}: {p['topic'].capitalize()} ({int(round(p['confidence']['value']*100))}% confidence)"
        for p in top
    )

# ───────────────────────────── endpoint ─────────────────────────────
@predictions_bp.post("/predictions")
def predictions_endpoint():
    rid = str(uuid.uuid4())

    # Parse JSON
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        if _PRED_REQS: _PRED_REQS.labels(outcome="invalid_json").inc()
        return jsonify({"error": "invalid_json", "message": "Body must be valid JSON", "request_id": rid}), 400

    # Validate payload
    try:
        valid = validate_prediction_payload(payload)
    except ValidationError as e:
        if _PRED_REQS: _PRED_REQS.labels(outcome="validation_error").inc()
        return jsonify({"error": "validation_error", "errors": e.errors(), "request_id": rid}), 422

    # Engine call
    engine_info: Dict[str, Any] = {}
    try:
        if ProfessionalAstrologyEngine is None:
            raise RuntimeError("ProfessionalAstrologyEngine unavailable")

        engine = ProfessionalAstrologyEngine()
        engine_info = {
            "name": getattr(engine, "name", "ProfessionalAstrologyEngine"),
            "mode": getattr(engine, "precision_mode", "STANDARD"),
        }

        try:
            results = engine.get_transit_predictions(
                birth_dt=valid.birth.as_aware_datetime(),
                lat=valid.birth.lat,
                lon=valid.birth.lon,
                window=(valid.window.start, valid.window.end),
                zodiac=valid.preferences.zodiac,
                ayanamsa=valid.preferences.ayanamsa,
                house_system=valid.preferences.house_system,
                topics=list(valid.topics),
                max_events=valid.max_events,
            )
        except TypeError:
            # fallback to older signature
            results = engine.get_transit_predictions(
                birth_dt=valid.birth.as_aware_datetime(),
                lat=valid.birth.lat,
                lon=valid.birth.lon,
                window=(valid.window.start, valid.window.end),
                topics=list(valid.topics),
            )

        if not isinstance(results, list):
            raise RuntimeError("Engine returned non-list predictions")

    except Exception as e:
        if _PRED_REQS: _PRED_REQS.labels(outcome="engine_error").inc()
        log.exception("Prediction engine failure: %s", e)
        return jsonify({
            "error": "engine_error",
            "message": "Prediction engine failed",
            "request_id": rid,
            "details": str(e),
        }), 502

    # Normalize predictions
    shaped = [_shape_prediction(x) for x in results]

    response = {
        "version": API_VERSION,
        "schema": SCHEMA_VERSION,
        "request_id": rid,
        "metadata": {
            "engine": engine_info,
            "inputs": {
                "birth": {
                    "date": valid.birth.date.isoformat(),
                    "time": valid.birth.time.isoformat(),
                    "tz": str(valid.birth.tz),
                    "lat": valid.birth.lat,
                    "lon": valid.birth.lon,
                },
                "window": {
                    "start": valid.window.start.isoformat(),
                    "end": valid.window.end.isoformat(),
                },
                "preferences": {
                    "zodiac": valid.preferences.zodiac,
                    "ayanamsa": valid.preferences.ayanamsa,
                    "house_system": valid.preferences.house_system,
                },
                "topics": list(valid.topics),
                "max_events": valid.max_events,
            },
            "summary": _summarize(shaped),
        },
        "predictions": shaped,
    }

    if _PRED_REQS: _PRED_REQS.labels(outcome="ok").inc()
    return jsonify(response), 200
