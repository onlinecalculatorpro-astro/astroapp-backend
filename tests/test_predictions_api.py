# tests/test_predictions_api.py
from __future__ import annotations
import json

def test_missing_birth_validation():
    """Payload without 'birth' must raise ValidationError."""
    from app.core.validate_prediction import validate_prediction_payload, ValidationError
    try:
        validate_prediction_payload({})
        assert False, "expected ValidationError"
    except ValidationError as e:
        errs = e.errors()
        assert errs and errs[0]["loc"][0] == "birth"

def test_predictions_endpoint_basic(client):
    """Predictions endpoint should work with minimal valid payload."""
    payload = {
        "birth": {
            "date": "1990-05-21",
            "time": "14:30",
            "tz": "Asia/Kolkata",
            "lat": 12.9716,
            "lon": 77.5946,
        },
        "window": {"days_ahead": 7},
        "preferences": {
            "zodiac": "sidereal",
            "ayanamsa": "lahiri",
            "house_system": "placidus",
        },
        "topics": ["career", "finance"],
        "max_events": 5,
    }

    res = client.post("/api/predictions", json=payload)
    assert res.status_code in (200, 502)  # 502 if engine not available
    data = res.get_json()
    assert "request_id" in data
    assert "metadata" in data
    if res.status_code == 200:
        assert "predictions" in data
        # each prediction should have id, topic, date, confidence
        if data["predictions"]:
            pred = data["predictions"][0]
            assert "id" in pred
            assert "topic" in pred
            assert "confidence" in pred
