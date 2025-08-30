import json
from app.main import app

def client():
    app.testing = True
    return app.test_client()

sample = {
    "date": "1992-11-04",
    "time": "05:25",
    "place_tz": "Asia/Kolkata",
    "latitude": 13.0827,
    "longitude": 80.2707,
    "mode": "sidereal",
    "ayanamsa": "lahiri"
}

def test_health():
    c = client()
    rv = c.get("/api/health")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["status"] == "ok"

def test_calculate():
    c = client()
    rv = c.post("/api/calculate", json=sample)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "chart" in data and "houses" in data
    assert "bodies" in data["chart"]
    assert "asc_deg" in data["houses"]

def test_predictions():
    c = client()
    payload = dict(sample)
    payload["horizon"] = "short"
    rv = c.post("/predictions", json=payload)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "predictions" in data
    assert len(data["predictions"]) == 5
    for p in data["predictions"]:
        assert "prediction_id" in p
        assert 0.0 <= p["probability_calibrated"] <= 1.0

def test_rectification_quick():
    c = client()
    payload = dict(sample)
    payload["window_minutes"] = 90
    rv = c.post("/rectification/quick", json=payload)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "best_time" in data
    assert "top3_times" in data
    assert "composite_scores" in data


def test_openapi():
    c = client()
    rv = c.get("/api/openapi")
    assert rv.status_code in (200, 404)
    if rv.status_code == 200:
        data = rv.get_json()
        assert "openapi" in data or "paths" in data


def test_hc_thresholds():
    c = client()
    payload = dict(sample)
    payload["horizon"] = "short"
    rv = c.post("/predictions", json=payload)
    assert rv.status_code == 200
    data = rv.get_json()
    assert "predictions" in data
    # ensure keys present
    p = data["predictions"][0]
    assert "hc_flag" in p and "abstained" in p
