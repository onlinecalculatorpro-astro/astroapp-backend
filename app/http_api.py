# app/http_api.py
from __future__ import annotations
from flask import Flask, request, jsonify
from flask_cors import CORS

from app.core.astronomy import compute_chart
from app.core.timescales import julian_day_utc, jd_tt_from_utc_jd

app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/time/timescales")
def timescales():
    data = request.get_json(force=True) or {}
    date = data.get("date")
    time = data.get("time")
    tz   = data.get("tz")
    if not (date and time and tz):
        return jsonify({"error": "missing_fields", "need": ["date","time","tz"]}), 400

    jd_ut = julian_day_utc(date, time, tz)
    year  = int(date[:4]); month = int(date[5:7])
    jd_tt = jd_tt_from_utc_jd(jd_ut, year, month)

    # For now, treat UT1≈UT; compute_chart accepts this.
    return jsonify({"jd_ut": jd_ut, "jd_tt": jd_tt, "jd_ut1": jd_ut})

@app.post("/api/chart")
def chart():
    payload = request.get_json(force=True) or {}
    try:
        out = compute_chart(payload)
        return jsonify(out)
    except Exception as e:
        # Keep responses friendly for the console tests
        return jsonify({"error": "chart_failed", "detail": str(e)}), 400
