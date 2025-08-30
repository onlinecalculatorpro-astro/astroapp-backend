# app/api/debug_endpoints.py
from __future__ import annotations

import json
from typing import Any, Dict, List
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from app.utils.validation import ChartRequest
from app.core.astronomy import compute_chart, compute_houses
from app.core.ephemeris_adapter import enhanced_ecliptic_longitudes
from app.core.astro_extras import (
    harmonic_longitudes,
    part_of_fortune,
    part_of_spirit,
    fixed_star_ecliptics,
    star_conjunctions,
    find_aspects,
    secondary_progressed_time,
    solar_arc_offset,
    apply_solar_arc,
)

debug_api = Blueprint("debug_api", __name__)


# -------- helpers ----------------------------------------------------

def _jd_ut_from_local(date_str: str, time_str: str, tz_name: str) -> float:
    dt_local = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(
        tzinfo=ZoneInfo(tz_name)
    )
    dt_utc = dt_local.astimezone(timezone.utc)

    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    frac = (dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600) / 24.0
    a = (14 - m) // 12
    y2 = y + 4800 - a
    m2 = m + 12 * a - 3
    jdn = d + ((153 * m2 + 2) // 5) + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045
    return jdn + frac - 0.5


# -------- endpoints you can hit from Postman -------------------------

@debug_api.post("/api/debug/enhanced-ecliptic")
def dbg_enhanced_ecliptic():
    """
    Body: {date,time,place_tz,latitude,longitude,mode, ayanamsa?, temperature_C?, pressure_mbar?}
    Returns true/mean/apparent ecliptic coords (topocentric).
    """
    try:
        raw = request.get_json(force=True) or {}
        payload = ChartRequest(**raw)
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    chart = compute_chart(
        payload.date, payload.time, payload.latitude, payload.longitude, payload.mode, payload.place_tz
    )
    jd_tt = chart.get("jd_tt")
    if jd_tt is None:
        # fallback: derive TT from UT (quick approximation)
        jd_tt = _jd_ut_from_local(payload.date, payload.time, payload.place_tz)

    precision = enhanced_ecliptic_longitudes(
        jd_tt=jd_tt,
        lat=payload.latitude,
        lon=payload.longitude,
        temperature_C=raw.get("temperature_C"),
        pressure_mbar=raw.get("pressure_mbar"),
    )
    return jsonify({"precision": precision, "chart_mode": chart.get("mode")}), 200


@debug_api.post("/api/debug/harmonics")
def dbg_harmonics():
    """
    Body: ChartRequest schema. Optional: {"n": 9} for nth harmonic (default 9).
    """
    try:
        raw = request.get_json(force=True) or {}
        payload = ChartRequest(**raw)
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    chart = compute_chart(
        payload.date, payload.time, payload.latitude, payload.longitude, payload.mode, payload.place_tz
    )
    n = int((raw.get("n") or 9))
    nat_lons = {b["name"]: b["longitude_deg"] for b in chart["bodies"] if b["name"] not in ("Rahu", "Ketu")}
    return jsonify({"harmonic_n": n, "longitudes": harmonic_longitudes(nat_lons, n)}), 200


@debug_api.post("/api/debug/arabic")
def dbg_arabic():
    """
    Body: ChartRequest schema.
    Returns Part of Fortune & Spirit (assumes daytime for now).
    """
    try:
        raw = request.get_json(force=True) or {}
        payload = ChartRequest(**raw)
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    chart = compute_chart(
        payload.date, payload.time, payload.latitude, payload.longitude, payload.mode, payload.place_tz
    )
    houses = compute_houses(payload.latitude, payload.longitude, payload.mode, chart.get("jd_ut"))

    nat_lons = {b["name"]: b["longitude_deg"] for b in chart["bodies"] if b["name"] not in ("Rahu", "Ketu")}
    asc = houses.get("asc_deg")
    if asc is None or "Sun" not in nat_lons or "Moon" not in nat_lons:
        return jsonify({"error": "missing_data"}), 400

    is_day = True  # TODO: compute via Sun altitude if you like
    return jsonify({
        "fortune": part_of_fortune(asc, nat_lons["Sun"], nat_lons["Moon"], is_day),
        "spirit":  part_of_spirit(asc, nat_lons["Sun"], nat_lons["Moon"], is_day),
    }), 200


@debug_api.post("/api/debug/fixedstars")
def dbg_fixedstars():
    """
    Body: ChartRequest schema. Optional: {"orb_deg": 1.0}
    Returns fixed-star ecliptics and planet-star conjunctions.
    """
    try:
        raw = request.get_json(force=True) or {}
        payload = ChartRequest(**raw)
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    chart = compute_chart(
        payload.date, payload.time, payload.latitude, payload.longitude, payload.mode, payload.place_tz
    )
    orb = float(raw.get("orb_deg") or 1.0)
    nat_lons = {b["name"]: b["longitude_deg"] for b in chart["bodies"] if b["name"] not in ("Rahu", "Ketu")}
    stars = fixed_star_ecliptics(chart["jd_tt"])
    hits = star_conjunctions(nat_lons, stars, orb_deg=orb)
    return jsonify({"stars": stars, "conjunctions": hits, "orb_deg": orb}), 200


@debug_api.post("/api/debug/aspects")
def dbg_aspects():
    """
    Body: ChartRequest schema. Optional: {"orb_deg": 2.0}
    Natal-natal aspects (unique pairs).
    """
    try:
        raw = request.get_json(force=True) or {}
        payload = ChartRequest(**raw)
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    chart = compute_chart(
        payload.date, payload.time, payload.latitude, payload.longitude, payload.mode, payload.place_tz
    )
    nat_lons = {b["name"]: b["longitude_deg"] for b in chart["bodies"] if b["name"] not in ("Rahu", "Ketu")}
    orb = float(raw.get("orb_deg") or 2.0)
    aspects = find_aspects(nat_lons, nat_lons, orb_deg=orb)
    return jsonify({"orb_deg": orb, "aspects": aspects}), 200


@debug_api.post("/api/debug/progressions-solararc")
def dbg_progressions_solararc():
    """
    Body: {
      date,time,place_tz,latitude,longitude,mode,
      "target_date":"YYYY-MM-DD","target_time":"HH:MM"
    }
    Returns secondary progressed JD, solar arc offset, and solar-arc longitudes.
    """
    try:
        raw = request.get_json(force=True) or {}
        payload = ChartRequest(**raw)
        tgt_date = raw.get("target_date")
        tgt_time = raw.get("target_time")
        if not (tgt_date and tgt_time):
            raise ValidationError([{"loc": ["target_date/target_time"], "msg": "required", "type": "value_error"}], ChartRequest)
    except ValidationError as e:
        return jsonify({"error": "validation_error", "details": e.errors()}), 400

    # Natal chart & longitudes
    chart = compute_chart(
        payload.date, payload.time, payload.latitude, payload.longitude, payload.mode, payload.place_tz
    )
    nat_lons = {b["name"]: b["longitude_deg"] for b in chart["bodies"] if b["name"] not in ("Rahu", "Ketu")}

    # Build target jd_tt (roughly via UT)
    jd_tt_birth = chart["jd_tt"]
    jd_tt_target = _jd_ut_from_local(tgt_date, tgt_time, payload.place_tz)

    jd_prog = secondary_progressed_time(jd_tt_birth, jd_tt_target)

    # Get progressed Sun lon (true-of-date)
    prog = enhanced_ecliptic_longitudes(jd_prog, payload.latitude, payload.longitude)
    prog_lons = {b["name"]: b["true"]["lon"] for b in prog["bodies"] if b["name"] in nat_lons}

    arc = solar_arc_offset(nat_lons["Sun"], prog_lons["Sun"])
    solar_arc_lons = apply_solar_arc(nat_lons, arc)

    return jsonify({
        "jd_tt_natal": jd_tt_birth,
        "jd_tt_target": jd_tt_target,
        "jd_tt_progressed": jd_prog,
        "progressed_true_lons": prog_lons,
        "solar_arc_deg": arc,
        "solar_arc_longitudes": solar_arc_lons
    }), 200
