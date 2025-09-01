# app/http_api.py
from __future__ import annotations
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from app.core.astronomy import compute_chart, AstronomyError
from app.core.timescales import julian_day_utc, jd_tt_from_utc_jd, to_utc_iso
from datetime import datetime
from zoneinfo import ZoneInfo

app = FastAPI(title="Astro API", version="1.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/time/timescales")
def timescales(payload: dict = Body(...)):
    try:
        date = payload["date"]; time = payload["time"]; tz = payload["tz"]
        # UTC JD
        jd_ut = float(julian_day_utc(date, time, tz))
        # Year/month for ΔT from local date (your poly expects calendar Y/M)
        dt_local = datetime.fromisoformat(f"{date}T{time}:00").replace(tzinfo=ZoneInfo(tz))
        jd_tt = float(jd_tt_from_utc_jd(jd_ut, dt_local.year, dt_local.month))
        jd_ut1 = jd_ut  # UT1≈UTC (good enough for now)
        return {
            "utc": to_utc_iso(date, time, tz),
            "jd_ut": jd_ut,
            "jd_tt": jd_tt,
            "jd_ut1": jd_ut1,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/api/chart")
def chart(payload: dict = Body(...)):
    """
    Accepts either:
      A) { jd_ut, jd_tt, jd_ut1, latitude, longitude, mode?, topocentric?, ayanamsa? }
      B) { date, time, tz, latitude, longitude, mode?, topocentric?, ayanamsa? }  (we derive jd_*)
    """
    try:
        if not all(k in payload for k in ("jd_ut", "jd_tt", "jd_ut1")):
            # derive timescales from civil inputs
            ts = timescales({"date": payload["date"], "time": payload["time"], "tz": payload["tz"]})
            if isinstance(ts, JSONResponse):  # error bubbled
                return ts
            payload = {
                **payload,
                "jd_ut": ts["jd_ut"],
                "jd_tt": ts["jd_tt"],
                "jd_ut1": ts["jd_ut1"],
            }

        result = compute_chart({
            "mode": payload.get("mode", "tropical"),
            "topocentric": bool(payload.get("topocentric", False)),
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "ayanamsa": payload.get("ayanamsa"),
            "jd_ut": float(payload["jd_ut"]),
            "jd_tt": float(payload["jd_tt"]),
            "jd_ut1": float(payload["jd_ut1"]),
        })
        return result
    except AstronomyError as e:
        return JSONResponse({"code": e.code, "error": "astro_error", "message": str(e)}, status_code=400)
    except KeyError as e:
        return JSONResponse({"error": "missing_field", "field": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": "server_error", "message": str(e)}, status_code=500)
