# app/http_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.core.astronomy import compute_chart
from app.core.timescales import julian_day_utc, jd_tt_from_utc_jd

app = FastAPI(title="Astro API")

@app.get("/health")
def health():
    return {"ok": True}

class TimescalesIn(BaseModel):
    date: str
    time: str
    tz: str

@app.post("/api/time/timescales")
def timescales(body: TimescalesIn):
    jd_ut = julian_day_utc(body.date, body.time, body.tz)
    jd_tt = jd_tt_from_utc_jd(jd_ut, int(body.date[:4]), int(body.date[5:7]))
    return {"jd_ut": jd_ut, "jd_tt": jd_tt, "jd_ut1": jd_ut}

@app.post("/api/chart")
def chart(payload: dict):
    try:
        return compute_chart(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
