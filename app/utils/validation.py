from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ChartRequest(BaseModel):
    date: str
    time: str
    place_tz: str
    latitude: float
    longitude: float
    mode: str = Field("sidereal", regex=r"^(sidereal|tropical)$")
    ayanamsa: Optional[str] = "lahiri"

class Body(BaseModel):
    name: str
    longitude_deg: float
    latitude_deg: float = 0.0
    speed_deg_per_day: float = 0.0
    retrograde: bool = False

class Chart(BaseModel):
    mode: str
    ayanamsa_deg: float = 24.1
    jd_ut: float
    jd_tt: float
    bodies: List[Body]

class Houses(BaseModel):
    house_system: str = "Placidus"
    asc_deg: float
    mc_deg: float
    cusps_deg: List[float]
    high_lat_fallback: bool = False
    warnings: List[str] = []

class PredictionRequest(ChartRequest):
    horizon: Optional[str] = Field("short", regex=r"^(short|medium|long)$") 

class Evidence(BaseModel):
    dasha: float
    transit: float
    varga: float
    yoga: float

class Prediction(BaseModel):
    prediction_id: str
    domain: str
    horizon: Optional[str] = "short"
    interval_start_utc: str
    interval_end_utc: str
    probability_calibrated: float
    hc_flag: bool = False
    abstained: bool = False
    evidence: Evidence
    mode: str
    ayanamsa_deg: float
    notes: Optional[str] = ""

class RectificationRequest(ChartRequest):
    window_minutes: int = 90

class RectificationResult(BaseModel):
    best_time: str
    top3_times: List[str]
    composite_scores: List[float]
    confidence_band: str
    margin_delta: float
    features_at_peak: str
