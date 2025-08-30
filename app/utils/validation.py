from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ChartRequest(BaseModel):
    date: str
    time: str
    place_tz: str
    latitude: float
    longitude: float
    mode: str = Field("sidereal")
    ayanamsa: Optional[str] = "lahiri"
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['sidereal', 'tropical']:
            raise ValueError('mode must be sidereal or tropical')
        return v

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
    horizon: Optional[str] = Field("short")
    
    @validator('horizon')
    def validate_horizon(cls, v):
        if v and v not in ['short', 'medium', 'long']:
            raise ValueError('horizon must be short, medium, or long')
        return v

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

class RectificationRequest(BaseModel):
    date: str
    time: str
    place_tz: str
    latitude: float
    longitude: float
    mode: str = Field("sidereal")
    ayanamsa: Optional[str] = "lahiri"
    window_minutes: int = 90
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['sidereal', 'tropical']:
            raise ValueError('mode must be sidereal or tropical')
        return v

class RectificationResult(BaseModel):
    best_time: str
    top3_times: List[str]
    composite_scores: List[float]
    confidence_band: str
    margin_delta: float
    features_at_peak: str
