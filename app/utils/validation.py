# Add these validators to your existing validation.py file
# Just add these @validator methods to your existing ChartRequest class

from datetime import datetime
import pytz

# Add these validators to your existing ChartRequest class:

@validator('date')
def validate_date(cls, v):
    """Validate date format and catch invalid dates before they cause 500 errors"""
    try:
        datetime.strptime(v, "%Y-%m-%d")
        return v
    except ValueError:
        raise ValueError('date must be in YYYY-MM-DD format')

@validator('time') 
def validate_time(cls, v):
    """Validate time format and catch invalid times before they cause 500 errors"""
    try:
        datetime.strptime(v, "%H:%M")
        return v
    except ValueError:
        raise ValueError('time must be in HH:MM format (24-hour)')

@validator('latitude')
def validate_latitude(cls, v):
    """Validate latitude bounds"""
    if not -90 <= v <= 90:
        raise ValueError('latitude must be between -90 and 90 degrees')
    return v

@validator('longitude')
def validate_longitude(cls, v):
    """Validate longitude bounds"""
    if not -180 <= v <= 180:
        raise ValueError('longitude must be between -180 and 180 degrees')
    return v

@validator('place_tz')
def validate_timezone(cls, v):
    """Validate timezone string"""
    try:
        pytz.timezone(v)
        return v
    except pytz.UnknownTimeZoneError:
        raise ValueError(f'Invalid timezone: {v}. Use format like "America/New_York"')

# Also add these same validators to your PredictionRequest and RectificationRequest classes
# since they inherit from ChartRequest, they'll automatically get these validations

# If you want to make some fields optional for minimal requests, 
# you can add these alternative minimal models:

class MinimalChartRequest(BaseModel):
    """For endpoints that can work with just date, time, place_tz"""
    date: str
    time: str
    place_tz: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    mode: str = Field("sidereal")
    ayanamsa: Optional[str] = "lahiri"
    
    # Include all the same validators as above

class MinimalPredictionRequest(MinimalChartRequest):
    """Minimal prediction request"""
    horizon: Optional[str] = Field("short")
    
    @validator('horizon')
    def validate_horizon(cls, v):
        if v and v not in ['short', 'medium', 'long']:
            raise ValueError('horizon must be short, medium, or long')
        return v

class MinimalRectificationRequest(MinimalChartRequest):
    """Minimal rectification request"""
    window_minutes: int = 90
