# app/api/routes.py
from __future__ import annotations
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo

from flask import Blueprint, jsonify, request

from app.version import VERSION
from app.utils.config import load_config
from app.utils.metrics import metrics
from app.utils.ratelimit import rate_limit
from app.core.validators import (
    ValidationError,
    parse_chart_payload,
    parse_rectification_payload,
)

# canonical chart call
from app.core.astronomy import compute_chart

# NOTE: predict() removed; use new /api/predictions blueprint
from app.core.rectify import rectification_candidates

# houses: prefer policy façade if present
try:
    from app.core.house import (
        compute_houses_with_policy as houses_fn,   # type: ignore
        list_supported_house_systems as list_house_systems,  # type: ignore
    )
    HOUSES_KIND = "policy"
except Exception:
    from app.core.astronomy import compute_houses as houses_fn  # type: ignore
    def list_house_systems() -> list[str]:
        return ["placidus", "koch", "regiomontanus", "campanus", "equal", "porphyry"]
    HOUSES_KIND = "legacy"

# time kernel
from app.core import time_kernel as _tk

# optional leap-seconds helper
try:
    from app.core import leapseconds as _leaps
except Exception:
    _leaps = None  # type: ignore

api = Blueprint("api", __name__)

DEBUG_VERBOSE = os.getenv("ASTRO_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "on")

# ───────────────────────────── helpers ─────────────────────────────

def get_bool(d: dict, key: str, default: bool = False) -> bool:
    """Extract boolean from dict with flexible string parsing."""
    val = d.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)

def datetime_to_jd_utc(dt: datetime) -> float:
    """Convert datetime to Julian Day UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    
    # Julian Day calculation
    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    
    if m <= 2:
        y -= 1
        m += 12
        
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
    
    # Add time fraction
    time_fraction = (dt_utc.hour + dt_utc.minute/60.0 + dt_utc.second/3600.0) / 24.0
    jd += time_fraction
    
    return jd

def find_kernel_callable(module, name: str) -> Optional[Callable]:
    """Find callable in time kernel module."""
    if hasattr(module, name):
        attr = getattr(module, name)
        if callable(attr):
            return attr
    return None

def compute_timescales_from_local(date: str, time: str, place_tz: str) -> Dict[str, Any]:
    """Compute timescales from local date/time."""
    try:
        # Parse the local datetime
        dt_str = f"{date}T{time}"
        if place_tz:
            tz = ZoneInfo(place_tz)
            dt_local = datetime.fromisoformat(dt_str).replace(tzinfo=tz)
        else:
            dt_local = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
        
        jd_utc = datetime_to_jd_utc(dt_local)
        
        # Try to use time kernel functions if available
        timescales = {"jd_utc": jd_utc}
        
        if _tk:
            try:
                jd_tt_fn = find_kernel_callable(_tk, "jd_utc_to_jd_tt")
                if jd_tt_fn:
                    jd_tt = jd_tt_fn(jd_utc)
                    timescales["jd_tt"] = jd_tt
                
                delta_t_fn = find_kernel_callable(_tk, "delta_t")
                if delta_t_fn:
                    delta_t = delta_t_fn(jd_utc)
                    timescales["delta_t"] = delta_t
                
                if _leaps:
                    delta_at_fn = find_kernel_callable(_leaps, "delta_at")
                    if delta_at_fn:
                        delta_at = delta_at_fn(jd_utc)
                        timescales["delta_at"] = delta_at
                    
                    dut1_fn = find_kernel_callable(_leaps, "dut1")
                    if dut1_fn:
                        dut1 = dut1_fn(jd_utc)
                        timescales["dut1"] = dut1
                        
            except Exception as e:
                if DEBUG_VERBOSE:
                    print(f"Time kernel error: {e}")
        
        # Fallback values
        if "jd_tt" not in timescales:
            timescales["jd_tt"] = jd_utc + 69.184 / 86400.0  # Approximate TT-UTC
        if "delta_t" not in timescales:
            timescales["delta_t"] = 69.184
        if "delta_at" not in timescales:
            timescales["delta_at"] = 32.184
        if "dut1" not in timescales:
            timescales["dut1"] = 0.0
            
        return timescales
        
    except Exception as e:
        return {
            "jd_utc": 2451545.0,  # J2000.0 fallback
            "jd_tt": 2451545.0 + 69.184 / 86400.0,
            "delta_t": 69.184,
            "delta_at": 32.184,
            "dut1": 0.0,
            "error": str(e)
        }

def sig_accepts(func: Callable, param: str) -> bool:
    """Check if function signature accepts parameter."""
    try:
        sig = inspect.signature(func)
        return param in sig.parameters
    except Exception:
        return False

def call_compute_chart(chart_request) -> Dict[str, Any]:
    """Call compute_chart with proper parameters."""
    try:
        if sig_accepts(compute_chart, "verbose"):
            return compute_chart(chart_request, verbose=DEBUG_VERBOSE)
        else:
            return compute_chart(chart_request)
    except Exception as e:
        return {"error": str(e)}

def call_compute_houses(chart_request) -> Dict[str, Any]:
    """Call houses computation function."""
    try:
        if sig_accepts(houses_fn, "verbose"):
            return houses_fn(chart_request, verbose=DEBUG_VERBOSE)
        else:
            return houses_fn(chart_request)
    except Exception as e:
        return {"error": str(e)}

def json_error(message: str, code: int = 400) -> tuple:
    """Return JSON error response."""
    return jsonify({"ok": False, "error": message}), code

def normalize_houses_payload(houses_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize houses data for consistent API response."""
    if not houses_data or "error" in houses_data:
        return houses_data
    
    # Ensure required fields exist
    normalized = dict(houses_data)
    
    if "asc_deg" not in normalized and "asc" in normalized:
        normalized["asc_deg"] = normalized["asc"]
    if "mc_deg" not in normalized and "mc" in normalized:
        normalized["mc_deg"] = normalized["mc"]
    if "cusps_deg" not in normalized and "cusps" in normalized:
        normalized["cusps_deg"] = normalized["cusps"]
    
    return normalized

# ───────────────────────────── endpoints ─────────────────────────────

@api.get("/api/health")
def health():
    return jsonify({"ok": True, "status": "up", "version": VERSION}), 200

@api.get("/api/houses/systems")
def houses_systems():
    try:
        systems = list_house_systems()
    except Exception:
        systems = []
    return jsonify({"ok": True, "engine": HOUSES_KIND, "systems": systems}), 200

@api.post("/api/calculate")
@metrics.middleware("calculate")
@rate_limit(10)
def calculate():
    """Main chart calculation endpoint."""
    try:
        # Parse and validate request
        data = request.get_json()
        if not data:
            return json_error("No JSON data provided")
        
        chart_request = parse_chart_payload(data)
        
        # Compute timescales
        timescales = compute_timescales_from_local(
            chart_request.date, 
            chart_request.time, 
            chart_request.place_tz
        )
        
        # Compute houses
        houses_data = call_compute_houses(chart_request)
        houses_data = normalize_houses_payload(houses_data)
        
        if "error" in houses_data:
            return json_error(f"Houses calculation failed: {houses_data['error']}")
        
        # Optionally compute full chart data
        chart_data = None
        if get_bool(data, "include_planets", False):
            chart_data = call_compute_chart(chart_request)
            if "error" in chart_data:
                chart_data = None  # Don't fail the whole request for chart errors
        
        # Build response
        response = {
            "ok": True,
            "houses": houses_data,
            "meta": {
                "timescales": timescales,
                "request": {
                    "date": chart_request.date,
                    "time": chart_request.time,
                    "latitude": chart_request.latitude,
                    "longitude": chart_request.longitude,
                    "place_tz": chart_request.place_tz
                }
            }
        }
        
        if chart_data:
            response["chart"] = chart_data
        
        return jsonify(response), 200
        
    except ValidationError as e:
        return json_error(str(e), 400)
    except Exception as e:
        if DEBUG_VERBOSE:
            import traceback
            traceback.print_exc()
        return json_error(f"Internal calculation error: {str(e)}", 500)

@api.post("/api/report")
@metrics.middleware("report")  
@rate_limit(5)
def report():
    """Extended chart report with interpretation."""
    try:
        data = request.get_json()
        if not data:
            return json_error("No JSON data provided")
        
        chart_request = parse_chart_payload(data)
        
        # Compute full chart
        chart_data = call_compute_chart(chart_request)
        if "error" in chart_data:
            return json_error(f"Chart calculation failed: {chart_data['error']}")
        
        # Compute houses
        houses_data = call_compute_houses(chart_request)
        houses_data = normalize_houses_payload(houses_data)
        if "error" in houses_data:
            return json_error(f"Houses calculation failed: {houses_data['error']}")
        
        # Compute timescales
        timescales = compute_timescales_from_local(
            chart_request.date,
            chart_request.time, 
            chart_request.place_tz
        )
        
        response = {
            "ok": True,
            "chart": chart_data,
            "houses": houses_data,
            "meta": {
                "timescales": timescales,
                "request": asdict(chart_request) if is_dataclass(chart_request) else dict(chart_request)
            }
        }
        
        return jsonify(response), 200
        
    except ValidationError as e:
        return json_error(str(e), 400)
    except Exception as e:
        if DEBUG_VERBOSE:
            import traceback
            traceback.print_exc()
        return json_error(f"Internal report error: {str(e)}", 500)

@api.post("/rectification/quick")
@metrics.middleware("rectification")
@rate_limit(2)
def rect_quick():
    """Quick birth time rectification."""
    try:
        data = request.get_json()
        if not data:
            return json_error("No JSON data provided")
        
        rect_request = parse_rectification_payload(data)
        
        # Call rectification function
        candidates = rectification_candidates(rect_request)
        
        return jsonify({
            "ok": True,
            "candidates": candidates,
            "meta": {
                "request": asdict(rect_request) if is_dataclass(rect_request) else dict(rect_request)
            }
        }), 200
        
    except ValidationError as e:
        return json_error(str(e), 400)
    except Exception as e:
        if DEBUG_VERBOSE:
            import traceback
            traceback.print_exc()
        return json_error(f"Rectification error: {str(e)}", 500)

@api.get("/api/openapi")
def openapi_spec():
    """OpenAPI specification."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Astrological Calculation API",
            "version": VERSION,
            "description": "Professional astrological calculation service"
        },
        "paths": {
            "/api/calculate": {
                "post": {
                    "summary": "Calculate astrological chart",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["date", "time", "latitude", "longitude"],
                                    "properties": {
                                        "date": {"type": "string", "format": "date"},
                                        "time": {"type": "string"},
                                        "latitude": {"type": "number"},
                                        "longitude": {"type": "number"},
                                        "place_tz": {"type": "string"},
                                        "house_system": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Calculation successful"},
                        "400": {"description": "Invalid request"},
                        "500": {"description": "Calculation error"}
                    }
                }
            }
        }
    }
    
    return jsonify(spec), 200

@api.get("/system-validation")
def system_validation():
    """System validation and policy information."""
    try:
        # Get available house systems
        available_systems = []
        try:
            available_systems = list_house_systems()
        except Exception:
            pass
        
        policy_info = {
            "houses_engine": HOUSES_KIND,
            "polar_policy": {
                "soft_fallback_lat_gt": 66.0,
                "hard_reject_lat_ge": 80.0
            },
            "available_systems": available_systems,
            "version": VERSION
        }
        
        return jsonify({
            "ok": True,
            "policy": policy_info
        }), 200
        
    except Exception as e:
        return json_error(f"System validation error: {str(e)}", 500)

@api.get("/api/config")
@metrics.middleware("config")
@rate_limit(1)
def config_info():
    """Configuration information."""
    try:
        config_data = {
            "version": VERSION,
            "houses_engine": HOUSES_KIND,
            "debug_verbose": DEBUG_VERBOSE,
            "features": {
                "time_kernel": _tk is not None,
                "leap_seconds": _leaps is not None,
                "policy_houses": HOUSES_KIND == "policy"
            }
        }
        
        return jsonify({
            "ok": True,
            "config": config_data
        }), 200
        
    except Exception as e:
        return json_error(f"Config error: {str(e)}", 500)
