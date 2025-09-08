# app/core/paran.py
# -*- coding: utf-8 -*-
"""
Parans (v11): Local co-risings/culminations/settings/anti-culminations

Public API
----------
compute_parans(
    subject: dict,                    # minimal natal-like dict (date,time,place_tz) for timescale resolution
    *,
    place: dict,                      # {latitude, longitude, elev_m}
    jd_tt_ref: float | None = None,   # reference epoch (TT); if None, resolved from subject
    jd_ut1_ref: float | None = None,  # reference epoch (UT1); if None, resolved from subject
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    bodies: tuple[str, ...] = ("Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"),
    tolerance_minutes: float = 4.0,   # max separation to qualify as a paran (absolute time diff)
    search_window_days: float = 1.0,  # events searched within ±window/2 around reference
    max_iters: int = 10,
    fd_step_minutes: float = 2.0,     # numeric derivative step for root-finding
    # Atmosphere / Earth (all optional; defaults keep legacy spherical/no-refraction behavior)
    earth_model: str = "spherical",   # "spherical" | "wgs84"
    apply_refraction: bool = False,   # Saemundsson near-horizon refraction
    pressure_hPa: float = 1010.0,
    temperature_C: float = 10.0,
    # Diagnostics
    profile: bool = False,            # include meta.profile timings
    validation: str = "basic",        # "none" | "basic"
) -> dict

Outputs
-------
{
  "ok": true,
  "meta": {
      "frame": "...", "zodiac_mode": "...", "ayanamsa_deg": float,
      "timescales_ref": {"jd_tt":..,"jd_ut1":..}, "place": {...},
      "earth_model": "spherical"|"wgs84", "refraction": {...},
      "tolerance_minutes": float, "notes":[...], "warnings":[...], "profile": {...}?
  },
  "events_by_body": {
      "Sun": [
        {"type":"RISE","jd_ut1":..,"jd_tt":..,"az_deg":..,"iterations":int,"converged":bool,"corrections":{"dip_deg":..,"refraction_deg":..}},
        {"type":"CULM",...}, {"type":"SET",...}, {"type":"ANTI",...}
      ],
      ...
  },
  "parans": [
      {
        "pair": "A_RISE ~ B_CULM",
        "a": {"body":"A","type":"RISE","jd_ut1":..,"jd_tt":..},
        "b": {"body":"B","type":"CULM","jd_ut1":..,"jd_tt":..},
        "delta_minutes": float,
        "within_tolerance": true|false
      },
      ...
  ]
}

Notes & Conventions
-------------------
- Strict timescales: prefers supplied jd_tt_ref & jd_ut1_ref. If missing, resolves via
  app.core.timescales.build_timescales(date,time,tz, dut1_seconds=0.0) and warns.
- Topocentric positions are used for event solving (horizon phenomena need parallax, esp. Moon).
- Refraction: Saemundsson (1986) near-horizon formula (scaled by pressure/temperature).
- Horizon dip: sqrt(2h/R) with WGS-84 or spherical Earth; h = observer elevation (meters).
- RA/Dec are derived from ecliptic-of-date lon/lat via mean obliquity; when zodiac_mode="sidereal",
  lon is offset by ayanamsa before conversion to RA/Dec to keep internal frames consistent with the rest of v11.
- Event solving uses Newton with numeric derivatives (central difference) and robust wrap-aware residuals.
- "Basic" validation checks that each body yields up to one of each event in-window and highlights circumpolar cases.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable, Tuple
from time import perf_counter
import math
import inspect

# ── resilient imports ─────────────────────────────────────────────────────────
try:
    from app.core.ephemeris_adapter import EphemerisAdapter
except Exception as _e:
    EphemerisAdapter = None  # type: ignore
    _EPH_ERR = _e

try:
    from app.core.timescales import build_timescales
except Exception as _e:
    build_timescales = None  # type: ignore
    _TS_ERR = _e


# ── constants ─────────────────────────────────────────────────────────────────
MAJORS: Tuple[str, ...] = (
    "Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"
)

GMST_RATE_DEG_PER_DAY = 360.98564736629  # mean sidereal rate
MEAN_EARTH_R_M = 6371008.8               # authalic mean radius
# WGS-84
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)


# ── math helpers ──────────────────────────────────────────────────────────────
def _wrap_deg(x: float) -> float:
    x = math.fmod(x, 360.0)
    return x + 360.0 if x < 0.0 else x

def _wrap_pm180(x: float) -> float:
    """Wrap to (-180, +180]."""
    y = (x + 180.0) % 360.0 - 180.0
    return y if y != -180.0 else 180.0

def _warn(ws: List[str], msg: str) -> None:
    if msg not in ws:
        ws.append(msg)

def _mean_obliquity_iau2006(jd_tt: float) -> float:
    T = (jd_tt - 2451545.0) / 36525.0
    eps0 = 84381.406 \
         - 46.836769*T \
         - 0.0001831*(T**2) \
         + 0.00200340*(T**3) \
         - 0.000000576*(T**4) \
         - 0.0000000434*(T**5)
    return eps0 / 3600.0

def _gmst_deg(jd_ut1: float) -> float:
    d = jd_ut1 - 2451545.0
    T = d / 36525.0
    gmst = 280.46061837 + GMST_RATE_DEG_PER_DAY * d + 0.000387933 * (T*T) - (T*T*T) / 38710000.0
    return _wrap_deg(gmst)

def _ecl_to_equ(lon_deg: float, lat_deg: float, jd_tt: float) -> Tuple[float, float]:
    """Ecliptic (λ,β) → Equatorial (α,δ), degrees."""
    eps = math.radians(_mean_obliquity_iau2006(jd_tt))
    lam = math.radians(_wrap_deg(lon_deg))
    beta = math.radians(lat_deg)
    # RA
    y = math.sin(lam) * math.cos(eps) - math.tan(beta) * math.sin(eps)
    x = math.cos(lam)
    alpha = math.degrees(math.atan2(y, x)) % 360.0
    # Dec
    s = math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam)
    delta = math.degrees(math.asin(max(-1.0, min(1.0, s))))
    return alpha, delta

def _earth_radius_m(latitude_deg: float, model: str) -> float:
    if model.lower() != "wgs84":
        return MEAN_EARTH_R_M
    phi = math.radians(latitude_deg)
    a2 = _WGS84_A * _WGS84_A
    b2 = _WGS84_B * _WGS84_B
    cosp = math.cos(phi)
    sinp = math.sin(phi)
    num = (a2*a2*cosp*cosp) + (b2*b2*sinp*sinp)
    den = (a2*cosp*cosp) + (b2*sinp*sinp)
    return math.sqrt(num / max(1e-9, den))

def _horizon_dip_deg(elev_m: float, latitude_deg: float, model: str) -> float:
    if elev_m <= 0.0:
        return 0.0
    R = _earth_radius_m(latitude_deg, model)
    return math.degrees(math.sqrt(2.0 * float(elev_m) / R))

def _saemundsson_refraction_deg(h_deg: float, pressure_hPa: float, temperature_C: float) -> float:
    """Saemundsson (1986) near-horizon refraction (deg), clamped."""
    h = max(-1.0, min(89.9, float(h_deg)))
    arg = math.radians(h + 10.3 / (h + 5.11))
    R_arcmin = 1.02 / max(1e-6, math.tan(arg))
    scale = (pressure_hPa / 1010.0) * (283.0 / (273.0 + float(temperature_C)))
    return min((R_arcmin * scale) / 60.0, 1.0)

def _alt_az_deg(alpha: float, delta: float, lat_deg: float, lst_deg: float) -> Tuple[float, float]:
    """Return (altitude, azimuth[N→E]) in degrees."""
    H = math.radians(_wrap_pm180(lst_deg - alpha))
    phi = math.radians(lat_deg)
    sd = math.sin(math.radians(delta))
    cd = math.cos(math.radians(delta))
    sh = math.sin(phi) * sd + math.cos(phi) * cd * math.cos(H)
    h = math.degrees(math.asin(max(-1.0, min(1.0, sh))))
    # Azimuth (from North, towards East)
    cosh = max(1e-9, math.cos(math.radians(h)))
    sinA = -cd * math.sin(H) / cosh
    cosA = (sd - math.sin(phi) * math.sin(math.radians(h))) / (math.cos(phi) * cosh + 1e-12)
    A = math.degrees(math.atan2(sinA, cosA)) % 360.0
    return h, A


# ── ephemeris helpers with defensive error handling ───────────────────────────
def _rows_for_bodies(
    jd_tt: float,
    bodies: Iterable[str],
    *,
    place: Optional[Dict[str, float]],
    frame: str,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """
    Get ephemeris data for bodies with comprehensive error handling.
    """
    if EphemerisAdapter is None:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPH_ERR}")
    
    try:
        adapter = EphemerisAdapter(frame=frame)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize EphemerisAdapter: {type(e).__name__}: {e}")
    
    bodies_list = list(bodies)
    if not bodies_list:
        raise ValueError("No bodies specified for ephemeris calculation")
    
    kwargs = {"jd_tt": float(jd_tt), "bodies": bodies_list, "center": "geocentric"}
    if place:
        kwargs.update({
            "center": "topocentric",
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
            "elevation_m": float(place.get("elev_m", 0.0)),
        })
    
    # Try available ephemeris methods
    last_error = None
    for method_name in ("ecliptic_longitudes_and_velocities", "ecliptic_longitudes"):
        if not hasattr(adapter, method_name):
            continue
        
        try:
            method = getattr(adapter, method_name)
            sig = inspect.signature(method)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            
            result = method(**filtered_kwargs)
            if result is None:
                _warn(warnings, f"ephemeris_method_{method_name}_returned_none")
                continue
            
            rows: List[Dict[str, Any]] = []
            
            # Handle different result formats
            if isinstance(result, dict):
                if not result:
                    _warn(warnings, f"ephemeris_method_{method_name}_returned_empty_dict")
                    continue
                
                for body_name, body_data in result.items():
                    try:
                        if isinstance(body_data, (int, float)):
                            rows.append({"name": str(body_name), "lon": float(body_data), "lat": 0.0})
                        elif isinstance(body_data, dict):
                            if "lon" not in body_data and "longitude" not in body_data:
                                _warn(warnings, f"ephemeris_missing_longitude_for_{body_name}")
                                continue
                            
                            row = {
                                "name": str(body_name), 
                                "lon": float(body_data.get("lon", body_data.get("longitude", 0.0))),
                                "lat": float(body_data.get("lat", body_data.get("latitude", 0.0)))
                            }
                            rows.append(row)
                        else:
                            _warn(warnings, f"ephemeris_unexpected_format_for_{body_name}")
                    except (ValueError, TypeError) as e:
                        _warn(warnings, f"ephemeris_data_conversion_error_{body_name}:{type(e).__name__}")
                        continue
            
            elif isinstance(result, list):
                if not result:
                    _warn(warnings, f"ephemeris_method_{method_name}_returned_empty_list")
                    continue
                
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    
                    try:
                        name = str(item.get("name") or item.get("body") or "unknown")
                        lon_key = "lon" if "lon" in item else "longitude"
                        lat_key = "lat" if "lat" in item else "latitude"
                        
                        if lon_key not in item:
                            _warn(warnings, f"ephemeris_missing_longitude_for_{name}")
                            continue
                        
                        row = {
                            "name": name,
                            "lon": float(item[lon_key]),
                            "lat": float(item.get(lat_key, 0.0))
                        }
                        rows.append(row)
                    except (ValueError, TypeError) as e:
                        _warn(warnings, f"ephemeris_data_conversion_error:{type(e).__name__}")
                        continue
            else:
                _warn(warnings, f"ephemeris_method_{method_name}_unexpected_result_type")
                continue
            
            # Validate we got results
            if not rows:
                _warn(warnings, f"ephemeris_method_{method_name}_produced_no_valid_rows")
                continue
            
            # Validate we have the requested bodies
            found_bodies = {row["name"] for row in rows}
            missing_bodies = set(bodies_list) - found_bodies
            if missing_bodies:
                _warn(warnings, f"ephemeris_missing_bodies:{','.join(missing_bodies)}")
            
            return rows
            
        except Exception as e:
            last_error = e
            _warn(warnings, f"ephemeris_method_{method_name}_failed:{type(e).__name__}")
            continue
    
    # If we get here, all methods failed
    error_msg = f"No usable ephemeris method. Last error: {type(last_error).__name__}: {last_error}" if last_error else "No usable ephemeris method found."
    raise RuntimeError(error_msg)

def _ra_dec_for_body(
    jd_tt: float,
    body: str,
    *,
    place: Optional[Dict[str, float]],
    frame: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    warnings: List[str],
) -> Tuple[float, float]:
    """
    Get RA/Dec for a single body with error handling.
    """
    try:
        rows = _rows_for_bodies(jd_tt, (body,), place=place, frame=frame, warnings=warnings)
        
        if not rows:
            raise ValueError(f"No ephemeris data returned for body '{body}'")
        
        # Find the requested body
        body_row = None
        for row in rows:
            if row["name"] == body:
                body_row = row
                break
        
        if body_row is None:
            available = [row["name"] for row in rows]
            raise ValueError(f"Body '{body}' not found in ephemeris results. Available: {available}")
        
        lon = float(body_row["lon"])
        lat = float(body_row.get("lat", 0.0))
        
        # Apply sidereal correction if needed
        if zodiac_mode.lower() == "sidereal":
            lon = _wrap_deg(lon - ayanamsa_deg)
        
        alpha, delta = _ecl_to_equ(lon, lat, jd_tt)
        return alpha, delta
        
    except Exception as e:
        raise RuntimeError(f"Failed to get RA/Dec for body '{body}': {type(e).__name__}: {e}")


# ── timescale resolution ──────────────────────────────────────────────────────
def _resolve_timescales(
    subject: Dict[str, Any],
    jd_tt_ref: Optional[float],
    jd_ut1_ref: Optional[float],
    warnings: List[str],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Resolve timescales with comprehensive error handling.
    """
    if jd_tt_ref is not None and jd_ut1_ref is not None:
        try:
            jd_tt = float(jd_tt_ref)
            jd_ut1 = float(jd_ut1_ref)
            
            # Basic sanity checks
            if not (1000000.0 <= jd_tt <= 5000000.0):
                _warn(warnings, f"unusual_jd_tt_value:{jd_tt}")
            if not (1000000.0 <= jd_ut1 <= 5000000.0):
                _warn(warnings, f"unusual_jd_ut1_value:{jd_ut1}")
            
            return jd_tt, jd_ut1, {
                "jd_tt": jd_tt, 
                "jd_ut1": jd_ut1, 
                "dut1_assumed": None,
                "source": "provided"
            }
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid timescale values: {type(e).__name__}: {e}")
    
    if build_timescales is None:
        raise RuntimeError(f"Timescales module unavailable ({_TS_ERR}) and strict values not supplied.")
    
    # Extract required fields
    date = subject.get("date")
    time = subject.get("time") 
    tz = subject.get("place_tz")
    
    if not date:
        raise ValueError("Missing 'date' field in subject for timescale resolution")
    if not time:
        raise ValueError("Missing 'time' field in subject for timescale resolution")
    if not tz:
        raise ValueError("Missing 'place_tz' field in subject for timescale resolution")
    
    try:
        ts = build_timescales(date_str=str(date), time_str=str(time), tz_name=str(tz), dut1_seconds=0.0)
        
        if not isinstance(ts, dict):
            raise ValueError(f"build_timescales returned unexpected type: {type(ts)}")
        
        if "jd_tt" not in ts or "jd_ut1" not in ts:
            raise ValueError(f"build_timescales missing required fields. Got: {list(ts.keys())}")
        
        jd_tt = float(ts["jd_tt"])
        jd_ut1 = float(ts["jd_ut1"])
        
        _warn(warnings, "strict_missing→computed_timescales_with_dut1=0.0s")
        
        return jd_tt, jd_ut1, {
            "jd_tt": jd_tt, 
            "jd_ut1": jd_ut1, 
            "dut1_assumed": 0.0,
            "source": "computed"
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to build timescales from subject data: {type(e).__name__}: {e}")


# ── event solvers with enhanced error handling ────────────────────────────────
def _lst_deg(jd_ut1: float, longitude_east_deg: float) -> float:
    """Local Sidereal Time in degrees."""
    return _wrap_deg(_gmst_deg(jd_ut1) + float(longitude_east_deg))

def _event_seed(
    kind: str, alpha0: float, delta0: float, lat_deg: float, lst0: float
) -> Optional[float]:
    """
    Return H (hour-angle, deg) seed for event (RISE/SET/CULM/ANTI) using δ at reference.
    Returns None for impossible events (circumpolar).
    """
    try:
        k = kind.upper()
        if k == "CULM":
            return 0.0
        if k == "ANTI":
            return 180.0
        
        # For RISE/SET: cos H0 = -tan φ tan δ
        lat_rad = math.radians(lat_deg)
        delta_rad = math.radians(delta0)
        
        # Check for extreme latitudes or declinations
        if abs(lat_deg) > 89.9:
            return None  # Too close to pole
        
        tphi = math.tan(lat_rad)
        tdel = math.tan(delta_rad)
        arg = -tphi * tdel
        
        # Check if rise/set is possible
        if arg < -1.0 or arg > 1.0:
            return None  # Circumpolar: no rise/set
        
        H0 = math.degrees(math.acos(arg))
        return -H0 if k == "RISE" else +H0
        
    except (ValueError, OverflowError):
        return None  # Mathematical error, treat as impossible

def _find_event_time(
    body: str,
    kind: str,                         # "RISE" | "SET" | "CULM" | "ANTI"
    jd_tt_ref: float,
    jd_ut1_ref: float,
    place: Dict[str, float],
    frame: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    *,
    max_iters: int,
    fd_step_minutes: float,
    apply_refraction: bool,
    pressure_hPa: float,
    temperature_C: float,
    earth_model: str,
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Solve for event time near the reference epoch with comprehensive error handling.
    Returns dict with times & azimuth, or None if not applicable (e.g., circumpolar).
    """
    try:
        # Validate inputs
        if not body:
            raise ValueError("Body name cannot be empty")
        
        kind = kind.upper()
        if kind not in ("RISE", "SET", "CULM", "ANTI"):
            raise ValueError(f"Invalid event kind: {kind}")
        
        lat = float(place["latitude"])
        lonE = float(place["longitude"])
        elev = float(place.get("elev_m", 0.0))
        
        # Validate coordinates
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180.0 <= lonE <= 180.0):
            raise ValueError(f"Invalid longitude: {lonE}")
        
        # Get reference RA/Dec and LST
        try:
            alpha0, delta0 = _ra_dec_for_body(
                jd_tt_ref, body, 
                place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg,
                warnings=warnings
            )
        except Exception as e:
            _warn(warnings, f"event_{body}_{kind}_ephemeris_failed:{type(e).__name__}")
            return None
        
        lst0 = _lst_deg(jd_ut1_ref, lonE)
        
        # Check if event is possible
        H_seed = _event_seed(kind, alpha0, delta0, lat, lst0)
        if H_seed is None:
            _warn(warnings, f"event_{body}_{kind}_circumpolar_or_impossible")
            return None
        
        # Calculate target altitude with corrections
        dip_deg = _horizon_dip_deg(elev, lat, earth_model)
        refr0_deg = _saemundsson_refraction_deg(0.0, pressure_hPa, temperature_C) if apply_refraction else 0.0
        h0_deg = -dip_deg + refr0_deg if kind in ("RISE", "SET") else None
        
        # Initial time guess
        x0_deg = _wrap_pm180((lst0 - alpha0) - H_seed)
        dt_seed_days = -x0_deg / GMST_RATE_DEG_PER_DAY
        
        # Iteration
        dt = dt_seed_days
        converged = False
        iterations = 0
        
        for iteration in range(1, max_iters + 1):
            iterations = iteration
            jd_ut1 = jd_ut1_ref + dt
            jd_tt = jd_tt_ref + dt
            
            try:
                alpha, delta = _ra_dec_for_body(
                    jd_tt, body,
                    place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg,
                    warnings=warnings
                )
            except Exception as e:
                _warn(warnings, f"event_{body}_{kind}_ephemeris_iteration_failed:{type(e).__name__}")
                break
            
            lst = _lst_deg(jd_ut1, lonE)
            
            # Calculate residual based on event type
            if kind in ("CULM", "ANTI"):
                target = 0.0 if kind == "CULM" else 180.0
                f = _wrap_pm180((lst - alpha) - target)
                
                # Numerical derivative for hour angle
                h = max(1e-6, float(fd_step_minutes) / 1440.0)
                try:
                    alpha_p, _ = _ra_dec_for_body(jd_tt + h, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg, warnings=warnings)
                    alpha_m, _ = _ra_dec_for_body(jd_tt - h, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg, warnings=warnings)
                    adot = _wrap_pm180(alpha_p - alpha_m) / (2.0 * h)
                    dfdt = GMST_RATE_DEG_PER_DAY - adot
                except Exception:
                    _warn(warnings, f"event_{body}_{kind}_derivative_calculation_failed")
                    break
            else:
                # RISE/SET: solve altitude equation
                alt, _ = _alt_az_deg(alpha, delta, lat, lst)
                f = alt - float(h0_deg)
                
                # Numerical derivative for altitude
                h = max(1e-6, float(fd_step_minutes) / 1440.0)
                try:
                    alpha_p, delta_p = _ra_dec_for_body(jd_tt + h, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg, warnings=warnings)
                    lst_p = _lst_deg(jd_ut1 + h, lonE)
                    alt_p, _ = _alt_az_deg(alpha_p, delta_p, lat, lst_p)
                    
                    alpha_m, delta_m = _ra_dec_for_body(jd_tt - h, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg, warnings=warnings)
                    lst_m = _lst_deg(jd_ut1 - h, lonE)
                    alt_m, _ = _alt_az_deg(alpha_m, delta_m, lat, lst_m)
                    
                    dfdt = (alt_p - alt_m) / (2.0 * h)
                except Exception:
                    _warn(warnings, f"event_{body}_{kind}_derivative_calculation_failed")
                    break
            
            # Check for convergence
            if abs(f) <= 0.01:  # 0.01 deg precision
                converged = True
                break
            
            # Newton step with safeguards
            if abs(dfdt) < 1e-6:
                _warn(warnings, f"event_{body}_{kind}_zero_derivative")
                break
            
            step = -f / dfdt
            if abs(step) > 0.5:  # Limit step size
                step = math.copysign(0.5, step)
            
            dt += step
        
        # Final evaluation
        jd_ut1_final = jd_ut1_ref + dt
        jd_tt_final = jd_tt_ref + dt
        
        try:
            alpha_final, delta_final = _ra_dec_for_body(
                jd_tt_final, body,
                place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg,
                warnings=warnings
            )
            lst_final = _lst_deg(jd_ut1_final, lonE)
            alt_final, az_final = _alt_az_deg(alpha_final, delta_final, lat, lst_final)
        except Exception as e:
            _warn(warnings, f"event_{body}_{kind}_final_evaluation_failed:{type(e).__name__}")
            return None
        
        if not converged:
            _warn(warnings, f"event_{body}_{kind}_not_converged_after_{max_iters}_iterations")
        
        return {
            "type": kind,
            "jd_ut1": float(jd_ut1_final),
            "jd_tt": float(jd_tt_final),
            "az_deg": float(az_final),
            "iterations": int(iterations),
            "converged": bool(converged),
            "corrections": {
                "dip_deg": float(dip_deg) if kind in ("RISE", "SET") else 0.0,
                "refraction_deg": float(refr0_deg) if kind in ("RISE", "SET") else 0.0,
            },
        }
        
    except Exception as e:
        _warn(warnings, f"event_{body}_{kind}_calculation_failed:{type(e).__name__}")
        return None

def _daily_events_for_body(
    body: str,
    jd_tt_ref: float,
    jd_ut1_ref: float,
    place: Dict[str, float],
    frame: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    *,
    max_iters: int,
    fd_step_minutes: float,
    apply_refraction: bool,
    pressure_hPa: float,
    temperature_C: float,
    earth_model: str,
    window_days: float,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """
    Compute up to one of each event near the reference epoch within ±window/2.
    """
    half_window = float(window_days) / 2.0
    events: List[Dict[str, Any]] = []
    
    for kind in ("RISE", "CULM", "SET", "ANTI"):
        try:
            event = _find_event_time(
                body, kind, jd_tt_ref, jd_ut1_ref, place, frame, zodiac_mode, ayanamsa_deg,
                max_iters=max_iters, fd_step_minutes=fd_step_minutes,
                apply_refraction=apply_refraction, pressure_hPa=pressure_hPa, temperature_C=temperature_C,
                earth_model=earth_model, warnings=warnings
            )
            
            if event is None:
                continue
            
            # Filter by search window
            time_diff = abs(event["jd_ut1"] - jd_ut1_ref)
            if time_diff <= half_window:
                events.append(event)
            else:
                _warn(warnings, f"event_{body}_{kind}_outside_window:{time_diff:.4f}_days")
                
        except Exception as e:
            _warn(warnings, f"daily_events_{body}_{kind}_failed:{type(e).__name__}")
            continue
    
    # Sort by time
    events.sort(key=lambda e: e["jd_ut1"])
    return events


# ── parans assembly with error handling ──────────────────────────────────────
_PARAN_COMBOS: Tuple[Tuple[str, str], ...] = (
    ("RISE", "CULM"), ("SET", "CULM"),
    ("RISE", "ANTI"), ("SET", "ANTI"),
    ("CULM", "RISE"), ("CULM", "SET"),
    ("ANTI", "RISE"), ("ANTI", "SET"),
)

def _nearest_event_of_type(events: List[Dict[str, Any]], event_type: str) -> Optional[Dict[str, Any]]:
    """Find the nearest event of the specified type."""
    candidates = [e for e in events if e.get("type") == event_type]
    if not candidates:
        return None
    # Return the first (earliest) event since events are sorted by time
    return candidates[0]

def compute_parans(
    subject: Dict[str, Any],
    *,
    place: Dict[str, Any],
    jd_tt_ref: Optional[float] = None,
    jd_ut1_ref: Optional[float] = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    bodies: Tuple[str, ...] = MAJORS,
    tolerance_minutes: float = 4.0,
    search_window_days: float = 1.0,
    max_iters: int = 10,
    fd_step_minutes: float = 2.0,
    earth_model: str = "spherical",
    apply_refraction: bool = False,
    pressure_hPa: float = 1010.0,
    temperature_C: float = 10.0,
    profile: bool = False,
    validation: str = "basic",
) -> Dict[str, Any]:
    """
    Compute local parans for selected bodies around a reference epoch at a given place.
    """
    try:
        t0 = perf_counter()
        prof: Dict[str, float] = {}
        warnings: List[str] = []
        
        # Input validation
        if not isinstance(subject, dict):
            return {"ok": False, "error": "validation_error", "details": "subject must be a dictionary"}
        
        if not isinstance(place, dict):
            return {"ok": False, "error": "validation_error", "details": "place must be a dictionary"}
        
        required_place_keys = ["latitude", "longitude"]
        missing_keys = [k for k in required_place_keys if k not in place]
        if missing_keys:
            return {"ok": False, "error": "validation_error", "details": f"place missing required keys: {missing_keys}"}
        
        try:
            lat = float(place["latitude"])
            lon = float(place["longitude"])
            if not (-90.0 <= lat <= 90.0):
                return {"ok": False, "error": "validation_error", "details": f"Invalid latitude: {lat}"}
            if not (-180.0 <= lon <= 180.0):
                return {"ok": False, "error": "validation_error", "details": f"Invalid longitude: {lon}"}
        except (ValueError, TypeError) as e:
            return {"ok": False, "error": "validation_error", "details": f"Invalid coordinate values: {e}"}
        
        if not bodies:
            return {"ok": False, "error": "validation_error", "details": "No bodies specified"}
        
        # Resolve timescales
        ts_start = perf_counter()
        try:
            jd_tt_ref, jd_ut1_ref, tsmeta = _resolve_timescales(subject, jd_tt_ref, jd_ut1_ref, warnings)
        except Exception as e:
            return {"ok": False, "error": "timescales_error", "details": {"message": str(e), "type": type(e).__name__}}
        prof["timescales_ms"] = (perf_counter() - ts_start) * 1000.0 if profile else 0.0
        
        # Calculate events for each body
        events_start = perf_counter()
        events_by_body: Dict[str, List[Dict[str, Any]]] = {}
        failed_bodies: List[str] = []
        
        for body in bodies:
            try:
                body_events = _daily_events_for_body(
                    str(body), jd_tt_ref, jd_ut1_ref, place, frame, zodiac_mode, ayanamsa_deg,
                    max_iters=max_iters, fd_step_minutes=fd_step_minutes,
                    apply_refraction=apply_refraction, pressure_hPa=pressure_hPa, temperature_C=temperature_C,
                    earth_model=earth_model, window_days=search_window_days, warnings=warnings
                )
                events_by_body[str(body)] = body_events
                
                if not body_events:
                    _warn(warnings, f"no_events_found_for_{body}")
                    
            except Exception as e:
                _warn(warnings, f"events_calculation_failed_for_{body}:{type(e).__name__}")
                failed_bodies.append(str(body))
                events_by_body[str(body)] = []
        
        prof["events_ms"] = (perf_counter() - events_start) * 1000.0 if profile else 0.0
        
        if len(failed_bodies) == len(bodies):
            return {"ok": False, "error": "parans_calculation_failed", "details": "All body calculations failed"}
        
        # Build parans
        parans_start = perf_counter()
        parans: List[Dict[str, Any]] = []
        tolerance_days = float(tolerance_minutes) / (24.0 * 60.0)
        
        successful_bodies = [b for b in bodies if str(b) in events_by_body and events_by_body[str(b)]]
        
        for i, body_a in enumerate(successful_bodies):
            events_a = events_by_body.get(str(body_a), [])
            if not events_a:
                continue
                
            for j in range(i + 1, len(successful_bodies)):
                body_b = successful_bodies[j]
                events_b = events_by_body.get(str(body_b), [])
                if not events_b:
                    continue
                
                # Check all paran combinations
                for type_a, type_b in _PARAN_COMBOS:
                    event_a = _nearest_event_of_type(events_a, type_a)
                    event_b = _nearest_event_of_type(events_b, type_b)
                    
                    if event_a is None or event_b is None:
                        continue
                    
                    time_diff = abs(event_a["jd_ut1"] - event_b["jd_ut1"])
                    delta_minutes = time_diff * 24.0 * 60.0
                    
                    parans.append({
                        "pair": f"{body_a}_{type_a} ~ {body_b}_{type_b}",
                        "a": {"body": str(body_a), "type": type_a, "jd_ut1": event_a["jd_ut1"], "jd_tt": event_a["jd_tt"]},
                        "b": {"body": str(body_b), "type": type_b, "jd_ut1": event_b["jd_ut1"], "jd_tt": event_b["jd_tt"]},
                        "delta_minutes": float(delta_minutes),
                        "within_tolerance": bool(time_diff <= tolerance_days),
                    })
        
        # Sort parans by time difference
        parans.sort(key=lambda p: p["delta_minutes"])
        prof["parans_ms"] = (perf_counter() - parans_start) * 1000.0 if profile else 0.0
        
        # Validation
        validation_start = perf_counter()
        validation_info = None
        if validation and validation.lower() != "none":
            checks: List[Dict[str, Any]] = []
            validation_ok = True
            
            for body, events in events_by_body.items():
                event_count = len(events)
                checks.append({
                    "name": f"{body}_event_count",
                    "count": event_count,
                    "pass": event_count <= 4
                })
                
                if event_count > 4:
                    validation_ok = False
                
                # Check for circumpolar conditions
                event_types = {e["type"] for e in events}
                if not any(t in event_types for t in ("RISE", "SET")):
                    checks.append({
                        "name": f"{body}_circumpolar_or_polar_day",
                        "types_found": list(event_types),
                        "pass": False
                    })
                    validation_ok = False
            
            validation_info = {
                "level": validation.lower(),
                "pass": bool(validation_ok),
                "checks": checks
            }
        
        prof["validation_ms"] = (perf_counter() - validation_start) * 1000.0 if profile else 0.0
        
        # Build metadata
        meta: Dict[str, Any] = {
            "frame": frame,
            "zodiac_mode": zodiac_mode.lower(),
            "ayanamsa_deg": float(ayanamsa_deg),
            "timescales_ref": tsmeta,
            "place": {
                "latitude": float(place["latitude"]),
                "longitude": float(place["longitude"]),
                "elev_m": float(place.get("elev_m", 0.0)),
            },
            "earth_model": earth_model.lower(),
            "refraction": {
                "enabled": bool(apply_refraction),
                "pressure_hPa": float(pressure_hPa),
                "temperature_C": float(temperature_C)
            },
            "tolerance_minutes": float(tolerance_minutes),
            "search_window_days": float(search_window_days),
            "warnings": warnings,
            "notes": [
                "Parans are detected as near-simultaneous horizon/meridian events within the given tolerance.",
                "Event solving uses topocentric RA/Dec and includes optional dip/refraction corrections.",
                "Events are filtered to the search window around the reference epoch.",
            ],
        }
        
        if profile:
            total_time = (perf_counter() - t0) * 1000.0
            prof["total_ms"] = total_time
            meta["profile"] = prof
        
        if validation_info is not None:
            meta["validation"] = validation_info
        
        return {
            "ok": True,
            "meta": meta,
            "events_by_body": events_by_body,
            "parans": parans,
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": "parans_internal_error",
            "details": {"message": str(e), "type": type(e).__name__}
        }
