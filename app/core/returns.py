# app/core/returns.py
# -*- coding: utf-8 -*-
"""
Solar & Lunar Returns (v12) — FIXED with scan functionality and proper error handling

Fixed Issues:
- Added scan_returns function for window scanning
- Enhanced error handling and validation of ephemeris results
- Better convergence handling and fallback strategies
- Defensive programming throughout
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
    from app.core.houses import compute_houses_with_policy as _compute_houses_policy
except Exception as _e:
    _compute_houses_policy = None
    _HOUSES_ERR = _e

try:
    from app.core.timescales import build_timescales
except Exception as _e:
    build_timescales = None  # type: ignore
    _TS_ERR = _e


# ── constants ─────────────────────────────────────────────────────────────────
MAJORS = ("Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto")
SOLAR_YEAR_D = 365.242189
LUNAR_SIDEREAL_D = 27.321582
LUNAR_SYNODIC_D  = 29.530588


# ── helpers ───────────────────────────────────────────────────────────────────
def _wrap_deg(x: float) -> float:
    x = math.fmod(x, 360.0)
    return x + 360.0 if x < 0.0 else x

def _delta_deg(a: float, b: float) -> float:
    d = _wrap_deg(b) - _wrap_deg(a)
    if d > 180.0: d -= 360.0
    elif d < -180.0: d += 360.0
    return d

def _abs_sep(a: float, b: float) -> float:
    return abs(_delta_deg(a, b))

def _warn(ws: List[str], msg: str) -> None:
    if msg not in ws:
        ws.append(msg)

def _apply_ayanamsa(rows: List[Dict[str, Any]], ay: float) -> None:
    if abs(ay) < 1e-12: return
    for r in rows:
        if "lon" in r:
            r["lon"] = _wrap_deg(float(r["lon"]) - ay)

def _resolve_ts_from_natal(natal: Dict[str, Any], jd_tt: Optional[float], jd_ut1: Optional[float], warnings: List[str]) -> Tuple[float,float,Dict[str,Any]]:
    if jd_tt is not None and jd_ut1 is not None:
        return float(jd_tt), float(jd_ut1), {"jd_tt": float(jd_tt), "jd_ut1": float(jd_ut1), "delta_t": None, "dut1": None}
    if build_timescales is None:
        raise RuntimeError(f"Timescales unavailable and strict values not supplied. Import error: {_TS_ERR}")
    date, time, tz = natal.get("date"), natal.get("time"), natal.get("place_tz")
    if not (date and time and tz):
        raise ValueError("Missing date/time/place_tz in natal for timescale resolution.")
    ts = build_timescales(date_str=str(date), time_str=str(time), tz_name=str(tz), dut1_seconds=0.0)
    _warn(warnings, "strict_missing→computed_timescales_with_dut1=0.0s")
    return float(ts["jd_tt"]), float(ts["jd_ut1"]), {"jd_tt": float(ts["jd_tt"]), "jd_ut1": float(ts["jd_ut1"]), "delta_t": float(ts.get("delta_t", 0.0)), "dut1": float(ts.get("dut1", 0.0))}

def _to_place(natal: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    src = override if override is not None else natal
    if src and all(k in src for k in ("latitude","longitude")):
        return {
            "latitude": float(src["latitude"]),
            "longitude": float(src["longitude"]),
            "elev_m": float(src.get("elev_m", 0.0)),
        }
    return None

def _planet_rows(jd_tt: float, place: Optional[Dict[str, float]], frame: str, bodies: Iterable[str], warnings: List[str]) -> List[Dict[str, Any]]:
    """
    FIXED: Enhanced error handling and validation of ephemeris results
    """
    if EphemerisAdapter is None:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPH_ERR}")
    
    # Validate inputs
    if not bodies:
        raise ValueError("No bodies specified for ephemeris calculation")
    
    bodies_list = list(bodies)
    if not bodies_list:
        raise ValueError("Empty bodies list provided")
    
    # Create adapter with proper error handling
    try:
        adapter = EphemerisAdapter(frame=frame)
    except Exception as e:
        raise RuntimeError(f"Failed to create EphemerisAdapter with frame '{frame}': {e}")
    
    # Build arguments for ephemeris call
    kwargs = {"jd_tt": jd_tt, "bodies": bodies_list}
    if place:
        kwargs.update({
            "topocentric": True,
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "elevation_m": place.get("elev_m", 0.0)
        })
    else:
        kwargs["topocentric"] = False

    # Try both available methods with enhanced error handling
    last_error = None
    for method_name in ("ecliptic_longitudes_and_velocities", "ecliptic_longitudes"):
        if not hasattr(adapter, method_name):
            continue
            
        try:
            method = getattr(adapter, method_name)
            sig = inspect.signature(method)
            # Filter kwargs to only include parameters the method accepts
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            
            result = method(**filtered_kwargs)
            
            # Parse result based on type
            rows: List[Dict[str, Any]] = []
            
            if isinstance(result, dict):
                # Handle direct dictionary response
                if "results" in result and isinstance(result["results"], list):
                    # Standard adapter response format
                    for item in result["results"]:
                        if isinstance(item, dict):
                            name = str(item.get("name") or item.get("body") or "unknown")
                            if "longitude" in item or "lon" in item:
                                lon = float(item.get("longitude") or item.get("lon"))
                                row = {"name": name, "lon": lon}
                                if "latitude" in item or "lat" in item:
                                    row["lat"] = float(item.get("latitude") or item.get("lat"))
                                if "velocity" in item or "speed" in item:
                                    row["speed"] = float(item.get("velocity") or item.get("speed"))
                                rows.append(row)
                else:
                    # Handle flat dictionary format
                    for k, v in result.items():
                        if isinstance(v, (int, float)):
                            rows.append({"name": k, "lon": float(v)})
                        elif isinstance(v, dict) and ("lon" in v or "longitude" in v):
                            lon = float(v.get("lon") or v.get("longitude"))
                            row = {"name": k, "lon": lon}
                            if "lat" in v or "latitude" in v:
                                row["lat"] = float(v.get("lat") or v.get("latitude"))
                            if "speed" in v or "velocity" in v:
                                row["speed"] = float(v.get("speed") or v.get("velocity"))
                            rows.append(row)
                            
            elif isinstance(result, list):
                # Handle list response
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or item.get("body") or "unknown")
                    if "lon" in item or "longitude" in item:
                        lon = float(item.get("lon") or item.get("longitude"))
                        row = {"name": name, "lon": lon}
                        if "lat" in item or "latitude" in item:
                            row["lat"] = float(item.get("lat") or item.get("latitude"))
                        if "speed" in item or "velocity" in item:
                            row["speed"] = float(item.get("speed") or item.get("velocity"))
                        rows.append(row)
            
            # Validate we got results
            if not rows:
                _warn(warnings, f"ephemeris_method_{method_name}_returned_empty_results")
                continue
                
            # Validate we got the requested bodies
            found_bodies = {row["name"] for row in rows}
            missing_bodies = set(bodies_list) - found_bodies
            if missing_bodies:
                _warn(warnings, f"ephemeris_missing_bodies: {missing_bodies}")
                # Continue if we got at least some results
                if not rows:
                    continue
            
            return rows
            
        except Exception as e:
            last_error = e
            _warn(warnings, f"ephemeris_method_failed:{method_name}:{type(e).__name__}:{str(e)}")
            continue
    
    # If we get here, all methods failed
    error_msg = f"No usable ephemeris method on adapter. Bodies: {bodies_list}, JD: {jd_tt}, Frame: {frame}"
    if last_error:
        error_msg += f". Last error: {last_error}"
    raise RuntimeError(error_msg)

def _get_body_lon_and_speed(jd_tt: float, place: Optional[Dict[str,float]], frame: str, body: str, warnings: List[str]) -> Tuple[float, Optional[float]]:
    """
    FIXED: Added proper bounds checking to prevent IndexError
    """
    if not body:
        raise ValueError("Body name cannot be empty")
    
    try:
        rows = _planet_rows(jd_tt, place, frame, (body,), warnings)
    except Exception as e:
        raise RuntimeError(f"Failed to get ephemeris data for body '{body}' at JD {jd_tt}: {e}")
    
    # CRITICAL FIX: Check if rows is empty before accessing
    if not rows:
        raise ValueError(f"No ephemeris data returned for body '{body}' at JD {jd_tt:.6f}. Check ephemeris coverage and body name.")
    
    # Find the requested body in results
    target_row = None
    for row in rows:
        if row.get("name", "").lower() == body.lower():
            target_row = row
            break
    
    if target_row is None:
        available_bodies = [row.get("name", "unknown") for row in rows]
        raise ValueError(f"Body '{body}' not found in ephemeris results. Available: {available_bodies}")
    
    # Extract longitude (required)
    if "lon" not in target_row:
        raise ValueError(f"No longitude data for body '{body}' in ephemeris results")
    
    lon = float(target_row["lon"])
    
    # Extract speed (optional)
    speed = None
    if "speed" in target_row:
        try:
            speed = float(target_row["speed"])
        except (ValueError, TypeError):
            _warn(warnings, f"invalid_speed_data_for_{body}")
    
    return lon, speed

def _find_return_jd_tt(
    body: str,
    natal_lon: float,
    jd_tt_seed: float,
    place: Optional[Dict[str,float]],
    frame: str,
    ayanamsa_deg: float,
    zodiac_mode: str,
    warnings: List[str],
    tol_deg: float,
    max_iters: int
) -> Tuple[float, float, int, bool]:
    """
    FIXED: Enhanced error handling and better convergence strategies
    """
    if not body:
        raise ValueError("Body name cannot be empty")
    
    def _delta_at(jd: float) -> Tuple[float, Optional[float]]:
        try:
            lon, spd = _get_body_lon_and_speed(jd, place, frame, body, warnings)
            if zodiac_mode == "sidereal":
                lon = _wrap_deg(lon - ayanamsa_deg)
            d = _delta_deg(natal_lon, lon)  # want 0
            return d, spd
        except Exception as e:
            raise RuntimeError(f"Failed to compute delta at JD {jd:.6f} for body '{body}': {e}")

    jd = jd_tt_seed
    
    try:
        d, spd = _delta_at(jd)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize return calculation at seed JD {jd_tt_seed:.6f}: {e}")

    prev_jd = None
    prev_d  = None
    
    for it in range(1, max_iters + 1):
        if abs(d) <= tol_deg:
            return jd, abs(d), it - 1, True
            
        step = None
        
        # Try Newton method if we have speed
        if spd is not None and abs(spd) > 1e-6:
            step = -d / spd    # deg / (deg/day) => days
            
        # Try secant method if we have previous point
        elif prev_jd is not None and prev_d is not None:
            denom = (d - prev_d)
            if abs(denom) > 1e-9:
                step = -d * (jd - prev_jd) / denom
                
        # Fallback to small step with adaptive size
        if step is None or abs(step) > 10.0:  # Increased fallback limit
            # Estimate step size based on body type and typical speeds
            if body.lower() == "sun":
                fallback_step = math.copysign(1.0, -d)  # Sun moves ~1°/day
            elif body.lower() == "moon":
                fallback_step = math.copysign(0.1, -d)  # Moon moves ~13°/day
            else:
                fallback_step = math.copysign(0.5, -d)  # Other bodies
            step = fallback_step
            
        prev_jd, prev_d = jd, d
        jd = jd + float(step)
        
        try:
            d, spd = _delta_at(jd)
        except Exception as e:
            _warn(warnings, f"iteration_{it}_failed_at_jd_{jd:.6f}: {type(e).__name__}")
            # Try a smaller step
            jd = prev_jd + step * 0.1
            try:
                d, spd = _delta_at(jd)
            except Exception:
                # Give up on this iteration
                break

    return jd, abs(d), max_iters, False

def _central_speed_deg_per_day(body: str, jd_tt: float, place: Optional[Dict[str,float]], frame: str, ay: float, zmode: str, warnings: List[str], h_days: float) -> float:
    """
    FIXED: Enhanced error handling for speed calculation
    """
    try:
        lon_p, _ = _get_body_lon_and_speed(jd_tt + h_days, place, frame, body, warnings)
        lon_m, _ = _get_body_lon_and_speed(jd_tt - h_days, place, frame, body, warnings)
        
        if zmode == "sidereal":
            lon_p = _wrap_deg(lon_p - ay)
            lon_m = _wrap_deg(lon_m - ay)
            
        d = _delta_deg(lon_m, lon_p)  # lon_p - lon_m along shortest arc
        return d / (2.0 * h_days)
    except Exception as e:
        _warn(warnings, f"central_speed_calculation_failed_for_{body}: {type(e).__name__}")
        return 0.0  # Return safe default

# ── public API ────────────────────────────────────────────────────────────────
def compute_return(
    natal: Dict[str, Any],
    *,
    kind: str = "solar",
    jd_tt_natal: Optional[float] = None,
    jd_ut1_natal: Optional[float] = None,
    place: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    lunar_month: str = "sidereal",
    guess_years_offset: Optional[int] = None,
    around_jd_tt: Optional[float] = None,
    tol_arcmin: float = 1.0,
    max_iters: int = 12,
    # NEW options
    estimate_uncertainty: bool = True,
    fd_step_minutes: float = 2.0,
    profile: bool = False,
    validation: str = "basic",
    validation_residual_arcmin: float = 1.0,
    # ── route-compat extras (accepted but currently unused) ────────────────────
    year: Optional[int] = None,
    approx_date: Optional[str] = None,
    jd_start_tt: Optional[float] = None,
    jd_end_tt: Optional[float] = None,
    topocentric: Optional[bool] = None,
    aspects_to_natal: Optional[bool] = None,
    parallels: Optional[bool] = None,
    antiscia: Optional[bool] = None,
    orbs: Optional[Dict[str, float]] = None,
    # Catch-all for any future fields from routes
    **_unused: Any,
) -> Dict[str, Any]:
    """
    FIXED: Compute a solar or lunar return with comprehensive error handling
    """
    t0 = perf_counter()
    prof: Dict[str, float] = {}
    warnings: List[str] = []

    try:
        # Validate inputs
        if not isinstance(natal, dict):
            raise ValueError("natal must be a dictionary")
        
        if kind.lower() not in ("solar", "lunar"):
            raise ValueError(f"kind must be 'solar' or 'lunar', got '{kind}'")
        
        if frame not in ("ecliptic-of-date", "ecliptic-j2000"):
            raise ValueError(f"frame must be 'ecliptic-of-date' or 'ecliptic-j2000', got '{frame}'")

        # timescales
        ts0 = perf_counter()
        try:
            jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt_natal, jd_ut1_natal, warnings)
        except Exception as e:
            raise RuntimeError(f"Failed to resolve timescales: {e}")
        prof["timescales_ms"] = (perf_counter() - ts0) * 1000.0 if profile else 0.0

        # place defaults
        place_natal = _to_place(natal, None)
        place = _to_place(natal, place) or place_natal

        # natal body longitude
        body = "Sun" if kind.lower() == "solar" else "Moon"
        ep0 = perf_counter()
        try:
            lon_nat, _ = _get_body_lon_and_speed(jd_tt0, place, frame, body, warnings)
            if zodiac_mode.lower() == "sidereal":
                lon_nat = _wrap_deg(lon_nat - ayanamsa_deg)
        except Exception as e:
            raise RuntimeError(f"Failed to get natal {body} longitude: {e}")
        prof["natal_lon_ms"] = (perf_counter() - ep0) * 1000.0 if profile else 0.0

        # seed with better defaults
        if around_jd_tt is not None:
            seed = float(around_jd_tt)
        else:
            k = 1 if guess_years_offset is None else int(guess_years_offset)
            if body == "Sun":
                seed = jd_tt0 + k * SOLAR_YEAR_D
            else:
                month_len = LUNAR_SIDEREAL_D if lunar_month == "sidereal" else LUNAR_SYNODIC_D
                seed = jd_tt0 + k * month_len

        # solve with enhanced parameters
        it0 = perf_counter()
        tol_deg = float(tol_arcmin) / 60.0
        try:
            jd_star, delta_deg, iters, ok = _find_return_jd_tt(
                body=body, natal_lon=lon_nat, jd_tt_seed=seed,
                place=place, frame=frame, ayanamsa_deg=ayanamsa_deg, zodiac_mode=zodiac_mode.lower(),
                warnings=warnings, tol_deg=tol_deg, max_iters=max(max_iters, 20)  # Ensure enough iterations
            )
        except Exception as e:
            raise RuntimeError(f"Failed to find {kind} return: {e}")
        prof["root_find_ms"] = (perf_counter() - it0) * 1000.0 if profile else 0.0

        # approximate UT1 at solution
        ut0 = perf_counter()
        jd_ut1_star = jd_star
        if ts_meta.get("delta_t") is not None:
            jd_ut1_star = jd_star - float(ts_meta["delta_t"]) / 86400.0
            _warn(warnings, "jd_ut1≈jd_tt-ΔT(natal); minor drift ignored")
        prof["ut1_approx_ms"] = (perf_counter() - ut0) * 1000.0 if profile else 0.0

        # snapshot positions
        snap0 = perf_counter()
        try:
            rows = _planet_rows(jd_star, place, frame, MAJORS, warnings)
            if zodiac_mode.lower() == "sidereal":
                _apply_ayanamsa(rows, ayanamsa_deg)
            positions = {r["name"]: float(_wrap_deg(r["lon"])) for r in rows if "lon" in r}
        except Exception as e:
            _warn(warnings, f"snapshot_positions_failed: {type(e).__name__}")
            positions = {}
        prof["snapshot_ms"] = (perf_counter() - snap0) * 1000.0 if profile else 0.0

        # houses
        hs0 = perf_counter()
        houses = None
        if _compute_houses_policy is None:
            _warn(warnings, f"houses_policy_unavailable: {_HOUSES_ERR}")
        elif place is None:
            _warn(warnings, "houses_missing_place")
        else:
            try:
                houses = _compute_houses_policy(
                    jd_tt=jd_star, jd_ut1=jd_ut1_star,
                    latitude=place["latitude"], longitude=place["longitude"],
                    elevation_m=place.get("elev_m", 0.0), system=house_system,
                )
            except Exception as e:
                _warn(warnings, f"houses_compute_failed:{type(e).__name__}")
        prof["houses_ms"] = (perf_counter() - hs0) * 1000.0 if profile else 0.0

        # --- Uncertainty estimation (linearized) ---------------------------------
        uncertainty: Optional[Dict[str, float | str]] = None
        if estimate_uncertainty:
            u0 = perf_counter()
            h_days = max(1e-6, float(fd_step_minutes) / 1440.0)
            
            # instantaneous speed via central difference (independent of adapter 'speed')
            spd = _central_speed_deg_per_day(
                body, jd_star, place, frame, ayanamsa_deg, zodiac_mode.lower(), warnings, h_days
            )
            spd_abs = abs(spd)
            if spd_abs < 1e-5:
                _warn(warnings, "low_angular_speed_near_station→time_uncertainty_large")

            # residual→time uncertainty (days)
            dt_resid_days = (delta_deg / max(1e-9, spd_abs)) if spd_abs > 0 else float("inf")

            # local linearized re-root using δ(jd±h)
            try:
                lon_p, _ = _get_body_lon_and_speed(jd_star + h_days, place, frame, body, warnings)
                lon_m, _ = _get_body_lon_and_speed(jd_star - h_days, place, frame, body, warnings)
                if zodiac_mode.lower() == "sidereal":
                    lon_p = _wrap_deg(lon_p - ayanamsa_deg)
                    lon_m = _wrap_deg(lon_m - ayanamsa_deg)
                # δ(j) = natal - lon(j)
                d_p = _delta_deg(lon_nat, lon_p)
                d_m = _delta_deg(lon_nat, lon_m)
                # linear interpolation of zero crossing around jd_star
                slope = (d_p - d_m) / (2.0 * h_days) if abs(h_days) > 0 else 0.0
                dt_lin_days = abs(delta_deg / max(1e-9, abs(slope)))  # conservative linear bound
                dt_days = max(dt_resid_days, dt_lin_days)
                lon_unc = spd_abs * dt_days

                uncertainty = {
                    "dt_days": float(dt_days),
                    "dt_seconds": float(dt_days * 86400.0),
                    "lon_deg": float(lon_unc),
                    "method": "residual/speed + linearized re-root",
                }
            except Exception as e:
                _warn(warnings, f"uncertainty_calculation_failed: {type(e).__name__}")
                uncertainty = {
                    "dt_days": float("inf"),
                    "dt_seconds": float("inf"),
                    "lon_deg": float("inf"),
                    "method": "failed",
                }
            prof["uncertainty_ms"] = (perf_counter() - u0) * 1000.0 if profile else 0.0

        # --- Validation -----------------------------------------------------------
        validation_info: Optional[Dict[str, Any]] = None
        v0 = perf_counter()
        if validation and validation.lower() != "none":
            checks: List[Dict[str, Any]] = []
            ok_all = True

            # Basic: residual within target
            basic_target = float(validation_residual_arcmin) / 60.0
            check_resid = {"name": "residual<=target", "target_deg": basic_target, "value_deg": float(delta_deg)}
            check_resid["pass"] = bool(delta_deg <= basic_target)
            ok_all = ok_all and check_resid["pass"]
            checks.append(check_resid)

            validation_info = {"level": validation.lower(), "pass": bool(ok_all), "checks": checks}
        prof["validation_ms"] = (perf_counter() - v0) * 1000.0 if profile else 0.0

        # Build response
        meta = {
            "frame": frame,
            "zodiac_mode": zodiac_mode.lower(),
            "ayanamsa_deg": float(ayanamsa_deg),
            "house_system": house_system,
            "natal_timescales": ts_meta,
            "warnings": warnings,
        }
        if profile:
            meta["profile"] = prof
        if validation_info is not None:
            meta["validation"] = validation_info

        return {
            "ok": True,
            "meta": meta,
            "event": {
                "kind": kind.lower(),
                "body": body,
                "jd_tt": float(jd_star),
                "jd_ut1": float(jd_ut1_star),
                "delta_deg": float(delta_deg),
                "iterations": int(iters),
                "converged": bool(ok),
                "uncertainty": uncertainty,
            },
            "positions": positions,
            "houses": houses,
        }

    except Exception as e:
        # Comprehensive error handling
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "kind": kind,
            "frame": frame,
            "zodiac_mode": zodiac_mode,
        }
        
        # Add context about which step failed
        if "timescales" in str(e).lower():
            error_details["failed_step"] = "timescale_resolution"
        elif "ephemeris" in str(e).lower() or "body" in str(e).lower():
            error_details["failed_step"] = "ephemeris_calculation"
        elif "return" in str(e).lower() or "iteration" in str(e).lower():
            error_details["failed_step"] = "return_finding"
        else:
            error_details["failed_step"] = "unknown"

        return {
            "ok": False,
            "error": "returns_internal",
            "details": error_details,
            "meta": {
                "warnings": warnings,
                "natal_provided": bool(natal),
                "profile": prof if profile else None,
            }
        }


def scan_returns(
    natal: Dict[str, Any],
    *,
    kind: str = "solar",
    jd_start_tt: float,
    jd_end_tt: float,
    jd_tt_natal: Optional[float] = None,
    jd_ut1_natal: Optional[float] = None,
    place: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    lunar_month: str = "sidereal",
    tol_arcmin: float = 1.0,
    max_iters: int = 12,
    estimate_uncertainty: bool = False,  # Disabled by default for performance
    fd_step_minutes: float = 2.0,
    profile: bool = False,
    validation: str = "basic",
    validation_residual_arcmin: float = 1.0,
    # Catch-all for compatibility
    **_unused: Any,
) -> Dict[str, Any]:
    """
    NEW: Scan a time window for multiple return events
    """
    t0 = perf_counter()
    warnings: List[str] = []
    results: List[Dict[str, Any]] = []

    try:
        # Validate inputs
        if not isinstance(natal, dict):
            raise ValueError("natal must be a dictionary")
        
        if kind.lower() not in ("solar", "lunar"):
            raise ValueError(f"kind must be 'solar' or 'lunar', got '{kind}'")
        
        if jd_end_tt <= jd_start_tt:
            raise ValueError("jd_end_tt must be greater than jd_start_tt")

        # Determine search parameters based on kind
        body = "Sun" if kind.lower() == "solar" else "Moon"
        if body == "Sun":
            period = SOLAR_YEAR_D
            max_returns = max(1, int((jd_end_tt - jd_start_tt) / period) + 2)
        else:
            period = LUNAR_SIDEREAL_D if lunar_month == "sidereal" else LUNAR_SYNODIC_D
            max_returns = max(1, int((jd_end_tt - jd_start_tt) / period) + 5)

        # Get initial guess offset
        try:
            jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt_natal, jd_ut1_natal, warnings)
        except Exception as e:
            raise RuntimeError(f"Failed to resolve natal timescales: {e}")

        # Search for returns in window
        search_start = jd_start_tt - period  # Start searching before window
        current_jd = search_start
        
        for i in range(max_returns):
            if current_jd > jd_end_tt + period:  # Stop searching after window
                break
                
            # Calculate guess years offset from natal
            years_offset = (current_jd - jd_tt0) / 365.25
            
            try:
                result = compute_return(
                    natal=natal,
                    kind=kind,
                    jd_tt_natal=jd_tt0,
                    jd_ut1_natal=jd_ut10,
                    place=place,
                    frame=frame,
                    house_system=house_system,
                    zodiac_mode=zodiac_mode,
                    ayanamsa_deg=ayanamsa_deg,
                    lunar_month=lunar_month,
                    guess_years_offset=int(years_offset),
                    tol_arcmin=tol_arcmin,
                    max_iters=max_iters,
                    estimate_uncertainty=estimate_uncertainty,
                    fd_step_minutes=fd_step_minutes,
                    profile=False,  # Disable profiling for scans
                    validation=validation,
                    validation_residual_arcmin=validation_residual_arcmin,
                )
                
                if result.get("ok") and result.get("event", {}).get("converged"):
                    event_jd = result["event"]["jd_tt"]
                    
                    # Check if this return is within our window
                    if jd_start_tt <= event_jd <= jd_end_tt:
                        results.append(result)
                    
                    # Move to next expected return
                    current_jd = event_jd + period * 0.8  # 80% of period to avoid missing returns
                else:
                    # If calculation failed, advance by expected period
                    current_jd += period
                    
            except Exception as e:
                _warn(warnings, f"scan_iteration_{i}_failed: {type(e).__name__}")
                current_jd += period
                continue

        # Sort results by JD
        results.sort(key=lambda r: r.get("event", {}).get("jd_tt", 0))

        total_time = perf_counter() - t0
        
        meta = {
            "scan_window": {"jd_start_tt": float(jd_start_tt), "jd_end_tt": float(jd_end_tt)},
            "expected_period_days": float(period),
            "returns_found": len(results),
            "scan_time_ms": float(total_time * 1000),
            "warnings": warnings,
        }

        return {
            "ok": True,
            "meta": meta,
            "results": results,
        }

    except Exception as e:
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "kind": kind,
            "window": {"start": jd_start_tt, "end": jd_end_tt},
        }

        return {
            "ok": False,
            "error": "scan_internal",
            "details": error_details,
            "meta": {
                "warnings": warnings,
                "partial_results": len(results),
            }
        }


# Export both functions for route discovery
__all__ = ["compute_return", "scan_returns"]
