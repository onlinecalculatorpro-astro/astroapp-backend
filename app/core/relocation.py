# app/core/relocation.py
# -*- coding: utf-8 -*-
"""
Relocated charts & Astrocartography (v11)

Enhancements:
- Optional WGS-84 geodetic model for local Earth radius.
- Horizon dip from elevation (DEM callback supported).
- Saemundsson refraction (scaled by P/T) for more realistic ASC/DC curves.
Defaults preserve the original spherical/no-refraction behavior.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable, Tuple, Callable
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
MAJORS: Tuple[str, ...] = (
    "Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"
)

# WGS-84 ellipsoid (meters)
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_MEAN_EARTH_R = 6371008.8  # authalic radius (meters), used for spherical model


# ── angle/time helpers ────────────────────────────────────────────────────────
def _wrap_deg(x: float) -> float:
    x = math.fmod(x, 360.0)
    return x + 360.0 if x < 0.0 else x

def _delta_deg(a: float, b: float) -> float:
    d = _wrap_deg(b) - _wrap_deg(a)
    if d > 180.0: d -= 360.0
    elif d < -180.0: d += 360.0
    return d

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
    gmst = 280.46061837 + 360.98564736629 * d + 0.000387933 * (T*T) - (T*T*T) / 38710000.0
    return _wrap_deg(gmst)

def _ecl_to_equ(lon_deg: float, lat_deg: float, jd_tt: float) -> Tuple[float, float]:
    """Ecliptic (λ,β) → Equatorial (α,δ), degrees."""
    eps = math.radians(_mean_obliquity_iau2006(jd_tt))
    lam = math.radians(_wrap_deg(lon_deg))
    beta = math.radians(lat_deg)
    y = math.sin(lam) * math.cos(eps) - math.tan(beta) * math.sin(eps)
    x = math.cos(lam)
    alpha = math.degrees(math.atan2(y, x)) % 360.0
    s = math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam)
    delta = math.degrees(math.asin(max(-1.0, min(1.0, s))))
    return alpha, delta


# ── common io helpers ────────────────────────────────────────────────────────
def _resolve_ts_from_natal(
    natal: Dict[str, Any],
    jd_tt: Optional[float], jd_ut1: Optional[float],
    warnings: List[str],
) -> Tuple[float, float, Dict[str, Any]]:
    if jd_tt is not None and jd_ut1 is not None:
        return float(jd_tt), float(jd_ut1), {"jd_tt": float(jd_tt), "jd_ut1": float(jd_ut1), "dut1_assumed": None}
    if build_timescales is None:
        raise RuntimeError("Timescales unavailable and strict values not supplied.")
    date, time, tz = natal.get("date"), natal.get("time"), natal.get("place_tz")
    if not (date and time and tz):
        raise ValueError("Missing date/time/place_tz for timescale resolution.")
    ts = build_timescales(date_str=str(date), time_str=str(time), tz_name=str(tz), dut1_seconds=0.0)
    _warn(warnings, "strict_missing→computed_timescales_with_dut1=0.0s")
    return float(ts["jd_tt"]), float(ts["jd_ut1"]), {"jd_tt": float(ts["jd_tt"]), "jd_ut1": float(ts["jd_ut1"]), "dut1_assumed": 0.0}

def _to_place(src: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if src and all(k in src for k in ("latitude","longitude")):
        return {
            "latitude": float(src["latitude"]),
            "longitude": float(src["longitude"]),
            "elev_m": float(src.get("elev_m", 0.0)),
        }
    return None

def _planet_rows(
    jd_tt: float, place: Optional[Dict[str, float]], frame: str,
    bodies: Iterable[str], warnings: List[str]
) -> List[Dict[str, Any]]:
    if EphemerisAdapter is None:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPH_ERR}")
    adapter = EphemerisAdapter(frame=frame)
    kwargs = {"jd_tt": jd_tt, "bodies": list(bodies), "center": "geocentric"}
    if place:
        kwargs.update({
            "center": "topocentric",
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "elevation_m": place.get("elev_m", 0.0),
        })
    for m in ("ecliptic_longitudes_and_velocities", "ecliptic_longitudes"):
        if hasattr(adapter, m):
            try:
                sig = inspect.signature(getattr(adapter, m))
                args = {k: v for k, v in kwargs.items() if k in sig.parameters}
                res = getattr(adapter, m)(**args)
                rows: List[Dict[str, Any]] = []
                if isinstance(res, dict):
                    for k, v in res.items():
                        if isinstance(v, (int, float)):
                            rows.append({"name": k, "lon": float(v)})
                        elif isinstance(v, dict) and "lon" in v:
                            r = {"name": k, "lon": float(v["lon"])}
                            if "lat" in v: r["lat"] = float(v["lat"])
                            if "speed" in v: r["speed"] = float(v["speed"])
                            rows.append(r)
                    return rows
                elif isinstance(res, list):
                    for r in res:
                        if not isinstance(r, dict): continue
                        name = str(r.get("name") or r.get("body") or "?")
                        lon  = float(r.get("lon") or r.get("longitude"))
                        row = {"name": name, "lon": lon}
                        if "lat" in r or "latitude" in r: row["lat"] = float(r.get("lat") or r.get("latitude"))
                        if "speed" in r: row["speed"] = float(r["speed"])
                        rows.append(row)
                    return rows
            except Exception as e:
                _warn(warnings, f"ephemeris_method_failed:{m}:{type(e).__name__}")
                continue
    raise RuntimeError("No usable ephemeris method on EphemerisAdapter.")

def _apply_ayanamsa(rows: List[Dict[str, Any]], ayanamsa_deg: float) -> None:
    if abs(ayanamsa_deg) < 1e-12:
        return
    for r in rows:
        r["lon"] = _wrap_deg(float(r["lon"]) - ayanamsa_deg)


# ── Relocated charts (unchanged) ─────────────────────────────────────────────
def compute_relocated(
    natal: Dict[str, Any],
    *,
    place_new: Dict[str, Any],
    jd_tt_natal: Optional[float] = None,
    jd_ut1_natal: Optional[float] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    topocentric_positions: bool = False,
) -> Dict[str, Any]:
    warnings: List[str] = []
    jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt_natal, jd_ut1_natal, warnings)
    place = _to_place(place_new)
    if place is None:
        raise ValueError("place_new must include latitude/longitude (and optional elev_m).")

    rows = _planet_rows(jd_tt0, place if topocentric_positions else None, frame, MAJORS, warnings)
    if (natal.get("mode") or zodiac_mode or "tropical").lower() == "sidereal":
        _apply_ayanamsa(rows, ayanamsa_deg)
    positions = {r["name"]: float(_wrap_deg(r["lon"])) for r in rows}

    houses = None
    if _compute_houses_policy is None:
        _warn(warnings, "houses_policy_unavailable")
    else:
        try:
            houses = _compute_houses_policy(
                jd_tt=jd_tt0, jd_ut1=jd_ut10,
                latitude=place["latitude"], longitude=place["longitude"],
                elevation_m=place.get("elev_m", 0.0),
                system=house_system,
            )
        except Exception as e:
            _warn(warnings, f"houses_compute_failed:{type(e).__name__}")

    axes = {
        "ASC": float(houses.get("asc_deg")) if houses and "asc_deg" in houses else None,
        "MC":  float(houses.get("mc_deg"))  if houses and "mc_deg" in houses  else None,
    }

    return {
        "meta": {
            "mode": (natal.get("mode") or zodiac_mode).lower(),
            "ayanamsa_deg": float(ayanamsa_deg),
            "frame": frame,
            "house_system": house_system,
            "topocentric_positions": bool(topocentric_positions),
            "timescales": ts_meta,
            "warnings": warnings,
            "notes": ["Relocation recomputes houses at new location; planets are geocentric by default."],
        },
        "positions": positions,
        "houses": houses,
        "axes": axes,
    }


# ── Astrocartography: advanced ASC/DC with dip + refraction + WGS-84 ─────────
def _saemundsson_refraction_deg(h_deg: float, pressure_hPa: float, temperature_C: float) -> float:
    """
    Saemundsson (1986) near-horizon refraction [arcmin]: R = 1.02 / tan((h + 10.3/(h+5.11))°)
    Scaled by (P/1010)*(283/(273+T)).
    Returns degrees. Clamp for stability when h ~ -1..+1 degrees.
    """
    h = max(-1.0, min(89.9, float(h_deg)))  # stabilize
    arg = math.radians(h + 10.3 / (h + 5.11))
    R_arcmin = 1.02 / max(1e-6, math.tan(arg))
    scale = (pressure_hPa / 1010.0) * (283.0 / (273.0 + float(temperature_C)))
    R_arcmin *= scale
    # Physical sanity: cap to 1 degree near/below horizon
    return min(R_arcmin / 60.0, 1.0)

def _earth_radius_m(latitude_deg: float, model: str) -> float:
    """
    Effective local Earth radius for horizon-dip. For 'wgs84', use ellipsoid
    geocentric radius formula; for 'spherical' return mean radius.
    """
    if model.lower() != "wgs84":
        return _MEAN_EARTH_R
    phi = math.radians(latitude_deg)
    a2 = _WGS84_A * _WGS84_A
    b2 = _WGS84_B * _WGS84_B
    cosp = math.cos(phi)
    sinp = math.sin(phi)
    num = (a2 * a2 * cosp * cosp) + (b2 * b2 * sinp * sinp)
    den = (a2 * cosp * cosp) + (b2 * sinp * sinp)
    R = math.sqrt(num / max(1e-9, den))
    return float(R)

def _horizon_dip_deg(elev_m: float, latitude_deg: float, model: str) -> float:
    """Dip ≈ sqrt(2h/R) (radians) → degrees. h in meters, R from Earth model."""
    if elev_m <= 0.0:
        return 0.0
    R = _earth_radius_m(latitude_deg, model)
    dip_rad = math.sqrt(2.0 * float(elev_m) / R)
    return math.degrees(dip_rad)

def _asc_dc_curves_advanced(
    alpha: float,
    delta: float,
    gmst: float,
    lon_step: float,
    lat_clip: float,
    *,
    apply_refraction: bool,
    pressure_hPa: float,
    temperature_C: float,
    dem_callback: Optional[Callable[[float, float], float]],
    default_elev_m: float,
    earth_model: str,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Solve A sinφ + B cosφ = sin(h0) for φ at each sampled longitude:
      A = sin δ, B = cos δ cos H, H = LST - α, h0 = -dip + R_refraction.
    We iterate once to refine h0 with DEM-derived altitude at (λ,φ).
    """
    asc_poly: List[Dict[str, float]] = []
    dc_poly:  List[Dict[str, float]] = []
    A = math.sin(math.radians(delta))
    for k in range(int(-180.0 / lon_step), int(180.0 / lon_step) + 1):
        lam_E = k * lon_step  # deg, -180..+180
        lst = _wrap_deg(gmst + lam_E)
        H = math.radians(_delta_deg(alpha, lst))
        B = math.cos(math.radians(delta)) * math.cos(H)

        # initial φ using zero-altitude (classic spherical)
        phi0 = math.degrees(math.atan2(-math.cos(H), math.tan(math.radians(delta))))
        phi0 = max(-lat_clip, min(lat_clip, phi0))

        # DEM elevation & corrections
        elev_m = float(default_elev_m)
        if callable(dem_callback):
            try:
                elev_m = float(dem_callback(lam_E, phi0))
            except Exception:
                pass

        dip_deg = _horizon_dip_deg(elev_m, phi0, earth_model) if elev_m > 0 else 0.0
        refr_deg = _saemundsson_refraction_deg(0.0, pressure_hPa, temperature_C) if apply_refraction else 0.0
        h0 = math.radians(-dip_deg + refr_deg)  # apparent horizon target altitude

        # Solve A sinφ + B cosφ = sin(h0)
        C = math.sin(h0)
        R = math.hypot(A, B)
        if R < 1e-12:
            # pathological (unlikely for real δ/H); skip sample
            continue
        phi1 = math.atan2(B, A)  # phase
        s = max(-1.0, min(1.0, C / R))
        phi = math.asin(s) - phi1
        phi_deg = math.degrees(phi)
        phi_deg = max(-lat_clip, min(lat_clip, phi_deg))

        # Branch by sin H: ASC (rising) vs DC (setting)
        (asc_poly if math.sin(H) < 0.0 else dc_poly).append({"lat": float(phi_deg), "lon": float(lam_E)})

    return asc_poly, dc_poly


# ── Astrocartography (MC/IC unchanged; ASC/DC now optionally corrected) ──────
def compute_astrocartography(
    natal: Dict[str, Any],
    *,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    bodies: Tuple[str, ...] = MAJORS,
    lon_step_deg: float = 1.0,
    lat_clip_deg: float = 89.5,

    # NEW (all optional; defaults keep legacy behavior)
    earth_model: str = "spherical",               # "spherical" | "wgs84"
    apply_refraction: bool = False,               # Saemundsson near-horizon correction
    pressure_hPa: float = 1010.0,
    temperature_C: float = 10.0,
    dem_callback: Optional[Callable[[float, float], float]] = None,  # fn(lon_deg, lat_deg)->elev_m
    default_elev_m: float = 0.0,
) -> Dict[str, Any]:
    """
    Build astrocartography lines (MC/IC meridians; ASC/DC horizon curves) for selected bodies
    at a given epoch (default: natal). Optional Earth/atmosphere corrections improve realism.
    """
    warnings: List[str] = []
    jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt, jd_ut1, warnings)

    gmst = _gmst_deg(jd_ut10)  # deg
    mode = (natal.get("mode") or zodiac_mode or "tropical").lower()
    lines: List[Dict[str, Any]] = []

    # Helper to get RA/Dec
    def _equ_params_for_body(jd_tt_local: float, body: str) -> Tuple[float, float]:
        rows = _planet_rows(jd_tt_local, None, frame, (body,), warnings)  # geocentric for world map
        lon = float(rows[0]["lon"])
        lat = float(rows[0].get("lat", 0.0))
        if mode == "sidereal":
            lon = _wrap_deg(lon - ayanamsa_deg)
        return _ecl_to_equ(lon, lat, jd_tt_local)

    # MC/IC (unchanged): fixed meridians
    def _mc_ic_lines(alpha: float) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], float, float]:
        lam_mc = _wrap_deg(alpha - gmst)
        lam_ic = _wrap_deg(lam_mc + 180.0)
        lat_samples = list(range(-int(lat_clip_deg), int(lat_clip_deg) + 1))
        mc_poly = [{"lat": float(y), "lon": float(lam_mc if lam_mc <= 180.0 else lam_mc - 360.0)} for y in lat_samples]
        ic_poly = [{"lat": float(y), "lon": float(lam_ic if lam_ic <= 180.0 else lam_ic - 360.0)} for y in lat_samples]
        return mc_poly, ic_poly, lam_mc, lam_ic

    for body in bodies:
        try:
            alpha, delta = _equ_params_for_body(jd_tt0, body)
        except Exception as e:
            _warn(warnings, f"equ_params_failed:{body}:{type(e).__name__}")
            continue

        # MC / IC meridians
        mc_poly, ic_poly, lam_mc, lam_ic = _mc_ic_lines(alpha)
        lines.append({"body": body, "type": "MC", "polyline": mc_poly, "info": {"lon_mc": float(lam_mc)}})
        lines.append({"body": body, "type": "IC", "polyline": ic_poly, "info": {"lon_ic": float(lam_ic)}})

        # ASC / DC curves (optionally corrected)
        if apply_refraction or dem_callback is not None or earth_model.lower() == "wgs84":
            asc_poly, dc_poly = _asc_dc_curves_advanced(
                alpha, delta, gmst, lon_step_deg, lat_clip_deg,
                apply_refraction=bool(apply_refraction),
                pressure_hPa=float(pressure_hPa),
                temperature_C=float(temperature_C),
                dem_callback=dem_callback,
                default_elev_m=float(default_elev_m),
                earth_model=earth_model,
            )
        else:
            # legacy spherical zero-altitude solution
            asc_poly: List[Dict[str, float]] = []
            dc_poly:  List[Dict[str, float]] = []
            tan_delta = math.tan(math.radians(delta))
            for k in range(int(-180.0 / lon_step_deg), int(180.0 / lon_step_deg) + 1):
                lam_E = k * lon_step_deg
                lst = _wrap_deg(gmst + lam_E)
                H = math.radians(_delta_deg(alpha, lst))
                phi = math.degrees(math.atan2(-math.cos(H), tan_delta))
                phi = max(-lat_clip_deg, min(lat_clip_deg, phi))
                (asc_poly if math.sin(H) < 0.0 else dc_poly).append({"lat": float(phi), "lon": float(lam_E)})

        lines.append({"body": body, "type": "ASC", "polyline": asc_poly, "info": {"earth_model": earth_model, "refraction": bool(apply_refraction)}})
        lines.append({"body": body, "type": "DC",  "polyline": dc_poly,  "info": {"earth_model": earth_model, "refraction": bool(apply_refraction)}})

    meta = {
        "epoch_timescales": ts_meta,
        "frame": frame,
        "zodiac_mode": mode,
        "ayanamsa_deg": float(ayanamsa_deg),
        "gmst_deg": float(gmst),
        "epsilon_deg": float(_mean_obliquity_iau2006(jd_tt0)),
        "sampling": {"lon_step_deg": float(lon_step_deg), "lat_clip_deg": float(lat_clip_deg)},
        "earth_model": earth_model.lower(),
        "refraction": {"enabled": bool(apply_refraction), "pressure_hPa": float(pressure_hPa), "temperature_C": float(temperature_C)},
        "dem": {"used": bool(dem_callback is not None), "default_elev_m": float(default_elev_m)},
        "warnings": warnings,
        "notes": [
            "MC/IC meridians are unaffected by refraction; ASC/DC incorporate dip and optional Saemundsson refraction.",
            "DEM is provided via a user callback dem_callback(lon_deg, lat_deg)→elev_m; if absent, default_elev_m is used.",
            "Earth model controls local radius for dip; spherical keeps legacy behavior.",
        ],
    }

    return {
        "meta": meta,
        "lines": lines,
    }
