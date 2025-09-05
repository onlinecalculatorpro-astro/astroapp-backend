# app/core/returns.py
# -*- coding: utf-8 -*-
"""
Solar & Lunar Returns (v11) — with Uncertainty, Profiling & Validation

APIs
----
compute_return(
    natal: dict,
    *,
    kind: str = "solar",              # "solar" | "lunar"
    jd_tt_natal: float | None = None,
    jd_ut1_natal: float | None = None,
    place: dict | None = None,        # {latitude, longitude, elev_m} for houses/topo (defaults to natal place if present)
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",    # "tropical" | "sidereal"
    ayanamsa_deg: float = 0.0,        # subtract when sidereal
    lunar_month: str = "sidereal",    # "sidereal" (27.321582 d) | "synodic" (29.530588 d)
    guess_years_offset: int | None = None,  # which return after natal (e.g., 1 => first SR after birth)
    around_jd_tt: float | None = None,      # alt: target epoch around which to search (TT)
    tol_arcmin: float = 1.0,          # stop when |Δλ| <= tol (arcmin)
    max_iters: int = 12,

    # NEW (optional)
    estimate_uncertainty: bool = True,
    fd_step_minutes: float = 2.0,     # central-diff step for speed/validation (minutes)
    profile: bool = False,            # include meta.profile timings
    validation: str = "basic",        # "none" | "basic" | "extended"
    validation_residual_arcmin: float = 1.0,  # residual target for basic validation
) -> dict

Notes
-----
- Prefers strict timescales (jd_tt & jd_ut1). If missing, resolves via
  app.core.timescales.build_timescales(..., dut1_seconds=0.0) and records a warning.
- Longitudes via app.core.ephemeris_adapter.EphemerisAdapter (default frame = ecliptic-of-date).
  If 'place' is provided, uses topocentric center; otherwise geocentric.
- Houses via app.core.houses.compute_houses_with_policy (needs jd_tt & jd_ut1).
- UT1 at the found TT epoch is approximated as jd_tt - ΔT/86400 using natal ΔT; warning included.
- Uncertainty is estimated from final angular residual + instantaneous angular speed and
  a local linearized re-root using central differences (no external ephemeris needed).
- Validation "extended" adds extra sanity checks (geocentric vs topocentric if place is supplied,
  and a second call path if the adapter exposes multiple methods).

Return shape
------------
{
  "meta": {...},                       # frames, zodiac, house system, timescales, warnings, profile?, validation?
  "event": {
    "kind": "solar"|"lunar",
    "body": "Sun"|"Moon",
    "jd_tt": float,
    "jd_ut1": float,                   # approximated (see warnings)
    "delta_deg": float,                # |Δλ| at solution (deg)
    "iterations": int,
    "converged": bool,
    "uncertainty": {                   # present when estimate_uncertainty=True
      "dt_days": float,
      "dt_seconds": float,
      "lon_deg": float,
      "method": "residual/speed + linearized re-root"
    }
  },
  "positions": { "Sun":deg, "Moon":deg, ... },   # longitudes at return (wrapped 0..360)
  "houses":   { "asc_deg":deg, "mc_deg":deg, "cusps_deg":[...]} | None
}
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
        r["lon"] = _wrap_deg(float(r["lon"]) - ay)

def _resolve_ts_from_natal(natal: Dict[str, Any], jd_tt: Optional[float], jd_ut1: Optional[float], warnings: List[str]) -> Tuple[float,float,Dict[str,Any]]:
    if jd_tt is not None and jd_ut1 is not None:
        return float(jd_tt), float(jd_ut1), {"jd_tt": float(jd_tt), "jd_ut1": float(jd_ut1), "delta_t": None, "dut1": None}
    if build_timescales is None:
        raise RuntimeError("Timescales unavailable and strict values not supplied.")
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
    if EphemerisAdapter is None:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPH_ERR}")
    adapter = EphemerisAdapter(frame=frame)
    kwargs = {"jd_tt": jd_tt, "bodies": list(bodies)}
    if place:
        kwargs.update({"center":"topocentric","latitude":place["latitude"],"longitude":place["longitude"],"elevation_m":place.get("elev_m",0.0)})
    else:
        kwargs["center"] = "geocentric"

    for m in ("ecliptic_longitudes_and_velocities","ecliptic_longitudes"):
        if hasattr(adapter, m):
            try:
                sig = inspect.signature(getattr(adapter, m))
                args = {k:v for k,v in kwargs.items() if k in sig.parameters}
                res = getattr(adapter, m)(**args)
                rows: List[Dict[str, Any]] = []
                if isinstance(res, dict):
                    for k,v in res.items():
                        if isinstance(v,(int,float)):
                            rows.append({"name":k,"lon":float(v)})
                        elif isinstance(v,dict) and "lon" in v:
                            r={"name":k,"lon":float(v["lon"])}
                            if "lat" in v: r["lat"]=float(v["lat"])
                            if "speed" in v: r["speed"]=float(v["speed"])
                            rows.append(r)
                    return rows
                elif isinstance(res, list):
                    for r in res:
                        if not isinstance(r, dict): continue
                        name = str(r.get("name") or r.get("body") or "?")
                        lon  = float(r.get("lon") or r.get("longitude"))
                        row = {"name":name, "lon":lon}
                        if "lat" in r or "latitude" in r: row["lat"] = float(r.get("lat") or r.get("latitude"))
                        if "speed" in r: row["speed"] = float(r["speed"])
                        rows.append(row)
                    return rows
            except Exception as e:
                _warn(warnings, f"ephemeris_method_failed:{m}:{type(e).__name__}")
                continue
    raise RuntimeError("No usable ephemeris method on adapter.")

def _get_body_lon_and_speed(jd_tt: float, place: Optional[Dict[str,float]], frame: str, body: str, warnings: List[str]) -> Tuple[float, Optional[float]]:
    rows = _planet_rows(jd_tt, place, frame, (body,), warnings)
    r = rows[0]
    lon = float(r["lon"])
    spd = float(r["speed"]) if "speed" in r else None
    return lon, spd

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
    Newton-secant hybrid: δ = wrap( lon_body(jd) - natal_lon ) in degrees.
    Derivative from ephemeris 'speed' when available; otherwise secant step.
    Returns (jd_tt, |δ|, iters, converged).
    """
    def _delta_at(jd: float) -> Tuple[float, Optional[float]]:
        lon, spd = _get_body_lon_and_speed(jd, place, frame, body, warnings)
        if zodiac_mode == "sidereal":
            lon = _wrap_deg(lon - ayanamsa_deg)
        d = _delta_deg(natal_lon, lon)  # want 0
        return d, spd

    jd = jd_tt_seed
    d, spd = _delta_at(jd)

    prev_jd = None
    prev_d  = None
    for it in range(1, max_iters+1):
        if abs(d) <= tol_deg:
            return jd, abs(d), it-1, True
        step = None
        if spd is not None and abs(spd) > 1e-6:
            step = - d / spd    # deg / (deg/day) => days
        elif prev_jd is not None and prev_d is not None:
            denom = (d - prev_d)
            if abs(denom) > 1e-9:
                step = - d * (jd - prev_jd) / denom
        if step is None or abs(step) > 3.0:
            step = - math.copysign(0.5, d)  # cautious fallback
        prev_jd, prev_d = jd, d
        jd = jd + float(step)
        d, spd = _delta_at(jd)

    return jd, abs(d), max_iters, False

def _central_speed_deg_per_day(body: str, jd_tt: float, place: Optional[Dict[str,float]], frame: str, ay: float, zmode: str, warnings: List[str], h_days: float) -> float:
    lon_p, _ = _get_body_lon_and_speed(jd_tt + h_days, place, frame, body, warnings)
    lon_m, _ = _get_body_lon_and_speed(jd_tt - h_days, place, frame, body, warnings)
    if zmode == "sidereal":
        lon_p = _wrap_deg(lon_p - ay)
        lon_m = _wrap_deg(lon_m - ay)
    d = _delta_deg(lon_m, lon_p)  # lon_p - lon_m along shortest arc
    return d / (2.0 * h_days)

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
) -> Dict[str, Any]:
    """
    Compute a solar or lunar return with optional uncertainty, profiling, and validation.
    """
    t0 = perf_counter()
    prof: Dict[str, float] = {}
    warnings: List[str] = []

    # timescales
    ts0 = perf_counter()
    jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt_natal, jd_ut1_natal, warnings)
    prof["timescales_ms"] = (perf_counter() - ts0) * 1000.0 if profile else 0.0

    # place defaults
    place_natal = _to_place(natal, None)
    place = _to_place(natal, place) or place_natal

    # natal body longitude
    body = "Sun" if kind.lower() == "solar" else "Moon"
    ep0 = perf_counter()
    lon_nat, _ = _get_body_lon_and_speed(jd_tt0, place, frame, body, warnings)
    if zodiac_mode.lower() == "sidereal":
        lon_nat = _wrap_deg(lon_nat - ayanamsa_deg)
    prof["natal_lon_ms"] = (perf_counter() - ep0) * 1000.0 if profile else 0.0

    # seed
    if around_jd_tt is not None:
        seed = float(around_jd_tt)
    else:
        k = 1 if guess_years_offset is None else int(guess_years_offset)
        if body == "Sun":
            seed = jd_tt0 + k * SOLAR_YEAR_D
        else:
            month_len = LUNAR_SIDEREAL_D if lunar_month == "sidereal" else LUNAR_SYNODIC_D
            seed = jd_tt0 + k * month_len

    # solve
    it0 = perf_counter()
    tol_deg = float(tol_arcmin) / 60.0
    jd_star, delta_deg, iters, ok = _find_return_jd_tt(
        body=body, natal_lon=lon_nat, jd_tt_seed=seed,
        place=place, frame=frame, ayanamsa_deg=ayanamsa_deg, zodiac_mode=zodiac_mode.lower(),
        warnings=warnings, tol_deg=tol_deg, max_iters=max_iters
    )
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
    rows = _planet_rows(jd_star, place, frame, MAJORS, warnings)
    if zodiac_mode.lower() == "sidereal":
        _apply_ayanamsa(rows, ayanamsa_deg)
    positions = {r["name"]: float(_wrap_deg(r["lon"])) for r in rows}
    prof["snapshot_ms"] = (perf_counter() - snap0) * 1000.0 if profile else 0.0

    # houses
    hs0 = perf_counter()
    houses = None
    if _compute_houses_policy is None:
        _warn(warnings, "houses_policy_unavailable")
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
        lon_p, _ = _get_body_lon_and_speed(jd_star + h_days, place, frame, body, warnings)
        lon_m, _ = _get_body_lon_and_speed(jd_star - h_days, place, frame, body, warnings)
        if zodiac_mode.lower() == "sidereal":
            lon_p = _wrap_deg(lon_p - ayanamsa_deg)
            lon_m = _wrap_deg(lon_m - ayanamsa_deg)
        # δ(j) = natal - lon(j)
        d_p = _delta_deg(lon_nat, lon_p)
        d_m = _delta_deg(lon_nat, lon_m)
        # linear interpolation of zero crossing around jd_star
        # slope ≈ (d_p - d_m) / (2h); zero offset ≈ jd_star - d0/slope with d0 ≈ 0 at convergence
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
        prof["uncertainty_ms"] = (perf_counter() - u0) * 1000.0 if profile else 0.0

    # --- Validation -----------------------------------------------------------
    validation_info: Optional[Dict[str, Any]] = None
    v0 = perf_counter()
    if validation and validation.lower() != "none":
        checks: List[Dict[str, Any]] = []
        ok_all = True

        # Basic: residual within target & central-diff self-consistency
        basic_target = float(validation_residual_arcmin) / 60.0
        check_resid = {"name": "residual<=target", "target_deg": basic_target, "value_deg": float(delta_deg)}
        check_resid["pass"] = bool(delta_deg <= basic_target)
        ok_all = ok_all and check_resid["pass"]
        checks.append(check_resid)

        # Self-consistency: recompute longitude with both adapter methods if available
        try:
            adapter = EphemerisAdapter(frame=frame) if EphemerisAdapter else None
            alt_delta = None
            if adapter:
                kwargs = {"jd_tt": jd_star, "bodies": (body,)}
                if place:
                    kwargs.update({"center":"topocentric","latitude":place["latitude"],"longitude":place["longitude"],"elevation_m":place.get("elev_m",0.0)})
                else:
                    kwargs["center"] = "geocentric"

                lon_primary = None
                lon_alt = None
                # primary path
                if hasattr(adapter, "ecliptic_longitudes_and_velocities"):
                    sig = inspect.signature(adapter.ecliptic_longitudes_and_velocities)
                    args = {k:v for k,v in kwargs.items() if k in sig.parameters}
                    res = adapter.ecliptic_longitudes_and_velocities(**args)
                    if isinstance(res, list) and res and isinstance(res[0], dict):
                        lon_primary = float(res[0].get("lon"))
                # alt path
                if hasattr(adapter, "ecliptic_longitudes"):
                    sig = inspect.signature(adapter.ecliptic_longitudes)
                    args = {k:v for k,v in kwargs.items() if k in sig.parameters}
                    res2 = adapter.ecliptic_longitudes(**args)
                    if isinstance(res2, list) and res2 and isinstance(res2[0], dict):
                        lon_alt = float(res2[0].get("lon"))

                if lon_primary is not None and lon_alt is not None:
                    if zodiac_mode.lower() == "sidereal":
                        lon_primary = _wrap_deg(lon_primary - ayanamsa_deg)
                        lon_alt     = _wrap_deg(lon_alt     - ayanamsa_deg)
                    alt_delta = abs(lon_primary - lon_alt)
                    # normalize around wrap
                    alt_delta = min(alt_delta, 360.0 - alt_delta)
            checks.append({"name":"adapter_method_consistency","diff_deg": float(alt_delta) if alt_delta is not None else None, "pass": (alt_delta is None) or (alt_delta <= 1e-3)})
            ok_all = ok_all and ((alt_delta is None) or (alt_delta <= 1e-3))
        except Exception as e:
            checks.append({"name":"adapter_method_consistency","error": type(e).__name__, "pass": False})
            ok_all = False

        # Extended: geocentric vs topocentric sanity (if place available)
        if validation.lower() == "extended" and place is not None:
            try:
                lon_geo, _ = _get_body_lon_and_speed(jd_star, None, frame, body, warnings)
                lon_top, _ = _get_body_lon_and_speed(jd_star, place, frame, body, warnings)
                if zodiac_mode.lower() == "sidereal":
                    lon_geo = _wrap_deg(lon_geo - ayanamsa_deg)
                    lon_top = _wrap_deg(lon_top - ayanamsa_deg)
                dif = abs(_delta_deg(lon_geo, lon_top))
                checks.append({"name":"geo_vs_topo_diff","deg": float(dif), "pass": dif < 1.0})  # usually << 1°
                ok_all = ok_all and (dif < 1.0)
            except Exception as e:
                checks.append({"name":"geo_vs_topo_diff","error": type(e).__name__, "pass": False})
                ok_all = False

        validation_info = {"level": validation.lower(), "pass": bool(ok_all), "checks": checks}
    prof["validation_ms"] = (perf_counter() - v0) * 1000.0 if profile else 0.0

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
    meta.setdefault("notes", []).append("UT1 at return is approximated from natal ΔT; acceptable for houses, tiny error.")

    return {
        "meta": meta,
        "event": {
            "kind": body.lower(),
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
