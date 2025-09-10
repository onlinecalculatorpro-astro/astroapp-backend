# app/core/returns.py
# -*- coding: utf-8 -*-
"""
Solar & Lunar Returns (v13) — performance rewrite with robust scanning

Goals
-----
- Keep API compatible: compute_return(...), scan_returns(...)
- Faster & more reliable root-finding (Newton/Secant + bracketing + step caps)
- Ephemeris call minimization via small LRU + JD quantization
- Correct lunar scan seeding by period (no months→years confusion)
- Solid error reporting; predictable metadata; optional profiling

Returns shape (unchanged keys where possible)
--------------------------------------------
{
  "ok": True/False,
  "kind": "solar"|"lunar",
  "meta": {...},               # frame/zodiac/ayanamsa/houses info, warnings, (optional) profile, validation
  "event": {                   # present when ok=True
     "kind": "solar"|"lunar",
     "body": "Sun"|"Moon",
     "jd_tt": float,
     "jd_ut1": float,
     "delta_deg": float,       # residual |Δλ|
     "iterations": int,
     "converged": bool,
     "uncertainty": { ... } | None
  },
  "positions": { "Sun":deg, ... }  # snapshot at solution (majors)
  "houses": {...} | None
}

Notes
-----
- For lunar: guess_years_offset is interpreted as “periods” (i.e., months) for backward compatibility.
- scan_returns() walks the window by **period-aligned seeds** and calls compute_return(around_jd_tt=seed),
  which is both faster and more accurate.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable, Tuple
from time import perf_counter
from functools import lru_cache
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
SOLAR_YEAR_D    = 365.242189
LUNAR_SIDEREAL_D = 27.321582
LUNAR_SYNODIC_D  = 29.530588

# Solver step caps (days)
MAX_NEWTON_STEP_D  = 10.0
FALLBACK_SUN_D     = 1.0    # ~1°/day
FALLBACK_MOON_D    = 0.08   # ~13°/day
FALLBACK_OTHER_D   = 0.5

# JD quantization for ephemeris caching (days). 1e-7 d ~ 0.00864 s.
JD_Q = 1e-7


# ── helpers ───────────────────────────────────────────────────────────────────
def _wrap_deg(x: float) -> float:
    x = math.fmod(x, 360.0)
    return x + 360.0 if x < 0.0 else x

def _delta_deg(a: float, b: float) -> float:
    """Shortest signed Δ from a→b in degrees (-180,+180]."""
    d = _wrap_deg(b) - _wrap_deg(a)
    if d > 180.0: d -= 360.0
    elif d <= -180.0: d += 360.0
    return d

def _warn(ws: List[str], msg: str) -> None:
    if msg not in ws:
        ws.append(msg)

def _apply_ayanamsa(rows: List[Dict[str, Any]], ay: float) -> None:
    if abs(ay) < 1e-12: return
    for r in rows:
        if "lon" in r:
            r["lon"] = _wrap_deg(float(r["lon"]) - ay)

def _to_place(natal: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    src = override if override is not None else natal
    if src and all(k in src for k in ("latitude","longitude")):
        return {
            "latitude": float(src["latitude"]),
            "longitude": float(src["longitude"]),
            "elev_m": float(src.get("elev_m", 0.0)),
        }
    return None

def _place_key(place: Optional[Dict[str, float]]) -> Tuple:
    if not place:
        return (False, 0.0, 0.0, 0.0)
    return (True, round(float(place["latitude"]), 7), round(float(place["longitude"]), 7), round(float(place.get("elev_m", 0.0)), 3))

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
    return float(ts["jd_tt"]), float(ts["jd_ut1"]), {
        "jd_tt": float(ts["jd_tt"]),
        "jd_ut1": float(ts["jd_ut1"]),
        "delta_t": float(ts.get("delta_t", 0.0)),
        "dut1": float(ts.get("dut1", 0.0))
    }

# ── ephemeris adapters & caching ──────────────────────────────────────────────

def _make_adapter(frame: str) -> Any:
    if EphemerisAdapter is None:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPH_ERR}")
    try:
        return EphemerisAdapter(frame=frame)
    except Exception as e:
        raise RuntimeError(f"Failed to create EphemerisAdapter with frame '{frame}': {e}")

@lru_cache(maxsize=8192)
def _cached_lon_speed(frame: str, topo: bool, plat: float, plon: float, pelev: float, jdq: float, body: str) -> Tuple[float, Optional[float]]:
    """
    Cached fetch of (lon, speed?) for a single body at a quantized JD.
    """
    adapter = _make_adapter(frame)
    kwargs = {"jd_tt": jdq, "bodies": [body]}
    if topo:
        kwargs.update({"topocentric": True, "latitude": plat, "longitude": plon, "elevation_m": pelev})
    else:
        kwargs["topocentric"] = False

    # prefer velocities method
    for method_name in ("ecliptic_longitudes_and_velocities", "ecliptic_longitudes"):
        if not hasattr(adapter, method_name):
            continue
        method = getattr(adapter, method_name)
        sig = inspect.signature(method)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        out = method(**filtered_kwargs)
        # parse
        if isinstance(out, dict):
            if "results" in out and isinstance(out["results"], list):
                for row in out["results"]:
                    if isinstance(row, dict) and (row.get("name") or row.get("body")):
                        name = str(row.get("name") or row.get("body"))
                        if name.lower() == body.lower():
                            lon = float(row.get("longitude") or row.get("lon"))
                            spd = row.get("velocity") or row.get("speed")
                            return float(lon), (float(spd) if spd is not None else None)
            # flat mapping fallback
            if body in out and isinstance(out[body], (int,float)):
                return float(out[body]), None
            if body in out and isinstance(out[body], dict):
                lon = float(out[body].get("longitude") or out[body].get("lon"))
                spd = out[body].get("velocity") or out[body].get("speed")
                return float(lon), (float(spd) if spd is not None else None)
        elif isinstance(out, list):
            for row in out:
                if isinstance(row, dict) and (row.get("name") or row.get("body")):
                    name = str(row.get("name") or row.get("body"))
                    if name.lower() == body.lower():
                        lon = float(row.get("longitude") or row.get("lon"))
                        spd = row.get("velocity") or row.get("speed")
                        return float(lon), (float(spd) if spd is not None else None)
    raise RuntimeError(f"Ephemeris method(s) unavailable/empty for body='{body}' @ JD={jdq:.8f}")

def _get_body_lon_and_speed(jd_tt: float, place: Optional[Dict[str,float]], frame: str, body: str) -> Tuple[float, Optional[float]]:
    topo, lat, lon, elev = _place_key(place)
    jdq = round(float(jd_tt) / JD_Q) * JD_Q
    lon_deg, spd = _cached_lon_speed(frame, topo, lat, lon, elev, jdq, body)
    return float(lon_deg), (float(spd) if spd is not None else None)

def _planet_rows(jd_tt: float, place: Optional[Dict[str, float]], frame: str, bodies: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Fast snapshot for multiple bodies. We do not cache this bulk call, but it is used once per result.
    """
    adapter = _make_adapter(frame)
    bodies_list = list(bodies) if bodies else []
    if not bodies_list:
        raise ValueError("No bodies specified")
    kwargs = {"jd_tt": float(jd_tt), "bodies": bodies_list}
    if place:
        kwargs.update({"topocentric": True, "latitude": place["latitude"], "longitude": place["longitude"], "elevation_m": place.get("elev_m", 0.0)})
    else:
        kwargs["topocentric"] = False

    for method_name in ("ecliptic_longitudes_and_velocities", "ecliptic_longitudes"):
        if not hasattr(adapter, method_name):
            continue
        method = getattr(adapter, method_name)
        sig = inspect.signature(method)
        out = method(**{k: v for k, v in kwargs.items() if k in sig.parameters})
        rows: List[Dict[str, Any]] = []
        if isinstance(out, dict) and "results" in out and isinstance(out["results"], list):
            for item in out["results"]:
                if not isinstance(item, dict): continue
                nm = str(item.get("name") or item.get("body") or "unknown")
                if "longitude" in item or "lon" in item:
                    row = {"name": nm, "lon": float(item.get("longitude") or item.get("lon"))}
                    if "velocity" in item or "speed" in item:
                        row["speed"] = float(item.get("velocity") or item.get("speed"))
                    rows.append(row)
        elif isinstance(out, list):
            for item in out:
                if not isinstance(item, dict): continue
                nm = str(item.get("name") or item.get("body") or "unknown")
                if "longitude" in item or "lon" in item:
                    row = {"name": nm, "lon": float(item.get("longitude") or item.get("lon"))}
                    if "velocity" in item or "speed" in item:
                        row["speed"] = float(item.get("velocity") or item.get("speed"))
                    rows.append(row)
        if rows:
            return rows
    raise RuntimeError(f"No usable ephemeris bulk method for bodies={bodies_list} @ JD={jd_tt:.8f}")

# ── numerics ──────────────────────────────────────────────────────────────────

def _central_speed_deg_per_day(body: str, jd_tt: float, place: Optional[Dict[str,float]], frame: str, ay: float, zmode: str, h_days: float) -> float:
    """Central difference speed in deg/day, ayanamsa-adjusted if needed."""
    lon_p, _ = _get_body_lon_and_speed(jd_tt + h_days, place, frame, body)
    lon_m, _ = _get_body_lon_and_speed(jd_tt - h_days, place, frame, body)
    if zmode == "sidereal":
        lon_p = _wrap_deg(lon_p - ay)
        lon_m = _wrap_deg(lon_m - ay)
    d = _delta_deg(lon_m, lon_p)  # lon_p - lon_m along the shortest arc
    return d / (2.0 * h_days)

def _body_typical_fallback_step(body: str) -> float:
    b = body.lower()
    if b == "sun": return FALLBACK_SUN_D
    if b == "moon": return FALLBACK_MOON_D
    return FALLBACK_OTHER_D

def _find_return_jd_tt(
    *,
    body: str,
    natal_lon: float,
    jd_seed: float,
    place: Optional[Dict[str,float]],
    frame: str,
    ayanamsa_deg: float,
    zodiac_mode: str,   # 'tropical' | 'sidereal'
    tol_deg: float,
    max_iters: int
) -> Tuple[float, float, int, bool]:
    """
    Hybrid solver:
      1) Evaluate Δ(j) = natal_lon − lon(j). If speed available, try Newton; else estimate central speed.
      2) If Newton step is too large or fails, fallback to secant or small directed steps.
      3) If function sign doesn't change, perform adaptive bracketing with bounded steps.
    Always caps steps to avoid "shooting past" for slow movers.
    """
    if not body:
        raise ValueError("Body name required")

    def f_and_speed(jd: float) -> Tuple[float, float]:
        lon, spd = _get_body_lon_and_speed(jd, place, frame, body)
        if zodiac_mode == "sidereal":
            lon = _wrap_deg(lon - ayanamsa_deg)
        d = _delta_deg(natal_lon, lon)
        if spd is None or abs(spd) < 1e-8:
            # estimate from central diff with a small h
            h = 1.0/1440.0  # 1 min in days
            spd = _central_speed_deg_per_day(body, jd, place, frame, ayanamsa_deg, zodiac_mode, h)
        return d, spd

    jd  = float(jd_seed)
    d, v = f_and_speed(jd)

    # Try to bracket a sign change near the seed (helps secant)
    # For Moon/Sun this is cheap, for others still safe with caps.
    step0 = _body_typical_fallback_step(body)
    left_jd, left_d = jd, d
    right_jd, right_d = jd, d

    # Expand bracket a few steps if needed
    for _ in range(6):
        if left_d * right_d <= 0.0:
            break
        left_jd  = left_jd  - step0
        right_jd = right_jd + step0
        left_d, _  = f_and_speed(left_jd)
        right_d, _ = f_and_speed(right_jd)

    prev_jd = None
    prev_d  = None

    for it in range(1, max_iters + 1):
        if abs(d) <= tol_deg:
            return jd, abs(d), it - 1, True

        # Newton step proposal
        step = None
        if v is not None and abs(v) > 1e-10:
            step = -d / v
            if abs(step) > MAX_NEWTON_STEP_D:
                step = math.copysign(MAX_NEWTON_STEP_D, step)

        # Secant step if we have history and Newton is unusable
        if (step is None) and (prev_jd is not None) and (prev_d is not None):
            denom = (d - prev_d)
            if abs(denom) > 1e-9:
                step = -d * (jd - prev_jd) / denom
                if abs(step) > MAX_NEWTON_STEP_D:
                    step = math.copysign(MAX_NEWTON_STEP_D, step)

        # Fallback small guided step
        if step is None:
            step = math.copysign(_body_typical_fallback_step(body), -d)

        prev_jd, prev_d = jd, d
        jd = jd + float(step)
        d, v = f_and_speed(jd)

    return jd, abs(d), max_iters, False


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
    guess_years_offset: Optional[int] = None,    # lunar: interpreted as periods (months)
    around_jd_tt: Optional[float] = None,        # preferred seed; used by scan()
    tol_arcmin: float = 1.0,
    max_iters: int = 20,
    # options
    estimate_uncertainty: bool = True,
    fd_step_minutes: float = 2.0,
    profile: bool = False,
    validation: str = "basic",
    validation_residual_arcmin: float = 1.0,
    # accepted but unused passthroughs (route-compat)
    year: Optional[int] = None,
    approx_date: Optional[str] = None,
    jd_start_tt: Optional[float] = None,
    jd_end_tt: Optional[float] = None,
    topocentric: Optional[bool] = None,
    aspects_to_natal: Optional[bool] = None,
    parallels: Optional[bool] = None,
    antiscia: Optional[bool] = None,
    orbs: Optional[Dict[str, float]] = None,
    **_unused: Any,
) -> Dict[str, Any]:
    t0 = perf_counter()
    prof: Dict[str, float] = {}
    warnings: List[str] = []

    try:
        # Validate
        k = kind.lower()
        if k not in ("solar", "lunar"):
            raise ValueError(f"kind must be 'solar' or 'lunar', got '{kind}'")
        if frame not in ("ecliptic-of-date", "ecliptic-j2000"):
            raise ValueError(f"frame must be 'ecliptic-of-date' or 'ecliptic-j2000', got '{frame}'")
        zmode = zodiac_mode.lower()
        if zmode not in ("tropical", "sidereal"):
            raise ValueError("zodiac_mode must be 'tropical' or 'sidereal'")
        if lunar_month not in ("sidereal", "synodic"):
            raise ValueError("lunar_month must be 'sidereal' or 'synodic'")

        # timescales
        ts0 = perf_counter()
        jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt_natal, jd_ut1_natal, warnings)
        if profile: prof["timescales_ms"] = (perf_counter() - ts0) * 1000.0

        # place
        place_natal = _to_place(natal, None)
        place = _to_place(natal, place) or place_natal

        # target body & natal longitude
        body = "Sun" if k == "solar" else "Moon"
        ep0 = perf_counter()
        lon_nat, _spd_nat = _get_body_lon_and_speed(jd_tt0, place, frame, body)
        if zmode == "sidereal":
            lon_nat = _wrap_deg(lon_nat - ayanamsa_deg)
        if profile: prof["natal_lon_ms"] = (perf_counter() - ep0) * 1000.0

        # seed
        if around_jd_tt is not None:
            seed = float(around_jd_tt)
        else:
            if k == "solar":
                offs = 1 if guess_years_offset is None else int(guess_years_offset)
                seed = jd_tt0 + offs * SOLAR_YEAR_D
            else:
                period = LUNAR_SIDEREAL_D if lunar_month == "sidereal" else LUNAR_SYNODIC_D
                offs = 1 if guess_years_offset is None else int(guess_years_offset)
                seed = jd_tt0 + offs * period

        # solve
        it0 = perf_counter()
        tol_deg = float(tol_arcmin) / 60.0
        jd_star, delta_deg, iters, ok = _find_return_jd_tt(
            body=body,
            natal_lon=lon_nat,
            jd_seed=seed,
            place=place,
            frame=frame,
            ayanamsa_deg=ayanamsa_deg,
            zodiac_mode=zmode,
            tol_deg=tol_deg,
            max_iters=max(int(max_iters), 12)
        )
        if profile: prof["root_find_ms"] = (perf_counter() - it0) * 1000.0

        # UT1 approximate at solution
        ut0 = perf_counter()
        jd_ut1_star = jd_star
        if ts_meta.get("delta_t") is not None:
            jd_ut1_star = jd_star - float(ts_meta["delta_t"]) / 86400.0
            _warn(warnings, "jd_ut1≈jd_tt-ΔT(natal); minor drift ignored")
        if profile: prof["ut1_approx_ms"] = (perf_counter() - ut0) * 1000.0

        # snapshot positions once
        snap0 = perf_counter()
        try:
            rows = _planet_rows(jd_star, place, frame, MAJORS)
            if zmode == "sidereal":
                _apply_ayanamsa(rows, ayanamsa_deg)
            positions = {r["name"]: float(_wrap_deg(r["lon"])) for r in rows if "lon" in r}
        except Exception as e:
            positions = {}
            _warn(warnings, f"snapshot_positions_failed:{type(e).__name__}:{str(e)}")
        if profile: prof["snapshot_ms"] = (perf_counter() - snap0) * 1000.0

        # houses (optional)
        hs0 = perf_counter()
        houses = None
        if _compute_houses_policy is None:
            _warn(warnings, f"houses_policy_unavailable:{_HOUSES_ERR}")
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
                _warn(warnings, f"houses_compute_failed:{type(e).__name__}:{str(e)}")
        if profile: prof["houses_ms"] = (perf_counter() - hs0) * 1000.0

        # uncertainty (linearized)
        uncertainty: Optional[Dict[str, float | str]] = None
        if estimate_uncertainty:
            u0 = perf_counter()
            try:
                h_days = max(1e-6, float(fd_step_minutes) / 1440.0)
                spd = abs(_central_speed_deg_per_day(body, jd_star, place, frame, ayanamsa_deg, zmode, h_days))
                dt_resid_days = (delta_deg / max(1e-9, spd)) if spd > 0 else float("inf")
                # linearized slope around jd_star
                lon_p, _ = _get_body_lon_and_speed(jd_star + h_days, place, frame, body)
                lon_m, _ = _get_body_lon_and_speed(jd_star - h_days, place, frame, body)
                if zmode == "sidereal":
                    lon_p = _wrap_deg(lon_p - ayanamsa_deg)
                    lon_m = _wrap_deg(lon_m - ayanamsa_deg)
                d_p = _delta_deg(lon_nat, lon_p)
                d_m = _delta_deg(lon_nat, lon_m)
                slope = (d_p - d_m) / (2.0 * h_days) if h_days > 0 else 0.0
                dt_lin_days = abs(delta_deg / max(1e-9, abs(slope))) if slope != 0 else float("inf")
                dt_days = max(dt_resid_days, dt_lin_days)
                uncertainty = {
                    "dt_days": float(dt_days),
                    "dt_seconds": float(dt_days * 86400.0),
                    "lon_deg": float(spd * dt_days if math.isfinite(dt_days) else float("inf")),
                    "method": "residual/speed + linearized re-root",
                }
            except Exception as e:
                _warn(warnings, f"uncertainty_calculation_failed:{type(e).__name__}:{str(e)}")
                uncertainty = {"dt_days": float("inf"), "dt_seconds": float("inf"), "lon_deg": float("inf"), "method": "failed"}
            if profile: prof["uncertainty_ms"] = (perf_counter() - u0) * 1000.0

        # validation
        validation_info: Optional[Dict[str, Any]] = None
        if validation and validation.lower() != "none":
            basic_target = float(validation_residual_arcmin) / 60.0
            pass_basic = bool(delta_deg <= basic_target)
            validation_info = {
                "level": validation.lower(),
                "pass": pass_basic,
                "checks": [{"name": "residual<=target", "target_deg": basic_target, "value_deg": float(delta_deg), "pass": pass_basic}],
            }

        meta: Dict[str, Any] = {
            "frame": frame,
            "zodiac_mode": zmode,
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
            "kind": k,
            "meta": meta,
            "event": {
                "kind": k,
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
        details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "kind": kind,
            "frame": frame,
            "zodiac_mode": zodiac_mode,
        }
        emsg = str(e).lower()
        if "timescale" in emsg:
            details["failed_step"] = "timescale_resolution"
        elif "ephemeris" in emsg or "body" in emsg:
            details["failed_step"] = "ephemeris_calculation"
        elif "houses" in emsg:
            details["failed_step"] = "houses"
        elif "return" in emsg or "iteration" in emsg or "solver" in emsg:
            details["failed_step"] = "return_finding"
        else:
            details["failed_step"] = "unknown"

        meta_err = {"warnings": warnings}
        if profile:
            meta_err["profile"] = prof

        return {"ok": False, "kind": kind, "error": "returns_internal", "details": details, "meta": meta_err}


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
    tol_arcmin: float = 3.0,          # looser for scanning
    max_iters: int = 25,
    estimate_uncertainty: bool = False,  # OFF for performance during scans
    fd_step_minutes: float = 2.0,
    profile: bool = False,
    validation: str = "basic",
    validation_residual_arcmin: float = 3.0,
    # compat passthrough
    **_unused: Any,
) -> Dict[str, Any]:
    t0 = perf_counter()
    warnings: List[str] = []
    results: List[Dict[str, Any]] = []

    try:
        k = kind.lower()
        if k not in ("solar","lunar"):
            raise ValueError(f"kind must be 'solar' or 'lunar', got '{kind}'")
        if jd_end_tt <= jd_start_tt:
            raise ValueError("jd_end_tt must be greater than jd_start_tt")
        if lunar_month not in ("sidereal","synodic"):
            raise ValueError("lunar_month must be 'sidereal' or 'synodic'")

        # Resolve natal timescales once
        jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt_natal, jd_ut1_natal, warnings)

        # Determine period & seeds aligned to natal epoch
        if k == "solar":
            period = SOLAR_YEAR_D
            padding = 30.0   # tolerance outside window for seed acceptance
        else:
            period = LUNAR_SIDEREAL_D if lunar_month == "sidereal" else LUNAR_SYNODIC_D
            padding = 5.0

        # Compute the integer index range of returns overlapping the window
        idx_start = math.floor((jd_start_tt - jd_tt0) / period)
        idx_end   = math.ceil((jd_end_tt   - jd_tt0) / period)

        # Expand slightly to be safe near boundaries
        idx_start -= 1
        idx_end   += 1

        seeds: List[float] = [jd_tt0 + i * period for i in range(idx_start, idx_end + 1)]
        _warn(warnings, f"scan_seed_count:{len(seeds)} period_days:{period:.6f}")

        # Run compute_return around each seed; this is both accurate and fast
        seen: List[float] = []
        for seed in seeds:
            r = compute_return(
                natal=natal,
                kind=k,
                jd_tt_natal=jd_tt0,
                jd_ut1_natal=jd_ut10,
                place=place,
                frame=frame,
                house_system=house_system,
                zodiac_mode=zodiac_mode,
                ayanamsa_deg=ayanamsa_deg,
                lunar_month=lunar_month,
                around_jd_tt=float(seed),        # <-- precise seeding
                tol_arcmin=tol_arcmin,
                max_iters=max_iters,
                estimate_uncertainty=estimate_uncertainty,
                fd_step_minutes=fd_step_minutes,
                profile=False,
                validation=validation,
                validation_residual_arcmin=validation_residual_arcmin,
            )

            if not r.get("ok"):
                _warn(warnings, f"seed_failed:{seed:.5f}:{r.get('details',{}).get('failed_step','unknown')}")
                continue

            ev = r["event"]
            ev_jd = float(ev["jd_tt"])
            # keep results only inside window (but allow slight padding during matching)
            if (jd_start_tt - padding) <= ev_jd <= (jd_end_tt + padding):
                # dedupe with 0.5 d threshold
                if not any(abs(ev_jd - s) < 0.5 for s in seen):
                    if jd_start_tt <= ev_jd <= jd_end_tt:
                        results.append(r)
                    seen.append(ev_jd)

        results.sort(key=lambda x: x["event"]["jd_tt"])

        meta = {
            "scan_window": {"jd_start_tt": float(jd_start_tt), "jd_end_tt": float(jd_end_tt)},
            "expected_period_days": float(period),
            "natal_jd_tt": float(jd_tt0),
            "returns_found": len(results),
            "scan_time_ms": float((perf_counter() - t0) * 1000.0),
            "warnings": warnings,
        }
        return {"ok": True, "kind": k, "meta": meta, "results": results}

    except Exception as e:
        return {
            "ok": False,
            "kind": kind,
            "error": "scan_internal",
            "details": {"error_type": type(e).__name__, "error_message": str(e), "kind": kind, "window": {"start": jd_start_tt, "end": jd_end_tt}},
            "meta": {
                "warnings": warnings,
                "scan_time_ms": float((perf_counter() - t0) * 1000.0),
                "partial_results": len(results)
            }
        }


__all__ = ["compute_return", "scan_returns"]
