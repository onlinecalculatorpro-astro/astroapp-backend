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
- “Basic” validation checks that each body yields up to one of each event in-window and highlights circumpolar cases.
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


# ── ephemeris helpers ─────────────────────────────────────────────────────────
def _rows_for_bodies(
    jd_tt: float,
    bodies: Iterable[str],
    *,
    place: Optional[Dict[str, float]],
    frame: str,
) -> List[Dict[str, Any]]:
    if EphemerisAdapter is None:
        raise RuntimeError(f"Ephemeris adapter unavailable: {_EPH_ERR}")
    adapter = EphemerisAdapter(frame=frame)
    kwargs = {"jd_tt": float(jd_tt), "bodies": list(bodies), "center": "geocentric"}
    if place:
        kwargs.update({
            "center": "topocentric",
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
            "elevation_m": float(place.get("elev_m", 0.0)),
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
                            rows.append(r)
                    return rows
                elif isinstance(res, list):
                    for r in res:
                        if not isinstance(r, dict): continue
                        name = str(r.get("name") or r.get("body") or "?")
                        lon  = float(r.get("lon") or r.get("longitude"))
                        row = {"name": name, "lon": lon}
                        if "lat" in r or "latitude" in r: row["lat"] = float(r.get("lat") or r.get("latitude"))
                        rows.append(row)
                    return rows
            except Exception:
                continue
    raise RuntimeError("No usable ephemeris method on EphemerisAdapter.")

def _ra_dec_for_body(
    jd_tt: float,
    body: str,
    *,
    place: Optional[Dict[str, float]],
    frame: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
) -> Tuple[float, float]:
    rows = _rows_for_bodies(jd_tt, (body,), place=place, frame=frame)
    lon = float(rows[0]["lon"])
    lat = float(rows[0].get("lat", 0.0))
    if zodiac_mode.lower() == "sidereal":
        lon = _wrap_deg(lon - ayanamsa_deg)
    alpha, delta = _ecl_to_equ(lon, lat, jd_tt)
    return alpha, delta


# ── timescale resolution ──────────────────────────────────────────────────────
def _resolve_timescales(
    subject: Dict[str, Any],
    jd_tt_ref: Optional[float],
    jd_ut1_ref: Optional[float],
    warnings: List[str],
) -> Tuple[float, float, Dict[str, Any]]:
    if jd_tt_ref is not None and jd_ut1_ref is not None:
        return float(jd_tt_ref), float(jd_ut1_ref), {"jd_tt": float(jd_tt_ref), "jd_ut1": float(jd_ut1_ref), "dut1_assumed": None}
    if build_timescales is None:
        raise RuntimeError("Timescales unavailable and strict values not supplied.")
    date, time, tz = subject.get("date"), subject.get("time"), subject.get("place_tz")
    if not (date and time and tz):
        raise ValueError("Missing date/time/place_tz for timescale resolution.")
    ts = build_timescales(date_str=str(date), time_str=str(time), tz_name=str(tz), dut1_seconds=0.0)
    _warn(warnings, "strict_missing→computed_timescales_with_dut1=0.0s")
    return float(ts["jd_tt"]), float(ts["jd_ut1"]), {"jd_tt": float(ts["jd_tt"]), "jd_ut1": float(ts["jd_ut1"]), "dut1_assumed": 0.0}


# ── event solvers ─────────────────────────────────────────────────────────────
def _lst_deg(jd_ut1: float, longitude_east_deg: float) -> float:
    return _wrap_deg(_gmst_deg(jd_ut1) + float(longitude_east_deg))

def _event_seed(
    kind: str, alpha0: float, delta0: float, lat_deg: float, lst0: float
) -> Optional[float]:
    """
    Return H (hour-angle, deg) seed for event (RISE/SET/CULM/ANTI) using δ at reference.
    For RISE/SET: cos H0 = -tan φ tan δ; RISE uses H=-H0, SET uses H=+H0.
    CULM: H=0; ANTI: H=180 (signed via wrap).
    """
    k = kind.upper()
    if k == "CULM":
        return 0.0
    if k == "ANTI":
        return 180.0
    # rise/set
    tphi = math.tan(math.radians(lat_deg))
    tdel = math.tan(math.radians(delta0))
    arg = -tphi * tdel
    if arg < -1.0 or arg > 1.0:
        return None  # circumpolar: no rise/set
    H0 = math.degrees(math.acos(arg))
    return -H0 if k == "RISE" else +H0

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
) -> Optional[Dict[str, Any]]:
    """
    Solve for event time near the reference epoch (Newton with numeric derivative).
    Returns dict with times & azimuth, or None if not applicable (e.g., circumpolar).
    """
    lat = float(place["latitude"])
    lonE = float(place["longitude"])
    elev = float(place.get("elev_m", 0.0))

    # reference RA/Dec and LST
    alpha0, delta0 = _ra_dec_for_body(jd_tt_ref, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg)
    lst0 = _lst_deg(jd_ut1_ref, lonE)

    H_seed = _event_seed(kind, alpha0, delta0, lat, lst0)
    if H_seed is None:
        return None  # no event (circumpolar)

    # target altitude (h0) with dip/refraction corrections for RISE/SET
    dip_deg = _horizon_dip_deg(elev, lat, earth_model)
    refr0_deg = _saemundsson_refraction_deg(0.0, pressure_hPa, temperature_C) if apply_refraction else 0.0
    h0_deg = -dip_deg + refr0_deg if kind.upper() in ("RISE", "SET") else None

    # initial guess in UT1 days from reference
    # Solve LST - α = H_target  ⇒  Δt ≈ (H_target - (LST0 - α0)) / (dLST/dt - dα/dt),
    # but use simpler sidereal rate seed and then iterate.
    x0_deg = _wrap_pm180((lst0 - alpha0) - H_seed)  # residual at ref (deg)
    dt_seed_days = - x0_deg / GMST_RATE_DEG_PER_DAY  # initial UT1 correction (αdot ignored)

    # iterate
    dt = dt_seed_days
    for it in range(1, max_iters + 1):
        jd_ut1 = jd_ut1_ref + dt
        jd_tt  = jd_tt_ref  + dt  # TT≈UT1 shift is small; acceptable for iteration here

        alpha, delta = _ra_dec_for_body(jd_tt, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg)
        lst = _lst_deg(jd_ut1, lonE)

        if kind.upper() in ("CULM", "ANTI"):
            target = 0.0 if kind.upper() == "CULM" else 180.0
            f = _wrap_pm180((lst - alpha) - target)  # deg
            # derivative df/dt ≈ (GMST_rate - αdot) using numeric αdot
            h = max(1e-6, float(fd_step_minutes) / 1440.0)
            alpha_p, _ = _ra_dec_for_body(jd_tt + h, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg)
            alpha_m, _ = _ra_dec_for_body(jd_tt - h, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg)
            adot = _wrap_pm180(alpha_p - alpha_m) / (2.0 * h)  # deg/day
            dfdt = (GMST_RATE_DEG_PER_DAY - adot)
        else:
            # RISE/SET: solve alt - h0 = 0
            alt, _ = _alt_az_deg(alpha, delta, lat, lst)
            f = (alt - float(h0_deg))  # deg
            # numeric derivative d(alt)/dt via central differences
            h = max(1e-6, float(fd_step_minutes) / 1440.0)
            # step forward/back
            alpha_p, delta_p = _ra_dec_for_body(jd_tt + h, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg)
            lst_p = _lst_deg(jd_ut1 + h, lonE)
            alt_p, _ = _alt_az_deg(alpha_p, delta_p, lat, lst_p)
            alpha_m, delta_m = _ra_dec_for_body(jd_tt - h, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg)
            lst_m = _lst_deg(jd_ut1 - h, lonE)
            alt_m, _ = _alt_az_deg(alpha_m, delta_m, lat, lst_m)
            dfdt = (alt_p - alt_m) / (2.0 * h)  # deg/day

        if abs(dfdt) < 1e-6:
            break  # avoid blow-up

        step = - f / dfdt  # days
        # damp overly large steps
        if abs(step) > 0.5:
            step = math.copysign(0.5, step)
        dt += step

        # stop if residual is tiny
        if abs(f) <= 0.01:  # 0.01 deg ~ 0.6 arcmin (~2.4 min alt or ~2.4 min in time at equator)
            break

    # final evaluation
    jd_ut1 = jd_ut1_ref + dt
    jd_tt  = jd_tt_ref  + dt
    alpha, delta = _ra_dec_for_body(jd_tt, body, place=place, frame=frame, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg)
    lst = _lst_deg(jd_ut1, lonE)
    alt, az = _alt_az_deg(alpha, delta, lat, lst)

    converged = True
    iterations = it if 'it' in locals() else 0
    return {
        "type": kind.upper(),
        "jd_ut1": float(jd_ut1),
        "jd_tt": float(jd_tt),
        "az_deg": float(az),
        "iterations": int(iterations),
        "converged": bool(converged),
        "corrections": {
            "dip_deg": float(dip_deg) if kind.upper() in ("RISE","SET") else 0.0,
            "refraction_deg": float(refr0_deg) if kind.upper() in ("RISE","SET") else 0.0,
        },
    }


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
) -> List[Dict[str, Any]]:
    """
    Compute up to one of each event near the reference epoch within ±window/2.
    """
    half = float(window_days) / 2.0
    events: List[Dict[str, Any]] = []
    for kind in ("RISE","CULM","SET","ANTI"):
        ev = _find_event_time(
            body, kind, jd_tt_ref, jd_ut1_ref, place, frame, zodiac_mode, ayanamsa_deg,
            max_iters=max_iters, fd_step_minutes=fd_step_minutes,
            apply_refraction=apply_refraction, pressure_hPa=pressure_hPa, temperature_C=temperature_C,
            earth_model=earth_model
        )
        if ev is None:
            continue
        # filter by window
        if abs(ev["jd_ut1"] - jd_ut1_ref) <= half:
            events.append(ev)
    # sort by time
    events.sort(key=lambda r: r["jd_ut1"])
    return events


# ── parans assembly ──────────────────────────────────────────────────────────
_PARAN_COMBOS: Tuple[Tuple[str,str], ...] = (
    ("RISE","CULM"), ("SET","CULM"),
    ("RISE","ANTI"), ("SET","ANTI"),
    ("CULM","RISE"), ("CULM","SET"),
    ("ANTI","RISE"), ("ANTI","SET"),
)

def _nearest_event_of_type(evts: List[Dict[str, Any]], typ: str) -> Optional[Dict[str, Any]]:
    cand = [e for e in evts if e["type"] == typ]
    if not cand:
        return None
    # choose the one closest to mid-window (list already filtered near ref)
    # simply return the earliest (sorted)
    return cand[0]

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
    t0 = perf_counter()
    prof: Dict[str, float] = {}
    warnings: List[str] = []

    # Resolve timescales
    ts0 = perf_counter()
    jd_tt_ref, jd_ut1_ref, tsmeta = _resolve_timescales(subject, jd_tt_ref, jd_ut1_ref, warnings)
    prof["timescales_ms"] = (perf_counter() - ts0) * 1000.0 if profile else 0.0

    if not all(k in place for k in ("latitude","longitude")):
        raise ValueError("place must include latitude and longitude (degrees); elev_m optional.")

    # Per-body daily events
    ev0 = perf_counter()
    by_body: Dict[str, List[Dict[str, Any]]] = {}
    for b in bodies:
        try:
            evs = _daily_events_for_body(
                b, jd_tt_ref, jd_ut1_ref, place, frame, zodiac_mode, ayanamsa_deg,
                max_iters=max_iters, fd_step_minutes=fd_step_minutes,
                apply_refraction=apply_refraction, pressure_hPa=pressure_hPa, temperature_C=temperature_C,
                earth_model=earth_model, window_days=search_window_days
            )
            by_body[b] = evs
        except Exception as e:
            _warn(warnings, f"events_failed:{b}:{type(e).__name__}")
    prof["events_ms"] = (perf_counter() - ev0) * 1000.0 if profile else 0.0

    # Build parans
    p0 = perf_counter()
    parans: List[Dict[str, Any]] = []
    tol_days = float(tolerance_minutes) / (24.0 * 60.0)

    body_list = [b for b in bodies if b in by_body]
    n = len(body_list)
    for i in range(n):
        Ai = body_list[i]
        evA = by_body.get(Ai, [])
        if not evA: continue
        for j in range(i+1, n):
            Bj = body_list[j]
            evB = by_body.get(Bj, [])
            if not evB: continue
            # For each pairing type
            for ta, tb in _PARAN_COMBOS:
                a = _nearest_event_of_type(evA, ta)
                b = _nearest_event_of_type(evB, tb)
                if a is None or b is None:
                    continue
                dt = abs(a["jd_ut1"] - b["jd_ut1"])
                parans.append({
                    "pair": f"{Ai}_{ta} ~ {Bj}_{tb}",
                    "a": {"body": Ai, "type": ta, "jd_ut1": a["jd_ut1"], "jd_tt": a["jd_tt"]},
                    "b": {"body": Bj, "type": tb, "jd_ut1": b["jd_ut1"], "jd_tt": b["jd_tt"]},
                    "delta_minutes": float(dt * 24.0 * 60.0),
                    "within_tolerance": bool(dt <= tol_days),
                })
    parans.sort(key=lambda x: x["delta_minutes"])
    prof["parans_ms"] = (perf_counter() - p0) * 1000.0 if profile else 0.0

    # Validation (light)
    v0 = perf_counter()
    if validation and validation.lower() != "none":
        # Simple checks: each body has <= 4 events; circumpolar flags
        checks: List[Dict[str, Any]] = []
        ok = True
        for b, evs in by_body.items():
            kinds = [e["type"] for e in evs]
            checks.append({"name": f"{b}_event_count", "count": len(evs), "pass": len(evs) <= 4})
            ok = ok and (len(evs) <= 4)
            if not any(k in kinds for k in ("RISE","SET")):
                checks.append({"name": f"{b}_circumpolar_or_polar_day", "pass": False})
                ok = False
        validation_info = {"level": validation.lower(), "pass": bool(ok), "checks": checks}
    else:
        validation_info = None
    prof["validation_ms"] = (perf_counter() - v0) * 1000.0 if profile else 0.0

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
        "refraction": {"enabled": bool(apply_refraction), "pressure_hPa": float(pressure_hPa), "temperature_C": float(temperature_C)},
        "tolerance_minutes": float(tolerance_minutes),
        "warnings": warnings,
        "notes": [
            "Parans are detected as near-simultaneous horizon/meridian events within the given tolerance.",
            "Event solving uses topocentric RA/Dec and includes optional dip/refraction corrections.",
        ],
    }
    if profile:
        meta["profile"] = prof
    if validation_info is not None:
        meta["validation"] = validation_info

    return {
        "meta": meta,
        "events_by_body": by_body,
        "parans": parans,
    }
