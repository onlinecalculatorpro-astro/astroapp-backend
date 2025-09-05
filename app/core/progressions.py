# app/core/progressions.py
# -*- coding: utf-8 -*-
"""
Progressions (v11): Secondary, Minor, Tertiary

Public API
----------
compute_progressions(
    natal: dict,
    *,
    method: str = "secondary",        # "secondary" | "minor" | "tertiary"
    target: dict | None = None,       # {"date","time","place_tz"} => defines "as-of" epoch
    years_after: float | None = None, # alternative: explicit age in (tropical) years
    jd_tt_natal: float | None = None,
    jd_ut1_natal: float | None = None,
    place: dict | None = None,        # {latitude, longitude, elev_m} for houses/topo (defaults to natal place)
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",    # "tropical" | "sidereal"
    ayanamsa_deg: float = 0.0,        # subtract when sidereal
    lunar_month: str = "synodic",     # for "minor": "synodic" (29.530588 d) | "sidereal" (27.321582 d)
    tertiary_mode: str = "day-for-month",  # "day-for-month" (Type I) | "lunar-day-for-year" (Type II)
    aspects_to_natal: bool = True,
    orbs: dict | None = None,         # same keys as synastry DEFAULT_ORBS
    parallels: bool = False,
    antiscia: bool = False,

    # Optional diagnostics
    profile: bool = False,            # include meta.profile timings
    validation: str = "basic",        # "none" | "basic"
) -> dict

Method mappings (time → ephemeris offset ΔT_ephem in DAYS)
----------------------------------------------------------
- secondary:           1 civil YEAR since birth  ≡ 1 civil DAY after birth
    ΔT_ephem = years_since_birth * 1.0

- minor (month-for-year):
    ΔT_ephem = years_since_birth * LUNAR_MONTH_DAYS
    where LUNAR_MONTH_DAYS = 29.530588 (synodic, default) or 27.321582 (sidereal)

- tertiary:
    • "day-for-month" (Type I, default): 1 calendar MONTH of life ≡ 1 DAY of ephemeris
        ΔT_ephem = months_since_birth * 1.0  = years_since_birth * 12.0
    • "lunar-day-for-year" (Type II): 1 lunar DAY of ephemeris ≡ 1 YEAR of life
        ΔT_ephem = years_since_birth * LUNAR_DAY_D  (≈ 1.03502 d)

Notes
-----
- Prefers strict timescales (jd_tt & jd_ut1) for the natal epoch. If missing, resolves via
  app.core.timescales.build_timescales(..., dut1_seconds=0.0) and records a warning.
- If a 'target' epoch is given, years_since_birth is derived from UT1 difference divided by the
  mean tropical year (365.242189 d). If 'years_after' is given, it is used directly.
- Planet positions are taken from app.core.ephemeris_adapter.EphemerisAdapter at
  jd_tt_prog = jd_tt_natal + ΔT_ephem (TT). When sidereal, subtract ayanamsa_deg.
- Houses via app.core.houses.compute_houses_with_policy at (jd_tt_prog, jd_ut1_prog).
  UT1 at the progressed epoch is approximated as jd_tt_prog - ΔT(natal)/86400; a warning is included.
- If aspects_to_natal=True, aspects are computed between progressed majors and natal majors (zodiacal by default,
  optional antiscia & declination parallels) using the same default orbs as synastry.

Return shape
------------
{
  "meta": {...},  # frames, zodiac, house system, timescales, mapping details, warnings, (optional) profile
  "epoch": {
    "jd_tt": float, "jd_ut1": float,
    "offset_days": float, "years_since_birth": float, "mapping": {...},
    "place_used": {latitude, longitude, elev_m} | None
  },
  "positions": { "Sun":deg, "Moon":deg, ... },   # progressed longitudes (0..360)
  "houses":   { "asc_deg":deg, "mc_deg":deg, "cusps_deg":[...]} | None,
  "aspects_to_natal": [...],                      # if requested; otherwise []
}

All outputs are JSON-serializable. No I/O/HTTP.
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

try:
    from app.core import aspects as _aspects  # optional dedicated engine
except Exception:
    _aspects = None


# ── constants ─────────────────────────────────────────────────────────────────
MAJORS = ("Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto")

TROPICAL_YEAR_D      = 365.242189
LUNAR_SYNODIC_D      = 29.530588
LUNAR_SIDEREAL_D     = 27.321582
LUNAR_DAY_D          = 1.03502       # ≈ 24h 50m 28s (mean)

# default orbs (aligned with synastry.py)
DEFAULT_ORBS: Dict[str, float] = {
    "conjunction": 8.0,
    "opposition": 6.0,
    "trine": 6.0,
    "square": 5.0,
    "sextile": 3.0,
    "quincunx": 2.0,
    "parallel_arcmin": 40.0,
    "antiscia": 2.0,
}
ASPECT_ANGLES: Dict[str, float] = {
    "conjunction": 0.0,
    "sextile": 60.0,
    "square": 90.0,
    "trine": 120.0,
    "opposition": 180.0,
    "quincunx": 150.0,
}

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

def _degmin_to_deg(arcmin: float) -> float:
    return arcmin / 60.0

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
    return float(ts["jd_tt"]), float(ts["jd_ut1"]), {
        "jd_tt": float(ts["jd_tt"]),
        "jd_ut1": float(ts["jd_ut1"]),
        "delta_t": float(ts.get("delta_t", 0.0)),
        "dut1": float(ts.get("dut1", 0.0)),
    }

def _resolve_years_since_birth(
    natal: Dict[str, Any],
    jd_ut1_natal: float,
    years_after: Optional[float],
    target: Optional[Dict[str, Any]],
    warnings: List[str],
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Returns (years_since_birth, target_ts_meta). If 'target' is provided, we compute UT1
    for target epoch and derive fractional years using TROPICAL_YEAR_D. Otherwise we use years_after.
    """
    if target:
        if build_timescales is None:
            raise RuntimeError("Timescales unavailable to resolve target epoch.")
        date, time, tz = target.get("date"), target.get("time"), target.get("place_tz")
        if not (date and time and tz):
            raise ValueError("Target requires date/time/place_tz.")
        ts = build_timescales(date_str=str(date), time_str=str(time), tz_name=str(tz), dut1_seconds=0.0)
        jd_ut1_target = float(ts["jd_ut1"])
        yrs = (jd_ut1_target - jd_ut1_natal) / TROPICAL_YEAR_D
        return float(yrs), {"jd_tt": float(ts["jd_tt"]), "jd_ut1": jd_ut1_target, "delta_t": float(ts.get("delta_t", 0.0))}
    if years_after is None:
        raise ValueError("Either 'target' or 'years_after' must be provided.")
    return float(years_after), None

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

def _rows_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["name"]): r for r in rows}

def _compute_declinations(rows: List[Dict[str, Any]], jd_tt: float) -> None:
    # light declination from ecliptic lon/lat via mean obliquity (same as synastry)
    def _mean_obliquity_iau2006(jd_tt: float) -> float:
        T = (jd_tt - 2451545.0) / 36525.0
        eps0 = 84381.406 - 46.836769*T - 0.0001831*(T**2) + 0.00200340*(T**3) - 0.000000576*(T**4) - 0.0000000434*(T**5)
        return eps0 / 3600.0
    def _decl(lon_deg: float, lat_deg: float, jd_tt: float) -> float:
        eps = math.radians(_mean_obliquity_iau2006(jd_tt))
        lam = math.radians(_wrap_deg(lon_deg))
        beta = math.radians(lat_deg)
        s = math.sin(beta)*math.cos(eps) + math.cos(beta)*math.sin(eps)*math.sin(lam)
        return math.degrees(math.asin(max(-1.0, min(1.0, s))))
    for r in rows:
        lat = float(r.get("lat", 0.0))
        r["dec"] = _decl(float(r["lon"]), lat, jd_tt)

# simple aspect finder (fallback if no dedicated engine)
def _zodiacal_aspects(rows_prog: List[Dict[str, Any]], rows_nat: List[Dict[str, Any]], orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pP = [r for r in rows_prog if r["name"] in MAJORS]
    pN = [r for r in rows_nat  if r["name"] in MAJORS]
    for a in pP:
        for b in pN:
            sep = _abs_sep(a["lon"], b["lon"])
            for name, ang in ASPECT_ANGLES.items():
                O = orbs.get(name, DEFAULT_ORBS.get(name, 0.0))
                if O <= 0: continue
                tight = abs(sep - ang)
                if tight <= O:
                    out.append({
                        "prog": a["name"], "natal": b["name"],
                        "type": name,
                        "exact_deg": ang, "sep_deg": sep, "orb_deg": tight, "mode": "zodiacal",
                    })
    return out

def _antiscia_of(lon_deg: float) -> float:
    return _wrap_deg(180.0 - _wrap_deg(lon_deg))

def _antiscia_aspects(rows_prog: List[Dict[str, Any]], rows_nat: List[Dict[str, Any]], orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    O = orbs.get("antiscia", DEFAULT_ORBS["antiscia"])
    if O <= 0: return out
    pP = [r for r in rows_prog if r["name"] in MAJORS]
    pN = [r for r in rows_nat  if r["name"] in MAJORS]
    for a in pP:
        a_anti = _antiscia_of(a["lon"])
        for b in pN:
            sep0 = _abs_sep(a_anti, b["lon"])
            if sep0 <= O:
                out.append({"prog":a["name"], "natal":b["name"], "type":"antiscia", "exact_deg":0.0, "sep_deg":sep0, "orb_deg":sep0, "mode":"antiscia"})
            sep180 = min(_abs_sep(a_anti, b["lon"]+180.0), _abs_sep(a_anti, b["lon"]-180.0))
            if sep180 <= O:
                out.append({"prog":a["name"], "natal":b["name"], "type":"contra-antiscia", "exact_deg":180.0, "sep_deg":sep180, "orb_deg":sep180, "mode":"antiscia"})
    return out

def _parallel_aspects(rows_prog: List[Dict[str, Any]], rows_nat: List[Dict[str, Any]], orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    arcmin = orbs.get("parallel_arcmin", DEFAULT_ORBS["parallel_arcmin"])
    if arcmin <= 0: return out
    O = _degmin_to_deg(arcmin)
    pP = [r for r in rows_prog if r["name"] in MAJORS and "dec" in r]
    pN = [r for r in rows_nat  if r["name"] in MAJORS and "dec" in r]
    for a in pP:
        for b in pN:
            d1, d2 = float(a["dec"]), float(b["dec"])
            if abs(d1 - d2) <= O:
                out.append({"prog":a["name"], "natal":b["name"], "type":"parallel", "exact_deg":0.0, "sep_deg":abs(d1-d2), "orb_deg":abs(d1-d2), "mode":"declination"})
            if abs(d1 + d2) <= O:
                out.append({"prog":a["name"], "natal":b["name"], "type":"contra-parallel", "exact_deg":0.0, "sep_deg":abs(d1+d2), "orb_deg":abs(d1+d2), "mode":"declination"})
    return out


# ── public API ────────────────────────────────────────────────────────────────
def compute_progressions(
    natal: Dict[str, Any],
    *,
    method: str = "secondary",
    target: Optional[Dict[str, Any]] = None,
    years_after: Optional[float] = None,
    jd_tt_natal: Optional[float] = None,
    jd_ut1_natal: Optional[float] = None,
    place: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    lunar_month: str = "synodic",
    tertiary_mode: str = "day-for-month",
    aspects_to_natal: bool = True,
    orbs: Optional[Dict[str, float]] = None,
    parallels: bool = False,
    antiscia: bool = False,
    profile: bool = False,
    validation: str = "basic",
) -> Dict[str, Any]:
    """
    Compute progressed positions (secondary/minor/tertiary) and optional houses & aspects to natal.
    """
    t0 = perf_counter()
    prof: Dict[str, float] = {}
    warnings: List[str] = []
    _orbs = {**DEFAULT_ORBS, **(orbs or {})}

    # Natal timescales (strict preferred)
    ts0 = perf_counter()
    jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt_natal, jd_ut1_natal, warnings)
    prof["timescales_ms"] = (perf_counter() - ts0) * 1000.0 if profile else 0.0

    # years since birth
    y0 = perf_counter()
    yrs, target_ts = _resolve_years_since_birth(natal, jd_ut10, years_after, target, warnings)
    prof["resolve_years_ms"] = (perf_counter() - y0) * 1000.0 if profile else 0.0

    # mapping → ephemeris offset (days)
    if method.lower() == "secondary":
        offset_days = yrs * 1.0
        mapping = {"kind":"secondary","formula":"ΔT_ephem = years * 1.0 day"}
    elif method.lower() == "minor":
        month_d = LUNAR_SYNODIC_D if lunar_month.lower() == "synodic" else LUNAR_SIDEREAL_D
        offset_days = yrs * month_d
        mapping = {"kind":"minor","lunar_month":lunar_month.lower(),"month_days":month_d,"formula":"ΔT_ephem = years * lunar_month_days"}
    elif method.lower() == "tertiary":
        if tertiary_mode.lower() == "day-for-month":
            offset_days = yrs * 12.0
            mapping = {"kind":"tertiary","variant":"day-for-month","formula":"ΔT_ephem = years * 12.0 days"}
        elif tertiary_mode.lower() == "lunar-day-for-year":
            offset_days = yrs * LUNAR_DAY_D
            mapping = {"kind":"tertiary","variant":"lunar-day-for-year","lunar_day_days":LUNAR_DAY_D,"formula":"ΔT_ephem = years * lunar_day_days"}
        else:
            raise ValueError("Unknown tertiary_mode. Use 'day-for-month' or 'lunar-day-for-year'.")
    else:
        raise ValueError("Unknown method. Use 'secondary', 'minor', or 'tertiary'.")

    # progressed epoch (TT)
    jd_prog_tt = jd_tt0 + float(offset_days)

    # approximate UT1 at progressed epoch using natal ΔT
    jd_prog_ut1 = jd_prog_tt
    if ts_meta.get("delta_t") is not None:
        jd_prog_ut1 = jd_prog_tt - float(ts_meta["delta_t"]) / 86400.0
        _warn(warnings, "jd_ut1≈jd_tt-ΔT(natal); minor drift ignored")

    # choose place (houses/topocentric)
    place_natal = _to_place(natal, None)
    place_used = _to_place(natal, place) or place_natal

    # progressed positions
    ep0 = perf_counter()
    rows_prog = _planet_rows(jd_prog_tt, place_used, frame, MAJORS, warnings)
    if zodiac_mode.lower() == "sidereal":
        _apply_ayanamsa(rows_prog, ayanamsa_deg)
    positions = {r["name"]: float(_wrap_deg(r["lon"])) for r in rows_prog}
    prof["ephemeris_ms"] = (perf_counter() - ep0) * 1000.0 if profile else 0.0

    # houses
    hs0 = perf_counter()
    houses = None
    if _compute_houses_policy is None:
        _warn(warnings, "houses_policy_unavailable")
    elif place_used is None:
        _warn(warnings, "houses_missing_place")
    else:
        try:
            houses = _compute_houses_policy(
                jd_tt=jd_prog_tt, jd_ut1=jd_prog_ut1,
                latitude=place_used["latitude"], longitude=place_used["longitude"],
                elevation_m=place_used.get("elev_m", 0.0), system=house_system,
            )
        except Exception as e:
            _warn(warnings, f"houses_compute_failed:{type(e).__name__}")
    prof["houses_ms"] = (perf_counter() - hs0) * 1000.0 if profile else 0.0

    # aspects to natal (progressed vs natal majors)
    aspects_list: List[Dict[str, Any]] = []
    if aspects_to_natal:
        # natal rows (geocentric/topo using natal place by default for consistency)
        nat0 = perf_counter()
        rows_nat = _planet_rows(jd_tt0, place_natal, frame, MAJORS, warnings)
        if zodiac_mode.lower() == "sidereal":
            _apply_ayanamsa(rows_nat, ayanamsa_deg)

        # declinations if parallels requested
        if parallels:
            _compute_declinations(rows_prog, jd_prog_tt)
            _compute_declinations(rows_nat, jd_tt0)

        if _aspects and hasattr(_aspects, "find_aspects"):
            # If your aspects engine exposes a generic finder, adapt as needed.
            # Here we’ll fall back to our simple finder to preserve contract stability.
            pass  # keep fallback for consistency
        aspects_list = _zodiacal_aspects(rows_prog, rows_nat, _orbs)
        if antiscia:
            aspects_list += _antiscia_aspects(rows_prog, rows_nat, _orbs)
        if parallels:
            aspects_list += _parallel_aspects(rows_prog, rows_nat, _orbs)
        prof["aspects_ms"] = (perf_counter() - nat0) * 1000.0 if profile else 0.0

    # validation (light)
    validation_info: Optional[Dict[str, Any]] = None
    if validation and validation.lower() != "none":
        checks: List[Dict[str, Any]] = []
        ok = True
        # basic monotonic sanity: progressed Sun must be near natal Sun + yrs*~0.9856° within broad band (not strict)
        try:
            sun_nat = positions["Sun"] if False else None  # placeholder, use rows_nat below
        except Exception:
            sun_nat = None
        try:
            if 'Sun' in positions:
                # If we computed rows_nat above, reuse; otherwise compute quickly (geocentric)
                if not aspects_to_natal:
                    rows_nat = _planet_rows(jd_tt0, None, frame, ("Sun",), warnings)
                    if zodiac_mode.lower() == "sidereal":
                        _apply_ayanamsa(rows_nat, ayanamsa_deg)
                natSun = [r for r in rows_nat if r["name"] == "Sun"][0]["lon"]
                progSun = positions["Sun"]
                # expected drift rough check (secondary ~ yrs*0.9856°, others vary widely so just non-degenerate separation)
                sep = _abs_sep(natSun, progSun)
                checks.append({"name":"sun_progressed_sanity","sep_deg": float(sep), "pass": sep >= 0.5})
                ok = ok and (sep >= 0.5)
        except Exception as e:
            checks.append({"name":"sun_progressed_sanity","error": type(e).__name__, "pass": False})
            ok = False
        validation_info = {"level": validation.lower(), "pass": bool(ok), "checks": checks}

    meta: Dict[str, Any] = {
        "frame": frame,
        "zodiac_mode": zodiac_mode.lower(),
        "ayanamsa_deg": float(ayanamsa_deg),
        "house_system": house_system,
        "natal_timescales": ts_meta,
        "target_timescales": target_ts,
        "warnings": warnings,
        "mapping": mapping,
    }
    if profile:
        meta["profile"] = prof

    return {
        "meta": meta,
        "epoch": {
            "jd_tt": float(jd_prog_tt),
            "jd_ut1": float(jd_prog_ut1),
            "offset_days": float(offset_days),
            "years_since_birth": float(yrs),
            "mapping": mapping,
            "place_used": place_used,
        },
        "positions": positions,
        "houses": houses,
        "aspects_to_natal": aspects_list if aspects_to_natal else [],
    }
