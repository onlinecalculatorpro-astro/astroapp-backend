# app/core/progressions.py
# -*- coding: utf-8 -*-
"""
Progressions (v13): Secondary, Minor, Tertiary
- Robust input normalization (method/zodiac/lunar/tertiary enums)
- Safe timescale resolution with clear warnings (never raises on normal cases)
- Stable ephemeris calls (class → module fallback; tolerant signatures)
- Houses/aspects computed only when feasible; degrade gracefully otherwise
- Sidereal ayanamsa applied only when zodiac_mode='sidereal' (ignored w/ warning otherwise)
- Parallel/antiscia optional; declinations computed only when needed
- Validation: light sanity check on Sun separation (optional)

Public API
----------
compute_progressions(
    natal: dict,
    *,
    method: str = "secondary",        # "secondary" | "minor" | "tertiary"
    target: dict | None = None,       # {"date","time","place_tz"| "timezone"}  (aliases supported)
    years_after: float | None = None, # explicit age in (tropical) years
    jd_tt_natal: float | None = None,
    jd_ut1_natal: float | None = None,
    place: dict | None = None,        # {latitude, longitude, elev_m} for houses/topo (defaults to natal place)
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",    # "tropical" | "sidereal"
    ayanamsa_deg: float = 0.0,        # subtract when sidereal
    lunar_month: str = "synodic",     # for "minor": "synodic" | "sidereal"
    tertiary_mode: str = "day-for-month",  # "day-for-month" | "lunar-day-for-year"
    aspects_to_natal: bool = True,
    orbs: dict | None = None,         # keys align with DEFAULT_ORBS
    parallels: bool = False,
    antiscia: bool = False,

    # diagnostics
    profile: bool = False,            # include meta.profile timings
    validation: str = "basic",        # "none" | "basic"
) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
from time import perf_counter
import math
import inspect

# ── optional imports (tolerant) ───────────────────────────────────────────────
try:
    from app.core.ephemeris_adapter import EphemerisAdapter  # optional class API
except Exception:
    EphemerisAdapter = None  # type: ignore

try:
    from app.core.houses import compute_houses_with_policy as _compute_houses_policy
except Exception:
    _compute_houses_policy = None

try:
    from app.core.timescales import build_timescales
except Exception:
    build_timescales = None  # type: ignore

# If a richer aspects engine exists, we still keep local fallbacks for stability.
try:
    from app.core import aspects as _aspects  # noqa: F401
except Exception:
    _aspects = None  # noqa: F841

# ── constants ─────────────────────────────────────────────────────────────────
MAJORS = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"
)

TROPICAL_YEAR_D  = 365.242189
LUNAR_SYNODIC_D  = 29.530588
LUNAR_SIDEREAL_D = 27.321582
LUNAR_DAY_D      = 1.03502     # ≈ 24h 50m 28s (mean)

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

# ── tiny math helpers ─────────────────────────────────────────────────────────
def _wrap_deg(x: float) -> float:
    x = math.fmod(float(x), 360.0)
    return x + 360.0 if x < 0.0 else x

def _delta_deg(a: float, b: float) -> float:
    """Signed shortest separation b−a in degrees (−180..+180]."""
    da = _wrap_deg(b) - _wrap_deg(a)
    if da > 180.0: da -= 360.0
    elif da <= -180.0: da += 360.0
    return da

def _abs_sep(a: float, b: float) -> float:
    return abs(_delta_deg(a, b))

def _degmin_to_deg(arcmin: float) -> float:
    return float(arcmin) / 60.0

def _warn(ws: List[str], msg: str) -> None:
    if msg not in ws:
        ws.append(msg)

def _apply_ayanamsa(rows: List[Dict[str, Any]], ay: float) -> None:
    if abs(ay) < 1e-12:
        return
    for r in rows:
        r["lon"] = _wrap_deg(float(r["lon"]) - ay)

# ── normalization helpers ─────────────────────────────────────────────────────
def _norm_method(method: Optional[str], warnings: List[str]) -> str:
    m = (method or "secondary").strip().lower()
    if m not in ("secondary", "minor", "tertiary"):
        _warn(warnings, f"unknown_method→default_secondary({method!r})")
        return "secondary"
    return m

def _norm_lunar_month(kind: Optional[str], warnings: List[str]) -> str:
    k = (kind or "synodic").strip().lower()
    if k not in ("synodic", "sidereal"):
        _warn(warnings, f"unknown_lunar_month→default_synodic({kind!r})")
        return "synodic"
    return k

def _norm_tertiary_mode(kind: Optional[str], warnings: List[str]) -> str:
    k = (kind or "day-for-month").strip().lower().replace("_", "-")
    if k not in ("day-for-month", "lunar-day-for-year"):
        _warn(warnings, f"unknown_tertiary_mode→default_day-for-month({kind!r})")
        return "day-for-month"
    return k

def _norm_zodiac_mode(mode: Optional[str], warnings: List[str]) -> str:
    z = (mode or "tropical").strip().lower()
    if z not in ("tropical", "sidereal"):
        _warn(warnings, f"unknown_zodiac_mode→default_tropical({mode!r})")
        return "tropical"
    return z

# ── timescales ────────────────────────────────────────────────────────────────
def _resolve_ts_from_natal(
    natal: Dict[str, Any],
    jd_tt: Optional[float],
    jd_ut1: Optional[float],
    warnings: List[str],
) -> Tuple[float, float, Dict[str, Any]]:
    if jd_tt is not None and jd_ut1 is not None:
        return float(jd_tt), float(jd_ut1), {
            "jd_tt": float(jd_tt), "jd_ut1": float(jd_ut1), "delta_t": None, "dut1": None
        }

    if build_timescales is None:
        # No timescales subsystem; last resort error.
        raise RuntimeError("Timescales unavailable and strict values not supplied.")

    date, time, tz = natal.get("date"), natal.get("time"), (natal.get("place_tz") or natal.get("timezone"))
    if not (date and time and tz):
        raise ValueError("Missing natal {date,time,place_tz} for timescale resolution.")

    ts = build_timescales(date_str=str(date), time_str=str(time), tz_name=str(tz), dut1_seconds=0.0)
    if profile:  # pass profile through or add a quiet flag in kwargs/meta
        _warn(warnings, "strict_missing→computed_timescales_with_dut1=0.0s")
    
    return float(ts.jd_tt), float(ts.jd_ut1), {
        "jd_tt": float(ts.jd_tt),
        "jd_ut1": float(ts.jd_ut1),
        "delta_t": float(getattr(ts, "delta_t", 0.0)),
        "dut1": float(getattr(ts, "dut1", 0.0)),
    }

def _resolve_years_since_birth(
    natal: Dict[str, Any],
    jd_ut1_natal: float,
    years_after: Optional[float],
    target: Optional[Dict[str, Any]],
    warnings: List[str],
) -> Tuple[float, Optional[Dict[str, Any]]]:
    if target:
        if build_timescales is None:
            raise RuntimeError("Timescales unavailable to resolve target epoch.")
        d, t = target.get("date"), target.get("time")
        tz = target.get("place_tz") or target.get("timezone")
        if not (d and t and tz):
            raise ValueError("Target requires {date,time,place_tz}.")
        ts = build_timescales(date_str=str(d), time_str=str(t), tz_name=str(tz), dut1_seconds=0.0)
        jd_ut1_target = float(ts.jd_ut1)
        yrs = (jd_ut1_target - jd_ut1_natal) / TROPICAL_YEAR_D
        return float(yrs), {
            "jd_tt": float(ts.jd_tt),
            "jd_ut1": jd_ut1_target,
            "delta_t": float(getattr(ts, "delta_t", 0.0)),
        }

    if years_after is None:
        raise ValueError("Either 'target' or 'years_after' must be provided.")

    return float(years_after), None

# ── place helpers ─────────────────────────────────────────────────────────────
def _to_place(natal: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    src = override if override is not None else natal
    if src and all(k in src for k in ("latitude", "longitude")):
        return {
            "latitude": float(src["latitude"]),
            "longitude": float(src["longitude"]),
            "elev_m": float(src.get("elev_m", 0.0)),
        }
    return None

# ── ephemeris integration (FIXED) ──────────────────────────────────────────────
def _normalize_ephem_result(res: Any) -> List[Dict[str, Any]]:
    """
    Normalize common adapter shapes into:
      [{'name': 'Sun', 'lon': float, 'lat': float?, 'speed': float?}, ...]
    """
    rows: List[Dict[str, Any]] = []
    if res is None:
        return rows

    # Handle the actual adapter response format
    if isinstance(res, dict):
        # Extract results from the response structure
        results = res.get("results", [])
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    name = r.get("name") or r.get("body")
                    lon = r.get("longitude") or r.get("lon")
                    lat = r.get("latitude") or r.get("lat")
                    speed = r.get("velocity") or r.get("speed")
                    
                    if name and isinstance(lon, (int, float)):
                        row = {"name": str(name), "lon": float(lon)}
                        if isinstance(lat, (int, float)):
                            row["lat"] = float(lat)
                        if isinstance(speed, (int, float)):
                            row["speed"] = float(speed)
                        rows.append(row)
            return rows
        
        # Fallback: try to extract from root level
        longitudes = res.get("longitudes", {})
        velocities = res.get("velocities", {})
        for name, lon in longitudes.items():
            if isinstance(lon, (int, float)):
                row = {"name": str(name), "lon": float(lon)}
                if name in velocities and isinstance(velocities[name], (int, float)):
                    row["speed"] = float(velocities[name])
                rows.append(row)
        return rows

    # Handle list format
    if isinstance(res, list):
        for r in res:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("body") or r.get("planet") or "?")
            lon = r.get("lon") or r.get("longitude") or r.get("lambda") or r.get("ecliptic_longitude")
            try:
                lonf = float(lon)
            except Exception:
                continue
            row = {"name": name, "lon": lonf}
            if r.get("lat") is not None:
                try: row["lat"] = float(r.get("lat"))
                except Exception: pass
            if r.get("latitude") is not None:
                try: row["lat"] = float(r.get("latitude"))
                except Exception: pass
            for spk in ("speed", "velocity", "deg_per_day", "dlambda_dt"):
                if r.get(spk) is not None:
                    try: row["speed"] = float(r.get(spk))
                    except Exception: pass
                    break
            rows.append(row)
        return rows

    return rows

def _planet_rows(
    jd_tt: float,
    place: Optional[Dict[str, float]],
    frame: str,
    bodies: Iterable[str],
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """
    FIXED: Use the working ephemeris adapter properly.
    """
    topo = bool(place)
    lat = place.get("latitude") if place else None
    lon = place.get("longitude") if place else None
    elev = place.get("elev_m") if place else None

    # Use the adapter that we know works for /api/ephemeris
    if EphemerisAdapter is not None:
        try:
            adapter = EphemerisAdapter()
            
            # Call the method that works for /api/ephemeris
            result = adapter.ecliptic_longitudes(
                jd_tt=jd_tt,
                names=list(bodies),  # Use 'names' parameter like /api/ephemeris
                frame=frame,
                topocentric=topo,
                latitude=lat,
                longitude=lon,
                elevation_m=elev
            )
            
            # Normalize the result
            rows = _normalize_ephem_result(result)
            if rows:
                return rows
            else:
                _warn(warnings, "ephemeris_adapter_returned_empty_results")
                
        except Exception as e:
            _warn(warnings, f"ephemeris_adapter_failed:{type(e).__name__}:{str(e)}")

    # Fallback: try module-level functions (simplified)
    try:
        from app.core import ephemeris_adapter as ea
        
        # Try the same method that works for /api/ephemeris
        result = ea.ecliptic_longitudes(
            jd_tt=jd_tt,
            names=list(bodies),
            frame=frame,
            topocentric=topo,
            latitude=lat,
            longitude=lon,
            elevation_m=elev
        )
        
        rows = _normalize_ephem_result(result)
        if rows:
            return rows
            
    except Exception as e:
        _warn(warnings, f"ephemeris_module_failed:{type(e).__name__}")

    # Last resort: return empty with clear warning
    _warn(warnings, "ephemeris_unavailable_for_longitudes")
    return []

# ── declinations (for parallels) ──────────────────────────────────────────────
def _compute_declinations(rows: List[Dict[str, Any]], jd_tt: float) -> None:
    # mean obliquity (IAU 2006) and ecliptic→equatorial conversion
    def _mean_obliquity_iau2006(jd_tt_: float) -> float:
        T = (jd_tt_ - 2451545.0) / 36525.0
        eps0 = 84381.406 - 46.836769*T - 0.0001831*(T**2) + 0.00200340*(T**3) - 0.000000576*(T**4) - 0.0000000434*(T**5)
        return eps0 / 3600.0
    eps = math.radians(_mean_obliquity_iau2006(jd_tt))
    for r in rows:
        lam = math.radians(_wrap_deg(r["lon"]))
        beta = math.radians(float(r.get("lat", 0.0)))
        s = math.sin(beta)*math.cos(eps) + math.cos(beta)*math.sin(eps)*math.sin(lam)
        r["dec"] = math.degrees(math.asin(max(-1.0, min(1.0, s))))

# ── simple aspect finders (fallback) ──────────────────────────────────────────
def _zodiacal_aspects(rows_prog: List[Dict[str, Any]], rows_nat: List[Dict[str, Any]], orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pP = [r for r in rows_prog if r["name"] in MAJORS]
    pN = [r for r in rows_nat  if r["name"] in MAJORS]
    for a in pP:
        for b in pN:
            sep = _abs_sep(a["lon"], b["lon"])
            for typ, ang in ASPECT_ANGLES.items():
                O = orbs.get(typ, DEFAULT_ORBS.get(typ, 0.0))
                if O <= 0:
                    continue
                tight = abs(sep - ang)
                if tight <= O:
                    out.append({
                        "prog": a["name"], "natal": b["name"],
                        "type": typ, "exact_deg": ang,
                        "sep_deg": sep, "orb_deg": tight, "mode": "zodiacal",
                    })
    return out

def _antiscia_of(lon_deg: float) -> float:
    return _wrap_deg(180.0 - _wrap_deg(lon_deg))

def _antiscia_aspects(rows_prog: List[Dict[str, Any]], rows_nat: List[Dict[str, Any]], orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    O = orbs.get("antiscia", DEFAULT_ORBS["antiscia"])
    if O <= 0:
        return out
    pP = [r for r in rows_prog if r["name"] in MAJORS]
    pN = [r for r in rows_nat  if r["name"] in MAJORS]
    for a in pP:
        a_anti = _antiscia_of(a["lon"])
        for b in pN:
            sep0 = _abs_sep(a_anti, b["lon"])
            if sep0 <= O:
                out.append({"prog": a["name"], "natal": b["name"], "type": "antiscia",
                            "exact_deg": 0.0, "sep_deg": sep0, "orb_deg": sep0, "mode": "antiscia"})
            sep180 = min(_abs_sep(a_anti, b["lon"] + 180.0), _abs_sep(a_anti, b["lon"] - 180.0))
            if sep180 <= O:
                out.append({"prog": a["name"], "natal": b["name"], "type": "contra-antiscia",
                            "exact_deg": 180.0, "sep_deg": sep180, "orb_deg": sep180, "mode": "antiscia"})
    return out

def _parallel_aspects(rows_prog: List[Dict[str, Any]], rows_nat: List[Dict[str, Any]], orbs: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    arcmin = orbs.get("parallel_arcmin", DEFAULT_ORBS["parallel_arcmin"])
    if arcmin <= 0:
        return out
    O = _degmin_to_deg(arcmin)
    pP = [r for r in rows_prog if r["name"] in MAJORS and "dec" in r]
    pN = [r for r in rows_nat  if r["name"] in MAJORS and "dec" in r]
    for a in pP:
        for b in pN:
            d1, d2 = float(a["dec"]), float(b["dec"])
            if abs(d1 - d2) <= O:
                out.append({"prog": a["name"], "natal": b["name"], "type": "parallel",
                            "exact_deg": 0.0, "sep_deg": abs(d1 - d2), "orb_deg": abs(d1 - d2),
                            "mode": "declination"})
            if abs(d1 + d2) <= O:
                out.append({"prog": a["name"], "natal": b["name"], "type": "contra-parallel",
                            "exact_deg": 0.0, "sep_deg": abs(d1 + d2), "orb_deg": abs(d1 + d2),
                            "mode": "declination"})
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
    Compute progressed positions (secondary/minor/tertiary) plus optional houses & aspects to natal.
    Never raises for normal adapter/house/aspect failures — returns partial results with warnings.
    """
    t0 = perf_counter()
    prof: Dict[str, float] = {}
    warnings: List[str] = []
    _orbs = {**DEFAULT_ORBS, **(orbs or {})}

    # Normalize enums early (prevents deep KeyErrors)
    m = _norm_method(method, warnings)
    zmode = _norm_zodiac_mode(zodiac_mode, warnings)
    lmonth = _norm_lunar_month(lunar_month, warnings)
    tmode = _norm_tertiary_mode(tertiary_mode, warnings)

    # Natal timescales (prefer strict)
    ts0 = perf_counter()
    jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt_natal, jd_ut1_natal, warnings)
    if profile: prof["timescales_ms"] = (perf_counter() - ts0) * 1000.0

    # Years since birth
    y0 = perf_counter()
    yrs, target_ts = _resolve_years_since_birth(natal, jd_ut10, years_after, target, warnings)
    if profile: prof["resolve_years_ms"] = (perf_counter() - y0) * 1000.0

    # Mapping → ephemeris offset (days)
    if m == "secondary":
        offset_days = yrs * 1.0
        mapping = {"kind": "secondary", "formula": "ΔT_ephem = years * 1.0 day"}
    elif m == "minor":
        month_d = LUNAR_SYNODIC_D if lmonth == "synodic" else LUNAR_SIDEREAL_D
        offset_days = yrs * month_d
        mapping = {"kind": "minor", "lunar_month": lmonth, "month_days": month_d,
                   "formula": "ΔT_ephem = years * lunar_month_days"}
    else:  # tertiary
        if tmode == "day-for-month":
            offset_days = yrs * 12.0
            mapping = {"kind": "tertiary", "variant": "day-for-month",
                       "formula": "ΔT_ephem = years * 12.0 days"}
        else:  # "lunar-day-for-year"
            offset_days = yrs * LUNAR_DAY_D
            mapping = {"kind": "tertiary", "variant": "lunar-day-for-year",
                       "lunar_day_days": LUNAR_DAY_D, "formula": "ΔT_ephem = years * lunar_day_days"}

    # Progressed epoch (TT)
    jd_prog_tt = jd_tt0 + float(offset_days)

    # Approx UT1 at progressed epoch using natal ΔT (documented approximation)
    jd_prog_ut1 = jd_prog_tt
    if ts_meta.get("delta_t") is not None:
        jd_prog_ut1 = jd_prog_tt - float(ts_meta["delta_t"]) / 86400.0
        _warn(warnings, "jd_ut1≈jd_tt-ΔT(natal); minor drift ignored")

    # Place for topo/houses
    place_natal = _to_place(natal, None)
    place_used = _to_place(natal, place) or place_natal

    # Progressed positions
    ep0 = perf_counter()
    rows_prog = _planet_rows(jd_prog_tt, place_used, frame, MAJORS, warnings)
    if not rows_prog:
        # Even if ephemeris failed, keep shape stable
        positions: Dict[str, float] = {}
    else:
        if zmode == "sidereal":
            _apply_ayanamsa(rows_prog, ayanamsa_deg)
        elif abs(ayanamsa_deg) > 1e-12:
            _warn(warnings, "ayanamsa_deg_ignored_in_tropical")
        positions = {r["name"]: float(_wrap_deg(r["lon"])) for r in rows_prog}
    if profile: prof["ephemeris_ms"] = (perf_counter() - ep0) * 1000.0

    # Houses (optional, only if we have a place & the policy function is present)
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
    if profile: prof["houses_ms"] = (perf_counter() - hs0) * 1000.0

    # Aspects to natal
    aspects_list: List[Dict[str, Any]] = []
    if aspects_to_natal:
        nat0 = perf_counter()
        rows_nat = _planet_rows(jd_tt0, place_natal, frame, MAJORS, warnings)
        if rows_nat and rows_prog:
            if zmode == "sidereal":
                _apply_ayanamsa(rows_nat, ayanamsa_deg)

            if parallels:
                _compute_declinations(rows_prog, jd_prog_tt)
                _compute_declinations(rows_nat, jd_tt0)

            # Fallback aspect finder; keeps output contract stable
            aspects_list = _zodiacal_aspects(rows_prog, rows_nat, _orbs)
            if antiscia:
                aspects_list += _antiscia_aspects(rows_prog, rows_nat, _orbs)
            if parallels:
                aspects_list += _parallel_aspects(rows_prog, rows_nat, _orbs)
        else:
            _warn(warnings, "aspects_skipped_due_to_missing_positions")
        if profile: prof["aspects_ms"] = (perf_counter() - nat0) * 1000.0

    # Light validation
    validation_info: Optional[Dict[str, Any]] = None
    if (validation or "basic").lower() != "none":
        checks: List[Dict[str, Any]] = []
        ok = True
        try:
            if "Sun" in positions:
                rows_nat_sun = _planet_rows(jd_tt0, None, frame, ("Sun",), warnings)
                if rows_nat_sun:
                    if zmode == "sidereal":
                        _apply_ayanamsa(rows_nat_sun, ayanamsa_deg)
                    natSun = [r for r in rows_nat_sun if r["name"] == "Sun"][0]["lon"]
                    progSun = positions["Sun"]
                    sep = _abs_sep(natSun, progSun)
                    checks.append({"name": "sun_progressed_sanity", "sep_deg": float(sep), "pass": sep >= 0.5})
                    ok = ok and (sep >= 0.5)
                else:
                    checks.append({"name": "sun_progressed_sanity", "error": "natal_sun_missing", "pass": False})
                    ok = False
        except Exception as e:
            checks.append({"name": "sun_progressed_sanity", "error": type(e).__name__, "pass": False})
            ok = False
        validation_info = {"level": (validation or "basic").lower(), "pass": bool(ok), "checks": checks}

    meta: Dict[str, Any] = {
        "frame": frame,
        "zodiac_mode": zmode,
        "ayanamsa_deg": float(ayanamsa_deg),
        "house_system": house_system,
        "natal_timescales": ts_meta,
        "target_timescales": target_ts,
        "warnings": warnings,
        "mapping": mapping,
    }
    if profile:
        meta["profile"] = {**prof, "total_ms": (perf_counter() - t0) * 1000.0}

    if validation_info is not None:
        meta["validation"] = validation_info

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
