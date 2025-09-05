# app/core/directions.py
# -*- coding: utf-8 -*-
"""
Directions (v11)
- Solar-Arc directions (Naibod & True-Sun), direct & converse.
- Hits to natal planets/angles/cusps with configurable orbs.
- Strict timescales; topocentric optional for snapshots; JSON-only outputs.

Public API
----------
compute_directions(
    natal: dict,
    *,
    method: str = "solar_arc",        # "solar_arc"
    rate: str = "naibod",             # "naibod" | "true_sun"
    target: dict | None = None,       # {"date","time","place_tz"} → “as-of” epoch
    years_after: float | None = None, # or explicit age in years
    jd_tt_natal: float | None = None,
    jd_ut1_natal: float | None = None,
    place: dict | None = None,        # {latitude, longitude, elev_m}; defaults to natal place
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",    # "tropical" | "sidereal"
    ayanamsa_deg: float = 0.0,
    arcs: str = "direct",             # "direct" | "converse" | "both"
    orbs: dict | None = None,         # keys like in synastry DEFAULT_ORBS
    include_hits_to: tuple[str,...] = ("planets","angles","cusps"),
    parallels: bool = False,          # (not typical for directions; default False)
    antiscia: bool = False,           # optional
    profile: bool = False,
    validation: str = "basic",        # "none" | "basic"
) -> dict

Return shape
------------
{
  "meta": {...},
  "epoch": {"years_since_birth": float, "jd_tt_progress_ref": float, ...},
  "arc_deg": {"direct": float, "converse": float},
  "positions": {
      "direct":   {"Sun":deg, ...},
      "converse": {"Sun":deg, ...} | None
  },
  "hits": {
      "direct":   [...],   # each: {dir:"direct", promissor, to, type, orb_deg, sep_deg, target}
      "converse": [...] | None
  }
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
AXES   = ("ASC","MC")
TROPICAL_YEAR_D = 365.242189
NAIBOD_DEG_PER_YEAR = 360.0 / TROPICAL_YEAR_D  # ≈ 0.985647 deg/yr

DEFAULT_ORBS: Dict[str, float] = {
    "conjunction": 1.5,   # tighter defaults common for directions
    "opposition": 1.5,
    "trine": 1.0,
    "square": 1.0,
    "sextile": 0.75,
    "quincunx": 0.5,
    "antiscia": 0.5,
}

ASPECT_ANGLES: Dict[str, float] = {
    "conjunction": 0.0,
    "sextile": 60.0,
    "square": 90.0,
    "trine": 120.0,
    "opposition": 180.0,
    "quincunx": 150.0,
}

# ── basic angle math ──────────────────────────────────────────────────────────
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

# ── helpers shared with other core files (inlined here) ───────────────────────
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

def _resolve_years_since_birth(natal: Dict[str, Any], jd_ut1_natal: float, years_after: Optional[float], target: Optional[Dict[str, Any]], warnings: List[str]) -> Tuple[float, Optional[Dict[str, Any]]]:
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

# ── aspect helpers (zodiacal + optional antiscia) ─────────────────────────────
def _zodiacal_hits(dir_rows: List[Dict[str, Any]], natal_targets: List[Dict[str, Any]], orbs: Dict[str, float], dir_tag: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pP = [r for r in dir_rows if r["name"] in MAJORS]
    for a in pP:
        for b in natal_targets:
            name_b = str(b["name"])
            lon_b  = float(b["lon"])
            sep = _abs_sep(a["lon"], lon_b)
            for typ, ang in ASPECT_ANGLES.items():
                O = orbs.get(typ, DEFAULT_ORBS.get(typ, 0.0))
                if O <= 0: continue
                tight = abs(sep - ang)
                if tight <= O:
                    out.append({
                        "dir": dir_tag,
                        "promissor": a["name"],
                        "to": name_b,
                        "type": typ,
                        "exact_deg": ang,
                        "sep_deg": sep,
                        "orb_deg": tight,
                        "target": "natal",
                        "mode": "zodiacal",
                    })
    return out

def _antiscia_of(lon_deg: float) -> float:
    return _wrap_deg(180.0 - _wrap_deg(lon_deg))

def _antiscia_hits(dir_rows: List[Dict[str, Any]], natal_targets: List[Dict[str, Any]], orbs: Dict[str, float], dir_tag: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    O = orbs.get("antiscia", DEFAULT_ORBS["antiscia"])
    if O <= 0: return out
    pP = [r for r in dir_rows if r["name"] in MAJORS]
    for a in pP:
        a_anti = _antiscia_of(a["lon"])
        for b in natal_targets:
            lon_b = float(b["lon"])
            sep0 = _abs_sep(a_anti, lon_b)
            if sep0 <= O:
                out.append({"dir":dir_tag, "promissor":a["name"], "to":b["name"], "type":"antiscia", "exact_deg":0.0, "sep_deg":sep0, "orb_deg":sep0, "target":"natal", "mode":"antiscia"})
            sep180 = min(_abs_sep(a_anti, lon_b+180.0), _abs_sep(a_anti, lon_b-180.0))
            if sep180 <= O:
                out.append({"dir":dir_tag, "promissor":a["name"], "to":b["name"], "type":"contra-antiscia", "exact_deg":180.0, "sep_deg":sep180, "orb_deg":sep180, "target":"natal", "mode":"antiscia"})
    return out

# ── core: solar-arc engine ────────────────────────────────────────────────────
def _solar_arc_degrees(
    natal_rows: List[Dict[str, Any]],
    jd_tt_natal: float,
    yrs: float,
    rate: str,
    place_for_progress: Optional[Dict[str, float]],
    frame: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    warnings: List[str],
) -> float:
    """Return direct-arc degrees for the chosen rate."""
    if rate == "naibod":
        return NAIBOD_DEG_PER_YEAR * yrs
    elif rate == "true_sun":
        # true-sun arc = (Sun longitude at natal + yrs days) - (Sun longitude at natal)
        jd_prog_tt = jd_tt_natal + float(yrs)  # 1 day per year
        rows_prog = _planet_rows(jd_prog_tt, place_for_progress, frame, ("Sun",), warnings)
        rows_nat  = [r for r in natal_rows if r["name"] == "Sun"]
        if not rows_nat:
            raise RuntimeError("Natal Sun longitude unavailable.")
        lon_nat = float(rows_nat[0]["lon"])
        lon_prog = float(rows_prog[0]["lon"])
        if zodiac_mode.lower() == "sidereal":
            lon_nat  = _wrap_deg(lon_nat  - ayanamsa_deg)
            lon_prog = _wrap_deg(lon_prog - ayanamsa_deg)
        # use forward arc (0..360)
        return _wrap_deg(lon_prog - lon_nat)
    else:
        raise ValueError("Unknown rate. Use 'naibod' or 'true_sun'.")

def _apply_arc(natal_rows: List[Dict[str, Any]], arc_deg: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in natal_rows:
        out.append({"name": r["name"], "lon": _wrap_deg(float(r["lon"]) + arc_deg)})
    return out

def _natal_targets(natal_rows: List[Dict[str, Any]], natal_houses: Optional[Dict[str, Any]], include: Tuple[str,...]) -> List[Dict[str, Any]]:
    t: List[Dict[str, Any]] = []
    if "planets" in include:
        for r in natal_rows:
            if r["name"] in MAJORS:
                t.append({"name": r["name"], "lon": float(r["lon"])})
    if "angles" in include and natal_houses:
        if "asc_deg" in natal_houses:
            t.append({"name":"ASC", "lon": float(natal_houses["asc_deg"])})
        if "mc_deg" in natal_houses:
            t.append({"name":"MC",  "lon": float(natal_houses["mc_deg"])})
    if "cusps" in include and natal_houses and isinstance(natal_houses.get("cusps_deg"), list):
        for i, c in enumerate(natal_houses["cusps_deg"], start=1):
            t.append({"name": f"Cusp {i}", "lon": float(c)})
    return t

# ── public API ────────────────────────────────────────────────────────────────
def compute_directions(
    natal: Dict[str, Any],
    *,
    method: str = "solar_arc",
    rate: str = "naibod",
    target: Optional[Dict[str, Any]] = None,
    years_after: Optional[float] = None,
    jd_tt_natal: Optional[float] = None,
    jd_ut1_natal: Optional[float] = None,
    place: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    arcs: str = "direct",
    orbs: Optional[Dict[str, float]] = None,
    include_hits_to: Tuple[str,...] = ("planets","angles","cusps"),
    parallels: bool = False,   # kept for extensibility; not used in solar-arc by default
    antiscia: bool = False,
    profile: bool = False,
    validation: str = "basic",
) -> Dict[str, Any]:
    """
    Compute Solar-Arc directions (direct/converse) and detect hits to natal targets.
    """
    if method.lower() != "solar_arc":
        raise ValueError("Only 'solar_arc' is implemented in v11 core. (Primary directions planned as Phase-2.)")

    t0 = perf_counter()
    prof: Dict[str, float] = {}
    warnings: List[str] = []
    _orbs = {**DEFAULT_ORBS, **(orbs or {})}
    arcs = arcs.lower()

    # Timescales (strict preferred)
    ts0 = perf_counter()
    jd_tt0, jd_ut10, ts_meta = _resolve_ts_from_natal(natal, jd_tt_natal, jd_ut1_natal, warnings)
    prof["timescales_ms"] = (perf_counter() - ts0) * 1000.0 if profile else 0.0

    # Years since birth
    y0 = perf_counter()
    yrs, target_ts = _resolve_years_since_birth(natal, jd_ut10, years_after, target, warnings)
    prof["resolve_years_ms"] = (perf_counter() - y0) * 1000.0 if profile else 0.0

    # Place (topocentric snapshots if given)
    natal_place = _to_place(natal, None)
    place_used = _to_place(natal, place) or natal_place

    # Natal positions (zodiac per mode)
    ep0 = perf_counter()
    natal_rows = _planet_rows(jd_tt0, place_used, frame, MAJORS, warnings)
    if zodiac_mode.lower() == "sidereal":
        for r in natal_rows:
            r["lon"] = _wrap_deg(float(r["lon"]) - ayanamsa_deg)
    prof["natal_ephemeris_ms"] = (perf_counter() - ep0) * 1000.0 if profile else 0.0

    # Natal houses (for target list)
    hs0 = perf_counter()
    natal_houses = None
    if _compute_houses_policy is None:
        _warn(warnings, "houses_policy_unavailable")
    elif place_used is None:
        _warn(warnings, "houses_missing_place")
    else:
        try:
            natal_houses = _compute_houses_policy(
                jd_tt=jd_tt0, jd_ut1=jd_ut10,
                latitude=place_used["latitude"], longitude=place_used["longitude"],
                elevation_m=place_used.get("elev_m", 0.0), system=house_system,
            )
        except Exception as e:
            _warn(warnings, f"houses_compute_failed:{type(e).__name__}")
    prof["houses_ms"] = (perf_counter() - hs0) * 1000.0 if profile else 0.0

    # Arc (direct) & optional converse
    arc_dir = _solar_arc_degrees(
        natal_rows, jd_tt0, yrs, rate, place_used, frame, zodiac_mode, ayanamsa_deg, warnings
    )
    arc_con = arc_dir if arcs in ("converse","both") else None

    # Apply arc(s)
    rows_direct = _apply_arc(natal_rows, arc_dir) if arcs in ("direct","both") else []
    rows_converse = _apply_arc(natal_rows, -arc_dir) if arcs in ("converse","both") else []

    # Build natal targets
    natal_targets = _natal_targets(natal_rows, natal_houses, include_hits_to)

    # Hits
    hits_direct: List[Dict[str, Any]] = []
    hits_converse: List[Dict[str, Any]] = []
    if rows_direct:
        hits_direct = _zodiacal_hits(rows_direct, natal_targets, _orbs, "direct")
        if antiscia:
            hits_direct += _antiscia_hits(rows_direct, natal_targets, _orbs, "direct")
    if rows_converse:
        hits_converse = _zodiacal_hits(rows_converse, natal_targets, _orbs, "converse")
        if antiscia:
            hits_converse += _antiscia_hits(rows_converse, natal_targets, _orbs, "converse")

    # Validation (light sanity)
    validation_info = None
    if validation and validation.lower() != "none":
        checks: List[Dict[str, Any]] = []
        ok = True
        try:
            # Sun should direct by ~yrs degrees (within generous 2° vs selected rate)
            idx = _rows_index(natal_rows)
            sun_nat = float(idx["Sun"]["lon"])
            if rows_direct:
                sun_dir = float([r for r in rows_direct if r["name"]=="Sun"][0]["lon"])
                sep = _abs_sep(sun_nat, sun_dir)
                exp = arc_dir
                checks.append({"name":"sun_arc_sanity","sep_deg": float(sep), "exp_arc_deg": float(exp), "pass": abs(sep-exp) <= 2.0})
                ok = ok and (abs(sep-exp) <= 2.0)
        except Exception as e:
            checks.append({"name":"sun_arc_sanity","error": type(e).__name__, "pass": False})
            ok = False
        validation_info = {"level": validation.lower(), "pass": bool(ok), "checks": checks}

    meta: Dict[str, Any] = {
        "frame": frame,
        "zodiac_mode": zodiac_mode.lower(),
        "ayanamsa_deg": float(ayanamsa_deg),
        "house_system": house_system,
        "natal_timescales": ts_meta,
        "target_timescales": target_ts,
        "method": "solar_arc",
        "rate": rate.lower(),
        "warnings": warnings,
    }
    if profile:
        meta["profile"] = prof
    meta.setdefault("notes", []).append("Primary (semi-arc) directions are planned as Phase-2; this module currently implements Solar-Arc (Naibod & True-Sun).")

    return {
        "meta": meta,
        "epoch": {
            "years_since_birth": float(yrs),
            "jd_tt_progress_ref": float(jd_tt0 + yrs),  # 1 day per year reference used for true_sun arc
            "place_used": place_used,
        },
        "arc_deg": {
            "direct": float(arc_dir),
            "converse": float(arc_con) if arc_con is not None else None,
        },
        "positions": {
            "direct":  {r["name"]: float(r["lon"]) for r in rows_direct} if rows_direct else None,
            "converse":{r["name"]: float(r["lon"]) for r in rows_converse} if rows_converse else None,
        },
        "hits": {
            "direct": hits_direct if rows_direct else None,
            "converse": hits_converse if rows_converse else None,
        },
    }
