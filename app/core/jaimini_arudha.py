# app/core/jaimini_arudha.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Jaimini Arudha Padas — research-grade, sidereal-first, deterministic

Outputs
- A1..A12 (Arudha padas for all twelve houses; A1 = Arudha Lagna)
- UL (Upapada) = Arudha of 12th
- Full diagnostics per house: house sign, lord, lord's sign, distance, initial pada,
  exception adjustments applied, final pada.

Design
- Sidereal pipeline: ecliptic-of-date longitudes minus ayanāṁśa
- Gold ascendant: imports Asc from houses_advanced (ERFA GAST + true ε)
- Strict timescales: jd_tt + jd_ut1 from time_kernel/timescales
- Options to match schools:
    * exception_scope: "all" (default), "lagna_only", or "none"
      (When the computed pada falls in the same sign as the house or its 7th:
       add 10 signs in the chosen scope.)
    * sign_lord_variant: "classical" (default; Scorpio→Mars, Aquarius→Saturn)
                         "nodes_modern" (Scorpio→Ketu, Aquarius→Rahu)
    * body_set: planets to fetch (defaults to grahas + nodes so variants always work)

Public API
----------
    compute_arudhas(payload: dict, **options) -> dict
        # payload must include birth site to compute Lagna:
        #   date, time, tz, latitude, longitude  (or jd_tt + jd_ut1 + site)
        # Optional: ayanamsa (key or explicit degrees)

    arudhas_from_longitudes(
        *,
        lagna_sidereal_deg: float | None,
        planet_longitudes_nirayana: dict[str, float],
        latitude: float | None = None,  # only needed if lagna not provided and you want us to derive it
        longitude: float | None = None,
        jd_tt: float | None = None,
        jd_ut1: float | None = None,
        ayanamsa_deg: float | None = None,
        **options
    ) -> dict
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math

# timescales / time_kernel (jd_tt, jd_ut1)
try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

# ephemeris (sidereal)
from app.core.ephem_singleton import TS, PLANETS
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    _EPH_OK = False

# ayanāṁśa
try:
    from app.core.ayanamsa import get_ayanamsa_deg as _get_ayanamsa_deg
except Exception:
    _get_ayanamsa_deg = None

# ascendant (gold path)
try:
    from app.core.houses_advanced import compute_house_system as _compute_houses
    _HOUSES_OK = True
except Exception:
    _HOUSES_OK = False

# ───────────────────────── utilities ─────────────────────────

_SIGNS = (
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
)

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _sign_index(lon: float) -> int:
    return int(math.floor(_norm360(lon) / 30.0)) % 12

def _in_sign_deg(lon: float) -> float:
    return _norm360(lon) % 30.0

def _add_signs(idx: int, n: int) -> int:
    return (int(idx) + int(n)) % 12

@dataclass(frozen=True)
class _TSOut:
    jd_tt: float
    jd_ut1: float
    warnings: List[str]

def _timescales_from_payload(payload: Dict[str, Any]) -> _TSOut:
    jd_tt = payload.get("jd_tt")
    jd_ut1 = payload.get("jd_ut1")
    warns: List[str] = []
    if isinstance(jd_tt, (int, float)) and isinstance(jd_ut1, (int, float)):
        return _TSOut(float(jd_tt), float(jd_ut1), warns)

    # Civil → timescales via time_kernel or timescales
    birth = payload.get("birth") or {}
    d = payload.get("date") or birth.get("date")
    t = payload.get("time") or birth.get("time") or "12:00:00"
    tz = payload.get("tz") or payload.get("place_tz") or birth.get("tz") or "UTC"

    if _tk is not None:
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(date=d, time=t, tz=tz)
                except TypeError:
                    out = fn(d, t, tz)
                if isinstance(out, dict):
                    ju = float(out.get("jd_ut") or out.get("jd_utc") or out.get("jd_ut1"))
                    jt = float(out["jd_tt"])
                    j1 = float(out.get("jd_ut1") or ju)
                    return _TSOut(jt, j1, warns)
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    ju, jt, j1 = map(float, out[:3])
                    return _TSOut(jt, j1, warns)

    if _ts is None:
        raise ValueError("timescales unavailable (need jd_tt & jd_ut1 or civil+tz)")

    jd_ut = float(_ts.julian_day_utc(d, t, tz))
    try:
        y, m = map(int, str(d).split("-")[:2])
    except Exception:
        y, m = 2000, 1
    try:
        jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m))
    except Exception:
        jd_tt = jd_ut + 69.0/86400.0
        warns.append("deltaT_fallback_69s")
    jd_ut1 = jd_ut  # DUT1 small; supply via time_kernel for strict if desired
    return _TSOut(jd_tt, jd_ut1, warns)

def _resolve_ayanamsa(jd_tt: float, ay_opt: Any) -> Tuple[str, float, List[str]]:
    warns: List[str] = []
    if isinstance(ay_opt, (int, float)):
        return "explicit", float(ay_opt), warns
    key = str(ay_opt or "lahiri").strip().lower()
    if _get_ayanamsa_deg is None:
        # linearized fallback around J2000 (keeps continuity; not for publication use)
        AY_J2000_DEG = (23 + 51/60 + 26.26/3600)
        RATE_AS_PER_YR = 50.290966
        years = (float(jd_tt) - 2451545.0) / 365.25
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
        warns.append("ayanamsa_fallback_lahiri_linearized")
        return key, ay, warns
    try:
        return key, float(_get_ayanamsa_deg(float(jd_tt), key)), warns
    except Exception as e:
        warns.append(f"ayanamsa_resolve_error:{e}")
        AY_J2000_DEG = (23 + 51/60 + 26.26/3600)
        RATE_AS_PER_YR = 50.290966
        years = (float(jd_tt) - 2451545.0) / 365.25
        ay = AY_J2000_DEG + (RATE_AS_PER_YR * years) / 3600.0
        warns.append("ayanamsa_fallback_lahiri_linearized")
        return key, ay, warns

# ───────────────────────── sign lords (variants) ─────────────────────────

def _sign_lords(variant: str = "classical") -> Dict[int, str]:
    if str(variant).lower().startswith("nodes"):
        # nodes_modern
        return {
            0:"Mars", 1:"Venus", 2:"Mercury", 3:"Moon", 4:"Sun", 5:"Mercury",
            6:"Venus", 7:"Ketu", 8:"Jupiter", 9:"Saturn", 10:"Rahu", 11:"Jupiter"
        }
    # classical
    return {
        0:"Mars", 1:"Venus", 2:"Mercury", 3:"Moon", 4:"Sun", 5:"Mercury",
        6:"Venus", 7:"Mars", 8:"Jupiter", 9:"Saturn", 10:"Saturn", 11:"Jupiter"
    }

# ───────────────────────── ephemeris (sidereal) ─────────────────────────

_DEFAULT_BODY_SET = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu")

def _sidereal_longitudes(jd_tt: float, names: List[str], *, ay_deg: float) -> Dict[str, float]:
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable")
    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore
    rows = ep.ecliptic_longitudes(float(jd_tt), names).get("results", [])
    out: Dict[str, float] = {}
    for r in rows or []:
        nm = str(r["name"])
        lon_trop = float(r["longitude"])
        out[nm] = _norm360(lon_trop - ay_deg)
    return out

# ───────────────────────── ascendant → Lagna sign ─────────────────────────

def _lagna_sidereal_deg_from_site(*, jd_tt: float, jd_ut1: float, lat: float, lon: float, ay_deg: float) -> float:
    if not _HOUSES_OK:
        raise RuntimeError("houses_advanced unavailable for ascendant computation")
    payload = _compute_houses(
        latitude=float(lat), longitude=float(lon),
        house_system="equal", jd_ut=float(jd_tt), jd_tt=float(jd_tt), jd_ut1=float(jd_ut1)
    )
    asc_tropical = float(payload["asc_deg"])
    return _norm360(asc_tropical - ay_deg)

# ───────────────────────── core arudha math ─────────────────────────

def _arudha_for_house(
    *,
    house_sign_idx: int,
    lords: Dict[int, str],
    nirayana_longs: Dict[str, float],
    exception_scope: str = "all",
) -> Tuple[int, Dict[str, Any]]:
    """
    Return (final_pada_sign_idx, debug)
    """
    lord_name = lords[house_sign_idx]
    lon_lord = nirayana_longs.get(lord_name)
    dbg: Dict[str, Any] = {
        "house_sign_index": int(house_sign_idx),
        "house_sign_name": _SIGNS[house_sign_idx],
        "lord": lord_name,
        "exception_applied": False,
        "exception_reason": None
    }
    if lon_lord is None or not math.isfinite(lon_lord):
        dbg["error"] = f"missing_longitude:{lord_name}"
        # Fail gracefully: keep pada at house sign (caller can mark warning)
        return int(house_sign_idx), dbg

    lord_sign_idx = _sign_index(lon_lord)
    d = 1 + ((lord_sign_idx - house_sign_idx) % 12)  # inclusive sign count
    initial = _add_signs(lord_sign_idx, d - 1)

    dbg.update({
        "lord_sign_index": int(lord_sign_idx),
        "lord_sign_name": _SIGNS[lord_sign_idx],
        "distance_signs": int(d),
        "initial_pada_index": int(initial),
        "initial_pada_name": _SIGNS[initial],
    })

    # exceptions: if pada equals house or 7th from it → add 10 signs
    def _needs_exception() -> bool:
        if exception_scope == "none":
            return False
        if exception_scope == "lagna_only":
            # caller will only pass house_sign_idx==lagna for AL
            pass
        same = (initial == house_sign_idx)
        opp = (initial == _add_signs(house_sign_idx, 6))
        return same or opp

    final = initial
    if _needs_exception():
        final = _add_signs(initial, 10)
        dbg["exception_applied"] = True
        dbg["exception_reason"] = "same_or_7th_add10"

    dbg["final_pada_index"] = int(final)
    dbg["final_pada_name"] = _SIGNS[final]
    return int(final), dbg

# ───────────────────────── public combinators ─────────────────────────

def arudhas_from_longitudes(
    *,
    lagna_sidereal_deg: Optional[float],
    planet_longitudes_nirayana: Dict[str, float],
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    ayanamsa_deg: Optional[float] = None,
    sign_lord_variant: str = "classical",
    exception_scope: str = "all",
) -> Dict[str, Any]:
    """
    Compute A1..A12 (and UL) from provided sidereal longitudes.
    If lagna_sidereal_deg is None, caller must also provide site (lat,lon), jd_tt, jd_ut1 and ayanamsa_deg.
    """
    warns: List[str] = []
    if lagna_sidereal_deg is None:
        if None in (latitude, longitude, jd_tt, jd_ut1, ayanamsa_deg):
            return {"ok": False, "error": "lagna_required_or_provide_site_and_timescales"}
        try:
            lagna_sidereal_deg = _lagna_sidereal_deg_from_site(
                jd_tt=float(jd_tt), jd_ut1=float(jd_ut1),
                lat=float(latitude), lon=float(longitude), ay_deg=float(ayanamsa_deg)
            )
        except Exception as e:
            return {"ok": False, "error": f"asc_compute_failed:{e}"}

    lagna_sidereal_deg = float(lagna_sidereal_deg)
    lagna_sign = _sign_index(lagna_sidereal_deg)
    lords = _sign_lords(sign_lord_variant)

    # house signs (whole sign houses)
    house_signs = [ (lagna_sign + i) % 12 for i in range(12) ]

    # Compute for all houses; apply exception_scope for all by default, but enforce lagna-only behavior for A1 explicitly.
    results: Dict[str, Dict[str, Any]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for i, hs in enumerate(house_signs, start=1):
        scope = exception_scope
        if exception_scope == "lagna_only" and i != 1:
            scope = "none"
        pada_idx, dbg = _arudha_for_house(
            house_sign_idx=hs,
            lords=lords,
            nirayana_longs=planet_longitudes_nirayana,
            exception_scope=scope
        )
        key = f"A{i}"
        results[key] = {"sign_index": int(pada_idx), "sign_name": _SIGNS[pada_idx]}
        diagnostics[key] = dbg

    # Upapada = A12
    results["UL"] = dict(results["A12"])

    out = {
        "ok": True,
        "arudhas": results,
        "meta": {
            "lagna_sign_index": int(lagna_sign),
            "lagna_sign_name": _SIGNS[lagna_sign],
            "sign_lord_variant": str(sign_lord_variant),
            "exception_scope": str(exception_scope),
        },
        "diagnostics": diagnostics
    }
    if warns:
        out["warnings"] = list(dict.fromkeys(warns))
    return out

def compute_arudhas(
    payload: Dict[str, Any],
    *,
    sign_lord_variant: str = "classical",
    exception_scope: str = "all",
    bodies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Orchestrator from civil or jd payload:
      Required for Lagna: latitude, longitude (degrees)
      Provide either (jd_tt & jd_ut1) or (date, time, tz)

    Optional:
      ayanamsa: key (e.g., "lahiri") or explicit degrees
      bodies: override body list used to fetch longitudes
    """
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable"}

    # site
    if not all(k in payload for k in ("latitude","longitude")):
        return {"ok": False, "error": "site_required_for_lagna"}
    lat = float(payload["latitude"]); lon = float(payload["longitude"])

    # timescales
    ts = _timescales_from_payload(payload)

    # ayanāṁśa
    ay_key, ay_deg, ay_warns = _resolve_ayanamsa(ts.jd_tt, payload.get("ayanamsa"))

    # sidereal longitudes
    names = list(bodies or _DEFAULT_BODY_SET)
    try:
        lons = _sidereal_longitudes(float(ts.jd_tt), names, ay_deg=float(ay_deg))
    except Exception as e:
        return {"ok": False, "error": f"ephemeris_failed:{e}"}

    # derive Lagna (sidereal)
    try:
        lagna_sid = _lagna_sidereal_deg_from_site(
            jd_tt=float(ts.jd_tt), jd_ut1=float(ts.jd_ut1),
            lat=lat, lon=lon, ay_deg=float(ay_deg)
        )
    except Exception as e:
        return {"ok": False, "error": f"asc_compute_failed:{e}"}

    # combine
    res = arudhas_from_longitudes(
        lagna_sidereal_deg=float(lagna_sid),
        planet_longitudes_nirayana=lons,
        sign_lord_variant=sign_lord_variant,
        exception_scope=exception_scope,
    )
    res.setdefault("meta", {})
    res["meta"].update({
        "ayanamsa_key": ay_key,
        "ayanamsa_deg": float(ay_deg),
        "jd_tt": float(ts.jd_tt),
        "jd_ut1": float(ts.jd_ut1),
        "latitude": float(lat),
        "longitude": float(lon),
    })
    if ay_warns:
        res["warnings"] = list(dict.fromkeys(list(res.get("warnings", [])) + ay_warns))
    return res
