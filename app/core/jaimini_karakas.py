# app/core/jaimini_karakas.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Jaimini Chara Karakas — research-grade (sidereal, options-aware)

Features
- Computes 7- or 8-karaka scheme:
  AK, AmK, BK, MK, PK, GK, DK  (+ optional Pitri/extra when using 8)
- Sidereal-first (subtract ayanāṁśa from ecliptic-of-date longitudes)
- Rahu handling options (reverse-in-sign per common Jaimini practice)
- Deterministic tie-breaking to arcsecond; stable final tiebreak by body order
- Clean timescale normalization (jd_tt strict; civil -> timescales fallback)
- Returns both assignment and full ranking diagnostics

Public API
----------
    compute_chara_karakas(payload: dict, **options) -> dict
    chara_karakas_from_longitudes(nirayana_longitudes: dict[str,float], **options) -> dict

Inputs (payload)
- Either provide jd_tt (preferred) or {date,time,tz}
- ayanamsa: key string (e.g., "lahiri") or explicit degrees
- Optional: bodies list override

Options
- karaka_count: 7 or 8 (default 7)
- include_rahu: bool (default True)
- allow_ketu: bool (default False)
- rahu_reverse_in_sign: bool (default True)
- body_order_tiebreak: tuple[str,...] custom deterministic last tiebreak
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math

# timescales / time_kernel (strict jd_tt path)
try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

# ephemeris + ayanamsa
from app.core.ephem_singleton import TS, PLANETS
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    _EPH_OK = False

try:
    from app.core.ayanamsa import get_ayanamsa_deg as _get_ayanamsa_deg
except Exception:
    _get_ayanamsa_deg = None

# ───────────────────────── Utils ─────────────────────────

_SIGNS = (
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
)
_SIGN_LORD = {
    0:"Mars", 1:"Venus", 2:"Mercury", 3:"Moon", 4:"Sun", 5:"Mercury",
    6:"Venus", 7:"Mars", 8:"Jupiter", 9:"Saturn", 10:"Saturn", 11:"Jupiter"
}

_DEFAULT_BODIES = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu")  # Ketu excluded by default
_TIE_ORDER = ("Saturn","Jupiter","Mars","Sun","Venus","Mercury","Moon","Rahu","Ketu")  # deterministic last tiebreak

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _sign_index(lon: float) -> int:
    return int(math.floor(_norm360(lon) / 30.0)) % 12

def _in_sign_deg(lon: float) -> float:
    return _norm360(lon) % 30.0

def _q_arcsec(x: float) -> int:
    # quantize to integer arcseconds for stable tie-handling
    return int(round(float(x) * 3600.0))

@dataclass(frozen=True)
class _TSOut:
    jd_tt: float
    warnings: List[str]

def _timescales_from_payload(payload: Dict[str, Any]) -> _TSOut:
    jd_tt = payload.get("jd_tt")
    if isinstance(jd_tt, (int, float)):
        return _TSOut(float(jd_tt), [])
    birth = payload.get("birth") or {}
    d = payload.get("date") or birth.get("date")
    t = payload.get("time") or birth.get("time") or "12:00:00"
    tz = payload.get("tz") or payload.get("place_tz") or birth.get("tz") or "UTC"
    warns: List[str] = []
    if _tk is not None:
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(date=d, time=t, tz=tz)
                except TypeError:
                    out = fn(d, t, tz)
                if isinstance(out, dict):
                    return _TSOut(float(out["jd_tt"]), warns)
                if isinstance(out, (list, tuple)) and len(out) >= 2:
                    return _TSOut(float(out[1]), warns)
    if _ts is None:
        raise ValueError("timescales unavailable for civil→jd_tt")
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
    return _TSOut(jd_tt, warns)

def _resolve_ayanamsa(jd_tt: float, ay_opt: Any) -> Tuple[str, float, List[str]]:
    warns: List[str] = []
    if isinstance(ay_opt, (int, float)):
        return "explicit", float(ay_opt), warns
    key = str(ay_opt or "lahiri").strip().lower()
    if _get_ayanamsa_deg is None:
        # linearized fallback around J2000 if ayanamsa module is missing
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

# ───────────────────────── Ephemeris (sidereal) ─────────────────────────

def _sidereal_longitudes(jd_tt: float, bodies: List[str], *, ay_deg: float) -> Dict[str, float]:
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable")
    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore
    rows = ep.ecliptic_longitudes(float(jd_tt), bodies).get("results", [])
    out: Dict[str, float] = {}
    for r in rows or []:
        nm = str(r["name"])
        lon_trop = float(r["longitude"])
        out[nm] = _norm360(lon_trop - ay_deg)
    return out

# ───────────────────────── Karaka engine ─────────────────────────

def _labels_for_count(n: int) -> Tuple[str, ...]:
    # canonical 7: AK, AmK, BK, MK, PK, GK, DK
    base7 = ("Atmakaraka","Amatyakaraka","Bhratrukaraka","Matrukaraka","Putrakaraka","Gnatikaraka","Darakaraka")
    if n <= 7:
        return base7[:n]
    # 8-karaka (conventional extra slot as Pitru or fallback-extra)
    return base7 + ("Pitru/Extra",)

def _karaka_score_in_sign(lon_nira: float, *, reverse_for_rahu: bool, is_rahu: bool) -> float:
    deg = _in_sign_deg(lon_nira)
    if is_rahu and reverse_for_rahu:
        return 30.0 - deg
    return deg

def chara_karakas_from_longitudes(
    nirayana_longitudes: Dict[str, float],
    *,
    karaka_count: int = 7,
    include_rahu: bool = True,
    allow_ketu: bool = False,
    rahu_reverse_in_sign: bool = True,
    body_order_tiebreak: Tuple[str, ...] = _TIE_ORDER,
) -> Dict[str, Any]:
    """
    Pure combinator: feed sidereal longitudes (deg) of bodies; get assignments.
    """
    # Filter bodies and build candidates
    cand: List[Tuple[str, float, int, float, int]] = []  # (name, score_deg, score_arcsec, lon_abs, lon_arcsec)
    for name, lon in nirayana_longitudes.items():
        nm = str(name)
        if nm not in _DEFAULT_BODIES and nm not in ("Ketu",):
            # allow arbitrary bodies if passed in explicitly
            pass
        if nm == "Rahu" and not include_rahu:
            continue
        if nm == "Ketu" and not allow_ketu:
            continue
        is_rahu = (nm == "Rahu")
        score = _karaka_score_in_sign(float(lon), reverse_for_rahu=bool(rahu_reverse_in_sign), is_rahu=is_rahu)
        cand.append((nm, float(score), _q_arcsec(score), float(lon), _q_arcsec(_norm360(lon))))

    if not cand:
        return {"ok": False, "error": "no_bodies"}

    # Sort by: primary score_arcsec DESC, then lon_arcsec DESC, then deterministic body order
    rank_order = {b: i for i, b in enumerate(body_order_tiebreak)}
    def _key(row: Tuple[str, float, int, float, int]):
        nm, score_deg, score_arc, lon_abs, lon_arc = row
        return (-score_arc, -lon_arc, rank_order.get(nm, 999))

    cand.sort(key=_key)

    # Assign karakas
    klabels = _labels_for_count(int(karaka_count))
    chosen = cand[:len(klabels)]

    karakas: Dict[str, str] = {}
    for lab, row in zip(klabels, chosen):
        karakas[lab] = row[0]

    # Prepare ranking diagnostics
    ranking: List[Dict[str, Any]] = []
    for i, row in enumerate(cand, start=1):
        nm, score_deg, score_arc, lon_abs, lon_arc = row
        si = _sign_index(lon_abs)
        ranking.append({
            "rank": i,
            "planet": nm,
            "score_in_sign_deg": float(score_deg),
            "score_in_sign_arcsec": int(score_arc),
            "nirayana_longitude_deg": float(lon_abs),
            "sign_index": si,                 # 0..11
            "sign_name": _SIGNS[si],
            "in_sign_deg": float(_in_sign_deg(lon_abs)),
        })

    return {
        "ok": True,
        "scheme": f"{len(klabels)}-karaka",
        "karakas": karakas,          # {"Atmakaraka": "Saturn", ...}
        "ranking": ranking,          # detailed scores and positions
        "meta": {
            "include_rahu": bool(include_rahu),
            "allow_ketu": bool(allow_ketu),
            "rahu_reverse_in_sign": bool(rahu_reverse_in_sign),
            "karaka_count": int(karaka_count),
        }
    }

# ───────────────────────── Orchestrator (from payload) ─────────────────────────

def compute_chara_karakas(
    payload: Dict[str, Any],
    *,
    karaka_count: int = 7,
    include_rahu: bool = True,
    allow_ketu: bool = False,
    rahu_reverse_in_sign: bool = True,
    bodies: Optional[List[str]] = None,
    body_order_tiebreak: Tuple[str, ...] = _TIE_ORDER,
) -> Dict[str, Any]:
    """
    Normalize timescales and ayanāṁśa, compute sidereal longitudes, return karakas.
    """
    ts = _timescales_from_payload(payload)
    ay_key, ay_deg, ay_warns = _resolve_ayanamsa(ts.jd_tt, payload.get("ayanamsa"))
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "warnings": ay_warns}

    names = list(bodies or _DEFAULT_BODIES)
    try:
        lons = _sidereal_longitudes(float(ts.jd_tt), names, ay_deg=float(ay_deg))
    except Exception as e:
        return {"ok": False, "error": f"ephemeris_failed:{e}", "warnings": ay_warns}

    res = chara_karakas_from_longitudes(
        lons,
        karaka_count=karaka_count,
        include_rahu=include_rahu,
        allow_ketu=allow_ketu,
        rahu_reverse_in_sign=rahu_reverse_in_sign,
        body_order_tiebreak=body_order_tiebreak,
    )
    res.setdefault("meta", {})
    res["meta"].update({
        "ayanamsa_key": ay_key,
        "ayanamsa_deg": float(ay_deg),
        "jd_tt": float(ts.jd_tt),
    })
    if ay_warns:
        res["warnings"] = list(dict.fromkeys(list(res.get("warnings", [])) + ay_warns))
    return res
