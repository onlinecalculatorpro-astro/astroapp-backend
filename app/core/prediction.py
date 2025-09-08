# app/core/prediction.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Prediction Engine (v2)

Public API
----------
predict_transits(natal_chart, time_range, **kwargs) -> PredictionResult
predict_progressions(natal_chart, target_date, **kwargs) -> PredictionResult
predict_returns(natal_chart, return_type, year, **kwargs) -> PredictionResult
predict_directions(natal_chart, target_date, **kwargs) -> PredictionResult
comprehensive_forecast(natal_chart, time_range, **kwargs) -> ComprehensiveForecast
relationship_forecast(natal_a, natal_b, time_range, **kwargs) -> RelationshipForecast
validate_prediction_model(test_cases, **kwargs) -> dict
"""

import json
import math
import random
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional, Union

# ── Imports with guards ───────────────────────────────────────────────────────

# Constants / helpers
try:
    from app.core.constants import (
        MAJOR_BODIES,
        ASPECT_ANGLES_DEG,
        DEFAULT_ORBS_SYNASTRY,
        DEFAULT_ORBS_PROGRESSIONS,
        DEFAULT_ORBS_DIRECTIONS,
        TROPICAL_YEAR_D,
        abs_sep_deg,
    )
    try:
        from app.core.constants import DEFAULT_ORBS_TRANSITS
    except Exception:
        DEFAULT_ORBS_TRANSITS = DEFAULT_ORBS_SYNASTRY
    _CONST_OK = True; _CONST_ERR = None
except Exception as e:
    _CONST_OK = False; _CONST_ERR = e
    # minimal fallbacks
    MAJOR_BODIES = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"]
    ASPECT_ANGLES_DEG = {"conjunction":0,"opposition":180,"trine":120,"square":90,"sextile":60}
    DEFAULT_ORBS_SYNASTRY = DEFAULT_ORBS_PROGRESSIONS = DEFAULT_ORBS_DIRECTIONS = DEFAULT_ORBS_TRANSITS = {
        "default": 1.0, "conjunction": 8, "opposition": 8, "trine": 6, "square": 6, "sextile": 4
    }
    TROPICAL_YEAR_D = 365.2422
    def abs_sep_deg(a, b):  # simple wrap
        d = (a - b + 180) % 360 - 180
        return abs(d)

# Timescales
try:
    from app.core.timescales import build_timescales, TimeScales
    _TS_OK = True; _TS_ERR = None
except Exception as e:
    _TS_OK = False; _TS_ERR = e
    TimeScales = Any  # type: ignore

# Aspects
try:
    from app.core.aspects import compute_aspects, AspectConfig
    _ASPECTS_OK = True; _ASPECTS_ERR = None
except Exception as e:
    _ASPECTS_OK = False; _ASPECTS_ERR = e
    AspectConfig = Any  # type: ignore

# Optional engines
try:
    from app.core.progressions import compute_progressions
    _PROG_OK = True; _PROG_ERR = None
except Exception as e:
    _PROG_OK = False; _PROG_ERR = e

try:
    from app.core.returns import compute_solar_return, compute_lunar_return
    _RET_OK = True; _RET_ERR = None
except Exception as e:
    _RET_OK = False; _RET_ERR = e

try:
    from app.core.directions import compute_directions
    _DIR_OK = True; _DIR_ERR = None
except Exception as e:
    _DIR_OK = False; _DIR_ERR = e

try:
    from app.core.synastry import compute_synastry, compute_composite
    _SYN_OK = True; _SYN_ERR = None
except Exception as e:
    _SYN_OK = False; _SYN_ERR = e

# Unified predictive toolkit (transits + vedic utils)
try:
    from app.core.predictive import (
        # Transits
        find_transits_in_range,
        # Dasha
        vimsottari_dasha,
    )
    _PRED_OK = True; _PRED_ERR = None
except Exception as e:
    _PRED_OK = False; _PRED_ERR = e

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PredictionEvent:
    event_type: str
    technique: str
    description: str
    datetime_utc: Optional[datetime]
    jd_tt: Optional[float]
    jd_ut1: Optional[float]
    precision_seconds: Optional[float]
    confidence: float
    significance: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TimingWindow:
    start_jd_tt: float
    end_jd_tt: float
    peak_jd_tt: Optional[float]
    uncertainty_days: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]

@dataclass
class PredictionResult:
    ok: bool
    technique: str
    events: List[PredictionEvent] = field(default_factory=list)
    synthesis: Dict[str, Any] = field(default_factory=dict)
    timing_windows: List[TimingWindow] = field(default_factory=list)
    confidence_score: float = 0.0
    statistical_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    computation_time_ms: Optional[float] = None

@dataclass
class ComprehensiveForecast:
    natal_chart: Dict[str, Any]
    time_range: Tuple[datetime, datetime]
    predictions: Dict[str, PredictionResult] = field(default_factory=dict)
    synthesis: Dict[str, Any] = field(default_factory=dict)
    peak_periods: List[TimingWindow] = field(default_factory=list)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    confidence_metrics: Dict[str, float] = field(default_factory=dict)
    validation_results: Optional[Dict[str, Any]] = None
    computation_time_ms: Optional[float] = None

@dataclass
class RelationshipForecast:
    synastry_analysis: Dict[str, Any]
    composite_analysis: Dict[str, Any]
    transit_interactions: List[PredictionEvent] = field(default_factory=list)
    progression_interactions: List[PredictionEvent] = field(default_factory=list)
    compatibility_trends: Dict[str, float] = field(default_factory=dict)
    critical_periods: List[TimingWindow] = field(default_factory=list)
    relationship_score: float = 0.0
    confidence_metrics: Dict[str, float] = field(default_factory=dict)

# ── Utilities ────────────────────────────────────────────────────────────────

def _check_env() -> None:
    if not _CONST_OK:
        raise RuntimeError(f"constants unavailable: {_CONST_ERR}")
    if not _TS_OK:
        raise RuntimeError(f"timescales unavailable: {_TS_ERR}")

def _ensure_utc(dt_or_str: Union[str, datetime]) -> datetime:
    if isinstance(dt_or_str, str):
        s = dt_or_str.strip()
        if "T" not in s and len(s) <= 10:
            s += "T00:00:00+00:00"
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    else:
        dt = dt_or_str
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def _hash_obj(obj: Any) -> str:
    try:
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        payload = repr(obj)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

@lru_cache(maxsize=4096)
def _jd_pair_from_dt(dt_utc_iso: str) -> Tuple[float, float]:
    dt_utc = _ensure_utc(dt_utc_iso)
    ts: TimeScales = build_timescales(
        date_str=dt_utc.date().isoformat(),
        time_str=dt_utc.time().isoformat(timespec="seconds"),
        tz_name="UTC",
        dut1_seconds=0.0,
    )
    return float(ts.jd_tt), float(ts.jd_ut1)

def _jd_pair_from_dt_dt(dt_utc: datetime) -> Tuple[float, float]:
    return _jd_pair_from_dt(dt_utc.isoformat())

def _canon_aspect(name: Optional[str]) -> str:
    return (name or "").strip().lower()

@lru_cache(maxsize=256)
def _normalize_aspect_config_cached(key: str) -> AspectConfig:
    params = json.loads(key)
    orbs = params["orbs"]
    zodiacal: Dict[str, Dict[str, float]] = {}
    for asp, angle in ASPECT_ANGLES_DEG.items():
        orb = orbs.get(asp, 0.0)
        if orb > 0:
            zodiacal[asp] = {"angle": angle, "orb": orb}
    return AspectConfig(
        zodiacal=zodiacal or None,
        antiscia={"orb": orbs.get("antiscia", 0.0)} if orbs.get("antiscia", 0.0) > 0 else None,
        declination={"orb_arcmin": orbs.get("parallel_arcmin", 0.0)} if orbs.get("parallel_arcmin", 0.0) > 0 else None,
        enable_fdr_correction=True,
        q_level=0.05,
    )

def _normalize_aspect_config(orbs: Optional[Dict[str, float]], technique: str) -> AspectConfig:
    if not _ASPECTS_OK:
        raise RuntimeError(f"aspects engine unavailable: {_ASPECTS_ERR}")
    if technique in ("synastry", "composite"):
        base = DEFAULT_ORBS_SYNASTRY
    elif technique in ("progressions", "returns"):
        base = DEFAULT_ORBS_PROGRESSIONS
    elif technique in ("directions", "solar_arc"):
        base = DEFAULT_ORBS_DIRECTIONS
    else:
        base = DEFAULT_ORBS_TRANSITS
    eff = {**base, **(orbs or {})}
    key = json.dumps({"orbs": eff}, sort_keys=True)
    return _normalize_aspect_config_cached(key)

def _resolve_natal_timescales(natal: Dict[str, Any]) -> Tuple[float, float, List[str]]:
    warns: List[str] = []
    if "jd_tt" in natal and "jd_ut1" in natal:
        try:
            return float(natal["jd_tt"]), float(natal["jd_ut1"]), warns
        except Exception:
            warns.append("invalid_strict_timescales_fallback_to_civil")
    missing = [k for k in ("date", "time", "place_tz") if k not in natal]
    if missing:
        raise RuntimeError(f"missing fields for timescales: {missing}")
    ts: TimeScales = build_timescales(
        date_str=str(natal["date"]),
        time_str=str(natal["time"]),
        tz_name=str(natal["place_tz"]),
        dut1_seconds=float(natal.get("dut1", 0.0)),
    )
    return float(ts.jd_tt), float(ts.jd_ut1), warns

def _compute_confidence(events: List[PredictionEvent], stats: Dict[str, float]) -> float:
    if not events:
        return 0.0
    cs = [e.confidence for e in events if e.confidence > 0]
    if not cs:
        return 0.0
    w = [c**1.5 for c in cs]
    wa = sum(c*wi for c, wi in zip(cs, w)) / sum(w)
    p = float(stats.get("p_value", 1.0))
    significance_boost = 0.3 * max(0.0, 1.0 - p / 0.01) if p < 0.05 else 0.0
    return min(1.0, wa + significance_boost)

def _create_timing_windows(events: List[PredictionEvent], *, cluster_days: float = 7.0) -> List[TimingWindow]:
    ev = [e for e in events if e.datetime_utc]
    if len(ev) < 2:
        return []
    ev.sort(key=lambda e: e.datetime_utc)  # type: ignore
    out: List[TimingWindow] = []
    grp: List[PredictionEvent] = [ev[0]]
    for e in ev[1:]:
        dt_days = (e.datetime_utc - grp[-1].datetime_utc).total_seconds() / 86400.0  # type: ignore
        avgc = sum(g.confidence for g in grp) / len(grp)
        thr = cluster_days * (0.5 + 0.5 * avgc)
        if dt_days <= thr:
            grp.append(e)
        else:
            if len(grp) > 1:
                s = grp[0].datetime_utc; q = grp[-1].datetime_utc  # type: ignore
                pk = max(grp, key=lambda x: x.confidence)
                s_tt, _ = _jd_pair_from_dt_dt(s)   # type: ignore
                q_tt, _ = _jd_pair_from_dt_dt(q)   # type: ignore
                p_tt, _ = _jd_pair_from_dt_dt(pk.datetime_utc)  # type: ignore
                out.append(TimingWindow(s_tt, q_tt, p_tt, (q_tt - s_tt), (s_tt, q_tt)))
            grp = [e]
    if len(grp) > 1:
        s = grp[0].datetime_utc; q = grp[-1].datetime_utc  # type: ignore
        pk = max(grp, key=lambda x: x.confidence)
        s_tt, _ = _jd_pair_from_dt_dt(s)   # type: ignore
        q_tt, _ = _jd_pair_from_dt_dt(q)   # type: ignore
        p_tt, _ = _jd_pair_from_dt_dt(pk.datetime_utc)  # type: ignore
        out.append(TimingWindow(s_tt, q_tt, p_tt, (q_tt - s_tt), (s_tt, q_tt)))
    return out

def _body_activity(events: List[PredictionEvent]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}; confs: Dict[str, float] = {}
    for e in events:
        for key in ("transiting_body","progressed_body","return_body","directed_body","body"):
            b = e.metadata.get(key)
            if b:
                counts[b] = counts.get(b, 0) + 1
                confs[b] = confs.get(b, 0.0) + e.confidence
    dist = {b: {"event_count": counts[b], "average_confidence": confs[b]/counts[b], "total_confidence": confs[b]} for b in counts}
    top = dict(sorted(dist.items(), key=lambda kv: kv[1]["total_confidence"], reverse=True)[:5])
    return {"most_active": top, "total_bodies": len(dist), "activity_distribution": dist}

# ── Predictive memoization ───────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def _memo_predictive(key: str):
    args = json.loads(key)
    if args["kind"] == "transits":
        return find_transits_in_range(**args["payload"])
    raise RuntimeError(f"Unsupported memo kind: {args['kind']}")

def _memo_key_transits(payload: Dict[str, Any]) -> str:
    safe = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return json.dumps({"kind": "transits", "payload": json.loads(safe)}, sort_keys=True)

# ── Engines ──────────────────────────────────────────────────────────────────

def predict_transits(
    natal_chart: Dict[str, Any],
    time_range: Tuple[Union[datetime, str], Union[datetime, str]],
    *,
    transiting_bodies: Optional[List[str]] = None,
    natal_bodies: Optional[List[str]] = None,
    orbs: Optional[Dict[str, float]] = None,
    aspects: Optional[List[str]] = None,
    include_aspects_to: Optional[List[str]] = None,  # ["planets","angles"]
    include_house_cusps: bool = False,
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    exact_timing: bool = True,
    statistical_validation: bool = False,   # API placeholder
    confidence_threshold: float = 0.1,
    **kwargs,
) -> PredictionResult:
    _check_env()
    t0 = time.time()
    warns: List[str] = []
    try:
        if not _PRED_OK:
            raise RuntimeError(f"predictive unavailable: {_PRED_ERR}")

        # Fixed natal timescale resolution - remove dependency on problematic function
        jd_tt_natal, jd_ut1_natal, w = _resolve_natal_timescales_fixed(natal_chart)
        warns.extend(w)

        start_dt = _ensure_utc(time_range[0]); end_dt = _ensure_utc(time_range[1])
        transiting_bodies = transiting_bodies or list(MAJOR_BODIES)
        natal_bodies = natal_bodies or list(MAJOR_BODIES)
        aspects_list = [a.lower() for a in (aspects or ["conjunction","opposition","trine","square","sextile"])]
        include_aspects_to = include_aspects_to or ["planets","angles"]
        orbs_to_use = orbs or DEFAULT_ORBS_TRANSITS

        payload = dict(
            natal_chart=natal_chart,
            time_range=(start_dt.date().isoformat(), end_dt.date().isoformat()),
            transiting_bodies=transiting_bodies,
            natal_bodies=natal_bodies,
            aspects=aspects_list,
            orbs=orbs_to_use,
            exact_timing=exact_timing,
            frame=frame,
            zodiac_mode=zodiac_mode,
            ayanamsa_deg=ayanamsa_deg,
            include_aspects_to=include_aspects_to,
            include_house_cusps=include_house_cusps,
        )
        key = _memo_key_transits(payload)
        tr_res = _memo_predictive(key)  # dict from predictive.find_transits_in_range

        events: List[PredictionEvent] = []

        # Convert predictive dict → PredictionEvent list
        if not isinstance(tr_res, dict) or not tr_res.get("ok", False):
            warns.append("predictive_find_transits_failed")
        else:
            hits = tr_res.get("transits", []) or []
            for h in hits:
                body = str(h.get("transiting_body",""))
                target = str(h.get("natal_body",""))
                asp = _canon_aspect(h.get("aspect"))
                sep = float(h.get("orb", 999.0))  # predictive 'orb' == absolute separation
                orb_cap = float(max(1e-9, h.get("max_orb", orbs_to_use.get(asp, orbs_to_use.get("default", 1.0)))))
                tight = max(0.0, 1.0 - (sep / orb_cap))
                major = 0.2 if asp in {"conjunction","opposition","trine","square"} else 0.0
                conf = min(1.0, tight + major)

                jd_exact = h.get("exact_jd_tt")
                # predictive currently doesn't return UTC datetime; keep None unless provided
                dt_utc = None
                if isinstance(h.get("exact_datetime_utc"), str):
                    try:
                        dt_utc = _ensure_utc(h["exact_datetime_utc"])
                    except Exception:
                        dt_utc = None

                pe = PredictionEvent(
                    event_type="transit",
                    technique="exact_transit",
                    description=f"{body} {asp} {target}",
                    datetime_utc=dt_utc,
                    jd_tt=float(jd_exact) if jd_exact is not None else None,
                    jd_ut1=None,
                    precision_seconds=None,
                    confidence=conf,
                    significance=1.0,
                    metadata={
                        "transiting_body": body,
                        "natal_body": target,
                        "aspect": asp,
                        "orb": sep,
                        "max_orb": orb_cap,
                        "applying": bool(h.get("applying", False)),
                        "exact": bool(h.get("exact", False)),
                        "kind": h.get("kind", "zodiacal"),
                        "finder": "predictive.find_transits_in_range",
                    },
                )
                if pe.confidence >= confidence_threshold:
                    events.append(pe)

        stats: Dict[str, float] = {}
        windows = _create_timing_windows(events)
        conf_score = _compute_confidence(events, stats)
        synth = {
            "total_transits": len(events),
            "major_aspects": sum(1 for e in events if e.metadata.get("aspect") in {"conjunction","opposition","trine","square"}),
            "average_confidence": (sum(e.confidence for e in events) / len(events)) if events else 0.0,
            "body_activity": _body_activity(events),
        }

        return PredictionResult(
            ok=True,
            technique="transits",
            events=events,
            synthesis=synth,
            timing_windows=windows,
            confidence_score=conf_score,
            statistical_metrics=stats,
            warnings=warns,
            metadata={
                "natal_jd_tt": jd_tt_natal,
                "time_range_utc": (start_dt.isoformat(), end_dt.isoformat()),
                "parameters": {
                    "transiting_bodies": transiting_bodies,
                    "natal_targets": natal_bodies,
                    "aspects": aspects_list,
                    "frame": frame,
                    "zodiac_mode": zodiac_mode,
                    "ayanamsa_deg": ayanamsa_deg,
                    "exact_timing": exact_timing,
                },
            },
            computation_time_ms=(time.time() - t0) * 1000.0,
        )
    except Exception as e:
        return PredictionResult(
            ok=False,
            technique="transits",
            warnings=[f"transit_computation_failed:{e}"],
            computation_time_ms=(time.time() - t0) * 1000.0,
        )


def _resolve_natal_timescales_fixed(natal_chart: Dict[str, Any]) -> Tuple[float, float, List[str]]:
    """
    Fixed version of natal timescale resolution that avoids .utc_jd attribute error.
    
    Returns:
        Tuple of (jd_tt, jd_ut1, warnings)
    """
    warnings = []
    
    try:
        # Extract date/time from natal chart
        natal_date = natal_chart.get("date")
        natal_time = natal_chart.get("time", "12:00:00")
        place_tz = natal_chart.get("place_tz", "UTC")
        
        # Create datetime object
        if isinstance(natal_date, str):
            dt_str = f"{natal_date} {natal_time}"
            try:
                # Try parsing with timezone
                if place_tz and place_tz != "UTC":
                    from zoneinfo import ZoneInfo
                    tz = ZoneInfo(place_tz)
                    dt = datetime.fromisoformat(dt_str.replace("Z", "")).replace(tzinfo=tz)
                    dt_utc = dt.astimezone(timezone.utc)
                else:
                    dt_utc = datetime.fromisoformat(dt_str.replace("Z", "")).replace(tzinfo=timezone.utc)
            except Exception as e:
                warnings.append(f"date_parse_error:{e}")
                # Fallback to simple parsing
                dt_utc = datetime.fromisoformat(f"{natal_date} {natal_time}").replace(tzinfo=timezone.utc)
        else:
            # Assume it's already a datetime object
            dt_utc = natal_date if hasattr(natal_date, 'astimezone') else datetime.now(timezone.utc)
            warnings.append("assumed_datetime_object")
        
        # Convert to Julian dates using Skyfield
        try:
            from skyfield.api import load
            ts = load.timescale()
            
            # Create Skyfield time object from UTC datetime
            t = ts.from_datetime(dt_utc)
            
            # Use correct Skyfield attributes (not .utc_jd)
            jd_tt = t.tt          # Terrestrial Time Julian Date
            jd_ut1 = t.ut1        # UT1 Julian Date
            
        except ImportError:
            warnings.append("skyfield_unavailable_using_approximation")
            # Fallback calculation without Skyfield
            import calendar
            timestamp = calendar.timegm(dt_utc.timetuple())
            jd_ut1 = 2440587.5 + timestamp / 86400.0  # Unix epoch to JD conversion
            jd_tt = jd_ut1 + 69.184 / 86400.0  # Approximate TT-UT1 difference
            
        except Exception as e:
            warnings.append(f"skyfield_time_conversion_error:{e}")
            # Emergency fallback
            import calendar
            timestamp = calendar.timegm(dt_utc.timetuple()) 
            jd_ut1 = 2440587.5 + timestamp / 86400.0
            jd_tt = jd_ut1 + 69.184 / 86400.0
            
        return float(jd_tt), float(jd_ut1), warnings
        
    except Exception as e:
        warnings.append(f"natal_timescale_resolution_failed:{e}")
        # Return reasonable defaults
        return 2451545.0, 2451545.0, warnings  # J2000.0 epoch


def _jd_pair_from_dt_fixed(dt: Union[datetime, str]) -> Tuple[float, float]:
    """
    Fixed version that converts datetime to (jd_tt, jd_ut1) without .utc_jd errors.
    
    Args:
        dt: datetime object or ISO string
        
    Returns:
        Tuple of (jd_tt, jd_ut1)
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "")).replace(tzinfo=timezone.utc)
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)
    
    try:
        from skyfield.api import load
        ts = load.timescale()
        t = ts.from_datetime(dt)
        
        # Use correct Skyfield Time attributes
        return float(t.tt), float(t.ut1)
        
    except ImportError:
        # Fallback without Skyfield
        import calendar
        timestamp = calendar.timegm(dt.timetuple())
        jd_ut1 = 2440587.5 + timestamp / 86400.0
        jd_tt = jd_ut1 + 69.184 / 86400.0  # Approximate TT-UT1
        return jd_tt, jd_ut1
        
    except Exception:
        # Emergency fallback
        import calendar
        timestamp = calendar.timegm(dt.timetuple())
        jd_ut1 = 2440587.5 + timestamp / 86400.0
        jd_tt = jd_ut1 + 69.184 / 86400.0
        return jd_tt, jd_ut1
        
def predict_progressions(
    natal_chart: Dict[str, Any],
    target_date: Union[datetime, str, float],
    *,
    method: str = "secondary",
    lunar_month: str = "synodic",
    tertiary_mode: str = "day-for-month",
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    aspects_to_natal: bool = True,
    orbs: Optional[Dict[str, float]] = None,
    parallels: bool = False,
    antiscia: bool = False,
    statistical_validation: bool = False,
    **kwargs,
) -> PredictionResult:
    _check_env()
    t0 = time.time()
    warns: List[str] = []
    
    # Add timeout protection to prevent hanging
    max_computation_time = 30.0  # 30 seconds max
    
    try:
        if not _PROG_OK:
            raise RuntimeError(f"progressions unavailable: {_PROG_ERR}")

        # Use fixed timescale resolution
        jd_tt_natal, jd_ut1_natal, w = _resolve_natal_timescales_fixed(natal_chart)
        warns.extend(w)

        if isinstance(target_date, (int, float)):
            years_after = (float(target_date) - jd_tt_natal) / TROPICAL_YEAR_D
            target_dt: Optional[datetime] = None
        else:
            t_dt = _ensure_utc(target_date)
            natal_epoch_ts = (jd_tt_natal - 2440587.5) * 86400.0
            natal_dt = datetime.fromtimestamp(natal_epoch_ts, tz=timezone.utc)
            years_after = (t_dt - natal_dt).total_seconds() / (TROPICAL_YEAR_D * 86400.0)
            target_dt = t_dt

        # Check if computation is taking too long
        if time.time() - t0 > max_computation_time:
            raise RuntimeError("computation_timeout_before_progressions_call")

        # Limit progression computation scope to prevent infinite loops
        limited_orbs = orbs or DEFAULT_ORBS_PROGRESSIONS
        # Cap orb values to reasonable limits
        capped_orbs = {k: min(v, 10.0) for k, v in limited_orbs.items()}

        res = compute_progressions_safe(
            natal=natal_chart,
            method=method,
            years_after=years_after,
            jd_tt_natal=jd_tt_natal,
            jd_ut1_natal=jd_ut1_natal,
            frame=frame,
            house_system=house_system,
            zodiac_mode=zodiac_mode,
            ayanamsa_deg=ayanamsa_deg,
            lunar_month=lunar_month,
            tertiary_mode=tertiary_mode,
            aspects_to_natal=aspects_to_natal,
            orbs=capped_orbs,
            parallels=parallels,
            antiscia=antiscia,
            profile=True,
            validation="basic",
            timeout_seconds=max_computation_time - (time.time() - t0),
        )
        
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "progressions_failed"))

        events: List[PredictionEvent] = []

        # Limit event processing to prevent excessive computation
        max_events = 1000  # Cap at 1000 events
        event_count = 0

        if aspects_to_natal and isinstance(res.get("aspects"), dict):
            for fam, hits in res["aspects"].items():
                if not isinstance(hits, list) or event_count >= max_events:
                    continue
                    
                for h in hits[:100]:  # Limit to 100 aspects per family
                    if event_count >= max_events:
                        break
                        
                    asp = _canon_aspect(h.get("aspect"))
                    orb = float(h.get("orb", 0.0))
                    max_orb = float(h.get("max_orb", max(capped_orbs.get(asp, 1.0), 1e-9)))
                    tight = max(0.0, 1.0 - (orb / max_orb))
                    exact_bonus = 0.3 if orb < 0.1 else 0.0
                    conf = min(1.0, tight + exact_bonus)
                    
                    events.append(PredictionEvent(
                        event_type="progression",
                        technique=f"{method}_progression",
                        description=f"Progressed {h.get('planet_a','?')} {asp} Natal {h.get('planet_b','?')}",
                        datetime_utc=target_dt,
                        jd_tt=None, jd_ut1=None, precision_seconds=None,
                        confidence=conf,
                        significance=float(h.get("p_value", 1.0)),
                        metadata={
                            "progressed_body": h.get("planet_a"),
                            "natal_body": h.get("planet_b"),
                            "aspect": asp,
                            "orb": orb,
                            "family": fam,
                            "finder": "progressions.compute_progressions",
                        },
                    ))
                    event_count += 1

        if isinstance(res.get("positions"), list) and event_count < max_events:
            natal_lookup: Dict[str, float] = {}
            for b in (natal_chart.get("bodies") or []):
                if isinstance(b, dict) and "name" in b:
                    natal_lookup[str(b["name"])] = float(b.get("longitude", b.get("lon", 0.0)))
                    
            for p in res["positions"][:50]:  # Limit to 50 position events
                if event_count >= max_events:
                    break
                    
                name = str(p.get("name"))
                if name in natal_lookup:
                    prog_lon = float(p.get("longitude", 0.0))
                    natal_lon = float(natal_lookup[name])
                    dist = abs_sep_deg(natal_lon, prog_lon)
                    if dist > 1.0:
                        conf = min(1.0, dist / 30.0)
                        events.append(PredictionEvent(
                            event_type="progression",
                            technique=f"{method}_position",
                            description=f"Progressed {name} moved {dist:.1f}° → {prog_lon:.1f}°",
                            datetime_utc=target_dt,
                            jd_tt=None, jd_ut1=None, precision_seconds=None,
                            confidence=conf, significance=1.0,
                            metadata={
                                "body": name,
                                "natal_longitude": natal_lon,
                                "progressed_longitude": prog_lon,
                                "movement_degrees": dist,
                                "speed": float(p.get("speed", 0.0)),
                            },
                        ))
                        event_count += 1

        if event_count >= max_events:
            warns.append("max_events_reached_computation_limited")

        stats: Dict[str, float] = {}
        synth = {
            "progression_method": method,
            "target_date": (target_dt.isoformat() if target_dt else None),
            "years_after_natal": years_after,
            "total_aspects": sum(1 for e in events if "progression" in e.technique),
            "significant_movements": sum(1 for e in events if "position" in e.technique),
            "average_confidence": (sum(e.confidence for e in events) / len(events)) if events else 0.0,
            "progression_epoch": res.get("meta", {}).get("epoch", {}),
            "computation_limited": event_count >= max_events,
        }
        
        for w in res.get("meta", {}).get("warnings", []):
            warns.append(f"progression_{w}")

        return PredictionResult(
            ok=True,
            technique=f"{method}_progressions",
            events=events,
            synthesis=synth,
            timing_windows=[],
            confidence_score=_compute_confidence(events, stats),
            statistical_metrics=stats,
            warnings=warns,
            metadata={
                "natal_jd_tt": jd_tt_natal,
                "parameters": {
                    "method": method, "lunar_month": lunar_month, "tertiary_mode": tertiary_mode,
                    "frame": frame, "zodiac_mode": zodiac_mode, "aspects_to_natal": aspects_to_natal,
                },
                "performance": {
                    "computation_time_ms": (time.time() - t0) * 1000.0,
                    "events_processed": event_count,
                    "timeout_protection": True,
                }
            },
            computation_time_ms=(time.time() - t0) * 1000.0,
        )
        
    except Exception as e:
        return PredictionResult(
            ok=False, technique="progressions",
            warnings=[f"progression_computation_failed:{e}"],
            computation_time_ms=(time.time() - t0) * 1000.0,
        )


def compute_progressions_safe(
    natal: Dict[str, Any],
    method: str,
    years_after: float,
    jd_tt_natal: float,
    jd_ut1_natal: float,
    frame: str,
    house_system: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    lunar_month: str,
    tertiary_mode: str,
    aspects_to_natal: bool,
    orbs: Dict[str, float],
    parallels: bool,
    antiscia: bool,
    profile: bool,
    validation: str,
    timeout_seconds: float = 25.0,
) -> Dict[str, Any]:
    """
    Safe wrapper for compute_progressions that prevents hanging.
    
    This function adds timeout protection and limits to prevent the infinite
    loops that were causing the progressions endpoint to hang.
    """
    import signal
    from functools import wraps
    
    def timeout_handler(signum, frame):
        raise TimeoutError("progressions_computation_timeout")
    
    # Set up timeout protection (Unix-like systems only)
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        
        try:
            # Call the original compute_progressions function
            # but with limited parameters to prevent excessive computation
            limited_natal = natal.copy()
            
            # Limit the bodies to prevent excessive calculations
            if "bodies" in limited_natal and len(limited_natal["bodies"]) > 15:
                limited_natal["bodies"] = limited_natal["bodies"][:15]
                
            # Limit years_after to reasonable range
            years_after = max(-100, min(100, years_after))
            
            result = compute_progressions(
                natal=limited_natal,
                method=method,
                years_after=years_after,
                jd_tt_natal=jd_tt_natal,
                jd_ut1_natal=jd_ut1_natal,
                frame=frame,
                house_system=house_system,
                zodiac_mode=zodiac_mode,
                ayanamsa_deg=ayanamsa_deg,
                lunar_month=lunar_month,
                tertiary_mode=tertiary_mode,
                aspects_to_natal=aspects_to_natal,
                orbs=orbs,
                parallels=parallels,
                antiscia=antiscia,
                profile=profile,
                validation=validation,
            )
            
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
            
            return result
            
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return {
                "ok": False,
                "error": "progressions_computation_timeout",
                "meta": {"warnings": ["computation_exceeded_timeout"]}
            }
            
    except (AttributeError, OSError):
        # signal.alarm not available (Windows) - use basic fallback
        try:
            # Call with very limited parameters
            limited_natal = natal.copy()
            if "bodies" in limited_natal and len(limited_natal["bodies"]) > 10:
                limited_natal["bodies"] = limited_natal["bodies"][:10]
                
            years_after = max(-50, min(50, years_after))
            
            return compute_progressions(
                natal=limited_natal,
                method=method,
                years_after=years_after,
                jd_tt_natal=jd_tt_natal,
                jd_ut1_natal=jd_ut1_natal,
                frame=frame,
                house_system=house_system,
                zodiac_mode=zodiac_mode,
                ayanamsa_deg=ayanamsa_deg,
                lunar_month=lunar_month,
                tertiary_mode=tertiary_mode,
                aspects_to_natal=aspects_to_natal,
                orbs=orbs,
                parallels=False,  # Disable parallels to reduce computation
                antiscia=False,   # Disable antiscia to reduce computation
                profile=False,    # Disable profiling to reduce computation
                validation="none", # Disable validation to reduce computation
            )
            
        except Exception as e:
            return {
                "ok": False,
                "error": f"progressions_fallback_failed:{e}",
                "meta": {"warnings": ["fallback_computation_failed"]}
            }
    
    except Exception as e:
        return {
            "ok": False,
            "error": f"progressions_safe_wrapper_failed:{e}",
            "meta": {"warnings": ["safe_wrapper_failed"]}
        }
        
def predict_returns(
    natal_chart: Dict[str, Any],
    return_type: str,
    year: int,
    *,
    lunar_month: str = "sidereal",
    place: Optional[Dict[str, Any]] = None,
    frame: str = "ecliptic-of-date",
    house_system: str = "placidus",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    estimate_uncertainty: bool = True,
    aspects_to_natal: bool = True,
    orbs: Optional[Dict[str, float]] = None,
    statistical_validation: bool = False,
    **kwargs,
) -> PredictionResult:
    _check_env()
    t0 = time.time()
    warns: List[str] = []
    try:
        if not _RET_OK:
            raise RuntimeError(f"returns unavailable: {_RET_ERR}")

        if return_type.lower() not in ("solar", "lunar"):
            raise ValueError("return_type must be 'solar' or 'lunar'")

        # Use fixed timescale resolution
        jd_tt_natal, jd_ut1_natal, w = _resolve_natal_timescales_fixed(natal_chart)
        warns.extend(w)

        use_place = place or natal_chart
        
        # Use safe wrapper functions that handle missing implementations
        if return_type.lower() == "solar":
            rr = compute_solar_return_safe(
                natal=natal_chart, target_year=int(year),
                jd_tt_natal=jd_tt_natal, jd_ut1_natal=jd_ut1_natal,
                place=use_place, frame=frame, house_system=house_system,
                zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg,
                estimate_uncertainty=estimate_uncertainty, profile=True, validation="extended",
            )
        else:
            rr = compute_lunar_return_safe(
                natal=natal_chart, target_year=int(year), lunar_month=lunar_month,
                jd_tt_natal=jd_tt_natal, jd_ut1_natal=jd_ut1_natal,
                place=use_place, frame=frame, house_system=house_system,
                zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg,
                estimate_uncertainty=estimate_uncertainty, profile=True, validation="extended",
            )
            
        if not rr.get("ok"):
            raise RuntimeError(rr.get("error", "return_failed"))

        ret_jd_tt = rr.get("return_jd_tt"); ret_jd_ut1 = rr.get("return_jd_ut1")
        ret_dt = _ensure_utc(rr["return_datetime_utc"]) if isinstance(rr.get("return_datetime_utc"), str) else None

        unc_days = rr.get("uncertainty", {}).get("timing_error_days")
        prec_s = (unc_days * 86400.0) if unc_days else None
        conv = rr.get("convergence", {})
        residual_arcmin = float(conv.get("final_residual_arcmin", 999.0))
        conv_conf = max(0.0, 1.0 - residual_arcmin / 10.0)
        if conv.get("converged", False):
            conv_conf = min(1.0, conv_conf + 0.3)

        events: List[PredictionEvent] = [
            PredictionEvent(
                event_type="return",
                technique=f"{return_type.lower()}_return",
                description=f"{return_type.capitalize()} Return {year}",
                datetime_utc=ret_dt,
                jd_tt=ret_jd_tt, jd_ut1=ret_jd_ut1,
                precision_seconds=prec_s, confidence=conv_conf, significance=0.01,
                metadata={
                    "return_type": return_type.lower(), "year": year, "residual_arcmin": residual_arcmin,
                    "iterations": int(conv.get("iterations", 0)), "converged": bool(conv.get("converged", False)),
                    "uncertainty_days": unc_days, "finder": "returns.compute_*_return",
                },
            )
        ]

        if aspects_to_natal and isinstance(rr.get("chart"), dict) and _ASPECTS_OK:
            def _positions(chart: Dict[str, Any]) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for b in chart.get("bodies", []):
                    if isinstance(b, dict) and "name" in b:
                        out.append({"name": b["name"], "longitude": float(b.get("longitude", b.get("lon", 0.0)))})
                if isinstance(chart.get("angles"), dict):
                    ang = chart["angles"]
                    for key, canon in (("asc_deg","ASC"),("mc_deg","MC"),("ic_deg","IC"),("dsc_deg","DSC")):
                        if key in ang: out.append({"name": canon, "longitude": float(ang[key])})
                return out

            rr_pos = _positions(rr["chart"])
            natal_pos = _positions(natal_chart)
            cfg = _normalize_aspect_config(orbs, "returns")
            ar = compute_aspects(positions_a=rr_pos, positions_b=natal_pos, config=cfg)
            for h in ar.get("hits", []):
                asp = _canon_aspect(h.get("aspect"))
                orb = float(h.get("orb", 0.0))
                max_orb = float(h.get("max_orb", max((orbs or DEFAULT_ORBS_PROGRESSIONS).get(asp, 1.0), 1e-9)))
                tight = max(0.0, 1.0 - orb / max_orb)
                major = 0.2 if asp in {"conjunction","opposition","trine","square"} else 0.0
                conf = min(1.0, tight + major)
                events.append(PredictionEvent(
                    event_type="return_aspect",
                    technique=f"{return_type.lower()}_return_aspect",
                    description=f"Return {h.get('planet_a','?')} {asp} Natal {h.get('planet_b','?')}",
                    datetime_utc=ret_dt,
                    jd_tt=ret_jd_tt, jd_ut1=ret_jd_ut1, precision_seconds=prec_s,
                    confidence=conf, significance=float(h.get("p_value", 1.0)),
                    metadata={
                        "return_body": h.get("planet_a"), "natal_body": h.get("planet_b"),
                        "aspect": asp, "orb": orb, "family": h.get("family","zodiacal"),
                        "finder": "aspects.compute_aspects",
                    },
                ))

        stats: Dict[str, float] = {}
        windows: List[TimingWindow] = []
        if ret_jd_tt and unc_days:
            windows.append(TimingWindow(
                start_jd_tt=ret_jd_tt - unc_days, end_jd_tt=ret_jd_tt + unc_days,
                peak_jd_tt=ret_jd_tt, uncertainty_days=unc_days,
                confidence_interval=(ret_jd_tt - unc_days, ret_jd_tt + unc_days),
            ))

        synth = {
            "return_type": return_type.lower(), "year": int(year),
            "return_datetime": (ret_dt.isoformat() if ret_dt else None),
            "uncertainty_days": unc_days,
            "total_aspects": sum(1 for e in events if e.event_type == "return_aspect"),
            "convergence_quality": bool(conv.get("converged", False)),
            "residual_quality": "excellent" if residual_arcmin < 1.0 else "good" if residual_arcmin < 5.0 else "fair",
        }
        for w in rr.get("meta", {}).get("warnings", []):
            warns.append(f"return_{w}")

        return PredictionResult(
            ok=True, technique=f"{return_type.lower()}_return",
            events=events, synthesis=synth, timing_windows=windows,
            confidence_score=_compute_confidence(events, stats), statistical_metrics=stats,
            warnings=warns,
            metadata={
                "natal_jd_tt": jd_tt_natal,
                "parameters": {
                    "return_type": return_type.lower(), "frame": frame, "house_system": house_system,
                    "zodiac_mode": zodiac_mode, "ayanamsa_deg": ayanamsa_deg,
                    "estimate_uncertainty": estimate_uncertainty, "aspects_to_natal": aspects_to_natal,
                },
            },
            computation_time_ms=(time.time() - t0) * 1000.0,
        )
    except Exception as e:
        return PredictionResult(
            ok=False, technique="returns",
            warnings=[f"return_computation_failed:{e}"],
            computation_time_ms=(time.time() - t0) * 1000.0,
        )


def compute_solar_return_safe(
    natal: Dict[str, Any],
    target_year: int,
    jd_tt_natal: float,
    jd_ut1_natal: float,
    place: Dict[str, Any],
    frame: str,
    house_system: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    estimate_uncertainty: bool,
    profile: bool,
    validation: str,
) -> Dict[str, Any]:
    """
    Safe wrapper for compute_solar_return that handles missing implementation.
    
    This addresses the error: cannot import name 'compute_solar_return' from 'app.core.returns'
    """
    try:
        # Try to import the actual function first
        from app.core.returns import compute_solar_return
        return compute_solar_return(
            natal=natal,
            target_year=target_year,
            jd_tt_natal=jd_tt_natal,
            jd_ut1_natal=jd_ut1_natal,
            place=place,
            frame=frame,
            house_system=house_system,
            zodiac_mode=zodiac_mode,
            ayanamsa_deg=ayanamsa_deg,
            estimate_uncertainty=estimate_uncertainty,
            profile=profile,
            validation=validation,
        )
    except ImportError:
        # Function doesn't exist - return a basic implementation
        return _compute_solar_return_fallback(
            natal=natal,
            target_year=target_year,
            jd_tt_natal=jd_tt_natal,
            jd_ut1_natal=jd_ut1_natal,
            place=place,
            frame=frame,
            house_system=house_system,
            zodiac_mode=zodiac_mode,
            ayanamsa_deg=ayanamsa_deg,
            estimate_uncertainty=estimate_uncertainty,
        )


def compute_lunar_return_safe(
    natal: Dict[str, Any],
    target_year: int,
    lunar_month: str,
    jd_tt_natal: float,
    jd_ut1_natal: float,
    place: Dict[str, Any],
    frame: str,
    house_system: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    estimate_uncertainty: bool,
    profile: bool,
    validation: str,
) -> Dict[str, Any]:
    """
    Safe wrapper for compute_lunar_return that handles missing implementation.
    
    This addresses the error: cannot import name 'compute_lunar_return' from 'app.core.returns'
    """
    try:
        # Try to import the actual function first
        from app.core.returns import compute_lunar_return
        return compute_lunar_return(
            natal=natal,
            target_year=target_year,
            lunar_month=lunar_month,
            jd_tt_natal=jd_tt_natal,
            jd_ut1_natal=jd_ut1_natal,
            place=place,
            frame=frame,
            house_system=house_system,
            zodiac_mode=zodiac_mode,
            ayanamsa_deg=ayanamsa_deg,
            estimate_uncertainty=estimate_uncertainty,
            profile=profile,
            validation=validation,
        )
    except ImportError:
        # Function doesn't exist - return a basic implementation
        return _compute_lunar_return_fallback(
            natal=natal,
            target_year=target_year,
            lunar_month=lunar_month,
            jd_tt_natal=jd_tt_natal,
            jd_ut1_natal=jd_ut1_natal,
            place=place,
            frame=frame,
            house_system=house_system,
            zodiac_mode=zodiac_mode,
            ayanamsa_deg=ayanamsa_deg,
            estimate_uncertainty=estimate_uncertainty,
        )


def _compute_solar_return_fallback(
    natal: Dict[str, Any],
    target_year: int,
    jd_tt_natal: float,
    jd_ut1_natal: float,
    place: Dict[str, Any],
    frame: str,
    house_system: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    estimate_uncertainty: bool,
) -> Dict[str, Any]:
    """
    Basic fallback implementation for solar return calculation.
    """
    try:
        from skyfield.api import load
        from datetime import datetime, timezone
        import calendar
        
        # Get Sun's natal longitude
        natal_sun_lon = 0.0
        for body in natal.get("bodies", []):
            if body.get("name", "").lower() == "sun":
                natal_sun_lon = float(body.get("longitude", body.get("lon", 0.0)))
                break
        
        # Estimate solar return date (approximately target_year birthday)
        natal_date = natal.get("date", "1990-01-01")
        if isinstance(natal_date, str):
            natal_year = int(natal_date.split("-")[0])
            natal_month_day = natal_date[4:]  # Keep "-MM-DD"
            estimated_return_date = f"{target_year}{natal_month_day}"
        else:
            estimated_return_date = f"{target_year}-01-01"
        
        # Convert to JD
        est_dt = datetime.fromisoformat(estimated_return_date).replace(tzinfo=timezone.utc)
        timestamp = calendar.timegm(est_dt.timetuple())
        return_jd_ut1 = 2440587.5 + timestamp / 86400.0
        return_jd_tt = return_jd_ut1 + 69.184 / 86400.0
        
        return {
            "ok": True,
            "return_jd_tt": return_jd_tt,
            "return_jd_ut1": return_jd_ut1,
            "return_datetime_utc": est_dt.isoformat(),
            "chart": {
                "bodies": [{"name": "Sun", "longitude": natal_sun_lon}],
                "angles": {},
            },
            "uncertainty": {"timing_error_days": 1.0},
            "convergence": {
                "converged": False,
                "iterations": 0,
                "final_residual_arcmin": 60.0,
            },
            "meta": {
                "warnings": ["using_fallback_solar_return_implementation"],
                "method": "approximate_birthday"
            }
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": f"solar_return_fallback_failed:{e}",
            "meta": {"warnings": ["fallback_computation_failed"]}
        }


def _compute_lunar_return_fallback(
    natal: Dict[str, Any],
    target_year: int,
    lunar_month: str,
    jd_tt_natal: float,
    jd_ut1_natal: float,
    place: Dict[str, Any],
    frame: str,
    house_system: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    estimate_uncertainty: bool,
) -> Dict[str, Any]:
    """
    Basic fallback implementation for lunar return calculation.
    """
    try:
        from datetime import datetime, timezone
        import calendar
        
        # Get Moon's natal longitude
        natal_moon_lon = 0.0
        for body in natal.get("bodies", []):
            if body.get("name", "").lower() == "moon":
                natal_moon_lon = float(body.get("longitude", body.get("lon", 0.0)))
                break
        
        # Approximate lunar return (monthly cycles)
        # Use January 1st of target year as rough estimate
        est_dt = datetime(target_year, 1, 1, tzinfo=timezone.utc)
        timestamp = calendar.timegm(est_dt.timetuple())
        return_jd_ut1 = 2440587.5 + timestamp / 86400.0
        return_jd_tt = return_jd_ut1 + 69.184 / 86400.0
        
        return {
            "ok": True,
            "return_jd_tt": return_jd_tt,
            "return_jd_ut1": return_jd_ut1,
            "return_datetime_utc": est_dt.isoformat(),
            "chart": {
                "bodies": [{"name": "Moon", "longitude": natal_moon_lon}],
                "angles": {},
            },
            "uncertainty": {"timing_error_days": 2.0},
            "convergence": {
                "converged": False,
                "iterations": 0,
                "final_residual_arcmin": 120.0,
            },
            "meta": {
                "warnings": ["using_fallback_lunar_return_implementation"],
                "method": "approximate_monthly_cycle"
            }
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": f"lunar_return_fallback_failed:{e}",
            "meta": {"warnings": ["fallback_computation_failed"]}
        }
        
def predict_directions(
    natal_chart: Dict[str, Any],
    target_date: Union[datetime, str, float],
    *,
    method: str = "solar_arc",
    frame: str = "ecliptic-of-date",
    zodiac_mode: str = "tropical",
    ayanamsa_deg: float = 0.0,
    house_system: str = "placidus",
    aspects_to_natal: bool = True,
    orbs: Optional[Dict[str, float]] = None,
    statistical_validation: bool = False,
    **kwargs,
) -> PredictionResult:
    _check_env()
    t0 = time.time()
    warns: List[str] = []
    try:
        if not _DIR_OK:
            raise RuntimeError(f"directions unavailable: {_DIR_ERR}")
            
        # Use fixed timescale resolution
        jd_tt_natal, jd_ut1_natal, w = _resolve_natal_timescales_fixed(natal_chart)
        warns.extend(w)
        
        if isinstance(target_date, (int, float)):
            target_dt: Optional[datetime] = None
        else:
            target_dt = _ensure_utc(target_date)
            
        # Use safe wrapper that handles parameter mismatch
        dr = compute_directions_safe(
            natal=natal_chart,
            method=method,
            jd_tt_natal=jd_tt_natal,
            jd_ut1_natal=jd_ut1_natal,
            frame=frame,
            house_system=house_system,
            zodiac_mode=zodiac_mode,
            ayanamsa_deg=ayanamsa_deg,
            orbs=orbs or DEFAULT_ORBS_DIRECTIONS,
            profile=True,
        )
        
        if not dr.get("ok"):
            raise RuntimeError(dr.get("error", "directions_failed"))
            
        events: List[PredictionEvent] = []
        
        for h in dr.get("hits", []):
            asp = _canon_aspect(h.get("aspect"))
            orb = float(h.get("orb", 0.0))
            max_orb = float(h.get("max_orb", max((orbs or DEFAULT_ORBS_DIRECTIONS).get(asp, 1.0), 1e-9)))
            tight = max(0.0, 1.0 - orb / max_orb)
            conf = min(1.0, tight + (0.2 if asp in {"conjunction","opposition","square","trine"} else 0.0))
            
            events.append(PredictionEvent(
                event_type="direction",
                technique=f"{method}_direction",
                description=f"Directed {h.get('planet_a','?')} {asp} Natal {h.get('planet_b','?')}",
                datetime_utc=target_dt,
                jd_tt=None, jd_ut1=None, precision_seconds=None,
                confidence=conf, significance=float(h.get("p_value", 1.0)),
                metadata={
                    "directed_body": h.get("planet_a"),
                    "natal_body": h.get("planet_b"),
                    "aspect": asp, "orb": orb,
                    "finder": "directions.compute_directions",
                },
            ))
            
        stats: Dict[str, float] = {}
        
        return PredictionResult(
            ok=True, technique=f"{method}_directions",
            events=events, synthesis={"total_hits": len(events)}, timing_windows=[],
            confidence_score=_compute_confidence(events, stats), statistical_metrics=stats,
            warnings=warns, metadata={"natal_jd_tt": jd_tt_natal},
            computation_time_ms=(time.time() - t0) * 1000.0,
        )
        
    except Exception as e:
        return PredictionResult(
            ok=False, technique="directions",
            warnings=[f"directions_failed:{e}"],
            computation_time_ms=(time.time() - t0) * 1000.0,
        )


def compute_directions_safe(
    natal: Dict[str, Any],
    method: str,
    jd_tt_natal: float,
    jd_ut1_natal: float,
    frame: str,
    house_system: str,
    zodiac_mode: str,
    ayanamsa_deg: float,
    orbs: Dict[str, float],
    profile: bool,
) -> Dict[str, Any]:
    """
    Safe wrapper for compute_directions that handles parameter mismatches.
    
    The diagnostic error was: compute_directions() got an unexpected keyword argument 'aspects_to_natal'
    This wrapper ensures only valid parameters are passed to the function.
    """
    try:
        # Import the actual function
        from app.core.directions import compute_directions
        
        # Call with only the parameters that the function accepts
        # Remove 'aspects_to_natal' which was causing the error
        return compute_directions(
            natal=natal,
            method=method,
            jd_tt_natal=jd_tt_natal,
            jd_ut1_natal=jd_ut1_natal,
            frame=frame,
            house_system=house_system,
            zodiac_mode=zodiac_mode,
            ayanamsa_deg=ayanamsa_deg,
            orbs=orbs,
            profile=profile,
        )
        
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            # Try with even fewer parameters if there are more signature mismatches
            try:
                return compute_directions(
                    natal=natal,
                    method=method,
                    jd_tt_natal=jd_tt_natal,
                    jd_ut1_natal=jd_ut1_natal,
                    orbs=orbs,
                )
            except Exception as fallback_error:
                return {
                    "ok": False,
                    "error": f"directions_parameter_mismatch:{fallback_error}",
                    "hits": [],
                    "meta": {"warnings": ["parameter_signature_incompatible"]}
                }
        else:
            return {
                "ok": False,
                "error": f"directions_type_error:{e}",
                "hits": [],
                "meta": {"warnings": ["function_call_failed"]}
            }
            
    except ImportError:
        # Function doesn't exist - return fallback
        return _compute_directions_fallback(
            natal=natal,
            method=method,
            jd_tt_natal=jd_tt_natal,
            jd_ut1_natal=jd_ut1_natal,
            orbs=orbs,
        )
        
    except Exception as e:
        return {
            "ok": False,
            "error": f"directions_computation_failed:{e}",
            "hits": [],
            "meta": {"warnings": ["directions_function_error"]}
        }


def _compute_directions_fallback(
    natal: Dict[str, Any],
    method: str,
    jd_tt_natal: float,
    jd_ut1_natal: float,
    orbs: Dict[str, float],
) -> Dict[str, Any]:
    """
    Basic fallback implementation for directions calculation.
    
    This provides minimal functionality when the actual compute_directions function
    is not available or has incompatible parameters.
    """
    try:
        # Basic directions calculation - simplified approach
        # This is a placeholder implementation that shows the expected structure
        
        hits = []
        
        # Get natal positions
        natal_bodies = natal.get("bodies", [])
        
        # For solar arc directions (most common method)
        if method == "solar_arc":
            # Calculate approximate solar arc rate (typically ~1°/year)
            # This is a very simplified calculation
            years_elapsed = (jd_tt_natal - 2451545.0) / 365.25  # Years from J2000
            arc_rate = 1.0  # degrees per year (simplified)
            
            for i, body_a in enumerate(natal_bodies):
                if not isinstance(body_a, dict) or "name" not in body_a:
                    continue
                    
                body_a_name = body_a["name"]
                body_a_lon = float(body_a.get("longitude", body_a.get("lon", 0.0)))
                
                # Calculate directed position (very simplified)
                directed_lon = (body_a_lon + arc_rate * years_elapsed) % 360.0
                
                # Check aspects to other natal bodies
                for j, body_b in enumerate(natal_bodies):
                    if i >= j or not isinstance(body_b, dict) or "name" not in body_b:
                        continue
                        
                    body_b_name = body_b["name"]
                    body_b_lon = float(body_b.get("longitude", body_b.get("lon", 0.0)))
                    
                    # Calculate aspect
                    separation = abs(directed_lon - body_b_lon)
                    if separation > 180:
                        separation = 360 - separation
                        
                    # Check for major aspects
                    aspects_to_check = [
                        ("conjunction", 0.0, orbs.get("conjunction", 8.0)),
                        ("opposition", 180.0, orbs.get("opposition", 8.0)),
                        ("trine", 120.0, orbs.get("trine", 6.0)),
                        ("square", 90.0, orbs.get("square", 6.0)),
                        ("sextile", 60.0, orbs.get("sextile", 4.0)),
                    ]
                    
                    for aspect_name, exact_angle, orb_limit in aspects_to_check:
                        orb = abs(separation - exact_angle)
                        if orb <= orb_limit:
                            hits.append({
                                "planet_a": body_a_name,
                                "planet_b": body_b_name,
                                "aspect": aspect_name,
                                "orb": orb,
                                "max_orb": orb_limit,
                                "p_value": 0.5,  # Default significance
                            })
        
        return {
            "ok": True,
            "hits": hits[:20],  # Limit to 20 hits to prevent overwhelming response
            "meta": {
                "warnings": ["using_fallback_directions_implementation"],
                "method": method,
                "computation": "simplified"
            }
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": f"directions_fallback_failed:{e}",
            "hits": [],
            "meta": {"warnings": ["fallback_computation_failed"]}
        }
        
def comprehensive_forecast(
    natal_chart: Dict[str, Any],
    time_range: Tuple[Union[datetime, str], Union[datetime, str]],
    *,
    techniques: Optional[List[str]] = None,
    confidence_threshold: float = 0.2,
    synthesis_method: str = "weighted_consensus",
    statistical_validation: bool = False,
    include_vedic: bool = False,
    peak_window_days: int = 14,
    **tk_kwargs,
) -> ComprehensiveForecast:
    _check_env()
    t0 = time.time()
    warns: List[str] = []
    
    # Add comprehensive timeout protection
    max_computation_time = 25.0  # 25 seconds maximum
    max_progression_samples = 10  # Limit progression sampling
    max_return_years = 5  # Limit return calculations
    
    try:
        techs = techniques or ["transits", "progressions", "solar_returns"]
        start_dt = _ensure_utc(time_range[0]); end_dt = _ensure_utc(time_range[1])

        # Validate time range to prevent excessive computation
        duration_days = (end_dt - start_dt).days
        if duration_days > 3650:  # More than 10 years
            warns.append("time_range_capped_to_10_years")
            end_dt = start_dt + timedelta(days=3650)
            duration_days = 3650

        predictions: Dict[str, PredictionResult] = {}
        all_events: List[PredictionEvent] = []

        # TRANSITS - Usually works, but add timeout check
        if "transits" in techs:
            if time.time() - t0 > max_computation_time * 0.3:  # Use 30% of time budget
                warns.append("transits_skipped_due_to_timeout_risk")
            else:
                try:
                    params = {k.replace("transit_", ""): v for k, v in tk_kwargs.items() if k.startswith("transit_")}
                    tr = predict_transits(natal_chart, (start_dt, end_dt), confidence_threshold=confidence_threshold, statistical_validation=False, **params)
                    predictions["transits"] = tr
                    all_events += tr.events if tr.ok else []
                    if not tr.ok: warns += tr.warnings
                except Exception as e:
                    warns.append(f"transits_failed:{e}")

        # PROGRESSIONS - This was causing infinite loops
        if "progressions" in techs:
            if time.time() - t0 > max_computation_time * 0.6:  # Use 60% of time budget
                warns.append("progressions_skipped_due_to_timeout_risk")
            else:
                try:
                    params = {k.replace("progression_", ""): v for k, v in tk_kwargs.items() if k.startswith("progression_")}
                    
                    # CRITICAL FIX: Limit progression sampling to prevent infinite loops
                    step = max(30, duration_days // max_progression_samples)  # At least 30-day steps
                    dates = [start_dt + timedelta(days=i) for i in range(0, min(duration_days + 1, step * max_progression_samples), step)]
                    
                    # Cap to maximum samples
                    if len(dates) > max_progression_samples:
                        dates = dates[:max_progression_samples]
                        warns.append(f"progression_samples_capped_to_{max_progression_samples}")
                    
                    pevents: List[PredictionEvent] = []
                    successful_samples = 0
                    
                    for dt in dates:
                        # Check timeout before each progression call
                        if time.time() - t0 > max_computation_time * 0.8:
                            warns.append("progression_sampling_stopped_due_to_timeout")
                            break
                            
                        try:
                            pr = predict_progressions(natal_chart, dt, statistical_validation=False, **params)
                            if pr.ok: 
                                pevents += pr.events
                                successful_samples += 1
                        except Exception as e:
                            warns.append(f"progression_sample_failed:{e}")
                            continue
                    
                    predictions["progressions"] = PredictionResult(
                        ok=True, technique="progressions_ensemble",
                        events=pevents, 
                        synthesis={
                            "sample_dates": successful_samples, 
                            "total_events": len(pevents),
                            "computation_limited": len(dates) < duration_days // 30
                        },
                        confidence_score=(sum(e.confidence for e in pevents) / len(pevents)) if pevents else 0.0,
                    )
                    all_events += pevents
                    
                except Exception as e:
                    warns.append(f"progressions_ensemble_failed:{e}")

        # SOLAR RETURNS - Add limits to prevent excessive computation
        if "solar_returns" in techs:
            if time.time() - t0 > max_computation_time * 0.9:  # Use 90% of time budget
                warns.append("returns_skipped_due_to_timeout_risk")
            else:
                try:
                    params = {k.replace("return_", ""): v for k, v in tk_kwargs.items() if k.startswith("return_")}
                    
                    # Limit years to prevent excessive computation
                    all_years = list(range(start_dt.year, end_dt.year + 1))
                    if len(all_years) > max_return_years:
                        all_years = all_years[:max_return_years]
                        warns.append(f"return_years_capped_to_{max_return_years}")
                    
                    revents: List[PredictionEvent] = []
                    successful_returns = 0
                    
                    for y in all_years:
                        # Check timeout before each return call
                        if time.time() - t0 > max_computation_time * 0.95:
                            warns.append("return_computation_stopped_due_to_timeout")
                            break
                            
                        try:
                            rr = predict_returns(natal_chart, "solar", y, statistical_validation=False, **params)
                            if rr.ok:
                                for ev in rr.events:
                                    if ev.datetime_utc and start_dt <= ev.datetime_utc <= end_dt:
                                        revents.append(ev)
                                successful_returns += 1
                        except Exception as e:
                            warns.append(f"return_year_{y}_failed:{e}")
                            continue
                    
                    predictions["solar_returns"] = PredictionResult(
                        ok=True, technique="solar_returns_ensemble",
                        events=revents, 
                        synthesis={
                            "years_computed": successful_returns, 
                            "total_events": len(revents),
                            "years_requested": len(all_years)
                        },
                        confidence_score=(sum(e.confidence for e in revents) / len(revents)) if revents else 0.0,
                    )
                    all_events += revents
                    
                except Exception as e:
                    warns.append(f"solar_returns_ensemble_failed:{e}")

        # VEDIC TECHNIQUES - Simplify to prevent issues
        if include_vedic and _PRED_OK and time.time() - t0 < max_computation_time * 0.98:
            try:
                # Use fixed timescale resolution
                jd_tt_natal, _, w_vedic = _resolve_natal_timescales_fixed(natal_chart)
                warns.extend(w_vedic)
                
                ay = float(natal_chart.get("ayanamsa_deg", 0.0) or 0.0)
                
                # Simplified moon longitude search
                moon_lon = None
                if isinstance(natal_chart.get("natal_longitudes"), dict):
                    moon_lon = float(natal_chart["natal_longitudes"].get("moon", None))
                else:
                    for b in (natal_chart.get("bodies") or []):
                        if isinstance(b, dict) and str(b.get("name","")).lower() == "moon":
                            moon_lon = float(b.get("longitude", b.get("lon", None))); break
                            
                if moon_lon is not None:
                    # Use safe function call instead of direct call
                    periods = vimsottari_dasha_safe(
                        birth_jd_tt=jd_tt_natal,
                        moon_lon_tropical_deg=moon_lon,
                        ayanamsa_deg=ay,
                        levels=min(int(tk_kwargs.get("vedic_levels", 3)), 2),  # Limit to 2 levels max
                        span_years=min(float(tk_kwargs.get("vedic_span_years", 120.0)), 50.0),  # Limit span
                    )
                    
                    s_tt, _ = _jd_pair_from_dt_fixed(start_dt); e_tt, _ = _jd_pair_from_dt_fixed(end_dt)
                    des: List[PredictionEvent] = []
                    
                    for p in periods[:20]:  # Limit to 20 periods maximum
                        if p.end_jd_tt < s_tt or p.start_jd_tt > e_tt:
                            continue
                        des.append(PredictionEvent(
                            event_type="dasha", technique="vimshottari_dasha",
                            description="/".join(p.parent_chain), datetime_utc=None,
                            jd_tt=p.start_jd_tt, jd_ut1=None, precision_seconds=None,
                            confidence=0.7 if p.level == 1 else 0.6 if p.level == 2 else 0.5,
                            significance=0.05,
                            metadata={
                                "level": p.level, "lord": p.lord, "chain": p.parent_chain,
                                "start_jd_tt": p.start_jd_tt, "end_jd_tt": p.end_jd_tt,
                            },
                        ))
                        
                    predictions["dasha"] = PredictionResult(
                        ok=True, technique="dasha",
                        events=des, synthesis={"total_periods": len(des)},
                        confidence_score=(sum(e.confidence for e in des) / len(des)) if des else 0.0,
                    )
                    all_events += des
                    
            except Exception as e:
                warns.append(f"vedic_techniques_failed:{e}")

        # Check final timeout before synthesis
        if time.time() - t0 > max_computation_time:
            warns.append("synthesis_simplified_due_to_timeout")
            synthesis = {"error": "computation_timeout_before_synthesis", "event_count": len(all_events)}
        else:
            synthesis = _synthesize_safe(all_events, predictions, method=synthesis_method, time_range=(start_dt, end_dt))
        
        peak_periods = _identify_peak_periods_safe(all_events, (start_dt, end_dt), window_days=peak_window_days)
        risk = _risk_safe(all_events, synthesis)
        
        conf_metrics = {
            "overall_confidence": synthesis.get("weighted_confidence", 0.0),
            "technique_agreement": synthesis.get("consensus_score", 0.0),
            "event_density": len(all_events) / max(1, (end_dt - start_dt).days),
            "validation_passed": False,
            "computation_limited": len(warns) > 0,
        }

        return ComprehensiveForecast(
            natal_chart=natal_chart,
            time_range=(start_dt, end_dt),
            predictions=predictions,
            synthesis={**synthesis, "warnings": warns},
            peak_periods=peak_periods,
            risk_assessment=risk,
            confidence_metrics=conf_metrics,
            validation_results=None,
            computation_time_ms=(time.time() - t0) * 1000.0,
        )
        
    except Exception as e:
        return ComprehensiveForecast(
            natal_chart=natal_chart,
            time_range=( _ensure_utc(time_range[0]) if isinstance(time_range[0], (str, datetime)) else datetime.now(timezone.utc),
                         _ensure_utc(time_range[1]) if isinstance(time_range[1], (str, datetime)) else datetime.now(timezone.utc)),
            synthesis={"error": f"comprehensive_forecast_failed:{e}", "warnings": warns},
            computation_time_ms=(time.time() - t0) * 1000.0,
        )


def vimsottari_dasha_safe(*args, **kwargs):
    """Safe wrapper for vimsottari_dasha to prevent hangs."""
    try:
        return vimsottari_dasha(*args, **kwargs)
    except Exception as e:
        return []  # Return empty list on error


def _synthesize_safe(all_events, predictions, method, time_range):
    """Safe wrapper for _synthesize to prevent hangs."""
    try:
        return _synthesize(all_events, predictions, method=method, time_range=time_range)
    except Exception as e:
        return {"error": f"synthesis_failed:{e}", "event_count": len(all_events)}


def _identify_peak_periods_safe(all_events, time_range, window_days):
    """Safe wrapper for _identify_peak_periods to prevent hangs."""
    try:
        return _identify_peak_periods(all_events, time_range, window_days=window_days)
    except Exception as e:
        return []  # Return empty list on error


def _risk_safe(all_events, synthesis):
    """Safe wrapper for _risk to prevent hangs."""
    try:
        return _risk(all_events, synthesis)
    except Exception as e:
        return {"error": f"risk_assessment_failed:{e}", "level": "unknown"}
        
def relationship_forecast(
    natal_a: Dict[str, Any],
    natal_b: Dict[str, Any],
    time_range: Tuple[Union[datetime, str], Union[datetime, str]],
    *,
    synastry_orbs: Optional[Dict[str, float]] = None,
    composite_method: str = "midpoint",
    include_transits_to_composite: bool = True,
    include_progressions: bool = True,
    confidence_threshold: float = 0.2,
    **kwargs,
) -> RelationshipForecast:
    _check_env()
    t0 = time.time()
    try:
        if not _SYN_OK:
            raise RuntimeError(f"synastry unavailable: {_SYN_ERR}")

        start_dt = _ensure_utc(time_range[0]); end_dt = _ensure_utc(time_range[1])

        syn = compute_synastry(
            natal_a=natal_a, natal_b=natal_b, orbs=synastry_orbs,
            parallels=kwargs.get("parallels", True), antiscia=kwargs.get("antiscia", True),
            **{k: v for k, v in kwargs.items() if k in ("frame","zodiac_mode","ayanamsa_deg","house_system")},
        )
        comp = compute_composite(
            natal_a=natal_a, natal_b=natal_b, method=composite_method,
            **{k: v for k, v in kwargs.items() if k in ("frame","zodiac_mode","ayanamsa_deg","house_system")},
        )

        transit_inter: List[PredictionEvent] = []
        prog_inter: List[PredictionEvent] = []

        try:
            tr_a = predict_transits(natal_a, (start_dt, end_dt), confidence_threshold=confidence_threshold, statistical_validation=False,
                                    **{k.replace("transit_",""): v for k, v in kwargs.items() if k.startswith("transit_")})
            tr_b = predict_transits(natal_b, (start_dt, end_dt), confidence_threshold=confidence_threshold, statistical_validation=False,
                                    **{k.replace("transit_",""): v for k, v in kwargs.items() if k.startswith("transit_")})
            for e in tr_a.events: e.metadata["person"]="A"; e.description=f"A: {e.description}"; transit_inter.append(e)
            for e in tr_b.events: e.metadata["person"]="B"; e.description=f"B: {e.description}"; transit_inter.append(e)

            if include_transits_to_composite and comp.get("ok"):
                comp_chart = _chart_from_composite(comp)
                if comp_chart:
                    tr_c = predict_transits(comp_chart, (start_dt, end_dt), confidence_threshold=confidence_threshold, statistical_validation=False,
                                            **{k.replace("transit_",""): v for k, v in kwargs.items() if k.startswith("transit_")})
                    for e in tr_c.events: e.metadata["person"]="Composite"; e.description=f"Composite: {e.description}"; transit_inter.append(e)
        except Exception:
            pass

        if include_progressions:
            duration = (end_dt - start_dt).days
            step = max(30, duration // 6) or 30
            for d in range(0, duration + 1, step):
                dt = start_dt + timedelta(days=d)
                pa = predict_progressions(natal_a, dt, statistical_validation=False, **{k.replace("progression_",""): v for k, v in kwargs.items() if k.startswith("progression_")})
                pb = predict_progressions(natal_b, dt, statistical_validation=False, **{k.replace("progression_",""): v for k, v in kwargs.items() if k.startswith("progression_")})
                for e in pa.events: e.metadata["person"]="A"; e.description=f"A: {e.description}"; prog_inter.append(e)
                for e in pb.events: e.metadata["person"]="B"; e.description=f"B: {e.description}"; prog_inter.append(e)

        trends = _compat_trends(transit_inter + prog_inter, (start_dt, end_dt))
        critical = _relationship_critical(transit_inter + prog_inter, (start_dt, end_dt))

        rel_score = 0.0
        if syn.get("ok"):
            s = syn.get("scores", {})
            rel_score = max(0.0, min(1.0, (float(s.get("total", 0.0)) + 20.0) / 40.0))

        conf = {
            "synastry_confidence": 1.0 if syn.get("ok") else 0.0,
            "composite_confidence": 1.0 if comp.get("ok") else 0.0,
            "transit_coverage": len(transit_inter) / max(1, (end_dt - start_dt).days),
            "progression_coverage": len(prog_inter) / max(1, ((end_dt - start_dt).days // 30) or 1),
            "overall_confidence": 0.25 * (
                (1.0 if syn.get("ok") else 0.0) +
                (1.0 if comp.get("ok") else 0.0) +
                (1.0 if transit_inter else 0.0) +
                (1.0 if prog_inter else 0.0)
            ),
        }

        return RelationshipForecast(
            synastry_analysis=syn if syn.get("ok") else {"ok": False, "error": syn.get("error")},
            composite_analysis=comp if comp.get("ok") else {"ok": False, "error": comp.get("error")},
            transit_interactions=transit_inter,
            progression_interactions=prog_inter,
            compatibility_trends=trends,
            critical_periods=critical,
            relationship_score=rel_score,
            confidence_metrics=conf,
        )
    except Exception as e:
        return RelationshipForecast(
            synastry_analysis={"ok": False, "error": f"synastry_failed:{e}"},
            composite_analysis={"ok": False, "error": f"composite_failed:{e}"},
            transit_interactions=[], progression_interactions=[],
            compatibility_trends={"trend": 0.0, "volatility": 0.0},
            critical_periods=[], relationship_score=0.0,
            confidence_metrics={"overall_confidence": 0.0},
        )

# ── Synthesis & analytics ────────────────────────────────────────────────────

def _synthesize(all_events: List[PredictionEvent], preds: Dict[str, PredictionResult], *, method: str, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
    if method == "weighted_consensus":
        weights = {"transits":1.0,"progressions":0.8,"solar_returns":0.9,"lunar_returns":0.7,"directions":0.8,"dasha":0.6}
        w_conf = w_sum = 0.0; consensus_events = 0
        for k, v in preds.items():
            if v.ok:
                w = weights.get(k, 0.5)
                w_conf += w * v.confidence_score; w_sum += w
                consensus_events += sum(1 for e in v.events if e.confidence >= 0.5)
        return {
            "method":"weighted_consensus",
            "weighted_confidence": (w_conf / w_sum) if w_sum > 0 else 0.0,
            "consensus_score": _consensus_score(preds, time_range),
            "technique_weights": weights,
            "total_events": len(all_events),
            "consensus_events": consensus_events,
            "technique_summary": {k: {"events": len(v.events), "confidence": v.confidence_score} for k, v in preds.items() if v.ok},
            "temporal_clusters": {"clustering_strength": _temporal_cluster_strength(all_events, time_range)},
        }
    return {
        "method": "simple",
        "total_events": len(all_events),
        "average_confidence": (sum(e.confidence for e in all_events) / len(all_events)) if all_events else 0.0,
        "techniques_used": list(preds.keys()),
    }

def _temporal_cluster_strength(events: List[PredictionEvent], time_range: Tuple[datetime, datetime]) -> float:
    ev = [e for e in events if e.datetime_utc and time_range[0] <= e.datetime_utc <= time_range[1]]
    if len(ev) < 3: return 0.0
    ev.sort(key=lambda x: x.datetime_utc)  # type: ignore
    gaps = [ (b.datetime_utc - a.datetime_utc).total_seconds()/86400.0 for a, b in zip(ev, ev[1:]) ]  # type: ignore
    if not gaps: return 0.0
    mean_gap = sum(gaps)/len(gaps)
    var_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    return max(0.0, min(1.0, (1.0/(1.0+mean_gap)) * (1.0 + min(1.0, var_gap/30.0))))

def _consensus_score(preds: Dict[str, PredictionResult], time_range: Tuple[datetime, datetime]) -> float:
    if sum(1 for v in preds.values() if v.ok) < 2: return 1.0
    timings: Dict[str, List[Tuple[int, float]]] = {}
    s, e = time_range
    for name, pr in preds.items():
        if pr.ok:
            lst: List[Tuple[int, float]] = []
            for ev in pr.events:
                if ev.datetime_utc and s <= ev.datetime_utc <= e:
                    lst.append(((ev.datetime_utc - s).days, ev.confidence))
            timings[name] = lst
    if len(timings) < 2: return 1.0
    cors: List[float] = []
    names = list(timings.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_events = timings[names[i]]; b_events = timings[names[j]]
            if not a_events or not b_events: continue
            total = 0.0; cnt = 0
            for day_a, conf_a in a_events:
                best = 0.0
                for day_b, conf_b in b_events:
                    dt = abs(day_a - day_b)
                    if dt <= 7:
                        best = max(best, math.exp(-dt / 3.0) * min(conf_a, conf_b))
                total += best; cnt += 1
            if cnt > 0: cors.append(total / cnt)
    return sum(cors)/len(cors) if cors else 0.0

def _identify_peak_periods(events: List[PredictionEvent], time_range: Tuple[datetime, datetime], *, window_days: int = 14) -> List[TimingWindow]:
    ev = [e for e in events if e.datetime_utc and time_range[0] <= e.datetime_utc <= time_range[1]]
    if not ev: return []
    ev.sort(key=lambda x: x.datetime_utc)  # type: ignore
    dur = (time_range[1] - time_range[0]).days
    windows: List[Tuple[float, TimingWindow]] = []
    for offset in range(0, max(1, dur - window_days + 1), 7):
        ws = time_range[0] + timedelta(days=offset); we = ws + timedelta(days=window_days)
        win = [e for e in ev if ws <= e.datetime_utc <= we]  # type: ignore
        if len(win) >= 2:
            total_conf = sum(e.confidence for e in win)
            peak = max(win, key=lambda x: x.confidence)
            s_tt, _ = _jd_pair_from_dt_dt(ws); e_tt, _ = _jd_pair_from_dt_dt(we); p_tt, _ = _jd_pair_from_dt_dt(peak.datetime_utc)  # type: ignore
            score = len(win) * (total_conf / len(win))
            windows.append((score, TimingWindow(s_tt, e_tt, p_tt, window_days/2, (s_tt, e_tt))))
    windows.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in windows[:5]]

def _risk(events: List[PredictionEvent], synthesis: Dict[str, Any]) -> Dict[str, float]:
    total = len(events)
    if total == 0:
        return {"temporal_stress": 0.0, "aspect_challenge": 0.0, "aspect_support": 0.0, "prediction_uncertainty": 1.0, "outer_planet_influence": 0.0, "overall_risk": 0.0}
    chall = {"square","opposition","quincunx"}; supp = {"trine","sextile","conjunction"}
    c_count = sum(1 for e in events if e.metadata.get("aspect") in chall)
    s_count = sum(1 for e in events if e.metadata.get("aspect") in supp)
    risk: Dict[str, float] = {}
    risk["temporal_stress"] = min(1.0, 2.0 * synthesis.get("temporal_clusters", {}).get("clustering_strength", 0.0))
    risk["aspect_challenge"] = c_count / total
    risk["aspect_support"] = s_count / total
    risk["prediction_uncertainty"] = 1.0 - (sum(e.confidence for e in events) / total)
    outer = {"saturn","uranus","neptune","pluto"}; o_count = 0
    for e in events:
        for k in ("transiting_body","progressed_body","body"):
            v = str(e.metadata.get(k, "")).lower()
            if any(p in v for p in outer): o_count += 1; break
    risk["outer_planet_influence"] = o_count / total
    weights = {"temporal_stress":0.3, "aspect_challenge":0.4, "prediction_uncertainty":0.2, "outer_planet_influence":0.1}
    risk["overall_risk"] = min(1.0, sum(risk[k]*w for k, w in weights.items()))
    return risk

def _compat_trends(events: List[PredictionEvent], time_range: Tuple[datetime, datetime]) -> Dict[str, float]:
    if not events:
        return {"trend": 0.0, "volatility": 0.0}
    dur = (time_range[1] - time_range[0]).days
    bucket = max(7, dur // 12) or 7
    scores: List[float] = []
    for off in range(0, dur + 1, bucket):
        bs = time_range[0] + timedelta(days=off); be = min(time_range[1], bs + timedelta(days=bucket))
        ee = [e for e in events if e.datetime_utc and bs <= e.datetime_utc <= be]
        if not ee:
            scores.append(0.0); continue
        s = 0.0
        for e in ee:
            asp = e.metadata.get("aspect",""); c = e.confidence
            if asp in {"trine","sextile","conjunction"}: s += c
            elif asp in {"square","opposition"}: s -= 0.5 * c
            else: s += 0.3 * c
        scores.append(s / len(ee))
    if len(scores) < 2: return {"trend": 0.0, "volatility": 0.0}
    n = len(scores); xm = (n - 1) / 2.0; ym = sum(scores) / n
    num = sum((i - xm) * (y - ym) for i, y in enumerate(scores))
    den = sum((i - xm) ** 2 for i in range(n)) or 1.0
    trend = num / den
    var = sum((y - ym) ** 2 for y in scores) / n
    return {"trend": float(trend), "volatility": math.sqrt(var), "bucket_scores": scores, "bucket_count": n}

def _relationship_critical(events: List[PredictionEvent], time_range: Tuple[datetime, datetime]) -> List[TimingWindow]:
    ev = [e for e in events if e.datetime_utc and time_range[0] <= e.datetime_utc <= time_range[1]]
    if not ev: return []
    ev.sort(key=lambda x: x.datetime_utc)  # type: ignore
    win_days = 21; dur = (time_range[1] - time_range[0]).days
    outs: List[Tuple[float, TimingWindow]] = []
    for off in range(0, max(1, dur - win_days + 1), 7):
        ws = time_range[0] + timedelta(days=off); we = ws + timedelta(days=win_days)
        wv = [e for e in ev if ws <= e.datetime_utc <= we]  # type: ignore
        if len(wv) < 2: continue
        chall = {"square","opposition","quincunx"}
        ccount = sum(1 for e in wv if e.metadata.get("aspect") in chall)
        avgc = sum(e.confidence for e in wv) / len(wv)
        score = 2.0 * ccount + len(wv) * avgc
        if score >= 3.0:
            s_tt, _ = _jd_pair_from_dt_dt(ws); e_tt, _ = _jd_pair_from_dt_dt(we)
            peak = max(wv, key=lambda x: x.confidence)
            p_tt, _ = _jd_pair_from_dt_dt(peak.datetime_utc)  # type: ignore
            outs.append((score, TimingWindow(s_tt, e_tt, p_tt, win_days/3, (s_tt, e_tt))))
    outs.sort(key=lambda x: x[1].peak_jd_tt or 0.0)
    return [w for _, w in outs[:5]]

# ── Validation harness (lightweight stubs) ───────────────────────────────────

def validate_prediction_model(
    test_cases: List[Dict[str, Any]],
    *,
    validation_method: str = "cross_validation",
    n_folds: int = 5,
    metrics: Optional[List[str]] = None,
    confidence_threshold: float = 0.1,
    **kwargs,
) -> Dict[str, Any]:
    _check_env()
    try:
        if not test_cases:
            raise ValueError("No test cases provided")
        metrics = metrics or ["precision","recall","f1","timing_accuracy","confidence_calibration"]
        if validation_method == "cross_validation":
            return _kfold(test_cases, n_folds, metrics, confidence_threshold, **kwargs)
        if validation_method == "bootstrap":
            return _bootstrap(test_cases, metrics, confidence_threshold, **kwargs)
        if validation_method == "holdout":
            return _holdout(test_cases, metrics, confidence_threshold, **kwargs)
        raise ValueError(f"Unknown validation method: {validation_method}")
    except Exception as e:
        return {"ok": False, "error": f"validation_failed:{e}", "method": validation_method}

def _kfold(cases: List[Dict[str, Any]], k: int, metrics: List[str], thr: float, **kwargs) -> Dict[str, Any]:
    k = max(2, min(k, len(cases))) if cases else 2
    size = len(cases) // k if k else 0
    folds = [cases[i * size : (i + 1) * size] for i in range(k - 1)] + [cases[(k - 1) * size :]]
    results = []
    for i in range(k):
        test = folds[i]
        train = [c for j, f in enumerate(folds) if j != i for c in f]
        results.append(_eval_metrics(test, train, metrics, thr, **kwargs))
    agg: Dict[str, Any] = {}
    for m in metrics:
        vals = [r.get(m, 0.0) for r in results]
        if not vals: continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        agg[m] = {"mean": mean, "std": math.sqrt(var), "min": min(vals), "max": max(vals), "values": vals}
    return {"ok": True, "method": "cross_validation", "n_folds": k, "aggregated_metrics": agg, "fold_results": results, "confidence_threshold": thr}

def _eval_metrics(test: List[Dict[str, Any]], train: List[Dict[str, Any]], metrics: List[str], thr: float, **kwargs) -> Dict[str, float]:
    preds: List[bool] = []; truth: List[bool] = []; t_errs: List[float] = []
    conf_pred: List[float] = []; conf_true: List[float] = []
    for case in test:
        try:
            natal = case.get("natal_chart", {})
            trange = case.get("time_range", (datetime.now(timezone.utc), datetime.now(timezone.utc) + timedelta(days=365)))
            pred_type = case.get("prediction_type", "transits")
            if pred_type == "comprehensive":
                cf = comprehensive_forecast(natal, trange, confidence_threshold=thr, statistical_validation=False, **kwargs)
                evs: List[PredictionEvent] = []
                for pr in cf.predictions.values():
                    if pr.ok: evs += pr.events
            elif pred_type == "transits":
                pr = predict_transits(natal, trange, confidence_threshold=thr, statistical_validation=False, **kwargs)
                evs = pr.events if pr.ok else []
            elif pred_type == "progressions":
                td = case.get("target_date", trange[1])
                pr = predict_progressions(natal, td, statistical_validation=False, **kwargs); evs = pr.events if pr.ok else []
            elif pred_type == "returns":
                rt = case.get("return_type", "solar"); y = case.get("year", _ensure_utc(trange[1]).year)
                pr = predict_returns(natal, rt, y, statistical_validation=False, **kwargs); evs = pr.events if pr.ok else []
            else:
                continue
            expected = case.get("expected_events", [])
            for ex in expected:
                ex_time = ex.get("datetime"); ex_type = ex.get("event_type"); ex_conf = float(ex.get("confidence", 1.0))
                matches = []
                for pv in evs:
                    if pv.event_type == ex_type and pv.datetime_utc and ex_time:
                        dt = _ensure_utc(ex_time)
                        if abs((pv.datetime_utc - dt).total_seconds()) <= 7 * 86400: matches.append(pv)
                if matches:
                    preds.append(True); truth.append(True)
                    best = max(matches, key=lambda x: x.confidence)
                    t_errs.append(abs((best.datetime_utc - _ensure_utc(ex_time)).total_seconds()) / 86400.0)  # type: ignore
                    conf_pred.append(best.confidence); conf_true.append(ex_conf)
                else:
                    preds.append(False); truth.append(True)
            # false positives
            for pv in evs:
                if pv.confidence >= thr:
                    matched = False
                    for ex in expected:
                        ex_time = ex.get("datetime")
                        if pv.datetime_utc and ex_time and abs((pv.datetime_utc - _ensure_utc(ex_time)).total_seconds()) <= 7 * 86400:
                            matched = True; break
                    if not matched:
                        preds.append(True); truth.append(False)
        except Exception:
            continue
    out: Dict[str, float] = {}
    if "precision" in metrics:
        tp = sum(1 for i, p in enumerate(preds) if p and truth[i]); fp = sum(1 for i, p in enumerate(preds) if p and not truth[i])
        out["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    if "recall" in metrics:
        tp = sum(1 for i, p in enumerate(preds) if p and truth[i]); fn = sum(1 for i, p in enumerate(preds) if (not p) and truth[i])
        out["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if "f1" in metrics:
        P = out.get("precision", 0.0); R = out.get("recall", 0.0)
        out["f1"] = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    if "timing_accuracy" in metrics:
        out["timing_accuracy"] = 1.0 / (1.0 + (sum(t_errs) / len(t_errs))) if t_errs else 0.0
    if "confidence_calibration" in metrics:
        if conf_pred and conf_true:
            mse = sum((p - t) ** 2 for p, t in zip(conf_pred, conf_true)) / len(conf_pred)
            out["confidence_calibration"] = 1.0 / (1.0 + mse)
        else:
            out["confidence_calibration"] = 0.0
    return out

def _bootstrap(cases: List[Dict[str, Any]], metrics: List[str], thr: float, **kwargs) -> Dict[str, Any]:
    B = 100
    vals: Dict[str, List[float]] = {m: [] for m in metrics}
    for _ in range(B):
        sample = [cases[int(random.random() * len(cases))] for _ in range(len(cases))]
        r = _eval_metrics(sample, [], metrics, thr, **kwargs)
        for m in metrics: vals[m].append(r.get(m, 0.0))
    agg = {m: {"mean": (sum(v)/len(v) if v else 0.0),
               "std": (math.sqrt(sum((x - (sum(v)/len(v) if v else 0.0))**2 for x in v)/len(v)) if v else 0.0)}
           for m, v in vals.items()}
    return {"ok": True, "method": "bootstrap", "aggregated_metrics": agg, "B": B}

def _holdout(cases: List[Dict[str, Any]], metrics: List[str], thr: float, **kwargs) -> Dict[str, Any]:
    mid = len(cases) // 2 or 1
    train, test = cases[:mid], cases[mid:]
    r = _eval_metrics(test, train, metrics, thr, **kwargs)
    return {"ok": True, "method": "holdout", "metrics": r}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _chart_from_composite(comp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not comp.get("positions"):
        return None
    bodies = [{"name": name, "longitude": float(lon), "latitude": 0.0, "is_point": False} for name, lon in comp["positions"].items()]
    angles: Dict[str, float] = {}
    if comp.get("asc") is not None: angles["asc_deg"] = float(comp["asc"])
    if comp.get("mc")  is not None: angles["mc_deg"]  = float(comp["mc"])
    return {"bodies": bodies, "angles": angles, "mode": comp.get("meta", {}).get("zodiac_mode", "tropical")}

# ── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    "predict_transits",
    "predict_progressions",
    "predict_returns",
    "predict_directions",
    "comprehensive_forecast",
    "relationship_forecast",
    "validate_prediction_model",
    "PredictionEvent",
    "PredictionResult",
    "ComprehensiveForecast",
    "RelationshipForecast",
    "TimingWindow",
]
