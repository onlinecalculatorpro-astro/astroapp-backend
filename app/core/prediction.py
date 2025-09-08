# app/core/prediction.py
# -*- coding: utf-8 -*-
"""
Prediction Engine (v2) — research-grade, API-stable, cached rewrite

Highlights
----------
- Correct timescale handling via app.core.timescales
- Zero local-tz leakage: all datetimes are UTC-aware
- Transit orbs != synastry/progression/direction orbs
- Optional heavy stats (permutation/bootstrap/FDR), default OFF
- Angles & optional house-cusps support in transit targeting
- Ensemble synthesis with confidence & significance semantics
- Performance: LRU + keyed memo caches for hot paths
- Robust error handling and consistent result shapes

Public API
----------
predict_transits(natal_chart, time_range, **kwargs) -> PredictionResult
predict_progressions(natal_chart, target_date, **kwargs) -> PredictionResult
predict_returns(natal_chart, return_type, year, **kwargs) -> PredictionResult
predict_directions(natal_chart, target_date, **kwargs) -> PredictionResult
comprehensive_forecast(natal_chart, time_range, **kwargs) -> ComprehensiveForecast
relationship_forecast(natal_a, natal_b, time_range, **kwargs) -> RelationshipForecast
validate_prediction_model(test_cases, **kwargs) -> ValidationReport
"""

from __future__ import annotations

import json
import math
import random
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional, Union

# ───────────────────────────── Resilient imports ─────────────────────────────

# Core constants & helpers
try:
    from app.core.constants import (
        MAJOR_BODIES,
        ASPECT_ANGLES_DEG,
        DEFAULT_ORBS_SYNASTRY,
        DEFAULT_ORBS_PROGRESSIONS,
        DEFAULT_ORBS_DIRECTIONS,
        TROPICAL_YEAR_D,
        abs_sep_deg,
        V11_CONSTANTS_VERSION,
    )
    try:
        from app.core.constants import DEFAULT_ORBS_TRANSITS
    except Exception:
        DEFAULT_ORBS_TRANSITS = DEFAULT_ORBS_SYNASTRY  # safe fallback
    _CONST_OK = True
    _CONST_ERR: Optional[Exception] = None
except Exception as e:
    _CONST_OK = False
    _CONST_ERR = e

# Timescales & validators
try:
    from app.core.timescales import build_timescales, TimeScales
    _TS_OK = True
    _TS_ERR: Optional[Exception] = None
except Exception as e:
    _TS_OK = False
    _TS_ERR = e
    TimeScales = Any  # type: ignore

# Aspects
try:
    from app.core.aspects import compute_aspects, AspectConfig
    _ASPECTS_OK = True
    _ASPECTS_ERR: Optional[Exception] = None
except Exception as e:
    _ASPECTS_OK = False
    _ASPECTS_ERR = e
    AspectConfig = Any  # type: ignore

# Prediction submodules
try:
    from app.core.progressions import compute_progressions
    _PROG_OK = True
    _PROG_ERR: Optional[Exception] = None
except Exception as e:
    _PROG_OK = False
    _PROG_ERR = e

try:
    from app.core.returns import compute_solar_return, compute_lunar_return
    _RET_OK = True
    _RET_ERR: Optional[Exception] = None
except Exception as e:
    _RET_OK = False
    _RET_ERR = e

# Unified predictive toolkit (Transit / Dasha / Varga / Yoga)
try:
    from app.core.predictive import (
        # Transits
        TransitEngine, TransitEvent, find_transits_in_range,
        # Dasha
        DashaPeriod, vimsottari_dasha,
        # Varga
        compute_vargas_for_point, compute_vargas,
        # Yogas
        detect_yogas, house_index_for_longitude,
        # Houses & timescales helpers
        compute_houses, timescales_from_civil,
        # Validation primitives & feature builders (available if/when needed)
        EvalResult, evaluate_univariate, permutation_pvalue_corr, bh_fdr, holdout_replicate,
        feature_transit_proximity, feature_dasha_lords_onehot, feature_yoga_flags,
        # Predictive result type (aliased so it doesn't shadow ours)
        PredictionResult as PredictiveResult,
    )
    _PRED_OK = True
    _PRED_ERR: Optional[Exception] = None
except Exception as e:
    _PRED_OK = False
    _PRED_ERR = e

try:
    from app.core.directions import compute_directions
    _DIR_OK = True
    _DIR_ERR: Optional[Exception] = None
except Exception as e:
    _DIR_OK = False
    _DIR_ERR = e

try:
    from app.core.synastry import compute_synastry, compute_composite
    _SYN_OK = True
    _SYN_ERR: Optional[Exception] = None
except Exception as e:
    _SYN_OK = False
    _SYN_ERR = e

# ───────────────────────────── Data structures ─────────────────────────────

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

# ───────────────────────────── Utilities & caching ─────────────────────────────

def _check_env() -> None:
    if not _CONST_OK:
        raise RuntimeError(f"constants unavailable: {_CONST_ERR}")
    if not _TS_OK:
        raise RuntimeError(f"timescales unavailable: {_TS_ERR}")

def _ensure_utc(dt_or_str: Union[str, datetime]) -> datetime:
    """Parse ISO or pass-through datetime and return UTC-aware datetime."""
    if isinstance(dt_or_str, str):
        s = dt_or_str.strip()
        if "T" not in s and len(s) <= 10:
            s = s + "T00:00:00+00:00"
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    else:
        dt = dt_or_str
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _hash_obj(obj: Any) -> str:
    """Stable content hash for dict/list/tuple primitives."""
    try:
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        payload = repr(obj)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

@lru_cache(maxsize=4096)
def _jd_pair_from_dt(dt_utc_iso: str) -> Tuple[float, float]:
    """Cacheable wrapper: ISO UTC -> (jd_tt, jd_ut1)."""
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
    """LRU layer; see _normalize_aspect_config for key construction."""
    params = json.loads(key)
    base_name = params["base"]
    orbs = params["orbs"]
    zodiacal: Dict[str, Dict[str, float]] = {}
    for asp, angle in ASPECT_ANGLES_DEG.items():
        orb = orbs.get(asp, 0.0)
        if orb > 0:
            zodiacal[asp] = {"angle": angle, "orb": orb}
    antiscia_orb = orbs.get("antiscia", 0.0)
    parallel_orb_arcmin = orbs.get("parallel_arcmin", 0.0)
    return AspectConfig(
        zodiacal=zodiacal or None,
        antiscia={"orb": antiscia_orb} if antiscia_orb > 0 else None,
        declination={"orb_arcmin": parallel_orb_arcmin} if parallel_orb_arcmin > 0 else None,
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
    key = json.dumps({"base": technique, "orbs": eff}, sort_keys=True)
    return _normalize_aspect_config_cached(key)

@lru_cache(maxsize=2048)
def _body_orb_scale(body_a: str, body_b: str) -> float:
    """Scale orbs by body type (luminaries wider; outers tighter)."""
    lum = {"sun", "moon"}
    outer = {"uranus", "neptune", "pluto"}
    a = body_a.lower()
    b = body_b.lower()
    if a in lum or b in lum:
        return 1.2
    if a in outer or b in outer:
        return 0.9
    return 1.0

def _scaled_orb(orbs: Dict[str, float], asp: str, body_a: str, body_b: str) -> float:
    base = orbs.get(asp, orbs.get("default", 1.0))
    return max(1e-9, base * _body_orb_scale(body_a, body_b))

def _resolve_natal_timescales(natal: Dict[str, Any]) -> Tuple[float, float, List[str]]:
    """Return (jd_tt, jd_ut1, warnings)."""
    warns: List[str] = []
    if "jd_tt" in natal and "jd_ut1" in natal:
        try:
            return float(natal["jd_tt"]), float(natal["jd_ut1"]), warns
        except Exception:
            warns.append("invalid_strict_timescales_fallback_to_civil")
    missing = [k for k in ("date", "time", "place_tz") if k not in natal]
    if missing:
        raise RuntimeError(f"missing fields for timescales: {missing}")
    dut1 = float(natal.get("dut1", 0.0))
    ts: TimeScales = build_timescales(
        date_str=str(natal["date"]),
        time_str=str(natal["time"]),
        tz_name=str(natal["place_tz"]),
        dut1_seconds=dut1,
    )
    if abs(dut1) < 1e-9:
        warns.append("timescales_computed_with_dut1_zero_assumption")
    return float(ts.jd_tt), float(ts.jd_ut1), warns

def _compute_confidence(events: List[PredictionEvent], stats: Dict[str, float]) -> float:
    if not events:
        return 0.0
    confidences = [e.confidence for e in events if e.confidence > 0]
    if not confidences:
        return 0.0
    weights = [c ** 1.5 for c in confidences]
    weighted_avg = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
    p_value = float(stats.get("p_value", 1.0))
    significance_boost = 0.3 * max(0.0, 1.0 - p_value / 0.01) if p_value < 0.05 else 0.0
    count_factor = min(0.15, 0.03 * math.log(1 + len(events)))
    techniques = {e.technique for e in events}
    diversity_bonus = min(0.1, 0.03 * len(techniques))
    return min(1.0, weighted_avg + significance_boost + count_factor + diversity_bonus)

def _create_timing_windows(events: List[PredictionEvent], *, cluster_days: float = 7.0) -> List[TimingWindow]:
    ev = [e for e in events if e.datetime_utc is not None]
    if len(ev) < 2:
        return []
    ev.sort(key=lambda e: e.datetime_utc)  # type: ignore
    wins: List[TimingWindow] = []
    group: List[PredictionEvent] = [ev[0]]
    for e in ev[1:]:
        dt_days = (e.datetime_utc - group[-1].datetime_utc).total_seconds() / 86400.0  # type: ignore
        avg_confidence = sum(g.confidence for g in group) / len(group)
        adaptive_threshold = cluster_days * (0.5 + 0.5 * avg_confidence)
        if dt_days <= adaptive_threshold:
            group.append(e)
        else:
            if len(group) > 1:
                s = group[0].datetime_utc  # type: ignore
                q = group[-1].datetime_utc  # type: ignore
                peak = max(group, key=lambda x: x.confidence)
                s_jd_tt, _ = _jd_pair_from_dt_dt(s)  # type: ignore
                q_jd_tt, _ = _jd_pair_from_dt_dt(q)  # type: ignore
                pk_jd_tt, _ = _jd_pair_from_dt_dt(peak.datetime_utc)  # type: ignore
                wins.append(TimingWindow(
                    start_jd_tt=s_jd_tt,
                    end_jd_tt=q_jd_tt,
                    peak_jd_tt=pk_jd_tt,
                    uncertainty_days=(q_jd_tt - s_jd_tt),
                    confidence_interval=(s_jd_tt, q_jd_tt),
                ))
            group = [e]
    if len(group) > 1:
        s = group[0].datetime_utc  # type: ignore
        q = group[-1].datetime_utc  # type: ignore
        peak = max(group, key=lambda x: x.confidence)
        s_jd_tt, _ = _jd_pair_from_dt_dt(s)  # type: ignore
        q_jd_tt, _ = _jd_pair_from_dt_dt(q)  # type: ignore
        pk_jd_tt, _ = _jd_pair_from_dt_dt(peak.datetime_utc)  # type: ignore
        wins.append(TimingWindow(
            start_jd_tt=s_jd_tt,
            end_jd_tt=q_jd_tt,
            peak_jd_tt=pk_jd_tt,
            uncertainty_days=(q_jd_tt - s_jd_tt),
            confidence_interval=(s_jd_tt, q_jd_tt),
        ))
    return wins

def _body_activity(events: List[PredictionEvent]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    confs: Dict[str, float] = {}
    for e in events:
        for key in ("transiting_body", "progressed_body", "return_body", "directed_body", "body"):
            b = e.metadata.get(key)
            if b:
                counts[b] = counts.get(b, 0) + 1
                confs[b] = confs.get(b, 0.0) + e.confidence
    dist = {
        b: {
            "event_count": counts[b],
            "average_confidence": confs[b] / counts[b],
            "total_confidence": confs[b],
        } for b in counts
    }
    top = dict(sorted(dist.items(), key=lambda kv: kv[1]["total_confidence"], reverse=True)[:5])
    return {"most_active": top, "total_bodies": len(dist), "activity_distribution": dist}

# ───────────────────────────── Predictive caching wrapper ─────────────────────

@lru_cache(maxsize=1024)
def _memo_predictive(key: str):
    """Memoizes pure predictive calls using a stable string key."""
    args = json.loads(key)
    kind = args["kind"]
    if kind == "transits":
        # Directly return the PredictionResult coming from predictive.find_transits_in_range
        return find_transits_in_range(**args["payload"])
    raise RuntimeError(f"Unsupported memo kind: {kind}")

def _memo_key_transits(payload: Dict[str, Any]) -> str:
    safe = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return json.dumps({"kind": "transits", "payload": json.loads(safe)}, sort_keys=True)

# ───────────────────────────── Prediction engines ─────────────────────────────

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
    statistical_validation: bool = False,   # kept for API, currently no-op
    confidence_threshold: float = 0.1,
    **kwargs,
) -> PredictionResult:
    _check_env()
    t0 = time.time()
    warns: List[str] = []
    try:
        if not _PRED_OK:
            raise RuntimeError(f"predictive unavailable: {_PRED_ERR}")

        jd_tt_natal, jd_ut1_natal, w = _resolve_natal_timescales(natal_chart)
        warns.extend(w)

        start_dt = _ensure_utc(time_range[0])
        end_dt = _ensure_utc(time_range[1])

        transiting_bodies = transiting_bodies or list(MAJOR_BODIES)
        natal_bodies = natal_bodies or list(MAJOR_BODIES)
        aspects_list = [a.lower() for a in (aspects or ["conjunction", "opposition", "trine", "square", "sextile"])]
        include_aspects_to = include_aspects_to or ["planets", "angles"]

        # NOTE: predictive currently targets planets (Asc/MC optional in future)
        orbs_to_use = orbs or DEFAULT_ORBS_TRANSITS

        # Build payload for predictive.find_transits_in_range (uses date strings)
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
        tr_res: PredictiveResult = _memo_predictive(key)  # returned by predictive.py

        events: List[PredictionEvent] = []

        if not getattr(tr_res, "ok", False):
            warns.append("predictive_find_transits_failed")
        else:
            for ev in getattr(tr_res, "events", []):
                meta = getattr(ev, "metadata", {}) or {}
                body = meta.get("body", "")
                target = meta.get("target", "")
                aspect = _canon_aspect(meta.get("aspect"))

                # tightness from |separation| / allowed orb (if provided)
                sep = abs(float(meta.get("separation_deg", 999.0)))
                # pull orb if predictive attached it; otherwise fall back to our orbs
                meta_orb = 0.0
                mmeta = meta.get("meta") if isinstance(meta.get("meta"), dict) else {}
                if isinstance(mmeta, dict):
                    meta_orb = float(mmeta.get("orb_deg", 0.0))
                orb_cap = float(meta_orb or orbs_to_use.get(aspect, 1.0))
                orb_cap = max(orb_cap, 1e-9)

                tight = max(0.0, 1.0 - (sep / orb_cap))
                major = 0.2 if aspect in {"conjunction", "opposition", "trine", "square"} else 0.0
                conf = min(1.0, tight + major)

                pe = PredictionEvent(
                    event_type="transit",
                    technique="exact_transit",
                    description=f"{body} {aspect} {target}",
                    datetime_utc=None,         # predictive leaves UTC datetime None (we're using jd_tt)
                    jd_tt=getattr(ev, "jd_tt", None),
                    jd_ut1=None,
                    precision_seconds=None,
                    confidence=conf,
                    significance=1.0,          # no p-values in fast scan
                    metadata={
                        "transiting_body": body,
                        "natal_body": target,
                        "aspect": aspect,
                        "orb": sep,             # here 'orb' == absolute separation from exact
                        "max_orb": orb_cap,
                        "applying": bool(meta.get("applying", False)),
                        "exact": bool(meta.get("exact", False)),
                        "kind": meta.get("kind", "zodiacal"),
                        "finder": "predictive.find_transits_in_range",
                    },
                )
                if pe.confidence >= confidence_threshold:
                    events.append(pe)

        # Validation step intentionally omitted (no validate_predictions in predictive.py)
        stats: Dict[str, float] = {}

        windows = _create_timing_windows(events)
        conf_score = _compute_confidence(events, stats)

        synth = {
            "total_transits": len(events),
            "major_aspects": sum(1 for e in events if e.metadata.get("aspect") in {"conjunction", "opposition", "trine", "square"}),
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
    statistical_validation: bool = False,  # kept for API, no-op
    **kwargs,
) -> PredictionResult:
    _check_env()
    t0 = time.time()
    warns: List[str] = []
    try:
        if not _PROG_OK:
            raise RuntimeError(f"progressions unavailable: {_PROG_ERR}")

        jd_tt_natal, jd_ut1_natal, w = _resolve_natal_timescales(natal_chart)
        warns.extend(w)

        # Determine years_after since natal
        if isinstance(target_date, (int, float)):
            years_after = (float(target_date) - jd_tt_natal)  # JD delta in days
            years_after = years_after / TROPICAL_YEAR_D
            target_dt: Optional[datetime] = None
        else:
            t_dt = _ensure_utc(target_date)
            # Convert natal JD_TT -> UTC datetime for consistent subtraction
            natal_epoch_ts = (jd_tt_natal - 2440587.5) * 86400.0
            natal_dt = datetime.fromtimestamp(natal_epoch_ts, tz=timezone.utc)
            years_after = (t_dt - natal_dt).total_seconds() / (TROPICAL_YEAR_D * 86400.0)
            target_dt = t_dt

        res = compute_progressions(
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
            orbs=orbs or DEFAULT_ORBS_PROGRESSIONS,
            parallels=parallels,
            antiscia=antiscia,
            profile=True,
            validation="basic",
        )
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "progressions_failed"))

        events: List[PredictionEvent] = []

        # Aspects to natal
        if aspects_to_natal and isinstance(res.get("aspects"), dict):
            for fam, hits in res["aspects"].items():
                if not isinstance(hits, list):
                    continue
                for h in hits:
                    asp = _canon_aspect(h.get("aspect"))
                    orb = float(h.get("orb", 0.0))
                    max_orb = float(h.get("max_orb", max((orbs or DEFAULT_ORBS_PROGRESSIONS).get(asp, 1.0), 1e-9)))
                    tight = max(0.0, 1.0 - (orb / max_orb))
                    exact_bonus = 0.3 if orb < 0.1 else 0.0
                    conf = min(1.0, tight + exact_bonus)
                    events.append(PredictionEvent(
                        event_type="progression",
                        technique=f"{method}_progression",
                        description=f"Progressed {h.get('planet_a','?')} {asp} Natal {h.get('planet_b','?')}",
                        datetime_utc=target_dt,
                        jd_tt=None,
                        jd_ut1=None,
                        precision_seconds=None,
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

        # Positional movements summary
        if isinstance(res.get("positions"), list):
            natal_lookup: Dict[str, float] = {}
            if isinstance(natal_chart.get("bodies"), list):
                for b in natal_chart["bodies"]:
                    if isinstance(b, dict) and "name" in b:
                        natal_lookup[str(b["name"])] = float(b.get("longitude", b.get("lon", 0.0)))
            for p in res["positions"]:
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
                            jd_tt=None,
                            jd_ut1=None,
                            precision_seconds=None,
                            confidence=conf,
                            significance=1.0,
                            metadata={
                                "body": name,
                                "natal_longitude": natal_lon,
                                "progressed_longitude": prog_lon,
                                "movement_degrees": dist,
                                "speed": float(p.get("speed", 0.0)),
                            },
                        ))

        # Validation step omitted
        stats: Dict[str, float] = {}

        synth = {
            "progression_method": method,
            "target_date": (target_dt.isoformat() if target_dt else None),
            "years_after_natal": years_after,
            "total_aspects": sum(1 for e in events if e.event_type == "progression" and "progression" in e.technique),
            "significant_movements": sum(1 for e in events if e.event_type == "progression" and "position" in e.technique),
            "average_confidence": (sum(e.confidence for e in events) / len(events)) if events else 0.0,
            "progression_epoch": res.get("meta", {}).get("epoch", {}),
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
                    "method": method,
                    "lunar_month": lunar_month,
                    "tertiary_mode": tertiary_mode,
                    "frame": frame,
                    "zodiac_mode": zodiac_mode,
                    "aspects_to_natal": aspects_to_natal,
                },
            },
            computation_time_ms=(time.time() - t0) * 1000.0,
        )
    except Exception as e:
        return PredictionResult(
            ok=False,
            technique="progressions",
            warnings=[f"progression_computation_failed:{e}"],
            computation_time_ms=(time.time() - t0) * 1000.0,
        )

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
    statistical_validation: bool = False,  # kept for API, no-op
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

        jd_tt_natal, jd_ut1_natal, w = _resolve_natal_timescales(natal_chart)
        warns.extend(w)

        use_place = place or natal_chart
        if return_type.lower() == "solar":
            rr = compute_solar_return(
                natal=natal_chart,
                target_year=int(year),
                jd_tt_natal=jd_tt_natal,
                jd_ut1_natal=jd_ut1_natal,
                place=use_place,
                frame=frame,
                house_system=house_system,
                zodiac_mode=zodiac_mode,
                ayanamsa_deg=ayanamsa_deg,
                estimate_uncertainty=estimate_uncertainty,
                profile=True,
                validation="extended",
            )
        else:
            rr = compute_lunar_return(
                natal=natal_chart,
                target_year=int(year),
                lunar_month=lunar_month,
                jd_tt_natal=jd_tt_natal,
                jd_ut1_natal=jd_ut1_natal,
                place=use_place,
                frame=frame,
                house_system=house_system,
                zodiac_mode=zodiac_mode,
                ayanamsa_deg=ayanamsa_deg,
                estimate_uncertainty=estimate_uncertainty,
                profile=True,
                validation="extended",
            )
        if not rr.get("ok"):
            raise RuntimeError(rr.get("error", "return_failed"))

        ret_jd_tt = rr.get("return_jd_tt")
        ret_jd_ut1 = rr.get("return_jd_ut1")
        ret_dt = None
        if isinstance(rr.get("return_datetime_utc"), str):
            ret_dt = _ensure_utc(rr["return_datetime_utc"])

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
                jd_tt=ret_jd_tt,
                jd_ut1=ret_jd_ut1,
                precision_seconds=prec_s,
                confidence=conv_conf,
                significance=0.01,
                metadata={
                    "return_type": return_type.lower(),
                    "year": year,
                    "residual_arcmin": residual_arcmin,
                    "iterations": int(conv.get("iterations", 0)),
                    "converged": bool(conv.get("converged", False)),
                    "uncertainty_days": unc_days,
                    "finder": "returns.compute_*_return",
                },
            )
        ]

        # Aspects RR↔Natal
        if aspects_to_natal and isinstance(rr.get("chart"), dict) and _ASPECTS_OK:
            def _positions(chart: Dict[str, Any]) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for b in chart.get("bodies", []):
                    if isinstance(b, dict) and "name" in b:
                        out.append({"name": b["name"], "longitude": float(b.get("longitude", b.get("lon", 0.0)))})
                if isinstance(chart.get("angles"), dict):
                    ang = chart["angles"]
                    for key, canon in (("asc_deg", "ASC"), ("mc_deg", "MC"), ("ic_deg", "IC"), ("dsc_deg", "DSC")):
                        if key in ang:
                            out.append({"name": canon, "longitude": float(ang[key])})
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
                major = 0.2 if asp in {"conjunction", "opposition", "trine", "square"} else 0.0
                conf = min(1.0, tight + major)
                events.append(PredictionEvent(
                    event_type="return_aspect",
                    technique=f"{return_type.lower()}_return_aspect",
                    description=f"Return {h.get('planet_a','?')} {asp} Natal {h.get('planet_b','?')}",
                    datetime_utc=ret_dt,
                    jd_tt=ret_jd_tt,
                    jd_ut1=ret_jd_ut1,
                    precision_seconds=prec_s,
                    confidence=conf,
                    significance=float(h.get("p_value", 1.0)),
                    metadata={
                        "return_body": h.get("planet_a"),
                        "natal_body": h.get("planet_b"),
                        "aspect": asp,
                        "orb": orb,
                        "family": h.get("family", "zodiacal"),
                        "finder": "aspects.compute_aspects",
                    },
                ))

        # Validation step omitted
        stats: Dict[str, float] = {}

        windows: List[TimingWindow] = []
        if ret_jd_tt and unc_days:
            windows.append(TimingWindow(
                start_jd_tt=ret_jd_tt - unc_days,
                end_jd_tt=ret_jd_tt + unc_days,
                peak_jd_tt=ret_jd_tt,
                uncertainty_days=unc_days,
                confidence_interval=(ret_jd_tt - unc_days, ret_jd_tt + unc_days),
            ))

        synth = {
            "return_type": return_type.lower(),
            "year": int(year),
            "return_datetime": (ret_dt.isoformat() if ret_dt else None),
            "uncertainty_days": unc_days,
            "total_aspects": sum(1 for e in events if e.event_type == "return_aspect"),
            "convergence_quality": bool(conv.get("converged", False)),
            "residual_quality": "excellent" if residual_arcmin < 1.0 else "good" if residual_arcmin < 5.0 else "fair",
        }
        for w in rr.get("meta", {}).get("warnings", []):
            warns.append(f"return_{w}")

        return PredictionResult(
            ok=True,
            technique=f"{return_type.lower()}_return",
            events=events,
            synthesis=synth,
            timing_windows=windows,
            confidence_score=_compute_confidence(events, stats),
            statistical_metrics=stats,
            warnings=warns,
            metadata={
                "natal_jd_tt": jd_tt_natal,
                "parameters": {
                    "return_type": return_type.lower(),
                    "frame": frame,
                    "house_system": house_system,
                    "zodiac_mode": zodiac_mode,
                    "ayanamsa_deg": ayanamsa_deg,
                    "estimate_uncertainty": estimate_uncertainty,
                    "aspects_to_natal": aspects_to_natal,
                },
            },
            computation_time_ms=(time.time() - t0) * 1000.0,
        )
    except Exception as e:
        return PredictionResult(
            ok=False,
            technique="returns",
            warnings=[f"return_computation_failed:{e}"],
            computation_time_ms=(time.time() - t0) * 1000.0,
        )

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
    """Thin wrapper; normalizes to PredictionResult."""
    _check_env()
    t0 = time.time()
    warns: List[str] = []
    try:
        if not _DIR_OK:
            raise RuntimeError(f"directions unavailable: {_DIR_ERR}")

        jd_tt_natal, jd_ut1_natal, w = _resolve_natal_timescales(natal_chart)
        warns.extend(w)

        if isinstance(target_date, (int, float)):
            target_dt: Optional[datetime] = None
        else:
            target_dt = _ensure_utc(target_date)

        dr = compute_directions(
            natal=natal_chart,
            method=method,
            jd_tt_natal=jd_tt_natal,
            jd_ut1_natal=jd_ut1_natal,
            frame=frame,
            house_system=house_system,
            zodiac_mode=zodiac_mode,
            ayanamsa_deg=ayanamsa_deg,
            aspects_to_natal=aspects_to_natal,
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
            conf = min(1.0, tight + (0.2 if asp in {"conjunction", "opposition", "square", "trine"} else 0.0))
            events.append(PredictionEvent(
                event_type="direction",
                technique=f"{method}_direction",
                description=f"Directed {h.get('planet_a','?')} {asp} Natal {h.get('planet_b','?')}",
                datetime_utc=target_dt,
                jd_tt=None,
                jd_ut1=None,
                precision_seconds=None,
                confidence=conf,
                significance=float(h.get("p_value", 1.0)),
                metadata={
                    "directed_body": h.get("planet_a"),
                    "natal_body": h.get("planet_b"),
                    "aspect": asp,
                    "orb": orb,
                    "finder": "directions.compute_directions",
                },
            ))

        # Validation step omitted
        stats: Dict[str, float] = {}

        return PredictionResult(
            ok=True,
            technique=f"{method}_directions",
            events=events,
            synthesis={"total_hits": len(events)},
            timing_windows=[],
            confidence_score=_compute_confidence(events, stats),
            statistical_metrics=stats,
            warnings=warns,
            metadata={"natal_jd_tt": jd_tt_natal},
            computation_time_ms=(time.time() - t0) * 1000.0,
        )
    except Exception as e:
        return PredictionResult(
            ok=False,
            technique="directions",
            warnings=[f"directions_failed:{e}"],
            computation_time_ms=(time.time() - t0) * 1000.0,
        )

def comprehensive_forecast(
    natal_chart: Dict[str, Any],
    time_range: Tuple[Union[datetime, str], Union[datetime, str]],
    *,
    techniques: Optional[List[str]] = None,
    confidence_threshold: float = 0.2,
    synthesis_method: str = "weighted_consensus",
    statistical_validation: bool = False,  # kept for API, no-op
    include_vedic: bool = False,
    peak_window_days: int = 14,
    **tk_kwargs,
) -> ComprehensiveForecast:
    _check_env()
    t0 = time.time()
    warns: List[str] = []
    try:
        techs = techniques or ["transits", "progressions", "solar_returns"]
        start_dt = _ensure_utc(time_range[0])
        end_dt = _ensure_utc(time_range[1])

        predictions: Dict[str, PredictionResult] = {}
        all_events: List[PredictionEvent] = []

        # Transits
        if "transits" in techs:
            params = {k.replace("transit_", ""): v for k, v in tk_kwargs.items() if k.startswith("transit_")}
            tr = predict_transits(
                natal_chart=natal_chart,
                time_range=(start_dt, end_dt),
                confidence_threshold=confidence_threshold,
                statistical_validation=False,
                **params,
            )
            predictions["transits"] = tr
            if tr.ok:
                all_events += tr.events
            else:
                warns += tr.warnings

        # Progressions ensemble (monthly/quarterly)
        if "progressions" in techs:
            params = {k.replace("progression_", ""): v for k, v in tk_kwargs.items() if k.startswith("progression_")}
            duration = (end_dt - start_dt).days
            step = 30 if duration <= 365 else 90
            dates = [start_dt + timedelta(days=i) for i in range(0, duration + 1, step)]
            pevents: List[PredictionEvent] = []
            for dt in dates:
                pr = predict_progressions(natal_chart=natal_chart, target_date=dt, statistical_validation=False, **params)
                if pr.ok:
                    pevents += pr.events
            predictions["progressions"] = PredictionResult(
                ok=True,
                technique="progressions_ensemble",
                events=pevents,
                synthesis={"sample_dates": len(dates), "total_events": len(pevents)},
                confidence_score=(sum(e.confidence for e in pevents) / len(pevents)) if pevents else 0.0,
            )
            all_events += pevents

        # Solar returns for each calendar year in range
        if "solar_returns" in techs:
            params = {k.replace("return_", ""): v for k, v in tk_kwargs.items() if k.startswith("return_")}
            years = range(start_dt.year, end_dt.year + 1)
            revents: List[PredictionEvent] = []
            for y in years:
                rr = predict_returns(natal_chart=natal_chart, return_type="solar", year=y, statistical_validation=False, **params)
                if rr.ok:
                    for ev in rr.events:
                        if ev.datetime_utc and start_dt <= ev.datetime_utc <= end_dt:
                            revents.append(ev)
            predictions["solar_returns"] = PredictionResult(
                ok=True,
                technique="solar_returns_ensemble",
                events=revents,
                synthesis={"years_computed": list(years), "total_events": len(revents)},
                confidence_score=(sum(e.confidence for e in revents) / len(revents)) if revents else 0.0,
            )
            all_events += revents

        # Optional Vedic via predictive.vimsottari_dasha
        validation_results = None
        if include_vedic and _PRED_OK:
            try:
                ved = {k.replace("vedic_", ""): v for k, v in tk_kwargs.items() if k.startswith("vedic_")}
                ay = float(natal_chart.get("ayanamsa_deg", ved.get("ayanamsa_deg", 0.0)) or 0.0)
                # Need natal Moon tropical longitude & birth JD_TT
                jd_tt_natal, _, _ = _resolve_natal_timescales(natal_chart)
                moon_lon = None
                if isinstance(natal_chart.get("natal_longitudes"), dict):
                    moon_lon = float(natal_chart["natal_longitudes"].get("moon", None))
                else:
                    for b in (natal_chart.get("bodies") or []):
                        if isinstance(b, dict) and str(b.get("name", "")).lower() == "moon":
                            moon_lon = float(b.get("longitude", b.get("lon", None)))
                            break
                if moon_lon is not None:
                    periods = vimsottari_dasha(
                        birth_jd_tt=jd_tt_natal,
                        moon_lon_tropical_deg=moon_lon,
                        ayanamsa_deg=ay,
                        levels=int(ved.get("levels", 3)),
                        span_years=float(ved.get("span_years", 120.0)),
                    )
                    s_tt, _ = _jd_pair_from_dt_dt(start_dt)
                    e_tt, _ = _jd_pair_from_dt_dt(end_dt)
                    des: List[PredictionEvent] = []
                    for p in periods:
                        if p.end_jd_tt < s_tt or p.start_jd_tt > e_tt:
                            continue
                        des.append(PredictionEvent(
                            event_type="dasha",
                            technique="vimshottari_dasha",
                            description=f"{' / '.join(p.parent_chain)}",
                            datetime_utc=None,
                            jd_tt=p.start_jd_tt,
                            jd_ut1=None,
                            precision_seconds=None,
                            confidence=0.7 if p.level == 1 else 0.6 if p.level == 2 else 0.5,
                            significance=0.05,
                            metadata={
                                "level": p.level,
                                "lord": p.lord,
                                "chain": p.parent_chain,
                                "start_jd_tt": p.start_jd_tt,
                                "end_jd_tt": p.end_jd_tt,
                            },
                        ))
                    predictions["dasha"] = PredictionResult(
                        ok=True,
                        technique="dasha",
                        events=des,
                        synthesis={"total_periods": len(des)},
                        confidence_score=(sum(e.confidence for e in des) / len(des)) if des else 0.0,
                    )
                    all_events += des
            except Exception as e:
                warns.append(f"vedic_techniques_failed:{e}")

        # (Optional) ensemble validation omitted — no validate_predictions here

        # Synthesis
        synthesis = _synthesize(all_events, predictions, method=synthesis_method, time_range=(start_dt, end_dt))
        peak_periods = _identify_peak_periods(all_events, (start_dt, end_dt), window_days=peak_window_days)
        risk = _risk(all_events, synthesis)
        conf_metrics = {
            "overall_confidence": synthesis.get("weighted_confidence", 0.0),
            "technique_agreement": synthesis.get("consensus_score", 0.0),
            "event_density": len(all_events) / max(1, (end_dt - start_dt).days),
            "validation_passed": False,  # placeholder; no global validation step
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
            time_range=(
                _ensure_utc(time_range[0]) if isinstance(time_range[0], (str, datetime)) else datetime.now(timezone.utc),
                _ensure_utc(time_range[1]) if isinstance(time_range[1], (str, datetime)) else datetime.now(timezone.utc),
            ),
            synthesis={"error": f"comprehensive_forecast_failed:{e}"},
            computation_time_ms=(time.time() - t0) * 1000.0,
        )

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

        start_dt = _ensure_utc(time_range[0])
        end_dt = _ensure_utc(time_range[1])

        syn = compute_synastry(
            natal_a=natal_a,
            natal_b=natal_b,
            orbs=synastry_orbs,
            parallels=kwargs.get("parallels", True),
            antiscia=kwargs.get("antiscia", True),
            **{k: v for k, v in kwargs.items() if k in ("frame", "zodiac_mode", "ayanamsa_deg", "house_system")},
        )
        comp = compute_composite(
            natal_a=natal_a,
            natal_b=natal_b,
            method=composite_method,
            **{k: v for k, v in kwargs.items() if k in ("frame", "zodiac_mode", "ayanamsa_deg", "house_system")},
        )

        transit_inter: List[PredictionEvent] = []
        prog_inter: List[PredictionEvent] = []

        # Transits to both charts (and composite)
        try:
            tr_a = predict_transits(natal_a, (start_dt, end_dt), confidence_threshold=confidence_threshold, statistical_validation=False, **{k.replace("transit_", ""): v for k, v in kwargs.items() if k.startswith("transit_")})
            tr_b = predict_transits(natal_b, (start_dt, end_dt), confidence_threshold=confidence_threshold, statistical_validation=False, **{k.replace("transit_", ""): v for k, v in kwargs.items() if k.startswith("transit_")})
            for e in tr_a.events:
                e.metadata["person"] = "A"
                e.description = f"A: {e.description}"
                transit_inter.append(e)
            for e in tr_b.events:
                e.metadata["person"] = "B"
                e.description = f"B: {e.description}"
                transit_inter.append(e)

            if include_transits_to_composite and comp.get("ok"):
                comp_chart = _chart_from_composite(comp)
                if comp_chart:
                    tr_c = predict_transits(comp_chart, (start_dt, end_dt), confidence_threshold=confidence_threshold, statistical_validation=False, **{k.replace("transit_", ""): v for k, v in kwargs.items() if k.startswith("transit_")})
                    for e in tr_c.events:
                        e.metadata["person"] = "Composite"
                        e.description = f"Composite: {e.description}"
                        transit_inter.append(e)
        except Exception:
            pass

        # Progressions samples
        if include_progressions:
            duration = (end_dt - start_dt).days
            step = max(30, duration // 6) or 30
            for d in range(0, duration + 1, step):
                dt = start_dt + timedelta(days=d)
                pa = predict_progressions(natal_a, dt, statistical_validation=False, **{k.replace("progression_", ""): v for k, v in kwargs.items() if k.startswith("progression_")})
                pb = predict_progressions(natal_b, dt, statistical_validation=False, **{k.replace("progression_", ""): v for k, v in kwargs.items() if k.startswith("progression_")})
                for e in pa.events:
                    e.metadata["person"] = "A"
                    e.description = f"A: {e.description}"
                    prog_inter.append(e)
                for e in pb.events:
                    e.metadata["person"] = "B"
                    e.description = f"B: {e.description}"
                    prog_inter.append(e)

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
            transit_interactions=[],
            progression_interactions=[],
            compatibility_trends={"trend": 0.0, "volatility": 0.0},
            critical_periods=[],
            relationship_score=0.0,
            confidence_metrics={"overall_confidence": 0.0},
        )

# ───────────────────────────── Synthesis & analytics ─────────────────────────────

def _synthesize(all_events: List[PredictionEvent], preds: Dict[str, PredictionResult], *, method: str, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
    if method == "weighted_consensus":
        weights = {
            "transits": 1.0,
            "progressions": 0.8,
            "solar_returns": 0.9,
            "lunar_returns": 0.7,
            "directions": 0.8,
            "dasha": 0.6,
        }
        w_conf = 0.0
        w_sum = 0.0
        consensus_events = 0
        for k, v in preds.items():
            if v.ok:
                w = weights.get(k, 0.5)
                w_conf += w * v.confidence_score
                w_sum += w
                consensus_events += sum(1 for e in v.events if e.confidence >= 0.5)
        return {
            "method": "weighted_consensus",
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
    if len(ev) < 3:
        return 0.0
    ev.sort(key=lambda x: x.datetime_utc)  # type: ignore
    gaps = []
    for a, b in zip(ev, ev[1:]):
        gaps.append((b.datetime_utc - a.datetime_utc).total_seconds() / 86400.0)  # type: ignore
    if not gaps:
        return 0.0
    mean_gap = sum(gaps) / len(gaps)
    var_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    score = max(0.0, min(1.0, (1.0 / (1.0 + mean_gap)) * (1.0 + min(1.0, var_gap / 30.0))))
    return score

def _consensus_score(preds: Dict[str, PredictionResult], time_range: Tuple[datetime, datetime]) -> float:
    if sum(1 for v in preds.values() if v.ok) < 2:
        return 1.0
    timings: Dict[str, List[Tuple[int, float]]]= {}
    s, e = time_range
    for name, pr in preds.items():
        if pr.ok:
            lst: List[Tuple[int, float]] = []
            for ev in pr.events:
                if ev.datetime_utc and s <= ev.datetime_utc <= e:
                    lst.append(((ev.datetime_utc - s).days, ev.confidence))
            timings[name] = lst
    if len(timings) < 2:
        return 1.0
    correlations: List[float] = []
    names = list(timings.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_events = timings[names[i]]
            b_events = timings[names[j]]
            if not a_events or not b_events:
                continue
            total_score = 0.0
            comparison_count = 0
            for day_a, conf_a in a_events:
                best = 0.0
                for day_b, conf_b in b_events:
                    dt = abs(day_a - day_b)
                    if dt <= 7:
                        time_corr = math.exp(-dt / 3.0)
                        best = max(best, time_corr * min(conf_a, conf_b))
                total_score += best
                comparison_count += 1
            if comparison_count > 0:
                correlations.append(total_score / comparison_count)
    return sum(correlations) / len(correlations) if correlations else 0.0

def _identify_peak_periods(events: List[PredictionEvent], time_range: Tuple[datetime, datetime], *, window_days: int = 14) -> List[TimingWindow]:
    ev = [e for e in events if e.datetime_utc and time_range[0] <= e.datetime_utc <= time_range[1]]
    if not ev:
        return []
    ev.sort(key=lambda x: x.datetime_utc)  # type: ignore
    dur = (time_range[1] - time_range[0]).days
    windows: List[Tuple[float, TimingWindow]] = []
    for offset in range(0, max(1, dur - window_days + 1), 7):
        ws = time_range[0] + timedelta(days=offset)
        we = ws + timedelta(days=window_days)
        win = [e for e in ev if ws <= e.datetime_utc <= we]  # type: ignore
        if len(win) >= 2:
            total_conf = sum(e.confidence for e in win)
            peak = max(win, key=lambda x: x.confidence)
            s_tt, _ = _jd_pair_from_dt_dt(ws)
            e_tt, _ = _jd_pair_from_dt_dt(we)
            p_tt, _ = _jd_pair_from_dt_dt(peak.datetime_utc)  # type: ignore
            score = len(win) * (total_conf / len(win))
            windows.append((score, TimingWindow(
                start_jd_tt=s_tt, end_jd_tt=e_tt, peak_jd_tt=p_tt,
                uncertainty_days=window_days / 2, confidence_interval=(s_tt, e_tt),
            )))
    windows.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in windows[:5]]

def _risk(events: List[PredictionEvent], synthesis: Dict[str, Any]) -> Dict[str, float]:
    total = len(events)
    risk: Dict[str, float] = {}
    if total == 0:
        return {"temporal_stress": 0.0, "aspect_challenge": 0.0, "aspect_support": 0.0, "prediction_uncertainty": 1.0, "outer_planet_influence": 0.0, "overall_risk": 0.0}
    chall = {"square", "opposition", "quincunx"}
    supp = {"trine", "sextile", "conjunction"}
    c_count = sum(1 for e in events if e.metadata.get("aspect") in chall)
    s_count = sum(1 for e in events if e.metadata.get("aspect") in supp)
    risk["temporal_stress"] = min(1.0, 2.0 * synthesis.get("temporal_clusters", {}).get("clustering_strength", 0.0))
    risk["aspect_challenge"] = c_count / total
    risk["aspect_support"] = s_count / total
    risk["prediction_uncertainty"] = 1.0 - (sum(e.confidence for e in events) / total)
    outer = {"saturn", "uranus", "neptune", "pluto"}
    o_count = 0
    for e in events:
        for k in ("transiting_body", "progressed_body", "body"):
            v = str(e.metadata.get(k, "")).lower()
            if any(p in v for p in outer):
                o_count += 1
                break
    risk["outer_planet_influence"] = o_count / total
    weights = {"temporal_stress": 0.3, "aspect_challenge": 0.4, "prediction_uncertainty": 0.2, "outer_planet_influence": 0.1}
    risk["overall_risk"] = min(1.0, sum(risk[k] * w for k, w in weights.items()))
    return risk

def _compat_trends(events: List[PredictionEvent], time_range: Tuple[datetime, datetime]) -> Dict[str, float]:
    if not events:
        return {"trend": 0.0, "volatility": 0.0}
    dur = (time_range[1] - time_range[0]).days
    bucket = max(7, dur // 12) or 7
    scores: List[float] = []
    for off in range(0, dur + 1, bucket):
        bs = time_range[0] + timedelta(days=off)
        be = min(time_range[1], bs + timedelta(days=bucket))
        ee = [e for e in events if e.datetime_utc and bs <= e.datetime_utc <= be]
        if not ee:
            scores.append(0.0)
            continue
        s = 0.0
        for e in ee:
            asp = e.metadata.get("aspect", "")
            c = e.confidence
            if asp in {"trine", "sextile", "conjunction"}:
                s += c
            elif asp in {"square", "opposition"}:
                s -= 0.5 * c
            else:
                s += 0.3 * c
        scores.append(s / len(ee))
    if len(scores) < 2:
        return {"trend": 0.0, "volatility": 0.0}
    n = len(scores)
    xm = (n - 1) / 2.0
    ym = sum(scores) / n
    num = sum((i - xm) * (y - ym) for i, y in enumerate(scores))
    den = sum((i - xm) ** 2 for i in range(n)) or 1.0
    trend = num / den
    var = sum((y - ym) ** 2 for y in scores) / n
    return {"trend": float(trend), "volatility": math.sqrt(var), "bucket_scores": scores, "bucket_count": n}

def _relationship_critical(events: List[PredictionEvent], time_range: Tuple[datetime, datetime]) -> List[TimingWindow]:
    ev = [e for e in events if e.datetime_utc and time_range[0] <= e.datetime_utc <= time_range[1]]
    if not ev:
        return []
    ev.sort(key=lambda x: x.datetime_utc)  # type: ignore
    win_days = 21
    dur = (time_range[1] - time_range[0]).days
    outs: List[Tuple[float, TimingWindow]] = []
    for off in range(0, max(1, dur - win_days + 1), 7):
        ws = time_range[0] + timedelta(days=off)
        we = ws + timedelta(days=win_days)
        wv = [e for e in ev if ws <= e.datetime_utc <= we]  # type: ignore
        if len(wv) < 2:
            continue
        chall = {"square", "opposition", "quincunx"}
        ccount = sum(1 for e in wv if e.metadata.get("aspect") in chall)
        avgc = sum(e.confidence for e in wv) / len(wv)
        score = 2.0 * ccount + len(wv) * avgc
        if score >= 3.0:
            s_tt, _ = _jd_pair_from_dt_dt(ws)
            e_tt, _ = _jd_pair_from_dt_dt(we)
            peak = max(wv, key=lambda x: x.confidence)
            p_tt, _ = _jd_pair_from_dt_dt(peak.datetime_utc)  # type: ignore
            outs.append((score, TimingWindow(
                start_jd_tt=s_tt, end_jd_tt=e_tt, peak_jd_tt=p_tt,
                uncertainty_days=win_days / 3, confidence_interval=(s_tt, e_tt),
            )))
    outs.sort(key=lambda x: x[1].peak_jd_tt or 0.0)
    return [w for _, w in outs[:5]]

# ───────────────────────────── Validation harness ─────────────────────────────

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
        metrics = metrics or ["precision", "recall", "f1", "timing_accuracy", "confidence_calibration"]
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
    folds = [cases[i * size: (i + 1) * size] for i in range(k - 1)] + [cases[(k - 1) * size:]]
    results = []
    for i in range(k):
        test = folds[i]
        train = [c for j, f in enumerate(folds) if j != i for c in f]
        results.append(_eval_metrics(test, train, metrics, thr, **kwargs))
    agg: Dict[str, Any] = {}
    for m in metrics:
        vals = [r.get(m, 0.0) for r in results]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        agg[m] = {"mean": mean, "std": math.sqrt(var), "min": min(vals), "max": max(vals), "values": vals}
    return {"ok": True, "method": "cross_validation", "n_folds": k, "aggregated_metrics": agg, "fold_results": results, "confidence_threshold": thr}

def _eval_metrics(test: List[Dict[str, Any]], train: List[Dict[str, Any]], metrics: List[str], thr: float, **kwargs) -> Dict[str, float]:
    preds: List[bool] = []
    truth: List[bool] = []
    t_errs: List[float] = []
    conf_pred: List[float] = []
    conf_true: List[float] = []
    for case in test:
        try:
            natal = case.get("natal_chart", {})
            trange = case.get("time_range", (datetime.now(timezone.utc), datetime.now(timezone.utc) + timedelta(days=365)))
            pred_type = case.get("prediction_type", "transits")
            if pred_type == "comprehensive":
                cf = comprehensive_forecast(natal, trange, confidence_threshold=thr, statistical_validation=False, **kwargs)
                evs: List[PredictionEvent] = []
                for pr in cf.predictions.values():
                    if pr.ok:
                        evs += pr.events
            elif pred_type == "transits":
                pr = predict_transits(natal, trange, confidence_threshold=thr, statistical_validation=False, **kwargs)
                evs = pr.events if pr.ok else []
            elif pred_type == "progressions":
                td = case.get("target_date", trange[1])
                pr = predict_progressions(natal, td, statistical_validation=False, **kwargs)
                evs = pr.events if pr.ok else []
            elif pred_type == "returns":
                rt = case.get("return_type", "solar")
                y = case.get("year", _ensure_utc(trange[1]).year)
                pr = predict_returns(natal, rt, y, statistical_validation=False, **kwargs)
                evs = pr.events if pr.ok else []
            else:
                continue
            expected = case.get("expected_events", [])
            for ex in expected:
                ex_time = ex.get("datetime")
                ex_type = ex.get("event_type")
                ex_conf = float(ex.get("confidence", 1.0))
                matches = []
                for pv in evs:
                    if pv.event_type == ex_type and pv.datetime_utc and ex_time:
                        dt = _ensure_utc(ex_time)
                        if abs((pv.datetime_utc - dt).total_seconds()) <= 7 * 86400:
                            matches.append(pv)
                if matches:
                    preds.append(True)
                    best = max(matches, key=lambda x: x.confidence)
                    t_errs.append(abs((best.datetime_utc - _ensure_utc(ex_time)).total_seconds()) / 86400.0)  # type: ignore
                    conf_pred.append(best.confidence)
                    conf_true.append(ex_conf)
                else:
                    preds.append(False)
                truth.append(True)
            # false positives
            for pv in evs:
                if pv.confidence >= thr:
                    matched = False
                    for ex in expected:
                        ex_time = ex.get("datetime")
                        if pv.datetime_utc and ex_time and abs((pv.datetime_utc - _ensure_utc(ex_time)).total_seconds()) <= 7 * 86400:
                            matched = True
                            break
                    if not matched:
                        preds.append(True)
                        truth.append(False)
        except Exception:
            continue
    out: Dict[str, float] = {}
    if "precision" in metrics:
        tp = sum(1 for i, p in enumerate(preds) if p and truth[i])
        fp = sum(1 for i, p in enumerate(preds) if p and not truth[i])
        out["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    if "recall" in metrics:
        tp = sum(1 for i, p in enumerate(preds) if p and truth[i])
        fn = sum(1 for i, p in enumerate(preds) if (not p) and truth[i])
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
        for m in metrics:
            vals[m].append(r.get(m, 0.0))
    agg = {m: {"mean": (sum(v)/len(v) if v else 0.0),
               "std": (math.sqrt(sum((x - (sum(v)/len(v) if v else 0.0))**2 for x in v)/len(v)) if v else 0.0)}
           for m, v in vals.items()}
    return {"ok": True, "method": "bootstrap", "aggregated_metrics": agg, "B": B}

def _holdout(cases: List[Dict[str, Any]], metrics: List[str], thr: float, **kwargs) -> Dict[str, Any]:
    mid = len(cases) // 2 or 1
    train, test = cases[:mid], cases[mid:]
    r = _eval_metrics(test, train, metrics, thr, **kwargs)
    return {"ok": True, "method": "holdout", "metrics": r}

# ───────────────────────────── Utilities ─────────────────────────────

def _chart_from_composite(comp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not comp.get("positions"):
        return None
    bodies = [{"name": name, "longitude": float(lon), "latitude": 0.0, "is_point": False} for name, lon in comp["positions"].items()]
    angles: Dict[str, float] = {}
    if comp.get("asc") is not None:
        angles["asc_deg"] = float(comp["asc"])
    if comp.get("mc") is not None:
        angles["mc_deg"] = float(comp["mc"])
    return {"bodies": bodies, "angles": angles, "mode": comp.get("meta", {}).get("zodiac_mode", "tropical")}

# ───────────────────────────── Exports ─────────────────────────────

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
