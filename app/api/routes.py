# app/api/routes.py
"""
AstroApp — Canonical API Routes
- Timescales (ERFA-aligned)
- Chart + Houses
- Predictions (legacy + v2 prediction engine)
- Ephemeris
- Predictive toolkit
- Progressions (secondary • minor • tertiary)
- Returns (solar • lunar • planetary)
- Parans (local horizon events)
- Synastry & Composite (relationship analysis)
- Relocation & Astrocartography
- Directions (solar arc)
- Ops: /api/health, /api/config, /api/openapi, /__debug/routes

Notes:
- Topocentric ephemeris honored via either center:"topocentric" or topocentric:true
- Ephemeris responses include adapter meta (with meta.topocentric)
- Adapter/kernel meta is bubbled up to API responses so dev tools can verify DE440s/DE421 quickly.
- V2 prediction engine provides comprehensive forecasting capabilities
"""

from __future__ import annotations

import json
import logging
import os
import re
import math
import inspect
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request, current_app
from zoneinfo import ZoneInfo

from app.version import VERSION
from app.utils.config import load_config
from app.utils.hc import flag_predictions
from app.utils.ratelimit import rate_limit, client_key, endpoint_key
from app.core.validators import (
    ValidationError,
    parse_chart_payload,
    parse_prediction_payload,
    parse_rectification_payload,  # noqa: F401 (kept for completeness)
    parse_ephemeris_payload,
    parse_frame,
    parse_latlon,
    parse_progressions_payload,
    parse_returns_payload,
    parse_parans_payload,
    parse_synastry_payload,
    parse_composite_payload,
    parse_synastry_report_payload,
    parse_relocation_payload,
    parse_astrocartography_payload,
    parse_directions_payload,
)

# Timescales core (guarded)
try:
    from app.core.timescales import build_timescales, TimeScales
    _TIMESCALES_OK = True
    _TIMESCALES_ERR: Optional[Exception] = None
except Exception as e:
    _TIMESCALES_OK = False
    _TIMESCALES_ERR = e
    build_timescales = None  # type: ignore
    TimeScales = None  # type: ignore

# Prediction Engine (NEW - comprehensive prediction system)
try:
    from app.core.prediction import (
        predict_transits,
        predict_progressions,
        predict_returns,
        predict_directions,
        comprehensive_forecast,
        relationship_forecast,
        validate_prediction_model,
        PredictionEvent,
        PredictionResult,
        ComprehensiveForecast,
        RelationshipForecast,
        TimingWindow,
    )
    _PREDICTION_ENGINE_OK = True
    _PREDICTION_ENGINE_ERR: Optional[Exception] = None
except Exception as e:
    _PREDICTION_ENGINE_OK = False
    _PREDICTION_ENGINE_ERR = e
    # Placeholders to avoid NameError downstream
    predict_transits = None  # type: ignore
    predict_progressions = None  # type: ignore
    predict_returns = None  # type: ignore
    predict_directions = None  # type: ignore
    comprehensive_forecast = None  # type: ignore
    relationship_forecast = None  # type: ignore
    validate_prediction_model = None  # type: ignore
    PredictionEvent = None  # type: ignore
    PredictionResult = None  # type: ignore
    ComprehensiveForecast = None  # type: ignore
    RelationshipForecast = None  # type: ignore
    TimingWindow = None  # type: ignore

# Progressions core (optional low-level helper; not required if v2 engine is used)
try:
    from app.core.progressions import compute_progressions
except Exception:
    compute_progressions = None  # type: ignore

# Returns core (prefer 'returns.py', fallback to single-file 'return.py')
_returns_mod = None
_RETURNS_IMPORT_ERROR: Optional[Exception] = None
try:
    from app.core import returns as _returns_mod  # app/core/returns.py (recommended)
except Exception as _e1:
    _RETURNS_IMPORT_ERROR = _e1
    try:
        import importlib.util as _importlib_util
        _base_dir = os.path.dirname(os.path.dirname(__file__))  # app/
        _path_return = os.path.join(_base_dir, "core", "return.py")
        if os.path.exists(_path_return):
            _spec = _importlib_util.spec_from_file_location("app.core._return_mod", _path_return)
            if _spec and _spec.loader:
                _mod = _importlib_util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
                _returns_mod = _mod
    except Exception as _e2:
        _RETURNS_IMPORT_ERROR = _e2  # keep last error for diagnostics

# Parans core (optional import guard)
_parans_compute = None  # function when available
_PARANS_IMPORT_ERROR: Optional[Exception] = None
try:
    from app.core.paran import compute_parans as _parans_compute
except Exception as _e:
    _PARANS_IMPORT_ERROR = _e
    _parans_compute = None  # type: ignore

# Synastry core (optional import guard)
_synastry_compute = None
_composite_compute = None
_synastry_report_compute = None
_SYNASTRY_IMPORT_ERROR: Optional[Exception] = None
try:
    from app.core.synastry import (
        compute_synastry as _synastry_compute,
        compute_composite as _composite_compute,
        synastry_report as _synastry_report_compute,
    )
    _SYN_OK = True
except Exception as _e:
    _SYNASTRY_IMPORT_ERROR = _e
    _synastry_compute = None
    _composite_compute = None
    _synastry_report_compute = None
    _SYN_OK = False

# Relocation core (optional import guard)
_compute_relocated = None
_compute_astrocartography = None
_RELOCATION_IMPORT_ERROR: Optional[Exception] = None
try:
    from app.core.relocation import (
        compute_relocated as _compute_relocated,
        compute_astrocartography as _compute_astrocartography,
    )
except Exception as _e:
    _RELOCATION_IMPORT_ERROR = _e
    _compute_relocated = None
    _compute_astrocartography = None

# Directions core (optional low-level helper; v2 engine exposes predict_directions)
_compute_directions = None
_DIRECTIONS_IMPORT_ERROR: Optional[Exception] = None
try:
    from app.core.directions import compute_directions as _compute_directions
    _DIR_OK = True
except Exception as _e:
    _DIRECTIONS_IMPORT_ERROR = _e
    _compute_directions = None
    _DIR_OK = False

log = logging.getLogger(__name__)
api = Blueprint("api", __name__)

DEBUG_VERBOSE = os.getenv("ASTRO_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "on")
ARCSEC_TOL = float(os.getenv("ASTRO_ASC_TOL_ARCSEC", "3.6"))  # 0.001°

# per-endpoint rate-limit caps (calls per minute, env-overridable)
_RL = lambda k, d: int(os.getenv(k, str(d)))
RL_TIMESCALES   = _RL("ASTRO_RL_TIMESCALES_PER_MIN",   60)
RL_CALCULATE    = _RL("ASTRO_RL_CALCULATE_PER_MIN",    24)
RL_REPORT       = _RL("ASTRO_RL_REPORT_PER_MIN",       12)
RL_ASPECTS      = _RL("ASTRO_RL_ASPECTS_PER_MIN",      18)
RL_EPHEM        = _RL("ASTRO_RL_EPHEM_PER_MIN",        30)
RL_PREDICTIONS  = _RL("ASTRO_RL_PREDICTIONS_PER_MIN",   6)
RL_PREDICTIVE   = _RL("ASTRO_RL_PREDICTIVE_PER_MIN",   12)
RL_DEBUG        = _RL("ASTRO_RL_DEBUG_PER_MIN",         6)
RL_PROGRESSIONS = _RL("ASTRO_RL_PROGRESSIONS_PER_MIN", 12)
RL_RETURNS      = _RL("ASTRO_RL_RETURNS_PER_MIN",      12)
RL_PARANS       = _RL("ASTRO_RL_PARANS_PER_MIN",       12)
RL_SYNASTRY     = _RL("ASTRO_RL_SYNASTRY_PER_MIN",     6)
RL_COMPOSITE    = _RL("ASTRO_RL_COMPOSITE_PER_MIN",    8)
RL_RELOCATION   = _RL("ASTRO_RL_RELOCATION_PER_MIN",   10)
RL_ASTROCARTOGRAPHY = _RL("ASTRO_RL_ASTROCARTOGRAPHY_PER_MIN", 4)
RL_DIRECTIONS   = _RL("ASTRO_RL_DIRECTIONS_PER_MIN",    8)
RL_PREDICTION_FORECAST = _RL("ASTRO_RL_PREDICTION_FORECAST_PER_MIN", 4)
RL_PREDICTION_TRANSITS = _RL("ASTRO_RL_PREDICTION_TRANSITS_PER_MIN", 8)
RL_PREDICTION_PROGRESSIONS = _RL("ASTRO_RL_PREDICTION_PROGRESSIONS_PER_MIN", 6)
RL_PREDICTION_RETURNS = _RL("ASTRO_RL_PREDICTION_RETURNS_PER_MIN", 6)
RL_PREDICTION_DIRECTIONS = _RL("ASTRO_RL_PREDICTION_DIRECTIONS_PER_MIN", 6)
RL_PREDICTION_RELATIONSHIP = _RL("ASTRO_RL_PREDICTION_RELATIONSHIP_PER_MIN", 3)
RL_PREDICTION_VALIDATION = _RL("ASTRO_RL_PREDICTION_VALIDATION_PER_MIN", 2)


# ───────────────────────── helpers ─────────────────────────
def _wrap360(x: float) -> float:
    try:
        v = float(x) % 360.0
        return 0.0 if abs(v) < 1e-12 else v
    except Exception:
        return x

def _shortest_delta_deg(a2: float, a1: float) -> float:
    d = (float(a2) - float(a1) + 540.0) % 360.0 - 180.0
    return -180.0 if d == 180.0 else d

def _delta_arcsec(a: float, b: float) -> float:
    return abs(_shortest_delta_deg(a, b)) * 3600.0

def _json_error(code: str, details: Any = None, http: int = 400):
    out: Dict[str, Any] = {"ok": False, "error": code}
    if details is not None:
        out["details"] = details
    return jsonify(out), http

def _split_jd(jd: float) -> tuple[float, float]:
    d = int(jd // 1)
    return float(d), float(jd - d)

def _sind(a: float) -> float:
    import math as _m
    return _m.sin(_m.radians(a))

def _cosd(a: float) -> float:
    import math as _m
    return _m.cos(_m.radians(a))

def _atan2d(y: float, x: float) -> float:
    import math as _m
    if abs(x) < 1e-18 and abs(y) < 1e-18:
        raise ValueError("atan2(0,0) undefined")
    return _wrap360(_m.degrees(_m.atan2(y, x)))

def _gast_deg(jd_ut1: float, jd_tt: float) -> float:
    try:
        import erfa  # type: ignore
        d1u, d2u = _split_jd(jd_ut1)
        d1t, d2t = _split_jd(jd_tt)
        gst_rad = erfa.gst06a(d1u, d2u, d1t, d2t)
        import math as _m
        return _wrap360(_m.degrees(gst_rad))
    except Exception:
        import math as _m
        T = (float(jd_ut1) - 2451545.0) / 36525.0
        theta = (
            280.46061837
            + 360.98564736629 * (float(jd_ut1) - 2451545.0)
            + 0.000387933 * (T**2)
            - (T**3) / 38710000.0
        )
        return _wrap360(theta)

def _true_obliquity_deg(jd_tt: float) -> float:
    try:
        import erfa  # type: ignore
        d1, d2 = _split_jd(jd_tt)
        eps0 = erfa.obl06(d1, d2)
        _dpsi, deps = erfa.nut06a(d1, d2)
        import math as _m
        return _m.degrees(eps0 + deps)
    except Exception:
        import math as _m
        T = (float(jd_tt) - 2451545.0) / 36525.0
        eps_arcsec = 84381.448 - 46.8150 * T - 0.00059 * (T**2) + 0.001813 * (T**3)
        return eps_arcsec / 3600.0

def _ramc_deg(jd_ut1: float, jd_tt: float, lon_east_deg: float) -> float:
    return _wrap360(_gast_deg(jd_ut1, jd_tt) + float(lon_east_deg))

def _mc_from_ramc(ramc: float, eps: float) -> float:
    return _atan2d(_sind(ramc) * _cosd(eps), _cosd(ramc))

def _asc_from_phi_ramc(phi: float, ramc: float, eps: float) -> float:
    import math as _m
    def _acotd(x: float) -> float:
        return _wrap360(_m.degrees(_m.atan2(1.0, x)))
    num = -((_m.tan(_m.radians(phi)) * _sind(eps)) + (_sind(ramc) * _cosd(eps)))
    den = _cosd(ramc)
    den = den if abs(den) > 1e-15 else _m.copysign(1e-15, den if den != 0 else 1.0)
    return _acotd(num / den)

def _recompute_angles_exact(
    *,
    jd_ut1: float,
    jd_tt: float,
    latitude: Optional[float],
    longitude_east: Optional[float],
    mode: str,
    ayanamsa_deg: Optional[float],
) -> Optional[Dict[str, float]]:
    if latitude is None or longitude_east is None:
        return None
    eps = _true_obliquity_deg(jd_tt)
    ramc = _ramc_deg(jd_ut1, jd_tt, float(longitude_east))
    mc = _mc_from_ramc(ramc, eps)
    asc = _asc_from_phi_ramc(float(latitude), ramc, eps)
    if (mode or "tropical").lower() == "sidereal" and isinstance(ayanamsa_deg, (int, float)):
        asc = _wrap360(asc - float(ayanamsa_deg))
        mc = _wrap360(mc - float(ayanamsa_deg))
    return {"asc_deg": asc, "mc_deg": mc}

# ─────────────────────────── Helpers & engine guard ───────────────────────────

# Timescales imported without a guard above; mark OK since import succeeded.
_TIMESCALES_OK = True

def _engine_required() -> None:
    """Ensure the v2 prediction engine is loaded before serving endpoints."""
    if not _PREDICTION_ENGINE_OK:
        raise RuntimeError(f"Prediction engine unavailable: {_PREDICTION_ENGINE_ERR}")

def _as_json(obj: Any) -> Any:
    """
    JSON-safe serializer that understands our v2 dataclasses
    (PredictionResult, ComprehensiveForecast, etc.).
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_as_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _as_json(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    return obj

@api.route("/v2/health", methods=["GET"])
def health_v2():
    status = {
        "version": VERSION,
        "prediction_engine": _PREDICTION_ENGINE_OK,
        "timescales": _TIMESCALES_OK,
        "synastry": _SYN_OK,
        "directions_lowlevel": _DIR_OK,
        "returns_module": bool(_returns_mod),
        "parans_available": _parans_compute is not None,
        "relocation_available": (_compute_relocated is not None) and (_compute_astrocartography is not None),
        "debug_verbose": DEBUG_VERBOSE,
    }
    return jsonify(status), 200


# ───────────────────────── timescales adapter ─────────────────────────
def _compute_timescales_from_local(
    date_str: str,
    time_str: str,
    tz_name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calls build_timescales(date, time, tz, dut1_seconds) and adapts result.

    DUT1 precedence:
      1) payload["dut1_seconds"] (preferred)
      2) payload["dut1"]         (legacy)
      3) env ASTRO_DUT1_BROADCAST / ASTRO_DUT1 (default "0.0")
    """
    # tz validation
    try:
        _ = ZoneInfo(tz_name)
    except Exception:
        raise ValidationError([{
            "loc": ["tz"],
            "msg": "must be a valid IANA zone like 'Asia/Kolkata'",
            "type": "value_error",
        }])

    def _env_dut1() -> float:
        env_val = os.getenv("ASTRO_DUT1_BROADCAST", os.getenv("ASTRO_DUT1", "0.0"))
        try:
            return float(env_val)
        except Exception:
            raise ValidationError([{
                "loc": ["dut1"],
                "msg": "environment DUT1 must be a valid number",
                "type": "value_error",
            }])

    # tolerant DUT1 parsing (treat "", None, "null", "undefined" as "not provided")
    def _parse_payload_dut1(p: Optional[Dict[str, Any]]) -> float:
        if not isinstance(p, dict):
            return _env_dut1()
        has_primary = "dut1_seconds" in p
        has_legacy = "dut1" in p
        if not (has_primary or has_legacy):
            return _env_dut1()
        key = "dut1_seconds" if has_primary else "dut1"
        raw = p.get(key)
        if raw in (None, "", "null", "undefined"):
            return _env_dut1()
        try:
            return float(raw)
        except Exception:
            raise ValidationError([{
                "loc": [key],
                "msg": "must be a number (seconds)",
                "type": "value_error",
            }])

    dut1_seconds = _parse_payload_dut1(payload)

    try:
        ts: TimeScales = build_timescales(date_str, time_str, tz_name, dut1_seconds)
    except ValueError as e:
        msg = str(e)
        if "DUT1" in msg.upper():
            raise ValidationError([{"loc": ["dut1"], "msg": msg, "type": "value_error"}])
        if "1960" in msg or "pre-1960" in msg.lower():
            raise ValidationError([{"loc": ["date"], "msg": msg, "type": "value_error"}])
        raise ValidationError([{"loc": ["timescales"], "msg": msg, "type": "value_error"}])

    return {
        "jd_utc": float(ts.jd_utc),
        "jd_tt": float(ts.jd_tt),
        "jd_ut1": float(ts.jd_ut1),
        "delta_t": float(ts.delta_t),
        "delta_at": float(ts.dat),
        "dut1": float(ts.dut1),
        "timezone": tz_name,
        "tz_offset_seconds": int(ts.tz_offset_seconds),
        "warnings": list(ts.warnings),
    }

# ───────────────────────── chart / houses engines ─────────────────────────
_compute_chart = None  # type: ignore
_CHART_ENGINE_NAME: Optional[str] = None
try:  # pragma: no cover
    from app.core.astronomy import compute_chart as _compute_chart  # type: ignore
    _CHART_ENGINE_NAME = "app.core.astronomy.compute_chart"
except Exception as e1:  # pragma: no cover
    try:
        from app.core.chart import compute_chart as _compute_chart  # <-- proper fallback
        _CHART_ENGINE_NAME = "app.core.chart.compute_chart"
        log.warning("Primary astronomy.compute_chart missing; fallback chart.compute_chart in use. err=%r", e1)
    except Exception as e2:
        _compute_chart = None  # type: ignore
        _CHART_ENGINE_NAME = None
        log.error("No compute_chart available: astronomy failed=%r, chart failed=%r", e1, e2)

_HOUSES_KIND = "policy"
_can_sys = None
try:
    from app.core.house import (  # type: ignore
        compute_houses_with_policy as _houses_fn,
        canonicalize_system as _can_sys,
        POLAR_SOFT_LIMIT_DEG,
        POLAR_HARD_LIMIT_DEG,
    )
except Exception:
    try:
        from app.core.houses_advanced import compute_house_system as _houses_fn  # type: ignore
        _HOUSES_KIND = "legacy"
        POLAR_SOFT_LIMIT_DEG = float(os.getenv("ASTRO_POLAR_SOFT_LAT", "66.0"))
        POLAR_HARD_LIMIT_DEG = float(os.getenv("ASTRO_POLAR_HARD_LAT", "80.0"))
    except Exception as e:
        _houses_fn = None  # type: ignore
        _HOUSES_KIND = "unavailable"
        POLAR_SOFT_LIMIT_DEG = float(os.getenv("ASTRO_POLAR_SOFT_LAT", "66.0"))
        POLAR_HARD_LIMIT_DEG = float(os.getenv("ASTRO_POLAR_HARD_LAT", "80.0"))
        log.error("No house engine available: %r", e)

def _sig_accepts(fn, *names: str) -> Dict[str, bool]:
    try:
        params = fn.__signature__.parameters  # type: ignore[attr-defined]
    except Exception:
        params = inspect.signature(fn).parameters
    return {n: (n in params) for n in names}

# ---------- adapter/kernel meta snapshot (for dev tools visibility) ----------
def _snapshot_ephemeris_meta(chart_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return a small dict with adapter/kernel information WITHOUT forcing a kernel
    load. Values populate lazily once a kernel is actually loaded.
    """
    info: Dict[str, Any] = {}
    try:
        from app.core import ephemeris_adapter as ea  # late import by design

        # Prefer diagnostics if available (it will also reflect SPICE status)
        try:
            diag = ea.ephemeris_diagnostics()
        except Exception:
            diag = None

        # Kernel short name
        try:
            info["kernel"] = ea.current_kernel_name()
        except Exception:
            pass

        # Kernels list (basenames) if diagnostics provided
        if isinstance(diag, dict) and isinstance(diag.get("kernels"), list):
            info["kernels"] = list(diag["kernels"])

        # Path + coverage (may be None until the kernel is loaded)
        try:
            info["ephemeris_path"] = ea.current_kernel_path()
        except Exception:
            pass
        try:
            info["ephemeris_coverage_jd"] = getattr(ea, "KERNEL_COVERAGE_JD", None)
        except Exception:
            pass

        # Tag the source engine that produced the chart data
        if isinstance(chart_meta, dict):
            src = chart_meta.get("source") or chart_meta.get("engine")
            if src:
                info["source"] = src
        if "source" not in info and _CHART_ENGINE_NAME:
            info["source"] = _CHART_ENGINE_NAME
    except Exception:
        # If adapter import fails, we still return what we can
        if _CHART_ENGINE_NAME:
            info["source"] = _CHART_ENGINE_NAME
    return info


def _call_compute_chart(payload: Dict[str, Any], ts: Dict[str, Any]) -> Dict[str, Any]:
    if _compute_chart is None:
        raise RuntimeError("chart_engine_unavailable")

    param_names = set(inspect.signature(_compute_chart).parameters.keys())

    def _normalize_payload_for_engine(p: Dict[str, Any]) -> Dict[str, Any]:
        q = dict(p)
        q.setdefault("center", "topocentric" if bool(q.get("topocentric")) else "geocentric")
        q.setdefault("frame", "ecliptic-of-date")
        bodies = q.get("bodies")
        if isinstance(bodies, list):
            if all(isinstance(b, str) for b in bodies):
                q.setdefault("names", bodies)
            elif all(isinstance(b, dict) for b in bodies):
                q.setdefault("names", [str(b.get("name") or b.get("body") or "").strip() for b in bodies])
        if "ayanamsa" in q and isinstance(q["ayanamsa"], str):
            q["ayanamsa"] = q["ayanamsa"].strip().lower()
        q.setdefault("jd_ut", ts["jd_utc"])
        q.setdefault("jd_tt", ts["jd_tt"])
        q.setdefault("jd_ut1", ts["jd_ut1"])
        q.setdefault("timescales", ts)
        return q

    if "payload" in param_names:
        payload2 = _normalize_payload_for_engine(payload)
        try:
            chart = _compute_chart(payload2)  # type: ignore[arg-type]
        except Exception as e:
            chart = {
                "mode": payload.get("mode"),
                "jd_ut": ts["jd_utc"], "jd_tt": ts["jd_tt"], "jd_ut1": ts["jd_ut1"],
                "meta": {"engine": _CHART_ENGINE_NAME, "warnings": ["chart_failed"]},
                "error": str(e) if DEBUG_VERBOSE else "chart_failed",
            }
    else:
        kwargs: Dict[str, Any] = {}
        if "date" in payload and (k := {"date","date_str","date_s"}.intersection(param_names)):
            kwargs[list(k)[0]] = payload.get("date")
        if "time" in payload and (k := {"time","time_str","time_s"}.intersection(param_names)):
            kwargs[list(k)[0]] = payload.get("time")
        if "latitude" in payload and (k := {"latitude","lat"}.intersection(param_names)):
            kwargs[list(k)[0]] = payload["latitude"]
        if "longitude" in payload and (k := {"longitude","lon"}.intersection(param_names)):
            kwargs[list(k)[0]] = payload["longitude"]
        tz_name = payload.get("place_tz") or payload.get("timezone")
        if tz_name and (k := {"place_tz","timezone","tz_name"}.intersection(param_names)):
            kwargs[list(k)[0]] = tz_name
        if "mode" in payload and (k := {"mode","system"}.intersection(param_names)):
            kwargs[list(k)[0]] = payload["mode"]
        if "ayanamsa" in payload and (k := {"ayanamsa","ayanamsha","aya"}.intersection(param_names)):
            kwargs[list(k)[0]] = payload["ayanamsa"]
        if "topocentric" in payload and (k := {"topocentric","observer_topocentric"}.intersection(param_names)):
            kwargs[list(k)[0]] = bool(payload["topocentric"])
        if ("elevation_m" in payload or "elev_m" in payload) and (k := {"elevation_m","elevation"}.intersection(param_names)):
            kwargs[list(k)[0]] = payload.get("elevation_m", payload.get("elev_m"))
        if "bodies" in payload and (k := {"bodies","names","planets"}.intersection(param_names)):
            kwargs[list(k)[0]] = payload["bodies"]
        if "frame" in payload and "frame" in param_names:
            kwargs["frame"] = payload["frame"]
        if "timescales" in param_names:
            kwargs["timescales"] = ts
        if "jd_tt" in param_names:
            kwargs["jd_tt"] = ts["jd_tt"]
        if "jd_ut1" in param_names:
            kwargs["jd_ut1"] = ts["jd_ut1"]
        if "jd_utc" in param_names:
            kwargs["jd_utc"] = ts["jd_utc"]
        elif "jd_ut" in param_names:
            kwargs["jd_ut"] = ts["jd_utc"]

        try:
            chart = _compute_chart(**kwargs)
        except Exception as e:
            chart = {
                "mode": payload.get("mode"),
                "jd_ut": ts["jd_utc"], "jd_tt": ts["jd_tt"], "jd_ut1": ts["jd_ut1"],
                "meta": {"engine": _CHART_ENGINE_NAME, "warnings": ["chart_failed"]},
                "error": str(e) if DEBUG_VERBOSE else "chart_failed",
            }

    chart = chart or {}
    chart.setdefault("meta", {})
    chart["meta"]["engine"] = _CHART_ENGINE_NAME or "unknown"
    chart["jd_ut"] = ts["jd_utc"]
    chart["jd_tt"] = ts["jd_tt"]
    chart["jd_ut1"] = ts["jd_ut1"]
    if "mode" not in chart and "mode" in payload:
        chart["mode"] = payload["mode"]
    return chart

def _sig_accepts_houses() -> Dict[str, bool]:
    if _houses_fn is None:
        return {}
    return _sig_accepts(
        _houses_fn, "lat", "lon", "latitude", "longitude",
        "system", "requested_house_system", "house_system",
        "mode", "jd_ut", "jd_tt", "jd_ut1", "diagnostics", "validation",
    )

def _call_compute_houses(payload: Dict[str, Any], ts: Dict[str, Any]) -> Any:
    if _houses_fn is None:
        raise RuntimeError("houses_engine_unavailable")

    acc = _sig_accepts_houses()
    lat_raw = payload.get("latitude")
    lon_raw = payload.get("longitude")
    if not isinstance(lat_raw, (int, float)) or not isinstance(lon_raw, (int, float)):
        raise ValueError("latitude and longitude are required (finite numbers) to compute houses")

    lat = float(lat_raw)
    lon = float(lon_raw)

    requested_system_raw = (payload.get("house_system") or "").strip()
    requested_system = (
        _can_sys(requested_system_raw) if (_can_sys and requested_system_raw)
        else (requested_system_raw.lower() or None)
    )

    kwargs: Dict[str, Any] = {}

    if acc.get("lat"):
        kwargs["lat"] = lat
    elif acc.get("latitude"):
        kwargs["latitude"] = lat
    else:
        kwargs["lat"] = lat

    if acc.get("lon"):
        kwargs["lon"] = lon
    elif acc.get("longitude"):
        kwargs["longitude"] = lon
    else:
        kwargs["lon"] = lon

    if requested_system:
        if acc.get("system"):
            kwargs["system"] = requested_system
        elif acc.get("requested_house_system"):
            kwargs["requested_house_system"] = requested_system
        elif acc.get("house_system"):
            kwargs["house_system"] = requested_system

    if acc.get("jd_tt"):
        kwargs["jd_tt"] = ts["jd_tt"]
    if acc.get("jd_ut1"):
        kwargs["jd_ut1"] = ts["jd_ut1"]
    if acc.get("jd_ut") and "jd_tt" not in kwargs and "jd_ut1" not in kwargs:
        kwargs["jd_ut"] = ts["jd_utc"]

    if acc.get("diagnostics"):
        kwargs["diagnostics"] = True
    if acc.get("validation"):
        kwargs["validation"] = True

    try:
        result = _houses_fn(**kwargs)
        return result
    except Exception as e:
        log.error("House calculation failed: %s: %s", type(e).__name__, e)
        raise

def _call_compute_aspects(payload: Dict[str, Any], chart: Dict[str, Any], houses: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a positions dict from chart (and optional houses) and delegate to the
    HTTP-facing aspects adapter (run_aspects_api). All angles are normalized to [0, 360).
    """
    try:
        from app.core.aspects import run_aspects_api
    except ImportError:
        raise RuntimeError("aspects engine not available")

    # --- Collect ecliptic longitudes (deg) from chart bodies/points ---
    positions: Dict[str, float] = {}
    for src in ("bodies", "points"):
        for rec in (chart.get(src) or []):
            if not isinstance(rec, dict):
                continue
            name = rec.get("name")
            L = rec.get("longitude_deg")
            if isinstance(name, str) and isinstance(L, (int, float)):
                positions[name] = _wrap360(float(L))

    # --- Optionally add house cusps and angles if provided ---
    if isinstance(houses, dict):
        cusps = houses.get("cusps") or houses.get("cusps_deg")
        if isinstance(cusps, list):
            for i, c in enumerate(cusps):
                if isinstance(c, (int, float)):
                    positions[f"House {i+1}"] = _wrap360(float(c))

        # Add angles (Asc/MC) when present
        asc_v = houses.get("asc") if isinstance(houses.get("asc"), (int, float)) else houses.get("asc_deg")
        mc_v  = houses.get("mc")  if isinstance(houses.get("mc"), (int, float))  else houses.get("mc_deg")
        if isinstance(asc_v, (int, float)):
            positions["Ascendant"] = _wrap360(float(asc_v))
        if isinstance(mc_v, (int, float)):
            positions["Midheaven"] = _wrap360(float(mc_v))

    # --- Optional declinations for parallels/contra-parallels ---
    decls: Dict[str, float] = {}
    for src in ("bodies", "points"):
        for rec in (chart.get(src) or []):
            if not isinstance(rec, dict):
                continue
            name = rec.get("name")
            d = rec.get("declination_deg")
            if isinstance(name, str) and isinstance(d, (int, float)):
                decls[name] = float(d)

    # --- Prepare adapter config ---
    aspects_config: Dict[str, Any] = {
        "positions": positions,
        "orbs": payload.get("orbs") or None,
        "aspects": payload.get("aspects") or None,
        "mode": (payload.get("mode") or "tropical"),
    }
    if decls:
        aspects_config["declinations"] = decls  # only enables parallels if requested in `aspects`

    # --- Run aspects engine (adapter handles empty positions gracefully) ---
    return run_aspects_api(**aspects_config)

def _extract_ayanamsa_from_chart(chart: Dict[str, Any]) -> Optional[float]:
    if not isinstance(chart, dict):
        return None
    meta = chart.get("meta") or {}
    ay = meta.get("ayanamsa_deg")
    if isinstance(ay, (int, float)):
        return float(ay)
    ay2 = chart.get("ayanamsa_deg")
    return float(ay2) if isinstance(ay2, (int, float)) else None

def _normalize_houses_payload(h: Any) -> Any:
    if not isinstance(h, dict):
        return h
    if "cusps" not in h and "cusps_deg" in h:
        h["cusps"] = h["cusps_deg"]
    if "cusps_deg" not in h and "cusps" in h:
        h["cusps_deg"] = h["cusps"]
    if "house_system" not in h and "system" in h:
        h["house_system"] = h["system"]
    if "system" not in h and "house_system" in h:
        h["system"] = h["house_system"]
    if "asc_deg" not in h and "asc" in h:
        h["asc_deg"] = h["asc"]
    if "mc_deg" not in h and "mc" in h:
        h["mc_deg"] = h["mc"]
    for k in ("asc", "asc_deg", "mc", "mc_deg", "vertex", "eastpoint", "armc", "ramc"):
        if k in h and isinstance(h[k], (int, float)):
            h[k] = _wrap360(h[k])
    for key in ("cusps", "cusps_deg"):
        if isinstance(h.get(key), list):
            h[key] = [_wrap360(c) if isinstance(c, (int, float)) else c for c in h[key]]
    return h

def _recompute_houses_angles_if_needed(
    h: Any, ts: Dict[str, Any], payload: Dict[str, Any], chart: Dict[str, Any]
) -> Any:
    if not isinstance(h, dict):
        return h
    lat = payload.get("latitude")
    lon = payload.get("longitude")
    mode = (payload.get("mode") or "tropical").lower()
    ay = _extract_ayanamsa_from_chart(chart) if mode == "sidereal" else None
    recomputed = _recompute_angles_exact(
        jd_ut1=float(ts["jd_ut1"]),
        jd_tt=float(ts["jd_tt"]),
        latitude=float(lat) if isinstance(lat, (int, float)) else None,
        longitude_east=float(lon) if isinstance(lon, (int, float)) else None,
        mode=mode,
        ayanamsa_deg=ay,
    )
    if not recomputed:
        return h
    asc_new, mc_new = recomputed["asc_deg"], recomputed["mc_deg"]
    asc_old = h.get("asc_deg") if isinstance(h.get("asc_deg"), (int, float)) else h.get("asc")
    mc_old = h.get("mc_deg") if isinstance(h.get("mc_deg"), (int, float)) else h.get("mc")
    warn_list = h.get("warnings") or []
    changed = False

    if isinstance(asc_old, (int, float)):
        _ = _delta_arcsec(asc_new, float(asc_old))
    else:
        h["asc_deg"] = _wrap360(asc_new); h["asc"] = h["asc_deg"]; changed = True

    if isinstance(mc_old, (int, float)):
        d_mc = _delta_arcsec(mc_new, float(mc_old))
        if d_mc > ARCSEC_TOL:
            h["mc_deg"] = _wrap360(mc_new); h["mc"] = h["mc_deg"]; changed = True
            warn_list.append(f"mc_corrected_for_parity({d_mc:.2f}arcsec)")
    else:
        h["mc_deg"] = _wrap360(mc_new); h["mc"] = h["mc_deg"]; changed = True

    if changed:
        h["warnings"] = warn_list
    return _normalize_houses_payload(h)

def _prepare_chart_for_predict(chart: Dict[str, Any]) -> Dict[str, Any]:
    ch = dict(chart)
    bodies = ch.get("bodies")
    if isinstance(bodies, list):
        name_map: Dict[str, Any] = {}
        for b in bodies:
            if isinstance(b, dict):
                nm = b.get("name")
                if isinstance(nm, str) and nm:
                    name_map[nm] = b
        if name_map:
            ch["bodies_map"] = name_map
    elif isinstance(bodies, dict):
        ch["bodies_map"] = bodies
    return ch

def _require_coords_for_houses(payload: Dict[str, Any]):
    lat = payload.get("latitude")
    lon = payload.get("longitude")
    try:
        lat_ok = isinstance(lat, (int, float)) and float(lat) == float(lat)
        lon_ok = isinstance(lon, (int, float)) and float(lon) == float(lon)
    except Exception:
        lat_ok = lon_ok = False
    if not (lat_ok and lon_ok):
        raise ValueError("latitude and longitude are required (finite numbers) to compute houses")

def _want_houses(body: Dict[str, Any]) -> bool:
    """
    Decide whether to compute houses.

    Accepts:
      - houses: true/false
      - houses: { compute: false }  (legacy-safe)
    Defaults to True for back-compat.
    """
    h = body.get("houses")
    if isinstance(h, bool):
        return h
    if isinstance(h, dict):
        if "compute" in h and h.get("compute") is False:
            return False
    return True

# ───────────────────────── health / ops ─────────────────────────
@api.get("/api/health")
def health():
    return jsonify({
        "ok": True, 
        "status": "up", 
        "version": VERSION,
        "engines": {
            "chart": _CHART_ENGINE_NAME is not None,
            "houses": _HOUSES_KIND != "unavailable",
            "predictions": predict_engine is not None,
            "prediction_engine_v2": _PREDICTION_ENGINE_OK,
            "progressions": compute_progressions is not None,
            "returns": _returns_available(),
            "synastry": _SYN_OK,
            "directions": _DIR_OK,
            "parans": _parans_compute is not None,
            "relocation": _compute_relocated is not None,
            "astrocartography": _compute_astrocartography is not None,
        }
    }), 200

@api.get("/api/config")
@rate_limit(1)
def config_info():
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    calib_path = os.environ.get("ASTRO_CALIBRATORS", "config/calibrators.json")
    th_path = os.environ.get("ASTRO_HC_THRESHOLDS", "config/hc_thresholds.json")

    cfg = load_config(cfg_path)
    calib_ver = None
    th_summary = None

    try:
        now_utc = datetime.now(timezone.utc)
        ts_now = _compute_timescales_from_local(
            now_utc.strftime("%Y-%m-%d"),
            now_utc.strftime("%H:%M:%S"),
            "UTC",
        )
        ts_sample = {
            "jd_utc": float(ts_now["jd_utc"]),
            "jd_tt": float(ts_now["jd_tt"]),
            "jd_ut1": float(ts_now["jd_ut1"]),
            "delta_t": ts_now["delta_t"],
            "delta_at": ts_now["delta_at"],
            "dut1": float(ts_now["dut1"]),
        }
    except Exception:
        ts_sample = None

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_ver = (json.load(f) or {}).get("version")
    except Exception:
        pass

    try:
        with open(th_path, "r", encoding="utf-8") as f:
            th = json.load(f) or {}
            th_summary = {"entropy_H": th.get("entropy_H"), "defaults": th.get("defaults")}
    except Exception:
        pass

    return jsonify(
        {
            "ok": True,
            "mode": cfg.mode,
            "ayanamsa": getattr(cfg, "ayanamsa", None),
            "rate_limits_per_hour": getattr(cfg, "rate_limits_per_hour", None),
            "pro_features_enabled": getattr(cfg, "pro_features_enabled", None),
            "calibrators_version": calib_ver,
            "hc_thresholds_summary": th_summary,
            "timescale_sample": ts_sample,
            "version": VERSION,
        }
    ), 200

@api.get("/api/openapi")
@rate_limit(RL_DEBUG)
def openapi_spec():
    import yaml
    base = os.path.dirname(__file__)
    for p in (os.path.join(base, "..", "openapi.yaml"), os.path.join(base, "..", "..", "openapi.yaml")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return jsonify(yaml.safe_load(f)), 200
        except Exception:
            continue
    return _json_error("openapi_not_found", None, 404)

@api.get("/__debug/routes")
@rate_limit(RL_DEBUG)
def debug_routes():
    """Simple route index used by the frontend 'Routes' page."""
    rules = []
    for r in current_app.url_map.iter_rules():
        if r.endpoint == "static":
            continue
        methods = sorted(m for m in r.methods if m not in {"HEAD", "OPTIONS"})
        rules.append({"rule": str(r), "methods": methods, "endpoint": r.endpoint})
    rules.sort(key=lambda x: x["rule"])
    return jsonify({"ok": True, "routes": rules}), 200

# ───────────────────────── timescales ─────────────────────────
@api.post("/api/timescales")
@rate_limit(RL_TIMESCALES)
def timescales_endpoint():
    body = request.get_json(force=True) or {}
    try:
        date = body.get("date")
        tz = body.get("tz") or body.get("place_tz") or body.get("timezone")
        time_ = str(body.get("time") or "").strip()
        if re.match(r"^\d{1,2}:\d{2}$", time_):
            time_ = f"{time_}:00"
        if not isinstance(date, str) or not isinstance(time_, str) or not isinstance(tz, str):
            errs = []
            if not isinstance(date, str):
                errs.append({"loc": ["date"], "msg": "required string", "type": "value_error"})
            if not isinstance(time_, str):
                errs.append({"loc": ["time"], "msg": "required string", "type": "value_error"})
            if not isinstance(tz, str):
                errs.append({"loc": ["tz"], "msg": "required string (IANA zone)", "type": "value_error"})
            raise ValidationError(errs or "invalid payload")
        ts = _compute_timescales_from_local(date, time_, tz, payload=body if isinstance(body, dict) else None)
        return jsonify({"ok": True, "timescales": ts}), 200
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

# ───────────────────────── endpoints ─────────────────────────
@api.post("/api/calculate")
@rate_limit(RL_CALCULATE)
def calculate():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)
        hs = str(body.get("house_system", "")).strip()
        if hs:
            payload["house_system"] = hs
        for k in ("bodies", "points", "ayanamsa", "topocentric", "elevation_m", "elev_m", "dut1", "houses"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    want_houses = _want_houses(body)
    payload["houses"] = bool(want_houses)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    try:
        ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name, payload=payload)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        chart = {
            "mode": payload.get("mode"),
            "jd_ut": ts["jd_utc"],
            "jd_tt": ts["jd_tt"],
            "jd_ut1": ts["jd_ut1"],
            "meta": {"engine": _CHART_ENGINE_NAME, "warnings": ["chart_failed"]},
        }
        if DEBUG_VERBOSE:
            chart["error"] = str(e)

    houses: Optional[Dict[str, Any]] = None
    if want_houses:
        try:
            _require_coords_for_houses(payload)
            houses = _call_compute_houses(payload, ts)
            houses = _normalize_houses_payload(houses)
        except ValueError as e:
            return _json_error("houses_coords_required", str(e), 422)
        except NotImplementedError as e:
            return _json_error("houses_not_implemented", str(e) if DEBUG_VERBOSE else None, 501)
        except Exception as e:
            return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

        mode = (payload.get("mode") or "tropical").lower()
        if mode == "sidereal":
            ay = _extract_ayanamsa_from_chart(chart)
            if isinstance(ay, (int, float)) and isinstance(houses, dict):
                def rot(v: Optional[float]) -> Optional[float]:
                    if v is None:
                        return None
                    x = (float(v) - float(ay)) % 360.0
                    return 0.0 if abs(x) < 1e-12 else x
                for k in ("asc", "asc_deg", "mc", "mc_deg", "vertex", "eastpoint"):
                    if k in houses and isinstance(houses[k], (int, float)):
                        houses[k] = rot(houses[k])
                if isinstance(houses.get("cusps"), list):
                    houses["cusps"] = [rot(c) for c in houses["cusps"]]
                if isinstance(houses.get("cusps_deg"), list):
                    houses["cusps_deg"] = [rot(c) for c in houses["cusps_deg"]]

        houses = _recompute_houses_angles_if_needed(houses, ts, payload, chart)

    meta = {
        "timescales": ts,
        "timescales_locked": True,
        "chart_engine": _CHART_ENGINE_NAME,
        "houses_engine": _HOUSES_KIND if want_houses else "skipped",
        "houses_requested": bool(want_houses),
    }
    meta.update(_snapshot_ephemeris_meta(chart.get("meta")))

    # Calculate aspects if requested
    aspects_result = None
    if body.get("aspects", False):
        try:
            aspects_result = _call_compute_aspects(payload, chart, houses)
        except Exception as e:
            if DEBUG_VERBOSE:
                aspects_result = {"error": str(e)}

    resp = {"ok": True, "timescales": ts, "chart": chart, "meta": meta}
    if want_houses:
        resp["houses"] = houses
    if aspects_result:
        resp["aspects"] = aspects_result
    return jsonify(resp), 200

@api.post("/api/report")
@rate_limit(RL_REPORT)
def report():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)
        hs = str(body.get("house_system", "")).strip()
        if hs:
            payload["house_system"] = hs
        for k in ("bodies", "points", "ayanamsa", "topocentric", "elevation_m", "elev_m", "dut1", "houses"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    want_houses = _want_houses(body)
    payload["houses"] = bool(want_houses)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    try:
        ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name, payload=payload)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        chart = {
            "mode": payload.get("mode"),
            "jd_ut": ts["jd_utc"],
            "jd_tt": ts["jd_tt"],
            "jd_ut1": ts["jd_ut1"],
            "meta": {"engine": _CHART_ENGINE_NAME, "warnings": ["chart_failed"]},
        }
        if DEBUG_VERBOSE:
            chart["error"] = str(e)

    houses: Optional[Dict[str, Any]] = None
    if want_houses:
        try:
            _require_coords_for_houses(payload)
            houses = _call_compute_houses(payload, ts)
            houses = _normalize_houses_payload(houses)
        except ValueError as e:
            return _json_error("houses_coords_required", str(e), 422)
        except NotImplementedError as e:
            return _json_error("houses_not_implemented", str(e) if DEBUG_VERBOSE else None, 501)
        except Exception as e:
            return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

        mode = (payload.get("mode") or "tropical").lower()
        if mode == "sidereal":
            ay = _extract_ayanamsa_from_chart(chart)
            if isinstance(ay, (int, float)) and isinstance(houses, dict):
                def rot(v: Optional[float]) -> Optional[float]:
                    if v is None:
                        return None
                    x = (float(v) - float(ay)) % 360.0
                    return 0.0 if abs(x) < 1e-12 else x
                for k in ("asc", "asc_deg", "mc", "mc_deg", "vertex", "eastpoint"):
                    if k in houses and isinstance(houses[k], (int, float)):
                        houses[k] = rot(houses[k])
                if isinstance(houses.get("cusps"), list):
                    houses["cusps"] = [rot(c) for c in houses["cusps"]]
                if isinstance(houses.get("cusps_deg"), list):
                    houses["cusps_deg"] = [rot(c) for c in houses["cusps_deg"]]

        houses = _recompute_houses_angles_if_needed(houses, ts, payload, chart)

    narrative = (
        "This is a placeholder narrative aligned to your mode and computed houses. "
        "Evidence will accompany predictions in /predictions."
    )

    meta = {
        "timescales": ts,
        "timescales_locked": True,
        "chart_engine": _CHART_ENGINE_NAME,
        "houses_engine": _HOUSES_KIND if want_houses else "skipped",
        "houses_requested": bool(want_houses),
    }
    meta.update(_snapshot_ephemeris_meta(chart.get("meta")))

    resp = {"ok": True, "chart": chart, "narrative": narrative, "meta": meta}
    if want_houses:
        resp["houses"] = houses
    return jsonify(resp), 200

# ───────────────────────── aspects ─────────────────────────
@api.post("/api/aspects")
@rate_limit(RL_ASPECTS)
def aspects():
    def _norm_orbs(orbs_raw):
        """Ensure orbs is a dict[str,float]."""
        if not isinstance(orbs_raw, dict):
            return {}
        out = {}
        for k, v in orbs_raw.items():
            try:
                f = float(v)
            except Exception:
                continue
            out[str(k).strip().lower()] = f
        return out

    def _norm_aspects(aspects_raw):
        """Ensure aspects is a list[str]."""
        if aspects_raw is None:
            return None
        if isinstance(aspects_raw, str):
            return [aspects_raw.strip().lower()]
        if isinstance(aspects_raw, (list, tuple)):
            return [str(a).strip().lower() for a in aspects_raw if a]
        return None

    def _norm_bodies(bodies_raw):
        """Ensure bodies is a list[str]."""
        if bodies_raw is None:
            return None
        if isinstance(bodies_raw, str):
            return [bodies_raw.strip()]
        if isinstance(bodies_raw, (list, tuple)):
            return [str(b).strip() for b in bodies_raw if b]
        return None

    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)

        # Normalized additional parameters
        if "orbs" in body:
            payload["orbs"] = _norm_orbs(body.get("orbs"))
        if "aspects" in body:
            payload["aspects"] = _norm_aspects(body.get("aspects"))
        if "bodies" in body:
            payload["bodies"] = _norm_bodies(body.get("bodies"))
        if "points" in body:
            payload["points"] = body.get("points")  # keep raw, engine decides
        if "mode" in body:
            payload["mode"] = str(body.get("mode")).strip().lower()
        if "houses" in body:
            payload["houses"] = bool(body.get("houses"))

    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    try:
        ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name, payload=payload)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    # Get chart data
    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        return _json_error("chart_internal", str(e) if DEBUG_VERBOSE else "chart_failed", 500)

    # Get houses if needed
    houses = None
    if payload.get("houses", True):
        try:
            _require_coords_for_houses(payload)
            houses = _call_compute_houses(payload, ts)
            houses = _normalize_houses_payload(houses)
        except Exception as e:
            return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "houses_failed", 500)

    # Calculate aspects
    try:
        aspects_result = _call_compute_aspects(payload, chart, houses)
    except Exception as e:
        return _json_error("aspects_internal", str(e) if DEBUG_VERBOSE else "aspects_failed", 500)

    meta = {
        "timescales": ts,
        "chart_engine": _CHART_ENGINE_NAME,
        "houses_engine": _HOUSES_KIND if houses else "skipped",
        "aspects_engine": "aspects.py",
    }
    meta.update(_snapshot_ephemeris_meta(chart.get("meta")))

    return jsonify({
        "ok": True,
        "timescales": ts,
        "chart": chart,
        "houses": houses,
        "aspects": aspects_result,
        "meta": meta,
    }), 200

# ───────────────────────── helper function for returns ─────────────────────────
def _returns_available() -> bool:
    return _returns_mod is not None

# ───────────────────────── PREDICTION ENGINE ─────────────────────────

def parse_prediction_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate prediction engine request payload (V2 shape, tolerant)."""
    errors = []

    if not isinstance(body, dict):
        raise ValidationError([{"loc": [], "msg": "payload must be an object", "type": "type_error.dict"}])

    # natal_chart is required for all endpoints in this block (except relationship/validate handle their own)
    if "natal_chart" in body and not isinstance(body.get("natal_chart"), dict):
        errors.append({"loc": ["natal_chart"], "msg": "required object", "type": "value_error"})

    # time_range: when present, must be [start, end]
    if "time_range" in body:
        tr = body.get("time_range")
        if not (isinstance(tr, (list, tuple)) and len(tr) == 2):
            errors.append({"loc": ["time_range"], "msg": "must be [start_date, end_date] array", "type": "value_error"})

    # target_date: when present, must be str|int|float
    if "target_date" in body:
        td = body.get("target_date")
        if not isinstance(td, (str, int, float)):
            errors.append({"loc": ["target_date"], "msg": "must be ISO string, JD float, or datetime", "type": "value_error"})

    if errors:
        raise ValidationError(errors)
    return body


@api.post("/api/prediction/forecast")
@rate_limit(RL_PREDICTION_FORECAST)
def prediction_comprehensive_forecast_route():
    """
    Comprehensive astrological forecast using the prediction engine.

    Body:
      natal_chart: { date, time, place_tz, bodies?, ... }
      time_range: [start_date, end_date]
      techniques?: ["transits", "progressions", "solar_returns", ...]
      confidence_threshold?: float (default 0.2)
      synthesis_method?: "weighted_consensus" | "simple"
      statistical_validation?: bool (default false)
      include_vedic?: bool (default false)
      peak_window_days?: int (default 14)
      ... technique-specific parameters with prefixes (transit_*, progression_*, return_*, vedic_*)
    """
    if not _PREDICTION_ENGINE_OK:
        det = {"import_error": repr(_PREDICTION_ENGINE_ERR)} if DEBUG_VERBOSE and _PREDICTION_ENGINE_ERR else None
        return _json_error("prediction_engine_unavailable", det or "prediction engine not available", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_prediction_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    try:
        natal_chart = payload["natal_chart"]
        time_range = tuple(payload["time_range"]) if "time_range" in payload else None
        if not time_range:
            return _json_error("validation_error", [{"loc": ["time_range"], "msg": "required"}], 400)

        techniques = payload.get("techniques")
        confidence_threshold = float(payload.get("confidence_threshold", 0.2))
        synthesis_method = payload.get("synthesis_method", "weighted_consensus")
        statistical_validation = bool(payload.get("statistical_validation", False))
        include_vedic = bool(payload.get("include_vedic", False))
        peak_window_days = int(payload.get("peak_window_days", 14))

        # Technique-specific kwargs passthrough
        tk_kwargs = {
            k: v
            for k, v in payload.items()
            if any(k.startswith(prefix) for prefix in ("transit_", "progression_", "return_", "vedic_"))
        }

        forecast = comprehensive_forecast(
            natal_chart=natal_chart,
            time_range=time_range,
            techniques=techniques,
            confidence_threshold=confidence_threshold,
            synthesis_method=synthesis_method,
            statistical_validation=statistical_validation,
            include_vedic=include_vedic,
            peak_window_days=peak_window_days,
            **tk_kwargs,
        )

        meta = {
            "prediction_engine": "app.core.prediction v2",
            "natal_chart_engine": _CHART_ENGINE_NAME,
            "houses_engine": _HOUSES_KIND,
            "computation_time_ms": getattr(forecast, "computation_time_ms", None),
        }
        meta.update(_snapshot_ephemeris_meta())

        return jsonify({
            "ok": True,
            "forecast": _serialize_comprehensive_forecast(forecast),
            "meta": meta
        }), 200

    except Exception as e:
        return _json_error("prediction_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/prediction/transits")
@rate_limit(RL_PREDICTION_TRANSITS)
def prediction_transits_route():
    """Calculate transit events using the prediction engine."""
    if not _PREDICTION_ENGINE_OK:
        det = {"import_error": repr(_PREDICTION_ENGINE_ERR)} if DEBUG_VERBOSE and _PREDICTION_ENGINE_ERR else None
        return _json_error("prediction_engine_unavailable", det or "prediction engine not available", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_prediction_payload(body)

        natal_chart = payload["natal_chart"]
        time_range = tuple(payload["time_range"]) if "time_range" in payload else None
        if not time_range:
            return _json_error("validation_error", [{"loc": ["time_range"], "msg": "required"}], 400)

        # Transit-specific accepted parameters
        accept = {
            "transiting_bodies", "natal_bodies", "orbs", "aspects", "include_aspects_to",
            "include_house_cusps", "frame", "zodiac_mode", "ayanamsa_deg", "exact_timing",
            "statistical_validation", "confidence_threshold",
        }
        kwargs = {k: payload[k] for k in payload.keys() & accept}

        result = predict_transits(natal_chart=natal_chart, time_range=time_range, **kwargs)

        return jsonify({
            "ok": getattr(result, "ok", False),
            "result": _serialize_prediction_result(result),
            "meta": {
                "prediction_engine": "app.core.prediction v2",
                "technique": getattr(result, "technique", "transits"),
                "computation_time_ms": getattr(result, "computation_time_ms", None),
            },
        }), 200

    except Exception as e:
        return _json_error("transits_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/prediction/progressions")
@rate_limit(RL_PREDICTION_PROGRESSIONS)
def prediction_progressions_route():
    """Calculate progression events using the prediction engine."""
    if not _PREDICTION_ENGINE_OK:
        det = {"import_error": repr(_PREDICTION_ENGINE_ERR)} if DEBUG_VERBOSE and _PREDICTION_ENGINE_ERR else None
        return _json_error("prediction_engine_unavailable", det or "prediction engine not available", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_prediction_payload(body)

        natal_chart = payload["natal_chart"]
        target_date = payload.get("target_date")
        if not target_date:
            return _json_error("validation_error", [{"loc": ["target_date"], "msg": "required"}], 400)

        accept = {
            "method", "lunar_month", "tertiary_mode", "frame", "house_system",
            "zodiac_mode", "ayanamsa_deg", "aspects_to_natal", "orbs",
            "parallels", "antiscia", "statistical_validation",
        }
        kwargs = {k: payload[k] for k in payload.keys() & accept}

        result = predict_progressions(natal_chart=natal_chart, target_date=target_date, **kwargs)

        return jsonify({
            "ok": getattr(result, "ok", False),
            "result": _serialize_prediction_result(result),
            "meta": {
                "prediction_engine": "app.core.prediction v2",
                "technique": getattr(result, "technique", "progressions"),
                "computation_time_ms": getattr(result, "computation_time_ms", None),
            },
        }), 200

    except Exception as e:
        return _json_error("progressions_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/prediction/returns")
@rate_limit(RL_PREDICTION_RETURNS)
def prediction_returns_route():
    """Calculate solar/lunar return events using the prediction engine."""
    if not _PREDICTION_ENGINE_OK:
        det = {"import_error": repr(_PREDICTION_ENGINE_ERR)} if DEBUG_VERBOSE and _PREDICTION_ENGINE_ERR else None
        return _json_error("prediction_engine_unavailable", det or "prediction engine not available", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_prediction_payload(body)

        natal_chart = payload["natal_chart"]
        return_type = payload.get("return_type")
        year = payload.get("year")

        if not return_type or not year:
            return _json_error("validation_error", [
                {"loc": ["return_type"], "msg": "required"},
                {"loc": ["year"], "msg": "required"},
            ], 400)

        if return_type not in ("solar", "lunar"):
            return _json_error("validation_error", [{"loc": ["return_type"], "msg": "must be 'solar' or 'lunar'"}], 400)

        accept = {
            "lunar_month", "place", "frame", "house_system", "zodiac_mode",
            "ayanamsa_deg", "estimate_uncertainty", "aspects_to_natal", "orbs",
            "statistical_validation",
        }
        kwargs = {k: payload[k] for k in payload.keys() & accept}

        result = predict_returns(natal_chart=natal_chart, return_type=return_type, year=int(year), **kwargs)

        return jsonify({
            "ok": getattr(result, "ok", False),
            "result": _serialize_prediction_result(result),
            "meta": {
                "prediction_engine": "app.core.prediction v2",
                "technique": getattr(result, "technique", "returns"),
                "computation_time_ms": getattr(result, "computation_time_ms", None),
            },
        }), 200

    except Exception as e:
        return _json_error("returns_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/prediction/directions")
@rate_limit(RL_PREDICTION_DIRECTIONS)
def prediction_directions_route():
    """Calculate direction events using the prediction engine."""
    if not _PREDICTION_ENGINE_OK:
        det = {"import_error": repr(_PREDICTION_ENGINE_ERR)} if DEBUG_VERBOSE and _PREDICTION_ENGINE_ERR else None
        return _json_error("prediction_engine_unavailable", det or "prediction engine not available", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_prediction_payload(body)

        natal_chart = payload["natal_chart"]
        target_date = payload.get("target_date")
        if not target_date:
            return _json_error("validation_error", [{"loc": ["target_date"], "msg": "required"}], 400)

        # IMPORTANT: do NOT pass 'aspects_to_natal' — legacy backends choke on it
        accept = {
            "method", "frame", "zodiac_mode", "ayanamsa_deg", "house_system",
            "orbs", "statistical_validation",
        }
        kwargs = {k: payload[k] for k in payload.keys() & accept}

        result = predict_directions(natal_chart=natal_chart, target_date=target_date, **kwargs)

        return jsonify({
            "ok": getattr(result, "ok", False),
            "result": _serialize_prediction_result(result),
            "meta": {
                "prediction_engine": "app.core.prediction v2",
                "technique": getattr(result, "technique", "directions"),
                "computation_time_ms": getattr(result, "computation_time_ms", None),
            },
        }), 200

    except Exception as e:
        return _json_error("directions_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/prediction/relationship")
@rate_limit(RL_PREDICTION_RELATIONSHIP)
def prediction_relationship_route():
    """Relationship forecast between two natal charts."""
    if not _PREDICTION_ENGINE_OK:
        det = {"import_error": repr(_PREDICTION_ENGINE_ERR)} if DEBUG_VERBOSE and _PREDICTION_ENGINE_ERR else None
        return _json_error("prediction_engine_unavailable", det or "prediction engine not available", 501)

    try:
        body = request.get_json(force=True) or {}

        natal_a = body.get("natal_a")
        natal_b = body.get("natal_b")
        time_range = body.get("time_range")

        if not natal_a or not natal_b or not time_range:
            return _json_error("validation_error", [
                {"loc": ["natal_a"], "msg": "required"},
                {"loc": ["natal_b"], "msg": "required"},
                {"loc": ["time_range"], "msg": "required"},
            ], 400)

        if not (isinstance(time_range, (list, tuple)) and len(time_range) == 2):
            return _json_error("validation_error", [{"loc": ["time_range"], "msg": "must be [start_date, end_date] array"}], 400)

        accept = {
            "synastry_orbs", "composite_method", "include_transits_to_composite",
            "include_progressions", "confidence_threshold", "parallels", "antiscia",
            "frame", "zodiac_mode", "ayanamsa_deg", "house_system",
        }
        kwargs = {k: body[k] for k in body.keys() & accept}

        # technique-specific passthroughs
        for k, v in body.items():
            if any(k.startswith(prefix) for prefix in ("transit_", "progression_")):
                kwargs[k] = v

        result = relationship_forecast(
            natal_a=natal_a,
            natal_b=natal_b,
            time_range=tuple(time_range),
            **kwargs,
        )

        return jsonify({
            "ok": True,
            "result": _serialize_relationship_forecast(result),
            "meta": {
                "prediction_engine": "app.core.prediction v2",
                "technique": "relationship_forecast",
            },
        }), 200

    except Exception as e:
        return _json_error("relationship_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/prediction/validate")
@rate_limit(RL_PREDICTION_VALIDATION)
def prediction_validation_route():
    """Validate prediction model using test cases."""
    if not _PREDICTION_ENGINE_OK:
        det = {"import_error": repr(_PREDICTION_ENGINE_ERR)} if DEBUG_VERBOSE and _PREDICTION_ENGINE_ERR else None
        return _json_error("prediction_engine_unavailable", det or "prediction engine not available", 501)

    try:
        body = request.get_json(force=True) or {}

        test_cases = body.get("test_cases")
        if not isinstance(test_cases, list) or not test_cases:
            return _json_error("validation_error", [{"loc": ["test_cases"], "msg": "required non-empty array"}], 400)

        accept = {"validation_method", "n_folds", "metrics", "confidence_threshold"}
        kwargs = {k: body[k] for k in body.keys() & accept}

        result = validate_prediction_model(test_cases=test_cases, **kwargs)

        return jsonify({
            "ok": result.get("ok", True),
            "result": result,
            "meta": {
                "prediction_engine": "app.core.prediction v2",
                "technique": "model_validation",
            },
        }), 200

    except Exception as e:
        return _json_error("validation_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


# ─────────────── Serialization helpers for prediction engine ───────────────

def _serialize_prediction_event(event: Any) -> Dict[str, Any]:
    """Convert PredictionEvent to JSON-serializable dict."""
    if event is None:
        return {}
    return {
        "event_type": getattr(event, "event_type", None),
        "technique": getattr(event, "technique", None),
        "description": getattr(event, "description", None),
        "datetime_utc": getattr(event, "datetime_utc", None).isoformat()
            if hasattr(event, "datetime_utc") and getattr(event, "datetime_utc") else None,
        "jd_tt": getattr(event, "jd_tt", None),
        "jd_ut1": getattr(event, "jd_ut1", None),
        "precision_seconds": getattr(event, "precision_seconds", None),
        "confidence": getattr(event, "confidence", None),
        "significance": getattr(event, "significance", None),
        "metadata": getattr(event, "metadata", {}),
    }


def _serialize_timing_window(window: Any) -> Dict[str, Any]:
    """Convert TimingWindow to JSON-serializable dict."""
    if window is None:
        return {}
    return {
        "start_jd_tt": getattr(window, "start_jd_tt", None),
        "end_jd_tt": getattr(window, "end_jd_tt", None),
        "peak_jd_tt": getattr(window, "peak_jd_tt", None),
        "uncertainty_days": getattr(window, "uncertainty_days", None),
        "confidence_interval": getattr(window, "confidence_interval", None),
    }


def _serialize_prediction_result(result: Any) -> Dict[str, Any]:
    """Convert PredictionResult to JSON-serializable dict."""
    if result is None:
        return {}
    return {
        "ok": getattr(result, "ok", True),
        "technique": getattr(result, "technique", None),
        "events": [_serialize_prediction_event(e) for e in getattr(result, "events", [])],
        "synthesis": getattr(result, "synthesis", None),
        "timing_windows": [_serialize_timing_window(w) for w in getattr(result, "timing_windows", [])],
        "confidence_score": getattr(result, "confidence_score", None),
        "statistical_metrics": getattr(result, "statistical_metrics", {}),
        "warnings": getattr(result, "warnings", []),
        "metadata": getattr(result, "metadata", {}),
        "computation_time_ms": getattr(result, "computation_time_ms", None),
    }


def _serialize_comprehensive_forecast(forecast: Any) -> Dict[str, Any]:
    """Convert ComprehensiveForecast to JSON-serializable dict."""
    if forecast is None:
        return {}
    predictions = {}
    if hasattr(forecast, "predictions") and forecast.predictions:
        predictions = {k: _serialize_prediction_result(v) for k, v in forecast.predictions.items()}

    time_range = getattr(forecast, "time_range", [])
    if time_range and len(time_range) >= 2:
        time_range_iso = [dt.isoformat() if hasattr(dt, "isoformat") else str(dt) for dt in time_range[:2]]
    else:
        time_range_iso = []

    return {
        "natal_chart": getattr(forecast, "natal_chart", {}),
        "time_range": time_range_iso,
        "predictions": predictions,
        "synthesis": getattr(forecast, "synthesis", {}),
        "peak_periods": [_serialize_timing_window(w) for w in getattr(forecast, "peak_periods", [])],
        "risk_assessment": getattr(forecast, "risk_assessment", {}),
        "confidence_metrics": getattr(forecast, "confidence_metrics", {}),
        "validation_results": getattr(forecast, "validation_results", {}),
        "computation_time_ms": getattr(forecast, "computation_time_ms", None),
    }


def _serialize_relationship_forecast(forecast: Any) -> Dict[str, Any]:
    """Convert RelationshipForecast to JSON-serializable dict."""
    if forecast is None:
        return {}
    return {
        "synastry_analysis": getattr(forecast, "synastry_analysis", {}),
        "composite_analysis": getattr(forecast, "composite_analysis", {}),
        "transit_interactions": [_serialize_prediction_event(e) for e in getattr(forecast, "transit_interactions", [])],
        "progression_interactions": [_serialize_prediction_event(e) for e in getattr(forecast, "progression_interactions", [])],
        "compatibility_trends": getattr(forecast, "compatibility_trends", {}),
        "critical_periods": [_serialize_timing_window(w) for w in getattr(forecast, "critical_periods", [])],
        "relationship_score": getattr(forecast, "relationship_score", None),
        "confidence_metrics": getattr(forecast, "confidence_metrics", {}),
    }


# ───────────────────────── predictive (transits • validation • dasha • varga • yogas) ─────────────────────────
from typing import Any, Dict, List, Tuple, Optional
import os, time, math, threading
from flask import request, jsonify

# Global adapter for performance optimization (per-frame cache)
from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig

# ───────────────────────── shared adapters & gating ─────────────────────────
_GLOBAL_ADAPTERS: Dict[str, EphemerisAdapter] = {}
_ADAPTER_LOCK = threading.Lock()

def _norm_frame_key(frame: Optional[str]) -> str:
    s = (frame or "ecliptic-of-date").strip().lower()
    return "ecliptic-j2000" if s in ("ecliptic-j2000", "j2000", "ecl-j2000") else "ecliptic-of-date"

def get_shared_adapter(frame: str = "ecliptic-of-date") -> EphemerisAdapter:
    """Get or create a shared EphemerisAdapter instance (per frame) for performance."""
    key = _norm_frame_key(frame)
    with _ADAPTER_LOCK:
        ep = _GLOBAL_ADAPTERS.get(key)
        if ep is None:
            ep = EphemerisAdapter(EphemConfig(frame=key, compute_velocity_default=False))
            _GLOBAL_ADAPTERS[key] = ep
        return ep

# small concurrency gate to avoid 429s under burst load from the same pod
_PRED_MAX_CONC = int(os.getenv("PREDICTIVE_MAX_CONCURRENCY", "2"))
_PRED_SEM_TIMEOUT_S = float(os.getenv("PREDICTIVE_SEM_TIMEOUT_S", "25"))
_PRED_SEM = threading.Semaphore(_PRED_MAX_CONC)

def _busy():
    resp = _json_error("server_busy", "try again shortly", 429)
    resp.headers["Retry-After"] = "2"
    return resp

def _take_gate():
    return _PRED_SEM.acquire(timeout=_PRED_SEM_TIMEOUT_S)

def _give_gate():
    try:
        _PRED_SEM.release()
    except Exception:
        pass

# ───────────────────────── math helpers ─────────────────────────
TAU = 360.0

def _norm360(x: float) -> float:
    r = float(x) % TAU
    return r + TAU if r < 0.0 else r

def _wrap180(x: float) -> float:
    v = ((float(x) + 180.0) % 360.0) - 180.0
    return 180.0 if v == -180.0 else v

def _wrap360(x: float) -> float:
    v = float(x) % 360.0
    return v + 360.0 if v < 0.0 else v

def _split_dt(s: str, fallback: str) -> Tuple[str, str]:
    """Split 'YYYY-MM-DD[ T]HH:MM:SS' into (date, time) with sensible fallbacks."""
    s = (s or "").strip().replace("T", " ")
    if len(s) <= 10:
        return s[:10], fallback
    return s[:10], (s[11:19] or fallback)

# ───────────────────────── time window parsing ─────────────────────────
def _parse_time_range_like(body: Dict[str, Any]) -> Tuple[float, float]:
    """
    Accepts any of:
      - jd_start_tt & jd_end_tt (floats)
      - time_range: [start, end] where each item can be ISO date or full ISO datetime
      - date*_*/time*_* + place_tz / timezone (old shape)
    Returns (jd0, jd1) in TT, raises ValidationError-compatible dict on problems.
    """
    # direct JDs
    jd0 = body.get("jd_start_tt")
    jd1 = body.get("jd_end_tt")
    if isinstance(jd0, (int, float)) and isinstance(jd1, (int, float)):
        jd0f, jd1f = float(jd0), float(jd1)
        if not (math.isfinite(jd0f) and math.isfinite(jd1f) and jd1f > jd0f):
            raise ValidationError([{"loc": ["jd_start_tt", "jd_end_tt"], "msg": "invalid range"}])
        return jd0f, jd1f

    # time_range: ["YYYY-MM-DD" | ISO, "YYYY-MM-DD" | ISO]
    tr = body.get("time_range")
    if isinstance(tr, (list, tuple)) and len(tr) == 2:
        tz = body.get("place_tz") or body.get("timezone") or "UTC"
        s0, s1 = tr[0], tr[1]
        if not (isinstance(s0, str) and isinstance(s1, str)):
            raise ValidationError([{"loc": ["time_range"], "msg": "items must be strings"}])
        d0, t0 = _split_dt(s0, "00:00:00")
        d1, t1 = _split_dt(s1, "23:59:59")
        ts0 = _compute_timescales_from_local(d0, t0, tz, payload=body)
        ts1 = _compute_timescales_from_local(d1, t1, tz, payload=body)
        jd0f, jd1f = float(ts0["jd_tt"]), float(ts1["jd_tt"])
        if jd1f <= jd0f:
            raise ValidationError([{"loc": ["time_range"], "msg": "end must be after start"}])
        return jd0f, jd1f

    # legacy civil fields
    date0 = body.get("date_start") or body.get("date")
    time0 = body.get("time_start") or body.get("time") or "00:00:00"
    date1 = body.get("date_end")
    time1 = body.get("time_end") or "23:59:59"
    tz = body.get("place_tz") or body.get("timezone") or "UTC"
    if not (isinstance(date0, str) and isinstance(time0, str)):
        raise ValidationError([{"loc": ["date_start/time_start"], "msg": "required"}])
    ts0 = _compute_timescales_from_local(date0, time0, tz, payload=body)
    if isinstance(date1, str):
        ts1 = _compute_timescales_from_local(date1, time1, tz, payload=body)
        jd0f, jd1f = float(ts0["jd_tt"]), float(ts1["jd_tt"])
    else:
        jd0f = float(ts0["jd_tt"])
        jd1f = jd0f + 1.0  # default 24h window
    if not (math.isfinite(jd0f) and math.isfinite(jd1f) and jd1f > jd0f):
        raise ValidationError([{"loc": ["jd_start_tt", "jd_end_tt"], "msg": "invalid range"}])
    return jd0f, jd1f

def _parse_step_minutes(v: Any, *, default_min: float) -> Any:
    """
    Accept numeric minutes (>0) or "auto"/0/None → "auto".
    Returns either float minutes or the string "auto" (engine understands both).
    """
    if v is None:
        return "auto" if default_min <= 0 else default_min
    if isinstance(v, str) and v.strip().lower() == "auto":
        return "auto"
    try:
        f = float(v)
        return "auto" if f <= 0 else f
    except Exception:
        return default_min

# ───────────────────────── ephemeris perf helpers ─────────────────────────
def _k(jd: float, *, ndigits: int = 6) -> float:
    """JD bucket: round to ~1e-6 day (~0.0864 s)."""
    return round(float(jd), ndigits)

class _PerRequestEphem:
    """Per-request memoized ephemeris accessor with sidereal toggle & batching."""
    def __init__(self, adapter: EphemerisAdapter, obs: dict, *, sidereal: bool, ay_deg: float):
        self.adapter = adapter
        self.obs = obs
        self.sidereal = bool(sidereal)
        self.ay = float(ay_deg if sidereal else 0.0)
        self.lon_cache: dict[tuple[float, str], float] = {}
        self.calls = 0  # count adapter calls

    def _nir(self, lon_trop: float) -> float:
        return _norm360(float(lon_trop) - self.ay)

    def map(self, jd_tt: float, names: list[str]) -> dict[str, float]:
        rows = self.adapter.ecliptic_longitudes(jd_tt, names, **self.obs).get("results", [])
        self.calls += 1
        out: dict[str, float] = {}
        if not rows:
            return out
        jd_key = _k(jd_tt)
        if self.sidereal:
            for row in rows:
                nm = row["name"]
                lon = self._nir(row["longitude"])
                out[nm] = lon
                self.lon_cache[(jd_key, nm)] = lon
        else:
            for row in rows:
                nm = row["name"]
                lon = float(row["longitude"])
                out[nm] = lon
                self.lon_cache[(jd_key, nm)] = lon
        return out

    def one(self, jd_tt: float, name: str) -> Optional[float]:
        key = (_k(jd_tt), name)
        if key in self.lon_cache:
            return self.lon_cache[key]
        rows = self.adapter.ecliptic_longitudes(jd_tt, [name], **self.obs).get("results", [])
        self.calls += 1
        if not rows:
            return None
        lon = float(rows[0]["longitude"])
        if self.sidereal:
            lon = self._nir(lon)
        self.lon_cache[key] = lon
        return lon

class _CountingAdapter:
    """Lightweight proxy to count ephemeris calls made by engines we don't control."""
    def __init__(self, base: EphemerisAdapter):
        self._base = base
        self.calls = 0
    def ecliptic_longitudes(self, jd_tt, names, **obs):
        self.calls += 1
        return self._base.ecliptic_longitudes(jd_tt, names, **obs)

# ───────────────────────── /predictive/transits ─────────────────────────
from typing import Any, Dict, List, Tuple, Optional
import os, time, math, threading
from flask import request, jsonify

# Global adapter for performance optimization (per-frame cache)
from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig

_GLOBAL_ADAPTERS: Dict[str, EphemerisAdapter] = {}
_ADAPTER_LOCK = threading.Lock()

def _norm_frame_key(frame: Optional[str]) -> str:
    s = (frame or "ecliptic-of-date").strip().lower()
    return "ecliptic-j2000" if s in ("ecliptic-j2000", "j2000", "ecl-j2000") else "ecliptic-of-date"

def get_shared_adapter(frame: str = "ecliptic-of-date") -> EphemerisAdapter:
    """Get or create a shared EphemerisAdapter instance (per frame) for performance."""
    key = _norm_frame_key(frame)
    with _ADAPTER_LOCK:
        ep = _GLOBAL_ADAPTERS.get(key)
        if ep is None:
            ep = EphemerisAdapter(EphemConfig(frame=key, compute_velocity_default=False))
            _GLOBAL_ADAPTERS[key] = ep
        return ep

# small concurrency gate to avoid 429s under burst load from the same pod
_PRED_MAX_CONC = int(os.getenv("PREDICTIVE_MAX_CONCURRENCY", "2"))
_PRED_SEM_TIMEOUT_S = float(os.getenv("PREDICTIVE_SEM_TIMEOUT_S", "25"))
_PRED_SEM = threading.Semaphore(_PRED_MAX_CONC)

def _busy():
    resp = _json_error("server_busy", "try again shortly", 429)
    resp.headers["Retry-After"] = "2"
    return resp

def _take_gate():
    return _PRED_SEM.acquire(timeout=_PRED_SEM_TIMEOUT_S)

def _give_gate():
    try:
        _PRED_SEM.release()
    except Exception:
        pass

def _split_dt(s: str, fallback: str) -> Tuple[str, str]:
    """Split 'YYYY-MM-DD[ T]HH:MM:SS' into (date, time) with sensible fallbacks."""
    s = (s or "").strip().replace("T", " ")
    if len(s) <= 10:
        return s[:10], fallback
    return s[:10], (s[11:19] or fallback)

def _parse_time_range_like(body: Dict[str, Any]) -> Tuple[float, float]:
    """
    Accepts any of:
    - jd_start_tt & jd_end_tt (floats)
    - time_range: [start, end] where each item can be ISO date or full ISO datetime
    - date*_*/time*_* + place_tz / timezone (old shape)
    Returns (jd0, jd1) in TT, raises ValidationError-compatible dict on problems.
    """
    # direct JDs
    jd0 = body.get("jd_start_tt")
    jd1 = body.get("jd_end_tt")
    if isinstance(jd0, (int, float)) and isinstance(jd1, (int, float)):
        jd0f, jd1f = float(jd0), float(jd1)
        if not (math.isfinite(jd0f) and math.isfinite(jd1f) and jd1f > jd0f):
            raise ValidationError([{"loc": ["jd_start_tt", "jd_end_tt"], "msg": "invalid range"}])
        return jd0f, jd1f

    # time_range: ["YYYY-MM-DD" | ISO, "YYYY-MM-DD" | ISO]
    tr = body.get("time_range")
    if isinstance(tr, (list, tuple)) and len(tr) == 2:
        tz = body.get("place_tz") or body.get("timezone") or "UTC"
        s0, s1 = tr[0], tr[1]
        if not (isinstance(s0, str) and isinstance(s1, str)):
            raise ValidationError([{"loc": ["time_range"], "msg": "items must be strings"}])
        d0, t0 = _split_dt(s0, "00:00:00")
        d1, t1 = _split_dt(s1, "23:59:59")
        ts0 = _compute_timescales_from_local(d0, t0, tz, payload=body)
        ts1 = _compute_timescales_from_local(d1, t1, tz, payload=body)
        jd0f, jd1f = float(ts0["jd_tt"]), float(ts1["jd_tt"])
        if jd1f <= jd0f:
            raise ValidationError([{"loc": ["time_range"], "msg": "end must be after start"}])
        return jd0f, jd1f

    # legacy civil fields
    date0 = body.get("date_start") or body.get("date")
    time0 = body.get("time_start") or body.get("time") or "00:00:00"
    date1 = body.get("date_end")
    time1 = body.get("time_end") or "23:59:59"
    tz = body.get("place_tz") or body.get("timezone") or "UTC"

    if not (isinstance(date0, str) and isinstance(time0, str)):
        raise ValidationError([{"loc": ["date_start/time_start"], "msg": "required"}])

    ts0 = _compute_timescales_from_local(date0, time0, tz, payload=body)
    if isinstance(date1, str):
        ts1 = _compute_timescales_from_local(date1, time1, tz, payload=body)
        jd0f, jd1f = float(ts0["jd_tt"]), float(ts1["jd_tt"])
    else:
        jd0f = float(ts0["jd_tt"])
        jd1f = jd0f + 1.0  # default 24h window

    if not (math.isfinite(jd0f) and math.isfinite(jd1f) and jd1f > jd0f):
        raise ValidationError([{"loc": ["jd_start_tt", "jd_end_tt"], "msg": "invalid range"}])
    return jd0f, jd1f

def _parse_step_minutes(v: Any, *, default_min: float) -> Any:
    """
    Accept numeric minutes (>0) or "auto"/0/None → "auto".
    Returns either float minutes or the string "auto" (engine understands both).
    """
    if v is None:
        return "auto" if default_min <= 0 else default_min
    if isinstance(v, str) and v.strip().lower() == "auto":
        return "auto"
    try:
        f = float(v)
        return "auto" if f <= 0 else f
    except Exception:
        return default_min

@api.post("/api/predictive/transits")
@rate_limit(RL_PREDICTIVE)
def predictive_transits():
    """
    Scan transits (moving bodies vs target longitudes) with robust refinement.
    Accepts either a direct targets map or a 'targets_chart' (natal chart params).
    Also accepts 'time_range' in addition to jd/date fields.
    """
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    if not _take_gate():
        return _busy()

    t0 = time.perf_counter()

    try:
        # -------- time window (jd_tt) --------
        try:
            jd0, jd1 = _parse_time_range_like(body)
        except ValidationError as e:
            return _json_error("validation_error", e.errors(), 400)
        except Exception as e:
            return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

        # -------- movers / targets --------
        raw_movers = body.get("movers") or ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn"]
        if not (isinstance(raw_movers, list) and all(isinstance(x, str) and x for x in raw_movers)):
            return _json_error("validation_error", [{"loc":["movers"],"msg":"must be a list of names"}], 400)

        # Pre-process movers once
        movers = [m.strip() for m in raw_movers if m and isinstance(m, str)]
        movers = list(dict.fromkeys(movers))  # deduplicate while preserving order

        # targets_longitudes: direct map (preferred, fastest path)
        targets: Dict[str, float] = {}
        raw_targets = body.get("targets_longitudes")
        if isinstance(raw_targets, dict):
            # Batch process targets for better performance
            for k, v in raw_targets.items():
                try:
                    targets[str(k)] = _wrap360(float(v))
                except (ValueError, TypeError):
                    continue  # Skip invalid entries

        # Or build targets from a chart payload (e.g., natal) - only if no direct targets
        if not targets and isinstance(body.get("targets_chart"), dict):
            targ = dict(body["targets_chart"])
            tz_nat = targ.get("place_tz") or targ.get("timezone") or "UTC"
            try:
                ts_nat = _compute_timescales_from_local(targ["date"], targ.get("time", "00:00:00"), tz_nat, payload=targ)
            except Exception as e:
                return _json_error("validation_error", [{"loc":["targets_chart"], "msg": str(e)}], 400)

            try:
                ch = _call_compute_chart(targ, ts_nat)
            except Exception as e:
                return _json_error("chart_internal", str(e) if DEBUG_VERBOSE else "chart_failed", 500)

            # Extract targets more efficiently
            for row_list in [ch.get("bodies", []), ch.get("points", [])]:
                for row in row_list:
                    if (isinstance(row, dict) and 
                        "name" in row and 
                        isinstance(row.get("longitude_deg"), (int, float))):
                        targets[str(row["name"])] = _wrap360(float(row["longitude_deg"]))

        if not targets:
            return _json_error("validation_error", [{"loc":["targets_longitudes|targets_chart"],"msg":"no targets to scan"}], 400)

        # -------- engine options --------
        lat = body.get("latitude")
        lon = body.get("longitude")
        topocentric = (bool(body.get("topocentric")) or 
                      (isinstance(lat, (int, float)) and isinstance(lon, (int, float))))
        
        # Only convert if actually numeric
        lat = float(lat) if isinstance(lat, (int, float)) else None
        lon = float(lon) if isinstance(lon, (int, float)) else None
        elev = float(body.get("elevation_m")) if isinstance(body.get("elevation_m"), (int, float)) else None
        
        frame = parse_frame(body.get("frame"))
        step_arg = _parse_step_minutes(body.get("step_minutes"), default_min=30.0)

        if topocentric and (lat is not None or lon is not None):
            try:
                lat, lon = parse_latlon(lat, lon)
            except ValidationError as e:
                return _json_error("validation_error", e.errors(), 400)

        include_antiscia = bool(body.get("include_antiscia", False))
        antiscia_orb_deg = float(body.get("antiscia_orb_deg", 2.0))
        include_minors = bool(body.get("include_minors", False))

        # sidereal options (thread through to engine)
        zodiac_mode = (body.get("zodiac_mode") or "tropical").lower()
        ayanamsa_deg = float(body.get("ayanamsa_deg", 0.0))

        # Use pre-computed aspect sets
        aspects = _ALL_ASPECTS if include_minors else _MAJOR_ASPECTS

        # -------- run engine with shared adapter --------
        try:
            shared_adapter = get_shared_adapter(frame)
            eng = pred.TransitEngine(
                ephem=shared_adapter,
                frame=frame,
                topocentric=topocentric,
                latitude=lat,
                longitude=lon,
                elevation_m=elev
            )

            # thread sidereal into engine (no API change)
            eng.sidereal_mode = zodiac_mode.startswith("sidereal")
            eng.ayanamsa_deg = ayanamsa_deg

            events = eng.scan_aspects(
                jd_start_tt=float(jd0),
                jd_end_tt=float(jd1),
                movers=movers,  # Already strings, no need to convert again
                targets=targets,
                aspects=aspects,
                step_minutes=step_arg,  # supports "auto"
                include_antiscia=include_antiscia,
                antiscia_orb_deg=antiscia_orb_deg,
            )
        except Exception as e:
            return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

        # Build response more efficiently
        out = [
            {
                "jd_tt": e.jd_tt,
                "body": e.body,
                "target": e.target,
                "aspect": e.aspect,
                "kind": e.kind,
                "separation_deg": e.separation_deg,
                "applying": e.applying,
                "exact": e.exact,
                "meta": e.meta,
            }
            for e in events
        ]

        resp = jsonify({
            "ok": True,
            "window": {
                "jd_start_tt": float(jd0),
                "jd_end_tt": float(jd1),
                "step_minutes": (step_arg if isinstance(step_arg, float) else "auto"),
            },
            "engine": {
                "frame": frame, 
                "topocentric": topocentric, 
                "zodiac_mode": zodiac_mode, 
                "ayanamsa_deg": ayanamsa_deg
            },
            "targets": targets,
            "movers": movers,
            "results": out
        })
        resp.status_code = 200
        resp.headers["X-Compute-Time-ms"] = f"{(time.perf_counter() - t0)*1000:.0f}"
        return resp

    finally:
        _give_gate()
# ───────────────────────── /predictive/ingresses ─────────────────────────
@api.post("/api/predictive/ingresses")
@rate_limit(RL_PREDICTIVE)
def predictive_ingresses():
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    if not _take_gate():
        return _busy()

    t_wall = time.perf_counter()
    try:
        # inputs / validation
        try:
            jd0, jd1 = _parse_time_range_like(body)
        except ValidationError as e:
            return _json_error("validation_error", e.errors(), 400)
        except Exception as e:
            return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

        raw_movers = body.get("movers") or ["Sun","Mercury","Venus","Mars","Jupiter","Saturn"]
        movers = [m.strip() for m in raw_movers if m and str(m).strip()]
        if not movers:
            return _json_error("validation_error", [{"msg":"movers cannot be empty"}], 400)

        frame = parse_frame(body.get("frame"))
        base_adapter = get_shared_adapter(frame)

        # observing options
        obs = dict(
            topocentric=bool(body.get("topocentric")),
            latitude=(float(body["latitude"]) if isinstance(body.get("latitude"), (int, float)) else None),
            longitude=(float(body["longitude"]) if isinstance(body.get("longitude"), (int, float)) else None),
            elevation_m=(float(body["elevation_m"]) if isinstance(body.get("elevation_m"), (int, float)) else None),
        )

        # sidereal/tropical
        zodiac_mode = (body.get("zodiac_mode") or "tropical").lower()
        sidereal = zodiac_mode.startswith("sidereal")
        ayanamsa_deg = float(body.get("ayanamsa_deg", 0.0))

        # step sizing
        step_arg = _parse_step_minutes(body.get("step_minutes"), default_min=60.0)

        # per-request ephemeris with memo
        E = _PerRequestEphem(base_adapter, obs, sidereal=sidereal, ay_deg=ayanamsa_deg)

        def _sign_idx(lon_deg: float) -> int:
            return int(math.floor(_norm360(lon_deg) / 30.0)) % 12

        def _auto_step_minutes(names: list[str]) -> float:
            # slightly bolder autos (Brent refinement guarantees accuracy)
            caps = []
            for m in names:
                n = (m or "").lower()
                if n == "moon": caps.append(20)                        # was 10
                elif n in ("mercury", "venus", "mars"): caps.append(60)
                elif n in ("sun", "jupiter", "saturn"): caps.append(120)
                else: caps.append(180)
            return float(max(15, min(caps) if caps else 90))

        # Brent–Dekker on wrapped delta
        def _refine_zero_brent(f, a, b, fa, fb, *, max_iter=64, tol_days=1e-6) -> float:
            if fa == 0.0: return a
            if fb == 0.0: return b
            # bracket if needed via bisection
            if fa * fb > 0.0:
                aa, bb = a, b
                for _ in range(max_iter):
                    m = 0.5 * (aa + bb)
                    fm = f(m)
                    if fm == 0.0 or (bb - aa) <= tol_days:
                        return m
                    if fa * fm <= 0.0:
                        bb, fb = m, fm
                    else:
                        aa, fa = m, fm
                return 0.5 * (aa + bb)
            c, fc = a, fa
            d = e = b - a
            for _ in range(max_iter):
                if abs(fb) < abs(fa):
                    a, b = b, a; fa, fb = fb, fa
                m = 0.5 * (a + b)
                tol = tol_days
                if abs(b - a) <= tol:
                    return b
                # inverse quadratic or secant
                if fa != fc and fb != fc:
                    s = (a*fb*fc)/((fa-fb)*(fa-fc)) + (b*fa*fc)/((fb-fa)*(fb-fc)) + (c*fa*fb)/((fc-fa)*(fc-fb))
                else:
                    s = b - fb*(b-a)/(fb-fa)
                # acceptability checks; else bisection
                cond = not ((3*a + b)/4 < s < b if a < b else b < s < (3*a + b)/4)
                cond |= (e and abs(s - b) >= abs(e) / 2)
                cond |= (not e and abs(s - b) >= abs(d) / 2)
                cond |= (abs(e) < tol) or (abs(d) < tol)
                if cond:
                    s = m
                    d = e = b - a
                else:
                    d, e = e, b - s
                fs = f(s)
                c, fc = a, fa
                if (fa * fs) < 0:
                    b, fb = s, fs
                else:
                    a, fa = s, fs
                if abs(fa) < abs(fb):
                    a, b = b, a; fa, fb = fb, fa
            return b

        # core scan
        step_minutes = _auto_step_minutes(movers) if isinstance(step_arg, str) else (
            float(step_arg) if float(step_arg) > 0.0 else _auto_step_minutes(movers)
        )
        dt = float(step_minutes) / (24.0 * 60.0)

        t0 = float(jd0)
        events: list[dict[str, object]] = []
        dedupe: set[tuple[str, int, int]] = set()

        l0 = E.map(t0, movers)
        s0 = {k: _sign_idx(v) for k, v in l0.items()}

        while t0 < jd1 - 1e-12:
            t1 = min(t0 + dt, jd1)
            l1 = E.map(t1, movers)

            for body in movers:
                if body not in l0 or body not in l1:
                    continue
                a = float(l0[body]); b = float(l1[body])
                s_prev = int(s0.get(body, _sign_idx(a)))
                s_next = _sign_idx(b)
                if s_prev == s_next:
                    continue

                # direction via shortest-path delta
                forward = (_wrap180(b - a) > 0.0)

                # boundary
                boundary = (s_prev + 1) % 12 if forward else s_prev
                edge_deg = 30.0 * boundary

                def f(tt: float) -> float:
                    lm = E.one(tt, body)
                    return 0.0 if lm is None else _wrap180(float(lm) - edge_deg)

                fa = _wrap180(a - edge_deg)
                fb = _wrap180(b - edge_deg)

                # try to shrink bracket using linear estimate
                ta, tb = t0, t1
                den = _wrap180(b - a)
                if den != 0.0:
                    frac = _wrap180(edge_deg - a) / den
                    if 0.0 <= frac <= 1.0:
                        t_est = t0 + frac * (t1 - t0)
                        pad = 0.25 * (t1 - t0)
                        ta, tb = max(t0, t_est - pad), min(t1, t_est + pad)
                        la = E.one(ta, body); lb = E.one(tb, body)
                        if la is not None and lb is not None:
                            fa2 = _wrap180(float(la) - edge_deg)
                            fb2 = _wrap180(float(lb) - edge_deg)
                            if fa2 * fb2 <= 0.0:
                                fa, fb = fa2, fb2

                # alt edge if unbracketed
                if fa == 0.0:
                    t_exact = t0
                elif fb == 0.0:
                    t_exact = t1
                elif fa * fb > 0.0:
                    alt_edge = 30.0 * ((boundary + (1 if forward else -1)) % 12)
                    fa2 = _wrap180(a - alt_edge)
                    fb2 = _wrap180(b - alt_edge)
                    if fa2 * fb2 > 0.0:
                        continue
                    def f2(tt: float) -> float:
                        lm = E.one(tt, body)
                        return 0.0 if lm is None else _wrap180(float(lm) - alt_edge)
                    t_exact = _refine_zero_brent(f2, ta, tb, fa2, fb2, tol_days=1e-6)
                    edge_deg = alt_edge
                else:
                    t_exact = _refine_zero_brent(f, ta, tb, fa, fb, tol_days=1e-6)

                n_exact = E.one(t_exact, body)
                if n_exact is None:
                    continue
                from_sign = s_prev
                to_sign = _sign_idx(n_exact)

                bucket = int(math.floor(t_exact * 86400.0 + 0.5))
                key = (body, bucket, int(edge_deg))
                if key in dedupe:
                    continue
                dedupe.add(key)

                events.append({
                    "jd_tt": float(t_exact),
                    "body": body,
                    "from_sign": int(from_sign),
                    "to_sign": int(to_sign),
                    "longitude_deg": float(n_exact),
                })

            # slide window
            t0 = t1
            l0 = l1
            s0 = {k: _sign_idx(v) for k, v in l0.items()}

        events.sort(key=lambda e: (e["jd_tt"], e["body"]))
        resp = jsonify({
            "ok": True,
            "window": {
                "jd_start_tt": float(jd0),
                "jd_end_tt": float(jd1),
                "step_minutes": (step_minutes if not isinstance(step_arg, str) else "auto"),
            },
            "engine": {
                "frame": frame,
                "topocentric": bool(obs["topocentric"]),
                "zodiac_mode": zodiac_mode,
                "ayanamsa_deg": ayanamsa_deg,
            },
            "movers": movers,
            "results": events,
        })
        resp.status_code = 200
        resp.headers["X-Compute-Time-ms"] = f"{(time.perf_counter() - t_wall)*1000:.0f}"
        resp.headers["X-Adapter-Calls"] = str(int(E.calls))
        return resp

    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)
    finally:
        _give_gate()

# ───────────────────────── /predictive/stations ─────────────────────────
@api.post("/api/predictive/stations")
@rate_limit(RL_PREDICTIVE)
def predictive_stations():
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    if not _take_gate():
        return _busy()
    t_wall = time.perf_counter()

    try:
        # ── window ────────────────────────────────────────────────────────────
        try:
            jd0, jd1 = _parse_time_range_like(body)
        except ValidationError as e:
            return _json_error("validation_error", e.errors(), 400)
        except Exception as e:
            return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

        # ── inputs ───────────────────────────────────────────────────────────
        raw_movers = body.get("movers") or ["Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"]
        movers = [str(m).strip() for m in raw_movers if m and str(m).strip()]
        if not movers:
            return _json_error("validation_error", [{"msg": "movers cannot be empty"}], 400)

        frame = parse_frame(body.get("frame"))
        shared_adapter = get_shared_adapter(frame)

        obs = dict(
            topocentric=bool(body.get("topocentric")),
            latitude=(float(body["latitude"]) if isinstance(body.get("latitude"), (int, float)) else None),
            longitude=(float(body["longitude"]) if isinstance(body.get("longitude"), (int, float)) else None),
            elevation_m=(float(body["elevation_m"]) if isinstance(body.get("elevation_m"), (int, float)) else None),
        )

        zodiac_mode = (body.get("zodiac_mode") or "tropical").lower()
        ayanamsa_deg = float(body.get("ayanamsa_deg", 0.0))
        sidereal = zodiac_mode.startswith("sidereal")
        ay = float(ayanamsa_deg if sidereal else 0.0)

        step_arg = _parse_step_minutes(body.get("step_minutes"), default_min=60.0)

        # ── helpers ──────────────────────────────────────────────────────────
        TAU = 360.0

        def _norm360(x: float) -> float:
            r = x % TAU
            return r + TAU if r < 0.0 else r

        def _wrap180(x: float) -> float:
            v = ((x + 180.0) % 360.0) - 180.0
            return 180.0 if v == -180.0 else v

        def _nir(lon_trop: float) -> float:
            return _norm360(float(lon_trop) - ay)

        # per-request caches
        lon_cache: dict[tuple[str, float], float] = {}   # (body, round(jd,6))
        spd_cache: dict[tuple[str, float, float], float] = {}  # (body, round(jd,6), h)

        def _k(jd: float) -> float:
            # ~0.0864 s bucket
            return round(float(jd), 6)

        def _lon_map(jd_tt: float, names: list[str]) -> dict[str, float]:
            rows = shared_adapter.ecliptic_longitudes(jd_tt, names, **obs).get("results", [])
            out: dict[str, float] = {}
            for row in rows or []:
                nm = row["name"]
                lon = float(row["longitude"])
                if sidereal:
                    lon = _nir(lon)
                out[nm] = lon
                lon_cache[(nm, _k(jd_tt))] = lon
            return out

        def _lon_one(jd_tt: float, body: str) -> float | None:
            kk = (body, _k(jd_tt))
            if kk in lon_cache:
                return lon_cache[kk]
            rows = shared_adapter.ecliptic_longitudes(jd_tt, [body], **obs).get("results", [])
            if not rows:
                return None
            lon = float(rows[0]["longitude"])
            if sidereal:
                lon = _nir(lon)
            lon_cache[kk] = lon
            return lon

        def _auto_step_minutes(names: list[str]) -> float:
            # Stations are slow; coarser is fine; refinement finds exact t
            caps = []
            for m in names:
                n = (m or "").lower()
                if n in ("mercury", "venus"): caps.append(90)   # inner planets: still coarse; refine later
                elif n in ("mars",):           caps.append(180)
                else:                          caps.append(360)
            return float(max(30, min(caps) if caps else 180))

        # central-difference speed (deg/day) with wrap
        def _speed(body: str, t: float, h: float, lo: float, hi: float) -> float:
            key = (body, _k(t), h)
            if key in spd_cache:
                return spd_cache[key]
            t_m = max(lo, t - h); t_p = min(hi, t + h)
            if t_p - t_m < 1e-9:  # widen a bit if degenerate
                t_m = max(lo, t - 2*h); t_p = min(hi, t + 2*h)
            la = _lon_one(t_m, body); lb = _lon_one(t_p, body)
            if la is None or lb is None:
                v = float("nan")
            else:
                diff = _wrap180(float(lb) - float(la))
                dt = (t_p - t_m)
                v = diff / dt if dt > 0.0 else float("nan")
            spd_cache[key] = v
            return v

        # Quadratic (parabolic) refine using three samples around mid
        def _refine_parabolic(body: str, a: float, b: float, h: float, lo: float, hi: float) -> float | None:
            m = 0.5*(a+b)
            # sample speeds at a, m, b
            s_a = _speed(body, a, h, lo, hi)
            s_m = _speed(body, m, h, lo, hi)
            s_b = _speed(body, b, h, lo, hi)
            if not (math.isfinite(s_a) and math.isfinite(s_m) and math.isfinite(s_b)):
                return None
            # fit parabola through (a,s_a), (m,s_m), (b,s_b)
            # x' ∈ { -1,0,1 } after mapping t ∈ {a,m,b} to x' = (t - m)/(b-a)/0.5
            # The vertex (zero) x0' = (s_a - s_b) / (2*(s_a - 2*s_m + s_b)) (if denom ≠ 0)
            denom = (s_a - 2*s_m + s_b)
            if abs(denom) < 1e-12:
                return None
            x0p = (s_a - s_b) / (2.0 * denom)
            if -1.5 <= x0p <= 1.5:
                t0 = m + x0p * (b - a) * 0.5
                return max(lo, min(hi, t0))
            return None

        # Brent–Dekker fallback on s(t)
        def _refine_zero_brent(body: str, a: float, b: float, fa: float, fb: float, *, h: float, lo: float, hi: float,
                               max_iter=64, tol_days=1e-6) -> float | None:
            if not (math.isfinite(fa) and math.isfinite(fb)):
                return None
            if fa == 0.0: return a
            if fb == 0.0: return b
            # ensure bracket by bisection if needed
            if fa * fb > 0.0:
                aa, bb = a, b
                for _ in range(32):
                    m = 0.5*(aa+bb)
                    fm = _speed(body, m, h, lo, hi)
                    if not math.isfinite(fm):
                        break
                    if fm == 0.0 or abs(bb-aa) <= tol_days:
                        return m
                    if fa * fm <= 0.0:
                        bb, fb = m, fm
                    else:
                        aa, fa = m, fm
                if fa * fb > 0.0:
                    return None
                a, b = aa, bb
            c, fc = a, fa
            d = e = b - a
            for _ in range(max_iter):
                if abs(fb) < abs(fa):
                    a, b = b, a; fa, fb = fb, fa
                m = 0.5*(a+b)
                tol = tol_days
                if abs(b-a) <= tol:
                    return b
                # inverse quadratic / secant
                if fa != fc and fb != fc:
                    s = (a*fb*fc)/((fa-fb)*(fa-fc)) + (b*fa*fc)/((fb-fa)*(fb-fc)) + (c*fa*fb)/((fc-fa)*(fc-fb))
                else:
                    s = b - fb*(b-a)/(fb-fa)
                # acceptability checks; else bisection
                cond = not ((3*a + b)/4 < s < b if a < b else b < s < (3*a + b)/4)
                cond |= (e and abs(s-b) >= abs(e)/2)
                cond |= (not e and abs(s-b) >= abs(d)/2)
                cond |= (abs(e) < tol) or (abs(d) < tol)
                if cond:
                    s = m
                    d = e = b - a
                else:
                    d, e = e, b - s
                fs = _speed(body, s, h, lo, hi)
                if not math.isfinite(fs):
                    s = m; fs = _speed(body, s, h, lo, hi)
                c, fc = a, fa
                if (fa * fs) < 0:
                    b, fb = s, fs
                else:
                    a, fa = s, fs
                if abs(fa) < abs(fb):
                    a, b = b, a; fa, fb = fb, fa
            return b

        # ── scan ──────────────────────────────────────────────────────────────
        if isinstance(step_arg, str):
            step_minutes = _auto_step_minutes(movers)
        else:
            step_minutes = float(step_arg) if float(step_arg) > 0.0 else _auto_step_minutes(movers)
        dt = float(step_minutes) / (24.0 * 60.0)

        # derivative half-stencil (tie to dt, clamp to ≤ 30m)
        h = min(0.5*dt, 30.0/(24.0*60.0)) or (15.0/(24.0*60.0))

        # warm caches at boundaries (batched)
        _ = _lon_map(float(jd0), movers)
        _ = _lon_map(float(jd1), movers)

        events: list[dict[str, object]] = []
        dedupe: set[tuple[str, int]] = set()

        t = float(jd0)
        while t < jd1 - 1e-12:
            t_next = min(t + dt, jd1)
            # warm at step endpoints (batched)
            _ = _lon_map(t, movers)
            _ = _lon_map(t_next, movers)

            for body in movers:
                s0 = _speed(body, t, h, jd0, jd1)
                s1 = _speed(body, t_next, h, jd0, jd1)
                if not (math.isfinite(s0) and math.isfinite(s1)):
                    continue

                # coarse detection: sign change or near zero at either end
                if not ((s0 == 0.0) or (s1 == 0.0) or (s0*s1 < 0.0) or (min(abs(s0),abs(s1)) <= 0.05)):
                    continue

                # parabolic refine first (cheap), then Brent fallback
                t_star = _refine_parabolic(body, t, t_next, h, jd0, jd1)
                if t_star is None:
                    t_star = _refine_zero_brent(body, t, t_next, s0, s1, h=h, lo=jd0, hi=jd1, tol_days=1e-6)
                if t_star is None or not (t - 1e-9 <= t_star <= t_next + 1e-9):
                    # fallback to minimum |s| among {t, mid, t_next}
                    mid = 0.5*(t + t_next)
                    sm = _speed(body, mid, h, jd0, jd1)
                    if not math.isfinite(sm):
                        continue
                    t_star = min([(abs(s0), t), (abs(sm), mid), (abs(s1), t_next)], key=lambda x: x[0])[1]

                lon_star = _lon_one(t_star, body)
                if lon_star is None:
                    continue

                # determine SR / SD (direction change)
                eps = max(1.0/(24.0*60.0), 0.25*h)  # ≥ 1 minute
                sb = _speed(body, max(jd0, t_star - eps), h, jd0, jd1)
                sa = _speed(body, min(jd1, t_star + eps), h, jd0, jd1)
                kind, direction = "station", None
                if math.isfinite(sb) and math.isfinite(sa):
                    if sb > 0 and sa < 0:
                        kind, direction = "SR", "retrograde"
                    elif sb < 0 and sa > 0:
                        kind, direction = "SD", "direct"

                bucket = int(round(float(t_star) * 86400.0))
                key = (body, bucket)
                if key in dedupe:
                    continue
                dedupe.add(key)

                events.append({
                    "jd_tt": float(t_star),
                    "body": body,
                    "kind": kind,
                    "direction": direction,
                    "longitude_deg": float(lon_star),
                    "speed_before_deg_per_day": float(sb) if math.isfinite(sb) else None,
                    "speed_after_deg_per_day": float(sa) if math.isfinite(sa) else None,
                })

            t = t_next

        events.sort(key=lambda e: (e["jd_tt"], e["body"]))
        resp = jsonify({
            "ok": True,
            "window": {
                "jd_start_tt": float(jd0),
                "jd_end_tt": float(jd1),
                "step_minutes": (step_minutes if not isinstance(step_arg, str) else "auto"),
            },
            "engine": {
                "frame": frame,
                "topocentric": bool(obs["topocentric"]),
                "zodiac_mode": zodiac_mode,
                "ayanamsa_deg": ayanamsa_deg,
            },
            "movers": movers,
            "results": events,
        })
        resp.status_code = 200
        resp.headers["X-Compute-Time-ms"] = f"{(time.perf_counter() - t_wall)*1000:.0f}"
        return resp

    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)
    finally:
        _give_gate()


@api.post("/api/predictive/evaluate")
@rate_limit(RL_PREDICTIVE)
def predictive_evaluate():
    """
    Univariate permutation test with optional time-series aware permutations.
    """

    # ── Local helpers ────────────────────────────────────────────────
    PLANET_NORMALIZATION = {
        'sun': 'Sun', 'Sun': 'Sun', 'SUN': 'Sun',
        'moon': 'Moon', 'Moon': 'Moon', 'MOON': 'Moon',
        'mercury': 'Mercury', 'Mercury': 'Mercury', 'MERCURY': 'Mercury',
        'venus': 'Venus', 'Venus': 'Venus', 'VENUS': 'Venus',
        'mars': 'Mars', 'Mars': 'Mars', 'MARS': 'Mars',
        'jupiter': 'Jupiter', 'Jupiter': 'Jupiter', 'JUPITER': 'Jupiter',
        'saturn': 'Saturn', 'Saturn': 'Saturn', 'SATURN': 'Saturn',
        'uranus': 'Uranus', 'Uranus': 'Uranus', 'URANUS': 'Uranus',
        'neptune': 'Neptune', 'Neptune': 'Neptune', 'NEPTUNE': 'Neptune',
        'pluto': 'Pluto', 'Pluto': 'Pluto', 'PLUTO': 'Pluto',
        'ascendant': 'Ascendant', 'Ascendant': 'Ascendant', 'ASCENDANT': 'Ascendant',
        'mc': 'MC', 'MC': 'MC', 'midheaven': 'MC', 'Midheaven': 'MC', 'MIDHEAVEN': 'MC',
    }

    def dup_titlecase_and_lower(name: str, value):
        canonical = PLANET_NORMALIZATION.get(name, name)
        if canonical.islower():
            canonical = canonical.capitalize()
        return [(canonical, value), (canonical.lower(), value)]

    def normalize_lons(lons: dict) -> dict:
        out = {}
        for k, v in (lons or {}).items():
            for kk, vv in dup_titlecase_and_lower(str(k), v):
                out[kk] = vv
        return out

    def normalize_records(records: list) -> list:
        norm = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            r = dict(rec)
            if "natal_longitudes" in r and isinstance(r["natal_longitudes"], dict):
                r["natal_longitudes"] = normalize_lons(r["natal_longitudes"])
            norm.append(r)
        return norm

    def normalize_movers(movers: list) -> list:
        out = []
        for m in (movers or []):
            canonical = PLANET_NORMALIZATION.get(m, m)
            if canonical.islower():
                canonical = canonical.capitalize()
            if canonical not in out:
                out.append(canonical)
        return out

    def clamp_int(x, lo, hi, default):
        try:
            v = int(x)
        except Exception:
            v = default
        return max(lo, min(hi, v))

    # ── Body parsing ────────────────────────────────────────────────
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    recs = body.get("records") or []
    if not (isinstance(recs, list) and recs):
        return _json_error("validation_error", [{"loc": ["records"], "msg": "non-empty list required"}], 400)

    normalized_recs = normalize_records(recs)

    feature = str(body.get("feature") or "transit_proximity").lower()
    perm_mode = str(body.get("perm_mode") or "iid").lower()
    group_by = body.get("group_by")
    alpha = float(body.get("alpha", 0.05))
    n_perm = clamp_int(body.get("n_perm", 2000), 1, 10000, 2000)

    from app.core import predictive as pred

    # ── Feature selection ───────────────────────────────────────────
    try:
        if feature == "transit_proximity":
            movers = normalize_movers(body.get("movers") or ["Sun", "Moon", "Mercury", "Venus", "Mars"])
            orb_deg = float(body.get("orb_deg", 1.0))
            ff = pred.feature_transit_proximity(movers=movers, orb_deg=orb_deg)

        elif feature == "dasha_lords_onehot":
            level = clamp_int(body.get("level", 1), 1, 3, 1)
            ff = pred.feature_dasha_lords_onehot(level=level)

        elif feature.startswith("dasha_l") and len(feature) == 8 and feature[7].isdigit():
            level = clamp_int(feature[7], 1, 3, 1)
            ff = pred.feature_dasha_lords_onehot(level=level)

        elif feature == "yoga_flags":
            names = body.get("yoga_names") or ["panch_mahapurusha", "gajakesari", "chandra_mangal", "parivartana"]
            ff = pred.feature_yoga_flags(names)

        elif feature == "angular_houses":
            return _json_error("validation_error", [{"loc": ["feature"], "msg": "angular_houses feature not implemented"}], 400)

        else:
            return _json_error("validation_error", [{"loc": ["feature"], "msg": "unknown feature"}], 400)

    except Exception as e:
        return _json_error("validation_error", [{"loc": ["feature"], "msg": str(e)}], 400)

    # ── Run evaluation ──────────────────────────────────────────────
    try:
        res = pred.evaluate_univariate(
            normalized_recs, ff,
            n_perm=n_perm, alpha=alpha,
            perm_mode=perm_mode, group_by=group_by,
            use_time=bool(body.get("use_time", True)),
            seed=body.get("seed")
        )
        return jsonify({"ok": True, "results": [getattr(r, "__dict__", r) for r in res]}), 200
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

@api.post("/api/predictive/holdout")
@rate_limit(RL_PREDICTIVE)
def predictive_holdout():
    """Train/holdout replication with the same feature menu as /evaluate."""

    # ── Local helpers (same as above) ───────────────────────────────
    PLANET_NORMALIZATION = {
        'sun': 'Sun', 'Sun': 'Sun', 'SUN': 'Sun',
        'moon': 'Moon', 'Moon': 'Moon', 'MOON': 'Moon',
        'mercury': 'Mercury', 'Mercury': 'Mercury', 'MERCURY': 'Mercury',
        'venus': 'Venus', 'Venus': 'Venus', 'VENUS': 'Venus',
        'mars': 'Mars', 'Mars': 'Mars', 'MARS': 'Mars',
        'jupiter': 'Jupiter', 'Jupiter': 'Jupiter', 'JUPITER': 'Jupiter',
        'saturn': 'Saturn', 'Saturn': 'Saturn', 'SATURN': 'Saturn',
        'uranus': 'Uranus', 'Uranus': 'Uranus', 'URANUS': 'Uranus',
        'neptune': 'Neptune', 'Neptune': 'Neptune', 'NEPTUNE': 'Neptune',
        'pluto': 'Pluto', 'Pluto': 'Pluto', 'PLUTO': 'Pluto',
        'ascendant': 'Ascendant', 'Ascendant': 'Ascendant', 'ASCENDANT': 'Ascendant',
        'mc': 'MC', 'MC': 'MC', 'midheaven': 'MC', 'Midheaven': 'MC', 'MIDHEAVEN': 'MC',
    }

    def dup_titlecase_and_lower(name: str, value):
        canonical = PLANET_NORMALIZATION.get(name, name)
        if canonical.islower():
            canonical = canonical.capitalize()
        return [(canonical, value), (canonical.lower(), value)]

    def normalize_lons(lons: dict) -> dict:
        out = {}
        for k, v in (lons or {}).items():
            for kk, vv in dup_titlecase_and_lower(str(k), v):
                out[kk] = vv
        return out

    def normalize_records(records: list) -> list:
        norm = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            r = dict(rec)
            if "natal_longitudes" in r and isinstance(r["natal_longitudes"], dict):
                r["natal_longitudes"] = normalize_lons(r["natal_longitudes"])
            norm.append(r)
        return norm

    def normalize_movers(movers: list) -> list:
        out = []
        for m in (movers or []):
            canonical = PLANET_NORMALIZATION.get(m, m)
            if canonical.islower():
                canonical = canonical.capitalize()
            if canonical not in out:
                out.append(canonical)
        return out

    def clamp_int(x, lo, hi, default):
        try:
            v = int(x)
        except Exception:
            v = default
        return max(lo, min(hi, v))

    # ── Body parsing ────────────────────────────────────────────────
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    recs = body.get("records") or []
    if not (isinstance(recs, list) and recs):
        return _json_error("validation_error", [{"loc": ["records"], "msg": "non-empty list required"}], 400)

    normalized_recs = normalize_records(recs)

    feature = str(body.get("feature") or "transit_proximity").lower()
    perm_mode = str(body.get("perm_mode") or "iid").lower()
    group_by = body.get("group_by")

    from app.core import predictive as pred

    # ── Feature selection ───────────────────────────────────────────
    try:
        if feature == "transit_proximity":
            movers = normalize_movers(body.get("movers") or ["Sun", "Moon", "Mercury", "Venus", "Mars"])
            orb_deg = float(body.get("orb_deg", 1.0))
            ff = pred.feature_transit_proximity(movers=movers, orb_deg=orb_deg)

        elif feature == "dasha_lords_onehot":
            level = clamp_int(body.get("level", 1), 1, 3, 1)
            ff = pred.feature_dasha_lords_onehot(level=level)

        elif feature.startswith("dasha_l") and len(feature) == 8 and feature[7].isdigit():
            level = clamp_int(feature[7], 1, 3, 1)
            ff = pred.feature_dasha_lords_onehot(level=level)

        elif feature == "yoga_flags":
            names = body.get("yoga_names") or ["panch_mahapurusha", "gajakesari", "chandra_mangal", "parivartana"]
            ff = pred.feature_yoga_flags(names)

        elif feature == "angular_houses":
            return _json_error("validation_error", [{"loc": ["feature"], "msg": "angular_houses feature not implemented"}], 400)

        else:
            return _json_error("validation_error", [{"loc": ["feature"], "msg": "unknown feature"}], 400)

    except Exception as e:
        return _json_error("validation_error", [{"loc": ["feature"], "msg": str(e)}], 400)

    # ── Params ──────────────────────────────────────────────────────
    train_frac = float(body.get("train_frac", 0.7))
    if not (0.05 <= train_frac <= 0.95):
        train_frac = 0.7
    alpha = float(body.get("alpha", 0.05))
    n_perm_train = clamp_int(body.get("n_perm_train", 2000), 1, 10000, 2000)
    n_perm_test = clamp_int(body.get("n_perm_test", 4000), 1, 20000, 4000)

    # ── Run holdout ─────────────────────────────────────────────────
    try:
        res = pred.holdout_replicate(
            normalized_recs, ff,
            train_frac=train_frac,
            alpha=alpha,
            n_perm_train=n_perm_train,
            n_perm_test=n_perm_test,
            perm_mode=perm_mode, group_by=group_by,
            use_time=bool(body.get("use_time", True)),
            seed=body.get("seed"),
        )
        return jsonify({"ok": True, **res}), 200
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

@api.post("/api/predictive/dasha")
@rate_limit(RL_PREDICTIVE)
def predictive_dasha():
    try:
        body = request.get_json(force=True) or {}
        from app.core import predictive as pred
        # Accept direct jd_tt or resolve from civil
        if isinstance(body.get("birth_jd_tt"), (int,float)):
            birth_jd_tt = float(body["birth_jd_tt"])
        else:
            tz = body.get("place_tz") or body.get("timezone") or "UTC"
            ts = _compute_timescales_from_local(body["birth_date"], body["birth_time"], tz, payload=body)
            birth_jd_tt = float(ts["jd_tt"])
        moon_lon_tropical = float(body["moon_lon_tropical_deg"])
        ay = float(body.get("ayanamsa_deg", 0.0))
        levels = int(body.get("levels", 3))
        span_years = float(body.get("span_years", 120.0))
        res = pred.vimsottari_dasha(
            birth_jd_tt=birth_jd_tt,
            moon_lon_tropical_deg=moon_lon_tropical,
            ayanamsa_deg=ay, levels=levels, span_years=span_years
        )
        return jsonify({"ok": True, "results": [r.__dict__ for r in res]}), 200
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/predictive/vargas")
@rate_limit(RL_PREDICTIVE)
def predictive_vargas():
    try:
        body = request.get_json(force=True) or {}
        from app.core import predictive as pred
        pts = body.get("points_deg") or {}
        pts2: Dict[str, float] = {}
        for k, v in pts.items():
            try:
                pts2[str(k)] = _wrap360(float(v))
            except Exception:
                pass
        if not pts2:
            return _json_error("validation_error", [{"loc":["points_deg"],"msg":"empty/invalid"}], 400)
        mode = (body.get("zodiac_mode") or "sidereal").lower()
        ay = float(body.get("ayanamsa_deg", 0.0))
        include = body.get("include") or ["D1","D2","D3","D9","D10","D12"]
        res = pred.compute_vargas(points_deg=pts2, zodiac_mode=("sidereal" if mode.startswith("sid") else "tropical"), ayanamsa_deg=ay, include=include)
        return jsonify({"ok": True, "results": res}), 200
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/predictive/yogas")
@rate_limit(RL_PREDICTIVE)
def predictive_yogas():
    try:
        body = request.get_json(force=True) or {}
        from app.core import predictive as pred
        pts = body.get("points_deg") or {}
        cusps = body.get("cusps_deg") or []
        pts2: Dict[str, float] = {}
        for k, v in pts.items():
            try:
                pts2[str(k)] = _wrap360(float(v))
            except Exception:
                pass
        cusps2: List[float] = []
        for x in cusps:
            if isinstance(x, (int,float)): cusps2.append(_wrap360(float(x)))
        if not pts2 or len(cusps2) != 12:
            return _json_error("validation_error", [{"loc":["points_deg|cusps_deg"],"msg":"need points and 12 cusps"}], 400)
        include = body.get("include") or ["panch_mahapurusha","gajakesari","chandra_mangal","parivartana"]
        res = pred.detect_yogas(points_deg=pts2, cusps_deg=cusps2, include=include, orbs=body.get("orbs") or {})
        return jsonify({"ok": True, "results": res}), 200
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

        
# ───────────────────────── PROGRESSIONS ─────────────────────────
@api.post("/api/progressions")
@rate_limit(RL_PROGRESSIONS)
def progressions_route():
    """
    Progressions: secondary / minor / tertiary.
    Validates payload, normalizes aliases, fills natal JDs if missing,
    filters kwargs to the engine's signature, and returns enriched meta.
    """
    if compute_progressions is None:
        return _json_error("progressions_unavailable", "progressions engine not wired", 501)

    # ---------- helpers: normalize any TimeScales objects to plain dicts ----------
    def _timescales_to_dict(ts_obj: Any) -> Optional[Dict[str, Any]]:
        # Already dict-like?
        if isinstance(ts_obj, dict):
            return ts_obj
        # Our canonical type
        try:
            from app.core.timescales import TimeScales as _TS  # local import
            if isinstance(ts_obj, _TS):
                return {
                    "jd_utc": float(ts_obj.jd_utc),
                    "jd_tt": float(ts_obj.jd_tt),
                    "jd_ut1": float(ts_obj.jd_ut1),
                    "delta_t": float(ts_obj.delta_t),
                    "delta_at": float(ts_obj.dat),
                    "dut1": float(ts_obj.dut1),
                    "tz_offset_seconds": int(ts_obj.tz_offset_seconds),
                    "timezone": getattr(ts_obj, "tz_name", None) or getattr(ts_obj, "timezone", None),
                    "warnings": list(getattr(ts_obj, "warnings", []) or []),
                }
        except Exception:
            pass
        # Dataclass fallback
        try:
            if is_dataclass(ts_obj):
                return asdict(ts_obj)  # type: ignore[arg-type]
        except Exception:
            pass
        # __dict__ fallback
        try:
            if hasattr(ts_obj, "__dict__"):
                return dict(ts_obj.__dict__)
        except Exception:
            pass
        return None  # unknown shape

    def _strip_or_fix_timescales(container: Any) -> Any:
        """Return a shallow-copied dict with any *.timescales normalized or removed."""
        if not isinstance(container, dict):
            return container
        out = dict(container)
        for key in ("timescales", "ts", "TimeScales"):
            if key in out:
                fixed = _timescales_to_dict(out[key])
                if fixed is None:
                    out.pop(key, None)     # drop unrecognized object to avoid engine errors
                else:
                    out[key] = fixed
        return out

    # ---- parse & validate body ------------------------------------------------
    try:
        body = request.get_json(force=True) or {}
        payload = parse_progressions_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # ---- sanitize nested dicts that may carry TimeScales -----------------------
    natal_clean  = _strip_or_fix_timescales(payload.get("natal")  or {})
    target_clean = _strip_or_fix_timescales(payload.get("target") or {})
    place_clean  = _strip_or_fix_timescales(payload.get("place")  or {})

    # ---- seed kwargs from payload --------------------------------------------
    kwargs = {
        "natal": natal_clean,
        "method": payload.get("method", "secondary"),
        "target": (target_clean or None) if isinstance(payload.get("target"), dict) and target_clean else None,
        "years_after": payload.get("years_after"),
        "jd_tt_natal": payload.get("jd_tt_natal"),
        "jd_ut1_natal": payload.get("jd_ut1_natal"),
        "place": (place_clean or None) if isinstance(payload.get("place"), dict) and place_clean else None,
        "frame": payload.get("frame", "ecliptic-of-date"),
        "house_system": payload.get("house_system", "placidus"),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "lunar_month": payload.get("lunar_month", "synodic"),
        "tertiary_mode": payload.get("tertiary_mode", "day-for-month"),
        "aspects_to_natal": bool(payload.get("aspects_to_natal", True)),
        "orbs": payload.get("orbs") or None,
        "parallels": bool(payload.get("parallels", False)),
        "antiscia": bool(payload.get("antiscia", False)),
        "profile": bool(payload.get("profile", False)),
        "validation": payload.get("validation", "basic"),
    }

    # ---- merge nested flags (if client sent payload.flags) --------------------
    f = payload.get("flags")
    if isinstance(f, dict):
        if "aspects_to_natal" in f:
            kwargs["aspects_to_natal"] = bool(kwargs.get("aspects_to_natal") or f.get("aspects_to_natal", False))
        for k in ("parallels", "antiscia", "profile"):
            if k in f:
                kwargs[k] = bool(kwargs.get(k) or f.get(k, False))
        if "orbs" in f and kwargs.get("orbs") is None and isinstance(f["orbs"], dict):
            kwargs["orbs"] = f["orbs"]

    # ---- fill natal timescales if not provided --------------------------------
    try:
        if (kwargs.get("jd_tt_natal") is None or kwargs.get("jd_ut1_natal") is None) and isinstance(kwargs["natal"], dict):
            nat = kwargs["natal"]
            d, t, tz = nat.get("date"), nat.get("time"), (nat.get("place_tz") or nat.get("timezone"))
            if isinstance(d, str) and isinstance(t, str) and isinstance(tz, str):
                ts_nat = _compute_timescales_from_local(d, t, tz, payload=nat)
                kwargs.setdefault("jd_tt_natal", float(ts_nat["jd_tt"]))
                kwargs.setdefault("jd_ut1_natal", float(ts_nat["jd_ut1"]))
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        if DEBUG_VERBOSE:
            return _json_error("timescales_error", {"type": type(e).__name__, "message": str(e)}, 400)
        pass

    # ---- filter to engine signature & map aliases -----------------------------
    try:
        import inspect
        engine_params = set(inspect.signature(compute_progressions).parameters.keys())

        # drop any accidental TimeScales again if user nested them deeper
        def _deep_clean(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _deep_clean(v) if k not in ("timescales", "ts", "TimeScales") else _timescales_to_dict(v) or None
                        for k, v in obj.items() if (k not in ("timescales", "ts", "TimeScales")) or (_timescales_to_dict(v) is not None)}
            return obj

        cleaned_kwargs = {k: _deep_clean(v) for k, v in kwargs.items()}
        safe_kwargs = {k: v for k, v in cleaned_kwargs.items() if k in engine_params}

        if "mode" in engine_params and "mode" not in safe_kwargs and "zodiac_mode" in cleaned_kwargs:
            safe_kwargs["mode"] = cleaned_kwargs["zodiac_mode"]

        for k in list(safe_kwargs.keys()):
            if safe_kwargs[k] is None and k not in ("years_after", "target"):
                safe_kwargs.pop(k, None)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("progressions_internal", det or "internal_error", 500)

    # ---- call engine ----------------------------------------------------------
    try:
        result = compute_progressions(**safe_kwargs)
    except ValueError as e:
        return _json_error("progressions_value_error", str(e), 400)
    except RuntimeError as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("progressions_internal", det or "internal_error", 500)
    except TypeError as e:
        det = {"type": "TypeError", "message": str(e), "accepted": sorted(engine_params)} if DEBUG_VERBOSE else None
        return _json_error("progressions_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("progressions_internal", det or "internal_error", 500)

    # ---- meta enrichment (adapter snapshot) -----------------------------------
    meta = dict(result.get("meta") or {})
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # ---- response -------------------------------------------------------------
    resp = {
        "ok": True,
        "mapping": (meta.get("mapping") if isinstance(meta.get("mapping"), dict) else None),
        "meta": meta,
        "warnings": list((meta.get("warnings") or [])),
        "epoch": result.get("epoch"),
        "positions": result.get("positions"),
        "houses": result.get("houses"),
        "aspects_to_natal": result.get("aspects_to_natal") or [],
    }
    return jsonify(resp), 200

# ───────────────────────── RETURNS (rewritten) ─────────────────────────
def _returns_pick_fn(kind: str) -> Optional[Any]:
    """
    Choose a function from the returns module based on intent.
    For 'compute': prefer one-shot compute fn; for 'scan': prefer window scanner.
    """
    if not _returns_available():
        return None

    candidates: list[str] = []
    if kind == "compute":
        candidates = [
            # primary / canonical
            "compute_return",
            # common alternates seen in legacy modules
            "run_return_api", "calculate_return", "compute_returns",
            "solar_return", "lunar_return",
        ]
    elif kind == "scan":
        candidates = ["scan_returns", "find_returns", "scan", "search_returns"]

    for name in candidates:
        fn = getattr(_returns_mod, name, None)
        if callable(fn):
            return fn
    return None

def _parse_return_kind(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return "solar"
    alias = {
        "sun": "solar", "sol": "solar",
        "moon": "lunar", "lun": "lunar",
    }
    return alias.get(s, s)

def _normalize_window(body: Dict[str, Any], default_days: float = 30.0) -> Tuple[Optional[float], Optional[float]]:
    """Return (jd_start_tt, jd_end_tt) if resolvable from body; else (None, None)."""
    jd0 = body.get("jd_start_tt")
    jd1 = body.get("jd_end_tt")
    if isinstance(jd0, (int, float)) and isinstance(jd1, (int, float)):
        try:
            jd0f = float(jd0); jd1f = float(jd1)
            if jd1f > jd0f:
                return jd0f, jd1f
        except Exception:
            pass

    # Civil window
    date0 = body.get("date_start") or body.get("start_date")
    time0 = body.get("time_start") or "00:00:00"
    date1 = body.get("date_end") or body.get("end_date")
    time1 = body.get("time_end") or "23:59:59"
    tz = body.get("tz") or body.get("place_tz") or body.get("timezone")

    if isinstance(date0, str) and isinstance(time0, str) and isinstance(tz, str):
        ts0 = _compute_timescales_from_local(date0, time0, tz, payload=body)
        jd0f = float(ts0["jd_tt"])
        if isinstance(date1, str) and isinstance(time1, str):
            ts1 = _compute_timescales_from_local(date1, time1, tz, payload=body)
            jd1f = float(ts1["jd_tt"])
            if jd1f > jd0f:
                return jd0f, jd1f
        else:
            return jd0f, jd0f + float(default_days)

    return None, None

def _build_returns_kwargs(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """
    Route-friendly builder for returns engines.
    - Validates minimal natal fields
    - Fills jd_tt_natal / jd_ut1_natal if missing
    - Normalizes frame/zodiac/house
    - Derives topocentric from coords
    - Computes optional scan window (jd_start_tt / jd_end_tt)
    """
    errs: List[Dict[str, Any]] = []

    # natal (required)
    natal = body.get("natal") or {}
    if not isinstance(natal, dict):
        errs.append({"loc": ["natal"], "msg": "required object", "type": "value_error"})
        natal = {}

    # natal essentials (only enforced here to be helpful; engine can still handle strict JDs)
    date = natal.get("date"); time_s = natal.get("time")
    tz = natal.get("place_tz") or natal.get("tz") or natal.get("timezone")
    if not (isinstance(date, str) and isinstance(time_s, str) and isinstance(tz, str)):
        errs.append({"loc": ["natal.date|time|place_tz"], "msg": "required strings", "type": "value_error"})

    # frame / zodiac / house / ayanamsa
    try:
        frame = parse_frame(body.get("frame"))
    except ValidationError as e:
        return {}, e.errors()
    zodiac_mode = (body.get("zodiac_mode") or body.get("mode") or "tropical").strip().lower()
    house_system = (body.get("house_system") or "placidus").strip().lower()
    ay = body.get("ayanamsa_deg")
    ay_f = None
    try:
        if isinstance(ay, (int, float)):
            ay_f = float(ay)
        elif ay is not None:
            ay_f = float(str(ay))
    except Exception:
        errs.append({"loc": ["ayanamsa_deg"], "msg": "must be a number", "type": "type_error.float"})

    # return kind
    kind = _parse_return_kind(body.get("kind") or body.get("type") or body.get("planet"))

    # natal strict timescales (fill if missing)
    jd_tt_natal = body.get("jd_tt_natal")
    jd_ut1_natal = body.get("jd_ut1_natal")
    try:
        if isinstance(date, str) and isinstance(time_s, str) and isinstance(tz, str):
            ts_nat = _compute_timescales_from_local(date, time_s, tz, payload=natal)
            if not isinstance(jd_tt_natal, (int, float)):
                jd_tt_natal = float(ts_nat["jd_tt"])
            if not isinstance(jd_ut1_natal, (int, float)):
                jd_ut1_natal = float(ts_nat["jd_ut1"])
    except ValidationError as e:
        errs.extend(e.errors())

    # Additional return parameters
    place = body.get("place")
    if isinstance(place, dict):
        try:
            la = float(place["latitude"]); lo = float(place["longitude"])
            place = {"latitude": la, "longitude": lo, "elev_m": float(place.get("elev_m", 0.0))}
        except Exception:
            place = None
    else:
        place = None

    def _has_coords(d: Dict[str, Any]) -> bool:
        try:
            return isinstance(d.get("latitude"), (int, float)) and isinstance(d.get("longitude"), (int, float))
        except Exception:
            return False

    topocentric = bool(body.get("topocentric")) or _has_coords(natal) or _has_coords(place or {})

    try:
        jd0, jd1 = _normalize_window(body)
    except ValidationError as e:
        errs.extend(e.errors())
        jd0 = jd1 = None

    # Additional solver & validation knobs
    tol_arcmin = float(body.get("tol_arcmin", 1.0))
    max_iters = int(body.get("max_iters", 12))
    estimate_uncertainty = bool(body.get("estimate_uncertainty", True))
    fd_step_minutes = float(body.get("fd_step_minutes", 2.0))
    profile = bool(body.get("profile", False))
    validation = (body.get("validation") or "basic").strip().lower()
    validation_residual_arcmin = float(body.get("validation_residual_arcmin", 1.0))

    guess_years_offset = body.get("guess_years_offset")
    around_jd_tt = body.get("around_jd_tt")

    orbs = body.get("orbs") if isinstance(body.get("orbs"), dict) else None
    aspects_to_natal = bool(body.get("aspects_to_natal", True))
    parallels = bool(body.get("parallels", False))
    antiscia = bool(body.get("antiscia", False))

    lunar_month = (body.get("lunar_month") or "sidereal").strip().lower()

    if errs:
        return {}, errs

    kwargs: Dict[str, Any] = {
        "natal": natal,
        "kind": kind,
        "jd_tt_natal": jd_tt_natal,
        "jd_ut1_natal": jd_ut1_natal,
        "place": place,
        "frame": frame,
        "house_system": house_system,
        "zodiac_mode": zodiac_mode,
        "ayanamsa_deg": ay_f if ay_f is not None else None,
        "lunar_month": lunar_month,
        "guess_years_offset": guess_years_offset,
        "around_jd_tt": around_jd_tt,
        "tol_arcmin": tol_arcmin,
        "max_iters": max_iters,
        "estimate_uncertainty": estimate_uncertainty,
        "fd_step_minutes": fd_step_minutes,
        "profile": profile,
        "validation": validation,
        "validation_residual_arcmin": validation_residual_arcmin,
        "aspects_to_natal": aspects_to_natal,
        "parallels": parallels,
        "antiscia": antiscia,
        "orbs": orbs,
        "jd_start_tt": jd0,
        "jd_end_tt": jd1,
        "topocentric": topocentric,
    }

    # prune explicit None (except window and ayanamsa which can be None safely)
    for k in list(kwargs.keys()):
        if kwargs[k] is None and k not in ("jd_start_tt", "jd_end_tt", "ayanamsa_deg"):
            kwargs.pop(k, None)

    return kwargs, None

def _returns_enrich_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(meta or {})
    try:
        m.update(_snapshot_ephemeris_meta(m))
    except Exception:
        pass
    return m

@api.post("/api/return")
@rate_limit(RL_RETURNS)
def returns_compute_route():
    """
    Compute a single return (solar/lunar).

    Body: see README; tolerant to extra fields.
    """
    if not _returns_available():
        det = {"import_error": repr(_RETURNS_IMPORT_ERROR)} if DEBUG_VERBOSE and _RETURNS_IMPORT_ERROR else None
        return _json_error("returns_unavailable", det or "returns engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    kwargs, errs = _build_returns_kwargs(body)
    if errs:
        return _json_error("validation_error", errs, 400)

    compute_fn = _returns_pick_fn("compute")
    if not callable(compute_fn):
        return _json_error("returns_unavailable", "no compute function exported by return(s) module", 501)

    try:
        result = compute_fn(**kwargs)  # type: ignore[misc]
    except ValueError as e:
        return _json_error("returns_value_error", str(e), 400)
    except TypeError as e:
        det = {"type": "TypeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("returns_internal", det or "internal_error", 500)
    except RuntimeError as e:
        det = {"type": "RuntimeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("returns_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("returns_internal", det or "internal_error", 500)

    # normalize response
    meta = _returns_enrich_meta(result.get("meta") or {})

    event = result.get("event") or {}
    epoch = None
    if isinstance(event, dict) and ("jd_tt" in event or "jd_ut1" in event):
        epoch = {"jd_tt": float(event.get("jd_tt")), "jd_ut1": float(event.get("jd_ut1"))}

    resp = {
        "ok": True,
        "kind": (event.get("kind") or kwargs.get("kind") or "solar"),
        "event": event,
        "epoch": epoch,
        "positions": result.get("positions"),
        "houses": result.get("houses"),
        "meta": meta,
        "warnings": list(meta.get("warnings") or []),
    }
    return jsonify(resp), 200

@api.post("/api/return/scan")
@rate_limit(RL_RETURNS)
def returns_scan_route():
    """
    Scan a time window for return events (e.g., all lunar returns in a month).

    Body:
      natal: { date, time, place_tz, latitude?, longitude? }
      kind|type|planet: "solar"|"lunar"|...
      jd_start_tt & jd_end_tt (preferred) OR date_start/time_start/date_end/time_end (+ tz)
      frame, zodiac_mode, house_system, place, etc.
    """
    if not _returns_available():
        det = {"import_error": repr(_RETURNS_IMPORT_ERROR)} if DEBUG_VERBOSE and _RETURNS_IMPORT_ERROR else None
        return _json_error("returns_unavailable", det or "returns engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    kwargs, errs = _build_returns_kwargs(body)
    if errs:
        return _json_error("validation_error", errs, 400)

    # ensure we have a real window
    jd0 = kwargs.get("jd_start_tt"); jd1 = kwargs.get("jd_end_tt")
    if not (isinstance(jd0, (int, float)) and isinstance(jd1, (int, float)) and jd1 > jd0):
        return _json_error("validation_error", [{"loc": ["jd_start_tt|date_start"], "msg": "window required"}], 400)

    scan_fn = _returns_pick_fn("scan")
    if not callable(scan_fn):
        return _json_error("returns_unavailable", "no scan function exported by return(s) module", 501)

    try:
        results = scan_fn(**kwargs)  # type: ignore[misc]
    except ValueError as e:
        return _json_error("returns_value_error", str(e), 400)
    except TypeError as e:
        det = {"type": "TypeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("returns_internal", det or "internal_error", 500)
    except RuntimeError as e:
        det = {"type": "RuntimeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("returns_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("returns_internal", det or "internal_error", 500)

    meta: Dict[str, Any] = {}
    if isinstance(results, dict):
        meta = _returns_enrich_meta(results.get("meta") or {})
        res_list = results.get("results") if isinstance(results.get("results"), list) else []
    elif isinstance(results, list):
        res_list = results
    else:
        res_list = []

    return jsonify({
        "ok": True,
        "kind": kwargs.get("kind"),
        "window": {"jd_start_tt": float(jd0), "jd_end_tt": float(jd1)},
        "results": res_list,
        "meta": meta,
    }), 200

# ───────────────────────── PARANS ─────────────────────────
@api.post("/api/parans")
@rate_limit(RL_PARANS)
def parans_route():
    """
    Compute local parans (co-risings/culminations/settings/anti-culminations).
    
    Body:
      subject?: { date, time, place_tz }  # for timescale resolution (optional if jd_tt_ref/jd_ut1_ref provided)
      place: { latitude, longitude, elev_m? }  # observation location (required)
      jd_tt_ref?: float  # reference epoch (TT); optional if subject provided
      jd_ut1_ref?: float  # reference epoch (UT1); optional if subject provided
      frame?: "ecliptic-of-date" | "ecliptic-j2000"
      zodiac_mode?: "tropical" | "sidereal"
      ayanamsa_deg?: float
      bodies?: ["Sun", "Moon", ...]  # default: major planets
      tolerance_minutes?: float  # max separation for paran detection (default: 4.0)
      search_window_days?: float  # search window around reference (default: 1.0)
      max_iters?: int  # iteration limit for event solving (default: 10)
      fd_step_minutes?: float  # finite difference step (default: 2.0)
      earth_model?: "spherical" | "wgs84"
      apply_refraction?: bool
      pressure_hPa?: float
      temperature_C?: float
      profile?: bool
      validation?: "none" | "basic"
    """
    if _parans_compute is None:
        det = {"import_error": repr(_PARANS_IMPORT_ERROR)} if DEBUG_VERBOSE and _PARANS_IMPORT_ERROR else None
        return _json_error("parans_unavailable", det or "parans engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_parans_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # Extract components
    subject = payload.get("subject", {})
    place = payload.get("place", {})
    
    # Handle strict timescales vs subject resolution
    jd_tt_ref = payload.get("jd_tt_ref")
    jd_ut1_ref = payload.get("jd_ut1_ref")
    
    # Resolve timescales if not provided directly
    if jd_tt_ref is None or jd_ut1_ref is None:
        # At this point, validator has ensured subject has required fields
        try:
            ts = _compute_timescales_from_local(
                subject["date"], 
                subject["time"], 
                subject["place_tz"], 
                payload=subject
            )
            jd_tt_ref = float(ts["jd_tt"])
            jd_ut1_ref = float(ts["jd_ut1"])
        except ValidationError as e:
            return _json_error("validation_error", e.errors(), 400)
        except Exception as e:
            return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

    # Build arguments for compute_parans function
    paran_kwargs = {
        "subject": subject,
        "place": place,
        "jd_tt_ref": jd_tt_ref,
        "jd_ut1_ref": jd_ut1_ref,
        "frame": payload.get("frame", "ecliptic-of-date"),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "bodies": tuple(payload.get("bodies", ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])),
        "tolerance_minutes": payload.get("tolerance_minutes", 4.0),
        "search_window_days": payload.get("search_window_days", 1.0),
        "max_iters": payload.get("max_iters", 10),
        "fd_step_minutes": payload.get("fd_step_minutes", 2.0),
        "earth_model": payload.get("earth_model", "spherical"),
        "apply_refraction": payload.get("apply_refraction", False),
        "pressure_hPa": payload.get("pressure_hPa", 1010.0),
        "temperature_C": payload.get("temperature_C", 10.0),
        "profile": payload.get("profile", False),
        "validation": payload.get("validation", "basic"),
    }

    # Filter arguments to match function signature
    try:
        import inspect
        paran_params = set(inspect.signature(_parans_compute).parameters.keys())
        filtered_kwargs = {k: v for k, v in paran_kwargs.items() if k in paran_params}
    except Exception:
        # Fallback: pass all arguments and let the function handle it
        filtered_kwargs = paran_kwargs

    # Call the parans computation engine
    try:
        result = _parans_compute(**filtered_kwargs)
    except ValueError as e:
        return _json_error("parans_value_error", str(e), 400)
    except TypeError as e:
        det = {"type": "TypeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("parans_internal", det or "internal_error", 500)
    except RuntimeError as e:
        det = {"type": "RuntimeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("parans_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("parans_internal", det or "internal_error", 500)

    # Check if result indicates success
    if not isinstance(result, dict) or not result.get("ok", False):
        error_details = result.get("details") if isinstance(result, dict) else None
        error_type = result.get("error", "parans_failed") if isinstance(result, dict) else "parans_failed"
        
        if error_type == "validation_error":
            return _json_error("validation_error", error_details, 400)
        elif error_type in ("timescales_error", "parans_calculation_failed"):
            return _json_error(error_type, error_details, 400)
        else:
            return _json_error("parans_internal", error_details if DEBUG_VERBOSE else None, 500)

    # Enrich metadata with ephemeris adapter info
    meta = dict(result.get("meta", {}))
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # Structure the response
    resp = {
        "ok": True,
        "meta": meta,
        "events_by_body": result.get("events_by_body", {}),
        "parans": result.get("parans", []),
        "warnings": list(meta.get("warnings", [])),
    }

    return jsonify(resp), 200

# ───────────────────────── SYNASTRY & COMPOSITE ─────────────────────────
@api.post("/api/synastry")
@rate_limit(RL_SYNASTRY)
def synastry_route():
    """
    Compute synastry between two natal charts.
    
    Body:
      natal_a: { date, time, place_tz, latitude?, longitude?, elev_m? }
      natal_b: { date, time, place_tz, latitude?, longitude?, elev_m? }
      jd_tt_a?, jd_ut1_a?, jd_tt_b?, jd_ut1_b?: strict timescales (optional)
      place_a?, place_b?: coordinate overrides (optional)
      frame?: "ecliptic-of-date" | "ecliptic-j2000"
      zodiac_mode?: "tropical" | "sidereal"
      ayanamsa_deg?: float
      house_system?: string
      orbs?: {aspect: orb_deg, ...}
      parallels?: bool (default true)
      antiscia?: bool (default true)
    """
    if _synastry_compute is None:
        det = {"import_error": repr(_SYNASTRY_IMPORT_ERROR)} if DEBUG_VERBOSE and _SYNASTRY_IMPORT_ERROR else None
        return _json_error("synastry_unavailable", det or "synastry engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_synastry_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # Build arguments for synastry computation
    synastry_kwargs = {
        "natal_a": payload["natal_a"],
        "natal_b": payload["natal_b"],
        "frame": payload.get("frame", "ecliptic-of-date"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "house_system": payload.get("house_system", "placidus"),
        "parallels": payload.get("parallels", True),
        "antiscia": payload.get("antiscia", True),
    }

    # Add optional fields
    for field in ("jd_tt_a", "jd_ut1_a", "jd_tt_b", "jd_ut1_b", "place_a", "place_b", "orbs"):
        if field in payload:
            synastry_kwargs[field] = payload[field]

    # Filter arguments to match function signature
    try:
        import inspect
        synastry_params = set(inspect.signature(_synastry_compute).parameters.keys())
        filtered_kwargs = {k: v for k, v in synastry_kwargs.items() if k in synastry_params}
    except Exception:
        filtered_kwargs = synastry_kwargs

    # Call synastry computation
    try:
        result = _synastry_compute(**filtered_kwargs)
    except ValueError as e:
        return _json_error("synastry_value_error", str(e), 400)
    except TypeError as e:
        det = {"type": "TypeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("synastry_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("synastry_internal", det or "internal_error", 500)

    # Enrich metadata
    meta = dict(result.get("meta", {}))
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # Structure response
    resp = {
        "ok": True,
        "meta": meta,
        "aspects": result.get("aspects", {}),
        "overlays": result.get("overlays", {}),
        "midpoints": result.get("midpoints", {}),
        "scores": result.get("scores", {}),
        "warnings": list(meta.get("warnings", [])),
    }

    return jsonify(resp), 200

@api.post("/api/composite")
@rate_limit(RL_COMPOSITE)
def composite_route():
    """
    Compute composite chart between two natal charts.
    
    Body:
      natal_a: { date, time, place_tz, latitude?, longitude?, elev_m? }
      natal_b: { date, time, place_tz, latitude?, longitude?, elev_m? }
      method?: "midpoint" | "davison" (default midpoint)
      jd_tt_ref?, jd_ut1_ref?: reference timescales (optional)
      place_ref?: reference place for davison method (optional)
      frame?: "ecliptic-of-date" | "ecliptic-j2000"
      house_system?: string
      ayanamsa_deg?: float
      zodiac_mode?: "tropical" | "sidereal"
    """
    if _composite_compute is None:
        det = {"import_error": repr(_SYNASTRY_IMPORT_ERROR)} if DEBUG_VERBOSE and _SYNASTRY_IMPORT_ERROR else None
        return _json_error("composite_unavailable", det or "composite engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_composite_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # Build arguments
    composite_kwargs = {
        "natal_a": payload["natal_a"],
        "natal_b": payload["natal_b"],
        "method": payload.get("method", "midpoint"),
        "frame": payload.get("frame", "ecliptic-of-date"),
        "house_system": payload.get("house_system", "placidus"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
    }

    # Add optional fields
    for field in ("jd_tt_ref", "jd_ut1_ref", "place_ref"):
        if field in payload:
            composite_kwargs[field] = payload[field]

    # Filter arguments
    try:
        import inspect
        composite_params = set(inspect.signature(_composite_compute).parameters.keys())
        filtered_kwargs = {k: v for k, v in composite_kwargs.items() if k in composite_params}
    except Exception:
        filtered_kwargs = composite_kwargs

    # Call composite computation
    try:
        result = _composite_compute(**filtered_kwargs)
    except ValueError as e:
        return _json_error("composite_value_error", str(e), 400)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("composite_internal", det or "internal_error", 500)

    # Enrich metadata
    meta = dict(result.get("meta", {}))
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # Structure response
    resp = {
        "ok": True,
        "meta": meta,
        "method": result.get("method"),
        "positions": result.get("positions", {}),
        "asc": result.get("asc"),
        "mc": result.get("mc"),
        "cusps": result.get("cusps"),
    }

    return jsonify(resp), 200

@api.post("/api/synastry/report")
@rate_limit(RL_SYNASTRY)
def synastry_report_route():
    """
    Comprehensive synastry report (synastry + composite + metrics).
    
    Body: combines synastry and composite parameters
      natal_a, natal_b: natal chart data
      composite_method?: "midpoint" | "davison"
      composite_place_ref?: reference place
      ... all synastry parameters ...
    """
    if _synastry_report_compute is None:
        det = {"import_error": repr(_SYNASTRY_IMPORT_ERROR)} if DEBUG_VERBOSE and _SYNASTRY_IMPORT_ERROR else None
        return _json_error("synastry_report_unavailable", det or "synastry report engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_synastry_report_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # Build arguments
    report_kwargs = {
        "natal_a": payload["natal_a"],
        "natal_b": payload["natal_b"],
        "frame": payload.get("frame", "ecliptic-of-date"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "house_system": payload.get("house_system", "placidus"),
        "parallels": payload.get("parallels", True),
        "antiscia": payload.get("antiscia", True),
        "composite_method": payload.get("composite_method", "midpoint"),
    }

    # Add optional fields
    optional_fields = (
        "jd_tt_a", "jd_ut1_a", "jd_tt_b", "jd_ut1_b", 
        "place_a", "place_b", "orbs", "composite_place_ref"
    )
    for field in optional_fields:
        if field in payload:
            report_kwargs[field] = payload[field]

    # Filter arguments
    try:
        import inspect
        report_params = set(inspect.signature(_synastry_report_compute).parameters.keys())
        filtered_kwargs = {k: v for k, v in report_kwargs.items() if k in report_params}
    except Exception:
        filtered_kwargs = report_kwargs

    # Call report computation
    try:
        result = _synastry_report_compute(**filtered_kwargs)
    except ValueError as e:
        return _json_error("synastry_report_value_error", str(e), 400)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("synastry_report_internal", det or "internal_error", 500)

    # Enrich metadata
    meta = dict(result.get("meta", {}))
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # Structure response
    resp = {
        "ok": True,
        "meta": meta,
        "aspects": result.get("aspects", {}),
        "overlays": result.get("overlays", {}),
        "midpoints": result.get("midpoints", {}),
        "composite": result.get("composite", {}),
        "scores": result.get("scores", {}),
        "metrics": result.get("metrics", {}),
        "warnings": list(meta.get("warnings", [])),
    }

    return jsonify(resp), 200

# ───────────────────────── RELOCATION & ASTROCARTOGRAPHY ─────────────────────────
@api.post("/api/relocation")
@rate_limit(RL_RELOCATION)
def relocation_route():
    """
    Compute relocated chart for new location while preserving natal time.
    
    Body:
      natal: { date, time, place_tz, latitude?, longitude?, elev_m? }
      place_new: { latitude, longitude, elev_m? } # required new location
      jd_tt_natal?, jd_ut1_natal?: strict timescales (optional)
      frame?: "ecliptic-of-date" | "ecliptic-j2000"
      house_system?: string
      zodiac_mode?: "tropical" | "sidereal"
      ayanamsa_deg?: float
      topocentric_positions?: bool (default false - use geocentric positions)
    """
    if _compute_relocated is None:
        det = {"import_error": repr(_RELOCATION_IMPORT_ERROR)} if DEBUG_VERBOSE and _RELOCATION_IMPORT_ERROR else None
        return _json_error("relocation_unavailable", det or "relocation engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_relocation_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # Build arguments for relocation computation
    relocation_kwargs = {
        "natal": payload["natal"],
        "place_new": payload["place_new"],
        "frame": payload.get("frame", "ecliptic-of-date"),
        "house_system": payload.get("house_system", "placidus"),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "topocentric_positions": payload.get("topocentric_positions", False),
    }

    # Add optional strict timescales
    for field in ("jd_tt_natal", "jd_ut1_natal"):
        if field in payload:
            relocation_kwargs[field] = payload[field]

    # Filter arguments to match function signature
    try:
        import inspect
        relocation_params = set(inspect.signature(_compute_relocated).parameters.keys())
        filtered_kwargs = {k: v for k, v in relocation_kwargs.items() if k in relocation_params}
    except Exception:
        filtered_kwargs = relocation_kwargs

    # Call relocation computation
    try:
        result = _compute_relocated(**filtered_kwargs)
    except ValueError as e:
        return _json_error("relocation_value_error", str(e), 400)
    except TypeError as e:
        det = {"type": "TypeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("relocation_internal", det or "internal_error", 500)
    except RuntimeError as e:
        det = {"type": "RuntimeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("relocation_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("relocation_internal", det or "internal_error", 500)

    # Enrich metadata with ephemeris adapter info
    meta = dict(result.get("meta", {}))
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # Structure response
    resp = {
        "ok": True,
        "meta": meta,
        "positions": result.get("positions", {}),
        "houses": result.get("houses"),
        "axes": result.get("axes", {}),
        "warnings": list(meta.get("warnings", [])),
    }

    return jsonify(resp), 200

@api.post("/api/astrocartography")
@rate_limit(RL_ASTROCARTOGRAPHY)
def astrocartography_route():
    """
    Compute astrocartography lines (MC/IC/ASC/DC) for selected bodies.
    
    Body:
      natal: { date, time, place_tz, latitude?, longitude?, elev_m? }
      jd_tt?, jd_ut1?: epoch timescales (optional, defaults to natal time)
      frame?: "ecliptic-of-date" | "ecliptic-j2000"
      zodiac_mode?: "tropical" | "sidereal"
      ayanamsa_deg?: float
      bodies?: ["Sun", "Moon", ...] # default: major planets
      lon_step_deg?: float # longitude sampling step (default 1.0)
      lat_clip_deg?: float # latitude clipping limit (default 89.5)
      earth_model?: "spherical" | "wgs84" # Earth model for dip calculation
      apply_refraction?: bool # Saemundsson atmospheric refraction
      pressure_hPa?: float # atmospheric pressure for refraction
      temperature_C?: float # temperature for refraction
      default_elev_m?: float # default elevation when no DEM available
    """
    if _compute_astrocartography is None:
        det = {"import_error": repr(_RELOCATION_IMPORT_ERROR)} if DEBUG_VERBOSE and _RELOCATION_IMPORT_ERROR else None
        return _json_error("astrocartography_unavailable", det or "astrocartography engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_astrocartography_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # Build arguments for astrocartography computation
    astro_kwargs = {
        "natal": payload["natal"],
        "frame": payload.get("frame", "ecliptic-of-date"),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "bodies": tuple(payload.get("bodies", ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])),
        "lon_step_deg": payload.get("lon_step_deg", 1.0),
        "lat_clip_deg": payload.get("lat_clip_deg", 89.5),
        "earth_model": payload.get("earth_model", "spherical"),
        "apply_refraction": payload.get("apply_refraction", False),
        "pressure_hPa": payload.get("pressure_hPa", 1010.0),
        "temperature_C": payload.get("temperature_C", 10.0),
        "default_elev_m": payload.get("default_elev_m", 0.0),
    }

    # Add optional epoch timescales
    for field in ("jd_tt", "jd_ut1"):
        if field in payload:
            astro_kwargs[field] = payload[field]

    # Filter arguments to match function signature
    try:
        import inspect
        astro_params = set(inspect.signature(_compute_astrocartography).parameters.keys())
        filtered_kwargs = {k: v for k, v in astro_kwargs.items() if k in astro_params}
    except Exception:
        filtered_kwargs = astro_kwargs

    # Call astrocartography computation
    try:
        result = _compute_astrocartography(**filtered_kwargs)
    except ValueError as e:
        return _json_error("astrocartography_value_error", str(e), 400)
    except TypeError as e:
        det = {"type": "TypeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("astrocartography_internal", det or "internal_error", 500)
    except RuntimeError as e:
        det = {"type": "RuntimeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("astrocartography_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("astrocartography_internal", det or "internal_error", 500)

    # Enrich metadata with ephemeris adapter info
    meta = dict(result.get("meta", {}))
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # Structure response
    resp = {
        "ok": True,
        "meta": meta,
        "lines": result.get("lines", []),
        "warnings": list(meta.get("warnings", [])),
    }

    return jsonify(resp), 200

# ───────────────────────── DIRECTIONS ─────────────────────────
@api.post("/api/directions")
@rate_limit(RL_DIRECTIONS)
def directions_route():
    """
    Compute Solar-Arc directions (direct/converse) and detect hits to natal targets.
    
    Body:
      natal: { date, time, place_tz, latitude?, longitude?, elev_m?, mode? }
      method?: "solar_arc" (only supported method currently)
      rate?: "naibod" | "true_sun" (default: "naibod")
      target?: { date, time, place_tz } # alternative to years_after
      years_after?: float # alternative to target
      jd_tt_natal?, jd_ut1_natal?: strict timescales (optional)
      place?: { latitude, longitude, elev_m? } # place override for directions
      frame?: "ecliptic-of-date" | "ecliptic-j2000"
      house_system?: string (default: "placidus")
      zodiac_mode?: "tropical" | "sidereal"
      ayanamsa_deg?: float
      arcs?: "direct" | "converse" | "both" (default: "direct")
      orbs?: { conjunction: float, opposition: float, ... }
      include_hits_to?: ["planets", "angles", "cusps"] # targets for hit detection
      parallels?: bool (default: false)
      antiscia?: bool (default: false)
      profile?: bool (default: false)
      validation?: "none" | "basic" (default: "basic")
    """
    if _compute_directions is None:
        det = {"import_error": repr(_DIRECTIONS_IMPORT_ERROR)} if DEBUG_VERBOSE and _DIRECTIONS_IMPORT_ERROR else None
        return _json_error("directions_unavailable", det or "directions engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        payload = parse_directions_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # Build arguments for directions computation
    directions_kwargs = {
        "natal": payload["natal"],
        "method": payload.get("method", "solar_arc"),
        "rate": payload.get("rate", "naibod"),
        "frame": payload.get("frame", "ecliptic-of-date"),
        "house_system": payload.get("house_system", "placidus"),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "arcs": payload.get("arcs", "direct"),
        "include_hits_to": tuple(payload.get("include_hits_to", ["planets", "angles", "cusps"])),
        "parallels": payload.get("parallels", False),
        "antiscia": payload.get("antiscia", False),
        "profile": payload.get("profile", False),
        "validation": payload.get("validation", "basic"),
    }

    # Add optional target or years_after
    if "target" in payload:
        directions_kwargs["target"] = payload["target"]
    elif "years_after" in payload:
        directions_kwargs["years_after"] = payload["years_after"]

    # Add optional strict timescales
    for field in ("jd_tt_natal", "jd_ut1_natal"):
        if field in payload:
            directions_kwargs[field] = payload[field]

    # Add optional place override
    if "place" in payload:
        directions_kwargs["place"] = payload["place"]

    # Add optional orbs
    if "orbs" in payload:
        directions_kwargs["orbs"] = payload["orbs"]

    # Filter arguments to match function signature
    try:
        import inspect
        directions_params = set(inspect.signature(_compute_directions).parameters.keys())
        filtered_kwargs = {k: v for k, v in directions_kwargs.items() if k in directions_params}
    except Exception:
        filtered_kwargs = directions_kwargs

    # Call directions computation
    try:
        result = _compute_directions(**filtered_kwargs)
    except ValueError as e:
        return _json_error("directions_value_error", str(e), 400)
    except TypeError as e:
        det = {"type": "TypeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("directions_internal", det or "internal_error", 500)
    except RuntimeError as e:
        det = {"type": "RuntimeError", "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("directions_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("directions_internal", det or "internal_error", 500)

    # Enrich metadata with ephemeris adapter info
    meta = dict(result.get("meta", {}))
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # Structure response
    resp = {
        "ok": True,
        "meta": meta,
        "epoch": result.get("epoch", {}),
        "arc_deg": result.get("arc_deg", {}),
        "positions": result.get("positions", {}),
        "hits": result.get("hits", {}),
        "warnings": list(meta.get("warnings", [])),
    }

    return jsonify(resp), 200

# ───────────────────────── ephemeris ─────────────────────────
def _coerce_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

def _truthy(val: Any) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None

def _norm_rows_from_longitudes(raw: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize longitude results to consistent format."""
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}

    if isinstance(raw, tuple) and len(raw) == 2:
        raw, meta = raw

    if isinstance(raw, dict):
        meta = dict(raw.get("meta") or {})

    if raw is None:
        return rows, meta

    if isinstance(raw, dict) and all(isinstance(v, (int, float)) for v in raw.values()):
        for k, v in raw.items():
            rows.append({"body": str(k).lower(), "name": str(k), "longitude": float(v)})
        return rows, meta

    data = raw.get("results") if isinstance(raw, dict) and isinstance(raw.get("results"), list) else raw
    if isinstance(data, list):
        for r in data:
            if isinstance(r, dict):
                if len(r) == 1 and isinstance(next(iter(r.values())), (int, float)):
                    k, v = next(iter(r.items()))
                    rows.append({"body": str(k).lower(), "name": str(k), "longitude": float(v)})
                    continue

                body = (r.get("body") or r.get("name") or r.get("planet") or r.get("id") or r.get("label"))
                L = r.get("longitude") or r.get("lon") or r.get("lambda") or r.get("ecliptic_longitude")

                if body and isinstance(L, (int, float)):
                    row: Dict[str, Any] = {
                        "body": str(body).lower(),
                        "name": str(r.get("name") or body),
                        "longitude": float(L),
                    }
                    if _coerce_float(r.get("speed")) is not None:
                        row["speed"] = float(r.get("speed"))
                    if _coerce_float(r.get("velocity")) is not None:
                        row["velocity"] = float(r.get("velocity"))
                    rows.append(row)

    return rows, meta

def _norm_rows_from_lv(raw: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize longitude+velocity results to consistent format."""
    rows_map: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}

    if isinstance(raw, dict):
        meta = dict(raw.get("meta") or {})

    if isinstance(raw, dict) and "longitudes" in raw and "velocities" in raw:
        for k, v in (raw.get("longitudes") or {}).items():
            rows_map.setdefault(str(k).lower(), {"body": str(k).lower(), "name": str(k)})["longitude"] = float(v)
        for k, v in (raw.get("velocities") or {}).items():
            rows_map.setdefault(str(k).lower(), {"body": str(k).lower(), "name": str(k)})["velocity"] = float(v)
        return list(rows_map.values()), meta

    data = raw.get("results") if isinstance(raw, dict) and isinstance(raw.get("results"), list) else raw
    if isinstance(data, list):
        for r in data:
            if not isinstance(r, dict):
                continue
            body = (r.get("body") or r.get("name") or r.get("planet") or r.get("id") or r.get("label"))
            if not body:
                continue
            key = str(body).lower()
            rec = rows_map.setdefault(key, {"body": key, "name": str(r.get("name") or body)})
            L = r.get("longitude") or r.get("lon") or r.get("lambda") or r.get("ecliptic_longitude")
            V = r.get("velocity") or r.get("vel") or r.get("dlambda_dt") or r.get("deg_per_day") or r.get("speed")
            if isinstance(L, (int, float)):
                rec["longitude"] = float(L)
            if isinstance(V, (int, float)):
                rec["velocity"] = float(V)

    return list(rows_map.values()), meta

@api.post("/api/ephemeris/longitudes")
@rate_limit(RL_EPHEM)
def ephemeris_longitudes_endpoint():
    """
    Fixed endpoint that properly handles topocentric flag:
      • If topocentric:true but no coords -> 422 topocentric_coords_required
      • If coords present -> force topocentric True (even if flag was false)
      • center = 'topocentric' iff meta.topocentric truthy; else 'geocentric'
      • Prefer adapter meta.topocentric; fallback to the resolved value
    """
    try:
        body = request.get_json(force=True) or {}
        payload = parse_ephemeris_payload(body, require_bodies=True)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e), 400)

    # Core parameters
    jd_tt = payload["jd_tt"]
    frame = payload.get("frame", "ecliptic-of-date")
    bodies = payload["bodies"]
    names = payload["names"]

    # Requested flag (body or query)
    requested_topo = _truthy(body.get("topocentric"))
    if requested_topo is None:
        requested_topo = _truthy(request.args.get("topocentric")) or False

    # Extract coordinates: body.observer, body aliases, query aliases
    ob = body.get("observer") or {}
    lat = (
        _coerce_float(ob.get("lat"))
        or _coerce_float(ob.get("latitude"))
        or _coerce_float(body.get("lat"))
        or _coerce_float(body.get("latitude"))
        or _coerce_float(request.args.get("lat"))
        or _coerce_float(request.args.get("latitude"))
    )
    lon = (
        _coerce_float(ob.get("lon"))
        or _coerce_float(ob.get("lng"))
        or _coerce_float(ob.get("longitude"))
        or _coerce_float(body.get("lon"))
        or _coerce_float(body.get("lng"))
        or _coerce_float(body.get("longitude"))
        or _coerce_float(request.args.get("lon"))
        or _coerce_float(request.args.get("lng"))
        or _coerce_float(request.args.get("longitude"))
    )
    elev = (
        _coerce_float(ob.get("elevation_m"))
        or _coerce_float(ob.get("elev_m"))
        or _coerce_float(body.get("elevation_m"))
        or _coerce_float(body.get("elev_m"))
        or _coerce_float(request.args.get("elevation_m"))
        or _coerce_float(request.args.get("elev_m"))
    )

    coords_valid = (lat is not None) and (lon is not None)
    observer = {"lat": lat, "lon": lon, "elevation_m": (elev if elev is not None else 0.0)} if coords_valid else None

    # Rule 1: explicit topocentric without coords => 422
    if requested_topo and not coords_valid:
        return _json_error(
            "topocentric_coords_required",
            {
                "message": "topocentric:true requires lat & lon (optional elevation_m) via body.observer or ?lat&lon[&elev_m].",
                "aliases": {"lat": ["lat", "latitude"], "lon": ["lon", "lng", "longitude"], "elevation_m": ["elev_m", "elevation_m"]},
            },
            422,
        )

    # Rule 2: coords force topocentric (override any false flag)
    resolved_topo = True if coords_valid else bool(requested_topo)

    # Call adapter robustly
    try:
        from app.core import ephemeris_adapter as ea

        def _call():
            if hasattr(ea, "ecliptic_longitudes"):
                try:
                    return ea.ecliptic_longitudes(
                        jd_tt=jd_tt,
                        bodies=names,                   # modern kw
                        frame=frame,
                        topocentric=resolved_topo,
                        observer=observer if resolved_topo else None,
                        latitude=lat if resolved_topo else None,   # back-compat
                        longitude=lon if resolved_topo else None,
                        elevation_m=(elev if elev is not None else None) if resolved_topo else None,
                    )
                except TypeError:
                    # older signature variants
                    try:
                        return ea.ecliptic_longitudes(
                            jd_tt,
                            names=names,
                            frame=frame,
                            topocentric=resolved_topo,
                            latitude=lat if resolved_topo else None,
                            longitude=lon if resolved_topo else None,
                            elevation_m=(elev if elev is not None else None) if resolved_topo else None,
                        )
                    except TypeError:
                        return ea.ecliptic_longitudes(
                            jd_tt=jd_tt,
                            names=names,
                            frame=frame,
                            topocentric=resolved_topo,
                            latitude=lat if resolved_topo else None,
                            longitude=lon if resolved_topo else None,
                            elev_m=(elev if elev is not None else None) if resolved_topo else None,
                        )
            elif hasattr(ea, "ecliptic_longitudes_and_velocities"):
                return ea.ecliptic_longitudes_and_velocities(
                    jd_tt=jd_tt,
                    bodies=names,
                    frame=frame,
                    topocentric=resolved_topo,
                    observer=observer if resolved_topo else None,
                )
            raise RuntimeError("adapter_missing")
        adapter_ret = _call()
    except Exception as e:
        return _json_error("adapter_error", str(e), 500)

    # Normalize
    if isinstance(adapter_ret, dict) and ("velocities" in adapter_ret or "longitudes" in adapter_ret):
        rows, meta = _norm_rows_from_lv(adapter_ret)
    else:
        rows, meta = _norm_rows_from_longitudes(adapter_ret)

    # Order by requested bodies
    requested = [b.lower() for b in bodies]
    name_map = {b.lower(): n for b, n in zip(bodies, names)}
    by_body = {r.get("body"): r for r in rows if r.get("body")}

    ordered_results = []
    for body_key in requested:
        r = by_body.get(body_key)
        if r and isinstance(r.get("longitude"), (int, float)):
            ordered_results.append({
                "body": body_key,
                "name": name_map.get(body_key, r.get("name", body_key)),
                "longitude": float(r["longitude"]),
            })

    # Meta + center resolution (prefer adapter meta; fallback to route)
    final_meta = dict(meta or {})
    if "topocentric" not in final_meta:
        final_meta["topocentric"] = resolved_topo
    center = "topocentric" if (final_meta.get("topocentric") is True or str(final_meta.get("topocentric")).lower() == "true") else "geocentric"
    final_meta.setdefault("frame", frame)

    # Augment with adapter snapshot so dev tools can see path/coverage
    final_meta.update(_snapshot_ephemeris_meta(final_meta))

    return jsonify({
        "ok": True,
        "jd_tt": float(jd_tt),
        "frame": frame,
        "center": center,
        "units": {"angles": "deg"},
        "meta": final_meta,
        "results": ordered_results,
    }), 200

# ───────────────────────── ephemeris diagnostics (for dev tools) ─────────────────────────
@api.get("/api/ephemeris/diagnostics")
@rate_limit(RL_DEBUG)
def ephemeris_diagnostics_route():
    """
    Lightweight adapter diagnostics surface.
    Does not force a Skyfield kernel load; if a kernel has been loaded earlier,
    path/coverage will show up. If only a BSP path is known, we lazily read
    coverage from it (via jplephem) without loading Skyfield.
    """
    try:
        from app.core import ephemeris_adapter as ea

        out = dict(ea.ephemeris_diagnostics() or {})

        # Always expose a single short kernel tag
        out["kernel"] = out.get("ephemeris_name") or ea.current_kernel_name()

        # Path (if known)
        try:
            out["ephemeris_path"] = ea.current_kernel_path()
        except Exception:
            pass

        # De-dupe kernels if adapter returned duplicates
        ks = out.get("kernels")
        if isinstance(ks, list):
            out["kernels"] = list(dict.fromkeys(ks))

        # Lazy coverage: if missing but a BSP path exists, compute via jplephem
        if not out.get("ephemeris_coverage_jd"):
            p = out.get("ephemeris_path")
            if p and os.path.isfile(p):
                try:
                    from jplephem.spk import SPK  # local import to keep it optional
                    with SPK.open(p) as spk:
                        cov_min = min(seg.start_jd for seg in spk.segments)
                        cov_max = max(seg.end_jd for seg in spk.segments)
                    out["ephemeris_coverage_jd"] = [cov_min, cov_max]
                except Exception:
                    # silently ignore if jplephem isn't available or file can't be read
                    pass

        return jsonify({"ok": True, **out}), 200

    except Exception as e:
        return _json_error("adapter_error", str(e) if DEBUG_VERBOSE else "adapter_error", 500)

# ───────────────────────── DEBUG: House Engine Detection ─────────────────────────
@api.get("/api/debug/engine-test")
@rate_limit(RL_DEBUG)
def debug_engine_test():
    """Test direct call to advanced engine with diagnostics enabled."""
    try:
        from app.core.houses_advanced import PreciseHouseCalculator

        calc = PreciseHouseCalculator(
            require_strict_timescales=True,
            enable_diagnostics=True,
            enable_validation=True
        )

        result = calc.calculate_houses(
            latitude=40.7128,
            longitude=-74.0060,
            jd_ut=2460000.5,
            house_system="placidus",
            jd_tt=2460000.501,
            jd_ut1=2460000.499
        )

        return jsonify({
            "ok": True,
            "direct_engine_result": {
                "system": result.system,
                "asc": result.ascendant,
                "mc": result.midheaven,
                "cusps": result.cusps,
                "has_solver_stats": result.solver_stats is not None,
                "solver_stats_keys": list(result.solver_stats.keys()) if result.solver_stats else None,
                "has_error_budget": result.error_budget is not None,
                "warnings": result.warnings,
                "validation_count": len(result.validation_results)
            }
        }), 200
    except Exception as e:
        return _json_error("engine_test_error", str(e), 500)

@api.get("/api/debug/precision-test")
@rate_limit(RL_DEBUG)
def debug_precision_test():
    """Run multiple calculations to check for micro-variations."""
    try:
        results = []

        for i in range(5):
            # Tiny time variations to trigger different computational paths
            ts = {
                "jd_tt": 2460000.5 + i * 1e-8,
                "jd_ut1": 2460000.499 + i * 1e-8,
                "jd_utc": 2460000.5
            }

            payload = {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "house_system": "placidus"
            }

            houses = _call_compute_houses(payload, ts)

            results.append({
                "run": i,
                "asc": getattr(houses, 'asc', getattr(houses, 'asc_deg', 'missing')),
                "mc": getattr(houses, 'mc', getattr(houses, 'mc_deg', 'missing')),
                "cusps_sample": getattr(houses, 'cusps', [])[:3] if hasattr(houses, 'cusps') else [],
                "has_solver_stats": hasattr(houses, 'solver_stats'),
                "warnings": getattr(houses, 'warnings', [])
            })

        # Check for variations
        asc_values = [r["asc"] for r in results if isinstance(r["asc"], (int, float))]
        variation = max(asc_values) - min(asc_values) if len(asc_values) > 1 else 0.0

        return jsonify({
            "ok": True,
            "results": results,
            "asc_variation_degrees": variation,
            "expected_variation": "> 1e-9 for advanced engine",
            "likely_advanced_engine": variation > 1e-9
        }), 200
    except Exception as e:
        return _json_error("precision_test_error", str(e), 500)

@api.get("/api/debug/function-signatures")
@rate_limit(RL_DEBUG)
def debug_function_signatures():
    """Examine what functions are actually being called."""
    try:
        import inspect

        info = {
            "houses_function": str(_houses_fn),
            "houses_function_name": getattr(_houses_fn, '__name__', 'unknown'),
            "houses_function_module": getattr(_houses_fn, '__module__', 'unknown'),
            "houses_kind": _HOUSES_KIND,
            "chart_engine": _CHART_ENGINE_NAME
        }

        if _houses_fn:
            try:
                info["houses_function_signature"] = str(inspect.signature(_houses_fn))
                info["houses_function_file"] = getattr(inspect.getmodule(_houses_fn), '__file__', 'unknown')
            except Exception as e:
                info["signature_error"] = str(e)

        # Test advanced engine import
        try:
            from app.core.houses_advanced import PreciseHouseCalculator
            info["advanced_engine_importable"] = True
            info["advanced_engine_file"] = inspect.getfile(PreciseHouseCalculator)
        except Exception as e:
            info["advanced_engine_importable"] = False
            info["advanced_engine_error"] = str(e)

        return jsonify({"ok": True, "debug_info": info}), 200
    except Exception as e:
        return _json_error("function_signatures_error", str(e), 500)

# ───────────────────────── system-validation (optional) ─────────────────────────
@api.get("/system-validation")
@rate_limit(RL_DEBUG)
def system_validation():
    cfg = load_config(os.environ.get("ASTRO_CONFIG", "config/defaults.yaml"))
    leap_status: Optional[Dict[str, Any]] = None
    try:
        from app.core import leapseconds as _leaps  # optional
        for name in ("get_status", "status", "summary"):
            fn = getattr(_leaps, name, None)
            if callable(fn):
                s = fn()
                if is_dataclass(s):
                    leap_status = asdict(s)  # type: ignore
                elif hasattr(s, "__dict__"):
                    leap_status = dict(s.__dict__)
                elif isinstance(s, dict):
                    leap_status = s
                break
    except Exception:
        pass

    try:
        now_utc = datetime.now(timezone.utc)
        ts_now = _compute_timescales_from_local(
            now_utc.strftime("%Y-%m-%d"),
            now_utc.strftime("%H:%M:%S"),
            "UTC",
        )
        ts_sample = {
            "jd_utc": float(ts_now["jd_utc"]),
            "jd_tt": float(ts_now["jd_tt"]),
            "jd_ut1": float(ts_now["jd_ut1"]),
            "delta_t": ts_now["delta_t"],
            "delta_at": ts_now["delta_at"],
            "dut1": float(ts_now["dut1"]),
        }
    except Exception:
        ts_sample = None

    policy = {
        "houses_engine": _HOUSES_KIND,
        "polar": {
            "soft_fallback_lat_gt": float(POLAR_SOFT_LIMIT_DEG),
            "hard_reject_lat_ge": float(POLAR_HARD_LIMIT_DEG),
            "numeric_fallback": os.getenv("ASTRO_HOUSES_NUMERIC_FALLBACK", "1").lower()
            in ("1", "true", "yes", "on"),
        },
    }

    return jsonify(
        {
            "ok": True,
            "astronomy_accuracy": "ERFA-first timescales (JD_TT/JD_UT1), strict where required",
            "performance_slo": {"calculate_p95_ms": 800, "rect_quick_p95_s": 20},
            "mode_consistency": {
                "sidereal_default": cfg.mode == "sidereal",
                "ayanamsa": getattr(cfg, "ayanamsa", None),
            },
            "policy": policy,
            "leap_seconds": leap_status,
            "version": VERSION,
            "timescale_sample": ts_sample,
        }
    ), 200




