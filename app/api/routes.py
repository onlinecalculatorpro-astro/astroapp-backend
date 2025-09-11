# app/api/routes.py
"""
AstroApp — Canonical API Routes (circular-safe header)
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
- V2 prediction engine provides comprehensive forecasting capabilities.

This header intentionally avoids importing anything from `app.core.__init__`
and only imports concrete modules directly to prevent circular imports.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request, current_app
from zoneinfo import ZoneInfo

from app.version import VERSION
from app.utils.config import load_config
from app.utils.hc import flag_predictions
from app.utils.ratelimit import rate_limit, client_key, endpoint_key

# ───────────────────────── validators / parsers (pure, no heavy deps) ─────────────────────────
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
    parse_synasry_payload_safe,     # internal alias used by synastry report helper
    parse_synastry_report_payload,
    parse_relocation_payload,
    parse_astrocartography_payload,
    parse_directions_payload,
)

# ───────────────────────── timescales core (ERFA-only; safe) ─────────────────────────
try:
    from app.core.timescales import build_timescales, TimeScales
    _TIMESCALES_OK = True
    _TIMESCALES_ERR: Optional[Exception] = None
except Exception as e:
    _TIMESCALES_OK = False
    _TIMESCALES_ERR = e
    build_timescales = None  # type: ignore
    TimeScales = None        # type: ignore

# ───────────────────────── prediction engine (v2, optional) ─────────────────────────
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
    # Shims to avoid NameError in routes where features are disabled
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

# ───────────────────────── progressions (legacy helper; optional) ─────────────────────────
try:
    from app.core.progressions import compute_progressions
except Exception:
    compute_progressions = None  # type: ignore

# ───────────────────────── returns core (prefer module; fallback to file) ─────────────────────────
_returns_mod = None
_RETURNS_IMPORT_ERROR: Optional[Exception] = None
try:
    from app.core import returns as _returns_mod  # app/core/returns.py (preferred)
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

# ───────────────────────── parans core (optional) ─────────────────────────
_parans_compute = None  # function when available
_PARANS_IMPORT_ERROR: Optional[Exception] = None
try:
    from app.core.paran import compute_parans as _parans_compute
except Exception as _e:
    _PARANS_IMPORT_ERROR = _e
    _parans_compute = None  # type: ignore

# ───────────────────────── synastry / composite (optional) ─────────────────────────
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

# ───────────────────────── relocation / astrocartography (optional) ─────────────────────────
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

# ───────────────────────── directions (legacy helper; optional) ─────────────────────────
_compute_directions = None
_DIRECTIONS_IMPORT_ERROR: Optional[Exception] = None
try:
    from app.core.directions import compute_directions as _compute_directions
    _DIR_OK = True
except Exception as _e:
    _DIRECTIONS_IMPORT_ERROR = _e
    _compute_directions = None
    _DIR_OK = False

# ───────────────────────── logging / blueprint / env caps ─────────────────────────
log = logging.getLogger(__name__)
api = Blueprint("api", __name__)

DEBUG_VERBOSE = os.getenv("ASTRO_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "on")
ARCSEC_TOL = float(os.getenv("ASTRO_ASC_TOL_ARCSEC", "3.6"))  # 0.001°

# Per-endpoint rate-limit caps (calls per minute, env-overridable)
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
    """Normalize any angle to [0, 360). Returns 0.0 for values within 1e-12 of 0."""
    try:
        v = float(x) % 360.0
        return 0.0 if abs(v) < 1e-12 else v
    except Exception:
        # Preserve original on non-numeric input to avoid surprising callers.
        return x

def _shortest_delta_deg(a2: float, a1: float) -> float:
    """Smallest signed angular difference a2−a1 in (−180, 180]."""
    d = (float(a2) - float(a1) + 540.0) % 360.0 - 180.0
    return -180.0 if d == 180.0 else d

def _delta_arcsec(a: float, b: float) -> float:
    """Absolute angular separation in arcseconds."""
    return abs(_shortest_delta_deg(a, b)) * 3600.0

def _json_error(code: str, details: Any = None, http: int = 400):
    """Uniform JSON error envelope."""
    out: Dict[str, Any] = {"ok": False, "error": code}
    if details is not None:
        out["details"] = details
    return jsonify(out), http

def _split_jd(jd: float) -> tuple[float, float]:
    """Split a JD into its integer day and fractional remainder."""
    d = math.floor(float(jd))
    return float(d), float(jd) - float(d)

def _sind(a: float) -> float:
    """sin(deg)."""
    return math.sin(math.radians(a))

def _cosd(a: float) -> float:
    """cos(deg)."""
    return math.cos(math.radians(a))

def _atan2d(y: float, x: float) -> float:
    """atan2 in degrees, normalized to [0, 360). Raises on 0/0 to catch bad inputs early."""
    if abs(x) < 1e-18 and abs(y) < 1e-18:
        raise ValueError("atan2(0,0) undefined")
    return _wrap360(math.degrees(math.atan2(y, x)))

def _gast_deg(jd_ut1: float, jd_tt: float) -> float:
    """
    Greenwich Apparent Sidereal Time (deg).
    Prefers ERFA gst06a; falls back to a standard approximation if ERFA unavailable.
    """
    try:
        import erfa  # type: ignore
        d1u, d2u = _split_jd(jd_ut1)
        d1t, d2t = _split_jd(jd_tt)
        gst_rad = erfa.gst06a(d1u, d2u, d1t, d2t)
        return _wrap360(math.degrees(gst_rad))
    except Exception:
        T = (float(jd_ut1) - 2451545.0) / 36525.0
        theta = (
            280.46061837
            + 360.98564736629 * (float(jd_ut1) - 2451545.0)
            + 0.000387933 * (T**2)
            - (T**3) / 38710000.0
        )
        return _wrap360(theta)

def _true_obliquity_deg(jd_tt: float) -> float:
    """
    True obliquity ε (deg). Prefers ERFA (obl06 + nut06a); falls back to
    a polynomial if ERFA unavailable.
    """
    try:
        import erfa  # type: ignore
        d1, d2 = _split_jd(jd_tt)
        eps0 = erfa.obl06(d1, d2)
        _dpsi, deps = erfa.nut06a(d1, d2)
        return math.degrees(eps0 + deps)
    except Exception:
        T = (float(jd_tt) - 2451545.0) / 36525.0
        eps_arcsec = 84381.448 - 46.8150 * T - 0.00059 * (T**2) + 0.001813 * (T**3)
        return eps_arcsec / 3600.0

def _ramc_deg(jd_ut1: float, jd_tt: float, lon_east_deg: float) -> float:
    """Right Ascension of the MC = GAST + longitude_E (deg), normalized."""
    return _wrap360(_gast_deg(jd_ut1, jd_tt) + float(lon_east_deg))

def _mc_from_ramc(ramc: float, eps: float) -> float:
    """Compute MC longitude (deg) from RAMC and true obliquity."""
    return _atan2d(_sind(ramc) * _cosd(eps), _cosd(ramc))

def _asc_from_phi_ramc(phi: float, ramc: float, eps: float) -> float:
    """Compute Ascendant longitude (deg) given latitude φ, RAMC, and true obliquity."""
    def _acotd(x: float) -> float:
        return _wrap360(math.degrees(math.atan2(1.0, x)))

    num = -((math.tan(math.radians(phi)) * _sind(eps)) + (_sind(ramc) * _cosd(eps)))
    den = _cosd(ramc)
    # Guard against den → 0 to avoid infs while preserving sign.
    if abs(den) <= 1e-15:
        den = math.copysign(1e-15, den if den != 0 else 1.0)
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
    """
    Recompute Ascendant/MC from fundamentals to correct parity issues from
    upstream house engines. Applies sidereal offset when mode == 'sidereal'.
    """
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
    """
    Lightweight health endpoint focused on core/optional engines.
    Mirrors /api/health but scoped to v2 components.
    """
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
    Build ERFA-aligned timescales from local civil input.

    DUT1 source precedence:
      1) payload["dut1_seconds"] (preferred)
      2) payload["dut1"]         (legacy key)
      3) env ASTRO_DUT1_BROADCAST or ASTRO_DUT1 (fallback, default "0.0")
    """
    # Validate tz early (nice error for bad IANA names)
    try:
        ZoneInfo(str(tz_name))
    except Exception:
        raise ValidationError([{
            "loc": ["tz"],
            "msg": "must be a valid IANA zone like 'Asia/Kolkata'",
            "type": "value_error",
        }])

    def _env_dut1() -> float:
        val = os.getenv("ASTRO_DUT1_BROADCAST", os.getenv("ASTRO_DUT1", "0.0"))
        try:
            return float(val)
        except Exception:
            raise ValidationError([{
                "loc": ["dut1"],
                "msg": "environment DUT1 must be a valid number",
                "type": "value_error",
            }])

    def _parse_payload_dut1(p: Optional[Dict[str, Any]]) -> float:
        """
        Tolerant extractor:
        - Treat None / "" / "null" / "undefined" as "not provided"
        - Accepts either 'dut1_seconds' or legacy 'dut1'
        """
        if not isinstance(p, dict):
            return _env_dut1()
        key = "dut1_seconds" if "dut1_seconds" in p else ("dut1" if "dut1" in p else None)
        if key is None:
            return _env_dut1()
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
        up = msg.upper()
        if "DUT1" in up:
            raise ValidationError([{"loc": ["dut1"], "msg": msg, "type": "value_error"}])
        if "1960" in up or "PRE-1960" in up:
            raise ValidationError([{"loc": ["date"], "msg": msg, "type": "value_error"}])
        raise ValidationError([{"loc": ["timescales"], "msg": msg, "type": "value_error"}])

    return {
        "jd_utc": float(ts.jd_utc),
        "jd_tt": float(ts.jd_tt),
        "jd_ut1": float(ts.jd_ut1),
        "delta_t": float(ts.delta_t),
        "delta_at": float(ts.dat),
        "dut1": float(ts.dut1),
        "timezone": str(tz_name),
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
        from app.core.chart import compute_chart as _compute_chart  # fallback
        _CHART_ENGINE_NAME = "app.core.chart.compute_chart"
        log.warning(
            "Primary astronomy.compute_chart missing; using chart.compute_chart. err=%r", e1
        )
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
    """Return a map of {param_name: bool} indicating if fn accepts each name."""
    try:
        params = fn.__signature__.parameters  # type: ignore[attr-defined]
    except Exception:
        params = inspect.signature(fn).parameters
    return {n: (n in params) for n in names}

# ---------- adapter/kernel meta snapshot (for dev tools visibility) ----------
def _snapshot_ephemeris_meta(chart_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Snapshot adapter/kernel info without forcing a kernel load.
    Values fill in lazily once kernels are actually loaded.
    """
    info: Dict[str, Any] = {}
    try:
        from app.core import ephemeris_adapter as ea  # late import by design

        # Prefer diagnostics (exposes kernels & coverage if available)
        try:
            diag = ea.ephemeris_diagnostics()
        except Exception:
            diag = None

        try:
            info["kernel"] = ea.current_kernel_name()
        except Exception:
            pass

        if isinstance(diag, dict) and isinstance(diag.get("kernels"), list):
            info["kernels"] = list(diag["kernels"])

        try:
            info["ephemeris_path"] = ea.current_kernel_path()
        except Exception:
            pass
        try:
            info["ephemeris_coverage_jd"] = getattr(ea, "KERNEL_COVERAGE_JD", None)
        except Exception:
            pass

        # Tag source engine that produced chart data
        if isinstance(chart_meta, dict):
            src = chart_meta.get("source") or chart_meta.get("engine")
            if src:
                info["source"] = src
        if "source" not in info and _CHART_ENGINE_NAME:
            info["source"] = _CHART_ENGINE_NAME
    except Exception:
        if _CHART_ENGINE_NAME:
            info["source"] = _CHART_ENGINE_NAME
    return info

def _call_compute_chart(payload: Dict[str, Any], ts: Dict[str, Any]) -> Dict[str, Any]:
    """Call whichever chart engine is present, adapting args as needed."""
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
        if isinstance(q.get("ayanamsa"), str):
            q["ayanamsa"] = q["ayanamsa"].strip().lower()
        q.setdefault("jd_ut", ts["jd_utc"])
        q.setdefault("jd_tt", ts["jd_tt"])
        q.setdefault("jd_ut1", ts["jd_ut1"])
        q.setdefault("timescales", ts)
        return q

    if "payload" in param_names:
        try:
            chart = _compute_chart(_normalize_payload_for_engine(payload))  # type: ignore[arg-type]
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
    """Call the available house-engine with normalized args."""
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
        return _houses_fn(**kwargs)
    except Exception as e:
        log.error("House calculation failed: %s: %s", type(e).__name__, e)
        raise

def _call_compute_aspects(payload: Dict[str, Any], chart: Dict[str, Any], houses: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build positions from chart/optional houses and delegate to run_aspects_api.
    Angles are normalized to [0, 360).
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
        aspects_config["declinations"] = decls  # Only enables parallels if requested.

    return run_aspects_api(**aspects_config)

def _extract_ayanamsa_from_chart(chart: Dict[str, Any]) -> Optional[float]:
    """Find ayanamsa in typical chart/meta locations."""
    if not isinstance(chart, dict):
        return None
    meta = chart.get("meta") or {}
    ay = meta.get("ayanamsa_deg")
    if isinstance(ay, (int, float)):
        return float(ay)
    ay2 = chart.get("ayanamsa_deg")
    return float(ay2) if isinstance(ay2, (int, float)) else None

def _normalize_houses_payload(h: Any) -> Any:
    """Normalize house payload keys and wrap angles to [0, 360)."""
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
    """Fix Asc/MC parity issues by recomputing from fundamentals when needed."""
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
    mc_old  = h.get("mc_deg")  if isinstance(h.get("mc_deg"), (int, float))  else h.get("mc")
    warn_list = h.get("warnings") or []
    changed = False

    if not isinstance(asc_old, (int, float)):
        h["asc_deg"] = _wrap360(asc_new); h["asc"] = h["asc_deg"]; changed = True
    # (We compute delta but only enforce correction threshold on MC parity below)
    _ = _delta_arcsec(asc_new, float(asc_old)) if isinstance(asc_old, (int, float)) else None

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
    """Attach a bodies_map[name] → body_record for quick prediction lookups."""
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
    """Strict check: both latitude and longitude must be finite numbers."""
    lat = payload.get("latitude"); lon = payload.get("longitude")
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
    if isinstance(h, dict) and h.get("compute") is False:
        return False
    return True

# ───────────────────────── health / ops ─────────────────────────
@api.get("/api/health")
def health():
    """Lightweight liveness + engine-capability snapshot."""
    def _returns_available() -> bool:
        # consider both the preferred module and the single-file fallback
        return bool(_returns_mod)

    engines = {
        "chart": _CHART_ENGINE_NAME is not None,
        "houses": _HOUSES_KIND != "unavailable",
        "predictions": _PREDICTION_ENGINE_OK,            # legacy alias for callers
        "prediction_engine_v2": _PREDICTION_ENGINE_OK,
        "progressions": compute_progressions is not None,
        "returns": _returns_available(),
        "synastry": _SYN_OK,
        "directions": _DIR_OK,
        "parans": _parans_compute is not None,
        "relocation": _compute_relocated is not None,
        "astrocartography": _compute_astrocartography is not None,
    }

    return jsonify({
        "ok": True,
        "status": "up",
        "version": VERSION,
        "engines": engines,
    }), 200


@api.get("/api/config")
@rate_limit(1)
def config_info():
    """Expose selected server configuration + sample timescales (UTC now)."""
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    calib_path = os.environ.get("ASTRO_CALIBRATORS", "config/calibrators.json")
    th_path = os.environ.get("ASTRO_HC_THRESHOLDS", "config/hc_thresholds.json")

    cfg = load_config(cfg_path)
    calib_ver = None
    th_summary = None

    # Try to compute a small, current timescale snapshot in UTC
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

    # Optional calibrators / thresholds metadata
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_ver = (json.load(f) or {}).get("version")
    except Exception:
        pass

    try:
        with open(th_path, "r", encoding="utf-8") as f:
            th = json.load(f) or {}
            th_summary = {
                "entropy_H": th.get("entropy_H"),
                "defaults": th.get("defaults"),
            }
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "mode": cfg.mode,
        "ayanamsa": getattr(cfg, "ayanamsa", None),
        "rate_limits_per_hour": getattr(cfg, "rate_limits_per_hour", None),
        "pro_features_enabled": getattr(cfg, "pro_features_enabled", None),
        "calibrators_version": calib_ver,
        "hc_thresholds_summary": th_summary,
        "timescale_sample": ts_sample,
        "version": VERSION,
    }), 200


@api.get("/api/openapi")
@rate_limit(RL_DEBUG)
def openapi_spec():
    """Return parsed OpenAPI YAML bundled with the app (if present)."""
    import yaml
    base = os.path.dirname(__file__)
    candidates = (
        os.path.join(base, "..", "openapi.yaml"),
        os.path.join(base, "..", "..", "openapi.yaml"),
    )
    for p in candidates:
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
    """Build ERFA-aligned timescales from a local civil instant."""
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

        ts = _compute_timescales_from_local(
            date, time_, tz, payload=body if isinstance(body, dict) else None
        )
        return jsonify({"ok": True, "timescales": ts}), 200

    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

# ───────────────────────── endpoints ─────────────────────────
from time import perf_counter

# Optional ephemeris fallback (same adapter the returns module uses)
try:
    from app.core.ephemeris_adapter import EphemerisAdapter
    _EPH_FALLBACK_ERR = None
except Exception as _e:
    EphemerisAdapter = None
    _EPH_FALLBACK_ERR = _e

MAJORS = ("Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")

# ---------- robust helpers ----------

def _coerce_float(v: Any) -> Optional[float]:
    """Safely coerce to finite float; return None on NaN/Inf/TypeError."""
    try:
        x = float(v)
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return x
    except Exception:
        return None


def _canon_name(name: Any) -> Optional[str]:
    """Normalize a body name; prefer canonical capitalization for MAJORS."""
    if not isinstance(name, str):
        return None
    n = name.strip()
    for m in MAJORS:
        if n.lower() == m.lower():
            return m
    return n or None


def _find_lon_in_dict(d: Dict[str, Any]) -> Optional[float]:
    """Search common longitude keys, including nested 'ecliptic' objects."""
    if not isinstance(d, dict):
        return None

    # direct candidates
    for k in ("lon", "longitude", "lon_deg", "lambda", "lam", "ecl_lon", "ecliptic_lon"):
        if k in d:
            v = _coerce_float(d[k])
            if v is not None:
                return v

    # nested ecliptic containers
    ecl = d.get("ecliptic")
    if isinstance(ecl, dict):
        for k in ("lon", "longitude", "lon_deg", "lambda", "lam"):
            if k in ecl:
                v = _coerce_float(ecl[k])
                if v is not None:
                    return v

    return None


def _rows_to_positions(rows: Any) -> Dict[str, float]:
    """Normalize a list of row dicts into {Name: longitude_deg}."""
    out: Dict[str, float] = {}
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = _canon_name(r.get("name") or r.get("body"))
        lon = _find_lon_in_dict(r)
        if name and lon is not None:
            out[name] = float(lon)
    return out


def _mapping_to_positions(mp: Any) -> Dict[str, float]:
    """Normalize a mapping into {Name: longitude_deg} handling nested dicts."""
    out: Dict[str, float] = {}
    if not isinstance(mp, dict):
        return out
    for k, v in mp.items():
        name = _canon_name(k)
        if not name:
            continue
        if isinstance(v, (int, float, str)):
            lon = _coerce_float(v)
        elif isinstance(v, dict):
            lon = _find_lon_in_dict(v)
        else:
            lon = None
        if lon is not None:
            out[name] = float(lon)
    return out


def _extract_positions_from_chart(chart: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize planet longitudes into: { "Sun": deg, "Moon": deg, ... }
    Handles multiple chart shapes; returns {} if nothing found.
    """
    if not isinstance(chart, dict):
        return {}

    # 1) Preferred dicts
    for key in ("positions", "planets"):
        if isinstance(chart.get(key), dict):
            out = _mapping_to_positions(chart[key])
            if out:
                return out

    # 2) Lists
    for key in ("planets", "rows"):
        if isinstance(chart.get(key), list):
            out = _rows_to_positions(chart[key])
            if out:
                return out

    # 3) Common nested container
    ephem = chart.get("ephemeris")
    if isinstance(ephem, dict):
        if isinstance(ephem.get("positions"), dict):
            out = _mapping_to_positions(ephem["positions"])
            if out:
                return out
        if isinstance(ephem.get("rows"), list):
            out = _rows_to_positions(ephem["rows"])
            if out:
                return out

    return {}


def _ayanamsa_from(chart: Dict[str, Any], payload: Dict[str, Any]) -> float:
    """Extract ayanamsa (deg) from chart/meta or payload; default 0.0."""
    try:
        ay = _extract_ayanamsa_from_chart(chart)  # defined earlier in this module
        if isinstance(ay, (int, float)):
            return float(ay)
    except Exception:
        pass
    ay = payload.get("ayanamsa")
    return float(ay) if isinstance(ay, (int, float)) else 0.0


def _snapshot_ephemeris_fallback(
    jd_tt: float,
    payload: Dict[str, Any],
    chart: Dict[str, Any],
) -> Dict[str, float]:
    """
    Use EphemerisAdapter directly if the chart didn’t provide positions (or Sun).
    Returns a {Name: longitude_deg} mapping in the requested mode.
    """
    if EphemerisAdapter is None:
        return {}

    bodies: List[str] = list(payload.get("bodies") or MAJORS)
    frame = "ecliptic-of-date"

    try:
        adapter = EphemerisAdapter(frame=frame)
    except Exception:
        return {}

    # topocentric if place present
    lat = payload.get("latitude")
    lon = payload.get("longitude")
    elev = payload.get("elev_m") or payload.get("elevation_m") or 0.0
    topocentric = isinstance(lat, (int, float)) and isinstance(lon, (int, float))

    kwargs = {"jd_tt": float(jd_tt), "bodies": bodies}
    if topocentric:
        kwargs.update({
            "topocentric": True,
            "latitude": float(lat),
            "longitude": float(lon),
            "elevation_m": float(elev),
        })
    else:
        kwargs["topocentric"] = False

    rows: List[Dict[str, Any]] = []

    # Prefer method with velocities (stable schema); fall back to longitudes
    for method_name in ("ecliptic_longitudes_and_velocities", "ecliptic_longitudes"):
        if not hasattr(adapter, method_name):
            continue
        try:
            method = getattr(adapter, method_name)
            res = method(**kwargs)
            # Normalize result to rows
            if isinstance(res, dict):
                if isinstance(res.get("results"), list):
                    rows = [r for r in res["results"] if isinstance(r, dict)]
                else:
                    # flat dict mapping name->lon or name->{lon:...}
                    rows = [{"name": k, **(v if isinstance(v, dict) else {"lon": v})} for k, v in res.items()]
            elif isinstance(res, list):
                rows = [r for r in res if isinstance(r, dict)]
            if rows:
                break
        except Exception:
            continue

    positions = _rows_to_positions(rows)

    # Sidereal correction if requested
    mode = (payload.get("mode") or "tropical").lower()
    if mode == "sidereal" and positions:
        ay = _ayanamsa_from(chart, payload)
        if ay:
            out = {}
            for k, v in positions.items():
                x = (v - ay) % 360.0
                out[k] = 0.0 if abs(x) < 1e-12 else x
            positions = out

    return positions


# ---------- endpoint: /api/calculate ----------

@api.post("/api/calculate")
@rate_limit(RL_CALCULATE)
def calculate():
    t0 = perf_counter()
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)

        # house system (normalized)
        hs = str(body.get("house_system", "")).strip()
        if hs:
            payload["house_system"] = hs

        # passthroughs
        for k in ("bodies", "points", "ayanamsa", "topocentric", "elevation_m", "elev_m", "dut1", "houses"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    want_houses = _want_houses(body)
    payload["houses"] = bool(want_houses)

    # ensure majors unless caller specified
    payload.setdefault("bodies", list(MAJORS))

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    try:
        ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name, payload=payload)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    # compute chart (resilient)
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

    # optional houses
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

        # apply sidereal rotation to houses if needed
        mode = (payload.get("mode") or "tropical").lower()
        if mode == "sidereal":
            ay = _ayanamsa_from(chart, payload)
            if isinstance(ay, (int, float)) and isinstance(houses, dict):
                def rot(v: Optional[float]) -> Optional[float]:
                    if v is None:
                        return None
                    x = (float(v) - float(ay)) % 360.0
                    return 0.0 if abs(x) < 1e-12 else x
                for k in ("asc", "asc_deg", "mc", "mc_deg", "vertex", "eastpoint"):
                    if isinstance(houses.get(k), (int, float)):
                        houses[k] = rot(houses[k])
                if isinstance(houses.get("cusps"), list):
                    houses["cusps"] = [rot(c) for c in houses["cusps"]]
                if isinstance(houses.get("cusps_deg"), list):
                    houses["cusps_deg"] = [rot(c) for c in houses["cusps_deg"]]

        houses = _recompute_houses_angles_if_needed(houses, ts, payload, chart)

    # meta
    meta = {
        "timescales": ts,
        "timescales_locked": True,
        "chart_engine": _CHART_ENGINE_NAME,
        "houses_engine": _HOUSES_KIND if want_houses else "skipped",
        "houses_requested": bool(want_houses),
    }
    meta.update(_snapshot_ephemeris_meta(chart.get("meta")))
    warnings = meta.setdefault("warnings", [])

    # normalize positions (from chart), then guarantee via fallback if Sun missing
    positions = _extract_positions_from_chart(chart)
    if not positions or "Sun" not in positions:
        snap = _snapshot_ephemeris_fallback(ts["jd_tt"], payload, chart)
        if snap:
            # merge, prefer chart values when present
            for k, v in snap.items():
                positions.setdefault(k, v)
            warnings.append("positions_fallback_ephemeris")
        else:
            warnings.append("positions_missing_and_no_fallback")

    # aspects (optional)
    aspects_result = None
    if body.get("aspects", False):
        try:
            aspects_result = _call_compute_aspects(payload, chart, houses)
        except Exception as e:
            if DEBUG_VERBOSE:
                aspects_result = {"error": str(e)}

    # response
    resp = {
        "ok": True,
        "timescales": ts,
        "chart": chart,
        "meta": meta,
        "positions": positions,   # stable place for majors (incl. Sun)
    }
    if want_houses:
        resp["houses"] = houses
    if aspects_result:
        resp["aspects"] = aspects_result

    ms = (perf_counter() - t0) * 1000.0
    return jsonify(resp), 200, {"X-Compute-Time-ms": f"{ms:.0f}"}


# ---------- endpoint: /api/report ----------

@api.post("/api/report")
@rate_limit(RL_REPORT)
def report():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)

        # normalize house system if provided
        hs = str(body.get("house_system", "")).strip()
        if hs:
            payload["house_system"] = hs

        # passthroughs
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
    def _norm_orbs(orbs_raw: Any) -> Dict[str, float]:
        """Normalize `orbs` into {aspect_name_lower: float_orb}."""
        if not isinstance(orbs_raw, dict):
            return {}
        out: Dict[str, float] = {}
        for k, v in orbs_raw.items():
            try:
                out[str(k).strip().lower()] = float(v)
            except Exception:
                # skip non-numeric values
                continue
        return out

    def _norm_aspects(aspects_raw: Any) -> Optional[List[str]]:
        """Normalize `aspects` into a list[str] (lower-cased)."""
        if aspects_raw is None:
            return None
        if isinstance(aspects_raw, str):
            s = aspects_raw.strip().lower()
            return [s] if s else []
        if isinstance(aspects_raw, (list, tuple)):
            return [str(a).strip().lower() for a in aspects_raw if str(a).strip()]
        return None

    def _norm_bodies(bodies_raw: Any) -> Optional[List[str]]:
        """Normalize `bodies` into a list[str] (original case preserved)."""
        if bodies_raw is None:
            return None
        if isinstance(bodies_raw, str):
            s = bodies_raw.strip()
            return [s] if s else []
        if isinstance(bodies_raw, (list, tuple)):
            return [str(b).strip() for b in bodies_raw if str(b).strip()]
        return None

    # ---- parse & normalize input ------------------------------------------------
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)

        # Optional knobs for aspects engine
        if "orbs" in body:
            payload["orbs"] = _norm_orbs(body.get("orbs"))
        if "aspects" in body:
            payload["aspects"] = _norm_aspects(body.get("aspects"))
        if "bodies" in body:
            payload["bodies"] = _norm_bodies(body.get("bodies"))
        if "points" in body:
            payload["points"] = body.get("points")  # pass-through (engine decides)
        if "mode" in body:
            payload["mode"] = str(body.get("mode")).strip().lower()
        if "houses" in body:
            payload["houses"] = bool(body.get("houses"))
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # ---- timescales -------------------------------------------------------------
    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    try:
        ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name, payload=payload)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    # ---- chart -----------------------------------------------------------------
    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        return _json_error("chart_internal", str(e) if DEBUG_VERBOSE else "chart_failed", 500)

    # ---- houses (optional) -----------------------------------------------------
    houses = None
    if payload.get("houses", True):
        try:
            _require_coords_for_houses(payload)
            houses = _call_compute_houses(payload, ts)
            houses = _normalize_houses_payload(houses)
        except Exception as e:
            return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "houses_failed", 500)

    # ---- aspects ----------------------------------------------------------------
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
    """Whether a returns engine (module or fallback) is available."""
    return _returns_mod is not None

# ───────────────────────── PREDICTION ENGINE API (Flask • Clean • Stable) ─────────────────────────
# Drop-in block: relies on globals already defined in your routes module:
#   api, rate_limit, RL_PREDICTION_*, _json_error, _snapshot_ephemeris_meta,
#   _CHART_ENGINE_NAME, _HOUSES_KIND, _PREDICTION_ENGINE_OK, _PREDICTION_ENGINE_ERR,
#   comprehensive_forecast, predict_transits, predict_progressions, predict_returns, predict_directions,
#   relationship_forecast, validate_prediction_model

from typing import Any, Dict, List, Optional, FrozenSet, Tuple
from flask import jsonify, Response, request

# ───────────────────────── Fast JSON (optional) ─────────────────────────
try:
    import orjson
    _FAST_JSON = True
except Exception:
    _FAST_JSON = False

def _json(payload: Dict[str, Any], status: int = 200) -> Response:
    if _FAST_JSON:
        return Response(orjson.dumps(payload), status=status, mimetype="application/json")
    resp = jsonify(payload)
    resp.status_code = status
    return resp

# ───────────────────────── Validation (prefer v2 from validators) ─────────────────────────
try:
    from app.core.validators import (
        ValidationError as _ValidationError,
        parse_prediction_payload_v2 as _parse_v2,
    )
except Exception:
    _parse_v2 = None
    class _ValidationError(Exception):
        def __init__(self, details):
            super().__init__("validation_error")
            self._details = details if isinstance(details, list) else [
                {"loc": [], "msg": str(details), "type": "value_error"}
            ]
        def errors(self):
            return self._details

def parse_prediction_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper over validators.parse_prediction_payload_v2 with a resilient fallback."""
    if _parse_v2 is not None:
        return _parse_v2(body)

    # Minimal, defensive fallback
    if not isinstance(body, dict):
        raise _ValidationError([{"loc": [], "msg": "payload must be an object", "type": "type_error.dict"}])
    if "natal_chart" not in body or not isinstance(body.get("natal_chart"), dict):
        raise _ValidationError([{"loc": ["natal_chart"], "msg": "required object", "type": "value_error"}])
    if "time_range" in body:
        tr = body.get("time_range")
        if not (isinstance(tr, (list, tuple)) and len(tr) == 2):
            raise _ValidationError([{"loc": ["time_range"], "msg": "must be [start_date, end_date] array", "type": "value_error"}])
    if "target_date" in body:
        from datetime import datetime
        td = body.get("target_date")
        if not isinstance(td, (str, int, float, datetime)):
            raise _ValidationError([{"loc": ["target_date"], "msg": "must be ISO string, JD float, or datetime", "type": "value_error"}])
    return body

# ───────────────────────── Helpers ─────────────────────────
def _engine_guard() -> Optional[Response]:
    """Return a 503 Response if the prediction engine is unavailable; else None."""
    if not _PREDICTION_ENGINE_OK:
        det = {"import_error": repr(_PREDICTION_ENGINE_ERR)} if _PREDICTION_ENGINE_ERR else None
        return _json_error("prediction_engine_unavailable", det or "prediction engine not available", 503)
    return None

def _build_meta(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = {
        "prediction_engine": "app.core.prediction v2",
        "natal_chart_engine": _CHART_ENGINE_NAME,
        "houses_engine": _HOUSES_KIND,
    }
    try:
        meta.update(_snapshot_ephemeris_meta())
    except Exception:
        pass
    if extra:
        meta.update(extra)
    return meta

def _accept(payload: Dict[str, Any], accept_keys: FrozenSet[str]) -> Dict[str, Any]:
    """Filter `payload` down to the keys the backend actually accepts."""
    return {k: payload[k] for k in payload.keys() & accept_keys}

# ───────────────────────── Serializer shims (guard against NameError) ─────────────────────────
if "_serialize_prediction_event" not in globals():
    def _serialize_prediction_event(event: Any) -> Dict[str, Any]:
        if event is None:
            return {}
        return {
            "event_type": getattr(event, "event_type", None),
            "technique": getattr(event, "technique", None),
            "description": getattr(event, "description", None),
            "datetime_utc": (
                getattr(event, "datetime_utc", None).isoformat()
                if hasattr(event, "datetime_utc") and getattr(event, "datetime_utc") else None
            ),
            "jd_tt": getattr(event, "jd_tt", None),
            "jd_ut1": getattr(event, "jd_ut1", None),
            "precision_seconds": getattr(event, "precision_seconds", None),
            "confidence": getattr(event, "confidence", None),
            "significance": getattr(event, "significance", None),
            "metadata": getattr(event, "metadata", {}) or {},
        }

if "_serialize_timing_window" not in globals():
    def _serialize_timing_window(window: Any) -> Dict[str, Any]:
        if window is None:
            return {}
        return {
            "start_jd_tt": getattr(window, "start_jd_tt", None),
            "end_jd_tt": getattr(window, "end_jd_tt", None),
            "peak_jd_tt": getattr(window, "peak_jd_tt", None),
            "uncertainty_days": getattr(window, "uncertainty_days", None),
            "confidence_interval": getattr(window, "confidence_interval", None),
        }

if "_serialize_prediction_result" not in globals():
    def _serialize_prediction_result(result: Any) -> Dict[str, Any]:
        if result is None:
            return {}
        return {
            "ok": getattr(result, "ok", True),
            "technique": getattr(result, "technique", None),
            "events": [_serialize_prediction_event(e) for e in getattr(result, "events", [])],
            "synthesis": getattr(result, "synthesis", None),
            "timing_windows": [_serialize_timing_window(w) for w in getattr(result, "timing_windows", [])],
            "confidence_score": getattr(result, "confidence_score", None),
            "statistical_metrics": getattr(result, "statistical_metrics", {}) or {},
            "warnings": getattr(result, "warnings", []) or [],
            "metadata": getattr(result, "metadata", {}) or {},
            "computation_time_ms": getattr(result, "computation_time_ms", None),
        }

if "_serialize_comprehensive_forecast" not in globals():
    def _serialize_comprehensive_forecast(forecast: Any) -> Dict[str, Any]:
        if forecast is None:
            return {}
        predictions: Dict[str, Any] = {}
        if hasattr(forecast, "predictions") and forecast.predictions:
            predictions = {k: _serialize_prediction_result(v) for k, v in forecast.predictions.items()}

        time_range_iso: List[str] = []
        tr = getattr(forecast, "time_range", [])
        if isinstance(tr, (list, tuple)) and len(tr) >= 2:
            time_range_iso = [
                (dt.isoformat() if hasattr(dt, "isoformat") else str(dt)) for dt in tr[:2]
            ]

        return {
            "natal_chart": getattr(forecast, "natal_chart", {}) or {},
            "time_range": time_range_iso,
            "predictions": predictions,
            "synthesis": getattr(forecast, "synthesis", {}) or {},
            "peak_periods": [_serialize_timing_window(w) for w in getattr(forecast, "peak_periods", [])],
            "risk_assessment": getattr(forecast, "risk_assessment", {}) or {},
            "confidence_metrics": getattr(forecast, "confidence_metrics", {}) or {},
            "validation_results": getattr(forecast, "validation_results", {}) or {},
            "computation_time_ms": getattr(forecast, "computation_time_ms", None),
        }

if "_serialize_relationship_forecast" not in globals():
    def _serialize_relationship_forecast(forecast: Any) -> Dict[str, Any]:
        if forecast is None:
            return {}
        return {
            "synastry_analysis": getattr(forecast, "synastry_analysis", {}) or {},
            "composite_analysis": getattr(forecast, "composite_analysis", {}) or {},
            "transit_interactions": [_serialize_prediction_event(e) for e in getattr(forecast, "transit_interactions", [])],
            "progression_interactions": [_serialize_prediction_event(e) for e in getattr(forecast, "progression_interactions", [])],
            "compatibility_trends": getattr(forecast, "compatibility_trends", {}) or {},
            "critical_periods": [_serialize_timing_window(w) for w in getattr(forecast, "critical_periods", [])],
            "relationship_score": getattr(forecast, "relationship_score", None),
            "confidence_metrics": getattr(forecast, "confidence_metrics", {}) or {},
        }

# ───────────────────────── Accept sets ─────────────────────────
_ACCEPT_TRANSITS: FrozenSet[str] = frozenset({
    "transiting_bodies", "natal_bodies", "orbs", "aspects", "include_aspects_to",
    "include_house_cusps", "frame", "zodiac_mode", "ayanamsa_deg", "exact_timing",
    "statistical_validation", "confidence_threshold",
})

_ACCEPT_PROGRESSIONS: FrozenSet[str] = frozenset({
    "method", "lunar_month", "tertiary_mode", "frame", "house_system",
    "zodiac_mode", "ayanamsa_deg", "aspects_to_natal", "orbs",
    "parallels", "antiscia", "statistical_validation",
})

_ACCEPT_RETURNS: FrozenSet[str] = frozenset({
    "lunar_month", "place", "frame", "house_system", "zodiac_mode",
    "ayanamsa_deg", "estimate_uncertainty", "aspects_to_natal", "orbs",
    "statistical_validation",
})

# IMPORTANT: Do NOT forward 'aspects_to_natal' to directions (legacy backends can choke)
_ACCEPT_DIRECTIONS: FrozenSet[str] = frozenset({
    "method", "frame", "zodiac_mode", "ayanamsa_deg", "house_system",
    "orbs", "statistical_validation",
})

_ACCEPT_RELATIONSHIP: FrozenSet[str] = frozenset({
    "synastry_orbs", "composite_method", "include_transits_to_composite",
    "include_progressions", "confidence_threshold", "parallels", "antiscia",
    "frame", "zodiac_mode", "ayanamsa_deg", "house_system",
})
_REL_PREFIXES: Tuple[str, ...] = ("transit_", "progression_")

# ───────────────────────── Routes ─────────────────────────

@api.post("/api/prediction/forecast")
@rate_limit(RL_PREDICTION_FORECAST)
def prediction_comprehensive_forecast_route():
    if (guard := _engine_guard()) is not None:
        return guard
    try:
        body = request.get_json(silent=False) or {}
        payload = parse_prediction_payload(body)
    except _ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e), 400)

    try:
        natal_chart = payload["natal_chart"]
        tr = payload.get("time_range")
        if not tr or not (isinstance(tr, (list, tuple)) and len(tr) == 2):
            return _json_error("validation_error", [{"loc": ["time_range"], "msg": "required"}], 400)
        time_range = (tr[0], tr[1])

        techniques = payload.get("techniques")
        confidence_threshold = float(payload.get("confidence_threshold", 0.2))
        synthesis_method = payload.get("synthesis_method", "weighted_consensus")
        statistical_validation = bool(payload.get("statistical_validation", False))
        include_vedic = bool(payload.get("include_vedic", False))
        peak_window_days = int(payload.get("peak_window_days", 14))

        tk_kwargs = {k: v for k, v in payload.items()
                     if k.startswith("transit_") or k.startswith("progression_") or
                        k.startswith("return_") or k.startswith("vedic_")}

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

        return _json({
            "ok": True,
            "forecast": _serialize_comprehensive_forecast(forecast),
            "meta": _build_meta({"computation_time_ms": getattr(forecast, "computation_time_ms", None)}),
        }, 200)

    except Exception as e:
        return _json_error("prediction_internal", str(e), 500)


@api.post("/api/prediction/transits")
@rate_limit(RL_PREDICTION_TRANSITS)
def prediction_transits_route():
    if (guard := _engine_guard()) is not None:
        return guard
    try:
        body = request.get_json(silent=False) or {}
        payload = parse_prediction_payload(body)

        natal_chart = payload["natal_chart"]
        tr = payload.get("time_range")
        if not tr or not (isinstance(tr, (list, tuple)) and len(tr) == 2):
            return _json_error("validation_error", [{"loc": ["time_range"], "msg": "required"}], 400)
        time_range = (tr[0], tr[1])

        kwargs = _accept(payload, _ACCEPT_TRANSITS)
        result = predict_transits(natal_chart=natal_chart, time_range=time_range, **kwargs)

        return _json({
            "ok": getattr(result, "ok", False),
            "result": _serialize_prediction_result(result),
            "meta": _build_meta({
                "technique": getattr(result, "technique", "transits"),
                "computation_time_ms": getattr(result, "computation_time_ms", None),
            }),
        }, 200)

    except _ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("transits_internal", str(e), 500)


@api.post("/api/prediction/progressions")
@rate_limit(RL_PREDICTION_PROGRESSIONS)
def prediction_progressions_route():
    if (guard := _engine_guard()) is not None:
        return guard
    try:
        body = request.get_json(silent=False) or {}
        payload = parse_prediction_payload(body)

        natal_chart = payload["natal_chart"]
        target_date = payload.get("target_date")
        if target_date is None:
            return _json_error("validation_error", [{"loc": ["target_date"], "msg": "required"}], 400)

        kwargs = _accept(payload, _ACCEPT_PROGRESSIONS)
        result = predict_progressions(natal_chart=natal_chart, target_date=target_date, **kwargs)

        return _json({
            "ok": getattr(result, "ok", False),
            "result": _serialize_prediction_result(result),
            "meta": _build_meta({
                "technique": getattr(result, "technique", "progressions"),
                "computation_time_ms": getattr(result, "computation_time_ms", None),
            }),
        }, 200)

    except _ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("progressions_internal", str(e), 500)


@api.post("/api/prediction/returns")
@rate_limit(RL_PREDICTION_RETURNS)
def prediction_returns_route():
    if (guard := _engine_guard()) is not None:
        return guard
    try:
        body = request.get_json(silent=False) or {}
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

        kwargs = _accept(payload, _ACCEPT_RETURNS)
        result = predict_returns(natal_chart=natal_chart, return_type=return_type, year=int(year), **kwargs)

        return _json({
            "ok": getattr(result, "ok", False),
            "result": _serialize_prediction_result(result),
            "meta": _build_meta({
                "technique": getattr(result, "technique", "returns"),
                "computation_time_ms": getattr(result, "computation_time_ms", None),
            }),
        }, 200)

    except _ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("returns_internal", str(e), 500)


@api.post("/api/prediction/directions")
@rate_limit(RL_PREDICTION_DIRECTIONS)
def prediction_directions_route():
    if (guard := _engine_guard()) is not None:
        return guard
    try:
        body = request.get_json(silent=False) or {}
        payload = parse_prediction_payload(body)

        natal_chart = payload["natal_chart"]
        target_date = payload.get("target_date")
        if target_date is None:
            return _json_error("validation_error", [{"loc": ["target_date"], "msg": "required"}], 400)

        kwargs = _accept(payload, _ACCEPT_DIRECTIONS)  # intentionally excludes 'aspects_to_natal'
        result = predict_directions(natal_chart=natal_chart, target_date=target_date, **kwargs)

        return _json({
            "ok": getattr(result, "ok", False),
            "result": _serialize_prediction_result(result),
            "meta": _build_meta({
                "technique": getattr(result, "technique", "directions"),
                "computation_time_ms": getattr(result, "computation_time_ms", None),
            }),
        }, 200)

    except _ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("directions_internal", str(e), 500)


@api.post("/api/prediction/relationship")
@rate_limit(RL_PREDICTION_RELATIONSHIP)
def prediction_relationship_route():
    if (guard := _engine_guard()) is not None:
        return guard
    try:
        body = request.get_json(silent=False) or {}

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

        kwargs = _accept(body, _ACCEPT_RELATIONSHIP)
        for k, v in body.items():
            if k.startswith(_REL_PREFIXES):
                kwargs[k] = v

        result = relationship_forecast(
            natal_a=natal_a,
            natal_b=natal_b,
            time_range=(time_range[0], time_range[1]),
            **kwargs,
        )

        return _json({
            "ok": True,
            "result": _serialize_relationship_forecast(result),
            "meta": _build_meta({"technique": "relationship_forecast"}),
        }, 200)

    except _ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("relationship_internal", str(e), 500)


@api.post("/api/prediction/validate")
@rate_limit(RL_PREDICTION_VALIDATION)
def prediction_validation_route():
    if (guard := _engine_guard()) is not None:
        return guard
    try:
        body = request.get_json(silent=False) or {}
        test_cases = body.get("test_cases")
        if not isinstance(test_cases, list) or not test_cases:
            return _json_error("validation_error", [{"loc": ["test_cases"], "msg": "required non-empty array"}], 400)

        kwargs = _accept(body, frozenset({"validation_method", "n_folds", "metrics", "confidence_threshold"}))
        result = validate_prediction_model(test_cases=test_cases, **kwargs)

        return _json({
            "ok": result.get("ok", True),
            "result": result,
            "meta": _build_meta({"technique": "model_validation"}),
        }, 200)

    except _ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("validation_internal", str(e), 500)


# ───────────────────────── predictive (transits • validation • dasha • varga • yogas) ─────────────────────────
# Cleaned & deduplicated drop-in. Keeps behavior; improves structure, typing, and guards.

import threading
import time
from datetime import date, datetime
from enum import Enum

# External app symbols expected to exist:
# - api (Flask Blueprint), rate_limit, RL_PREDICTIVE
# - _json_error, DEBUG_VERBOSE, ValidationError
# - parse_frame, parse_latlon, _compute_timescales_from_local, _call_compute_chart, _wrap360
# - app.core.predictive as pred  (must expose TransitEngine, MAJOR_ASPECTS, MINOR_ASPECTS)

from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
from app.core import predictive as pred  # TransitEngine, aspect presets


# ───────────────────────── JSON safety (utility) ─────────────────────────

try:
    import numpy as _np
    _HAS_NP = True
except Exception:
    _HAS_NP = False


def _json_safe(x: Any) -> Any:
    """Recursively coerce common non-JSON types into JSON-safe primitives."""
    if x is None or isinstance(x, (bool, int, float, str)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, Enum):
        return x.name.lower()
    if _HAS_NP:
        if isinstance(x, _np.floating):
            v = float(x)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(x, _np.integer):
            return int(x)
        if isinstance(x, _np.ndarray):
            return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (list, tuple, set)):
        it = x if not isinstance(x, set) else sorted(x, key=lambda y: str(y))
        return [_json_safe(v) for v in it]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    return str(x)


# ───────────────────────── Shared ephemeris adapters (perf) ─────────────────────────

_GLOBAL_ADAPTERS: Dict[str, EphemerisAdapter] = {}
_ADAPTER_LOCK = threading.Lock()


def _norm_frame_key(frame: Optional[str]) -> str:
    s = (frame or "ecliptic-of-date").strip().lower()
    return "ecliptic-j2000" if s in {"ecliptic-j2000", "j2000", "ecl-j2000"} else "ecliptic-of-date"


def get_shared_adapter(frame: str = "ecliptic-of-date") -> EphemerisAdapter:
    """Get or create a shared EphemerisAdapter instance (per-frame cache)."""
    key = _norm_frame_key(frame)
    with _ADAPTER_LOCK:
        ep = _GLOBAL_ADAPTERS.get(key)
        if ep is None:
            ep = EphemerisAdapter(EphemConfig(frame=key, compute_velocity_default=False))
            _GLOBAL_ADAPTERS[key] = ep
        return ep


# ───────────────────────── Concurrency gate (burst protection) ─────────────────────────

_PRED_MAX_CONC = int(os.getenv("PREDICTIVE_MAX_CONCURRENCY", "2"))
_PRED_SEM_TIMEOUT_S = float(os.getenv("PREDICTIVE_SEM_TIMEOUT_S", "25"))
_PRED_SEM = threading.Semaphore(_PRED_MAX_CONC)


def _busy():
    resp = _json_error("server_busy", {"msg": "try again shortly"}, 429)
    resp.headers["Retry-After"] = "2"
    return resp


def _take_gate() -> bool:
    try:
        return bool(_PRED_SEM.acquire(timeout=_PRED_SEM_TIMEOUT_S))
    except Exception:
        return False


def _give_gate():
    try:
        _PRED_SEM.release()
    except Exception:
        pass


# ───────────────────────── Time parsing helpers ─────────────────────────

def _split_dt(s: Any, fallback_time: str) -> Tuple[str, str]:
    """Split 'YYYY-MM-DD[ T]HH:MM:SS' into (date, time) with sensible fallbacks."""
    s = (str(s or "")).strip().replace("T", " ")
    if len(s) <= 10:
        return s[:10], fallback_time
    return s[:10], (s[11:19] or fallback_time)


def _parse_time_range_like(body: Dict[str, Any]) -> Tuple[float, float]:
    """
    Accepts:
      - jd_start_tt & jd_end_tt (floats)
      - time_range: [start, end] (each ISO date or ISO datetime)
      - legacy date/time + place_tz/timezone
    Returns (jd0, jd1) in TT.
    """
    # Direct JDs
    jd0, jd1 = body.get("jd_start_tt"), body.get("jd_end_tt")
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
        if not (math.isfinite(jd0f) and math.isfinite(jd1f) and jd1f > jd0f):
            raise ValidationError([{"loc": ["time_range"], "msg": "end must be after start"}])
        return jd0f, jd1f

    # Legacy civil fields
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


def _parse_step_minutes(v: Any, *, default_min: float) -> float | str:
    """
    Accept numeric minutes (>0) or "auto"/0/None → "auto".
    Returns float minutes or the string "auto".
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


# ───────────────────────── Aspect presets & normalization ─────────────────────────

_MAJOR_NAMES = list(getattr(pred, "MAJOR_ASPECTS", ("conjunction", "opposition", "square", "trine", "sextile")))
_MINOR_NAMES = list(getattr(pred, "MINOR_ASPECTS", ("quincunx", "semisextile", "semisquare", "sesquisquare", "quintile", "biquintile")))


def _normalize_aspects(x: Any, *, include_minors: bool) -> List[str]:
    """
    Accepts list/tuple/set/str or dict {name:angle} or list of dicts with 'name'.
    Falls back to major/minor defaults by flag.
    """
    if x is None:
        return (_MAJOR_NAMES + _MINOR_NAMES) if include_minors else list(_MAJOR_NAMES)

    if isinstance(x, dict):
        return [str(k).strip().lower() for k in x.keys()]

    if isinstance(x, (list, tuple, set)):
        out: List[str] = []
        for a in x:
            if isinstance(a, str):
                out.append(a.strip().lower())
            elif isinstance(a, dict) and "name" in a:
                out.append(str(a["name"]).strip().lower())
        return out or ((_MAJOR_NAMES + _MINOR_NAMES) if include_minors else list(_MAJOR_NAMES))

    if isinstance(x, str):
        return [x.strip().lower()]

    return (_MAJOR_NAMES + _MINOR_NAMES) if include_minors else list(_MAJOR_NAMES)


# ───────────────────────── Route: /api/predictive/transits ─────────────────────────

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
        raw_movers = body.get("movers") or ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
        if not (isinstance(raw_movers, list) and all(isinstance(x, str) and x for x in raw_movers)):
            return _json_error("validation_error", [{"loc": ["movers"], "msg": "must be a list of names"}], 400)
        movers = list(dict.fromkeys(m.strip() for m in raw_movers if m))

        # targets_longitudes: direct map (preferred)
        targets: Dict[str, float] = {}
        raw_targets = body.get("targets_longitudes")
        if isinstance(raw_targets, dict):
            for k, v in raw_targets.items():
                try:
                    targets[str(k)] = _wrap360(float(v))
                except Exception:
                    pass

        # Or build targets from a chart payload (e.g., natal)
        if not targets and isinstance(body.get("targets_chart"), dict):
            targ = dict(body["targets_chart"])
            tz_nat = targ.get("place_tz") or targ.get("timezone") or "UTC"
            try:
                ts_nat = _compute_timescales_from_local(
                    targ["date"],
                    targ.get("time", "00:00:00"),
                    tz_nat,
                    payload=targ,
                )
            except Exception as e:
                return _json_error("validation_error", [{"loc": ["targets_chart"], "msg": str(e)}], 400)

            try:
                ch = _call_compute_chart(targ, ts_nat)
            except Exception as e:
                return _json_error("chart_internal", str(e) if DEBUG_VERBOSE else "chart_failed", 500)

            for row in (ch.get("bodies") or []) + (ch.get("points") or []):
                if isinstance(row, dict) and "name" in row and isinstance(row.get("longitude_deg"), (int, float)):
                    targets[str(row["name"])] = _wrap360(float(row["longitude_deg"]))

        if not targets:
            return _json_error(
                "validation_error",
                [{"loc": ["targets_longitudes|targets_chart"], "msg": "no targets to scan"}],
                400,
            )

        # -------- engine options --------
        topocentric = bool(body.get("topocentric")) or (
            isinstance(body.get("latitude"), (int, float)) and isinstance(body.get("longitude"), (int, float))
        )
        lat = float(body.get("latitude")) if isinstance(body.get("latitude"), (int, float)) else None
        lon = float(body.get("longitude")) if isinstance(body.get("longitude"), (int, float)) else None
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

        # sidereal options
        zodiac_mode = (body.get("zodiac_mode") or "tropical").lower()
        ayanamsa_deg = float(body.get("ayanamsa_deg", 0.0))

        # aspect set
        aspects = _normalize_aspects(body.get("aspects"), include_minors=include_minors)

        # -------- run engine with shared adapter --------
        try:
            shared_adapter = get_shared_adapter(frame)
            eng = pred.TransitEngine(
                ephem=shared_adapter,
                frame=frame,
                topocentric=topocentric,
                latitude=lat,
                longitude=lon,
                elevation_m=elev,
            )

            # thread sidereal into engine (no API change)
            eng.sidereal_mode = zodiac_mode.startswith("sidereal")
            eng.ayanamsa_deg = ayanamsa_deg

            events = eng.scan_aspects(
                jd_start_tt=float(jd0),
                jd_end_tt=float(jd1),
                movers=[str(m) for m in movers],
                targets=targets,
                aspects=aspects,
                step_minutes=step_arg,  # supports "auto"
                include_antiscia=include_antiscia,
                antiscia_orb_deg=antiscia_orb_deg,
            )
        except Exception as e:
            return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

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

        resp = jsonify(
            {
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
                    "ayanamsa_deg": ayanamsa_deg,
                },
                "targets": targets,
                "movers": movers,
                "results": out,
            }
        )
        resp.status_code = 200
        resp.headers["X-Compute-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.0f}"
        return resp

    finally:
        _give_gate()


# ───────────────────────── /predictive/ingresses ─────────────────────────
@api.post("/api/predictive/ingresses")
@rate_limit(RL_PREDICTIVE)
def predictive_ingresses():
    """Find sign-ingress moments for moving bodies within a time window."""
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    if not _take_gate():
        return _busy()

    t_wall = time.perf_counter()
    try:
        # ── Window (TT) ──────────────────────────────────────────────────────
        try:
            jd0, jd1 = _parse_time_range_like(body)
        except ValidationError as e:
            return _json_error("validation_error", e.errors(), 400)
        except Exception as e:
            return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

        # ── Inputs ──────────────────────────────────────────────────────────
        raw_movers = body.get("movers") or ["Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
        movers = [str(m).strip() for m in raw_movers if str(m).strip()]
        if not movers:
            return _json_error("validation_error", [{"loc": ["movers"], "msg": "movers cannot be empty"}], 400)

        frame = parse_frame(body.get("frame"))
        adapter = get_shared_adapter(frame)

        # Observing options
        obs = {
            "topocentric": bool(body.get("topocentric")),
            "latitude": float(body["latitude"]) if isinstance(body.get("latitude"), (int, float)) else None,
            "longitude": float(body["longitude"]) if isinstance(body.get("longitude"), (int, float)) else None,
            "elevation_m": float(body["elevation_m"]) if isinstance(body.get("elevation_m"), (int, float)) else None,
        }

        # Sidereal/tropical
        zodiac_mode = (body.get("zodiac_mode") or "tropical").lower()
        sidereal = zodiac_mode.startswith("sidereal")
        ayanamsa_deg = float(body.get("ayanamsa_deg", 0.0))

        # Step sizing
        step_arg = _parse_step_minutes(body.get("step_minutes"), default_min=60.0)

        # ── Per-request ephemeris (memoized) ─────────────────────────────────
        E = _PerRequestEphem(adapter, obs, sidereal=sidereal, ay_deg=ayanamsa_deg)

        # ── Helpers ─────────────────────────────────────────────────────────
        def _sign_idx(lon_deg: float) -> int:
            return int(math.floor(_norm360(lon_deg) / 30.0)) % 12

        def _auto_step_minutes(names: List[str]) -> float:
            """
            Heuristic coarse step (minutes). Fine timing is refined with Brent,
            so we can be a bit bold here.
            """
            caps: List[int] = []
            for m in names:
                n = (m or "").lower()
                if n == "moon":
                    caps.append(20)           # faster mover
                elif n in {"mercury", "venus", "mars"}:
                    caps.append(60)
                elif n in {"sun", "jupiter", "saturn"}:
                    caps.append(120)
                else:
                    caps.append(180)
            return float(max(15, min(caps) if caps else 90))

        # Brent–Dekker root on wrapped delta (deg)
        def _refine_zero_brent(f, a, b, fa, fb, *, max_iter=64, tol_days=1e-6) -> float:
            if fa == 0.0:
                return a
            if fb == 0.0:
                return b

            # Ensure bracket via bisection if needed
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
                    a, b = b, a
                    fa, fb = fb, fa
                m = 0.5 * (a + b)
                tol = tol_days
                if abs(b - a) <= tol:
                    return b

                # Inverse quadratic or secant
                if fa != fc and fb != fc:
                    s = (
                        (a * fb * fc) / ((fa - fb) * (fa - fc))
                        + (b * fa * fc) / ((fb - fa) * (fb - fc))
                        + (c * fa * fb) / ((fc - fa) * (fc - fb))
                    )
                else:
                    s = b - fb * (b - a) / (fb - fa)

                # Acceptability checks; else bisection
                cond = not ((3 * a + b) / 4 < s < b if a < b else b < s < (3 * a + b) / 4)
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
                    a, b = b, a
                    fa, fb = fb, fa
            return b

        # ── Core scan ────────────────────────────────────────────────────────
        step_minutes = _auto_step_minutes(movers) if isinstance(step_arg, str) else (
            float(step_arg) if float(step_arg) > 0.0 else _auto_step_minutes(movers)
        )
        dt = float(step_minutes) / (24.0 * 60.0)

        t0 = float(jd0)
        events: List[Dict[str, object]] = []
        dedupe: set[Tuple[str, int, int]] = set()

        l0 = E.map(t0, movers)
        s0 = {k: _sign_idx(v) for k, v in l0.items()}

        while t0 < jd1 - 1e-12:
            t1 = min(t0 + dt, jd1)
            l1 = E.map(t1, movers)

            for body in movers:
                if body not in l0 or body not in l1:
                    continue

                a = float(l0[body])
                b = float(l1[body])
                s_prev = int(s0.get(body, _sign_idx(a)))
                s_next = _sign_idx(b)
                if s_prev == s_next:
                    continue  # no ingress this step

                # Direction via shortest-path delta
                forward = (_wrap180(b - a) > 0.0)

                # Primary edge to test
                boundary = (s_prev + 1) % 12 if forward else s_prev
                edge_deg = 30.0 * boundary

                def f(tt: float) -> float:
                    lm = E.one(tt, body)
                    return 0.0 if lm is None else _wrap180(float(lm) - edge_deg)

                fa = _wrap180(a - edge_deg)
                fb = _wrap180(b - edge_deg)

                # Try to shrink the bracket using a linear estimate
                ta, tb = t0, t1
                den = _wrap180(b - a)
                if den != 0.0:
                    frac = _wrap180(edge_deg - a) / den
                    if 0.0 <= frac <= 1.0:
                        t_est = t0 + frac * (t1 - t0)
                        pad = 0.25 * (t1 - t0)
                        ta, tb = max(t0, t_est - pad), min(t1, t_est + pad)
                        la = E.one(ta, body)
                        lb = E.one(tb, body)
                        if la is not None and lb is not None:
                            fa2 = _wrap180(float(la) - edge_deg)
                            fb2 = _wrap180(float(lb) - edge_deg)
                            if fa2 * fb2 <= 0.0:
                                fa, fb = fa2, fb2  # better bracket

                # If still unbracketed, try adjacent edge (rare wrap cases)
                if fa == 0.0:
                    t_exact = ta
                elif fb == 0.0:
                    t_exact = tb
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

                lon_exact = E.one(t_exact, body)
                if lon_exact is None:
                    continue

                from_sign = s_prev
                to_sign = _sign_idx(lon_exact)

                # Dedupe (per sec & edge)
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
                    "longitude_deg": float(lon_exact),
                })

            # Slide window
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
        resp.headers["X-Compute-Time-ms"] = f"{(time.perf_counter() - t_wall) * 1000:.0f}"
        resp.headers["X-Adapter-Calls"] = str(int(E.calls))
        return resp

    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)
    finally:
        _give_gate()

        
# ───────────────────────── /predictive/station ─────────────────────────
@api.post("/api/predictive/stations")
def predictive_stations():
    try:
        # Try optional engine (rename to your actual module/function if different)
        from app.core.predictive import compute_stations as _compute
    except Exception:
        return _json_error("stations_unavailable", "predictive stations engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        res = _compute(body)  # adapt if your signature differs
        return jsonify({"ok": True, **(res if isinstance(res, dict) else {"result": res})}), 200
    except ValueError as e:
        return _json_error("stations_value_error", str(e), 400)
    except Exception as e:
        return _json_error("stations_internal", {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else "internal_error", 500)

# ───────────────────────── /predictive/Dasha ─────────────────────────

@api.post("/api/predictive/dasha")
def predictive_dasha():
    try:
        from app.core.vedic import compute_dasha as _compute
    except Exception:
        return _json_error("dasha_unavailable", "vedic/dasha engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        res = _compute(body)
        return jsonify({"ok": True, **(res if isinstance(res, dict) else {"result": res})}), 200
    except ValueError as e:
        return _json_error("dasha_value_error", str(e), 400)
    except Exception as e:
        return _json_error("dasha_internal", {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else "internal_error", 500)


# ───────────────────────── /predictive/varga ─────────────────────────

@api.post("/api/predictive/vargas")
def predictive_vargas():
    try:
        from app.core.vedic import compute_vargas as _compute
    except Exception:
        return _json_error("vargas_unavailable", "vedic/vargas engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        res = _compute(body)
        return jsonify({"ok": True, **(res if isinstance(res, dict) else {"result": res})}), 200
    except ValueError as e:
        return _json_error("vargas_value_error", str(e), 400)
    except Exception as e:
        return _json_error("vargas_internal", {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else "internal_error", 500)


# ───────────────────────── /predictive/yoga ─────────────────────────

@api.post("/api/predictive/yogas")
def predictive_yogas():
    try:
        from app.core.vedic import compute_yogas as _compute
    except Exception:
        return _json_error("yogas_unavailable", "vedic/yogas engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        res = _compute(body)
        return jsonify({"ok": True, **(res if isinstance(res, dict) else {"result": res})}), 200
    except ValueError as e:
        return _json_error("yogas_value_error", str(e), 400)
    except Exception as e:
        return _json_error("yogas_internal", {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else "internal_error", 500)


# ───────────────────────── /predictive/evaluate ─────────────────────────
@api.post("/api/evaluate")
def evaluate_transit_prox_sun():
    try:
        from app.core.evaluate import transit_prox_sun as _compute
    except Exception:
        return _json_error("evaluate_unavailable", "evaluate engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        res = _compute(body)
        return jsonify({"ok": True, **(res if isinstance(res, dict) else {"result": res})}), 200
    except ValueError as e:
        return _json_error("evaluate_value_error", str(e), 400)
    except Exception as e:
        return _json_error("evaluate_internal", {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else "internal_error", 500)


# ───────────────────────── /predictive/holdout ─────────────────────────

@api.post("/api/holdout")
def holdout_transit():
    try:
        from app.core.holdout import transit_holdout as _compute
    except Exception:
        return _json_error("holdout_unavailable", "holdout engine not wired", 501)

    try:
        body = request.get_json(force=True) or {}
        res = _compute(body)
        return jsonify({"ok": True, **(res if isinstance(res, dict) else {"result": res})}), 200
    except ValueError as e:
        return _json_error("holdout_value_error", str(e), 400)
    except Exception as e:
        return _json_error("holdout_internal", {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else "internal_error", 500)



# ───────────────────────── PROGRESSIONS ─────────────────────────
@api.post("/api/progressions")
@rate_limit(RL_PROGRESSIONS)
def progressions_route():
    """
    Progressions: secondary / minor / tertiary.

    • Validates payload and normalizes options.
    • Fills missing natal JDs from civil datetime + tz (ERFA-aligned).
    • Filters kwargs to the engine signature to avoid TypeErrors.
    • Returns enriched meta and deterministic epoch scalars:
        - epoch_seconds (jd_tt * 86400)
        - epoch_jd_tt   (jd_tt)
      For back-compat with suites expecting a number in `epoch`, we set
      `epoch` to `epoch_seconds` and provide the original object in `epoch_detail`.
    • Adds mover/target/aspect aliases in `aspects_to_natal` for schema-agnostic loggers.
    """
    from dataclasses import is_dataclass, asdict
    from time import perf_counter
    from flask import request, jsonify, make_response
    from typing import Any, Dict, Optional, Tuple, List
    import inspect

    if compute_progressions is None:
        return _json_error("progressions_unavailable", "progressions engine not wired", 501)

    # ---------- helpers --------------------------------------------------------
    def _timescales_to_dict(ts_obj: Any) -> Optional[Dict[str, Any]]:
        """Best-effort normalize a TimeScales-like object into a plain dict."""
        if isinstance(ts_obj, dict):
            return ts_obj
        # Canonical type (import locally to avoid hard import at module load)
        try:
            from app.core.timescales import TimeScales as _TS  # type: ignore
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
        # Generic dataclass
        try:
            if is_dataclass(ts_obj):
                return asdict(ts_obj)  # type: ignore[arg-type]
        except Exception:
            pass
        # Plain __dict__
        try:
            if hasattr(ts_obj, "__dict__"):
                return dict(ts_obj.__dict__)
        except Exception:
            pass
        return None

    def _strip_or_fix_timescales_shallow(container: Any) -> Any:
        """Normalize/strip 'timescales' in a shallow dict (no recursion)."""
        if not isinstance(container, dict):
            return container
        out = dict(container)
        for key in ("timescales", "ts", "TimeScales"):
            if key in out:
                fixed = _timescales_to_dict(out[key])
                if fixed is None:
                    out.pop(key, None)
                else:
                    out[key] = fixed
        return out

    def _deep_clean(obj: Any) -> Any:
        """Recursively drop/normalize any nested TimeScales-like objects."""
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in ("timescales", "ts", "TimeScales"):
                    tv = _timescales_to_dict(v)
                    if tv is not None:
                        cleaned[k] = tv
                    # else drop unknown TS object
                else:
                    cleaned[k] = _deep_clean(v)
            return cleaned
        if isinstance(obj, list):
            return [_deep_clean(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_deep_clean(v) for v in obj)
        return obj

    def _epoch_scalars(epoch_obj: Any) -> Tuple[Optional[float], Optional[float]]:
        """(epoch_jd_tt, epoch_seconds) if available; else (None, None)."""
        try:
            jd_tt = float((epoch_obj or {}).get("jd_tt"))
            return jd_tt, jd_tt * 86400.0
        except Exception:
            return None, None

    def _augment_aspects(lst: Any) -> Any:
        """Add friendly aliases to aspects_to_natal items."""
        if not isinstance(lst, list):
            return lst
        out: List[Dict[str, Any]] = []
        for a in lst:
            if not isinstance(a, dict):
                out.append(a)
                continue
            b = dict(a)
            b.setdefault("mover_name", b.get("prog"))
            b.setdefault("target_name", b.get("natal"))
            b.setdefault("aspect", b.get("type"))
            if "orb_deg" not in b and isinstance(b.get("orb"), (int, float)):
                b["orb_deg"] = float(b["orb"])
            out.append(b)
        return out

    t0 = perf_counter()

    # ---------- parse & validate body -----------------------------------------
    try:
        body = request.get_json(force=True) or {}
        payload = parse_progressions_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # ---------- sanitize shallow nested dicts ---------------------------------
    natal_clean  = _strip_or_fix_timescales_shallow(payload.get("natal")  or {})
    target_clean = _strip_or_fix_timescales_shallow(payload.get("target") or {})
    place_clean  = _strip_or_fix_timescales_shallow(payload.get("place")  or {})

    # ---------- assemble kwargs (defaults + payload) ---------------------------
    kwargs: Dict[str, Any] = {
        "natal": natal_clean,
        "method": payload.get("method", "secondary"),
        "target": (target_clean or None) if isinstance(payload.get("target"), dict) and target_clean else None,
        "years_after": payload.get("years_after"),
        "jd_tt_natal": payload.get("jd_tt_natal"),
        "jd_ut1_natal": payload.get("jd_ut1_natal"),
        "place": (place_clean or None) if isinstance(payload.get("place"), dict) and place_clean else None,
        "frame": payload.get("frame", "ecliptic-of-date"),
        "house_system": payload.get("house_system", "none"),
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

    # ---------- merge nested flags (payload.flags) -----------------------------
    flags = payload.get("flags")
    if isinstance(flags, dict):
        if "aspects_to_natal" in flags:
            kwargs["aspects_to_natal"] = bool(kwargs.get("aspects_to_natal") or flags.get("aspects_to_natal", False))
        for k in ("parallels", "antiscia", "profile"):
            if k in flags:
                kwargs[k] = bool(kwargs.get(k) or flags.get(k, False))
        if "orbs" in flags and kwargs.get("orbs") is None and isinstance(flags["orbs"], dict):
            kwargs["orbs"] = flags["orbs"]

    # ---------- fill natal timescales if missing -------------------------------
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

    # ---------- deep clean payload to avoid TS objects sneaking in -------------
    cleaned_kwargs = _deep_clean(kwargs)

    # ---------- filter to engine signature & map aliases -----------------------
    try:
        engine_params = set(inspect.signature(compute_progressions).parameters.keys())
        safe_kwargs = {k: v for k, v in cleaned_kwargs.items() if k in engine_params}

        # Map 'zodiac_mode' → 'mode' if engine expects 'mode'
        if "mode" in engine_params and "mode" not in safe_kwargs and "zodiac_mode" in cleaned_kwargs:
            safe_kwargs["mode"] = cleaned_kwargs["zodiac_mode"]

        # prune explicit Nones (except for fields that legitimately use None)
        for k in list(safe_kwargs.keys()):
            if safe_kwargs[k] is None and k not in ("years_after", "target"):
                safe_kwargs.pop(k, None)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("progressions_internal", det or "internal_error", 500)

    # ---------- call engine ----------------------------------------------------
    try:
        result = compute_progressions(**safe_kwargs)  # expected to return a dict
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

    # ---------- enrich meta (ephemeris snapshot) -------------------------------
    meta = dict(result.get("meta") or {})
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # ---------- epoch scalars & aspects aliases --------------------------------
    epoch_obj = result.get("epoch") or {}
    epoch_jd_tt, epoch_seconds = _epoch_scalars(epoch_obj)
    aspects_out = _augment_aspects(result.get("aspects_to_natal") or [])

    # ---------- response --------------------------------------------------------
    body_out = {
        "ok": True,
        "mapping": (meta.get("mapping") if isinstance(meta.get("mapping"), dict) else None),
        "meta": meta,
        "warnings": list(meta.get("warnings") or []),
        "epoch": (epoch_seconds if epoch_seconds is not None else epoch_obj),
        "epoch_detail": (epoch_obj if epoch_seconds is not None else None),
        "epoch_seconds": epoch_seconds,
        "epoch_jd_tt": epoch_jd_tt,
        "positions": result.get("positions"),
        "houses": result.get("houses"),
        "aspects_to_natal": aspects_out,
    }

    resp = make_response(jsonify(body_out), 200)
    try:
        resp.headers["X-Compute-Time-ms"] = str(int((perf_counter() - t0) * 1000))
    except Exception:
        pass
    return resp

# ───────────────────────── RETURNS (clean rewrite) ─────────────────────────
from typing import Any, Dict, List, Optional, Tuple

def _returns_pick_fn(kind: str) -> Optional[Any]:
    """
    Pick a callable from the loaded returns module for the given intent.
    kind: "compute" | "scan"
    """
    if not _returns_available():
        return None

    if kind == "compute":
        names = (
            "compute_return",               # canonical
            "run_return_api", "calculate_return", "compute_returns",
            "solar_return", "lunar_return", # legacy alternates
        )
    elif kind == "scan":
        names = ("scan_returns", "find_returns", "scan", "search_returns")
    else:
        return None

    for n in names:
        fn = getattr(_returns_mod, n, None)
        if callable(fn):
            return fn
    return None


def _parse_return_kind(raw: Any) -> str:
    """Normalize/validate kind → 'solar'|'lunar' (supports common aliases)."""
    s = str(raw or "").strip().lower()
    if not s:
        return "solar"
    valid = {"solar", "lunar", "sun", "sol", "moon", "lun"}
    if s not in valid:
        raise ValidationError([{
            "loc": ["kind"],
            "msg": f"must be 'solar' or 'lunar', got '{s}'",
            "type": "value_error",
        }])
    return {"sun": "solar", "sol": "solar", "moon": "lunar", "lun": "lunar"}.get(s, s)


def _normalize_window(body: Dict[str, Any], default_days: float = 30.0) -> Tuple[Optional[float], Optional[float]]:
    """
    Resolve a [jd_start_tt, jd_end_tt] from body. Falls back to civil window.
    Returns (None, None) if the window cannot be resolved.
    """
    # Direct JDs
    jd0, jd1 = body.get("jd_start_tt"), body.get("jd_end_tt")
    if isinstance(jd0, (int, float)) and isinstance(jd1, (int, float)):
        try:
            a, b = float(jd0), float(jd1)
            if b > a:
                return a, b
        except Exception:
            pass

    # Civil fields
    date0 = body.get("date_start") or body.get("start_date")
    time0 = body.get("time_start") or "00:00:00"
    date1 = body.get("date_end") or body.get("end_date")
    time1 = body.get("time_end") or "23:59:59"
    tz    = body.get("tz") or body.get("place_tz") or body.get("timezone")

    if isinstance(date0, str) and isinstance(time0, str) and isinstance(tz, str):
        try:
            ts0 = _compute_timescales_from_local(date0, time0, tz, payload=body)
            a = float(ts0["jd_tt"])
            if isinstance(date1, str) and isinstance(time1, str):
                ts1 = _compute_timescales_from_local(date1, time1, tz, payload=body)
                b = float(ts1["jd_tt"])
                if b > a:
                    return a, b
            else:
                return a, a + float(default_days)
        except Exception:
            pass

    return None, None


def _build_returns_kwargs(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """
    Construct kwargs for the returns engine with strict, friendly validation.

    Ensures:
      • natal.{date,time,place_tz} present (unless jd_* provided)
      • jd_tt_natal / jd_ut1_natal filled when possible
      • frame/zodiac/house normalized
      • topocentric derived from coordinates if present
      • optional scan window (jd_start_tt / jd_end_tt)
    """
    errs: List[Dict[str, Any]] = []

    # natal (required object)
    natal = body.get("natal") or {}
    if not isinstance(natal, dict):
        errs.append({"loc": ["natal"], "msg": "required object", "type": "value_error"})
        natal = {}

    date = natal.get("date")
    time_s = natal.get("time")
    tz = natal.get("place_tz") or natal.get("tz") or natal.get("timezone")

    # frame (validated via parse_frame)
    try:
        frame = parse_frame(body.get("frame"))
    except ValidationError as e:
        return {}, e.errors()

    # zodiac mode
    zodiac_mode = (body.get("zodiac_mode") or body.get("mode") or "tropical").strip().lower()
    if zodiac_mode not in ("tropical", "sidereal"):
        errs.append({"loc": ["zodiac_mode"], "msg": "must be 'tropical' or 'sidereal'", "type": "value_error"})

    # house system
    house_system = (body.get("house_system") or "placidus").strip().lower()

    # ayanamsa
    ay_f: Optional[float] = None
    try:
        ay_raw = body.get("ayanamsa_deg")
        if isinstance(ay_raw, (int, float)):
            ay_f = float(ay_raw)
        elif ay_raw is not None:
            ay_f = float(str(ay_raw))
    except Exception:
        errs.append({"loc": ["ayanamsa_deg"], "msg": "must be a number", "type": "type_error.float"})

    # kind
    try:
        kind = _parse_return_kind(body.get("kind") or body.get("type") or body.get("planet"))
    except ValidationError as e:
        errs.extend(e.errors())
        kind = "solar"

    # natal JDs (fill from civil when possible)
    jd_tt_natal = body.get("jd_tt_natal")
    jd_ut1_natal = body.get("jd_ut1_natal")

    if not (isinstance(jd_tt_natal, (int, float)) and isinstance(jd_ut1_natal, (int, float))):
        # require civil fields to compute them
        if not (isinstance(date, str) and isinstance(time_s, str) and isinstance(tz, str)):
            errs.append({
                "loc": ["natal.date|time|place_tz"],
                "msg": "required strings (or provide jd_tt_natal & jd_ut1_natal)",
                "type": "value_error",
            })
        else:
            try:
                ts_nat = _compute_timescales_from_local(date, time_s, tz, payload=natal)
                jd_tt_natal  = float(ts_nat["jd_tt"])  if not isinstance(jd_tt_natal, (int, float)) else float(jd_tt_natal)
                jd_ut1_natal = float(ts_nat["jd_ut1"]) if not isinstance(jd_ut1_natal, (int, float)) else float(jd_ut1_natal)
            except ValidationError as e:
                errs.extend(e.errors())
            except Exception as e:
                errs.append({"loc": ["natal"], "msg": f"timescale computation failed: {e}", "type": "value_error"})

    # optional place
    place = body.get("place")
    if isinstance(place, dict):
        try:
            la = float(place["latitude"])
            lo = float(place["longitude"])
            place = {"latitude": la, "longitude": lo, "elev_m": float(place.get("elev_m", 0.0))}
        except Exception:
            place = None
    else:
        place = None

    # topocentric?
    def _has_coords(d: Dict[str, Any]) -> bool:
        return isinstance(d.get("latitude"), (int, float)) and isinstance(d.get("longitude"), (int, float))

    topocentric = bool(body.get("topocentric")) or _has_coords(natal) or _has_coords(place or {})

    # window
    try:
        jd0, jd1 = _normalize_window(body)
    except ValidationError as e:
        errs.extend(e.errors())
        jd0 = jd1 = None

    # numeric knobs
    def _num(key: str, cast, default, positive=False, int_type=False) -> Any:
        try:
            v = cast(body.get(key, default))
            if positive and v <= 0:
                raise ValueError
            return v
        except Exception:
            errs.append({"loc": [key], "msg": f"must be a {'positive ' if positive else ''}{'integer' if int_type else 'number'}",
                         "type": f"type_error.{'int' if int_type else 'float'}"})
            return default

    tol_arcmin             = _num("tol_arcmin", float, 1.0, positive=True)
    max_iters              = _num("max_iters", int,   12,   positive=True, int_type=True)
    fd_step_minutes        = _num("fd_step_minutes", float, 2.0, positive=True)
    validation_residual_am = _num("validation_residual_arcmin", float, 1.0, positive=True)

    # misc toggles
    estimate_uncertainty = bool(body.get("estimate_uncertainty", True))
    profile              = bool(body.get("profile", False))
    validation           = (body.get("validation") or "basic").strip().lower()
    if validation not in ("none", "basic", "extended"):
        errs.append({"loc": ["validation"], "msg": "must be 'none', 'basic', or 'extended'", "type": "value_error"})
        validation = "basic"

    # hints
    guess_years_offset = body.get("guess_years_offset")
    if guess_years_offset is not None:
        try:
            guess_years_offset = int(guess_years_offset)
        except Exception:
            errs.append({"loc": ["guess_years_offset"], "msg": "must be an integer", "type": "type_error.int"})
            guess_years_offset = None

    around_jd_tt = body.get("around_jd_tt")
    if around_jd_tt is not None:
        try:
            around_jd_tt = float(around_jd_tt)
        except Exception:
            errs.append({"loc": ["around_jd_tt"], "msg": "must be a number", "type": "type_error.float"})
            around_jd_tt = None

    # aspects config
    orbs = body.get("orbs") if isinstance(body.get("orbs"), dict) else None
    aspects_to_natal = bool(body.get("aspects_to_natal", True))
    parallels = bool(body.get("parallels", False))
    antiscia  = bool(body.get("antiscia", False))

    # lunar month
    lunar_month = (body.get("lunar_month") or "sidereal").strip().lower()
    if lunar_month not in ("sidereal", "synodic"):
        errs.append({"loc": ["lunar_month"], "msg": "must be 'sidereal' or 'synodic'", "type": "value_error"})
        lunar_month = "sidereal"

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
        "ayanamsa_deg": (ay_f if ay_f is not None else 0.0),
        "lunar_month": lunar_month,
        "guess_years_offset": guess_years_offset,
        "around_jd_tt": around_jd_tt,
        "tol_arcmin": tol_arcmin,
        "max_iters": max_iters,
        "estimate_uncertainty": estimate_uncertainty,
        "fd_step_minutes": fd_step_minutes,
        "profile": profile,
        "validation": validation,
        "validation_residual_arcmin": validation_residual_am,
        "aspects_to_natal": aspects_to_natal,
        "parallels": parallels,
        "antiscia": antiscia,
        "orbs": orbs,
        "jd_start_tt": jd0,
        "jd_end_tt": jd1,
        "topocentric": topocentric,
    }

    # prune explicit None (except allowed window/hints/orbs/place)
    for k in list(kwargs.keys()):
        if kwargs[k] is None and k not in {"jd_start_tt", "jd_end_tt", "guess_years_offset", "around_jd_tt", "orbs", "place"}:
            kwargs.pop(k, None)

    return kwargs, None


def _returns_enrich_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(meta or {})
    try:
        out.update(_snapshot_ephemeris_meta(out))
    except Exception:
        pass
    return out


# ───────────────────────── Routes ─────────────────────────

@api.post("/api/return")
@rate_limit(RL_RETURNS)
def returns_compute_route():
    """Compute a single solar/lunar return."""
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

    fn = _returns_pick_fn("compute")
    if not callable(fn):
        return _json_error("returns_unavailable", "no compute function exported by return(s) module", 501)

    try:
        result = fn(**kwargs)  # type: ignore[misc]
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

    if not isinstance(result, dict):
        return _json_error("returns_internal", "invalid result format from compute function", 500)

    # Engine-level error passthrough
    if not result.get("ok", True):
        det = {"result": result, "error_details": result.get("details", {})} if DEBUG_VERBOSE else None
        return _json_error(result.get("error", "computation_failed"), det, 500)

    meta = _returns_enrich_meta(result.get("meta") or {})
    event = result.get("event") or {}
    if not isinstance(event, dict):
        return _json_error("returns_internal", "invalid event structure", 500)

    # epoch (tolerant)
    epoch = None
    if any(k in event for k in ("jd_tt", "jd_ut1")):
        try:
            epoch = {"jd_tt": float(event.get("jd_tt", 0.0)), "jd_ut1": float(event.get("jd_ut1", 0.0))}
        except Exception:
            epoch = None

    positions = result.get("positions")
    if positions is not None and not isinstance(positions, dict):
        positions = None

    resp = {
        "ok": True,
        "kind": (event.get("kind") or kwargs.get("kind") or "solar"),
        "event": event,
        "epoch": epoch,
        "positions": positions,
        "houses": result.get("houses"),
        "meta": meta,
        "warnings": list(meta.get("warnings") or []),
    }
    return jsonify(resp), 200


@api.post("/api/return/scan")
@rate_limit(RL_RETURNS)
def returns_scan_route():
    """
    Scan a window for return events (e.g., all lunar returns in a month).

    Body:
      natal: { date, time, place_tz, latitude?, longitude? }
      kind|type|planet: "solar" | "lunar"
      jd_start_tt & jd_end_tt (preferred) OR date_start/time_start/date_end/time_end (+ tz)
      frame, zodiac_mode, house_system, place, …
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

    jd0, jd1 = kwargs.get("jd_start_tt"), kwargs.get("jd_end_tt")
    if not (isinstance(jd0, (int, float)) and isinstance(jd1, (int, float)) and jd1 > jd0):
        return _json_error("validation_error", [{"loc": ["jd_start_tt|date_start"], "msg": "window required"}], 400)

    fn = _returns_pick_fn("scan")
    if not callable(fn):
        return _json_error("returns_unavailable", "no scan function exported by return(s) module", 501)

    try:
        results = fn(**kwargs)  # type: ignore[misc]
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

    # Normalize results shape
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

    

# ───────────────────────── PARANS (clean rewrite) ─────────────────────────

@api.post("/api/parans")
@rate_limit(RL_PARANS)
def parans_route():
    from time import perf_counter
    t0 = perf_counter()

    # Engine availability
    if _parans_compute is None:
        det = {"import_error": repr(_PARANS_IMPORT_ERROR)} if DEBUG_VERBOSE and _PARANS_IMPORT_ERROR else None
        return _json_error("parans_unavailable", det or "parans engine not wired", 501)

    # ---------- Parse & validate payload ----------
    try:
        body = request.get_json(force=True) or {}
        payload = parse_parans_payload(body)
    except ValidationError as e:
        errs = e.errors()

        # Map coordinate-style validation errors to a friendlier code
        def _norm_loc(loc) -> list[str]:
            if loc is None:
                return []
            if isinstance(loc, (list, tuple)):
                parts: list[str] = []
                for x in loc:
                    parts.extend(str(x).split("."))
                return [p.strip() for p in parts if str(p).strip()]
            return [p.strip() for p in str(loc).split(".") if p.strip()]

        def _is_coord_error(details: list[dict]) -> bool:
            for d in details or []:
                loc_parts = [s.lower() for s in _norm_loc(d.get("loc"))]
                dot = ".".join(loc_parts)
                msg = (str(d.get("msg", "")) or "").lower()
                typ = (str(d.get("type", "")) or "").lower()
                if (
                    dot in ("place.latitude", "place.longitude")
                    or (len(loc_parts) >= 2 and loc_parts[0] == "place" and loc_parts[1] in {"latitude", "longitude"})
                    or ("latitude" in loc_parts) or ("longitude" in loc_parts)
                    or ("latitude" in msg) or ("longitude" in msg)
                    or (("float" in typ or "number" in msg) and ("place" in loc_parts or dot.startswith("place")))
                ):
                    return True
            return False

        return (
            _json_error("parans_value_error", errs, 400)
            if _is_coord_error(errs)
            else _json_error("validation_error", errs, 400)
        )
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # ---------- Resolve reference timescales if needed ----------
    subject = payload.get("subject") or {}
    place   = payload.get("place") or {}

    jd_tt_ref  = payload.get("jd_tt_ref")
    jd_ut1_ref = payload.get("jd_ut1_ref")

    if jd_tt_ref is None or jd_ut1_ref is None:
        try:
            ts = _compute_timescales_from_local(subject["date"], subject["time"], subject["place_tz"], payload=subject)
            jd_tt_ref  = float(ts["jd_tt"])
            jd_ut1_ref = float(ts["jd_ut1"])
        except ValidationError as e:
            return _json_error("validation_error", e.errors(), 400)
        except Exception as e:
            return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

    # ---------- Build call kwargs (filter to engine signature) ----------
    paran_kwargs = {
        "subject": subject,
        "place": place,
        "jd_tt_ref": jd_tt_ref,
        "jd_ut1_ref": jd_ut1_ref,
        "frame": payload.get("frame", "ecliptic-of-date"),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "bodies": tuple(payload.get("bodies", ("Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"))),
        "tolerance_minutes": float(payload.get("tolerance_minutes", 4.0)),
        "search_window_days": float(payload.get("search_window_days", 1.0)),
        "max_iters": int(payload.get("max_iters", 10)),
        "fd_step_minutes": float(payload.get("fd_step_minutes", 2.0)),
        "earth_model": payload.get("earth_model", "spherical"),
        "apply_refraction": bool(payload.get("apply_refraction", False)),
        "pressure_hPa": float(payload.get("pressure_hPa", 1010.0)),
        "temperature_C": float(payload.get("temperature_C", 10.0)),
        "profile": bool(payload.get("profile", False)),
        "validation": payload.get("validation", "basic"),
    }

    try:
        import inspect
        sig = inspect.signature(_parans_compute)
        paran_kwargs = {k: v for k, v in paran_kwargs.items() if k in sig.parameters}
    except Exception:
        # If reflection fails, just pass the conservative set above.
        pass

    # ---------- Run engine ----------
    try:
        result = _parans_compute(**paran_kwargs)
    except ValueError as e:
        return _json_error("parans_value_error", str(e), 400)
    except (TypeError, RuntimeError) as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("parans_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("parans_internal", det or "internal_error", 500)

    # ---------- Normalize engine result ----------
    if not isinstance(result, dict) or not result.get("ok", False):
        details = result.get("details") if isinstance(result, dict) else None
        etype = result.get("error", "parans_failed") if isinstance(result, dict) else "parans_failed"
        if etype in {"validation_error", "timescales_error", "parans_calculation_failed", "parans_value_error"}:
            return _json_error(etype, details, 400)
        return _json_error("parans_internal", details if DEBUG_VERBOSE else None, 500)

    meta = dict(result.get("meta") or {})
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    resp = {
        "ok": True,
        "meta": meta,
        "events_by_body": result.get("events_by_body", {}),
        "parans": result.get("parans", []),
        "warnings": list(meta.get("warnings", [])),
    }

    ms = (perf_counter() - t0) * 1000.0
    return jsonify(resp), 200, {"X-Compute-Time-ms": f"{ms:.0f}"}


# ───────────────────────── SYNASTRY & COMPOSITE (clean) ─────────────────────────

@api.post("/api/synastry")
@rate_limit(RL_SYNASTRY)
def synastry_route():
    """Compute synastry between two natal charts."""
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
    for field in ("jd_tt_a", "jd_ut1_a", "jd_tt_b", "jd_ut1_b", "place_a", "place_b", "orbs"):
        if field in payload:
            synastry_kwargs[field] = payload[field]

    # Filter to function signature if needed
    try:
        import inspect
        sig = inspect.signature(_synastry_compute)
        params = set(sig.parameters.keys())
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        filtered_kwargs = synastry_kwargs if has_kwargs else {k: v for k, v in synastry_kwargs.items() if k in params}
    except Exception:
        filtered_kwargs = synastry_kwargs

    # Compute
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
    meta = dict(result.get("meta", {}) or {})
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    # Response
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
    """Compute composite chart between two natal charts."""
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

    composite_kwargs = {
        "natal_a": payload["natal_a"],
        "natal_b": payload["natal_b"],
        "method": payload.get("method", "midpoint"),
        "frame": payload.get("frame", "ecliptic-of-date"),
        "house_system": payload.get("house_system", "placidus"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
    }
    for field in ("jd_tt_ref", "jd_ut1_ref", "place_ref"):
        if field in payload:
            composite_kwargs[field] = payload[field]

    try:
        import inspect
        sig = inspect.signature(_composite_compute)
        params = set(sig.parameters.keys())
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        filtered_kwargs = composite_kwargs if has_kwargs else {k: v for k, v in composite_kwargs.items() if k in params}
    except Exception:
        filtered_kwargs = composite_kwargs

    try:
        result = _composite_compute(**filtered_kwargs)
    except ValueError as e:
        return _json_error("composite_value_error", str(e), 400)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("composite_internal", det or "internal_error", 500)

    meta = dict(result.get("meta", {}) or {})
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

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
    """Comprehensive synastry report (synastry + composite + metrics)."""
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
    for field in ("jd_tt_a", "jd_ut1_a", "jd_tt_b", "jd_ut1_b", "place_a", "place_b", "orbs", "composite_place_ref"):
        if field in payload:
            report_kwargs[field] = payload[field]

    try:
        result = _synastry_report_compute(**report_kwargs)
    except ValueError as e:
        return _json_error("synastry_report_value_error", str(e), 400)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("synastry_report_internal", det or "internal_error", 500)

    meta = dict(result.get("meta", {}) or {})
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

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
    Relocate a natal chart to a new place while keeping the natal instant.
    Body:
      natal: { date, time, place_tz, latitude?, longitude?, elev_m? }
      place_new: { latitude, longitude, elev_m? }   # required
      jd_tt_natal?, jd_ut1_natal?                   # optional strict timescales
      frame?, house_system?, zodiac_mode?, ayanamsa_deg?, topocentric_positions?
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

    args = {
        "natal": payload["natal"],
        "place_new": payload["place_new"],
        "frame": payload.get("frame", "ecliptic-of-date"),
        "house_system": payload.get("house_system", "placidus"),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "topocentric_positions": payload.get("topocentric_positions", False),
    }
    for k in ("jd_tt_natal", "jd_ut1_natal"):
        if k in payload:
            args[k] = payload[k]

    try:
        import inspect
        params = set(inspect.signature(_compute_relocated).parameters.keys())
        args = {k: v for k, v in args.items() if k in params}
    except Exception:
        pass

    try:
        result = _compute_relocated(**args)
    except ValueError as e:
        return _json_error("relocation_value_error", str(e), 400)
    except (TypeError, RuntimeError) as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("relocation_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("relocation_internal", det or "internal_error", 500)

    meta = dict(result.get("meta", {}) or {})
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "meta": meta,
        "positions": result.get("positions", {}),
        "houses": result.get("houses"),
        "axes": result.get("axes", {}),
        "warnings": list(meta.get("warnings", [])),
    }), 200


@api.post("/api/astrocartography")
@rate_limit(RL_ASTROCARTOGRAPHY)
def astrocartography_route():
    """
    Compute astrocartography lines (ASC/DC/MC/IC) for selected bodies.
    Body:
      natal: { date, time, place_tz, latitude?, longitude?, elev_m? }
      jd_tt?, jd_ut1? (optional epoch; defaults to natal time)
      frame?, zodiac_mode?, ayanamsa_deg?, bodies?
      lon_step_deg?, lat_clip_deg?, earth_model?, apply_refraction?,
      pressure_hPa?, temperature_C?, default_elev_m?
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

    args = {
        "natal": payload["natal"],
        "frame": payload.get("frame", "ecliptic-of-date"),
        "zodiac_mode": payload.get("zodiac_mode", "tropical"),
        "ayanamsa_deg": payload.get("ayanamsa_deg", 0.0),
        "bodies": tuple(payload.get("bodies", ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"])),
        "lon_step_deg": payload.get("lon_step_deg", 1.0),
        "lat_clip_deg": payload.get("lat_clip_deg", 89.5),
        "earth_model": payload.get("earth_model", "spherical"),
        "apply_refraction": payload.get("apply_refraction", False),
        "pressure_hPa": payload.get("pressure_hPa", 1010.0),
        "temperature_C": payload.get("temperature_C", 10.0),
        "default_elev_m": payload.get("default_elev_m", 0.0),
    }
    for k in ("jd_tt", "jd_ut1"):
        if k in payload:
            args[k] = payload[k]

    try:
        import inspect
        params = set(inspect.signature(_compute_astrocartography).parameters.keys())
        args = {k: v for k, v in args.items() if k in params}
    except Exception:
        pass

    try:
        result = _compute_astrocartography(**args)
    except ValueError as e:
        return _json_error("astrocartography_value_error", str(e), 400)
    except (TypeError, RuntimeError) as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("astrocartography_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("astrocartography_internal", det or "internal_error", 500)

    meta = dict(result.get("meta", {}) or {})
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "meta": meta,
        "lines": result.get("lines", []),
        "warnings": list(meta.get("warnings", [])),
    }), 200

# ───────────────────────── DIRECTIONS ─────────────────────────
@api.post("/api/directions")
@rate_limit(RL_DIRECTIONS)
def directions_route():
    """
    Solar-Arc directions (direct/converse) with optional hit detection.
    Body:
      natal: { date, time, place_tz, latitude?, longitude?, elev_m?, mode? }
      method?: "solar_arc"                     (default)
      rate?: "naibod" | "true_sun"            (default: "naibod")
      target?: { date, time, place_tz }       # alt to years_after
      years_after?: float                      # alt to target
      jd_tt_natal?, jd_ut1_natal?: floats
      place?: { latitude, longitude, elev_m? }
      frame?: "ecliptic-of-date" | "ecliptic-j2000"
      house_system?: string                    (default: "placidus")
      zodiac_mode?: "tropical" | "sidereal"
      ayanamsa_deg?: float
      arcs?: "direct" | "converse" | "both"   (default: "direct")
      orbs?: { conjunction: float, opposition: float, ... }
      include_hits_to?: ["planets","angles","cusps"] (default all)
      parallels?: bool                         (default: false)
      antiscia?: bool                          (default: false)
      profile?: bool                           (default: false)
      validation?: "none" | "basic"            (default: "basic")
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

    args = {
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
    if "target" in payload:
        args["target"] = payload["target"]
    elif "years_after" in payload:
        args["years_after"] = payload["years_after"]
    for k in ("jd_tt_natal", "jd_ut1_natal", "place", "orbs"):
        if k in payload:
            args[k] = payload[k]

    try:
        import inspect
        params = set(inspect.signature(_compute_directions).parameters.keys())
        args = {k: v for k, v in args.items() if k in params}
    except Exception:
        pass

    try:
        result = _compute_directions(**args)
    except ValueError as e:
        return _json_error("directions_value_error", str(e), 400)
    except (TypeError, RuntimeError) as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("directions_internal", det or "internal_error", 500)
    except Exception as e:
        det = {"type": type(e).__name__, "message": str(e)} if DEBUG_VERBOSE else None
        return _json_error("directions_internal", det or "internal_error", 500)

    meta = dict(result.get("meta", {}) or {})
    try:
        meta.update(_snapshot_ephemeris_meta(meta))
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "meta": meta,
        "epoch": result.get("epoch", {}),
        "arc_deg": result.get("arc_deg", {}),
        "positions": result.get("positions", {}),
        "hits": result.get("hits", {}),
        "warnings": list(meta.get("warnings", [])),
    }), 200


# ───────────────────────── Ephemeris utilities ─────────────────────────
def _coerce_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def _truthy(val: Any) -> Optional[bool]:
    if isinstance(val, bool): return val
    if val is None: return None
    s = str(val).strip().lower()
    if s in {"1","true","t","yes","y","on"}: return True
    if s in {"0","false","f","no","n","off"}: return False
    return None

def _norm_rows_from_longitudes(raw: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}

    if isinstance(raw, tuple) and len(raw) == 2:
        raw, meta = raw
    if isinstance(raw, dict):
        meta = dict(raw.get("meta") or {})

    if raw is None:
        return rows, meta

    # mapping: {"Sun": 123.4, ...}
    if isinstance(raw, dict) and all(isinstance(v, (int, float)) for v in raw.values()):
        for k, v in raw.items():
            rows.append({"body": str(k).lower(), "name": str(k), "longitude": float(v)})
        return rows, meta

    data = raw.get("results") if isinstance(raw, dict) and isinstance(raw.get("results"), list) else raw
    if isinstance(data, list):
        for r in data:
            if not isinstance(r, dict):
                continue
            body = (r.get("body") or r.get("name") or r.get("planet") or r.get("id") or r.get("label"))
            L = r.get("longitude") or r.get("lon") or r.get("lambda") or r.get("ecliptic_longitude")
            if body and isinstance(L, (int, float)):
                rec: Dict[str, Any] = {"body": str(body).lower(), "name": str(r.get("name") or body), "longitude": float(L)}
                sp = _coerce_float(r.get("speed"))
                if sp is not None: rec["speed"] = sp
                vel = _coerce_float(r.get("velocity"))
                if vel is not None: rec["velocity"] = vel
                rows.append(rec)

    return rows, meta

def _norm_rows_from_lv(raw: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
            if isinstance(L, (int, float)): rec["longitude"] = float(L)
            if isinstance(V, (int, float)): rec["velocity"] = float(V)

    return list(rows_map.values()), meta


# ───────────────────────── /ephemeris/longitudes ─────────────────────────
@api.post("/api/ephemeris/longitudes")
@rate_limit(RL_EPHEM)
def ephemeris_longitudes_endpoint():
    """
    Proper topocentric handling:
      • If topocentric:true but no coords → 422 topocentric_coords_required
      • If coords present → force topocentric True
      • center = 'topocentric' iff meta.topocentric truthy; else 'geocentric'
    """
    try:
        body = request.get_json(force=True) or {}
        payload = parse_ephemeris_payload(body, require_bodies=True)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e), 400)

    jd_tt = payload["jd_tt"]
    frame = payload.get("frame", "ecliptic-of-date")
    bodies = payload["bodies"]
    names = payload["names"]

    requested_topo = _truthy(body.get("topocentric"))
    if requested_topo is None:
        requested_topo = _truthy(request.args.get("topocentric")) or False

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

    coords_ok = (lat is not None) and (lon is not None)
    observer = {"lat": lat, "lon": lon, "elevation_m": (elev if elev is not None else 0.0)} if coords_ok else None

    if requested_topo and not coords_ok:
        return _json_error(
            "topocentric_coords_required",
            {
                "message": "topocentric:true requires lat & lon (optional elevation_m) via body.observer or ?lat&lon[&elev_m].",
                "aliases": {"lat": ["lat","latitude"], "lon": ["lon","lng","longitude"], "elevation_m": ["elev_m","elevation_m"]},
            },
            422,
        )

    resolved_topo = True if coords_ok else bool(requested_topo)

    # Call adapter
    try:
        from app.core import ephemeris_adapter as ea

        def _call():
            if hasattr(ea, "ecliptic_longitudes"):
                try:
                    return ea.ecliptic_longitudes(
                        jd_tt=jd_tt,
                        bodies=names,
                        frame=frame,
                        topocentric=resolved_topo,
                        observer=observer if resolved_topo else None,
                        latitude=lat if resolved_topo else None,
                        longitude=lon if resolved_topo else None,
                        elevation_m=(elev if elev is not None else None) if resolved_topo else None,
                    )
                except TypeError:
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
            if hasattr(ea, "ecliptic_longitudes_and_velocities"):
                return ea.ecliptic_longitudes_and_velocities(
                    jd_tt=jd_tt, bodies=names, frame=frame, topocentric=resolved_topo, observer=observer if resolved_topo else None
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

    requested = [b.lower() for b in bodies]
    name_map = {b.lower(): n for b, n in zip(bodies, names)}
    by_body = {r.get("body"): r for r in rows if r.get("body")}

    ordered = []
    for key in requested:
        r = by_body.get(key)
        if r and isinstance(r.get("longitude"), (int, float)):
            ordered.append({"body": key, "name": name_map.get(key, r.get("name", key)), "longitude": float(r["longitude"])})

    final_meta = dict(meta or {})
    if "topocentric" not in final_meta:
        final_meta["topocentric"] = resolved_topo
    center = "topocentric" if str(final_meta.get("topocentric")).lower() in {"1","true","t","yes"} else "geocentric"
    final_meta.setdefault("frame", frame)
    final_meta.update(_snapshot_ephemeris_meta(final_meta))

    return jsonify({
        "ok": True,
        "jd_tt": float(jd_tt),
        "frame": frame,
        "center": center,
        "units": {"angles": "deg"},
        "meta": final_meta,
        "results": ordered,
    }), 200

# ───────────────────────── EPHEMERIS INFO (production-safe) ─────────────────────────
@api.get("/api/ephemeris/diagnostics")
@rate_limit(RL_DEBUG)
def ephemeris_diagnostics_route():
    """
    Lightweight adapter info surface for ops/monitoring.
    Does not force kernel loads or perform file introspection.
    """
    try:
        from app.core import ephemeris_adapter as ea

        info = dict(ea.ephemeris_diagnostics() or {})

        # Stable, minimal fields only
        info["kernel"] = info.get("ephemeris_name") or ea.current_kernel_name()
        try:
            info["ephemeris_path"] = ea.current_kernel_path()
        except Exception:
            pass

        # De-dupe any kernel list the adapter may return
        ks = info.get("kernels")
        if isinstance(ks, list):
            info["kernels"] = list(dict.fromkeys(ks))

        # Do NOT attempt lazy coverage reads or other heavyweight diagnostics
        info.pop("ephemeris_coverage_jd_lazy", None)

        return jsonify({"ok": True, **info}), 200

    except Exception as e:
        return _json_error("adapter_error", str(e) if DEBUG_VERBOSE else "adapter_error", 500)


# ───────────────────────── REMOVED DEBUG ROUTES ─────────────────────────
# The following developer-only diagnostic endpoints were intentionally removed
# for production hardening:
#   • /api/debug/engine-test
#   • /api/debug/precision-test
#   • /api/debug/function-signatures
#
# If you need them again locally, keep them behind a strict feature flag and
# never enable them in production builds.

# ───────────────────────── SYSTEM VALIDATION ─────────────────────────
@api.get("/system-validation")
@rate_limit(RL_DEBUG)
def system_validation():
    """
    Lightweight system validation surface:
      • Shows core config values
      • Current timescale sample (UTC/TT/UT1)
      • House engine policy
      • Leap second status (if available)
    """
    # Config load
    cfg = load_config(os.environ.get("ASTRO_CONFIG", "config/defaults.yaml"))

    # Leap second status (optional module)
    leap_status: Optional[Dict[str, Any]] = None
    try:
        from app.core import leapseconds as _leaps
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

    # Timescale sample at "now"
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

    # Policy snapshot
    policy = {
        "houses_engine": _HOUSES_KIND,
        "polar": {
            "soft_fallback_lat_gt": float(POLAR_SOFT_LIMIT_DEG),
            "hard_reject_lat_ge": float(POLAR_HARD_LIMIT_DEG),
            "numeric_fallback": os.getenv("ASTRO_HOUSES_NUMERIC_FALLBACK", "1").lower()
            in ("1", "true", "yes", "on"),
        },
    }

    # Response
    return jsonify({
        "ok": True,
        "astronomy_accuracy": "ERFA-first timescales (JD_TT/JD_UT1), strict where required",
        "performance_slo": {
            "calculate_p95_ms": 800,
            "rect_quick_p95_s": 20,
        },
        "mode_consistency": {
            "sidereal_default": cfg.mode == "sidereal",
            "ayanamsa": getattr(cfg, "ayanamsa", None),
        },
        "policy": policy,
        "leap_seconds": leap_status,
        "version": VERSION,
        "timescale_sample": ts_sample,
    }), 200




