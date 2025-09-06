# app/api/routes.py
"""
AstroApp — Canonical API Routes
- Timescales (ERFA-aligned)
- Chart + Houses
- Predictions
- Ephemeris
- Predictive toolkit
- Ops: /api/health, /api/config, /api/openapi, /__debug/routes

Notes:
- Topocentric ephemeris honored via either center:"topocentric" or topocentric:true
- Ephemeris responses include adapter meta (with meta.topocentric)
- Adapter/kernel meta is bubbled up to API responses so dev tools can verify DE440s/DE421 quickly.
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
from app.utils.ratelimit import rate_limit
from app.core.validators import (
    ValidationError,
    parse_chart_payload,
    parse_prediction_payload,
    parse_rectification_payload,  # noqa: F401 (kept for completeness)
    parse_ephemeris_payload,
)

# Timescales core
from app.core.timescales import build_timescales, TimeScales

# Optional predictions engine
try:
    from app.core.predict import predict as predict_engine
except Exception:
    predict_engine = None  # type: ignore

log = logging.getLogger(__name__)
api = Blueprint("api", __name__)

DEBUG_VERBOSE = os.getenv("ASTRO_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "on")
ARCSEC_TOL = float(os.getenv("ASTRO_ASC_TOL_ARCSEC", "3.6"))  # 0.001°

# ── per-endpoint rate-limit caps (calls per minute, env-overridable) ───────────
_RL = lambda k, d: int(os.getenv(k, str(d)))
RL_TIMESCALES   = _RL("ASTRO_RL_TIMESCALES_PER_MIN",   60)
RL_CALCULATE    = _RL("ASTRO_RL_CALCULATE_PER_MIN",    24)
RL_REPORT       = _RL("ASTRO_RL_REPORT_PER_MIN",       12)
RL_ASPECTS      = _RL("ASTRO_RL_ASPECTS_PER_MIN",      18)
RL_EPHEM        = _RL("ASTRO_RL_EPHEM_PER_MIN",        30)
RL_PREDICTIONS  = _RL("ASTRO_RL_PREDICTIONS_PER_MIN",   6)
RL_PREDICTIVE   = _RL("ASTRO_RL_PREDICTIVE_PER_MIN",   12)
RL_DEBUG        = _RL("ASTRO_RL_DEBUG_PER_MIN",         6)


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

    # tolerant DUT1 parsing (treat "", None, "null", "undefined" as “not provided”)
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


# ───────────────────────── health / ops ─────────────────────────
@api.get("/api/health")
def health():
    return jsonify({"ok": True, "status": "up", "version": VERSION}), 200


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


# ───────────────────────── chart / houses ─────────────────────────
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

    # DEBUG: Add comprehensive logging
    log.info("=== HOUSE CALC START ===")
    log.info("Function: %s from %s", getattr(_houses_fn, "__name__", "?"), getattr(_houses_fn, "__module__", "?"))
    log.info("Houses kind: %s", _HOUSES_KIND)
    log.info("Payload keys: %s", list(payload.keys()))
    log.info("Timescales: jd_tt=%s, jd_ut1=%s", ts.get("jd_tt"), ts.get("jd_ut1"))

    acc = _sig_accepts_houses()
    log.info("Function signature accepts: %s", acc)

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

    log.info("System: %s -> %s", requested_system_raw, requested_system)

    # Force diagnostic parameters
    kwargs: Dict[str, Any] = {}

    # Coordinate parameters
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

    # System parameter
    if requested_system:
        if acc.get("system"):
            kwargs["system"] = requested_system
        elif acc.get("requested_house_system"):
            kwargs["requested_house_system"] = requested_system
        elif acc.get("house_system"):
            kwargs["house_system"] = requested_system

    # Time parameters
    if acc.get("jd_tt"):
        kwargs["jd_tt"] = ts["jd_tt"]
    if acc.get("jd_ut1"):
        kwargs["jd_ut1"] = ts["jd_ut1"]
    if acc.get("jd_ut") and "jd_tt" not in kwargs and "jd_ut1" not in kwargs:
        kwargs["jd_ut"] = ts["jd_utc"]

    # Force diagnostics and validation
    if acc.get("diagnostics"):
        kwargs["diagnostics"] = True
    if acc.get("validation"):
        kwargs["validation"] = True

    log.info("Calling %s with kwargs: %s", getattr(_houses_fn, "__name__", "?"), list(kwargs.keys()))
    log.info("Kwargs values: %s", kwargs)

    try:
        result = _houses_fn(**kwargs)
        log.info("Result type: %s", type(result))
        log.info("Result keys: %s", list(result.keys()) if hasattr(result, "keys") else "not_dict")

        # Check for advanced engine indicators
        has_solver = hasattr(result, "solver_stats") or (hasattr(result, "get") and result.get("solver_stats"))
        has_budget = hasattr(result, "error_budget") or (hasattr(result, "get") and result.get("error_budget"))
        log.info("Advanced indicators: solver_stats=%s, error_budget=%s", has_solver, has_budget)

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
        # (keep ASC parity correction disabled; advanced engine authoritative)
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
        # allow both elevation_m and elev_m to flow through
        for k in ("bodies", "points", "ayanamsa", "topocentric", "elevation_m", "elev_m", "dut1", "houses"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # decide houses policy up-front
    want_houses = _want_houses(body)
    payload["houses"] = bool(want_houses)  # harmless for engines that ignore it

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

    # ---- meta (now includes ephemeris/adapter snapshot for dev tools) ----
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
    if body.get("aspects", False):  # Only if explicitly requested
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


# ───────────────────────── predictions ─────────────────────────
@api.post("/api/predictions")
@rate_limit(RL_PREDICTIONS)
def predictions_route():
    if predict_engine is None:
        return _json_error("predictions_unavailable", "predict engine not wired", 501)

    body = request.get_json(force=True) or {}
    try:
        payload, horizon = parse_prediction_payload(body)
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
        return _json_error("chart_internal", str(e) if DEBUG_VERBOSE else "chart_failed", 500)

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

    chart_for_predict = _prepare_chart_for_predict(chart)

    th_path = os.environ.get("ASTRO_HC_THRESHOLDS", "config/hc_thresholds.json")
    try:
        with open(th_path, "r", encoding="utf-8") as f:
            hc = json.load(f) or {}
    except Exception:
        hc = {}

    defaults = hc.get("defaults", {}) or {}
    tau = float(defaults.get("tau", 0.88))
    floor = float(defaults.get("floor", 0.60))

    overrides = body.get("hc_overrides") or {}
    if isinstance(overrides, dict):
        if "tau" in overrides:
            tau = float(overrides["tau"])
        if "floor" in overrides:
            floor = float(overrides["floor"])

    env_over = os.environ.get("ASTRO_HC_DEBUG_OVERRIDES")
    if env_over:
        try:
            env_dict = json.loads(env_over)
            if isinstance(env_dict, dict):
                if "tau" in env_dict:
                    tau = float(env_dict["tau"])
                if "floor" in env_dict:
                    floor = float(env_dict["floor"])
        except Exception:
            pass

    try:
        result = predict_engine(chart_for_predict, houses, horizon)
        raw_preds = result.get("predictions") if isinstance(result, dict) else result
        if not isinstance(raw_preds, list):
            raise TypeError("predict() returned unexpected shape")
    except Exception as e:
        return jsonify({"ok": False, "error": "internal_error", "message": str(e), "type": type(e).__name__}), 500

    preds: List[Dict[str, Any]] = []
    for i, pr in enumerate(raw_preds):
        p = float(pr.get("probability", 0.0))
        abstained = p < floor
        hc_flag = (not abstained) and (p >= tau)
        interval = pr.get("interval") or {}
        preds.append(
            {
                "prediction_id": f"pred_{i}",
                "domain": pr.get("domain"),
                "horizon": horizon,
                "interval_start_utc": interval.get("start"),
                "interval_end_utc": interval.get("end"),
                "probability_calibrated": p,
                "hc_flag": hc_flag,
                "abstained": abstained,
                "evidence": pr.get("evidence"),
                "mode": chart.get("mode"),
                "ayanamsa_deg": _extract_ayanamsa_from_chart(chart),
                "notes": "QIA+calibrated placeholder; subject to M3 tuning "
                         + ("abstained" if abstained else "accepted"),
            }
        )

    if not overrides and not os.environ.get("ASTRO_HC_DEBUG_OVERRIDES"):
        try:
            preds = flag_predictions(preds, horizon, th_path)
        except Exception as e:
            if DEBUG_VERBOSE:
                log.warning("flag_predictions failed: %r", e)

    meta = {
        "timescales": ts,
        "timescales_locked": True,
        "chart_engine": _CHART_ENGINE_NAME,
        "houses_engine": _HOUSES_KIND if want_houses else "skipped",
        "houses_requested": bool(want_houses),
    }
    meta.update(_snapshot_ephemeris_meta(chart.get("meta")))
    resp = {"ok": True, "predictions": preds, "meta": meta}
    return jsonify(resp), 200


# ───────────────────────── aspects ─────────────────────────
@api.post("/api/aspects")
@rate_limit(RL_ASPECTS)
def aspects():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)

        # Allow additional aspects-specific parameters
        for k in ("orbs", "aspects", "bodies", "points", "mode", "houses"):
            if k in body:
                payload[k] = body[k]

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
        "meta": meta
    }), 200


# ───────────────────────── predictive (transits • validation • dasha • varga • yogas) ─────────────────────────
@api.post("/api/predictive/transits")
@rate_limit(RL_PREDICTIVE)
def predictive_transits():
    """
    Scan transits (moving bodies vs target longitudes) with bisection refinement.
    Accepts either a concrete targets map or a "targets_chart" (natal chart params).
    """
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    # -------- time window (jd_tt) --------
    try:
        jd0 = body.get("jd_start_tt")
        jd1 = body.get("jd_end_tt")
        if not (isinstance(jd0, (int, float)) and isinstance(jd1, (int, float))):
            # derive from civil times
            date0 = body.get("date_start") or body.get("date")
            time0 = body.get("time_start") or body.get("time") or "00:00:00"
            date1 = body.get("date_end")
            time1 = body.get("time_end") or "23:59:59"
            tz = body.get("place_tz") or body.get("timezone") or "UTC"
            if not (isinstance(date0, str) and isinstance(time0, str)):
                return _json_error("validation_error", [{"loc": ["date_start/time_start"], "msg": "required"}], 400)
            ts0 = _compute_timescales_from_local(date0, time0, tz, payload=body)
            if not isinstance(date1, str):
                # default: 24h window from start
                jd0 = float(ts0["jd_tt"]); jd1 = jd0 + 1.0
            else:
                ts1 = _compute_timescales_from_local(date1, time1, tz, payload=body)
                jd0 = float(ts0["jd_tt"]); jd1 = float(ts1["jd_tt"])
        if not (math.isfinite(jd0) and math.isfinite(jd1)) or jd1 <= jd0:
            return _json_error("validation_error", [{"loc": ["jd_start_tt", "jd_end_tt"], "msg": "invalid range"}], 400)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

    # -------- movers / targets --------
    movers = body.get("movers") or ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
    if not (isinstance(movers, list) and all(isinstance(x, str) and x for x in movers)):
        return _json_error("validation_error", [{"loc": ["movers"], "msg": "must be a list of names"}], 400)

    # targets_longitudes: direct map (preferred, fastest path)
    targets: Dict[str, float] = {}
    raw_targets = body.get("targets_longitudes")
    if isinstance(raw_targets, dict):
        for k, v in raw_targets.items():
            try:
                targets[str(k)] = _wrap360(float(v))
            except Exception:
                pass

    # Or: build targets from a chart payload (e.g., natal)
    if not targets and isinstance(body.get("targets_chart"), dict):
        targ = dict(body["targets_chart"])
        tz = targ.get("place_tz") or targ.get("timezone") or "UTC"
        try:
            ts_nat = _compute_timescales_from_local(targ["date"], targ["time"], tz, payload=targ)
        except Exception as e:
            return _json_error("validation_error", [{"loc": ["targets_chart"], "msg": str(e)}], 400)
        try:
            ch = _call_compute_chart(targ, ts_nat)
        except Exception as e:
            return _json_error("chart_internal", str(e) if DEBUG_VERBOSE else "chart_failed", 500)

        # Extract longitudes from chart
        bodies = ch.get("bodies", []) or []
        points = ch.get("points", []) or []
        for row in bodies + points:
            if isinstance(row, dict) and "name" in row and isinstance(row.get("longitude_deg"), (int, float)):
                targets[str(row["name"])] = _wrap360(float(row["longitude_deg"]))

    if not targets:
        return _json_error("validation_error", [{"loc": ["targets_longitudes|targets_chart"], "msg": "no targets to scan"}], 400)

    # -------- engine options --------
    # topocentric honored if lat/lon provided or topocentric:true explicitly
    topocentric = bool(body.get("topocentric")) or (isinstance(body.get("latitude"), (int, float)) and isinstance(body.get("longitude"), (int, float)))
    lat = float(body.get("latitude")) if isinstance(body.get("latitude"), (int, float)) else None
    lon = float(body.get("longitude")) if isinstance(body.get("longitude"), (int, float)) else None
    elev = float(body.get("elevation_m")) if isinstance(body.get("elevation_m"), (int, float)) else None
    frame = body.get("frame") or "ecliptic-of-date"
    step_min = float(body.get("step_minutes") or 30.0)

    include_antiscia = bool(body.get("include_antiscia", False))
    antiscia_orb_deg = float(body.get("antiscia_orb_deg", 2.0))

    # aspect set: majors by default; allow minor switch
    from app.core import predictive as pred  # local import to avoid circulars at module import
    aspects = list(pred.MAJOR_ASPECTS)
    if body.get("include_minors"):
        aspects += list(pred.MINOR_ASPECTS)

    # -------- run engine --------
    try:
        eng = pred.TransitEngine(
            frame=frame,
            topocentric=topocentric,
            latitude=lat, longitude=lon, elevation_m=elev
        )
        events = eng.scan_aspects(
            jd_start_tt=float(jd0),
            jd_end_tt=float(jd1),
            movers=[str(m) for m in movers],
            targets=targets,
            aspects=aspects,
            step_minutes=step_min,
            include_antiscia=include_antiscia,
            antiscia_orb_deg=antiscia_orb_deg,
        )
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

    # shape response
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
    return jsonify({
        "ok": True,
        "window": {"jd_start_tt": float(jd0), "jd_end_tt": float(jd1), "step_minutes": step_min},
        "engine": {"frame": frame, "topocentric": topocentric},
        "targets": targets,
        "movers": movers,
        "results": out
    }), 200


@api.post("/api/predictive/ingresses")
@rate_limit(RL_PREDICTIVE)
def predictive_ingresses():
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    try:
        # time window
        jd0 = body.get("jd_start_tt"); jd1 = body.get("jd_end_tt")
        if not (isinstance(jd0, (int, float)) and isinstance(jd1, (int, float))):
            date0 = body.get("date_start") or body.get("date")
            time0 = body.get("time_start") or body.get("time") or "00:00:00"
            date1 = body.get("date_end"); time1 = body.get("time_end") or "23:59:59"
            tz = body.get("place_tz") or body.get("timezone") or "UTC"
            ts0 = _compute_timescales_from_local(date0, time0, tz, payload=body)
            jd0 = float(ts0["jd_tt"])
            if isinstance(date1, str):
                jd1 = float(_compute_timescales_from_local(date1, time1, tz, payload=body)["jd_tt"])
            else:
                jd1 = jd0 + 1.0
        movers = body.get("movers") or ["Sun","Mercury","Venus","Mars","Jupiter","Saturn"]
        from app.core import predictive as pred
        eng = pred.TransitEngine(
            frame=body.get("frame") or "ecliptic-of-date",
            topocentric=bool(body.get("topocentric")),
            latitude=float(body.get("latitude")) if isinstance(body.get("latitude"), (int,float)) else None,
            longitude=float(body.get("longitude")) if isinstance(body.get("longitude"), (int,float)) else None,
            elevation_m=float(body.get("elevation_m")) if isinstance(body.get("elevation_m"), (int,float)) else None,
        )
        res = eng.find_ingresses(
            jd_start_tt=float(jd0), jd_end_tt=float(jd1),
            movers=[str(m) for m in movers],
            step_minutes=float(body.get("step_minutes") or 60.0)
        )
        return jsonify({"ok": True, "window": {"jd_start_tt": jd0, "jd_end_tt": jd1}, "movers": movers, "results": res}), 200
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/predictive/stations")
@rate_limit(RL_PREDICTIVE)
def predictive_stations():
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)
    try:
        jd0 = body.get("jd_start_tt"); jd1 = body.get("jd_end_tt")
        if not (isinstance(jd0, (int, float)) and isinstance(jd1, (int, float))):
            date0 = body.get("date_start") or body.get("date")
            time0 = body.get("time_start") or body.get("time") or "00:00:00"
            date1 = body.get("date_end"); time1 = body.get("time_end") or "23:59:59"
            tz = body.get("place_tz") or body.get("timezone") or "UTC"
            ts0 = _compute_timescales_from_local(date0, time0, tz, payload=body)
            jd0 = float(ts0["jd_tt"])
            if isinstance(date1, str):
                jd1 = float(_compute_timescales_from_local(date1, time1, tz, payload=body)["jd_tt"])
            else:
                jd1 = jd0 + 30.0/1440.0 * 60.0  # default ~30 hours
        movers = body.get("movers") or ["Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"]
        from app.core import predictive as pred
        eng = pred.TransitEngine(
            frame=body.get("frame") or "ecliptic-of-date",
            topocentric=bool(body.get("topocentric")),
            latitude=float(body.get("latitude")) if isinstance(body.get("latitude"), (int,float)) else None,
            longitude=float(body.get("longitude")) if isinstance(body.get("longitude"), (int,float)) else None,
            elevation_m=float(body.get("elevation_m")) if isinstance(body.get("elevation_m"), (int,float)) else None,
        )
        res = eng.find_stations(
            jd_start_tt=float(jd0), jd_end_tt=float(jd1),
            movers=[str(m) for m in movers],
            step_minutes=float(body.get("step_minutes") or 60.0)
        )
        return jsonify({"ok": True, "window": {"jd_start_tt": jd0, "jd_end_tt": jd1}, "movers": movers, "results": res}), 200
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/predictive/evaluate")
@rate_limit(RL_PREDICTIVE)
def predictive_evaluate():
    """
    Univariate permutation test with optional time-series aware permutations.
    Body:
      records: [{ jd_tt, outcome(0/1), natal_longitudes:{}, natal_cusps:[], birth_jd_tt, ayanamsa_deg }]
      feature:
        - "transit_proximity"   (+ movers[], orb_deg?)
        - "dasha_l1"|"dasha_l2"|"dasha_l3"
        - "yoga_flags"          (+ names[])
      perm_mode: "iid"|"within"|"circular"
      group_by: "subject_id" (optional)
      n_perm: 2000 (default)
    """
    try:
        body = request.get_json(force=True) or {}
        recs = body.get("records") or []
        if not (isinstance(recs, list) and recs):
            return _json_error("validation_error", [{"loc":["records"],"msg":"non-empty list required"}], 400)
        feature = (body.get("feature") or "transit_proximity").lower()
        perm_mode = (body.get("perm_mode") or "iid").lower()
        group_by = body.get("group_by")
        alpha = float(body.get("alpha", 0.05))
        n_perm = int(body.get("n_perm", 2000))
        from app.core import predictive as pred
        # build feature fn
        if feature == "transit_proximity":
            movers = body.get("movers") or ["Sun","Moon","Mercury","Venus","Mars"]
            orb_deg = float(body.get("orb_deg", 1.0))
            ff = pred.feature_transit_proximity(movers=movers, orb_deg=orb_deg)
        elif feature.startswith("dasha_l"):
            level = int(feature.split("_l")[1])
            ff = pred.feature_dasha_lords_onehot(level=level)
        elif feature == "yoga_flags":
            names = body.get("yoga_names") or ["panch_mahapurusha","gajakesari","chandra_mangal","parivartana"]
            ff = pred.feature_yoga_flags(names)
        else:
            return _json_error("validation_error", [{"loc":["feature"],"msg":"unknown"}], 400)
        res = pred.evaluate_univariate(
            recs, ff, n_perm=n_perm, alpha=alpha,
            perm_mode=perm_mode, group_by=group_by, use_time=bool(body.get("use_time", True)),
            seed=body.get("seed")
        )
        return jsonify({"ok": True, "results": [r.__dict__ for r in res]}), 200
    except Exception as e:
        return _json_error("predictive_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)


@api.post("/api/predictive/holdout")
@rate_limit(RL_PREDICTIVE)
def predictive_holdout():
    """
    Train/holdout replication with the same feature menu as /evaluate.
    """
    try:
        body = request.get_json(force=True) or {}
        recs = body.get("records") or []
        if not (isinstance(recs, list) and recs):
            return _json_error("validation_error", [{"loc":["records"],"msg":"non-empty list required"}], 400)
        feature = (body.get("feature") or "transit_proximity").lower()
        perm_mode = (body.get("perm_mode") or "iid").lower()
        group_by = body.get("group_by")
        from app.core import predictive as pred
        if feature == "transit_proximity":
            movers = body.get("movers") or ["Sun","Moon","Mercury","Venus","Mars"]
            orb_deg = float(body.get("orb_deg", 1.0))
            ff = pred.feature_transit_proximity(movers=movers, orb_deg=orb_deg)
        elif feature.startswith("dasha_l"):
            level = int(feature.split("_l")[1])
            ff = pred.feature_dasha_lords_onehot(level=level)
        elif feature == "yoga_flags":
            names = body.get("yoga_names") or ["panch_mahapurusha","gajakesari","chandra_mangal","parivartana"]
            ff = pred.feature_yoga_flags(names)
        else:
            return _json_error("validation_error", [{"loc":["feature"],"msg":"unknown"}], 400)
        res = pred.holdout_replicate(
            recs, ff,
            train_frac=float(body.get("train_frac", 0.7)),
            alpha=float(body.get("alpha", 0.05)),
            n_perm_train=int(body.get("n_perm_train", 2000)),
            n_perm_test=int(body.get("n_perm_test", 4000)),
            perm_mode=perm_mode, group_by=group_by,
            use_time=bool(body.get("use_time", True)),
            seed=body.get("seed")
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
