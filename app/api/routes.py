# app/api/routes.py
"""
AstroApp — Canonical API Routes
- Timescales (ERFA-aligned)
- Chart + Houses
- Predictions
- Ephemeris
- Rectification
- Ops: /api/health, /api/config, /api/openapi
No aliases, no dev echo endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request
from zoneinfo import ZoneInfo

from app.version import VERSION
from app.utils.config import load_config
from app.utils.hc import flag_predictions
from app.utils.ratelimit import rate_limit
from app.core.validators import (
    ValidationError,
    parse_chart_payload,
    parse_prediction_payload,
    parse_rectification_payload,
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
    import math
    return math.sin(math.radians(a))


def _cosd(a: float) -> float:
    import math
    return math.cos(math.radians(a))


def _atan2d(y: float, x: float) -> float:
    import math
    if abs(x) < 1e-18 and abs(y) < 1e-18:
        raise ValueError("atan2(0,0) undefined")
    return _wrap360(math.degrees(math.atan2(y, x)))


def _gast_deg(jd_ut1: float, jd_tt: float) -> float:
    try:
        import erfa  # type: ignore
        d1u, d2u = _split_jd(jd_ut1)
        d1t, d2t = _split_jd(jd_tt)
        gst_rad = erfa.gst06a(d1u, d2u, d1t, d2t)
        import math
        return _wrap360(math.degrees(gst_rad))
    except Exception:
        import math
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
        import math
        return math.degrees(eps0 + deps)
    except Exception:
        import math
        T = (float(jd_tt) - 2451545.0) / 36525.0
        eps_arcsec = 84381.448 - 46.8150 * T - 0.00059 * (T**2) + 0.001813 * (T**3)
        return eps_arcsec / 3600.0


def _ramc_deg(jd_ut1: float, jd_tt: float, lon_east_deg: float) -> float:
    return _wrap360(_gast_deg(jd_ut1, jd_tt) + float(lon_east_deg))


def _mc_from_ramc(ramc: float, eps: float) -> float:
    return _atan2d(_sind(ramc) * _cosd(eps), _cosd(ramc))


def _asc_from_phi_ramc(phi: float, ramc: float, eps: float) -> float:
    import math

    def _acotd(x: float) -> float:
        return _wrap360(math.degrees(math.atan2(1.0, x)))

    num = -((math.tan(math.radians(phi)) * _sind(eps)) + (_sind(ramc) * _cosd(eps)))
    den = _cosd(ramc)
    den = den if abs(den) > 1e-15 else math.copysign(1e-15, den if den != 0 else 1.0)
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

    if isinstance(payload, dict):
        if "dut1_seconds" in payload:
            try:
                dut1_seconds = float(payload["dut1_seconds"])
            except Exception:
                raise ValidationError([{"loc": ["dut1_seconds"], "msg": "must be a number (seconds)", "type": "value_error"}])
        elif "dut1" in payload:
            try:
                dut1_seconds = float(payload["dut1"])
            except Exception:
                raise ValidationError([{"loc": ["dut1"], "msg": "must be a number (seconds)", "type": "value_error"}])
        else:
            dut1_seconds = _env_dut1()
    else:
        dut1_seconds = _env_dut1()

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


# ───────────────────────── timescales ─────────────────────────
@api.post("/api/timescales")
def timescales_endpoint():
    body = request.get_json(force=True) or {}
    try:
        date = body.get("date")
        # accept tz aliases
        tz = body.get("tz") or body.get("place_tz") or body.get("timezone")

        # normalize time: allow "HH:MM" by appending ":00"
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
# compute_chart engine discovery (primary + fallback)
_compute_chart = None  # type: ignore
_CHART_ENGINE_NAME: Optional[str] = None
try:  # pragma: no cover
    from app.core.astronomy import compute_chart as _compute_chart  # type: ignore
    _CHART_ENGINE_NAME = "app.core.astronomy.compute_chart"
except Exception as e1:  # pragma: no cover
    try:
        from app.core.chart import compute_chart as _compute_chart  # type: ignore
        _CHART_ENGINE_NAME = "app.core.chart.compute_chart"
        log.warning("Primary astronomy.compute_chart missing; fallback chart.compute_chart in use. err=%r", e1)
    except Exception as e2:
        _compute_chart = None  # type: ignore
        _CHART_ENGINE_NAME = None
        log.error("No compute_chart available: astronomy failed=%r, chart failed=%r", e1, e2)

# house engine (policy façade preferred)
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
        import inspect
        params = inspect.signature(fn).parameters
    return {n: (n in params) for n in names}


def _call_compute_chart(payload: Dict[str, Any], ts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls the chart engine with maximum compatibility:
      • Path A: engine(payload) — inject normalized fields & timescales into payload copy
      • Path B: legacy signatures — only pass kwargs the engine advertises
    Always attaches engine label + JDs to returned chart.
    """
    if _compute_chart is None:
        raise RuntimeError("chart_engine_unavailable")

    import inspect
    param_names = set(inspect.signature(_compute_chart).parameters.keys())

    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in param_names:
                return c
        return None

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

    # Path A
    if "payload" in param_names:
        payload2 = _normalize_payload_for_engine(payload)
        try:
            chart = _compute_chart(payload2)  # type: ignore[arg-type]
        except Exception as e:
            chart = {
                "mode": payload.get("mode"),
                "jd_ut": ts["jd_utc"],
                "jd_tt": ts["jd_tt"],
                "jd_ut1": ts["jd_ut1"],
                "meta": {"engine": _CHART_ENGINE_NAME, "warnings": ["chart_failed"]},
                "error": str(e) if DEBUG_VERBOSE else "chart_failed",
            }
    else:
        # Path B (legacy)
        kwargs: Dict[str, Any] = {}
        if pick("date", "date_str", "date_s"):
            kwargs[pick("date", "date_str", "date_s")] = payload.get("date")
        if pick("time", "time_str", "time_s"):
            kwargs[pick("time", "time_str", "time_s")] = payload.get("time")

        if "latitude" in payload and pick("latitude", "lat"):
            kwargs[pick("latitude", "lat")] = payload["latitude"]
        if "longitude" in payload and pick("longitude", "lon"):
            kwargs[pick("longitude", "lon")] = payload["longitude"]

        tz_name = payload.get("place_tz") or payload.get("timezone")
        if tz_name and pick("place_tz", "timezone", "tz_name"):
            kwargs[pick("place_tz", "timezone", "tz_name")] = tz_name

        if pick("mode", "system") and "mode" in payload:
            kwargs[pick("mode", "system")] = payload["mode"]
        if "ayanamsa" in payload and pick("ayanamsa", "ayanamsha", "aya"):
            kwargs[pick("ayanamsa", "ayanamsha", "aya")] = payload["ayanamsa"]

        if "topocentric" in payload and pick("topocentric", "observer_topocentric"):
            kwargs[pick("topocentric", "observer_topocentric")] = bool(payload["topocentric"])
        if "elevation_m" in payload and pick("elevation_m", "elevation"):
            kwargs[pick("elevation_m", "elevation")] = payload["elevation_m"]

        if "bodies" in payload and pick("bodies", "names", "planets"):
            kwargs[pick("bodies", "names", "planets")] = payload["bodies"]
        if "frame" in payload and pick("frame"):
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
                "jd_ut": ts["jd_utc"],
                "jd_tt": ts["jd_tt"],
                "jd_ut1": ts["jd_ut1"],
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
    lat = float(payload["latitude"])
    lon = float(payload["longitude"])

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

    if acc.get("diagnostics") and "diagnostics" in payload:
        kwargs["diagnostics"] = bool(payload["diagnostics"])
    if acc.get("validation") and "validation" in payload:
        kwargs["validation"] = bool(payload["validation"])

    return _houses_fn(**kwargs)


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
        d_asc = _delta_arcsec(asc_new, float(asc_old))
        if d_asc > ARCSEC_TOL:
            h["asc_deg"] = _wrap360(asc_new); h["asc"] = h["asc_deg"]; changed = True
            warn_list.append(f"asc_corrected_for_parity({d_asc:.2f}arcsec)")
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


# ───────────────────────── endpoints ─────────────────────────
@api.post("/api/calculate")
def calculate():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)
        hs = str(body.get("house_system", "")).strip()
        if hs:
            payload["house_system"] = hs
        for k in ("bodies", "points", "ayanamsa", "topocentric", "elevation_m", "dut1"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name, payload=payload)

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

    try:
        houses = _call_compute_houses(payload, ts)
        houses = _normalize_houses_payload(houses)
    except NotImplementedError as e:
        return _json_error("houses_not_implemented", str(e) if DEBUG_VERBOSE else None, 501)
    except ValueError as e:
        return _json_error("houses_error", str(e), 422)
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
        "houses_engine": _HOUSES_KIND,
    }
    return jsonify({"ok": True, "timescales": ts, "chart": chart, "houses": houses, "meta": meta}), 200


@api.post("/api/report")
def report():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)
        hs = str(body.get("house_system", "")).strip()
        if hs:
            payload["house_system"] = hs
        for k in ("bodies", "points", "ayanamsa", "topocentric", "elevation_m", "dut1"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name, payload=payload)

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

    try:
        houses = _call_compute_houses(payload, ts)
        houses = _normalize_houses_payload(houses)
    except NotImplementedError as e:
        return _json_error("houses_not_implemented", str(e) if DEBUG_VERBOSE else None, 501)
    except ValueError as e:
        return _json_error("houses_error", str(e), 422)
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
        "houses_engine": _HOUSES_KIND,
    }
    return jsonify({"ok": True, "chart": chart, "houses": houses, "narrative": narrative, "meta": meta}), 200


# ───────────────────────── predictions ─────────────────────────
@api.post("/api/predictions")
def predictions_route():
    if predict_engine is None:
        return _json_error("predictions_unavailable", "predict engine not wired", 501)

    body = request.get_json(force=True) or {}
    try:
        payload, horizon = parse_prediction_payload(body)
        hs = str(body.get("house_system", "")).strip()
        if hs:
            payload["house_system"] = hs
        for k in ("bodies", "points", "ayanamsa", "topocentric", "elevation_m", "dut1"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name, payload=payload)

    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        return _json_error("chart_internal", str(e) if DEBUG_VERBOSE else "chart_failed", 500)

    try:
        houses = _call_compute_houses(payload, ts)
        houses = _normalize_houses_payload(houses)
    except NotImplementedError as e:
        return _json_error("houses_not_implemented", str(e) if DEBUG_VERBOSE else None, 501)
    except ValueError as e:
        return _json_error("houses_error", str(e), 422)
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
        "houses_engine": _HOUSES_KIND,
    }
    return jsonify({"ok": True, "predictions": preds, "meta": meta}), 200


# ───────────────────────── ephemeris ─────────────────────────
def _norm_rows_from_longitudes(raw: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if raw is None:
        return rows
    if isinstance(raw, dict) and all(isinstance(v, (int, float)) for v in raw.values()):
        for k, v in raw.items():
            rows.append({"body": str(k).lower(), "name": str(k), "longitude": float(v)})
        return rows
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
                    rows.append({"body": str(body).lower(), "name": str(r.get("name") or body), "longitude": float(L)})
    return rows


def _norm_rows_from_lv(raw: Any) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict) and "longitudes" in raw and "velocities" in raw:
        for k, v in (raw.get("longitudes") or {}).items():
            rows.setdefault(str(k).lower(), {"body": str(k).lower(), "name": str(k)})["longitude"] = float(v)
        for k, v in (raw.get("velocities") or {}).items():
            rows.setdefault(str(k).lower(), {"body": str(k).lower(), "name": str(k)})["velocity"] = float(v)
        return list(rows.values())
    data = raw.get("results") if isinstance(raw, dict) and isinstance(raw.get("results"), list) else raw
    if isinstance(data, list):
        for r in data:
            if not isinstance(r, dict):
                continue
            body = (r.get("body") or r.get("name") or r.get("planet") or r.get("id") or r.get("label"))
            if not body:
                continue
            key = str(body).lower()
            rec = rows.setdefault(key, {"body": key, "name": str(r.get("name") or body)})
            L = r.get("longitude") or r.get("lon") or r.get("lambda") or r.get("ecliptic_longitude")
            V = r.get("velocity") or r.get("vel") or r.get("dlambda_dt") or r.get("deg_per_day")
            if isinstance(L, (int, float)):
                rec["longitude"] = float(L)
            if isinstance(V, (int, float)):
                rec["velocity"] = float(V)
    return list(rows.values())


@api.post("/api/ephemeris/longitudes")
def ephemeris_longitudes_endpoint():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_ephemeris_payload(body, require_bodies=True)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    jd_tt = payload["jd_tt"]
    frame = payload["frame"]
    center = payload["center"]
    lat = payload.get("latitude")
    lon = payload.get("longitude")
    elev_m = payload.get("elev_m", 0.0)
    bodies = payload["bodies"]
    names = payload["names"]
    topocentric = (center == "topocentric")

    try:
        from app.core import ephemeris_adapter as ea
        raw = ea.ecliptic_longitudes(
            jd_tt,
            names=names,
            frame=frame,
            topocentric=topocentric,
            latitude=lat,
            longitude=lon,
            elevation_m=elev_m,
        )
    except Exception as e:
        return _json_error("ephemeris_longitudes_error", str(e) if DEBUG_VERBOSE else None, 500)

    rows = _norm_rows_from_longitudes(raw)
    requested = [b.lower() for b in bodies]
    name_map = {b.lower(): n for b, n in zip(bodies, names)}
    by_body = {r["body"]: r for r in rows if isinstance(r.get("longitude"), (int, float))}
    ordered = [
        {"body": b, "name": name_map[b], "longitude": float(by_body[b]["longitude"])}
        for b in requested
        if b in by_body
    ]

    return jsonify(
        {"ok": True, "jd_tt": float(jd_tt), "frame": frame, "center": center, "units": {"angles": "deg"}, "results": ordered}
    ), 200


@api.post("/api/ephemeris/longitudes_and_velocities")
def ephemeris_lv_endpoint():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_ephemeris_payload(body, require_bodies=True)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    jd_tt = payload["jd_tt"]
    frame = payload["frame"]
    center = payload["center"]
    lat = payload.get("latitude")
    lon = payload.get("longitude")
    elev_m = payload.get("elev_m", 0.0)
    bodies = payload["bodies"]
    names = payload["names"]
    topocentric = (center == "topocentric")

    try:
        from app.core import ephemeris_adapter as ea
        raw = ea.ecliptic_longitudes_and_velocities(
            jd_tt,
            names=names,
            frame=frame,
            topocentric=topocentric,
            latitude=lat,
            longitude=lon,
            elevation_m=elev_m,
        )
    except Exception as e:
        return _json_error("ephemeris_lv_error", str(e) if DEBUG_VERBOSE else None, 500)

    rows = _norm_rows_from_lv(raw)
    requested = [b.lower() for b in bodies]
    name_map = {b.lower(): n for b, n in zip(bodies, names)}
    by_body = {
        r["body"]: r
        for r in rows
        if isinstance(r.get("longitude"), (int, float)) or isinstance(r.get("velocity"), (int, float))
    }
    ordered = [
        {
            "body": b,
            "name": name_map[b],
            "longitude": float(by_body[b]["longitude"]) if "longitude" in by_body[b] else None,
            "velocity": float(by_body[b]["velocity"]) if "velocity" in by_body[b] else None,
        }
        for b in requested
        if b in by_body
    ]

    return jsonify(
        {
            "ok": True,
            "jd_tt": float(jd_tt),
            "frame": frame,
            "center": center,
            "units": {"angles": "deg", "velocities": "deg/day"},
            "results": ordered,
        }
    ), 200


# ───────────────────────── rectification ─────────────────────────
@api.post("/api/rectification/quick")
def rect_quick():
    try:
        body = request.get_json(force=True) or {}
        payload, window_minutes = parse_rectification_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    from app.core.rectify import rectification_candidates
    result = rectification_candidates(payload, window_minutes)
    return jsonify({"ok": True, **result}), 200


# ───────────────────────── system-validation (optional) ─────────────────────────
@api.get("/system-validation")
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
