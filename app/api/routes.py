# app/api/routes.py
"""
Ops & Diagnostics routes (mounted with NO prefix by main.py)

Wires common core modules used by both Western & Vedic:
- app/core/timescales.py
- app/core/time_kernel.py
- app/core/ephem_singleton.py
- app/core/ephemeris_adapter.py
- app/core/astronomy.py
- app/core/house.py
- app/core/houses.py
- app/core/houses_advanced.py

Also surfaces validator presence:
- app/core/validator.py
- app/core/western_validator.py
- app/core/vedic_validator.py

NEW:
- POST /ops/common/calc  → unified lightweight dispatcher for common ops
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple, Callable
import os
import inspect

from flask import Blueprint, jsonify, request

ops_api = Blueprint("ops_api", __name__)

# ───────────────────────── import helpers ─────────────────────────
def _try_import(modpath: str, attr: Optional[str] = None):
    mod, obj, err = None, None, None
    try:
        mod = __import__(modpath, fromlist=["*"])
        if attr:
            obj = getattr(mod, attr)
    except Exception as e:
        err = repr(e)
    return mod, obj, err

def _sig(obj) -> Optional[str]:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None

# ───────────────────────── core modules ─────────────────────────
_ts_mod, _build_timescales, _TS_ERR     = _try_import("app.core.timescales", "build_timescales")
_, _TimeScales, _TS_CLS_ERR             = _try_import("app.core.timescales", "TimeScales")
_tk_mod, _, _TK_ERR                     = _try_import("app.core.time_kernel")
_es_mod, _, _ES_ERR                     = _try_import("app.core.ephem_singleton")
_ea_mod, _, _EA_ERR                     = _try_import("app.core.ephemeris_adapter")
_ast_mod, _, _AST_ERR                   = _try_import("app.core.astronomy")
_h_mod, _, _H_ERR                       = _try_import("app.core.house")
_hs_mod, _, _HS_ERR                     = _try_import("app.core.houses")
_hsa_mod, _, _HSA_ERR                   = _try_import("app.core.houses_advanced")

# Prefer the time_kernel forwarders for public/common endpoints
try:
    from app.core.time_kernel import (
        build_timescales as _tk_build_timescales,
        julian_day_utc as _tk_jd_utc,
        jd_tt_from_utc_jd as _tk_tt_from_utc_jd,
        jd_ut1_from_utc_jd as _tk_ut1_from_utc_jd,
        TIMEKERNEL_VERSION as _TK_VERSION,
    )
    _TK_FUN_ERR = None
except Exception as e:
    _tk_build_timescales = _tk_jd_utc = _tk_tt_from_utc_jd = _tk_ut1_from_utc_jd = None  # type: ignore
    _TK_VERSION = "unknown"
    _TK_FUN_ERR = repr(e)

# ───────────────────────── validators ─────────────────────────
_val_mod, _, _VAL_ERR                   = _try_import("app.core.validator")
_wval_mod, _, _WVAL_ERR                 = _try_import("app.core.western_validator")
_vval_mod, _normalize_vim_payload, _VVAL_ERR = _try_import("app.core.vedic_validator", "normalize_vim_payload")

# Best-effort: locate a common timescales parser in validator.py (optional)
_TIMESCALES_VALIDATOR: Optional[Callable[..., Any]] = None
if _val_mod is not None:
    for name in (
        "parse_timescales_request",
        "parse_timescales_payload",
        "normalize_timescales_payload",
        "parse_timescales",  # lenient candidates
    ):
        fn = getattr(_val_mod, name, None)
        if callable(fn):
            _TIMESCALES_VALIDATOR = fn
            break

# ───────────────────────── ops basics ─────────────────────────
@ops_api.get("/ops/health")
def ops_health():
    return jsonify(ok=True, service="astro-backend", scope="ops", status="ok"), 200

@ops_api.get("/api/health")
def api_health_backcompat():
    # /api/health is deprecated; deprecation headers are added by main.py after_request hook.
    return jsonify(ok=True, service="astro-backend", scope="api", status="ok"), 200

@ops_api.get("/ops/version")
def ops_version():
    try:
        from app.version import VERSION  # type: ignore
    except Exception:
        VERSION = os.environ.get("APP_VERSION", "0.0.0")
    git_sha = os.environ.get("GIT_SHA") or os.environ.get("COMMIT_SHA")
    return jsonify(ok=True, service="astro-backend", version=str(VERSION), git_sha=git_sha), 200

@ops_api.get("/ops/config")
def ops_config():
    cfg: Dict[str, Any] = {
        "enable_vedic_api": os.getenv("ENABLE_VEDIC_API", "1"),
        "cors_allow_origin": os.getenv("CORS_ALLOW_ORIGIN", "*"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "env": os.getenv("ENVIRONMENT") or os.getenv("ENV") or "unknown",
    }
    return jsonify(ok=True, config=cfg), 200

# ───────────────────────── diagnostics ─────────────────────────
@ops_api.get("/ops/diag/cores")
def ops_diag_cores():
    """Report presence and key signatures of shared core modules."""
    return jsonify({
        "ok": True,
        "timescales": {
            "loaded": _ts_mod is not None,
            "error": _TS_ERR,
            "build_timescales_sig": _sig(_build_timescales) if _build_timescales else None,
            "TimeScales_loaded": _TimeScales is not None,
            "TimeScales_error": _TS_CLS_ERR,
        },
        "time_kernel": {
            "loaded": _tk_mod is not None,
            "error": _TK_ERR,
            "version": _TK_VERSION,
            "forwarders_ok": _TK_FUN_ERR is None,
        },
        "ephem_singleton": {"loaded": _es_mod is not None, "error": _ES_ERR},
        "ephemeris_adapter": {"loaded": _ea_mod is not None, "error": _EA_ERR},
        "astronomy": {"loaded": _ast_mod is not None, "error": _AST_ERR},
        "house": {"loaded": _h_mod is not None, "error": _H_ERR},
        "houses": {"loaded": _hs_mod is not None, "error": _HS_ERR},
        "houses_advanced": {"loaded": _hsa_mod is not None, "error": _HSA_ERR},
    }), 200

@ops_api.get("/ops/diag/validators")
def ops_diag_validators():
    """Report presence of validator modules and important callables."""
    def list_callables(mod, limit=12) -> List[str]:
        try:
            names = [n for n in dir(mod) if callable(getattr(mod, n, None)) and not n.startswith("_")]
            names.sort()
            return names[:limit]
        except Exception:
            return []
    return jsonify({
        "ok": True,
        "validator": {
            "loaded": _val_mod is not None,
            "error": _VAL_ERR,
            "functions": list_callables(_val_mod) if _val_mod else None,
            "timescales_parser": getattr(_TIMESCALES_VALIDATOR, "__name__", None),
        },
        "western_validator": {
            "loaded": _wval_mod is not None,
            "error": _WVAL_ERR,
            "functions": list_callables(_wval_mod) if _wval_mod else None,
        },
        "vedic_validator": {
            "loaded": _vval_mod is not None,
            "error": _VVAL_ERR,
            "normalize_vim_payload_sig": _sig(_normalize_vim_payload) if _normalize_vim_payload else None,
            "functions": list_callables(_vval_mod) if _vval_mod else None,
        },
    }), 200

# ───────────────────────── timescales probe (kept) ─────────────────────────
@ops_api.post("/ops/timescales")
def ops_timescales():
    """
    POST a light preview for build_timescales (time_kernel forwarder).
    Body: { "date": "YYYY-MM-DD", "time": "HH:MM:SS[.frac]", "tz": "Area/City", "dut1_seconds": 0.0 }
    Returns a sanitized view (no jd_utc by default) so callers don’t rely on it downstream.
    """
    if _tk_build_timescales is None:
        return jsonify(ok=False, error="timescales_unavailable", detail=_TK_FUN_ERR or _TK_ERR), 503

    body = request.get_json(silent=True) or {}
    date = body.get("date"); time_str = body.get("time"); tz = body.get("tz")
    dut1 = body.get("dut1_seconds", os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", 0.0)))
    include_jd_utc = bool(body.get("include_jd_utc", False))

    # Optional validator
    if _TIMESCALES_VALIDATOR:
        try:
            parsed = _TIMESCALES_VALIDATOR(body)
            if isinstance(parsed, tuple) and parsed:
                body = parsed[0] or body
            elif isinstance(parsed, dict):
                body = parsed
        except Exception:
            # best-effort only; fall back to simple checks
            pass

    missing = [k for k in ("date", "time", "tz") if not body.get(k)]
    if missing:
        return jsonify(ok=False, error="bad_request", detail=f"Missing keys: {', '.join(missing)}"), 400

    try:
        dut1 = float(dut1)
    except Exception:
        return jsonify(ok=False, error="bad_request", detail="dut1_seconds must be a float"), 400

    try:
        ts = _tk_build_timescales(str(date), str(time_str), str(tz), float(dut1))
    except Exception as e:
        return jsonify(ok=False, error="timescales_failed", detail=str(e)), 400

    ts_dict = dict(ts) if isinstance(ts, dict) else getattr(ts, "to_dict", lambda: {} )()

    # Sanitize output to avoid downstream reliance on jd_utc unless requested
    out = {
        "jd_tt": ts_dict.get("jd_tt"),
        "jd_ut1": ts_dict.get("jd_ut1"),
        "delta_t": ts_dict.get("delta_t"),
        "dut1": ts_dict.get("dut1"),
        "dat": ts_dict.get("dat"),
        "tz_offset_seconds": ts_dict.get("tz_offset_seconds"),
        "timezone": ts_dict.get("timezone"),
        "warnings": ts_dict.get("warnings", []),
        "precision": ts_dict.get("precision"),
    }
    if include_jd_utc:
        out["jd_utc"] = ts_dict.get("jd_utc")

    return jsonify(ok=True, input={"date": date, "time": time_str, "tz": tz, "dut1_seconds": float(dut1)}, timescales=out), 200

# ───────────────────────── NEW: common calc endpoint ─────────────────────────
@ops_api.post("/ops/common/calc")
def ops_common_calc():
    """
    Unified lightweight dispatcher for common operations that both Western & Vedic consume.

    Body:
    {
      "op": "timescales" | "jd_utc" | "tt_from_utc_jd" | "ut1_from_utc_jd" | "help",
      "params": { ... },
      "include_jd_utc": false   // only used for "timescales"
    }

    Examples:
      {"op":"timescales","params":{"date":"1989-01-26","time":"20:44:00","tz":"Asia/Kolkata","dut1_seconds":0.0}}
      {"op":"jd_utc","params":{"date":"1989-01-26","time":"20:44:00","tz":"Asia/Kolkata"}}
      {"op":"tt_from_utc_jd","params":{"jd_utc":2447553.13}}
      {"op":"ut1_from_utc_jd","params":{"jd_utc":2447553.13,"dut1_seconds":0.0}}
    """
    body = request.get_json(silent=True) or {}
    op = str(body.get("op") or "").strip().lower()
    params: Dict[str, Any] = body.get("params") or {}
    include_jd_utc = bool(body.get("include_jd_utc", False))

    # Advertise capabilities
    if op in ("help", "list", ""):
        return jsonify(ok=True, ops=[
            {"op": "timescales", "requires": ["date", "time", "tz"], "optional": ["dut1_seconds"], "notes": "Uses time_kernel.build_timescales; jd_utc omitted by default."},
            {"op": "jd_utc", "requires": ["date", "time", "tz"], "notes": "Deprecated helper via time_kernel.julian_day_utc"},
            {"op": "tt_from_utc_jd", "requires": ["jd_utc"], "notes": "Deprecated helper via time_kernel.jd_tt_from_utc_jd"},
            {"op": "ut1_from_utc_jd", "requires": ["jd_utc", "dut1_seconds"], "notes": "Deprecated helper via time_kernel.jd_ut1_from_utc_jd"},
        ], modules={
            "time_kernel_loaded": _tk_mod is not None,
            "time_kernel_forwarders_ok": _TK_FUN_ERR is None,
            "timescales_engine_loaded": _ts_mod is not None,
            "validator_present": _val_mod is not None,
            "timescales_parser": getattr(_TIMESCALES_VALIDATOR, "__name__", None),
        }), 200

    if _tk_mod is None or _TK_FUN_ERR is not None:
        return jsonify(ok=False, error="time_kernel_unavailable", detail=_TK_FUN_ERR or _TK_ERR), 503

    # Optional validator for timescales-like shapes
    if op == "timescales" and _TIMESCALES_VALIDATOR:
        try:
            parsed = _TIMESCALES_VALIDATOR(params)
            if isinstance(parsed, tuple) and parsed:
                params = parsed[0] or params
            elif isinstance(parsed, dict):
                params = parsed
        except Exception:
            pass  # best-effort only

    # Dispatch
    try:
        if op == "timescales":
            date = params.get("date"); time_str = params.get("time"); tz = params.get("tz")
            dut1 = params.get("dut1_seconds", os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", 0.0)))
            missing = [k for k in ("date", "time", "tz") if not params.get(k)]
            if missing:
                return jsonify(ok=False, error="bad_request", detail=f"Missing keys: {', '.join(missing)}"), 400
            dut1 = float(dut1)
            ts = _tk_build_timescales(str(date), str(time_str), str(tz), float(dut1))  # type: ignore[arg-type]
            ts_dict = dict(ts) if isinstance(ts, dict) else getattr(ts, "to_dict", lambda: {} )()
            out = {
                "jd_tt": ts_dict.get("jd_tt"),
                "jd_ut1": ts_dict.get("jd_ut1"),
                "delta_t": ts_dict.get("delta_t"),
                "dut1": ts_dict.get("dut1"),
                "dat": ts_dict.get("dat"),
                "tz_offset_seconds": ts_dict.get("tz_offset_seconds"),
                "timezone": ts_dict.get("timezone"),
                "warnings": ts_dict.get("warnings", []),
                "precision": ts_dict.get("precision"),
            }
            if include_jd_utc:
                out["jd_utc"] = ts_dict.get("jd_utc")
            return jsonify(ok=True, op=op, input={"date": date, "time": time_str, "tz": tz, "dut1_seconds": float(dut1)}, result=out), 200

        if op == "jd_utc":
            date = params.get("date"); time_str = params.get("time"); tz = params.get("tz")
            missing = [k for k in ("date", "time", "tz") if not params.get(k)]
            if missing:
                return jsonify(ok=False, error="bad_request", detail=f"Missing keys: {', '.join(missing)}"), 400
            jd = _tk_jd_utc(str(date), str(time_str), str(tz))  # type: ignore[arg-type]
            return jsonify(ok=True, op=op, input={"date": date, "time": time_str, "tz": tz}, result={"jd_utc": float(jd)}), 200

        if op == "tt_from_utc_jd":
            if "jd_utc" not in params:
                return jsonify(ok=False, error="bad_request", detail="Missing key: jd_utc"), 400
            jd_utc = float(params["jd_utc"])
            jd_tt = _tk_tt_from_utc_jd(jd_utc)  # type: ignore[arg-type]
            return jsonify(ok=True, op=op, input={"jd_utc": jd_utc}, result={"jd_tt": float(jd_tt)}), 200

        if op == "ut1_from_utc_jd":
            missing = [k for k in ("jd_utc", "dut1_seconds") if k not in params]
            if missing:
                return jsonify(ok=False, error="bad_request", detail=f"Missing keys: {', '.join(missing)}"), 400
            jd_utc = float(params["jd_utc"])
            dut1 = float(params["dut1_seconds"])
            jd_ut1 = _tk_ut1_from_utc_jd(jd_utc, dut1)  # type: ignore[arg-type]
            return jsonify(ok=True, op=op, input={"jd_utc": jd_utc, "dut1_seconds": dut1}, result={"jd_ut1": float(jd_ut1)}), 200

        # Unknown op
        return jsonify(ok=False, error="unknown_op", detail=f"Unsupported op '{op}'. Try 'help'."), 400

    except Exception as e:
        return jsonify(ok=False, error="calc_failed", op=op, detail=str(e)), 400
