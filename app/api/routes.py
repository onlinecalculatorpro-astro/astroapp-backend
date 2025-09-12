# app/api/routes.py
"""
Ops & Diagnostics routes (mounted with NO prefix by main.py)

Single dispatcher for shared/core ops:
- POST /ops/calculate

Diagnostics & health:
- GET  /ops/health
- GET  /api/health                  (back-compat; main.py adds Deprecation header)
- GET  /ops/version
- GET  /ops/config
- GET  /ops/diag/cores
- GET  /ops/diag/validators
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import os
import inspect

from flask import Blueprint, jsonify, request
from app.utils.ratelimit import rate_limit

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
# Canonical engine
_ts_mod, _engine_build_timescales, _TS_ERR = _try_import("app.core.timescales", "build_timescales")
_, _TimeScales, _TS_CLS_ERR                = _try_import("app.core.timescales", "TimeScales")

# Forwarders (single surface used by dispatcher)
_tk_mod, _tk_build_ts, _TK_ERR             = _try_import("app.core.time_kernel", "build_timescales")
_, _tk_jd_utc, _TK_JDU_ERR                 = _try_import("app.core.time_kernel", "julian_day_utc")
_, _tk_tt_from_utc, _TK_TT_ERR             = _try_import("app.core.time_kernel", "jd_tt_from_utc_jd")
_, _tk_ut1_from_utc, _TK_UT1_ERR           = _try_import("app.core.time_kernel", "jd_ut1_from_utc_jd")

# Other shared cores (presence only; useful for diag)
_es_mod, _, _ES_ERR                        = _try_import("app.core.ephem_singleton")
_ea_mod, _, _EA_ERR                        = _try_import("app.core.ephemeris_adapter")
_ast_mod, _, _AST_ERR                      = _try_import("app.core.astronomy")
_h_mod, _, _H_ERR                          = _try_import("app.core.house")
_hs_mod, _, _HS_ERR                        = _try_import("app.core.houses")
_hsa_mod, _, _HSA_ERR                      = _try_import("app.core.houses_advanced")

# Validators
_val_mod, _, _VAL_ERR                      = _try_import("app.core.validator")
_vval_mod, _normalize_times_input, _VVAL_ERR = _try_import("app.core.validator", "normalize_timescales_input")
_wval_mod, _, _WVAL_ERR                    = _try_import("app.core.western_validator")
_ved_val_mod, _normalize_vim_payload, _VEDVAL_ERR = _try_import("app.core.vedic_validator", "normalize_vim_payload")

# ───────────────────────── helpers ─────────────────────────
def _env_dut1_seconds() -> float:
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST",
                                    os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0

def _ok(data: Dict[str, Any], **meta):
    return jsonify({"ok": True, **data, "meta": meta}), 200

def _err(status: int, code: str, detail: str, **meta):
    return jsonify({"ok": False, "error": code, "detail": detail, "meta": meta}), status

# One shared bucket for /ops/calculate (env overridable)
RL_OPS_CALCULATE = int(os.getenv("ASTRO_RL_OPS_CALCULATE_PER_MIN", "60"))
def _ops_bucket(*_a, **_k) -> str:
    return "ops-calc"

# ───────────────────────── basic ops & health ─────────────────────────
@ops_api.get("/ops/health")
def ops_health():
    return jsonify(ok=True, service="astro-backend", scope="ops", status="ok"), 200

@ops_api.get("/api/health")
def api_health_backcompat():
    # Deprecation headers are added in main.py after_request
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
            "build_timescales_sig": _sig(_engine_build_timescales) if _engine_build_timescales else None,
            "TimeScales_loaded": _TimeScales is not None,
            "TimeScales_error": _TS_CLS_ERR,
        },
        "time_kernel": {
            "loaded": _tk_mod is not None,
            "error": _TK_ERR,
            "forwarders_ok": all([_tk_build_ts, _tk_jd_utc, _tk_tt_from_utc, _tk_ut1_from_utc]),
            "build_timescales_sig": _sig(_tk_build_ts) if _tk_build_ts else None,
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
            "normalize_timescales_input_sig": _sig(_normalize_times_input) if _normalize_times_input else None,
            "functions": list_callables(_val_mod) if _val_mod else None,
        },
        "western_validator": {
            "loaded": _wval_mod is not None,
            "error": _WVAL_ERR,
            "functions": list_callables(_wval_mod) if _wval_mod else None,
        },
        "vedic_validator": {
            "loaded": _ved_val_mod is not None,
            "error": _VEDVAL_ERR,
            "normalize_vim_payload_sig": _sig(_normalize_vim_payload) if _normalize_vim_payload else None,
            "functions": list_callables(_ved_val_mod) if _ved_val_mod else None,
        },
    }), 200

# ───────────────────────── unified dispatcher ─────────────────────────
def _unwrap_params(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts multiple shapes and returns a flat params dict:

    - { op, params: {date,time,tz,...}, include_jd_utc? }
    - { op, date, time, tz, ... }
    - { op, timescales: {date,time,tz,...} }
    - { op, data: { timescales: { ... } } }
    """
    if not isinstance(body, dict):
        return {}

    params = body.get("params")
    if isinstance(params, dict):
        return params

    # Flat top-level keys
    if any(k in body for k in ("date", "time", "tz", "timezone", "place_tz")):
        return {
            "date": body.get("date"),
            "time": body.get("time"),
            "tz": body.get("tz") or body.get("timezone") or body.get("place_tz"),
            "dut1_seconds": body.get("dut1_seconds"),
        }

    # timescales envelope
    ts = body.get("timescales")
    if isinstance(ts, dict):
        return ts

    # data.timescales envelope
    data = body.get("data")
    if isinstance(data, dict) and isinstance(data.get("timescales"), dict):
        return data["timescales"]

    return {}

@ops_api.post("/ops/calculate")
@rate_limit(RL_OPS_CALCULATE, key_fn=_ops_bucket)
def ops_calculate():
    """
    Unified common ops endpoint (backed by time_kernel forwarders).

    Body:
    {
      "op": "<timescales|jd_utc|tt_from_utc_jd|ut1_from_utc_jd>",
      "params": {...}                  // optional; also supports flat or {timescales:{...}}
      "include_jd_utc": false          // only used by op="timescales"
    }
    """
    if _tk_mod is None or _tk_build_ts is None:
        return _err(503, "core_unavailable", "time_kernel not loaded", op=None)

    body = request.get_json(silent=True) or {}
    op = str(body.get("op") or "").strip().lower()
    include_jd_utc = bool(body.get("include_jd_utc"))

    # Allow multiple body shapes
    params = _unwrap_params(body)

    # ── TIMESCALES ────────────────────────────────────────────────────────────
    if op == "timescales":
        # Normalize inputs using common validator if present
        if _normalize_times_input:
            try:
                norm, warns, tz_norm = _normalize_times_input(params)  # type: ignore[misc]
            except Exception as e:
                return _err(400, "bad_request", f"normalize_timescales_input failed: {e}", op=op)
            date = norm.get("date"); time_str = norm.get("time"); tz = norm.get("tz")
            dut1 = norm.get("dut1_seconds", _env_dut1_seconds())
            if not (date and time_str and tz):
                return _err(400, "bad_request", "Missing keys: date, time, tz", op=op)
        else:
            # Best-effort fallback
            date = params.get("date")
            time_str = params.get("time")
            tz = params.get("tz") or params.get("tz_name") or params.get("place_tz")
            dut1 = params.get("dut1_seconds", _env_dut1_seconds())
            if not (date and time_str and tz):
                return _err(400, "bad_request", "Missing keys: date, time, tz", op=op)

        try:
            dut1 = float(dut1)
        except Exception:
            return _err(400, "bad_request", "dut1_seconds must be a float", op=op)

        try:
            ts = _tk_build_ts(str(date), str(time_str), str(tz), float(dut1))  # returns dict via time_kernel
        except Exception as e:
            return _err(400, "timescales_failed", str(e), op=op)

        # Sanitize (hide jd_utc unless explicitly requested)
        ts_out = {
            "jd_tt": ts.get("jd_tt"),
            "jd_ut1": ts.get("jd_ut1"),
            "delta_t": ts.get("delta_t"),
            "dut1": ts.get("dut1"),
            "dat": ts.get("dat"),
            "tz_offset_seconds": ts.get("tz_offset_seconds"),
            "timezone": ts.get("timezone"),
            "warnings": ts.get("warnings", []),
            "precision": ts.get("precision"),
        }
        if include_jd_utc:
            ts_out["jd_utc"] = ts.get("jd_utc")

        return _ok(
            {"timescales": ts_out,
             "input": {"date": date, "time": time_str, "tz": tz, "dut1_seconds": float(dut1)}},
            op=op
        )

    # ── JD_UTC (deprecated helper) ───────────────────────────────────────────
    if op == "jd_utc":
        date = params.get("date")
        time_str = params.get("time")
        tz = params.get("tz") or params.get("tz_name") or params.get("place_tz")
        if not (date and time_str and tz):
            return _err(400, "bad_request", "Missing keys: date, time, tz", op=op)
        try:
            jd_utc = float(_tk_jd_utc(str(date), str(time_str), str(tz)))  # type: ignore[misc]
        except Exception as e:
            return _err(400, "jd_utc_failed", str(e), op=op)
        return _ok({"jd_utc": jd_utc, "input": {"date": date, "time": time_str, "tz": tz}}, op=op)

    # ── TT from UTC JD ───────────────────────────────────────────────────────
    if op == "tt_from_utc_jd":
        if "jd_utc" not in params:
            return _err(400, "bad_request", "Missing key: jd_utc", op=op)
        try:
            jd_tt = float(_tk_tt_from_utc(float(params["jd_utc"])))  # type: ignore[misc]
        except Exception as e:
            return _err(400, "tt_from_utc_jd_failed", str(e), op=op)
        return _ok({"jd_tt": jd_tt, "input": {"jd_utc": float(params["jd_utc"])}}, op=op)

    # ── UT1 from UTC JD ──────────────────────────────────────────────────────
    if op == "ut1_from_utc_jd":
        missing = [k for k in ("jd_utc", "dut1_seconds") if k not in params]
        if missing:
            return _err(400, "bad_request", f"Missing keys: {', '.join(missing)}", op=op)
        try:
            jd_ut1 = float(_tk_ut1_from_utc(float(params["jd_utc"]), float(params["dut1_seconds"])))  # type: ignore[misc]
        except Exception as e:
            return _err(400, "ut1_from_utc_jd_failed", str(e), op=op)
        return _ok(
            {"jd_ut1": jd_ut1,
             "input": {"jd_utc": float(params["jd_utc"]), "dut1_seconds": float(params["dut1_seconds"])}},
            op=op
        )

    # Unknown op
    return _err(
        400,
        "unsupported_op",
        "op must be one of: timescales, jd_utc, tt_from_utc_jd, ut1_from_utc_jd",
        op=op or None
    )
