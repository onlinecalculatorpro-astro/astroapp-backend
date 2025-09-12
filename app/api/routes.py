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
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
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

# ───────────────────────── validators ─────────────────────────
_val_mod, _, _VAL_ERR                   = _try_import("app.core.validator")
_wval_mod, _, _WVAL_ERR                 = _try_import("app.core.western_validator")
_vval_mod, _normalize_vim_payload, _VVAL_ERR = _try_import("app.core.vedic_validator", "normalize_vim_payload")

# ───────────────────────── ops basics ─────────────────────────
@ops_api.get("/ops/health")
def ops_health():
    """Ops-only health."""
    return jsonify(ok=True, service="astro-backend", scope="ops", status="ok"), 200

@ops_api.get("/api/health")
def api_health_backcompat():
    """
    Back-compat health endpoint. Prefer `/healthz`.
    main.py also sets Deprecation & Link headers in after_request,
    but we include them here too for robustness.
    """
    resp = jsonify(ok=True, service="astro-backend", scope="api", status="ok", deprecated=True)
    try:
        resp.headers["Deprecation"] = "true"
        resp.headers["Link"] = "</healthz>; rel=\"successor-version\""
    except Exception:
        pass
    return resp, 200

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
        "time_kernel": {"loaded": _tk_mod is not None, "error": _TK_ERR},
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

# ───────────────────────── timescales probe ─────────────────────────
@ops_api.post("/ops/timescales")
def ops_timescales():
    """
    POST a light preview for build_timescales.
    Body: { "date": "YYYY-MM-DD", "time": "HH:MM:SS[.frac]", "tz": "Area/City", "dut1_seconds": 0.0 }
    Returns a sanitized view (no jd_utc) so callers don’t rely on it downstream.
    """
    if _build_timescales is None:
        return jsonify(ok=False, error="timescales_unavailable", detail=_TS_ERR), 503

    body = request.get_json(silent=True) or {}
    date = body.get("date"); time_str = body.get("time"); tz = body.get("tz")
    dut1 = body.get("dut1_seconds", os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", 0.0)))

    missing = [k for k in ("date", "time", "tz") if not body.get(k)]
    if missing:
        return jsonify(ok=False, error="bad_request", detail=f"Missing keys: {', '.join(missing)}"), 400

    try:
        dut1 = float(dut1)
    except Exception:
        return jsonify(ok=False, error="bad_request", detail="dut1_seconds must be a float"), 400

    try:
        ts = _build_timescales(str(date), str(time_str), str(tz), float(dut1))
    except Exception as e:
        return jsonify(ok=False, error="timescales_failed", detail=str(e)), 400

    # Sanitize output to avoid downstream reliance on jd_utc
    try:
        ts_dict = ts.to_dict() if hasattr(ts, "to_dict") else dict(ts.__dict__)
    except Exception:
        ts_dict = {}

    ts_out = {
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

    return jsonify(
        ok=True,
        input={"date": date, "time": time_str, "tz": tz, "dut1_seconds": float(dut1)},
        timescales=ts_out
    ), 200
