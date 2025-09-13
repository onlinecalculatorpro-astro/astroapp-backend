# app/api/routes.py
# -*- coding: utf-8 -*-
"""
Ops & Diagnostics routes (mounted with NO prefix by main.py)

Single dispatcher for shared/core ops:
- POST /ops/calculate
  • op: "timescales" | "jd_utc" | "tt_from_utc_jd" | "ut1_from_utc_jd"
  • op: "ephem_vector" | "ephem_equatorial" | "ephem_ecliptic" | "ephem_sidereal"
  • op: "chart" (astronomy.compute_chart)

Diagnostics & health:
- GET  /ops/health
- GET  /api/health                  (back-compat; main.py may add Deprecation header)
- GET  /ops/version
- GET  /ops/config
- GET  /ops/diag/cores              (includes leap-seconds + ephemeris diagnostics)
- GET  /ops/diag/validators
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import os
import inspect
import math

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

# Leap-seconds (diagnostics)
_ls_mod, _ls_delta_at, _LS_ERR             = _try_import("app.core.leapseconds", "delta_at")

# Ephemeris cores (diagnostics + runtime access)
_es_mod, _es_dummy, _ES_ERR                = _try_import("app.core.ephem_singleton")
_, _ep_get_ts, _ES_TS_ERR                  = _try_import("app.core.ephem_singleton", "get_timescale")
_, _ep_get_eph, _ES_EPH_ERR                = _try_import("app.core.ephem_singleton", "get_planets")

_ea_mod, _ea_dummy, _EA_ERR                = _try_import("app.core.ephemeris_adapter")
_, _ea_diag, _EA_DIAG_ERR                  = _try_import("app.core.ephemeris_adapter", "ephemeris_diagnostics")
_, _ea_ecl, _EA_ECL_ERR                    = _try_import("app.core.ephemeris_adapter", "ecliptic_longitudes")
_, _ea_many, _EA_MANY_ERR                  = _try_import("app.core.ephemeris_adapter", "ecliptic_longitudes_many")

# Astronomy core (THIS is the one we call for charts)
_ast_mod, _, _AST_ERR                      = _try_import("app.core.astronomy")
_, _compute_chart, _ASTRO_ERR              = _try_import("app.core.astronomy", "compute_chart")

# Houses (presence/diag only; routing doesn’t call directly)
_h_mod,   _, _H_ERR                        = _try_import("app.core.house")
_hs_mod,  _, _HS_ERR                       = _try_import("app.core.houses")
_hsa_mod, _, _HSA_ERR                      = _try_import("app.core.houses_advanced")

# Validators (ALL validation flows go through validator.py)
_val_mod, _normalize_common, _VAL_ERR          = _try_import("app.core.validator", "normalize_common_payload")
_, _normalize_timescales_input, _VAL_ALIAS_ERR = _try_import("app.core.validator", "normalize_timescales_input")
_, _normalize_body, _NB_ERR                    = _try_import("app.core.validator", "normalize_body")
_, _normalize_for_vedic, _NV_ERR               = _try_import("app.core.validator", "normalize_for_vedic")
_, _normalize_for_western, _NW_ERR             = _try_import("app.core.validator", "normalize_for_western")
_, _normalize_chart_payload, _NC_ERR           = _try_import("app.core.validator", "normalize_chart_payload")

# Presence-only (diagnostics visibility)
_wval_mod, _, _WVAL_ERR                    = _try_import("app.core.western_validator")
_ved_val_mod, _, _VEDVAL_ERR               = _try_import("app.core.vedic_validator")

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

# ───────────────────────── Ephemeris helpers ─────────────────────────
_BODY_KEY_CANDIDATES: Dict[str, List[Any]] = {
    "sun":      ["sun", 10],
    "moon":     ["moon", 301],
    "mercury":  ["mercury", "mercury barycenter", 199, 1],
    "venus":    ["venus", "venus barycenter", 299, 2],
    "earth":    ["earth", "earth barycenter", 399, 3],
    "mars":     ["mars", "mars barycenter", 499, 4],
    "jupiter":  ["jupiter", "jupiter barycenter", 599, 5],
    "saturn":   ["saturn", "saturn barycenter", 699, 6],
    "uranus":   ["uranus", "uranus barycenter", 799, 7],
    "neptune":  ["neptune", "neptune barycenter", 899, 8],
    "pluto":    ["pluto", "pluto barycenter", 999, 9],
}

def _resolve_kernel_body(eph, body_norm: str):
    if not eph or not body_norm:
        return None
    for key in _BODY_KEY_CANDIDATES.get(body_norm, []):
        try:
            return eph[key]
        except Exception:
            continue
    return None

def _ephem_status() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "loaded": False,
        "error": _ES_ERR or _ES_TS_ERR or _ES_EPH_ERR,
        "data_dir": None,
        "kernel_candidates": None,
        "kernel_active": None,
    }
    if _es_mod:
        info["data_dir"] = getattr(_es_mod, "_EPHEM_DIR", None)
        info["kernel_candidates"] = list(getattr(_es_mod, "_CANDIDATES", ()) or ())
        try:
            ts = _ep_get_ts() if _ep_get_ts else None
            eph = _ep_get_eph() if _ep_get_eph else None
            if ts and eph:
                meta = getattr(_es_mod, "_META", None)
                if isinstance(meta, dict):
                    info["kernel_active"] = meta.get("kernel")
                info["loaded"] = True
                info["error"] = None
        except Exception as e:
            info["loaded"] = False
            info["error"] = f"{type(e).__name__}: {e}"
    return info

def _compute_vector(eph, t, body_norm: str) -> Dict[str, float]:
    earth = eph["earth"]
    if body_norm == "earth":
        return {"x_au": 0.0, "y_au": 0.0, "z_au": 0.0, "distance_au": 0.0}
    target = _resolve_kernel_body(eph, body_norm)
    if target is None:
        raise ValueError(f"Body not present in kernel: '{body_norm}'")
    g = earth.at(t).observe(target)  # geometric (ICRS)
    x, y, z = (float(g.position.au[0]), float(g.position.au[1]), float(g.position.au[2]))
    dist = math.sqrt(x*x + y*y + z*z)
    return {"x_au": x, "y_au": y, "z_au": z, "distance_au": dist}

def _compute_equatorial(eph, t, body_norm: str) -> Dict[str, float]:
    earth = eph["earth"]
    if body_norm == "earth":
        return {"ra_deg": float("nan"), "dec_deg": float("nan"), "distance_au": 0.0}
    target = _resolve_kernel_body(eph, body_norm)
    if target is None:
        raise ValueError(f"Body not present in kernel: '{body_norm}'")
    a = earth.at(t).observe(target).apparent()  # apparent RA/Dec
    ra, dec, dist = a.radec()
    return {"ra_deg": float(ra.hours) * 15.0, "dec_deg": float(dec.degrees), "distance_au": float(dist.au)}

def _compute_ecliptic_true(eph, t, body_norm: str) -> Dict[str, float]:
    if body_norm == "earth":
        return {"lon_deg": float("nan"), "lat_deg": float("nan"), "distance_au": 0.0}
    from skyfield import framelib as _fl
    earth = eph["earth"]
    target = _resolve_kernel_body(eph, body_norm)
    if target is None:
        raise ValueError(f"Body not present in kernel: '{body_norm}'")
    a = earth.at(t).observe(target).apparent()
    lat, lon, dist = a.frame_latlon(_fl.ecliptic_frame)  # true-of-date
    return {"lon_deg": float(lon.degrees) % 360.0, "lat_deg": float(lat.degrees), "distance_au": float(dist.au)}

def _compute_sidereal(eph, t, body_norm: str, ayanamsa_offset_deg: float = 0.0) -> Dict[str, float]:
    base = _compute_ecliptic_true(eph, t, body_norm)
    true_lon = float(base["lon_deg"])
    try:
        off = float(ayanamsa_offset_deg)
    except Exception:
        off = 0.0
    return {"lon_sidereal_deg": (true_lon - off) % 360.0, "lon_true_deg": true_lon, "ayanamsa_offset_deg": off}

# ───────────────────────── basic ops & health ─────────────────────────
@ops_api.get("/ops/health")
def ops_health():
    return jsonify(ok=True, service="astro-backend", scope="ops", status="ok"), 200

@ops_api.get("/api/health")
def api_health_backcompat():
    # Deprecation headers are added in main.py after_request (if any)
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
    """
    Report presence and signatures for shared core modules.
    Optionally, if query provides ?date=&time=&tz=, include a leap-seconds
    sample probe comparing timescales.dat vs leapseconds.delta_at(MJD).
    """
    payload: Dict[str, Any] = {
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
        "ephem_singleton": {
            "loaded": _es_mod is not None,
            "error": _ES_ERR,
            "get_timescale_sig": _sig(_ep_get_ts) if _ep_get_ts else None,
            "get_planets_sig": _sig(_ep_get_eph) if _ep_get_eph else None,
            "status": _ephem_status(),
        },
        "ephemeris_adapter": {
            "loaded": _ea_mod is not None,
            "error": _EA_ERR,
            "ecliptic_longitudes_sig": _sig(_ea_ecl) if _ea_ecl else None,
            "ecliptic_longitudes_many_sig": _sig(_ea_many) if _ea_many else None,
            "diagnostics_sig": _sig(_ea_diag) if _ea_diag else None,
        },
        "astronomy": {
            "loaded": _ast_mod is not None,
            "error": _AST_ERR or _ASTRO_ERR,
            "compute_chart_sig": _sig(_compute_chart) if _compute_chart else None,
        },
        "house":            {"loaded": _h_mod   is not None, "error": _H_ERR},
        "houses":           {"loaded": _hs_mod  is not None, "error": _HS_ERR},
        "houses_advanced":  {"loaded": _hsa_mod is not None, "error": _HSA_ERR},
        "leapseconds": {
            "loaded": (_ls_mod is not None) or (_ls_delta_at is not None),
            "error": _LS_ERR,
        },
    }

    # Optional: include a trimmed ephemeris_adapter diagnostics block if callable
    try:
        if _ea_diag:
            diag = _ea_diag()  # may include coverage, node cache, etc.
            payload["ephemeris_adapter"]["diagnostics_sample"] = {
                "ephemeris_name": diag.get("ephemeris_name"),
                "kernels": diag.get("kernels"),
                "node_model": diag.get("node_model"),
                "smalls_enabled": diag.get("smalls_enabled"),
                "coverage_jd": diag.get("coverage_jd"),
            }
    except Exception as e:
        payload["ephemeris_adapter"]["diagnostics_error"] = str(e)

    # Optional sample probe if date/time/tz provided
    q = request.args or {}
    date = q.get("date")
    time_str = q.get("time")
    tz = q.get("tz")
    dut1 = q.get("dut1_seconds", _env_dut1_seconds())

    if date and time_str and tz and _tk_build_ts:
        try:
            ts = _tk_build_ts(str(date), str(time_str), str(tz), float(dut1))  # dict via time_kernel
            jd_utc = ts.get("jd_utc")
            dat_ts = ts.get("dat")
            sample = {"input": {"date": date, "time": time_str, "tz": tz, "dut1_seconds": float(dut1)}}

            if isinstance(jd_utc, (int, float)) and _ls_delta_at:
                mjd = float(jd_utc) - 2400000.5
                try:
                    li = _ls_delta_at(mjd)  # type: ignore[misc]
                    li_dict = {
                        "delta_at": float(getattr(li, "delta_at", None)),
                        "source": getattr(li, "source", None),
                        "status": getattr(li, "status", None),
                        "last_known_mjd": float(getattr(li, "last_known_mjd", 0.0) or 0.0),
                        "erfa_status_code": getattr(li, "erfa_status_code", None),
                        "notes": getattr(li, "notes", None),
                    }
                    sample.update({
                        "jd_utc": float(jd_utc),
                        "dat_from_timescales": float(dat_ts) if isinstance(dat_ts, (int, float)) else None,
                        "delta_at_probe": li_dict,
                        "delta_diff_seconds": (
                            (float(dat_ts) - float(li_dict["delta_at"]))
                            if isinstance(dat_ts, (int, float)) and isinstance(li_dict["delta_at"], (int, float))
                            else None
                        ),
                    })
                except Exception as e:
                    sample.update({"probe_error": str(e)})
            else:
                sample.update({"note": "jd_utc or leapseconds not available"})

            payload["leapseconds"]["sample"] = sample
        except Exception as e:
            payload["leapseconds"]["sample_error"] = str(e)

    return jsonify(payload), 200

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
        # Routing uses ONLY validator.py; vfiles presence shown for visibility.
        "validator": {
            "loaded": _val_mod is not None,
            "error": _VAL_ERR,
            "normalize_common_payload_sig": _sig(_normalize_common) if _normalize_common else None,
            "normalize_timescales_input_sig": _sig(_normalize_timescales_input) if _normalize_timescales_input else None,
            "normalize_for_vedic_sig": _sig(_normalize_for_vedic) if _normalize_for_vedic else None,
            "normalize_for_western_sig": _sig(_normalize_for_western) if _normalize_for_western else None,
            "normalize_chart_payload_sig": _sig(_normalize_chart_payload) if _normalize_chart_payload else None,
            "normalize_body_sig": _sig(_normalize_body) if _normalize_body else None,
            "functions": list_callables(_val_mod) if _val_mod else None,
        },
        "western_validator": {
            "loaded": _wval_mod is not None,
            "error": _WVAL_ERR,
            "note": "routes use app.core.validator wrappers; this module is optional",
            "functions": list_callables(_wval_mod) if _wval_mod else None,
        },
        "vedic_validator": {
            "loaded": _ved_val_mod is not None,
            "error": _VEDVAL_ERR,
            "note": "routes use app.core.validator wrappers; this module is optional",
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

    if any(k in body for k in ("date", "time", "tz", "timezone", "place_tz", "body", "ayanamsa_offset_deg")):
        return {
            "date": body.get("date"),
            "time": body.get("time"),
            "tz": body.get("tz") or body.get("timezone") or body.get("place_tz"),
            "dut1_seconds": body.get("dut1_seconds"),
            "body": body.get("body"),
            "ayanamsa_offset_deg": body.get("ayanamsa_offset_deg"),
            # chart-related passthroughs if user goes flat
            "mode": body.get("mode"),
            "frame": body.get("frame"),
            "center": body.get("center"),
            "topocentric": body.get("topocentric"),
            "latitude": body.get("latitude"),
            "longitude": body.get("longitude"),
            "elevation_m": body.get("elevation_m"),
            "elev_m": body.get("elev_m"),
            "elevation": body.get("elevation"),
            "bodies": body.get("bodies"),
            "points": body.get("points"),
            "ayanamsa": body.get("ayanamsa"),
        }

    ts = body.get("timescales")
    if isinstance(ts, dict):
        return ts

    data = body.get("data")
    if isinstance(data, dict) and isinstance(data.get("timescales"), dict):
        return data["timescales"]

    return {}

@ops_api.post("/ops/calculate")
@rate_limit(RL_OPS_CALCULATE, key_fn=_ops_bucket)
def ops_calculate():
    """
    Unified common ops endpoint.

    Body:
    {
      "op": "<timescales|jd_utc|tt_from_utc_jd|ut1_from_utc_jd|ephem_vector|ephem_equatorial|ephem_ecliptic|ephem_sidereal|chart>",
      "params": {...}                  // optional; also supports flat or {timescales:{...}}
      "include_jd_utc": false          // only used by op="timescales" (and normalization if chart needs it)
    }
    """
    if _tk_mod is None or _tk_build_ts is None:
        return _err(503, "core_unavailable", "time_kernel not loaded", op=None)

    body = request.get_json(silent=True) or {}
    op = str(body.get("op") or "").strip().lower()
    include_jd_utc = bool(body.get("include_jd_utc"))

    raw_params = _unwrap_params(body)

    # ── TIMESCALES ────────────────────────────────────────────────────────────
    if op == "timescales":
        if _normalize_common:
            try:
                norm, _warns, _tz_norm = _normalize_common(raw_params, compute_timescales=False)  # type: ignore[misc]
            except Exception as e:
                return _err(400, "bad_request", f"normalize_common_payload failed: {e}", op=op)
            date = norm.get("date"); time_str = norm.get("time"); tz = norm.get("tz")
            dut1 = norm.get("dut1_seconds", _env_dut1_seconds())
            if not (date and time_str and tz):
                return _err(400, "bad_request", "Missing keys: date, time, tz", op=op)
        else:
            date = raw_params.get("date")
            time_str = raw_params.get("time")
            tz = raw_params.get("tz") or raw_params.get("tz_name") or raw_params.get("place_tz")
            dut1 = raw_params.get("dut1_seconds", _env_dut1_seconds())
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
        params = raw_params
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
        params = raw_params
        if "jd_utc" not in params:
            return _err(400, "bad_request", "Missing key: jd_utc", op=op)
        try:
            jd_tt = float(_tk_tt_from_utc(float(params["jd_utc"])))  # type: ignore[misc]
        except Exception as e:
            return _err(400, "tt_from_utc_jd_failed", str(e), op=op)
        return _ok({"jd_tt": jd_tt, "input": {"jd_utc": float(params["jd_utc"])}}, op=op)

    # ── UT1 from UTC JD ──────────────────────────────────────────────────────
    if op == "ut1_from_utc_jd":
        params = raw_params
        missing = [k for k in ("jd_utc", "dut1_seconds") if k not in params]
        if missing:
            return _err(400, "bad_request", f"Missing keys: {', '.join(missing)}", op=op)
        try:
            jd_ut1 = float(_tk_ut1_from_utc(float(params["jd_utc"]), float(params["dut1_seconds"])))  # type: ignore[misc]
        except Exception as e:
            return _err(400, "ut1_from_utc_jd_failed", str(e), op=op)
        return _ok(
            {"jd_ut1": jd_ut1,
             "input": {"jd_utc": float(params["jd_utc"]), "dut1_seconds": float(params["dut1_seconds"])}} ,
            op=op
        )

    # ── Ephemeris ops (validator-backed) ─────────────────────────────────────
    if op in ("ephem_vector", "ephem_equatorial", "ephem_ecliptic", "ephem_sidereal"):
        if not _normalize_common or not _normalize_body:
            return _err(503, "validator_unavailable", "validator functions not loaded", op=op)

        try:
            norm, warns, _tz = _normalize_common(raw_params, compute_timescales=True, include_jd_utc=False)  # type: ignore[misc]
        except Exception as e:
            return _err(400, "bad_request", f"normalize_common_payload failed: {e}", op=op)

        jd_tt = norm.get("jd_tt") or (norm.get("timescales") or {}).get("jd_tt")
        if not isinstance(jd_tt, (int, float)):
            return _err(400, "bad_request", "jd_tt not available after normalization", op=op)

        body_norm, bwarn = _normalize_body(raw_params)  # type: ignore[misc]
        if not body_norm:
            return _err(400, "bad_body", "Missing or unsupported 'body'", op=op)

        try:
            eph = _ep_get_eph() if _ep_get_eph else None
            ts = _ep_get_ts() if _ep_get_ts else None
            if eph is None or ts is None:
                return _err(503, "ephem_core_unavailable", "Skyfield/JPL ephemeris not available", op=op)

            t = ts.tt_jd(float(jd_tt))
            if op == "ephem_vector":
                result = _compute_vector(eph, t, body_norm)
            elif op == "ephem_equatorial":
                result = _compute_equatorial(eph, t, body_norm)
            elif op == "ephem_ecliptic":
                result = _compute_ecliptic_true(eph, t, body_norm)
            else:  # ephem_sidereal
                off = raw_params.get("ayanamsa_offset_deg", 0.0)
                try:
                    off = float(off)
                except Exception:
                    off = 0.0
                result = _compute_sidereal(eph, t, body_norm, ayanamsa_offset_deg=off)

        except ValueError as e:
            return _err(400, "bad_body", str(e), op=op)
        except Exception as e:
            msg = str(e)
            if "ephemeris" in msg.lower() or "skyfield" in msg.lower():
                return _err(503, "ephem_core_unavailable", msg, op=op)
            return _err(500, "ephem_compute_error", msg, op=op)

        return _ok(
            {
                "result": result,
                "input": {"date": norm.get("date"), "time": norm.get("time"), "tz": norm.get("tz"), "body": body_norm},
                "warnings": list(warns or []) + list(bwarn or []),
            },
            op=op, body=body_norm
        )

        # ── Astronomy chart (compute_chart) ──────────────────────────────────────
    if op in ("chart", "astro_chart", "compute_chart"):
        # Ensure dependencies are available
        if not (callable(_normalize_chart_payload) and callable(_compute_chart)):
            return _err(503, "astronomy_unavailable", "astronomy module or validator not loaded", op=op)

        # Normalize incoming params (optionally include jd_utc for debugging)
        try:
            norm, warns, tz_norm = _normalize_chart_payload(
                raw_params,
                compute_timescales=True,
                include_jd_utc=bool(include_jd_utc),
            )  # type: ignore[misc]
        except Exception as e:
            return _err(400, "bad_request", f"normalize_chart_payload failed: {e}", op=op)

        # Make sure DUT1 survives normalization if the client provided it
        try:
            if "dut1_seconds" in raw_params and "dut1_seconds" not in norm:
                norm["dut1_seconds"] = raw_params["dut1_seconds"]
            if "dut1" in raw_params and "dut1" not in norm:
                norm["dut1"] = raw_params["dut1"]
        except Exception:
            pass  # best-effort passthrough

        # Compute the chart
        try:
            out = _compute_chart(norm)  # dict as defined by astronomy.compute_chart
        except Exception as e:
            code = getattr(e, "code", None)
            msg = str(e)
            if code:
                status = 400 if any(tok in code for tok in ("invalid", "unsupported", "missing", "bad", "not_")) else 500
                return _err(status, code, msg, op=op)
            return _err(500, "chart_compute_error", msg, op=op)

        # Merge warnings from normalization + engine (dedupe, preserve order)
        engine_warns = list(out.get("warnings") or [])
        merged_warns = list(dict.fromkeys((warns or []) + engine_warns))
        out["warnings"] = merged_warns

        # Echo back key input fields (plus DUT1 if present) for transparency
        input_echo = {
            "date": norm.get("date"),
            "time": norm.get("time"),
            "tz": norm.get("tz") or norm.get("place_tz"),
            "mode": norm.get("mode"),
            "frame": norm.get("frame"),
            "center": norm.get("center"),
            "topocentric": bool(norm.get("topocentric")),
            "latitude": norm.get("latitude"),
            "longitude": norm.get("longitude"),
            "elevation_m": norm.get("elevation_m"),
        }
        if "dut1_seconds" in norm:
            input_echo["dut1_seconds"] = norm["dut1_seconds"]
        if "dut1" in norm:
            input_echo["dut1"] = norm["dut1"]

        return _ok({"chart": out, "input": input_echo}, op=op, timezone=tz_norm)

    # Unknown op
    return _err(
        400,
        "unsupported_op",
        "op must be one of: timescales, jd_utc, tt_from_utc_jd, ut1_from_utc_jd, ephem_vector, ephem_equatorial, ephem_ecliptic, ephem_sidereal, chart",
        op=op or None
    )
