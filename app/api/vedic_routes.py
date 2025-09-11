# -*- coding: utf-8 -*-
from __future__ import annotations

from flask import Blueprint, request, jsonify
import json

vedic_bp = Blueprint("vedic", __name__, url_prefix="/api/vedic")

# ── guarded imports (don’t explode if a module is missing) ────────────────
def _maybe_import(path):
    try:
        module = __import__(path, fromlist=["*"])
        return module
    except Exception:
        return None

_panchanga           = _maybe_import("app.core.panchanga")
_panchanga_events    = _maybe_import("app.core.panchanga_events")
_varga               = _maybe_import("app.core.varga_charts")
_gochar              = _maybe_import("app.core.vedic_gochar")
_synastry            = _maybe_import("app.core.vedic_synastry")
_yogas               = _maybe_import("app.core.yogas")
_jaimini_arudha      = _maybe_import("app.core.jaimini_arudha")
_jaimini_argala      = _maybe_import("app.core.jaimini_argala")
_dasha               = _maybe_import("app.core.dasha_registry")
_houses              = _maybe_import("app.core.houses_advanced")
_ashtakavarga        = _maybe_import("app.core.ashtakavarga")
_ayanamsa            = _maybe_import("app.core.ayanamsa")
_graha_drishti       = _maybe_import("app.core.graha_drishti")
_drik_sidereal       = _maybe_import("app.core.drik_sidereal_positions")

# ── small helpers ─────────────────────────────────────────────────────────
def _json_body() -> dict:
    if request.is_json:
        return request.get_json(silent=True) or {}
    if request.data:
        try:
            return json.loads(request.data.decode("utf-8") or "{}")
        except Exception:
            pass
    return {}

def _respond(obj):
    if isinstance(obj, tuple) and len(obj) == 2:
        body, code = obj
        return jsonify(body), code
    if isinstance(obj, dict) and "ok" in obj:
        return jsonify(obj)
    return jsonify({"ok": True, "result": obj})

def _fail(msg: str, code: int = 400, **meta):
    return jsonify({"ok": False, "error": str(msg), "meta": meta}), code

def _call_first(mod, candidates, *args, **kwargs):
    if mod is None:
        raise RuntimeError("module_unavailable")
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise AttributeError(f"none_of_functions_found:{candidates}")

# ── health ────────────────────────────────────────────────────────────────
@vedic_bp.get("/health")
def vedic_health():
    present = {
        "panchanga": bool(_panchanga),
        "panchanga_events": bool(_panchanga_events),
        "varga": bool(_varga),
        "gochar": bool(_gochar),
        "synastry": bool(_synastry),
        "yogas": bool(_yogas),
        "jaimini_arudha": bool(_jaimini_arudha),
        "jaimini_argala": bool(_jaimini_argala),
        "dasha_registry": bool(_dasha),
        "houses_advanced": bool(_houses),
        "ashtakavarga": bool(_ashtakavarga),
        "ayanamsa": bool(_ayanamsa),
        "graha_drishti": bool(_graha_drishti),
        "drik_sidereal_positions": bool(_drik_sidereal),
    }
    return jsonify({"ok": True, "modules": present})

# ── Panchanga & Events ────────────────────────────────────────────────────
@vedic_bp.post("/panchanga")
def route_panchanga():
    try:
        payload = _json_body()
        out = _call_first(_panchanga, ["compute_panchanga"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"panchanga_failed:{e}")

@vedic_bp.post("/panchanga/events")
def route_panchanga_events():
    try:
        payload = _json_body()
        out = _call_first(_panchanga_events, ["compute_panchanga_events"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"panchanga_events_failed:{e}")

# ── Vargas ────────────────────────────────────────────────────────────────
@vedic_bp.post("/vargas")
def route_vargas():
    try:
        payload = _json_body()
        out = _call_first(_varga, ["compute_vargas", "build_vargas", "compute"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"vargas_failed:{e}")

# ── Gochar (transits/progressions) ────────────────────────────────────────
@vedic_bp.post("/gochar")
def route_gochar():
    try:
        payload = _json_body()
        out = _call_first(_gochar, ["compute_gochar", "find_transits_in_range", "compute"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"gochar_failed:{e}")

# ── Synastry ──────────────────────────────────────────────────────────────
@vedic_bp.post("/synastry")
def route_synastry():
    try:
        payload = _json_body()
        out = _call_first(_synastry, ["compute_synastry", "analyze_synastry", "compute"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"synastry_failed:{e}")

# ── Yogas (rule engine) ───────────────────────────────────────────────────
@vedic_bp.post("/yogas")
def route_yogas():
    try:
        payload = _json_body()
        out = _call_first(_yogas, ["evaluate_yogas", "compute_yogas", "evaluate", "compute"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"yogas_failed:{e}")

# ── Jaimini Arudha & Argala ───────────────────────────────────────────────
@vedic_bp.post("/jaimini/arudha")
def route_jaimini_arudha():
    try:
        payload = _json_body()
        out = _call_first(_jaimini_arudha, ["compute_arudha_padas", "arudha_padas", "compute"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"jaimini_arudha_failed:{e}")

@vedic_bp.post("/jaimini/argala")
def route_jaimini_argala():
    try:
        payload = _json_body()
        out = _call_first(_jaimini_argala, ["compute_argala", "argala", "compute"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"jaimini_argala_failed:{e}")

# ── Dasha systems via registry ────────────────────────────────────────────
@vedic_bp.get("/dasha/systems")
def route_dasha_systems():
    try:
        if _dasha is None:
            raise RuntimeError("module_unavailable")
        if hasattr(_dasha, "list_systems"):
            out = _dasha.list_systems()
        elif hasattr(_dasha, "registered_systems"):
            out = _dasha.registered_systems()
        else:
            out = getattr(_dasha, "SYSTEMS", {})
        return _respond({"ok": True, "systems": out})
    except Exception as e:
        return _fail(f"dasha_list_failed:{e}")

def _dispatch_dasha(system: str, payload: dict):
    if _dasha is None:
        raise RuntimeError("module_unavailable")
    if hasattr(_dasha, "dispatch_dasha"):
        return _dasha.dispatch_dasha(system, payload)
    if hasattr(_dasha, "compute_dasha_for_system"):
        return _dasha.compute_dasha_for_system(system=system, payload=payload)
    if hasattr(_dasha, "compute"):
        p = dict(payload); p["system"] = system
        return _dasha.compute(p)
    raise AttributeError("no dasha dispatcher found in dasha_registry")

@vedic_bp.post("/dasha/<system>")
def route_dasha(system: str):
    try:
        payload = _json_body()
        out = _dispatch_dasha(system, payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"dasha_failed:{e}", system=system)

# ── Houses (advanced) ─────────────────────────────────────────────────────
@vedic_bp.post("/houses")
def route_houses():
    try:
        payload = _json_body()
        out = _call_first(
            _houses,
            ["compute_house_system"],
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            house_system=str(payload.get("house_system", "placidus")),
            jd_ut=float(payload["jd_ut"]),
            jd_tt=payload.get("jd_tt"),
            jd_ut1=payload.get("jd_ut1"),
        )
        return _respond(out)
    except Exception as e:
        return _fail(f"houses_failed:{e}")

# ── Ashtakavarga ──────────────────────────────────────────────────────────
@vedic_bp.post("/ashtakavarga")
def route_ashtakavarga():
    try:
        payload = _json_body()
        out = _call_first(_ashtakavarga, ["compute_ashtakavarga", "compute"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"ashtakavarga_failed:{e}")

# ── Ayanāṁśa helpers ─────────────────────────────────────────────────────
@vedic_bp.get("/ayanamsa/<key>")
def route_ayanamsa_get(key: str):
    try:
        if _ayanamsa is None:
            raise RuntimeError("module_unavailable")
        jd_tt = float(request.args.get("jd") or request.args.get("jd_tt"))
        val = _ayanamsa.get_ayanamsa_deg(jd_tt, key)
        return _respond({"ok": True, "ayanamsa": key, "jd_tt": jd_tt, "value_deg": float(val)})
    except Exception as e:
        return _fail(f"ayanamsa_failed:{e}", key=key)

@vedic_bp.post("/ayanamsa")
def route_ayanamsa_post():
    try:
        payload = _json_body()
        jd_tt = float(payload["jd_tt"])
        key = str(payload.get("key", "lahiri"))
        val = _ayanamsa.get_ayanamsa_deg(jd_tt, key)
        return _respond({"ok": True, "ayanamsa": key, "jd_tt": jd_tt, "value_deg": float(val)})
    except Exception as e:
        return _fail(f"ayanamsa_failed:{e}")

# ── Graha Dṛṣṭi (Parāśari aspects) ────────────────────────────────────────
@vedic_bp.post("/drishti")
def route_graha_drishti():
    try:
        payload = _json_body()
        out = _call_first(_graha_drishti, ["compute_graha_drishti", "compute"], payload)
        return _respond(out)
    except Exception as e:
        return _fail(f"graha_drishti_failed:{e}")

# ── Sidereal positions (Drik) ─────────────────────────────────────────────
@vedic_bp.post("/positions")
def route_drik_sidereal_positions():
    try:
        payload = _json_body()
        out = _call_first(
            _drik_sidereal,
            ["compute_sidereal_positions", "sidereal_positions", "compute"],
            payload
        )
        return _respond(out)
    except Exception as e:
        return _fail(f"drik_sidereal_positions_failed:{e}")
