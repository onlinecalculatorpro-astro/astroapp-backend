from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import os
from flask import Blueprint, jsonify, request

# rate limiting
from app.utils.ratelimit import rate_limit, client_key, endpoint_key

# timescales (optional high-precision)
try:
    from app.core.timescales import build_timescales
    _TIMESCALES_OK = True
except Exception:
    _TIMESCALES_OK = False
    build_timescales = None  # type: ignore

# Vedic dasha engines — registry preferred, module fallback
try:
    from app.core.dasha_registry import compute_dasha as _compute_dasha_registry
except Exception:
    _compute_dasha_registry = None  # type: ignore

try:
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim_module
except Exception:
    _compute_vim_module = None  # type: ignore

vedic_api = Blueprint("vedic_api", __name__)

# per-endpoint rate-limit (env override-capable; matches your style)
RL_PREDICTIVE = int(os.getenv("ASTRO_RL_PREDICTIVE_PER_MIN", "12"))

# common tz aliases
_TZ_ALIAS = {
    "asia/patna": "Asia/Kolkata",
    "asia/calcutta": "Asia/Kolkata",
    "ist": "Asia/Kolkata",
}

def _normalize_tz(tz: str) -> str:
    if not isinstance(tz, str):
        return "UTC"
    key = tz.strip().lower()
    return _TZ_ALIAS.get(key, tz)

def _normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Massage the request into a shape the engine understands.
    - Computes (jd_tt, jd_ut1, jd_utc) if date/time/tz present and timescales available
    - Accepts 'levels'/'depth'/'max_levels' → depth 1..5 (default 5)
    """
    warns: List[str] = []

    date = str(payload.get("date") or payload.get("birth_date") or "")
    time = str(payload.get("time") or payload.get("birth_time") or "12:00:00")
    tz   = _normalize_tz(str(payload.get("tz") or payload.get("place_tz") or "UTC"))

    jd_tt = payload.get("jd_tt")
    jd_ut1 = payload.get("jd_ut1")
    jd_ut = payload.get("jd_ut")

    if (jd_tt is None or jd_ut1 is None) and _TIMESCALES_OK and date:
        try:
            ts = build_timescales(date=date, time=time, tz=tz)
            jd_tt = float(ts.jd_tt)
            jd_ut1 = float(ts.jd_ut1)
            jd_ut = float(ts.jd_utc)
        except Exception as e:
            warns.append(f"timescales_failed:{e!s}")

    # levels/depth
    levels = payload.get("levels", payload.get("depth", payload.get("max_levels", 5)))
    if isinstance(levels, list):
        depth = max(1, min(5, len(levels)))
    else:
        try:
            depth = int(levels)
            depth = max(1, min(5, depth))
        except Exception:
            depth = 5

    ayanamsa = str(payload.get("ayanamsa") or "lahiri").lower()
    lat = payload.get("latitude"); lon = payload.get("longitude")

    norm = {
        "system": "vimshottari",
        "date": date, "time": time, "tz": tz,
        "jd_tt": jd_tt, "jd_ut1": jd_ut1, "jd_ut": jd_ut,
        "ayanamsa": ayanamsa,
        "latitude": float(lat) if isinstance(lat, (int, float)) else None,
        "longitude": float(lon) if isinstance(lon, (int, float)) else None,
        "levels": depth,
        "max_levels": depth,
        "raw": payload,
    }
    return norm, warns

def _run_vimshottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    norm, warns = _normalize_vim_payload(payload)

    # 1) unified registry
    if _compute_dasha_registry:
        try:
            out = _compute_dasha_registry("vimshottari", **norm)
            if isinstance(out, dict):
                out.setdefault("meta", {}).update({"route": "vimshottari", "tz_normalized": norm["tz"]})
                if warns: out.setdefault("warnings", []).extend(warns)
                return out
        except Exception as e:
            return {"ok": False, "error": "vimshottari_registry_failed", "detail": str(e), "meta": {"tz_normalized": norm["tz"]}}

    # 2) module fallback
    if _compute_vim_module:
        try:
            out = _compute_vim_module(**norm) if callable(_compute_vim_module) else _compute_vim_module(norm)
            if isinstance(out, dict):
                out.setdefault("meta", {}).update({"route": "vimshottari", "tz_normalized": norm["tz"]})
                if warns: out.setdefault("warnings", []).extend(warns)
                return out
        except Exception as e:
            return {"ok": False, "error": "vimshottari_module_failed", "detail": str(e), "meta": {"tz_normalized": norm["tz"]}}

    return {"ok": False, "error": "vimshottari_engine_unavailable", "meta": {"tz_normalized": norm["tz"]}}

# ─────────────── Endpoints (POST) ───────────────

@vedic_api.post("/api/vedic/dasha/vimshottari")
@rate_limit(client_key, endpoint_key, RL_PREDICTIVE)
def vedic_vimshottari():
    body = request.get_json(silent=True) or {}
    res = _run_vimshottari(body)
    status = 200 if res.get("ok") else (503 if res.get("error","").endswith("unavailable") else 400)
    return jsonify(res), status

@vedic_api.post("/api/vedic/dasha/vimshottari/compute")
@rate_limit(client_key, endpoint_key, RL_PREDICTIVE)
def vedic_vimshottari_compute():
    body = request.get_json(silent=True) or {}
    res = _run_vimshottari(body)
    status = 200 if res.get("ok") else (503 if res.get("error","").endswith("unavailable") else 400)
    return jsonify(res), status

# convenience aliases
@vedic_api.post("/api/vedic/vimshottari")
@rate_limit(client_key, endpoint_key, RL_PREDICTIVE)
def vedic_vimshottari_alias1():
    body = request.get_json(silent=True) or {}
    res = _run_vimshottari(body)
    status = 200 if res.get("ok") else (503 if res.get("error","").endswith("unavailable") else 400)
    return jsonify(res), status

@vedic_api.post("/api/dasha/vimshottari")
@rate_limit(client_key, endpoint_key, RL_PREDICTIVE)
def vedic_vimshottari_alias2():
    body = request.get_json(silent=True) or {}
    res = _run_vimshottari(body)
    status = 200 if res.get("ok") else (503 if res.get("error","").endswith("unavailable") else 400)
    return jsonify(res), status
