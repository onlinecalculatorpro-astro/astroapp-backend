from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import os
from flask import Blueprint, jsonify, request

# ───────────────────────── rate limiting ─────────────────────────
from app.utils.ratelimit import rate_limit, client_key, endpoint_key

# ───────────────────────── timescales (optional) ─────────────────────────
try:
    from app.core.timescales import build_timescales
    _TIMESCALES_OK = True
except Exception:
    _TIMESCALES_OK = False
    build_timescales = None  # type: ignore

# ───────────────────────── dasha engines (registry preferred) ─────────────────────────
_compute_dasha_registry = None
_run_dasha = None
try:
    # Allow either naming style from your registry module
    from app.core.dasha_registry import compute_dasha as _compute_dasha_registry  # type: ignore
except Exception:
    try:
        from app.core.dasha_registry import run_dasha as _run_dasha  # type: ignore
    except Exception:
        _run_dasha = None

# Module fallback for Vimśottari
try:
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim_module  # type: ignore
except Exception:
    _compute_vim_module = None  # type: ignore

# ───────────────────────── blueprint ─────────────────────────
vedic_api = Blueprint("vedic_api", __name__)

# Per-endpoint rate-limit (env-override)
RL_PREDICTIVE = int(os.getenv("ASTRO_RL_PREDICTIVE_PER_MIN", "12"))

# ───────────────────────── helpers ─────────────────────────
_TZ_ALIAS = {
    "asia/patna": "Asia/Kolkata",    # legacy alias → IANA canonical
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
    Normalize request:
    - (date,time,tz) → (jd_tt, jd_ut1, jd_ut) if timescales available
    - levels/depth/max_levels → depth 1..5 (default 5)
    """
    warns: List[str] = []

    date = str(payload.get("date") or payload.get("birth_date") or "")
    time = str(payload.get("time") or payload.get("birth_time") or "12:00:00")
    tz   = _normalize_tz(str(payload.get("tz") or payload.get("place_tz") or "UTC"))

    jd_tt  = payload.get("jd_tt")
    jd_ut1 = payload.get("jd_ut1")
    jd_ut  = payload.get("jd_ut")

    if (jd_tt is None or jd_ut1 is None) and _TIMESCALES_OK and date:
        try:
            ts = build_timescales(date=date, time=time, tz=tz)  # type: ignore[operator]
            jd_tt  = float(ts.jd_tt)
            jd_ut1 = float(ts.jd_ut1)
            jd_ut  = float(ts.jd_utc)
        except Exception as e:
            warns.append(f"timescales_failed:{e!s}")

    # depth / levels
    lv = payload.get("levels", payload.get("depth", payload.get("max_levels", 5)))
    if isinstance(lv, list):
        depth = max(1, min(5, len(lv)))
    else:
        try:
            depth = max(1, min(5, int(lv)))
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

    # 1) unified registry (prefer)
    if _compute_dasha_registry is not None:
        try:
            # support both signatures: compute_dasha("vimshottari", **norm) OR compute_dasha("vimshottari", norm)
            try:
                out = _compute_dasha_registry("vimshottari", **norm)  # type: ignore[misc]
            except TypeError:
                out = _compute_dasha_registry("vimshottari", norm)   # type: ignore[misc]
            if isinstance(out, dict):
                out.setdefault("meta", {}).update({"route": "vimshottari", "tz_normalized": norm["tz"]})
                if warns: out.setdefault("warnings", []).extend(warns)
                return out
        except Exception as e:
            return {"ok": False, "error": "vimshottari_registry_failed", "detail": str(e),
                    "meta": {"tz_normalized": norm["tz"]}}

    if _run_dasha is not None:
        try:
            out = _run_dasha("vimshottari", norm)  # type: ignore[misc]
            if isinstance(out, dict):
                out.setdefault("meta", {}).update({"route": "vimshottari", "tz_normalized": norm["tz"]})
                if warns: out.setdefault("warnings", []).extend(warns)
                return out
        except Exception as e:
            return {"ok": False, "error": "vimshottari_registry_failed", "detail": str(e),
                    "meta": {"tz_normalized": norm["tz"]}}

    # 2) module fallback
    if _compute_vim_module is not None:
        try:
            out = _compute_vim_module(**norm) if callable(_compute_vim_module) else _compute_vim_module(norm)  # type: ignore[misc]
            if isinstance(out, dict):
                out.setdefault("meta", {}).update({"route": "vimshottari", "tz_normalized": norm["tz"]})
                if warns: out.setdefault("warnings", []).extend(warns)
                return out
        except Exception as e:
            return {"ok": False, "error": "vimshottari_module_failed", "detail": str(e),
                    "meta": {"tz_normalized": norm["tz"]}}

    return {"ok": False, "error": "vimshottari_engine_unavailable",
            "meta": {"tz_normalized": norm["tz"]}}

# ───────────────────────── endpoints (mounted under /api/vedic) ─────────────────────────

@vedic_api.get("/health")
def vedic_health():
    return jsonify(ok=True, vedic=True), 200

@vedic_api.post("/dasha/vimshottari")
@rate_limit(RL_PREDICTIVE, key_func=client_key, ep_key_func=endpoint_key)
def vedic_vimshottari():
    body = request.get_json(silent=True) or {}
    res = _run_vimshottari(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error","")).endswith("unavailable") else 400)
    return jsonify(res), status

# Alias
@vedic_api.post("/dasha/vimshottari/compute")
@rate_limit(RL_PREDICTIVE, key_func=client_key, ep_key_func=endpoint_key)
def vedic_vimshottari_compute():
    return vedic_vimshottari()

# Short alias
@vedic_api.post("/vimshottari")
@rate_limit(RL_PREDICTIVE, key_func=client_key, ep_key_func=endpoint_key)
def vedic_vimshottari_short():
    return vedic_vimshottari()

# Common misspelling alias (optional)
@vedic_api.post("/dasha/vimshotri")
@rate_limit(RL_PREDICTIVE, key_func=client_key, ep_key_func=endpoint_key)
def vedic_vimshotri_alias():
    return vedic_vimshottari()
