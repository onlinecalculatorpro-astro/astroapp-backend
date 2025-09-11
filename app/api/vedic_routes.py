# app/api/vedic_routes.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import os
from flask import Blueprint, jsonify, request

# ── rate limiting (single key_fn) ──
from app.utils.ratelimit import rate_limit, client_key, endpoint_key

def client_endpoint_key() -> str:
    try:
        return f"{client_key()}::{endpoint_key()}"
    except Exception:
        return str(client_key())

# ── timescales (optional) ──
try:
    from app.core.timescales import build_timescales  # build_timescales(date, time, tz, dut1_seconds)
    _TIMESCALES_OK = True
except Exception:
    build_timescales = None  # type: ignore
    _TIMESCALES_OK = False

# ── dasha engines (registry preferred) ──
_compute_dasha_registry = None
_run_dasha = None
try:
    from app.core.dasha_registry import compute_dasha as _compute_dasha_registry  # type: ignore
except Exception:
    _compute_dasha_registry = None  # type: ignore
    try:
        from app.core.dasha_registry import run_dasha as _run_dasha  # type: ignore
    except Exception:
        _run_dasha = None  # type: ignore

# module fallback
try:
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim_module  # type: ignore
except Exception:
    _compute_vim_module = None  # type: ignore

vedic_api = Blueprint("vedic_api", __name__)

# per-endpoint RL
RL_VEDIC_PREDICTIVE = int(os.getenv("ASTRO_RL_VEDIC_PREDICTIVE_PER_MIN", "12"))

_TZ_ALIAS = {
    "asia/patna": "Asia/Kolkata",
    "asia/calcutta": "Asia/Kolkata",
    "ist": "Asia/Kolkata",
}

def _normalize_tz(tz: str) -> str:
    if not isinstance(tz, str):
        return "UTC"
    return _TZ_ALIAS.get(tz.strip().lower(), tz)

def _normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warns: List[str] = []

    date = str(payload.get("date") or payload.get("birth_date") or "")
    time_str = str(payload.get("time") or payload.get("birth_time") or "12:00:00")
    tz_name = _normalize_tz(str(payload.get("tz") or payload.get("place_tz") or "UTC"))

    jd_tt: Optional[float] = payload.get("jd_tt")
    jd_ut1: Optional[float] = payload.get("jd_ut1")

    if (jd_tt is None or jd_ut1 is None) and _TIMESCALES_OK and date:
        try:
            ts = build_timescales(date, time_str, tz_name, 0.0)  # type: ignore[call-arg]
            if isinstance(ts, dict):
                if ts.get("jd_tt") is not None: jd_tt = float(ts["jd_tt"])
                if ts.get("jd_ut1") is not None: jd_ut1 = float(ts["jd_ut1"])
            else:
                jd_tt = float(getattr(ts, "jd_tt"))
                jd_ut1 = float(getattr(ts, "jd_ut1"))
        except Exception as e:
            warns.append(f"timescales_failed:{e!s}")

    lv = payload.get("levels", payload.get("depth", payload.get("max_levels", 5)))
    if isinstance(lv, list):
        depth = max(1, min(5, len(lv)))
    else:
        try:
            depth = max(1, min(5, int(lv)))
        except Exception:
            depth = 5

    ayanamsa = str(payload.get("ayanamsa") or "lahiri").lower()
    lat = payload.get("latitude")
    lon = payload.get("longitude")

    norm: Dict[str, Any] = {
        "system": "vimshottari",
        "date": date,
        "time": time_str,
        "tz": tz_name,
        "jd_tt": jd_tt,
        "jd_ut1": jd_ut1,
        "ayanamsa": ayanamsa,
        "latitude": float(lat) if isinstance(lat, (int, float)) else None,
        "longitude": float(lon) if isinstance(lon, (int, float)) else None,
        "levels": depth,
        "max_levels": depth,
        "raw": payload,
    }
    return norm, warns

def _wrap_ok(out: Dict[str, Any], warns: List[str], tz_norm: str) -> Dict[str, Any]:
    out.setdefault("ok", True)
    out.setdefault("meta", {})
    out["meta"].update({"route": "vimshottari", "tz_normalized": tz_norm})
    if warns:
        out.setdefault("warnings", [])
        seen = set(map(str, out["warnings"]))
        for w in warns:
            s = str(w)
            if s not in seen:
                out["warnings"].append(s)
                seen.add(s)
    return out

def _run_vimshottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    norm, warns = _normalize_vim_payload(payload)
    tz_norm = str(norm.get("tz", "UTC"))

    if _compute_dasha_registry is not None:
        try:
            try:
                out = _compute_dasha_registry("vimshottari", **norm)  # type: ignore[misc]
            except TypeError:
                out = _compute_dasha_registry("vimshottari", norm)   # type: ignore[misc]
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm)
        except Exception as e:
            return {"ok": False, "error": "vimshottari_registry_failed", "detail": str(e),
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm}}

    if _run_dasha is not None:
        try:
            out = _run_dasha("vimshottari", norm)  # type: ignore[misc]
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm)
        except Exception as e:
            return {"ok": False, "error": "vimshottari_registry_failed", "detail": str(e),
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm}}

    if _compute_vim_module is not None:
        try:
            out = (_compute_vim_module(**norm) if callable(_compute_vim_module)
                   else _compute_vim_module(norm))  # type: ignore[misc]
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm)
        except Exception as e:
            return {"ok": False, "error": "vimshottari_module_failed", "detail": str(e),
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm}}

    return {"ok": False, "error": "vimshottari_engine_unavailable",
            "meta": {"route": "vimshottari", "tz_normalized": tz_norm}}

# ── ABSOLUTE PATHS (because main.py registers blueprint WITHOUT url_prefix) ──

@vedic_api.get("/api/vedic/health")
def vedic_health():
    return jsonify(ok=True, vedic=True), 200

@vedic_api.post("/api/vedic/dasha/vimshottari")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=client_endpoint_key)
def vedic_vimshottari():
    body = request.get_json(silent=True) or {}
    res = _run_vimshottari(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status

@vedic_api.post("/api/vedic/dasha/vimshottari/compute")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=client_endpoint_key)
def vedic_vimshottari_compute():
    return vedic_vimshottari()

@vedic_api.post("/api/vedic/vimshottari")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=client_endpoint_key)
def vedic_vimshottari_short():
    return vedic_vimshottari()

@vedic_api.post("/api/vedic/dasha/vimshotri")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=client_endpoint_key)
def vedic_vimshotri_alias():
    return vedic_vimshottari()
