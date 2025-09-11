# app/api/vedic_routes.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import inspect
import os

from flask import Blueprint, jsonify, request

# ── rate limiting (single key_fn) ──
from app.utils.ratelimit import rate_limit, client_key, endpoint_key

def client_endpoint_key() -> str:
    try:
        return f"{client_key()}::{endpoint_key()}"
    except Exception:
        return str(client_key())

# ── timescales (optional; NO jd_utc) ──
try:
    # build_timescales(date_str, time_str, tz_name, dut1_seconds) → has jd_tt/jd_ut1
    from app.core.timescales import build_timescales
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

# fallback module
try:
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim_module  # type: ignore
except Exception:
    _compute_vim_module = None  # type: ignore

vedic_api = Blueprint("vedic_api", __name__)
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

def _filter_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Only pass parameters the callable can accept (prevents TypeError: unexpected kwarg)."""
    try:
        sig = inspect.signature(func)
    except Exception:
        return kwargs  # best effort
    kept = {}
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_varkw:
        return kwargs
    allowed = set(sig.parameters.keys())
    for k, v in kwargs.items():
        if k in allowed:
            kept[k] = v
    return kept

def _normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalize inputs; fill jd_tt/jd_ut1 (if timescales available); compute depth 1..5.
    NO jd_utc anywhere.
    """
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
                if ts.get("jd_tt") is not None:  jd_tt  = float(ts["jd_tt"])
                if ts.get("jd_ut1") is not None: jd_ut1 = float(ts["jd_ut1"])
            else:
                jd_tt  = float(getattr(ts, "jd_tt"))
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

    # base kwargs (civil + optional timescales)
    norm: Dict[str, Any] = {
        "system": "vimshottari",
        "date": date,
        "time": time_str,
        "tz": tz_name,
        "ayanamsa": ayanamsa,
        "levels": depth,
        "max_levels": depth,
        "latitude": float(lat) if isinstance(lat, (int, float)) else None,
        "longitude": float(lon) if isinstance(lon, (int, float)) else None,
        "jd_tt": jd_tt,
        "jd_ut1": jd_ut1,
        "raw": payload,
    }

    # common aliases some registries expect (harmless if unused)
    norm.update({
        "tz_name": tz_name,
        "ayanamsa_key": ayanamsa,
        "birth_date": date,
        "birth_time": time_str,
        "place_tz": tz_name,
        "depth": depth,
    })
    return norm, warns

def _wrap_ok(out: Dict[str, Any], warns: List[str], tz_norm: str, branch: str) -> Dict[str, Any]:
    out.setdefault("ok", True)
    out.setdefault("meta", {})
    out["meta"].update({"route": "vimshottari", "tz_normalized": tz_norm, "branch": branch})
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

    # 1) unified registry (preferred)
    if _compute_dasha_registry is not None:
        try:
            k = _filter_kwargs(_compute_dasha_registry, norm)
            try:
                out = _compute_dasha_registry("vimshottari", **k)  # type: ignore[misc]
            except TypeError:
                out = _compute_dasha_registry("vimshottari", k)   # type: ignore[misc]
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.compute_dasha")
        except Exception as e:
            return {"ok": False, "error": "vimshottari_registry_failed", "detail": str(e),
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "registry.compute_dasha"}}

    # 2) alternate registry name
    if _run_dasha is not None:
        try:
            k = _filter_kwargs(_run_dasha, norm)
            out = _run_dasha("vimshottari", k)  # most run_dasha(kind, payload) use a single dict
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.run_dasha")
        except Exception as e:
            return {"ok": False, "error": "vimshottari_registry_failed", "detail": str(e),
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "registry.run_dasha"}}

    # 3) module fallback: be conservative (civil + ayanamsa/levels only)
    if _compute_vim_module is not None:
        try:
            civ_only_keys = ["date","time","tz","tz_name","ayanamsa","ayanamsa_key","levels","depth","max_levels","latitude","longitude"]
            civ = {k: norm[k] for k in civ_only_keys if k in norm and norm[k] is not None}
            k = _filter_kwargs(_compute_vim_module, civ)
            out = (_compute_vim_module(**k) if callable(_compute_vim_module)
                   else _compute_vim_module(k))  # type: ignore[misc]
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="module.compute_vimshottari")
        except Exception as e:
            return {"ok": False, "error": "vimshottari_module_failed", "detail": str(e),
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "module.compute_vimshottari"}}

    return {"ok": False, "error": "vimshottari_engine_unavailable",
            "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "none"}}

# ── ABSOLUTE PATHS (main.py registers this bp WITHOUT url_prefix) ──

@vedic_api.get("/api/vedic/health")
def vedic_health():
    return jsonify(ok=True, vedic=True), 200

@vedic_api.get("/api/vedic/diag")
def vedic_diag():
    def sigs(fn):
        try:
            return str(inspect.signature(fn))
        except Exception:
            return None
    return jsonify({
        "timescales_ok": _TIMESCALES_OK,
        "registry_compute_present": bool(_compute_dasha_registry),
        "registry_run_present": bool(_run_dasha),
        "module_vimshottari_present": bool(_compute_vim_module),
        "registry_compute_sig": sigs(_compute_dasha_registry) if _compute_dasha_registry else None,
        "registry_run_sig": sigs(_run_dasha) if _run_dasha else None,
    }), 200

@vedic_api.post("/api/vedic/dasha/vimshottari")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=client_endpoint_key)
def vedic_vimshottari():
    body = request.get_json(silent=True) or {}
    try:
        res = _run_vimshottari(body)
        status = 200 if res.get("ok") else (503 if str(res.get("error","")).endswith("unavailable") else 400)
        return jsonify(res), status
    except Exception as e:
        # Ensure JSON detail reaches the client for quick triage
        return jsonify(ok=False, error="vedic_internal_error", detail=str(e)), 500

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
