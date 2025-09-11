# app/api/vedic_routes.py
from __future__ import annotations
from typing import Any, Dict, Optional
import inspect
import os

from flask import Blueprint, jsonify, request as _flask_request

# ── rate limiting (single key_fn) ──
from app.utils.ratelimit import rate_limit, client_key, endpoint_key

def _call_key(fn):
    """Call key function that may be defined as fn() or fn(request)."""
    try:
        return fn(_flask_request)   # signature: fn(request)
    except TypeError:
        return fn()                 # signature: fn()
    except Exception:
        return None

def client_endpoint_key(*_args, **_kwargs) -> str:
    """
    Accepts positional args so it works with decorators that pass (request).
    Falls back gracefully if inner key fns raise.
    """
    ck = _call_key(client_key)
    ek = _call_key(endpoint_key)
    if ck is None:
        try:
            ck = _flask_request.remote_addr or "anon"
        except Exception:
            ck = "anon"
    if ek is None:
        try:
            ek = _flask_request.path or "unknown"
        except Exception:
            ek = "unknown"
    return f"{ck}::{ek}"

# ── payload normalization (from core; NO jd_utc inside) ──
try:
    from app.core.vedic_validator import normalize_vim_payload  # type: ignore
except Exception as _e:
    normalize_vim_payload = None  # type: ignore
    _VALIDATOR_IMPORT_ERR = repr(_e)
else:
    _VALIDATOR_IMPORT_ERR = None

# ── dasha engines (registry preferred; module fallback) ──
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

try:
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim_module  # type: ignore
except Exception:
    _compute_vim_module = None  # type: ignore

# ── blueprint & rate limits ──
vedic_api = Blueprint("vedic_api", __name__)
RL_VEDIC_PREDICTIVE = int(os.getenv("ASTRO_RL_VEDIC_PREDICTIVE_PER_MIN", "12"))

# ── helpers ──
def _wrap_ok(out: Dict[str, Any], warns: list[str], tz_norm: str, branch: str) -> Dict[str, Any]:
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
    # Validator must be available
    if normalize_vim_payload is None:
        return {
            "ok": False,
            "error": "validator_unavailable",
            "detail": f"app.core.vedic_validator.normalize_vim_payload import failed: {_VALIDATOR_IMPORT_ERR}",
        }

    # normalize_vim_payload → (norm, warns, tz_norm)
    norm, warns, tz_norm = normalize_vim_payload(payload)  # type: ignore[misc]

    # 1) Unified registry (preferred) — signature:
    #    compute_dasha(system: str, payload: Dict[str, Any], *, depth: int | None = None)
    if _compute_dasha_registry is not None:
        try:
            depth_val = int(norm.get("levels") or norm.get("depth") or norm.get("max_levels") or 5)
            out = _compute_dasha_registry("vimshottari", norm, depth=depth_val)
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.compute_dasha")
        except Exception as e:
            return {
                "ok": False,
                "error": "vimshottari_registry_failed",
                "detail": str(e),
                "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "registry.compute_dasha"},
            }

    # 2) Alternate registry name (if present)
    if _run_dasha is not None:
        try:
            out = _run_dasha("vimshottari", norm)
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.run_dasha")
        except Exception as e:
            return {
                "ok": False,
                "error": "vimshottari_registry_failed",
                "detail": str(e),
                "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "registry.run_dasha"},
            }

    # 3) Module fallback (conservative civil kwargs only)
    if _compute_vim_module is not None:
        try:
            civ_keys = [
                "date", "time", "tz", "tz_name", "ayanamsa", "ayanamsa_key",
                "levels", "depth", "max_levels", "latitude", "longitude"
            ]
            civ = {k: norm[k] for k in civ_keys if k in norm and norm[k] is not None}
            out = (_compute_vim_module(**civ) if callable(_compute_vim_module) else _compute_vim_module(civ))  # type: ignore[misc]
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="module.compute_vimshottari")
        except Exception as e:
            return {
                "ok": False,
                "error": "vimshottari_module_failed",
                "detail": str(e),
                "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "module.compute_vimshottari"},
            }

    # No engine available
    return {
        "ok": False,
        "error": "vimshottari_engine_unavailable",
        "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "none"},
    }

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
        "validator_loaded": normalize_vim_payload is not None,
        "validator_error": _VALIDATOR_IMPORT_ERR,
        "registry_compute_present": bool(_compute_dasha_registry),
        "registry_compute_sig": sigs(_compute_dasha_registry) if _compute_dasha_registry else None,
        "registry_run_present": bool(_run_dasha),
        "registry_run_sig": sigs(_run_dasha) if _run_dasha else None,
        "module_vimshottari_present": bool(_compute_vim_module),
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
