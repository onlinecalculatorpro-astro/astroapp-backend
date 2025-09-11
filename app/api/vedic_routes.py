# app/api/vedic_routes.py
from __future__ import annotations
from typing import Any, Dict, Optional
import inspect
import os

from flask import Blueprint, jsonify, request
from app.utils.ratelimit import rate_limit  # fixed bucket

# ── Fixed shared bucket key (all callers share one bucket) ──
def fixed_key(*_a, **_k) -> str:
    return "20"

# ── Validator (no jd_utc inside) ──
try:
    from app.core.vedic_validator import normalize_vim_payload  # type: ignore
except Exception as _e:
    normalize_vim_payload = None  # type: ignore
    _VALIDATOR_IMPORT_ERR = repr(_e)
else:
    _VALIDATOR_IMPORT_ERR = None

# ── Dasha engines: registry (preferred) + module fallback ──
_compute_dasha_registry = None
_run_dasha = None
_dasha_registry_mod = None
try:
    import app.core.dasha_registry as _dasha_registry_mod  # module handle (for diagnostics)
    from app.core.dasha_registry import compute_dasha as _compute_dasha_registry  # type: ignore
except Exception:
    _compute_dasha_registry = None
    try:
        from app.core.dasha_registry import run_dasha as _run_dasha  # type: ignore
    except Exception:
        _run_dasha = None

try:
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim_module  # type: ignore
except Exception:
    _compute_vim_module = None  # type: ignore

# ── Blueprint & RL cap ──
vedic_api = Blueprint("vedic_api", __name__)
RL_VEDIC_PREDICTIVE = int(os.getenv("ASTRO_RL_VEDIC_PREDICTIVE_PER_MIN", "20"))

# ── Helpers ──
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

def _env_dut1_seconds() -> float:
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST",
                       os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0

def _run_vimshottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    if normalize_vim_payload is None:
        return {
            "ok": False,
            "error": "validator_unavailable",
            "detail": f"app.core.vedic_validator.normalize_vim_payload import failed: {_VALIDATOR_IMPORT_ERR}",
        }

    # Normalize; NO jd_utc anywhere
    norm, warns, tz_norm = normalize_vim_payload(payload)  # type: ignore[misc]

    # Ensure registry gets dut1_seconds it expects (even if it ignores it)
    if "dut1_seconds" not in norm or norm["dut1_seconds"] is None:
        norm["dut1_seconds"] = _env_dut1_seconds()

    # 1) Registry (preferred)
    if _compute_dasha_registry is not None:
        try:
            depth_val = int(norm.get("levels") or norm.get("depth") or norm.get("max_levels") or 5)
            out = _compute_dasha_registry("vimshottari", norm, depth=depth_val)
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.compute_dasha")
        except Exception as e:
            # If the registry is compiled against a 3-arg build_timescales, it throws this exact error.
            msg = str(e)
            if "build_timescales() missing 1 required positional argument: 'dut1_seconds'" in msg and _compute_vim_module:
                try:
                    civ_keys = ["date","time","tz","tz_name","ayanamsa","ayanamsa_key",
                                "levels","depth","max_levels","latitude","longitude"]
                    civ = {k: norm[k] for k in civ_keys if k in norm and norm[k] is not None}
                    out = (_compute_vim_module(**civ) if callable(_compute_vim_module) else _compute_vim_module(civ))  # type: ignore[misc]
                    if isinstance(out, dict):
                        return _wrap_ok(out, warns, tz_norm, branch="fallback.module.compute_vimshottari")
                except Exception as e2:
                    return {"ok": False, "error": "vimshottari_module_failed", "detail": str(e2),
                            "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "fallback.module"}}
            # Generic registry error
            return {"ok": False, "error": "vimshottari_registry_failed", "detail": msg,
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "registry.compute_dasha"}}

    # 2) Alternate registry name
    if _run_dasha is not None:
        try:
            out = _run_dasha("vimshottari", norm)
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.run_dasha")
        except Exception as e:
            return {"ok": False, "error": "vimshottari_registry_failed", "detail": str(e),
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "registry.run_dasha"}}

    # 3) Module fallback
    if _compute_vim_module is not None:
        try:
            civ_keys = ["date","time","tz","tz_name","ayanamsa","ayanamsa_key",
                        "levels","depth","max_levels","latitude","longitude"]
            civ = {k: norm[k] for k in civ_keys if k in norm and norm[k] is not None}
            out = (_compute_vim_module(**civ) if callable(_compute_vim_module) else _compute_vim_module(civ))  # type: ignore[misc]
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="module.compute_vimshottari")
        except Exception as e:
            return {"ok": False, "error": "vimshottari_module_failed", "detail": str(e),
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "module.compute_vimshottari"}}

    return {"ok": False, "error": "vimshottari_engine_unavailable",
            "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "none"}}

# ── Absolute paths (no url_prefix in main.py) ──
@vedic_api.get("/api/vedic/health")
def vedic_health():
    return jsonify(ok=True, vedic=True), 200

@vedic_api.get("/api/vedic/diag")
def vedic_diag():
    def sigs(fn):
        try: return str(inspect.signature(fn))
        except Exception: return None
    return jsonify({
        "validator_loaded": normalize_vim_payload is not None,
        "validator_error": _VALIDATOR_IMPORT_ERR,
        "registry_compute_present": bool(_compute_dasha_registry),
        "registry_compute_sig": sigs(_compute_dasha_registry) if _compute_dasha_registry else None,
        "registry_module_loaded": _dasha_registry_mod is not None,
        "module_vimshottari_present": bool(_compute_vim_module),
        "rl_cap_per_min": RL_VEDIC_PREDICTIVE,
        "rl_bucket_key": "20",
        "dut1_seconds_env": _env_dut1_seconds(),
    }), 200

@vedic_api.post("/api/vedic/dasha/vimshottari")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)  # shared bucket "20"
def vedic_vimshottari():
    body = request.get_json(silent=True) or {}
    try:
        res = _run_vimshottari(body)
        status = 200 if res.get("ok") else (503 if str(res.get("error","")).endswith("unavailable") else 400)
        return jsonify(res), status
    except Exception as e:
        return jsonify(ok=False, error="vedic_internal_error", detail=str(e)), 500
