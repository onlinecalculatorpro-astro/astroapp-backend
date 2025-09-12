# app/api/western_routes.py
from __future__ import annotations

from typing import Any, Dict, Optional
import inspect
import os

from flask import Blueprint, jsonify
from app.utils.ratelimit import rate_limit

western_api = Blueprint("western_api", __name__)

# ── fixed shared bucket key ("20") for parity with Vedic ──
def fixed_key(*_a, **_k) -> str:
    return "20"

RL_WESTERN_PREDICTIVE = int(os.getenv("ASTRO_RL_WESTERN_PREDICTIVE_PER_MIN", "20"))

# ───────────────────────── validator wiring ─────────────────────────
_wval_mod = None
_WVAL_ERR: Optional[str] = None
try:
    # We don't assume specific function names; we introspect safely below.
    import app.core.western_validator as _wval_mod  # type: ignore
except Exception as _e:
    _WVAL_ERR = repr(_e)

def _list_public_callables(mod) -> list[str]:
    try:
        names = [
            n for n in dir(mod)
            if callable(getattr(mod, n, None)) and not n.startswith("_")
        ]
        names.sort()
        return names
    except Exception:
        return []

def _sig_safe(obj) -> Optional[str]:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None

def _collect_key_signatures(mod) -> Dict[str, Optional[str]]:
    """
    Try to surface signatures of commonly named normalizers if present.
    We don't mandate names; this avoids hard failures.
    """
    if not mod:
        return {}
    suspects = [
        # common patterns — only included if present
        "normalize_payload",
        "normalize_chart_payload",
        "normalize_prediction_payload",
        "normalize_ephemeris_payload",
        "normalize_western_payload",
    ]
    out: Dict[str, Optional[str]] = {}
    for name in suspects:
        fn = getattr(mod, name, None)
        if callable(fn):
            out[name] = _sig_safe(fn)
    return out

# ───────────────────────── routes (RELATIVE; mounted at /api/western) ─────────────────────────
@western_api.get("/health")
def western_health():
    """Basic health for the Western API blueprint."""
    return jsonify(ok=True, western=True), 200

@western_api.get("/diag")
def western_diag():
    """Light diagnostic surface for Western stack."""
    funcs = _list_public_callables(_wval_mod) if _wval_mod else None
    sigs = _collect_key_signatures(_wval_mod) if _wval_mod else None
    return jsonify({
        "validator_loaded": _wval_mod is not None,
        "validator_error": _WVAL_ERR,
        "validator_functions": funcs,
        "validator_signatures": sigs,
        "rl_cap_per_min": RL_WESTERN_PREDICTIVE,
        "rl_bucket_key": "20",
    }), 200

# (When you add Western compute endpoints later, mirror the Vedic pattern:
#  @western_api.post("/...") + @rate_limit(RL_WESTERN_PREDICTIVE, key_fn=fixed_key))
