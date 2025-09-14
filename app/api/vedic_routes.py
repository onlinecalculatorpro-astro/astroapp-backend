# app/api/vedic_routes.py
from __future__ import annotations

from typing import Any, Dict, List
import inspect
import os

from flask import Blueprint, jsonify, request
from app.utils.ratelimit import rate_limit  # fixed shared bucket

# ──────────────────────────────────────────────────────────────────────────────
# Blueprint & rate limits
# ──────────────────────────────────────────────────────────────────────────────
vedic_api = Blueprint("vedic_api", __name__)
RL_VEDIC_PREDICTIVE = int(os.getenv("ASTRO_RL_VEDIC_PREDICTIVE_PER_MIN", "20"))

def fixed_key(*_a, **_k) -> str:
    """Shared bucket key ('20') used by all Vimśottarī calls."""
    return "20"


# ──────────────────────────────────────────────────────────────────────────────
# Validator (NO jd_utc)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.vedic_validator import normalize_vim_payload  # type: ignore
except Exception as _e:
    normalize_vim_payload = None  # type: ignore
    _VALIDATOR_IMPORT_ERR = repr(_e)
else:
    _VALIDATOR_IMPORT_ERR = None


# ──────────────────────────────────────────────────────────────────────────────
# Engines: registry (preferred) + module fallback
# ──────────────────────────────────────────────────────────────────────────────
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
    # Preferred module surface: compute_vimshottari(payload: Dict[str, Any]) -> Dict[str, Any]
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim_module  # type: ignore
except Exception:
    _compute_vim_module = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _env_dut1_seconds() -> float:
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST",
                                    os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0


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


def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _levels_from(norm: Dict[str, Any]) -> int:
    """Clamp requested depth to [1..5]."""
    L = _coerce_int(norm.get("levels", norm.get("depth", norm.get("max_levels", 5))), 5)
    return max(1, min(5, L))


def _build_civic_payload(original: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare payload for compute_vimshottari(payload_dict).

    Accept either civil (date,time,tz) or a direct birth_jd_tt. Prefer the *original*
    birth_jd_tt if the client sent it; otherwise use norm.jd_tt if available.
    """
    civ: Dict[str, Any] = {}

    # Prefer client-provided birth_jd_tt; else fallback to normalized jd_tt (if computed).
    if isinstance(original.get("birth_jd_tt"), (int, float)):
        civ["birth_jd_tt"] = float(original["birth_jd_tt"])
    elif isinstance(norm.get("jd_tt"), (int, float)):
        civ["birth_jd_tt"] = float(norm["jd_tt"])

    # Civil triplet (strings only)
    for k_src, k_dst in (("date", "date"), ("time", "time"), ("tz", "tz")):
        v = norm.get(k_src)
        if isinstance(v, str) and v.strip():
            civ[k_dst] = v.strip()

    # Ayanamsa (float or str)
    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]

    # Levels (clamped)
    civ["levels"] = _levels_from(norm)

    # Optional extras (from the original body)
    for k in (
        "span_years", "end_jd_tt", "year_days",
        "query_jd_tt", "q_date", "q_time", "q_tz",
        "flatten_level",
    ):
        if k in original and original[k] is not None:
            civ[k] = original[k]

    # Never pass validator internals
    for k in ("timescales", "jd_tt", "jd_ut1", "dut1_seconds"):
        civ.pop(k, None)

    return civ


def _call_vim_module(civ: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the module function with signature awareness:
      - If it takes a single parameter (payload), call positionally.
      - Else, call with **kwargs.
    """
    if _compute_vim_module is None:
        raise RuntimeError("vimshottari module not available")

    try:
        sig = inspect.signature(_compute_vim_module)  # type: ignore[arg-type]
        params = list(sig.parameters.values())
    except Exception:
        # If we cannot inspect, use the canonical/expected API (single dict positional)
        return _compute_vim_module(civ)  # type: ignore[misc]

    if len(params) == 1 and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return _compute_vim_module(civ)  # type: ignore[misc]
    else:
        return _compute_vim_module(**civ)  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────
def _run_vimshottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Validator required
    if normalize_vim_payload is None:
        return {
            "ok": False,
            "error": "validator_unavailable",
            "detail": f"app.core.vedic_validator.normalize_vim_payload import failed: {_VALIDATOR_IMPORT_ERR}",
        }

    # Normalize (may compute jd_tt/jd_ut1; NO jd_utc). Also adds common alias keys.
    norm, warns, tz_norm = normalize_vim_payload(payload)  # type: ignore[misc]

    # Ensure dut1_seconds for any registry code that may call build_timescales(...)
    if "dut1_seconds" not in norm or norm["dut1_seconds"] is None:
        norm["dut1_seconds"] = _env_dut1_seconds()

    # 1) Registry (preferred)
    if _compute_dasha_registry is not None:
        try:
            depth_val = _levels_from(norm)
            out = _compute_dasha_registry("vimshottari", norm, depth=depth_val)
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.compute_dasha")
        except Exception as e:
            msg = str(e)
            # Allow fallback if a known 3-arg build_timescales signature is the problem
            if "build_timescales() missing 1 required positional argument: 'dut1_seconds'" not in msg:
                return {
                    "ok": False,
                    "error": "vimshottari_registry_failed",
                    "detail": msg,
                    "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "registry.compute_dasha"},
                }

    # 2) Legacy registry alias (if present)
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

    # 3) Module fallback (clean civic payload; signature-aware call)
    if _compute_vim_module is not None:
        try:
            civ = _build_civic_payload(payload, norm)
            out = _call_vim_module(civ)
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="module.compute_vimshottari")
            return {
                "ok": False,
                "error": "vimshottari_module_invalid_return",
                "detail": f"Expected dict, got {type(out).__name__}",
                "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "module.compute_vimshottari"},
            }
        except Exception as e:
            return {
                "ok": False,
                "error": "vimshottari_module_failed",
                "detail": str(e),
                "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "module.compute_vimshottari"},
            }

    # 4) No engine available
    return {
        "ok": False,
        "error": "vimshottari_engine_unavailable",
        "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "none"},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Routes (mounted at /api/vedic)
# ──────────────────────────────────────────────────────────────────────────────
@vedic_api.get("/health")
def vedic_health():
    return jsonify(ok=True, vedic=True), 200


@vedic_api.get("/diag")
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
        "module_vimshottari_sig": sigs(_compute_vim_module) if _compute_vim_module else None,
        "rl_cap_per_min": RL_VEDIC_PREDICTIVE,
        "rl_bucket_key": "20",
        "dut1_seconds_env": _env_dut1_seconds(),
    }), 200


@vedic_api.post("/dasha/vimshottari")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)  # shared bucket "20"
def vedic_vimshottari():
    body = request.get_json(silent=True) or {}
    res = _run_vimshottari(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status
