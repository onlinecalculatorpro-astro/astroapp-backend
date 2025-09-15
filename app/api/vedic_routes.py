# app/api/vedic_routes.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
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
    """Shared bucket key ('20') used by all dasha calls."""
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

# Optional module fallbacks
try:
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim_module  # type: ignore
except Exception:
    _compute_vim_module = None  # type: ignore

try:
    from app.core.ashtottari_dasha import compute_ashtottari as _compute_ashto_module  # type: ignore
except Exception:
    _compute_ashto_module = None  # type: ignore

try:
    from app.core.yogini_dasha import compute_yogini as _compute_yogini_module  # type: ignore
except Exception:
    _compute_yogini_module = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _env_dut1_seconds() -> float:
    try:
        return float(os.environ.get("ASTRO_DUT1_BROADCAST",
                                    os.environ.get("ASTRO_DUT1", "0.0")) or 0.0)
    except Exception:
        return 0.0


def _wrap_ok(out: Dict[str, Any], warns: List[str], tz_norm: str, branch: str, *, route_name: str) -> Dict[str, Any]:
    out.setdefault("ok", True)
    out.setdefault("meta", {})
    out["meta"].update({"route": route_name, "tz_normalized": tz_norm, "branch": branch})
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


# ---- civic payload builders ---------------------------------------------------
def _build_civic_payload_vim(original: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare payload for compute_vimshottari(payload_dict).

    Accept either civil (date,time,tz) or a direct birth_jd_tt. Prefer the *original*
    birth_jd_tt if the client sent it; otherwise use norm.jd_tt if available.
    """
    civ: Dict[str, Any] = {}

    if isinstance(original.get("birth_jd_tt"), (int, float)):
        civ["birth_jd_tt"] = float(original["birth_jd_tt"])
    elif isinstance(norm.get("jd_tt"), (int, float)):
        civ["birth_jd_tt"] = float(norm["jd_tt"])

    for k_src, k_dst in (("date", "date"), ("time", "time"), ("tz", "tz")):
        v = norm.get(k_src)
        if isinstance(v, str) and v.strip():
            civ[k_dst] = v.strip()

    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]

    civ["levels"] = _levels_from(norm)

    for k in ("span_years", "end_jd_tt", "year_days", "query_jd_tt", "q_date", "q_time", "q_tz", "flatten_level"):
        if k in original and original[k] is not None:
            civ[k] = original[k]

    for k in ("timescales", "jd_tt", "jd_ut1", "dut1_seconds"):
        civ.pop(k, None)

    return civ


def _build_civic_payload_ashto(original: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare payload for compute_ashtottari(payload_dict).
    Accepts either jd_tt or civil date/time/tz; forwards ayanamsa, levels, and optional knobs.
    """
    civ: Dict[str, Any] = {}

    if isinstance(original.get("jd_tt"), (int, float)):
        civ["jd_tt"] = float(original["jd_tt"])
    elif isinstance(norm.get("jd_tt"), (int, float)):
        civ["jd_tt"] = float(norm["jd_tt"])

    for k in ("date", "time", "tz"):
        v = norm.get(k)
        if isinstance(v, str) and v.strip():
            civ[k] = v.strip()

    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]
    civ["levels"] = _levels_from(norm)

    for k in ("start_mode", "year_days", "limit_jd_tt", "moon_nirayana_deg", "compact", "include_spans"):
        if k in original and original[k] is not None:
            civ[k] = original[k]

    for k in ("timescales", "jd_ut1", "dut1_seconds"):
        civ.pop(k, None)

    return civ


def _build_civic_payload_yogini(original: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare payload for compute_yogini(payload_dict).
    Accepts either jd_tt or civil date/time/tz; forwards ayanamsa, levels, and optional knobs.
    """
    civ: Dict[str, Any] = {}

    if isinstance(original.get("jd_tt"), (int, float)):
        civ["jd_tt"] = float(original["jd_tt"])
    elif isinstance(norm.get("jd_tt"), (int, float)):
        civ["jd_tt"] = float(norm["jd_tt"])

    for k in ("date", "time", "tz"):
        v = norm.get(k)
        if isinstance(v, str) and v.strip():
            civ[k] = v.strip()

    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]
    civ["levels"] = _levels_from(norm)

    for k in ("start_mode", "year_days", "limit_jd_tt", "start_lord", "moon_nirayana_deg"):
        if k in original and original[k] is not None:
            civ[k] = original[k]

    for k in ("timescales", "jd_ut1", "dut1_seconds"):
        civ.pop(k, None)

    return civ


# ---- tiny module-call shims ---------------------------------------------------
def _call_single_param_or_kwargs(fn, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)  # type: ignore[arg-type]
        params = list(sig.parameters.values())
    except Exception:
        return fn(payload)  # type: ignore[misc]

    if len(params) == 1 and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return fn(payload)  # type: ignore[misc]
    else:
        return fn(**payload)  # type: ignore[misc]


def _coerce_tree_like(res: Dict[str, Any], *, scheme: str) -> Dict[str, Any]:
    """
    Ensure a consistent tree-like shape. If the module returns 'nested' (forest),
    wrap it into a single envelope node under 'tree'.
    """
    if isinstance(res, dict) and "tree" in res:
        return res
    if isinstance(res, dict) and isinstance(res.get("nested"), list):
        nodes = res["nested"]
        if nodes:
            s0 = min(float(n.get("start_jd_tt", 0.0)) for n in nodes)
            e1 = max(float(n.get("end_jd_tt", 0.0)) for n in nodes)
        else:
            s0, e1 = 0.0, 0.0
        res = dict(res)
        res["tree"] = {
            "level": 0,
            "lord": None,
            "label": scheme,
            "start_jd_tt": s0,
            "end_jd_tt": e1,
            "children": nodes,
        }
        return res
    return res


# --- helper for registry call that may/may not accept depth -------------------
def _call_registry_compute(system: str, norm: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], str]:
    """
    Try (system, payload, depth=...), then positional 3-arg, then 2-arg.
    Returns (result or None, branch_name). Any non-TypeError exceptions fall through.
    """
    if _compute_dasha_registry is None:
        return None, "registry.absent"

    L = _levels_from(norm)

    # 1) depth as keyword
    try:
        out = _compute_dasha_registry(system, norm, depth=L)  # type: ignore[misc]
        return out, "registry.compute_dasha(depth_kw)"
    except TypeError:
        pass
    except Exception:
        # fall through to fallback paths
        return None, "registry.compute_dasha(error_depth_kw)"

    # 2) depth as positional
    try:
        out = _compute_dasha_registry(system, norm, L)  # type: ignore[misc]
        return out, "registry.compute_dasha(depth_pos)"
    except TypeError:
        pass
    except Exception:
        return None, "registry.compute_dasha(error_depth_pos)"

    # 3) no depth supported
    try:
        out = _compute_dasha_registry(system, norm)  # type: ignore[misc]
        return out, "registry.compute_dasha(no_depth)"
    except Exception:
        return None, "registry.compute_dasha(failed)"


# ──────────────────────────────────────────────────────────────────────────────
# Core runners
# ──────────────────────────────────────────────────────────────────────────────
def _run_vimshottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    if normalize_vim_payload is None:
        return {
            "ok": False,
            "error": "validator_unavailable",
            "detail": f"app.core.vedic_validator.normalize_vim_payload import failed: {_VALIDATOR_IMPORT_ERR}",
        }

    norm, warns, tz_norm = normalize_vim_payload(payload)  # type: ignore[misc]

    if "dut1_seconds" not in norm or norm["dut1_seconds"] is None:
        norm["dut1_seconds"] = _env_dut1_seconds()

    # 1) Registry (preferred) — signature-aware
    if _compute_dasha_registry is not None:
        out, branch = _call_registry_compute("vimshottari", norm)
        if isinstance(out, dict):
            return _wrap_ok(out, warns, tz_norm, branch=branch, route_name="vimshottari")

    # 2) Legacy registry alias (if present)
    if _run_dasha is not None:
        try:
            out = _run_dasha("vimshottari", norm)
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.run_dasha", route_name="vimshottari")
        except Exception as e:
            return {
                "ok": False,
                "error": "vimshottari_registry_failed",
                "detail": str(e),
                "meta": {"route": "vimshottari", "tz_normalized": tz_norm, "branch": "registry.run_dasha"},
            }

    # 3) Module fallback
    if _compute_vim_module is not None:
        try:
            civ = _build_civic_payload_vim(payload, norm)
            out = _call_single_param_or_kwargs(_compute_vim_module, civ)
            if isinstance(out, dict):
                out = _coerce_tree_like(out, scheme="vimshottari")
                return _wrap_ok(out, warns, tz_norm, branch="module.compute_vimshottari", route_name="vimshottari")
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


def _run_ashtottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    if normalize_vim_payload is None:
        return {
            "ok": False,
            "error": "validator_unavailable",
            "detail": f"app.core.vedic_validator.normalize_vim_payload import failed: {_VALIDATOR_IMPORT_ERR}",
        }

    norm, warns, tz_norm = normalize_vim_payload(payload)  # type: ignore[misc]

    if "dut1_seconds" not in norm or norm["dut1_seconds"] is None:
        norm["dut1_seconds"] = _env_dut1_seconds()

    # 1) Registry (preferred) — signature-aware
    if _compute_dasha_registry is not None:
        out, branch = _call_registry_compute("ashtottari", norm)
        if isinstance(out, dict):
            return _wrap_ok(out, warns, tz_norm, branch=branch, route_name="ashtottari")

    # 2) Legacy registry alias (rare)
    if _run_dasha is not None:
        try:
            out = _run_dasha("ashtottari", norm)
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.run_dasha", route_name="ashtottari")
        except Exception as e:
            return {
                "ok": False,
                "error": "ashtottari_registry_failed",
                "detail": str(e),
                "meta": {"route": "ashtottari", "tz_normalized": tz_norm, "branch": "registry.run_dasha"},
            }

    # 3) Module fallback (if available)
    if _compute_ashto_module is not None:
        try:
            civ = _build_civic_payload_ashto(payload, norm)
            out = _call_single_param_or_kwargs(_compute_ashto_module, civ)
            if isinstance(out, dict):
                out = _coerce_tree_like(out, scheme="ashtottari")
                return _wrap_ok(out, warns, tz_norm, branch="module.compute_ashtottari", route_name="ashtottari")
            return {
                "ok": False,
                "error": "ashtottari_module_invalid_return",
                "detail": f"Expected dict, got {type(out).__name__}",
                "meta": {"route": "ashtottari", "tz_normalized": tz_norm, "branch": "module.compute_ashtottari"},
            }
        except Exception as e:
            return {
                "ok": False,
                "error": "ashtottari_module_failed",
                "detail": str(e),
                "meta": {"route": "ashtottari", "tz_normalized": tz_norm, "branch": "module.compute_ashtottari"},
            }

    # 4) No engine available
    return {
        "ok": False,
        "error": "ashtottari_engine_unavailable",
        "meta": {"route": "ashtottari", "tz_normalized": tz_norm, "branch": "none"},
    }


def _run_yogini(payload: Dict[str, Any]) -> Dict[str, Any]:
    if normalize_vim_payload is None:
        return {
            "ok": False,
            "error": "validator_unavailable",
            "detail": f"app.core.vedic_validator.normalize_vim_payload import failed: {_VALIDATOR_IMPORT_ERR}",
        }

    norm, warns, tz_norm = normalize_vim_payload(payload)  # type: ignore[misc]

    if "dut1_seconds" not in norm or norm["dut1_seconds"] is None:
        norm["dut1_seconds"] = _env_dut1_seconds()

    # 1) Registry (preferred) — signature-aware
    if _compute_dasha_registry is not None:
        out, branch = _call_registry_compute("yogini", norm)
        if isinstance(out, dict):
            return _wrap_ok(out, warns, tz_norm, branch=branch, route_name="yogini")

    # 2) Legacy registry alias (rare)
    if _run_dasha is not None:
        try:
            out = _run_dasha("yogini", norm)
            if isinstance(out, dict):
                return _wrap_ok(out, warns, tz_norm, branch="registry.run_dasha", route_name="yogini")
        except Exception as e:
            return {
                "ok": False,
                "error": "yogini_registry_failed",
                "detail": str(e),
                "meta": {"route": "yogini", "tz_normalized": tz_norm, "branch": "registry.run_dasha"},
            }

    # 3) Module fallback (if available)
    if _compute_yogini_module is not None:
        try:
            civ = _build_civic_payload_yogini(payload, norm)
            out = _call_single_param_or_kwargs(_compute_yogini_module, civ)
            if isinstance(out, dict):
                out = _coerce_tree_like(out, scheme="yogini")
                return _wrap_ok(out, warns, tz_norm, branch="module.compute_yogini", route_name="yogini")
            return {
                "ok": False,
                "error": "yogini_module_invalid_return",
                "detail": f"Expected dict, got {type(out).__name__}",
                "meta": {"route": "yogini", "tz_normalized": tz_norm, "branch": "module.compute_yogini"},
            }
        except Exception as e:
            return {
                "ok": False,
                "error": "yogini_module_failed",
                "detail": str(e),
                "meta": {"route": "yogini", "tz_normalized": tz_norm, "branch": "module.compute_yogini"},
            }

    # 4) No engine available
    return {
        "ok": False,
        "error": "yogini_engine_unavailable",
        "meta": {"route": "yogini", "tz_normalized": tz_norm, "branch": "none"},
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
        "module_ashtottari_present": bool(_compute_ashto_module),
        "module_ashtottari_sig": sigs(_compute_ashto_module) if _compute_ashto_module else None,
        "module_yogini_present": bool(_compute_yogini_module),
        "module_yogini_sig": sigs(_compute_yogini_module) if _compute_yogini_module else None,
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


# ASCII alias + Unicode canonical for Aṣṭottarī
@vedic_api.post("/dasha/ashtottari")
@vedic_api.post("/dasha/Aṣṭottarī")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_ashtottari():
    body = request.get_json(silent=True) or {}
    res = _run_ashtottari(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status


# ASCII alias + Unicode canonical for Yoginī
@vedic_api.post("/dasha/yogini")
@vedic_api.post("/dasha/Yoginī")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_yogini():
    body = request.get_json(silent=True) or {}
    res = _run_yogini(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status
