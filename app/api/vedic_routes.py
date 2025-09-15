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
# Engines: central registry (preferred) + direct module fallbacks
# ──────────────────────────────────────────────────────────────────────────────
_compute_dasha_registry = None
_available_schemes_fn = None
try:
    from app.core.dasha_registry import compute_dasha as _compute_dasha_registry  # type: ignore
    from app.core.dasha_registry import available_schemes as _available_schemes_fn  # type: ignore
except Exception:
    _compute_dasha_registry = None  # type: ignore
    _available_schemes_fn = None  # type: ignore

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

try:
    from app.core.chara_dasha import compute_chara_dasha as _compute_chara_module  # type: ignore
except Exception:
    _compute_chara_module = None  # type: ignore

try:
    from app.core.kala_chakra_dasha import compute_kalachakra_dasha as _compute_kcd_module  # type: ignore
except Exception:
    _compute_kcd_module = None  # type: ignore


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


def _add_levels_and_limit(payload: Dict[str, Any], norm: Dict[str, Any]) -> None:
    payload["levels"] = _levels_from(norm)
    if "limit_jd_tt" in norm and norm["limit_jd_tt"] is not None:
        payload["limit_jd_tt"] = norm["limit_jd_tt"]


# ---- civic payload builders ---------------------------------------------------
def _pick_times(norm: Dict[str, Any], original: Dict[str, Any]) -> Dict[str, Any]:
    civ: Dict[str, Any] = {}
    # Prefer direct jd_tt if given
    if isinstance(original.get("jd_tt"), (int, float)):
        civ["jd_tt"] = float(original["jd_tt"])
    elif isinstance(norm.get("jd_tt"), (int, float)):
        civ["jd_tt"] = float(norm["jd_tt"])
    # Pass UT1 when present (helps asc calc)
    if isinstance(original.get("jd_ut1"), (int, float)):
        civ["jd_ut1"] = float(original["jd_ut1"])
    elif isinstance(norm.get("jd_ut1"), (int, float)):
        civ["jd_ut1"] = float(norm["jd_ut1"])
    # Civil strings
    for k in ("date", "time", "tz"):
        v = norm.get(k)
        if isinstance(v, str) and v.strip():
            civ[k] = v.strip()
    return civ


def _build_civic_payload_vim(original: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    civ: Dict[str, Any] = {}
    # legacy engine expects birth_jd_tt OR civil
    if isinstance(original.get("birth_jd_tt"), (int, float)):
        civ["birth_jd_tt"] = float(original["birth_jd_tt"])
    elif isinstance(norm.get("jd_tt"), (int, float)):
        civ["birth_jd_tt"] = float(norm["jd_tt"])
    for k in ("date", "time", "tz"):
        v = norm.get(k)
        if isinstance(v, str) and v.strip():
            civ[k] = v.strip()
    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]
    _add_levels_and_limit(civ, norm)
    for k in ("span_years", "end_jd_tt", "year_days", "query_jd_tt", "q_date", "q_time", "q_tz", "flatten_level"):
        if k in original and original[k] is not None:
            civ[k] = original[k]
    return civ


def _build_civic_payload_ashto(original: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    civ = _pick_times(norm, original)
    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]
    _add_levels_and_limit(civ, norm)
    for k in ("start_mode", "year_days", "moon_nirayana_deg", "compact", "include_spans"):
        if k in original and original[k] is not None:
            civ[k] = original[k]
    return civ


def _build_civic_payload_yogini(original: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    civ = _pick_times(norm, original)
    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]
    _add_levels_and_limit(civ, norm)
    for k in ("start_mode", "year_days", "start_lord", "moon_nirayana_deg"):
        if k in original and original[k] is not None:
            civ[k] = original[k]
    return civ


def _build_civic_payload_chara(original: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    civ = _pick_times(norm, original)

    # site/asc inputs (from original body)
    for k in ("latitude", "longitude", "asc_sidereal_deg", "asc_tropical_deg"):
        if isinstance(original.get(k), (int, float)):
            civ[k] = float(original[k])

    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]
    _add_levels_and_limit(civ, norm)

    # Chara-specific knobs
    for k in (
        "start_from", "include_rahu_in_karakas", "direction_mode",
        "year_days", "balance_years", "balance_fraction",
        "override_start_sign_index", "planet_longitudes_sidereal"
    ):
        if k in original and original[k] is not None:
            civ[k] = original[k]
    return civ


def _build_civic_payload_kcd(original: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare payload for compute_kalachakra_dasha(payload_dict).
    Requires a kcd_table or kcd_preset supplied by the client.
    """
    civ = _pick_times(norm, original)

    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]
    _add_levels_and_limit(civ, norm)

    # KCD-specific settings
    for k in ("kcd_table", "kcd_preset", "use_demo_kcd_table", "override_start_sign_index",
              "year_days", "balance_years", "balance_fraction", "planet_longitudes_sidereal"):
        if k in original and original[k] is not None:
            civ[k] = original[k]
    return civ


# ---- tiny module-call shim ----------------------------------------------------
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


def _ensure_tree_envelope(res: Dict[str, Any], *, scheme: str) -> Dict[str, Any]:
    """
    Normalize shapes to a consistent tree envelope + nested:
      - If res.tree.periods is present (some engines), mirror to children + nested.
      - If res.nested list exists, create/ensure a 'tree' wrapper using min/max bounds.
    """
    if not isinstance(res, dict):
        return res

    t = res.get("tree")
    if isinstance(t, dict) and isinstance(t.get("periods"), list):
        periods = t["periods"]
        s0 = min((float(p.get("start_jd_tt", 0.0)) for p in periods), default=0.0)
        e1 = max((float(p.get("end_jd_tt", 0.0)) for p in periods), default=0.0)
        out = dict(res)
        out["nested"] = periods
        out["tree"] = {
            "level": 0,
            "lord": None,
            "label": scheme,
            "start_jd_tt": s0,
            "end_jd_tt": e1,
            "children": periods,
        }
        return out

    if isinstance(res.get("nested"), list):
        nodes = res["nested"]
        if nodes:
            s0 = min(float(n.get("start_jd_tt", 0.0)) for n in nodes)
            e1 = max(float(n.get("end_jd_tt", 0.0)) for n in nodes)
        else:
            s0, e1 = 0.0, 0.0
        out = dict(res)
        out["tree"] = {
            "level": 0,
            "lord": None,
            "label": scheme,
            "start_jd_tt": s0,
            "end_jd_tt": e1,
            "children": nodes,
        }
        return out

    return res


# ──────────────────────────────────────────────────────────────────────────────
# Core runners (each: validator → registry (preferred) → module fallback)
# ──────────────────────────────────────────────────────────────────────────────
def _run_with_registry_first(
    *,
    route_name: str,
    scheme_key: str,
    civic_builder,
    module_fallback_fn,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    if normalize_vim_payload is None:
        return {
            "ok": False,
            "error": "validator_unavailable",
            "detail": f"app.core.vedic_validator.normalize_vim_payload import failed: {_VALIDATOR_IMPORT_ERR}",
        }

    norm, warns, tz_norm = normalize_vim_payload(body)  # type: ignore[misc]

    # Attempt central registry (preferred) — return ONLY on success
    if _compute_dasha_registry is not None:
        try:
            civ = civic_builder(body, norm)
            civ["scheme"] = scheme_key
            out = _compute_dasha_registry(civ)  # type: ignore[misc]
            if isinstance(out, dict) and out.get("ok"):
                out = _ensure_tree_envelope(out, scheme=scheme_key)
                return _wrap_ok(out, warns, tz_norm, branch="registry.compute_dasha", route_name=route_name)
            # else: fall through to module fallback
        except Exception:
            # fall through to module fallback
            pass

    # Module fallback, if available
    if module_fallback_fn is not None:
        try:
            civ = civic_builder(body, norm)
            out = _call_single_param_or_kwargs(module_fallback_fn, civ)
            if isinstance(out, dict):
                out = _ensure_tree_envelope(out, scheme=scheme_key)
                return _wrap_ok(out, warns, tz_norm, branch=f"module.compute_{scheme_key}", route_name=route_name)
            return {
                "ok": False,
                "error": f"{scheme_key}_module_invalid_return",
                "detail": f"Expected dict, got {type(out).__name__}",
                "meta": {"route": route_name, "tz_normalized": tz_norm, "branch": f"module.compute_{scheme_key}"},
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"{scheme_key}_module_failed",
                "detail": str(e),
                "meta": {"route": route_name, "tz_normalized": tz_norm, "branch": f"module.compute_{scheme_key}"},
            }

    # No engine available
    return {
        "ok": False,
        "error": f"{scheme_key}_engine_unavailable",
        "meta": {"route": route_name, "tz_normalized": tz_norm, "branch": "none"},
    }


def _run_vimshottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _run_with_registry_first(
        route_name="vimshottari",
        scheme_key="vimshottari",
        civic_builder=_build_civic_payload_vim,
        module_fallback_fn=_compute_vim_module,
        body=payload,
    )


def _run_ashtottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _run_with_registry_first(
        route_name="ashtottari",
        scheme_key="ashtottari",
        civic_builder=_build_civic_payload_ashto,
        module_fallback_fn=_compute_ashto_module,
        body=payload,
    )


def _run_yogini(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _run_with_registry_first(
        route_name="yogini",
        scheme_key="yogini",
        civic_builder=_build_civic_payload_yogini,
        module_fallback_fn=_compute_yogini_module,
        body=payload,
    )


def _run_chara(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _run_with_registry_first(
        route_name="chara",
        scheme_key="chara",
        civic_builder=_build_civic_payload_chara,
        module_fallback_fn=_compute_chara_module,
        body=payload,
    )


def _run_kalachakra(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _run_with_registry_first(
        route_name="kalachakra",
        scheme_key="kalachakra",
        civic_builder=_build_civic_payload_kcd,
        module_fallback_fn=_compute_kcd_module,
        body=payload,
    )


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
        "registry_available": _available_schemes_fn() if callable(_available_schemes_fn) else None,
        "module_vimshottari_present": bool(_compute_vim_module),
        "module_vimshottari_sig": sigs(_compute_vim_module) if _compute_vim_module else None,
        "module_ashtottari_present": bool(_compute_ashto_module),
        "module_ashtottari_sig": sigs(_compute_ashto_module) if _compute_ashto_module else None,
        "module_yogini_present": bool(_compute_yogini_module),
        "module_yogini_sig": sigs(_compute_yogini_module) if _compute_yogini_module else None,
        "module_chara_present": bool(_compute_chara_module),
        "module_chara_sig": sigs(_compute_chara_module) if _compute_chara_module else None,
        "module_kalachakra_present": bool(_compute_kcd_module),
        "module_kalachakra_sig": sigs(_compute_kcd_module) if _compute_kcd_module else None,
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


# Chara (Jaimini)
@vedic_api.post("/dasha/chara")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_chara():
    body = request.get_json(silent=True) or {}
    res = _run_chara(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status


# Kālachakra
@vedic_api.post("/dasha/kalachakra")
@vedic_api.post("/dasha/kalacakra")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_kalachakra():
    """
    Body must include either:
      - kcd_table: { "pada_to_sequence": {1:[...],...,108:[...]}, "sign_years": {1:.., ... 12:..} }
        (and optionally "name"), or
      - kcd_preset: "your-preset-name" if your kalachakra module wires a preset loader.
    """
    body = request.get_json(silent=True) or {}
    res = _run_kalachakra(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status
