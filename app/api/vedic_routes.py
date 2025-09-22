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
    """Shared bucket key ('20') used by all predictive/varga/yoga calls."""
    return "20"


# ──────────────────────────────────────────────────────────────────────────────
# Validators
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.vedic_validator import (
        normalize_vim_payload,    # for dasha routes & common tz/site hints
        normalize_yoga_payload,   # for yoga routes (Mode-C only)
    )  # type: ignore
    _VALIDATOR_IMPORT_ERR = None
except Exception as _e:
    normalize_vim_payload = None  # type: ignore
    normalize_yoga_payload = None  # type: ignore
    _VALIDATOR_IMPORT_ERR = repr(_e)


# ──────────────────────────────────────────────────────────────────────────────
# Engines: central registry (preferred) + direct module fallbacks (dasha only)
# ──────────────────────────────────────────────────────────────────────────────
_compute_dasha_registry = None
_available_schemes_fn = None
try:
    from app.core.dasha_registry import compute_dasha as _compute_dasha_registry  # type: ignore
    from app.core.dasha_registry import available_schemes as _available_schemes_fn  # type: ignore
except Exception:
    _compute_dasha_registry = None  # type: ignore
    _available_schemes_fn = None  # type: ignore

# Optional dasha module fallbacks
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
# Varga engine (varga_charts)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.varga_charts import (
        varga_position as _varga_position,
        compute_varga_chart as _compute_varga_chart,
        compute_many_vargas as _compute_many_vargas,
    )
    _VARGA_OK = True
except Exception:
    _varga_position = None  # type: ignore
    _compute_varga_chart = None  # type: ignore
    _compute_many_vargas = None  # type: ignore
    _VARGA_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Timescales + Ayanāṁśa (used by varga routes)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.timescales import build_timescales  # type: ignore
    _TS_OK = True
except Exception:
    build_timescales = None  # type: ignore
    _TS_OK = False

try:
    from app.core.ayanamsa import get_ayanamsa_deg  # type: ignore
    _AY_OK = True
except Exception:
    get_ayanamsa_deg = None  # type: ignore
    _AY_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Yoga core (Mode-C only) — prefer singular module name, then fallback
# ──────────────────────────────────────────────────────────────────────────────
_YOGA_OK = False
_compute_yogas = None  # type: ignore
_yoga_list = None      # type: ignore

try:
    # Preferred: app/core/yoga.py
    from app.core.yoga import (                                     # type: ignore
        compute_yogas as _compute_yogas,
        list_registered_yogas as _yoga_list,
    )
    _YOGA_OK = True
except Exception:
    try:
        # Fallback: app/core/yogas.py (alternate filename)
        from app.core.yogas import (                                # type: ignore
            compute_yogas as _compute_yogas,
            list_registered_yogas as _yoga_list,
        )
        _YOGA_OK = True
    except Exception:
        _YOGA_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Gochar / Ingress / Stations wrappers (from vedic_predictive)
# ──────────────────────────────────────────────────────────────────────────────
_GOCHAR_OK = False
try:
    from app.core.vedic_predictive import (  # type: ignore
        gochar_drishti as _gochar_drishti,
        ingresses_rashi as _ingresses_rashi,
        ingresses_nakshatra as _ingresses_nakshatra,
        stations_retro_direct as _stations_retro_direct,
        feature_drishti_proximity as _feature_drishti_proximity,
    )
    _GOCHAR_OK = True
except Exception:
    _GOCHAR_OK = False
    _gochar_drishti = None               # type: ignore
    _ingresses_rashi = None              # type: ignore
    _ingresses_nakshatra = None          # type: ignore
    _stations_retro_direct = None        # type: ignore
    _feature_drishti_proximity = None    # type: ignore


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


def _tz_from_payload(body: Dict[str, Any]) -> str:
    """Best-effort: prefer validator normalization, else raw tz, else UTC."""
    try:
        if normalize_vim_payload is not None:
            _, _warns, tz_norm = normalize_vim_payload(body)  # type: ignore[misc]
            return tz_norm or str(body.get("tz") or "UTC")
    except Exception:
        pass
    return str(body.get("tz") or body.get("place_tz") or "UTC")


# ---- civic payload builders (dasha) ------------------------------------------
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
    """
    Flexible flags for Vimśottarī (method/observer/ayanamsa/place/etc.).
    """
    civ: Dict[str, Any] = {}

    # Birth time: prefer explicit birth_jd_tt → else jd_tt → else (date,time,tz)
    if isinstance(original.get("birth_jd_tt"), (int, float)):
        civ["birth_jd_tt"] = float(original["birth_jd_tt"])
    elif isinstance(norm.get("jd_tt"), (int, float)):
        civ["birth_jd_tt"] = float(norm["jd_tt"])
    for k in ("date", "time", "tz"):
        v = norm.get(k)
        if isinstance(v, str) and v.strip():
            civ[k] = v.strip()

    # Method/observer/ayanamsa
    if norm.get("method"):
        civ["method"] = norm["method"]
    if norm.get("coordinate_mode"):
        civ["coordinate_mode"] = norm["coordinate_mode"]
        civ["topocentric"] = (str(norm["coordinate_mode"]).lower() == "topocentric")
    elif "observer" in original:
        obs = str(original.get("observer") or "").strip().lower()
        civ["coordinate_mode"] = "topocentric" if obs == "topocentric" else "geocentric"
        civ["topocentric"] = (civ["coordinate_mode"] == "topocentric")
    elif "topocentric" in original:
        civ["topocentric"] = bool(original.get("topocentric"))
        civ["coordinate_mode"] = "topocentric" if civ["topocentric"] else "geocentric"

    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]

    # Geography
    for k in ("latitude", "longitude", "elevation"):
        if norm.get(k) is not None:
            civ[k] = norm[k]
        elif original.get(k) is not None:
            civ[k] = original[k]

    # Optional place fields (pass-through)
    for k in ("place", "place_city", "place_state", "place_country"):
        if original.get(k) is not None:
            civ[k] = original[k]
        elif norm.get(k) is not None:
            civ[k] = norm[k]

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
    civ = _pick_times(norm, original)

    if norm.get("ayanamsa") is not None:
        civ["ayanamsa"] = norm["ayanamsa"]

    civ["levels"] = _levels_from(norm)

    lim = original.get("limit_jd_tt", norm.get("limit_jd_tt"))
    if isinstance(lim, (int, float)):
        civ["limit_jd_tt"] = float(lim)

    for k in (
        "kcd_table", "kcd_preset", "use_demo_kcd_table",
        "override_start_sign_index", "year_days",
        "balance_years", "balance_fraction",
        "planet_longitudes_sidereal"
    ):
        if k in original and original[k] is not None:
            civ[k] = original[k]
    return civ


# ---- tiny module-call shim (dasha) -------------------------------------------
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
    Normalize shapes to a consistent tree envelope + nested.
    """
    if not isinstance(res, dict):
        return res

    t = res.get("tree")
    if isinstance(t, dict) and isinstance(t.get("periods"), list):
        periods = t["periods"]
        s0 = min((float(p.get("start_jd_tt", 0.0)) for p in periods), default=0.0)
        e1 = max((float(p.get("end_jd_tt", 0.0)) for p in periods), default=0.0)

        extras_keys = (
            "nakshatra", "moon_nirayana_deg", "birth_jd_tt", "year_days",
            "levels", "ayanamsa", "method", "coordinate_mode", "topocentric",
        )
        extras = {k: t[k] for k in extras_keys if k in t}

        out = dict(res)
        out["nested"] = periods
        root = {
            "level": 0,
            "lord": None,
            "label": scheme,
            "start_jd_tt": s0,
            "end_jd_tt": e1,
            "children": periods,
            **extras,
        }
        out["tree"] = root
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
# Core runners (dasha): validator → registry (preferred) → module fallback
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

    # Attempt central registry (preferred)
    if _compute_dasha_registry is not None:
        try:
            civ = civic_builder(body, norm)
            civ["scheme"] = scheme_key
            out = _compute_dasha_registry(civ)  # type: ignore[misc]
            if isinstance(out, dict) and out.get("ok"):
                out = _ensure_tree_envelope(out, scheme=scheme_key)
                return _wrap_ok(out, warns, tz_norm, branch="registry.compute_dasha", route_name=route_name)
        except Exception:
            pass  # fall through

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
# Varga helpers (request-scoped; will compute ayanāṁśa deg from jd_tt if possible)
# ──────────────────────────────────────────────────────────────────────────────
def _norm_method(v: Any, default: str = "sidereal") -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("sidereal", "nirayana", "nirāyaṇa", "sid", "s"): return "sidereal"
        if s in ("tropical", "sayana", "sāyana", "trop", "t"):   return "tropical"
    return default

def _norm_ayanamsa(v: Any) -> Any:
    # pass through floats/ints; normalize strings; default lahiri
    if v is None: return "lahiri"
    if isinstance(v, (int, float)): return float(v)
    return str(v).strip().lower() or "lahiri"

def _coerce_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, str) and x.strip() not in ("", "null", "None"):
            return float(x.strip())
    except Exception:
        return None
    return None

def _pick_longitudes(body: Dict[str, Any]) -> Dict[str, float]:
    for key in ("longitudes", "points_deg", "longitudes_by_name"):
        raw = body.get(key)
        if isinstance(raw, dict):
            out: Dict[str, float] = {}
            for name, val in raw.items():
                f = _coerce_float(val)
                if f is not None:
                    out[str(name)] = f
            return out
    return {}

def _pick_vargas(body: Dict[str, Any]) -> List[str]:
    v = body.get("vargas") or body.get("include")
    if isinstance(v, (list, tuple)):
        return [str(x).upper() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [s.strip().upper() for s in v.split(",") if s.strip()]
    # sensible default if caller forgot (match common UI)
    return ["D1", "D9", "D10", "D12"]

def _wrap_varga_meta(route: str, method: str, ay_meta: Dict[str, Any]) -> Dict[str, Any]:
    base = {"route": route, "branch": "varga_charts", "zodiac_mode": method}
    base.update(ay_meta or {})
    return base

def _pad_hms(t: Any) -> str:
    s = str(t or "").strip()
    if not s: return "12:00:00"
    parts = s.split(":")
    if len(parts) == 1: return f"{parts[0]}:00:00"
    if len(parts) == 2: return f"{parts[0]}:{parts[1]}:00"
    return s

def _jd_tt_from_body(body: Dict[str, Any]) -> Optional[float]:
    # direct overrides
    jd = _coerce_float(body.get("jd_tt") or body.get("birth_jd_tt"))
    if isinstance(jd, float):
        return jd
    # civil → JD_TT
    if not _TS_OK:
        return None
    date = str(body.get("date") or body.get("birth_date") or "").strip()
    if not date:
        return None
    time = _pad_hms(body.get("time") or body.get("birth_time") or "12:00")
    tz   = str(body.get("tz") or body.get("place_tz") or "UTC").strip() or "UTC"
    try:
        ts = build_timescales(date, time, tz, _env_dut1_seconds())  # type: ignore[misc]
        if isinstance(ts, dict):
            j = ts.get("jd_tt")
        else:
            j = getattr(ts, "jd_tt", None)
        return float(j) if j is not None else None
    except Exception:
        return None

def _ayanamsa_deg_from_key(jd_tt: Optional[float], key: str) -> Optional[float]:
    if not _AY_OK or get_ayanamsa_deg is None:
        return None
    try:
        if jd_tt is not None:
            try:
                return float(get_ayanamsa_deg(jd_tt, key))  # type: ignore[misc]
            except Exception:
                pass
        return float(get_ayanamsa_deg(None, key))  # type: ignore[misc]
    except Exception:
        return None


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
        "validator_loaded": (normalize_vim_payload is not None) and (normalize_yoga_payload is not None),
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
        "varga_engine_present": _VARGA_OK,
        "timescales_present": _TS_OK,
        "ayanamsa_adapter_present": _AY_OK,
        # Yoga diagnostics
        "yoga_core_present": _YOGA_OK,
        # Gochar diagnostics
        "gochar_present": _GOCHAR_OK,
        "rl_cap_per_min": RL_VEDIC_PREDICTIVE,
        "rl_bucket_key": "20",
        "dut1_seconds_env": _env_dut1_seconds(),
    }), 200


# ──────────────── Dasha routes ────────────────
@vedic_api.post("/dasha/vimshottari")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_vimshottari():
    body = request.get_json(silent=True) or {}
    res = _run_vimshottari(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status


@vedic_api.post("/dasha/ashtottari")
@vedic_api.post("/dasha/Aṣṭottarī")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_ashtottari():
    body = request.get_json(silent=True) or {}
    res = _run_ashtottari(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status


@vedic_api.post("/dasha/yogini")
@vedic_api.post("/dasha/Yoginī")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_yogini():
    body = request.get_json(silent=True) or {}
    res = _run_yogini(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status


@vedic_api.post("/dasha/chara")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_chara():
    body = request.get_json(silent=True) or {}
    res = _run_chara(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status


@vedic_api.post("/dasha/kalachakra")
@vedic_api.post("/dasha/kalacakra")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_kalachakra():
    body = request.get_json(silent=True) or {}
    res = _run_kalachakra(body)
    status = 200 if res.get("ok") else (503 if str(res.get("error", "")).endswith("unavailable") else 400)
    return jsonify(res), status


# ──────────────── Yoga routes (Mode-C only) ────────────────
@vedic_api.get("/yoga/catalog")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_yoga_catalog():
    if not (_YOGA_OK and callable(_yoga_list)):
        # Gracefully return empty when yoga core absent
        return jsonify({"ok": True, "catalog": [], "meta": {"route": "yoga/catalog", "branch": "none"}}), 200
    try:
        catalog = _yoga_list() or []
        return jsonify({"ok": True, "catalog": catalog, "meta": {"route": "yoga/catalog", "branch": "yoga_core"}}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": "yoga_catalog_failed", "detail": str(e)}), 400


@vedic_api.post("/yoga/detect")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_yoga_detect():
    """
    Yoga detection (civil/time + site only).
    Requires: date, time, tz, latitude, longitude (place is optional; if provided and a resolver is available, it can fill missing tz/coords).
    No precomputed points/cusps accepted here.
    """
    if normalize_yoga_payload is None:
        return jsonify({"ok": False, "error": "validator_unavailable"}), 503
    if not (_YOGA_OK and callable(_compute_yogas)):
        return jsonify({"ok": False, "error": "yoga_core_unavailable"}), 503

    body = request.get_json(silent=True) or {}
    norm, warns, tz_norm = normalize_yoga_payload(body)  # type: ignore[misc]

    # Basic input guardrails
    if not norm.get("date") or not norm.get("time"):
        return jsonify({
            "ok": False,
            "error": "missing_date_or_time",
            "warnings": (warns or []),
            "meta": {"route": "yoga/detect", "branch": "needs_civil", "tz_normalized": tz_norm},
        }), 400
    if norm.get("latitude") is None or norm.get("longitude") is None:
        return jsonify({
            "ok": False,
            "error": "missing_coordinates",
            "warnings": (warns or []),
            "meta": {"route": "yoga/detect", "branch": "needs_coordinates", "tz_normalized": tz_norm},
            "hints": [
                "Provide 'latitude' and 'longitude' (in degrees).",
                "Optionally send a 'place' string (city/state/country) if a resolver is enabled.",
            ],
        }), 400

    # Call the core yoga engine
    try:
        res = _compute_yogas(
            {
                "date": norm["date"],
                "time": norm["time"],
                "tz": norm["tz"],
                "latitude": norm["latitude"],
                "longitude": norm["longitude"],
                "elevation_m": norm.get("elevation_m"),
                # passthrough for any future core-side needs
                "include": norm.get("include") or [],
            },
            ayanamsa=norm.get("ayanamsa", "lahiri"),
            house_system=norm.get("house_system", "placidus"),
            # defaults below mirror core defaults; override if you expose UI toggles
            sign_lord_variant="classical",
            chandra_mangala_by_sign=True,
            conj_orb_deg=6.0,
            gajakesari_include_same_house=True,
            include_mooltrikona_in_mahapurusha=True,
            use_vargas_for_scoring=bool(norm.get("use_vargas_for_scoring", True)),
            varga_keys_for_boost=tuple(norm.get("varga_keys_for_boost") or ("D9", "D10")),
            include_arudha_notes=False,
            enable_catalog_tags=tuple(norm.get("enable_catalog_tags") or ()),
            disable_catalog_tags=tuple(norm.get("disable_catalog_tags") or ()),
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "yoga_detect_failed", "detail": str(e)}), 400

    ok = bool(res.get("ok"))
    yogas = res.get("yogas", []) if isinstance(res, dict) else []
    present = [y.get("name") for y in yogas if isinstance(y, dict) and y.get("present")]
    out = {
        "ok": ok,
        "yogas": yogas,
        "present": present,
        "context": res.get("context"),
        "warnings": (warns or []) + (res.get("warnings", []) if isinstance(res, dict) else []),
        "meta": {"route": "yoga/detect", "branch": "yoga_core", "tz_normalized": tz_norm},
    }
    if not ok:
        out["error"] = res.get("error", "yoga_detect_failed")
        return jsonify(out), (503 if str(out["error"]).endswith("unavailable") else 400)
    return jsonify(out), 200


# ──────────────── Gochar / Ingress / Stations routes ────────────────
@vedic_api.post("/gochar/drishti")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_gochar_drishti():
    """
    Degree-true graha dṛṣṭi transit hits within a civil window.
    Body:
      - date_from, date_to (YYYY-MM-DD or RFC3339)
      - Optional: transiting_bodies, natal_targets, zodiac_mode/method, ayanamsa,
                  include_nodes, treat_nodes_like_saturn, orb_deg, orb_map,
                  step_minutes ("auto"|number), prebatch_refinement (bool),
                  frame ("ecliptic-of-date"), plus any natal/site fields.
    """
    if not (_GOCHAR_OK and callable(_gochar_drishti)):
        return jsonify({"ok": False, "error": "gochar_engine_unavailable"}), 503

    body = request.get_json(silent=True) or {}
    date_from = str(body.get("date_from") or body.get("from") or "").strip()
    date_to   = str(body.get("date_to") or body.get("to") or "").strip()
    if not date_from or not date_to:
        return jsonify({"ok": False, "error": "missing_date_window",
                        "hints": ["Provide 'date_from' and 'date_to' (YYYY-MM-DD)."]}), 400

    # Normalize natal payload enough to pass through (tz/place/coords if available)
    if normalize_vim_payload is None:
        natal_chart = body
        warns: List[str] = []
        tz_norm = _tz_from_payload(body)
    else:
        natal_chart, warns, tz_norm = normalize_vim_payload(body)  # type: ignore[misc]

    try:
        res = _gochar_drishti(
            natal_chart=natal_chart,
            date_from=date_from,
            date_to=date_to,
            transiting_bodies=body.get("transiting_bodies"),
            natal_targets=body.get("natal_targets"),
            zodiac_mode=(body.get("zodiac_mode") or body.get("method") or "sidereal"),
            ayanamsa=body.get("ayanamsa", "lahiri"),
            frame=str(body.get("frame") or "ecliptic-of-date"),
            include_nodes=bool(body.get("include_nodes", False)),
            treat_nodes_like_saturn=bool(body.get("treat_nodes_like_saturn", False)),
            orb_deg=float(body.get("orb_deg", 12.0)),
            orb_map=body.get("orb_map"),
            step_minutes=body.get("step_minutes", "auto"),
            prebatch_refinement=bool(body.get("prebatch_refinement", False)),
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "gochar_drishti_failed", "detail": str(e)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "gochar/drishti", "tz_normalized": tz_norm, "branch": "vedic_predictive"})
        if normalize_vim_payload is not None and warns:
            res.setdefault("warnings", []).extend(warns)
    ok = bool(res.get("ok", False))
    return jsonify(res), (200 if ok else 400)


@vedic_api.post("/gochar/drishti/proximity")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_gochar_drishti_proximity():
    """
    Score helper for drishti hits: returns [0..1] proximity per hit to the axis.
    Body:
      - hits: array from /gochar/drishti response's "gochar"
      - cap_deg (default 12)
    """
    if not (_GOCHAR_OK and callable(_feature_drishti_proximity)):
        return jsonify({"ok": False, "error": "gochar_engine_unavailable"}), 503
    body = request.get_json(silent=True) or {}
    hits = body.get("hits") or body.get("gochar") or []
    cap = float(body.get("cap_deg", 12.0))
    try:
        scores = _feature_drishti_proximity(hits=hits, cap_deg=cap)
        return jsonify({"ok": True, "scores": scores, "meta": {"route": "gochar/drishti/proximity"}}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": "feature_drishti_proximity_failed", "detail": str(e)}), 400


@vedic_api.post("/ingress/rashi")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_ingress_rashi():
    if not (_GOCHAR_OK and callable(_ingresses_rashi)):
        return jsonify({"ok": False, "error": "gochar_engine_unavailable"}), 503

    body = request.get_json(silent=True) or {}
    date_from = str(body.get("date_from") or body.get("from") or "").strip()
    date_to   = str(body.get("date_to") or body.get("to") or "").strip()
    if not date_from or not date_to:
        return jsonify({"ok": False, "error": "missing_date_window"}), 400

    tz_name = _tz_from_payload(body)
    try:
        res = _ingresses_rashi(
            date_from=date_from,
            date_to=date_to,
            movers=body.get("movers"),
            zodiac_mode=(body.get("zodiac_mode") or body.get("method") or "sidereal"),
            ayanamsa=body.get("ayanamsa", "lahiri"),
            frame=str(body.get("frame") or "ecliptic-of-date"),
            topocentric=bool(body.get("topocentric", False)),
            latitude=_coerce_float(body.get("latitude")),
            longitude=_coerce_float(body.get("longitude")),
            elevation_m=_coerce_float(body.get("elevation_m") or body.get("elevation")),
            step_minutes=body.get("step_minutes", "auto"),
            tz_name=tz_name,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "rashi_ingress_failed", "detail": str(e)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "ingress/rashi", "tz_normalized": tz_name, "branch": "vedic_predictive"})
    return jsonify(res), (200 if res.get("ok") else 400)


@vedic_api.post("/ingress/nakshatra")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_ingress_nakshatra():
    if not (_GOCHAR_OK and callable(_ingresses_nakshatra)):
        return jsonify({"ok": False, "error": "gochar_engine_unavailable"}), 503

    body = request.get_json(silent=True) or {}
    date_from = str(body.get("date_from") or body.get("from") or "").strip()
    date_to   = str(body.get("date_to") or body.get("to") or "").strip()
    if not date_from or not date_to:
        return jsonify({"ok": False, "error": "missing_date_window"}), 400

    tz_name = _tz_from_payload(body)
    try:
        res = _ingresses_nakshatra(
            date_from=date_from,
            date_to=date_to,
            movers=body.get("movers"),
            zodiac_mode=(body.get("zodiac_mode") or body.get("method") or "sidereal"),
            ayanamsa=body.get("ayanamsa", "lahiri"),
            frame=str(body.get("frame") or "ecliptic-of-date"),
            topocentric=bool(body.get("topocentric", False)),
            latitude=_coerce_float(body.get("latitude")),
            longitude=_coerce_float(body.get("longitude")),
            elevation_m=_coerce_float(body.get("elevation_m") or body.get("elevation")),
            step_minutes=body.get("step_minutes", "auto"),
            tz_name=tz_name,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "nakshatra_ingress_failed", "detail": str(e)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "ingress/nakshatra", "tz_normalized": tz_name, "branch": "vedic_predictive"})
    return jsonify(res), (200 if res.get("ok") else 400)


@vedic_api.post("/stations")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_stations():
    if not (_GOCHAR_OK and callable(_stations_retro_direct)):
        return jsonify({"ok": False, "error": "gochar_engine_unavailable"}), 503

    body = request.get_json(silent=True) or {}
    date_from = str(body.get("date_from") or body.get("from") or "").strip()
    date_to   = str(body.get("date_to") or body.get("to") or "").strip()
    if not date_from or not date_to:
        return jsonify({"ok": False, "error": "missing_date_window"}), 400

    tz_name = _tz_from_payload(body)
    try:
        res = _stations_retro_direct(
            date_from=date_from,
            date_to=date_to,
            movers=body.get("movers"),
            zodiac_mode=(body.get("zodiac_mode") or body.get("method") or "sidereal"),
            ayanamsa=body.get("ayanamsa", "lahiri"),
            frame=str(body.get("frame") or "ecliptic-of-date"),
            topocentric=bool(body.get("topocentric", False)),
            latitude=_coerce_float(body.get("latitude")),
            longitude=_coerce_float(body.get("longitude")),
            elevation_m=_coerce_float(body.get("elevation_m") or body.get("elevation")),
            step_minutes=body.get("step_minutes", "auto"),
            tz_name=tz_name,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "stations_failed", "detail": str(e)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "stations", "tz_normalized": tz_name, "branch": "vedic_predictive"})
    return jsonify(res), (200 if res.get("ok") else 400)


# ──────────────── Varga routes (varga_charts) ────────────────
@vedic_api.post("/varga/position")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_varga_position():
    if not _VARGA_OK or _varga_position is None:
        return jsonify({"ok": False, "error": "varga_engine_unavailable"}), 503

    body = request.get_json(silent=True) or {}
    varga = str(body.get("varga") or "").strip() or "D9"
    lon = _coerce_float(body.get("lon_deg"))
    if lon is None:
        return jsonify({"ok": False, "error": "missing_or_invalid_lon_deg"}), 400

    method = _norm_method(body.get("method") or body.get("zodiac_mode") or "sidereal")
    jd_tt = _jd_tt_from_body(body)
    ay_for_engine, ay_meta = _resolve_ayanamsa_for_engine(body, method, jd_tt)

    try:
        result = _varga_position(float(lon), varga, zodiac_mode=method, ayanamsa=ay_for_engine)
        out = {
            "ok": True,
            "varga": result.get("varga", varga),
            "result": result,
            "meta": _wrap_varga_meta("varga/position", method, ay_meta),
        }
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"ok": False, "error": "varga_position_failed", "detail": str(e)}), 400


@vedic_api.post("/varga/chart")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_varga_chart():
    if not _VARGA_OK or _compute_varga_chart is None:
        return jsonify({"ok": False, "error": "varga_engine_unavailable"}), 503

    body = request.get_json(silent=True) or {}
    varga = str(body.get("varga") or "").strip() or "D9"
    longitudes = _pick_longitudes(body)
    if not longitudes:
        return jsonify({"ok": False, "error": "missing_or_invalid_longitudes"}), 400

    method = _norm_method(body.get("method") or body.get("zodiac_mode") or "sidereal")
    jd_tt = _jd_tt_from_body(body)
    ay_for_engine, ay_meta = _resolve_ayanamsa_for_engine(body, method, jd_tt)

    try:
        placements = _compute_varga_chart(longitudes, varga, zodiac_mode=method, ayanamsa=ay_for_engine)
        out = {
            "ok": True,
            "varga": varga.upper(),
            "placements": placements,
            "options": {"zodiac_mode": method, "ayanamsa": ay_for_engine},
            "meta": _wrap_varga_meta("varga/chart", method, ay_meta),
        }
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"ok": False, "error": "varga_chart_failed", "detail": str(e)}), 400


@vedic_api.post("/varga/many")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_varga_many():
    if not _VARGA_OK or _compute_many_vargas is None:
        return jsonify({"ok": False, "error": "varga_engine_unavailable"}), 503

    body = request.get_json(silent=True) or {}
    longitudes = _pick_longitudes(body)
    if not longitudes:
        return jsonify({"ok": False, "error": "missing_or_invalid_longitudes"}), 400

    vargas = _pick_vargas(body)
    method = _norm_method(body.get("method") or body.get("zodiac_mode") or "sidereal")
    jd_tt = _jd_tt_from_body(body)
    ay_for_engine, ay_meta = _resolve_ayanamsa_for_engine(body, method, jd_tt)

    try:
        placements = _compute_many_vargas(longitudes, vargas, zodiac_mode=method, ayanamsa=ay_for_engine)
        out = {
            "ok": True,
            "vargas": [v.upper() for v in vargas],
            "placements": placements,
            "options": {"zodiac_mode": method, "ayanamsa": ay_for_engine},
            "meta": _wrap_varga_meta("varga/many", method, ay_meta),
        }
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"ok": False, "error": "varga_many_failed", "detail": str(e)}), 400


# ──────────────────────────────────────────────────────────────────────────────
# Local ayanamsa resolver for varga helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_ayanamsa_for_engine(body: Dict[str, Any], method: str, jd_tt: Optional[float]) -> tuple[Any, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "ts_available": _TS_OK,
        "ayanamsa_adapter_available": _AY_OK,
        "jd_tt_used": jd_tt,
    }
    method_lc = (method or "sidereal").lower()
    ay_in = body.get("ayanamsa")
    ay_deg_in = _coerce_float(body.get("ayanamsa_deg"))

    if method_lc.startswith("trop"):
        meta.update({
            "ayanamsa_input": ay_in if ay_in is not None else ("ayanamsa_deg=" + str(ay_deg_in) if ay_deg_in is not None else None),
            "ayanamsa_effective": None,
            "ayanamsa_resolve": "not_applied_tropical",
        })
        return ay_in, meta

    if isinstance(ay_deg_in, float):
        meta.update({
            "ayanamsa_input": ay_deg_in,
            "ayanamsa_effective": ay_deg_in,
            "ayanamsa_resolve": "explicit_numeric",
        })
        return ay_deg_in, meta
    if isinstance(ay_in, (int, float)):
        val = float(ay_in)
        meta.update({
            "ayanamsa_input": val,
            "ayanamsa_effective": val,
            "ayanamsa_resolve": "explicit_numeric",
        })
        return val, meta

    key = _norm_ayanamsa(ay_in)
    if isinstance(key, str) and key:
        deg = _ayanamsa_deg_from_key(jd_tt, key)
        if isinstance(deg, float):
            meta.update({
                "ayanamsa_input": key,
                "ayanamsa_effective": deg,
                "ayanamsa_resolve": "computed_from_key",
                "ayanamsa_key": key,
            })
            return deg, meta
        meta.update({
            "ayanamsa_input": key,
            "ayanamsa_effective": key,
            "ayanamsa_resolve": "pass_through_string_fallback",
            "ayanamsa_key": key,
            "warning": "could_not_compute_ayanamsa_degrees_from_key",
        })
        return key, meta

    meta.update({
        "ayanamsa_input": key,
        "ayanamsa_effective": key,
        "ayanamsa_resolve": "default_key_passthrough",
    })
    return key, meta
