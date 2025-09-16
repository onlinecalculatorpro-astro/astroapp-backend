# app/api/vedic_routes.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Iterable
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
# Validator (NO jd_utc) — used by dasha routes and yoga (for civil/time/geo)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.vedic_validator import normalize_vim_payload  # type: ignore
except Exception as _e:
    normalize_vim_payload = None  # type: ignore
    _VALIDATOR_IMPORT_ERR = repr(_e)
else:
    _VALIDATOR_IMPORT_ERR = None


# ──────────────────────────────────────────────────────────────────────────────
# Engines: central registry (preferred) + direct module fallbacks (dasha)
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
# Yoga engine wiring (primary via app.core.yoga)
# ──────────────────────────────────────────────────────────────────────────────
_YOGA_CORE_OK = False
try:
    # Prefer yoga.py
    from app.core.yoga import (
        compute_yogas as _compute_yogas_core,
        list_registered_yogas as _yoga_list,
    )
    _YOGA_CORE_OK = True
except Exception:
    try:
        # Alternate file name yogas.py
        from app.core.yogas import (  # type: ignore
            compute_yogas as _compute_yogas_core,
            list_registered_yogas as _yoga_list,
        )
        _YOGA_CORE_OK = True
    except Exception:
        _compute_yogas_core = None  # type: ignore
        _yoga_list = None  # type: ignore
        _YOGA_CORE_OK = False


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
    Flexible flags for Vimśottarī (method/observer/ayanamsa/place/etc.)."
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
        # Prefer (jd_tt, key); fall back to (None, key) if the impl allows it.
        if jd_tt is not None:
            try:
                return float(get_ayanamsa_deg(jd_tt, key))  # type: ignore[misc]
            except Exception:
                pass
        return float(get_ayanamsa_deg(None, key))  # type: ignore[misc]
    except Exception:
        return None

def _resolve_ayanamsa_for_engine(body: Dict[str, Any], method: str, jd_tt: Optional[float]) -> tuple[Any, Dict[str, Any]]:
    """
    Decide what to pass into varga_charts:
      - Tropical → ayanamsa not applied (pass-through; note why)
      - If numeric provided (ayanamsa or ayanamsa_deg) → use numeric (explicit)
      - If string key and we can compute deg via get_ayanamsa_deg(jd_tt,key) → use numeric
      - Else pass string through (engine may fallback to 0°); include warning in meta
    Returns (ayanamsa_for_engine, meta_dict)
    """
    meta: Dict[str, Any] = {
        "ts_available": _TS_OK,
        "ayanamsa_adapter_available": _AY_OK,
        "jd_tt_used": jd_tt,
    }
    method_lc = (method or "sidereal").lower()
    ay_in = body.get("ayanamsa")
    ay_deg_in = _coerce_float(body.get("ayanamsa_deg"))

    # Tropical: ignore ayanamsa, but still echo what was received for transparency
    if method_lc.startswith("trop"):
        meta.update({
            "ayanamsa_input": ay_in if ay_in is not None else ("ayanamsa_deg=" + str(ay_deg_in) if ay_deg_in is not None else None),
            "ayanamsa_effective": None,
            "ayanamsa_resolve": "not_applied_tropical",
        })
        return ay_in, meta  # value is ignored downstream anyway

    # Sidereal
    # 1) explicit numeric wins
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

    # 2) string key → try to compute degrees with jd_tt
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
        # Could not compute — pass the key through (engine may fallback to 0°)
        meta.update({
            "ayanamsa_input": key,
            "ayanamsa_effective": key,
            "ayanamsa_resolve": "pass_through_string_fallback",
            "ayanamsa_key": key,
            "warning": "could_not_compute_ayanamsa_degrees_from_key",
        })
        return key, meta

    # 3) last resort default
    meta.update({
        "ayanamsa_input": key,
        "ayanamsa_effective": key,
        "ayanamsa_resolve": "default_key_passthrough",
    })
    return key, meta


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

    # Try discovering legacy detectors availability without importing at module load.
    try:
        from app.core.vedic_predictive import detect_yogas as _legacy_detect  # type: ignore
        _legacy_ok = callable(_legacy_detect)
        _legacy_sig = sigs(_legacy_detect)
    except Exception:
        _legacy_ok = False
        _legacy_sig = None

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
        "varga_engine_present": _VARGA_OK,
        "timescales_present": _TS_OK,
        "ayanamsa_adapter_present": _AY_OK,
        # Yoga diagnostics
        "yoga_core_present": _YOGA_CORE_OK,
        "yoga_core_sig": sigs(_compute_yogas_core) if _YOGA_CORE_OK else None,
        "yoga_legacy_present": _legacy_ok,
        "yoga_legacy_sig": _legacy_sig,
        "rl_cap_per_min": RL_VEDIC_PREDICTIVE,
        "rl_bucket_key": "20",
        "dut1_seconds_env": _env_dut1_seconds(),
    }), 200


# ──────────────── Dasha routes (unchanged) ────────────────
@vedic_api.post("/dasha/vimshottari")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)  # shared bucket "20"
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


# ──────────────── Yoga routes (core first; legacy fallback; NO shim required) ────────────────

def _parse_listish(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return []

def _pick_points_deg(body: Dict[str, Any]) -> Dict[str, float]:
    # Legacy fallback when yoga core is unavailable or fails
    for key in ("points_deg", "longitudes", "longitudes_by_name"):
        raw = body.get(key)
        if isinstance(raw, dict):
            out: Dict[str, float] = {}
            for k, v in raw.items():
                f = _coerce_float(v)
                if f is not None:
                    out[str(k).lower()] = f
            if out:
                return out
    return {}

def _pick_cusps_deg(body: Dict[str, Any]) -> List[float]:
    for key in ("cusps_deg", "house_cusps", "houses"):
        raw = body.get(key)
        if isinstance(raw, list) and len(raw) >= 12:
            vals: List[float] = []
            for i in range(12):
                f = _coerce_float(raw[i])
                if f is None:
                    break
                vals.append(float(f))
            if len(vals) == 12:
                return vals
    return []


@vedic_api.get("/yoga/catalog")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_yoga_catalog():
    # Prefer direct catalog from yoga core if present
    if _YOGA_CORE_OK and callable(_yoga_list):
        try:
            catalog = _yoga_list()  # type: ignore[misc]
            return jsonify({"ok": True, "catalog": catalog, "meta": {"route": "yoga/catalog", "branch": "yoga_core"}}), 200
        except Exception as e:
            return jsonify({"ok": False, "error": "yoga_catalog_failed", "detail": str(e)}), 400

    # If yoga core is missing, expose a small hint
    return jsonify({"ok": False, "error": "yoga_core_unavailable"}), 503


@vedic_api.post("/yoga/detect")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_yoga_detect():
    """
    Detect yogas using the primary engine (app.core.yoga) when available.
    Fallback: legacy-basic detectors require `points_deg` and `cusps_deg` (12).
    Body (accepts either civil+place or precomputed geometry):
    {
      "date": "YYYY-MM-DD", "time": "HH:MM[:SS]", "tz": "Area/City",
      "latitude": 12.34, "longitude": 56.78,
      "ayanamsa": "lahiri" | 23.85,
      "house_system": "placidus|whole_sign|...",

      // Optional filters:
      "include": ["Gajakesari","Parivartana"],             // or comma string
      "enable_catalog_tags": ["classic","mahapurusha"],    // or comma string
      "disable_catalog_tags": ["experimental"],

      // Legacy fallback only:
      "points_deg": { "sun": 123.4, "moon": 210.6, ... },  # sidereal longitudes expected
      "cusps_deg": [Asc, 2nd, ..., 12th]                   # 12 floats, degrees
    }
    """
    body = request.get_json(silent=True) or {}

    # Normalize civil fields (date/time/tz/lat/lon/ayanamsa/jd_tt).
    if normalize_vim_payload is None:
        norm = {}
        warns: List[str] = ["validator_unavailable"]
        tz_norm = str(body.get("tz") or body.get("place_tz") or "UTC")
    else:
        norm, warns, tz_norm = normalize_vim_payload(body)  # type: ignore[misc]

    # Pass through yoga-specific hints
    norm["house_system"] = body.get("house_system") or norm.get("house_system") or "placidus"
    norm["include"] = _parse_listish(body.get("include"))
    norm["enable_catalog_tags"] = _parse_listish(body.get("enable_catalog_tags"))
    norm["disable_catalog_tags"] = _parse_listish(body.get("disable_catalog_tags"))

    # Legacy fallback inputs in request (used only if core not present / fails)
    pts = _pick_points_deg(body)
    cusps = _pick_cusps_deg(body)
    if pts:
        norm["points_deg"] = pts
    if cusps:
        norm["cusps_deg"] = cusps

    # ---------------- Primary path: yoga core ----------------
    if _YOGA_CORE_OK and callable(_compute_yogas_core):
        req: Dict[str, Any] = {
            "date": norm.get("date"),
            "time": norm.get("time"),
            "tz": norm.get("tz") or norm.get("tz_name") or norm.get("place_tz"),
            "latitude": norm.get("latitude"),
            "longitude": norm.get("longitude"),
            "ayanamsa": norm.get("ayanamsa"),
            "house_system": norm.get("house_system"),
            "include": norm.get("include") or [],
            "enable_catalog_tags": norm.get("enable_catalog_tags") or [],
            "disable_catalog_tags": norm.get("disable_catalog_tags") or [],
            # precomputed geometry (some cores can use it)
            "points_deg": norm.get("points_deg"),
            "cusps_deg": norm.get("cusps_deg"),
        }
        try:
            res = _compute_yogas_core(req)  # type: ignore[misc]
            if isinstance(res, dict) and res.get("ok"):
                out = {
                    "ok": True,
                    "yogas": res.get("yogas", []),
                    "present": res.get("present", []),
                    "context": res.get("context"),
                    "warnings": (res.get("warnings", []) or []) + warns,
                    "meta": {"route": "yoga/detect", "branch": "yoga_core", "tz_normalized": tz_norm},
                }
                return jsonify(out), 200
            # If core returns not-ok, try legacy fallback if geometry provided
        except Exception as _core_err:
            # fall through to legacy if available
            pass

    # ---------------- Legacy fallback (basic detectors) ----------------
    if pts and cusps:
        try:
            # Import lazily to avoid module-load dependency
            from app.core.vedic_predictive import detect_yogas as _legacy_detect  # type: ignore
            yogas = _legacy_detect(points_deg=pts, cusps_deg=cusps)  # type: ignore[misc]
            out = {
                "ok": True,
                "yogas": yogas,
                "present": [y.get("yoga") for y in yogas],
                "warnings": ["yoga_core_unavailable_fallback"] + warns,
                "meta": {"route": "yoga/detect", "branch": "legacy_basic", "tz_normalized": tz_norm},
            }
            return jsonify(out), 200
        except Exception as e:
            return jsonify({"ok": False, "error": "legacy_detector_failed", "detail": str(e)}), 400

    # No engine available and no fallback geometry
    return jsonify({
        "ok": False,
        "error": "yoga_unavailable_or_insufficient_inputs",
        "meta": {"route": "yoga/detect", "branch": "none", "tz_normalized": tz_norm},
        "warnings": warns,
    }), (503 if not _YOGA_CORE_OK else 400)


# ──────────────── Varga routes (varga_charts) ────────────────
@vedic_api.post("/varga/position")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_varga_position():
    """
    Compute a SINGLE point’s varga placement.

    Body:
      {
        "varga": "D9",
        "lon_deg": 123.456,         # tropical by default unless method='sidereal'
        "method": "sidereal|tropical",
        "ayanamsa": "lahiri" | 22.5,
        // optionally include one of:
        "jd_tt": 2447762.123,
        "date": "1989-07-26", "time": "20:44", "tz": "Asia/Kolkata"
      }
    """
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
    """
    Compute placements for MANY points in a SINGLE varga.

    Body:
      {
        "varga": "D9",
        "longitudes": { "Sun": 123.4, "Moon": 210.6, ... },  # tropical unless method='sidereal'
        "method": "sidereal|tropical",
        "ayanamsa": "lahiri" | 22.5,
        // optionally include either jd_tt or {date,time,tz}
        "jd_tt": 2447762.123,
        "date": "1989-07-26", "time": "20:44", "tz": "Asia/Kolkata"
      }
    """
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
    """
    Compute placements for MANY points across MULTIPLE vargas.

    Body:
      {
        "vargas": ["D1","D9","D10"],
        "longitudes": { "Sun": 123.4, "Moon": 210.6, ... },  # tropical unless method='sidereal'
        "method": "sidereal|tropical",
        "ayanamsa": "lahiri" | 22.5,
        // optionally include either jd_tt or {date,time,tz}
        "jd_tt": 2447762.123,
        "date": "1989-07-26", "time": "20:44", "tz": "Asia/Kolkata"
      }
    """
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
            "placements": placements,  # { 'D9': { 'Sun': {...}, ... }, ... }
            "options": {"zodiac_mode": method, "ayanamsa": ay_for_engine},
            "meta": _wrap_varga_meta("varga/many", method, ay_meta),
        }
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"ok": False, "error": "varga_many_failed", "detail": str(e)}), 400
