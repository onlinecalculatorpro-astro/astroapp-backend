# app/api/vedic_routes.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import inspect
import os
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from app.utils.ratelimit import rate_limit  # fixed shared bucket

# ──────────────────────────────────────────────────────────────────────────────
# Blueprint & rate limits
# ──────────────────────────────────────────────────────────────────────────────
vedic_api = Blueprint("vedic_api", __name__)
RL_VEDIC_PREDICTIVE = int(os.getenv("ASTRO_RL_VEDIC_PREDICTIVE_PER_MIN", "20"))

def fixed_key(*_a, **_k) -> str:
    """Shared bucket key ('20') used by all predictive/varga/yoga/horary calls."""
    return "20"


# ──────────────────────────────────────────────────────────────────────────────
# Validators
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.vedic_validator import (  # type: ignore
        normalize_vim_payload,         # dasha & common tz/site hints
        normalize_yoga_payload,        # yoga (mode-C)
        normalize_gochar_payload,      # gochar/drishti
        normalize_ingress_payload,     # ingress (rashi/nakshatra)
        normalize_stations_payload,    # stations
        # (NEW) strength helpers
        normalize_shadbala_payload,    # shadbala (if present)
        normalize_ashtakavarga_payload # ashtakavarga (if present)
    )
    _VALIDATOR_IMPORT_ERR = None
except Exception as _e:
    normalize_vim_payload = None            # type: ignore
    normalize_yoga_payload = None           # type: ignore
    normalize_gochar_payload = None         # type: ignore
    normalize_ingress_payload = None        # type: ignore
    normalize_stations_payload = None       # type: ignore
    normalize_shadbala_payload = None       # type: ignore
    normalize_ashtakavarga_payload = None   # type: ignore
    _VALIDATOR_IMPORT_ERR = repr(_e)


# ──────────────────────────────────────────────────────────────────────────────
# Dasha engines: registry (preferred) + direct module fallbacks
# ──────────────────────────────────────────────────────────────────────────────
_compute_dasha_registry = None
_available_schemes_fn = None
try:
    from app.core.dasha_registry import compute_dasha as _compute_dasha_registry  # type: ignore
    from app.core.dasha_registry import available_schemes as _available_schemes_fn  # type: ignore
except Exception:
    _compute_dasha_registry = None  # type: ignore
    _available_schemes_fn = None    # type: ignore

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
    from app.core.varga_charts import (  # type: ignore
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
# Timescales + Ayanāṁśa (used by varga helpers)
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
# Yoga core (diag visibility); route uses vedic_predictive.yoga_detect
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
        # Fallback: app/core/yogas.py
        from app.core.yogas import (                                # type: ignore
            compute_yogas as _compute_yogas,
            list_registered_yogas as _yoga_list,
        )
        _YOGA_OK = True
    except Exception:
        _YOGA_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Gochar / Ingress / Stations wrappers (for diagnostics)
# ──────────────────────────────────────────────────────────────────────────────
_GOCHAR_OK = False
_GOCHAR_BRANCH = "none"
try:
    from app.core.vedic_gochar import (          # type: ignore
        gochar_drishti as _gochar_drishti,
        ingresses_rashi as _ingresses_rashi,
        ingresses_nakshatra as _ingresses_nakshatra,
        stations_retro_direct as _stations_retro_direct,
        feature_drishti_proximity as _feature_drishti_proximity,
    )
    _GOCHAR_OK = True
    _GOCHAR_BRANCH = "vedic_gochar"
except Exception:
    _GOCHAR_OK = False
    _GOCHAR_BRANCH = "none"
    _gochar_drishti = None               # type: ignore
    _ingresses_rashi = None              # type: ignore
    _ingresses_nakshatra = None          # type: ignore
    _stations_retro_direct = None        # type: ignore
    _feature_drishti_proximity = None    # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Predictive helpers from vedic_predictive (classic)
# ──────────────────────────────────────────────────────────────────────────────
_PRED_BRANCH = "none"
_predict_dasha_periods = None   # type: ignore
_yoga_detect = None             # type: ignore
_shadbala_wrapper = None        # type: ignore
_ashtakavarga_wrapper = None    # type: ignore

try:
    from app.core.vedic_predictive import (  # type: ignore
        predict_dasha_periods as _predict_dasha_periods,
        yoga_detect as _yoga_detect,
        shadbala as _shadbala_wrapper,
        compute_ashtakavarga as _ashtakavarga_wrapper,  # shim exists in predictive
    )
    _PRED_BRANCH = "vedic_predictive"
except Exception:
    _PRED_BRANCH = "none"


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Payload-wired helpers from vedic_predictive (preferred)
# ──────────────────────────────────────────────────────────────────────────────
_PRED_PAYLOAD_BRANCH = "none"
_dasha_from_payload = None
_yoga_from_payload = None
_gochar_from_payload = None
_ingresses_rashi_from_payload = None
_ingresses_nakshatra_from_payload = None
_stations_from_payload = None
_shadbala_from_payload = None
_ashtakavarga_from_payload = None
_horary_from_payload = None  # NEW
try:
    from app.core.vedic_predictive import (  # type: ignore
        dasha_from_payload as _dasha_from_payload,
        yoga_from_payload as _yoga_from_payload,
        gochar_from_payload as _gochar_from_payload,
        ingresses_rashi_from_payload as _ingresses_rashi_from_payload,
        ingresses_nakshatra_from_payload as _ingresses_nakshatra_from_payload,
        stations_from_payload as _stations_from_payload,
        shadbala_from_payload as _shadbala_from_payload,
        ashtakavarga_from_payload as _ashtakavarga_from_payload,
        horary_from_payload as _horary_from_payload,  # NEW
    )
    _PRED_PAYLOAD_BRANCH = "predictive.payload"
except Exception:
    _PRED_PAYLOAD_BRANCH = "none"


# ──────────────────────────────────────────────────────────────────────────────
# Horary (Prasna) core
# ──────────────────────────────────────────────────────────────────────────────
_HORARY_OK = False
_horary_err = None
try:
    from app.core.horary import (  # type: ignore
        analyze_prasna_enhanced as _analyze_prasna_enhanced,
        HoraryInput as _HoraryInput,
        QuestionType as _QuestionType,
    )
    _HORARY_OK = True
except Exception as _he:
    _HORARY_OK = False
    _horary_err = repr(_he)
    _analyze_prasna_enhanced = None  # type: ignore
    _HoraryInput = None               # type: ignore
    _QuestionType = None              # type: ignore


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
    out = dict(out or {})
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
    """Flexible flags for Vimśottarī (method/observer/ayanamsa/place/etc.)."""
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


# ---- tiny module-call shim (generic) -----------------------------------------
def _call_single_param_or_kwargs(fn, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls a function that may accept either a single 'payload' positional arg
    or a set of keyword args. Returns the dict result or raises.
    """
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


# ---- dasha window helpers ----------------------------------------------------
def _iso_to_utc_dt(s: str, default_time: str = "00:00:00") -> Optional[datetime]:
    if not s:
        return None
    t = s.strip()
    try:
        if "T" not in t:
            t = f"{t}T{default_time}"
        # Allow "Z"
        t = t.replace("Z", "+00:00")
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _pick_window(body: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Accepts:
      - window/time_window/range: {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}
      - date_from/date_to
      - from/to
    Returns UTC datetimes spanning the civil days.
    """
    win = body.get("window") or body.get("time_window") or body.get("range") or {}
    d0 = (win or {}).get("from") or (win or {}).get("start")
    d1 = (win or {}).get("to") or (win or {}).get("end")
    d0 = d0 or body.get("date_from") or body.get("from")
    d1 = d1 or body.get("date_to") or body.get("to")

    dt0 = _iso_to_utc_dt(str(d0 or ""), "00:00:00")
    dt1 = _iso_to_utc_dt(str(d1 or ""), "23:59:59")
    return dt0, dt1


# ──────────────────────────────────────────────────────────────────────────────
# Core runners (dasha): validator → registry (preferred) → module/predictive
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
            "detail": f"app.core.vedic_validator import failed: {_VALIDATOR_IMPORT_ERR}",
        }

    norm, warns, tz_norm = normalize_vim_payload(body)  # type: ignore[misc]

    # If a civil window is provided and predictive helper exists, prefer it.
    # (Keeps old behavior when no window provided.)
    if callable(_predict_dasha_periods):
        dt0, dt1 = _pick_window(body)
        if dt0 and dt1:
            natal_chart = {
                "date": norm.get("date"),
                "time": norm.get("time"),
                "tz": norm.get("tz") or tz_norm,
                "place_tz": norm.get("tz") or tz_norm,
                "latitude": norm.get("latitude"),
                "longitude": norm.get("longitude"),
                "elevation_m": norm.get("elevation_m"),
                "ayanamsa": norm.get("ayanamsa"),
                "jd_tt": norm.get("jd_tt"),
                "jd_ut1": norm.get("jd_ut1"),
            }
            try:
                res = _predict_dasha_periods(
                    natal_chart=natal_chart,
                    start_date=dt0,
                    end_date=dt1,
                    dasha_system=scheme_key,
                    include_antardasha=_levels_from(norm) >= 2,
                    levels=_levels_from(norm),
                )
                if isinstance(res, dict) and res.get("ok"):
                    return _wrap_ok(res, warns, tz_norm, branch=f"predictive.{scheme_key}", route_name=route_name)
            except Exception:
                # fall back silently if predictive path fails
                pass

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
    # Prefer payload helper if present
    if callable(_dasha_from_payload):
        try:
            res = _dasha_from_payload(payload)  # type: ignore
            if isinstance(res, dict) and res.get("ok"):
                res.setdefault("meta", {}).update({"route": "vimshottari", "branch": _PRED_PAYLOAD_BRANCH,
                                                   "tz_normalized": _tz_from_payload(payload)})
                return res
        except Exception:
            pass
    # Fallback chain
    return _run_with_registry_first(
        route_name="vimshottari",
        scheme_key="vimshottari",
        civic_builder=_build_civic_payload_vim,
        module_fallback_fn=_compute_vim_module,
        body=payload,
    )


def _run_ashtottari(payload: Dict[str, Any]) -> Dict[str, Any]:
    if callable(_dasha_from_payload):
        try:
            # dasha_from_payload defaults to vimshottari; preserve old behavior by setting scheme in body
            body = dict(payload or {})
            body.setdefault("scheme", "ashtottari")
            res = _dasha_from_payload(body)  # type: ignore
            if isinstance(res, dict) and res.get("ok") and res.get("system") in ("ashtottari", "ashto", "ashtottari"):
                res.setdefault("meta", {}).update({"route": "ashtottari", "branch": _PRED_PAYLOAD_BRANCH,
                                                   "tz_normalized": _tz_from_payload(payload)})
                return res
        except Exception:
            pass
    return _run_with_registry_first(
        route_name="ashtottari",
        scheme_key="ashtottari",
        civic_builder=_build_civic_payload_ashto,
        module_fallback_fn=_compute_ashto_module,
        body=payload,
    )


def _run_yogini(payload: Dict[str, Any]) -> Dict[str, Any]:
    if callable(_dasha_from_payload):
        try:
            body = dict(payload or {})
            body.setdefault("scheme", "yogini")
            res = _dasha_from_payload(body)  # type: ignore
            if isinstance(res, dict) and res.get("ok") and str(res.get("system","")).startswith("yogini"):
                res.setdefault("meta", {}).update({"route": "yogini", "branch": _PRED_PAYLOAD_BRANCH,
                                                   "tz_normalized": _tz_from_payload(payload)})
                return res
        except Exception:
            pass
    return _run_with_registry_first(
        route_name="yogini",
        scheme_key="yogini",
        civic_builder=_build_civic_payload_yogini,
        module_fallback_fn=_compute_yogini_module,
        body=payload,
    )


def _run_chara(payload: Dict[str, Any]) -> Dict[str, Any]:
    if callable(_dasha_from_payload):
        try:
            body = dict(payload or {})
            body.setdefault("scheme", "chara")
            res = _dasha_from_payload(body)  # type: ignore
            if isinstance(res, dict) and res.get("ok") and res.get("system") == "chara":
                res.setdefault("meta", {}).update({"route": "chara", "branch": _PRED_PAYLOAD_BRANCH,
                                                   "tz_normalized": _tz_from_payload(payload)})
                return res
        except Exception:
            pass
    return _run_with_registry_first(
        route_name="chara",
        scheme_key="chara",
        civic_builder=_build_civic_payload_chara,
        module_fallback_fn=_compute_chara_module,
        body=payload,
    )


def _run_kalachakra(payload: Dict[str, Any]) -> Dict[str, Any]:
    if callable(_dasha_from_payload):
        try:
            body = dict(payload or {})
            body.setdefault("scheme", "kalachakra")
            res = _dasha_from_payload(body)  # type: ignore
            if isinstance(res, dict) and res.get("ok") and res.get("system") == "kalachakra":
                res.setdefault("meta", {}).update({"route": "kalachakra", "branch": _PRED_PAYLOAD_BRANCH,
                                                   "tz_normalized": _tz_from_payload(payload)})
                return res
        except Exception:
            pass
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
        return float(get_ayanamsa_deg(None, key))  # type: ignore/misc
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
        "gochar_branch": _GOCHAR_BRANCH,
        # Predictive helpers (classic)
        "predictive_branch": _PRED_BRANCH,
        "predictive_dasha_present": bool(_predict_dasha_periods),
        "yoga_detect_present": bool(_yoga_detect),
        # NEW payload helpers
        "payload_branch": _PRED_PAYLOAD_BRANCH,
        "payload_dasha_present": bool(_dasha_from_payload),
        "payload_yoga_present": bool(_yoga_from_payload),
        "payload_gochar_present": bool(_gochar_from_payload),
        "payload_rashi_present": bool(_ingresses_rashi_from_payload),
        "payload_nakshatra_present": bool(_ingresses_nakshatra_from_payload),
        "payload_stations_present": bool(_stations_from_payload),
        "payload_shadbala_present": bool(_shadbala_from_payload),
        "payload_ashtakavarga_present": bool(_ashtakavarga_from_payload),
        "payload_horary_present": bool(_horary_from_payload),  # NEW
        # Strength diagnostics
        "shadbala_present": bool(_shadbala_wrapper),
        "shadbala_branch": "vedic_predictive" if _shadbala_wrapper else "none",
        "ashtakavarga_present": bool(_ashtakavarga_wrapper),
        "ashtakavarga_branch": "vedic_predictive" if _ashtakavarga_wrapper else "none",
        # Horary diagnostics
        "horary_present": _HORARY_OK,
        "horary_error": _horary_err,
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


# ──────────────── Yoga routes ────────────────
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
    Yoga detection: prefer payload-wired wrapper → fallback to legacy route that
    calls vedic_predictive.yoga_detect directly.
    """
    body = request.get_json(silent=True) or {}

    # Payload-first
    if callable(_yoga_from_payload):
        try:
            res = _yoga_from_payload(body)  # type: ignore
            if isinstance(res, dict):
                res.setdefault("meta", {}).update({"route": "yoga/detect", "tz_normalized": _tz_from_payload(body),
                                                   "branch": _PRED_PAYLOAD_BRANCH})
                return jsonify(res), (200 if res.get("ok") else 400)
        except Exception:
            pass

    # Fallback path (original)
    if normalize_yoga_payload is None:
        return jsonify({"ok": False, "error": "validator_unavailable"}), 503
    if not callable(_yoga_detect):
        return jsonify({"ok": False, "error": "yoga_engine_unavailable"}), 503

    norm, warns, tz_norm = normalize_yoga_payload(body)  # type: ignore[misc]

    if not norm.get("date") or not norm.get("time"):
        return jsonify({
            "ok": False,
            "error": "missing_date_or_time",
            "warnings": (warns or []),
            "meta": {"route": "yoga/detect", "branch": _PRED_BRANCH, "tz_normalized": tz_norm},
        }), 400

    try:
        res = _yoga_detect(norm)
    except Exception as e:
        return jsonify({"ok": False, "error": "yoga_detect_failed", "detail": str(e)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "yoga/detect", "tz_normalized": tz_norm, "branch": _PRED_BRANCH})
        if warns:
            res.setdefault("warnings", []).extend(warns)

    return jsonify(res), (200 if res.get("ok") else 400)


# ──────────────── Gochar / Ingress / Stations routes ────────────────
@vedic_api.post("/gochar/drishti")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_gochar_drishti():
    """
    Prefer payload wrapper; fallback to vedic_gochar direct call.
    """
    body = request.get_json(silent=True) or {}

    if callable(_gochar_from_payload):
        try:
            res = _gochar_from_payload(body)  # type: ignore
            if isinstance(res, dict):
                res.setdefault("meta", {}).update({"route": "gochar/drishti", "tz_normalized": _tz_from_payload(body),
                                                   "branch": _PRED_PAYLOAD_BRANCH})
                return jsonify(res), (200 if res.get("ok") else 400)
        except Exception:
            pass

    # Fallback
    if not (_GOCHAR_OK and callable(_gochar_drishti)):
        return jsonify({"ok": False, "error": "gochar_engine_unavailable"}), 503
    if normalize_gochar_payload is None:
        return jsonify({"ok": False, "error": "validator_unavailable"}), 503

    norm, warns, tz_norm = normalize_gochar_payload(body)  # type: ignore[misc]
    tr = norm.get("time_range")
    if not (isinstance(tr, list) and len(tr) == 2 and tr[0] and tr[1]):
        return jsonify({
            "ok": False,
            "error": "missing_date_window",
            "warnings": (warns or []),
            "meta": {"route": "gochar/drishti", "branch": _GOCHAR_BRANCH, "tz_normalized": tz_norm},
        }), 400

    if norm.get("fatal"):
        return jsonify({
            "ok": False,
            "error": norm.get("fatal"),
            "warnings": (warns or []) + ["fatal"],
            "meta": {"route": "gochar/drishti", "branch": _GOCHAR_BRANCH, "tz_normalized": tz_norm},
        }), 400

    try:
        res = _gochar_drishti(
            natal_chart=norm.get("natal_chart") or {},
            date_from=tr[0], date_to=tr[1],
            transiting_bodies=norm.get("movers"),
            natal_targets=norm.get("natal_targets"),
            zodiac_mode=norm.get("zodiac_mode", "sidereal"),
            ayanamsa=norm.get("ayanamsa", "lahiri"),
            frame=str(norm.get("frame") or "ecliptic-of-date"),
            include_nodes=bool(norm.get("include_nodes", False)),
            treat_nodes_like_saturn=bool(norm.get("treat_nodes_like_saturn", False)),
            orb_deg=float(norm.get("orb_deg", 12.0)),
            orb_map=norm.get("orb_map") or {},
            step_minutes=norm.get("step_minutes", "auto"),
            tz_name=tz_norm,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "gochar_drishti_failed", "detail": str(e)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "gochar/drishti", "tz_normalized": tz_norm, "branch": _GOCHAR_BRANCH})
        if warns:
            res.setdefault("warnings", []).extend(warns)
    return jsonify(res), (200 if res.get("ok") else 400)


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
    body = request.get_json(silent=True) or {}

    if callable(_ingresses_rashi_from_payload):
        try:
            res = _ingresses_rashi_from_payload(body)  # type: ignore
            if isinstance(res, dict):
                res.setdefault("meta", {}).update({"route": "ingress/rashi", "tz_normalized": _tz_from_payload(body),
                                                   "branch": _PRED_PAYLOAD_BRANCH})
                return jsonify(res), (200 if res.get("ok") else 400)
        except Exception:
            pass

    # Fallback
    if not (_GOCHAR_OK and callable(_ingresses_rashi)):
        return jsonify({"ok": False, "error": "gochar_engine_unavailable"}), 503
    if normalize_ingress_payload is None:
        return jsonify({"ok": False, "error": "validator_unavailable"}), 503

    norm, warns, tz_norm = normalize_ingress_payload(body)  # type: ignore[misc]
    tr = norm.get("time_range")
    if not (isinstance(tr, list) and len(tr) == 2 and tr[0] and tr[1]):
        return jsonify({"ok": False, "error": "missing_date_window"}), 400

    try:
        res = _ingresses_rashi(
            date_from=tr[0],
            date_to=tr[1],
            movers=norm.get("movers"),
            zodiac_mode=norm.get("zodiac_mode", "sidereal"),
            ayanamsa=norm.get("ayanamsa", "lahiri"),
            frame=str(norm.get("frame") or "ecliptic-of-date"),
            observer=("topocentric" if bool(norm.get("topocentric", False)) else "geocentric"),
            latitude=_coerce_float(norm.get("latitude")),
            longitude=_coerce_float(norm.get("longitude")),
            elevation_m=_coerce_float(norm.get("elevation_m")),
            step_minutes=norm.get("step_minutes", "auto"),
            tz_name=tz_norm,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "rashi_ingress_failed", "detail": str(e)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "ingress/rashi", "tz_normalized": tz_norm, "branch": _GOCHAR_BRANCH})
        if warns:
            res.setdefault("warnings", []).extend(warns)
    return jsonify(res), (200 if res.get("ok") else 400)


@vedic_api.post("/ingress/nakshatra")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_ingress_nakshatra():
    body = request.get_json(silent=True) or {}

    if callable(_ingresses_nakshatra_from_payload):
        try:
            res = _ingresses_nakshatra_from_payload(body)  # type: ignore
            if isinstance(res, dict):
                res.setdefault("meta", {}).update({"route": "ingress/nakshatra", "tz_normalized": _tz_from_payload(body),
                                                   "branch": _PRED_PAYLOAD_BRANCH})
                return jsonify(res), (200 if res.get("ok") else 400)
        except Exception:
            pass

    # Fallback
    if not (_GOCHAR_OK and callable(_ingresses_nakshatra)):
        return jsonify({"ok": False, "error": "gochar_engine_unavailable"}), 503
    if normalize_ingress_payload is None:
        return jsonify({"ok": False, "error": "validator_unavailable"}), 503

    norm, warns, tz_norm = normalize_ingress_payload(body)  # type: ignore[misc]
    tr = norm.get("time_range")
    if not (isinstance(tr, list) and len(tr) == 2 and tr[0] and tr[1]):
        return jsonify({"ok": False, "error": "missing_date_window"}), 400

    try:
        res = _ingresses_nakshatra(
            date_from=tr[0],
            date_to=tr[1],
            movers=norm.get("movers"),
            zodiac_mode=norm.get("zodiac_mode", "sidereal"),
            ayanamsa=norm.get("ayanamsa", "lahiri"),
            frame=str(norm.get("frame") or "ecliptic-of-date"),
            observer=("topocentric" if bool(norm.get("topocentric", False)) else "geocentric"),
            latitude=_coerce_float(norm.get("latitude")),
            longitude=_coerce_float(norm.get("longitude")),
            elevation_m=_coerce_float(norm.get("elevation_m")),
            step_minutes=norm.get("step_minutes", "auto"),
            tz_name=tz_norm,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "nakshatra_ingress_failed", "detail": str(e)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "ingress/nakshatra", "tz_normalized": tz_norm, "branch": _GOCHAR_BRANCH})
        if warns:
            res.setdefault("warnings", []).extend(warns)
    return jsonify(res), (200 if res.get("ok") else 400)


@vedic_api.post("/stations")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def vedic_stations():
    body = request.get_json(silent=True) or {}

    if callable(_stations_from_payload):
        try:
            res = _stations_from_payload(body)  # type: ignore
            if isinstance(res, dict):
                res.setdefault("meta", {}).update({"route": "stations", "tz_normalized": _tz_from_payload(body),
                                                   "branch": _PRED_PAYLOAD_BRANCH})
                return jsonify(res), (200 if res.get("ok") else 400)
        except Exception:
            pass

    # Fallback
    if not (_GOCHAR_OK and callable(_stations_retro_direct)):
        return jsonify({"ok": False, "error": "gochar_engine_unavailable"}), 503
    if normalize_stations_payload is None:
        return jsonify({"ok": False, "error": "validator_unavailable"}), 503

    norm, warns, tz_norm = normalize_stations_payload(body)  # type: ignore[misc]
    tr = norm.get("time_range")
    if not (isinstance(tr, list) and len(tr) == 2 and tr[0] and tr[1]):
        return jsonify({"ok": False, "error": "missing_date_window"}), 400

    try:
        res = _stations_retro_direct(
            date_from=tr[0],
            date_to=tr[1],
            movers=norm.get("movers"),
            zodiac_mode=norm.get("zodiac_mode", "sidereal"),
            ayanamsa=norm.get("ayanamsa", "lahiri"),
            frame=str(norm.get("frame") or "ecliptic-of-date"),
            observer=("topocentric" if bool(norm.get("topocentric", False)) else "geocentric"),
            latitude=_coerce_float(norm.get("latitude")),
            longitude=_coerce_float(norm.get("longitude")),
            elevation_m=_coerce_float(norm.get("elevation_m")),
            step_minutes=norm.get("step_minutes", "auto"),
            tz_name=tz_norm,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": "stations_failed", "detail": str(e)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "stations", "tz_normalized": tz_norm, "branch": _GOCHAR_BRANCH})
        if warns:
            res.setdefault("warnings", []).extend(warns)
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


# =============================================================================
# Śaḍbala & Aṣṭakavarga routes
# =============================================================================

def _normalize_strength_payload_generic(body: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], str]:
    """
    Fallback normalizer for strength endpoints when a dedicated validator
    is not available in vedic_validator. Mirrors 'yoga' normalizer semantics
    and, importantly, *preserves* house/angles fields so engines can use them.
    """
    if callable(normalize_yoga_payload):
        norm, warns, tz = normalize_yoga_payload(body)  # type: ignore/misc
    else:
        # Last-resort: minimal pass-through
        warns = ["validator_unavailable_minimal_fallback"]
        tz = str(body.get("tz") or body.get("place_tz") or "UTC")
        norm = {
            "date": body.get("date") or body.get("birth_date"),
            "time": body.get("time") or body.get("birth_time") or "12:00:00",
            "tz": tz,
            "latitude": _coerce_float(body.get("latitude") or body.get("lat")),
            "longitude": _coerce_float(body.get("longitude") or body.get("lon")),
            "elevation_m": _coerce_float(body.get("elevation_m") or body.get("elevation")),
            "zodiac_mode": _norm_method(body.get("zodiac_mode") or body.get("mode") or body.get("method") or "sidereal"),
            # ⬇ improvement: accept ayanamsa_key alias too
            "ayanamsa": _norm_ayanamsa(body.get("ayanamsa") or body.get("ayanamsa_key")),
            "place_tz": tz,
        }

    # ── Do NOT drop helpful extras (engines may rely on these) ──────────────
    # Houses / angles
    if body.get("house_system") is not None:
        norm["house_system"] = body["house_system"]
    if body.get("house_cusps_deg") is not None:
        norm["house_cusps_deg"] = body["house_cusps_deg"]
    if isinstance(body.get("houses"), dict):
        norm["houses"] = body["houses"]
    if isinstance(body.get("angles"), dict):
        norm["angles"] = body["angles"]

    # Engine-specific optional knobs
    if body.get("include_components") is not None:
        norm["include_components"] = body["include_components"]
    if body.get("vargas") is not None:
        norm["vargas"] = body["vargas"]

    return norm, warns, tz


def _strength_call(fn, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Strength engines may want payload or kwargs; unify the call & enforce dict return."""
    out = _call_single_param_or_kwargs(fn, payload)
    if not isinstance(out, dict):
        raise TypeError(f"invalid_return:{type(out).__name__}")
    return out


def _run_shadbala(body: Dict[str, Any]) -> Dict[str, Any]:
    # Payload-first
    if callable(_shadbala_from_payload):
        try:
            res = _shadbala_from_payload(body)  # type: ignore
            if isinstance(res, dict):
                res.setdefault("meta", {}).update({"route": "strength/shadbala",
                                                   "tz_normalized": _tz_from_payload(body),
                                                   "branch": _PRED_PAYLOAD_BRANCH})
                return res
        except Exception:
            pass

    # Predictive wrapper fallback (classic)
    if _shadbala_wrapper is None:
        return {"ok": False, "error": "shadbala_engine_unavailable",
                "meta": {"route": "strength/shadbala", "branch": "none"}}

    if callable(normalize_shadbala_payload):
        norm, warns, tz_norm = normalize_shadbala_payload(body)  # type: ignore/misc
    else:
        norm, warns, tz_norm = _normalize_strength_payload_generic(body)

    if not norm.get("date") or not norm.get("time"):
        return {"ok": False, "error": "missing_date_or_time", "warnings": warns,
                "meta": {"route": "strength/shadbala", "branch": "vedic_predictive", "tz_normalized": tz_norm}}
    if norm.get("latitude") is None or norm.get("longitude") is None:
        return {"ok": False, "error": "missing_coordinates", "warnings": warns,
                "meta": {"route": "strength/shadbala", "branch": "vedic_predictive", "tz_normalized": tz_norm}}

    natal_chart = {
        "date": norm["date"],
        "time": norm["time"],
        "place_tz": norm.get("tz") or tz_norm,
        "tz": norm.get("tz") or tz_norm,
        "latitude": float(norm["latitude"]),
        "longitude": float(norm["longitude"]),
        "elevation_m": norm.get("elevation_m"),
    }

    try:
        res = _shadbala_wrapper(
            natal_chart=natal_chart,
            zodiac_mode=norm.get("zodiac_mode", "sidereal"),
            ayanamsa=norm.get("ayanamsa", "lahiri"),
            house_system=norm.get("house_system", "placidus"),
            include_velocity=bool(body.get("include_velocity", True)),
            prefer_houses_advanced=bool(body.get("prefer_houses_advanced", True)),
            observer=str(body.get("observer") or "geocentric"),
        )
    except Exception as e:
        return {"ok": False, "error": "shadbala_failed", "detail": str(e),
                "meta": {"route": "strength/shadbala", "branch": "vedic_predictive", "tz_normalized": tz_norm}}

    res.setdefault("meta", {}).update({"route": "strength/shadbala",
                                       "tz_normalized": tz_norm, "branch": "vedic_predictive"})
    if warns:
        res.setdefault("warnings", []).extend(warns)
    return res


def _run_ashtakavarga(body: Dict[str, Any]) -> Dict[str, Any]:
    # Payload-first
    if callable(_ashtakavarga_from_payload):
        try:
            res = _ashtakavarga_from_payload(body)  # type: ignore
            if isinstance(res, dict):
                res.setdefault("meta", {}).update({"route": "ashtakavarga",
                                                   "tz_normalized": _tz_from_payload(body),
                                                   "branch": _PRED_PAYLOAD_BRANCH})
                return res
        except Exception:
            pass

    # Classic fallback using predictive shim
    if _ashtakavarga_wrapper is None:
        return {
            "ok": False,
            "error": "ashtakavarga_engine_unavailable",
            "meta": {"route": "ashtakavarga", "branch": "none"},
        }

    if callable(normalize_ashtakavarga_payload):
        norm, warns, tz_norm = normalize_ashtakavarga_payload(body)  # type: ignore/misc
    else:
        norm, warns, tz_norm = _normalize_strength_payload_generic(body)

    if not norm.get("date") or not norm.get("time"):
        return {
            "ok": False,
            "error": "missing_date_or_time",
            "warnings": warns,
            "meta": {"route": "ashtakavarga", "branch": "vedic_predictive", "tz_normalized": tz_norm},
        }
    if norm.get("latitude") is None or norm.get("longitude") is None:
        return {
            "ok": False,
            "error": "missing_coordinates",
            "warnings": warns,
            "meta": {"route": "ashtakavarga", "branch": "vedic_predictive", "tz_normalized": tz_norm},
        }

    natal_chart = {
        "date": norm["date"],
        "time": norm["time"],
        "place_tz": norm.get("tz") or tz_norm,
        "tz": norm.get("tz") or tz_norm,
        "latitude": float(norm["latitude"]),
        "longitude": float(norm["longitude"]),
        "elevation_m": norm.get("elevation_m"),
        "angles": body.get("angles"),
    }

    try:
        res = _ashtakavarga_wrapper(
            {
                "date": natal_chart["date"],
                "time": natal_chart["time"],
                "tz": natal_chart["tz"],
                "latitude": natal_chart["latitude"],
                "longitude": natal_chart["longitude"],
                "elevation_m": natal_chart["elevation_m"],
                "zodiac_mode": norm.get("zodiac_mode", "sidereal"),
                "ayanamsa": norm.get("ayanamsa", "lahiri"),
                "house_system": norm.get("house_system", "placidus"),
                "angles": body.get("angles"),
                "ruleset": body.get("ruleset"),
                "ruleset_map": body.get("ruleset_map"),
                "include": body.get("include") or body.get("include_keys"),
                "spec_path": body.get("spec_path") or body.get("spec") or norm.get("spec_path"),
            }
        )
    except Exception as e:
        return {
            "ok": False,
            "error": "ashtakavarga_failed",
            "detail": str(e),
            "meta": {"route": "ashtakavarga", "branch": "vedic_predictive", "tz_normalized": tz_norm},
        }

    res.setdefault("meta", {})
    res["meta"].update({"route": "ashtakavarga", "tz_normalized": tz_norm, "branch": "vedic_predictive"})
    if warns:
        res.setdefault("warnings", []).extend(warns)
    return res


@vedic_api.post("/strength/shadbala")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def route_shadbala():
    body = request.get_json(silent=True) or {}
    res = _run_shadbala(body)
    status = 200 if (isinstance(res, dict) and res.get("ok")) else (503 if str(res.get("error","")).endswith("unavailable") else 400)
    return jsonify(res), status


@vedic_api.post("/strength/ashtakavarga")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def route_strength_ashtakavarga():
    body = request.get_json(silent=True) or {}
    res = _run_ashtakavarga(body)
    status = 200 if (isinstance(res, dict) and res.get("ok")) else (503 if str(res.get("error","")).endswith("unavailable") else 400)
    return jsonify(res), status


# Back-compat alias
@vedic_api.post("/ashtakavarga")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def route_ashtakavarga_alias():
    body = request.get_json(silent=True) or {}
    res = _run_ashtakavarga(body)
    status = 200 if (isinstance(res, dict) and res.get("ok")) else (503 if str(res.get("error","")).endswith("unavailable") else 400)
    return jsonify(res), status


# =============================================================================
# Horary (Prasna) routes
# =============================================================================

def _parse_question_type(val: Any) -> Optional[Any]:
    """
    Accepts enum name/value like 'job', 'JOB', 'QuestionType.JOB'.
    Returns a _QuestionType or None.
    """
    if not _HORARY_OK or _QuestionType is None or val is None:
        return None
    if isinstance(val, _QuestionType):  # type: ignore
        return val
    s = str(val).strip()
    if not s:
        return None
    s = s.replace("QuestionType.", "").strip().lower()
    for qt in _QuestionType:  # type: ignore
        if qt.value == s or qt.name.lower() == s:
            return qt
    return None


def _merge_horary_results(parashari: Dict[str, Any], kp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Non-opinionated merge: report scalar-field agreements and unique keys.
    """
    if not isinstance(parashari, dict) or not isinstance(kp, dict):
        return {"agreement_keys": [], "parashari_only_keys": [], "kp_only_keys": []}

    def _is_scalar(v: Any) -> bool:
        return isinstance(v, (str, int, float, bool))

    a_keys = set(parashari.keys())
    k_keys = set(kp.keys())
    agree = sorted(
        k for k in (a_keys & k_keys)
        if _is_scalar(parashari.get(k)) and parashari.get(k) == kp.get(k)
    )
    return {
        "agreement_keys": agree,
        "parashari_only_keys": sorted(list(a_keys - k_keys)),
        "kp_only_keys": sorted(list(k_keys - a_keys)),
    }


# --- strict input validation for horary --------------------------------------
def _validate_civil_raw(body: Dict[str, Any], tz_fallback: str) -> Optional[str]:
    """Validate raw date/time/tz from the request body (not the normalized copy)."""
    date = str(body.get("date") or body.get("birth_date") or "").strip()
    time = str(body.get("time") or body.get("birth_time") or "").strip()
    tz   = str(body.get("tz") or body.get("place_tz") or tz_fallback or "UTC").strip()

    if not date or not time:
        return "missing_date_or_time"

    if not _TS_OK or build_timescales is None:
        # Lightweight format sanity (YYYY-MM-DD / HH:MM[:SS])
        import re
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            return "invalid_date_or_time"
        if not re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", time):
            return "invalid_date_or_time"
        return None

    try:
        ts = build_timescales(date, time, tz, _env_dut1_seconds())  # type: ignore
        jd = ts.get("jd_tt") if isinstance(ts, dict) else getattr(ts, "jd_tt", None)
        float(jd)  # raises if None/NaN
    except Exception:
        return "invalid_date_or_time"
    return None


def _validate_coords_raw(body: Dict[str, Any]) -> Optional[str]:
    lat = body.get("latitude") or body.get("lat")
    lon = body.get("longitude") or body.get("lon")
    if lat is None or lon is None:
        return "missing_coordinates"
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return "missing_coordinates"  # or "invalid_coordinates"
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return "invalid_coordinates_range"
    return None


def _normalize_horary_payload(body: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], str]:
    """
    Use yoga normalizer when available to align date/time/tz/place/coords.
    Return a dict of HoraryInput kwargs + warns + tz_norm.
    """
    warns: List[str] = []
    tz_norm = str(body.get("tz") or body.get("place_tz") or "UTC")
    norm: Dict[str, Any] = {}

    if callable(normalize_yoga_payload):
        norm, warns, tz_norm = normalize_yoga_payload(body)  # type: ignore/misc
    else:
        warns.append("validator_unavailable_minimal_fallback")
        norm = {
            "date": body.get("date") or body.get("birth_date"),
            "time": body.get("time") or body.get("birth_time") or "12:00:00",
            "tz": tz_norm,
            "latitude": _coerce_float(body.get("latitude") or body.get("lat")),
            "longitude": _coerce_float(body.get("longitude") or body.get("lon")),
            "elevation_m": _coerce_float(body.get("elevation_m") or body.get("elevation")),
            "zodiac_mode": _norm_method(body.get("zodiac_mode") or body.get("mode") or "sidereal"),
            "ayanamsa": _norm_ayanamsa(body.get("ayanamsa")),
            "place": body.get("place"),
        }

    out = {
        "date": norm.get("date"),
        "time": norm.get("time"),
        "tz_name": norm.get("tz") or tz_norm,
        "place": body.get("place") or norm.get("place"),
        "latitude": norm.get("latitude"),
        "longitude": norm.get("longitude"),
        "zodiac_mode": norm.get("zodiac_mode", "sidereal"),
        "ayanamsa": norm.get("ayanamsa", "lahiri"),
        "ayanamsa_deg": _coerce_float(body.get("ayanamsa_deg")),
        "house_system": body.get("house_system") or "sripati",
        # KP options
        "kp_house_system": body.get("kp_house_system") or "placidus",
        "kp_ayanamsa": body.get("kp_ayanamsa") or "krishnamurti",
        "kp_number": _coerce_int(body.get("kp_number"), None) if body.get("kp_number") is not None else None,  # type: ignore
        "kp_number_mode": str(body.get("kp_number_mode") or "anchor_asc"),
        # Question
        "question_type": _parse_question_type(body.get("question_type")),
        "question_text": body.get("question_text"),
        "querent_house": _coerce_int(body.get("querent_house"), 1),
        "quesited_house": _coerce_int(body.get("quesited_house"), None) if body.get("quesited_house") is not None else None,  # type: ignore
    }
    return out, warns, tz_norm


def _horary_error(route: str, tz_norm: str, error: str, warns: List[str], status: int = 400):
    res = {
        "ok": False,
        "error": error,
        "warnings": warns or [],
        "meta": {"route": route, "branch": "horary_core", "tz_normalized": tz_norm},
    }
    return jsonify(res), status


def _run_horary_single(method: str, body: Dict[str, Any], route_name: str):
    """Shared runner for single-method endpoints with strict raw validation."""
    # NEW: Prefer payload helper first
    if callable(_horary_from_payload):
         body = request.get_json(silent=True) or {}
         try:
             b = dict(body or {}); b["method"] = "hybrid"
             res = _horary_from_payload(b)  # type: ignore
             if isinstance(res, dict) and res.get("ok"):
                 res.setdefault("meta", {}).update({
                     "route": "horary/hybrid",
                     "tz_normalized": _tz_from_payload(body),
                     "branch": _PRED_PAYLOAD_BRANCH
                 })
                 return jsonify(res), 200
             # otherwise, fall through to core fallback (parashari+kp merge)
         except Exception:
             pass

    # Fallback to horary core
    if not _HORARY_OK or not callable(_analyze_prasna_enhanced):
        return jsonify({"ok": False, "error": "horary_engine_unavailable"}), 503

    hw, warns, tz_norm = _normalize_horary_payload(body)

    # Strict raw validation (pre-normalization guarantees)
    e = _validate_civil_raw(body, tz_norm)
    if e:
        return _horary_error(route_name, tz_norm, e, warns, 400)
    e = _validate_coords_raw(body)
    if e:
        return _horary_error(route_name, tz_norm, e, warns, 400)

    try:
        inp = _HoraryInput(**hw)  # type: ignore
        res = _analyze_prasna_enhanced(inp, method=method)  # type: ignore
    except Exception as ex:
        return jsonify({"ok": False, "error": f"{route_name.replace('/', '_')}_failed", "detail": str(ex)}), 400

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": route_name, "tz_normalized": tz_norm, "branch": "horary_core"})
        if warns:
            res.setdefault("warnings", []).extend(warns)
    return jsonify(res), (200 if res.get("ok") else 400)


@vedic_api.post("/horary/parashari")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def horary_parashari():
    body = request.get_json(silent=True) or {}
    return _run_horary_single("parashari", body, "horary/parashari")


@vedic_api.post("/horary/kp")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def horary_kp():
    body = request.get_json(silent=True) or {}
    return _run_horary_single("kp", body, "horary/kp")


@vedic_api.post("/horary/hybrid")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def horary_hybrid():
    """
    Hybrid horary: run both Parāśarī and KP on the same normalized payload
    and return both results plus a lightweight agreement summary.
    """
    # NEW: Prefer payload helper first
    if callable(_horary_from_payload):
        body = request.get_json(silent=True) or {}
        try:
            b = dict(body or {})
            b["method"] = "hybrid"
            res = _horary_from_payload(b)  # type: ignore
            if isinstance(res, dict):
                res.setdefault("meta", {}).update({"route": "horary/hybrid",
                                                   "tz_normalized": _tz_from_payload(body),
                                                   "branch": _PRED_PAYLOAD_BRANCH})
                return jsonify(res), (200 if res.get("ok") else 400)
        except Exception:
            pass

    # Fallback to horary core
    if not _HORARY_OK or not callable(_analyze_prasna_enhanced):
        return jsonify({"ok": False, "error": "horary_engine_unavailable"}), 503

    body = request.get_json(silent=True) or {}
    hw, warns, tz_norm = _normalize_horary_payload(body)

    # Strict raw validation
    e = _validate_civil_raw(body, tz_norm)
    if e:
        return _horary_error("horary/hybrid", tz_norm, e, warns, 400)
    e = _validate_coords_raw(body)
    if e:
        return _horary_error("horary/hybrid", tz_norm, e, warns, 400)

    try:
        inp = _HoraryInput(**hw)  # type: ignore
        res_par = _analyze_prasna_enhanced(inp, method="parashari")  # type: ignore
        res_kp  = _analyze_prasna_enhanced(inp, method="kp")         # type: ignore
    except Exception as ex:
        return jsonify({"ok": False, "error": "horary_hybrid_failed", "detail": str(ex)}), 400

    ok_par = bool(isinstance(res_par, dict) and res_par.get("ok"))
    ok_kp  = bool(isinstance(res_kp, dict) and res_kp.get("ok"))

    if not (ok_par or ok_kp):
        return jsonify({
            "ok": False,
            "error": "hybrid_both_failed",
            "parashari": res_par,
            "kp": res_kp,
            "meta": {"route": "horary/hybrid", "tz_normalized": tz_norm, "branch": "horary_core"},
            "warnings": warns or [],
        }), 400

    merged = _merge_horary_results(
        res_par if isinstance(res_par, dict) else {},
        res_kp if isinstance(res_kp, dict) else {}
    )

    out = {
        "ok": ok_par and ok_kp,
        "method": "hybrid",
        "partial": not (ok_par and ok_kp),
        "parashari": res_par,
        "kp": res_kp,
        "merged": merged,
        "meta": {"route": "horary/hybrid", "tz_normalized": tz_norm, "branch": "horary_core"},
    }
    if warns:
        out.setdefault("warnings", []).extend(warns)
    return jsonify(out), 200


@vedic_api.post("/horary")
@rate_limit(RL_VEDIC_PREDICTIVE, key_fn=fixed_key)
def horary_generic():
    """
    DEPRECATED: Generic horary endpoint.
    Accepts body.method: 'parashari' (default), 'kp', or 'hybrid'.
    Prefer the explicit routes:
      - /api/vedic/horary/parashari
      - /api/vedic/horary/kp
      - /api/vedic/horary/hybrid
    """
    body = request.get_json(silent=True) or {}
    method = str(body.get("method") or "parashari").lower()
    if method not in ("parashari", "kp", "hybrid"):
        method = "parashari"

    # NEW: Prefer payload helper first
    if callable(_horary_from_payload):
        try:
            b = dict(body or {})
            b["method"] = method
            res = _horary_from_payload(b)  # type: ignore
            if isinstance(res, dict):
                res.setdefault("meta", {}).update({"route": "horary",
                                                   "tz_normalized": _tz_from_payload(body),
                                                   "branch": _PRED_PAYLOAD_BRANCH,
                                                   "deprecated": True,
                                                   "preferred_endpoints": ["/api/vedic/horary/parashari", "/api/vedic/horary/kp", "/api/vedic/horary/hybrid"]})
                status = 200 if res.get("ok") else 400
                return jsonify(res), status, {"X-Deprecated-Endpoint": "/api/vedic/horary"}
        except Exception:
            pass

    # Fallback to horary core
    if not _HORARY_OK or not callable(_analyze_prasna_enhanced):
        return jsonify({"ok": False, "error": "horary_engine_unavailable"}), 503

    hw, warns, tz_norm = _normalize_horary_payload(body)

    # Strict raw validation
    e = _validate_civil_raw(body, tz_norm)
    if e:
        res, status = _horary_error("horary", tz_norm, e, warns, 400)
        return res, status, {"X-Deprecated-Endpoint": "/api/vedic/horary"}
    e = _validate_coords_raw(body)
    if e:
        res, status = _horary_error("horary", tz_norm, e, warns, 400)
        return res, status, {"X-Deprecated-Endpoint": "/api/vedic/horary"}

    try:
        inp = _HoraryInput(**hw)  # type: ignore
        if method == "hybrid":
            res_par = _analyze_prasna_enhanced(inp, method="parashari")  # type: ignore
            res_kp  = _analyze_prasna_enhanced(inp, method="kp")         # type: ignore
            ok_par = bool(isinstance(res_par, dict) and res_par.get("ok"))
            ok_kp  = bool(isinstance(res_kp, dict) and res_kp.get("ok"))
            merged = _merge_horary_results(
                res_par if isinstance(res_par, dict) else {},
                res_kp  if isinstance(res_kp, dict)  else {}
            )
            out = {
                "ok": ok_par and ok_kp,
                "method": "hybrid",
                "partial": not (ok_par and ok_kp),
                "parashari": res_par,
                "kp": res_kp,
                "merged": merged,
                "warnings": warns or [],
                "meta": {"route": "horary", "tz_normalized": tz_norm, "branch": "horary_core", "method": "hybrid",
                         "deprecated": True,
                         "preferred_endpoints": ["/api/vedic/horary/parashari", "/api/vedic/horary/kp", "/api/vedic/horary/hybrid"]},
            }
            status = 200 if (ok_par or ok_kp) else 400
            return jsonify(out), status, {"X-Deprecated-Endpoint": "/api/vedic/horary"}

        # Single method
        res = _analyze_prasna_enhanced(inp, method=method)  # type: ignore
    except Exception as ex:
        return jsonify({"ok": False, "error": "horary_failed", "detail": str(ex)}), 400, {"X-Deprecated-Endpoint": "/api/vedic/horary"}

    if isinstance(res, dict):
        res.setdefault("meta", {})
        res["meta"].update({"route": "horary", "tz_normalized": tz_norm, "branch": "horary_core",
                            "method": method, "deprecated": True,
                            "preferred_endpoints": ["/api/vedic/horary/parashari", "/api/vedic/horary/kp", "/api/vedic/horary/hybrid"]})
        if warns:
            res.setdefault("warnings", []).extend(warns)

    status = 200 if res.get("ok") else 400
    return jsonify(res), status, {"X-Deprecated-Endpoint": "/api/vedic/horary"}
