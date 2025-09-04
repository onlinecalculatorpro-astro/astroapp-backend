# app/main.py
from __future__ import annotations

import logging
import os
import sys
import traceback
from time import perf_counter
from typing import Any, Dict, Final

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, BadRequest
from werkzeug.middleware.proxy_fix import ProxyFix

# ───────────────────────── Optional blueprints ─────────────────────────
_routes_import_err: str | None = None
_pred_import_err: str | None = None

try:
    from app.api import routes as _routes_mod  # type: ignore
    _routes_bp = _routes_mod.api
except Exception as e:  # pragma: no cover
    _routes_bp = None
    _routes_import_err = repr(e)
    print("WARNING: routes blueprint failed to import:", _routes_import_err, file=sys.stderr)
    traceback.print_exc()

try:
    from app.api.predictions import predictions_bp as _pred_bp  # type: ignore
except Exception as e:  # pragma: no cover
    _pred_bp = None
    _pred_import_err = repr(e)
    print("WARNING: predictions blueprint failed to import:", _pred_import_err, file=sys.stderr)
    traceback.print_exc()

# ───────────────────────── Optional config loader ─────────────────────────
try:
    from app.utils.config import load_config  # type: ignore
except Exception as e:  # pragma: no cover
    print("INFO: app.utils.config.load_config not available:", repr(e), file=sys.stderr)
    load_config = None  # type: ignore

# ───────────────────────── Core engines & policy façade ─────────────────────────
# IMPORTANT: do NOT import astronomy here; we will lazy-import in the handler
from app.core import timescales as _ts
from app.core.house import (
    list_supported_house_systems,
    GATED_NOT_IMPLEMENTED,
    POLAR_POLICY,
    POLAR_SOFT_LIMIT_DEG,
    POLAR_HARD_LIMIT_DEG,
)

# Detect advanced houses backend for labeling
try:
    from app.core.houses_advanced import PreciseHouseCalculator  # noqa: F401
    _ADV_OK = True
except Exception:
    _ADV_OK = False

# ───────────────────────── Prometheus (with safe shim) ─────────────────────────
try:
    from prometheus_client import (  # type: ignore
        Counter,
        Gauge,
        Histogram,
        CONTENT_TYPE_LATEST,
        REGISTRY,
        generate_latest,
    )
except Exception:  # pragma: no cover
    class _NoOpMetric:
        def labels(self, **_kwargs): return self
        def inc(self, *_a, **_k): return None
        def observe(self, *_a, **_k): return None
        def set(self, *_a, **_k): return None
    def Counter(_n: str, _h: str, _lbls: list[str] | tuple[str, ...]): return _NoOpMetric()  # type: ignore
    def Gauge(_n: str, _h: str): return _NoOpMetric()  # type: ignore
    def Histogram(_n: str, _h: str, _lbls: list[str] | tuple[str, ...]): return _NoOpMetric()  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # type: ignore
    class _NoRegistry: pass
    REGISTRY = _NoRegistry()  # type: ignore
    def generate_latest(_reg=None): return b""  # type: ignore

# Metrics used by house.py via lazy import (keep names stable!)
MET_REQUESTS: Final = Counter("astro_api_requests_total", "API requests", ["route"])
MET_FALLBACKS: Final = Counter(
    "astro_house_fallback_total", "House fallbacks at high latitude", ["requested", "fallback"]
)
MET_WARNINGS: Final = Counter("astro_warning_total", "Non-fatal warnings", ["kind"])
GAUGE_DUT1: Final = Gauge("astro_dut1_broadcast_seconds", "DUT1 broadcast seconds")
GAUGE_APP_UP: Final = Gauge("astro_app_up", "1 if app is running")
REQ_LATENCY: Final = Histogram("astro_request_seconds", "API request latency", ["route"])

# ───────────────────────── policy façade label ─────────────────────────
ENGINE_LABEL: Final[str] = (
    os.getenv("HOUSES_ENGINE_NAME")
    or ("houses-advanced" if _ADV_OK else "proto-basic")
)

# Keep last astronomy import status for debugging
_ASTRONOMY_IMPORT_OK: bool | None = None
_ASTRONOMY_IMPORT_ERR: str | None = None

# ───────────────────────── helpers: logging & errors ─────────────────────────
def _configure_logging(app: Flask) -> None:
    gerr = logging.getLogger("gunicorn.error")
    if gerr.handlers:
        app.logger.handlers = gerr.handlers
        app.logger.setLevel(gerr.level)
    else:
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

def _register_errors(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        app.logger.warning("HTTP %s at %s %s: %s", e.code, request.method, request.path, e.description)
        return jsonify(
            ok=False,
            error="http_error",
            code=e.code,
            name=e.name,
            message=e.description,
            path=request.path,
        ), e.code

    @app.errorhandler(Exception)
    def _any(e: Exception):
        tb = traceback.format_exc()
        app.logger.error("UNHANDLED %s at %s %s\n%s", type(e).__name__, request.method, request.path, tb)
        return jsonify(
            ok=False,
            error="internal_error",
            type=type(e).__name__,
            message=str(e),
            path=request.path,
        ), 500

# ───────────────────────── health & utils ─────────────────────────
def _register_health(app: Flask) -> None:
    @app.route("/", methods=["GET"])
    def root():
        return jsonify(ok=True, service="astro-backend", health="/health"), 200

    @app.route("/api/health-check", methods=["GET"])
    def api_health():
        return jsonify(ok=True), 200

    @app.route("/health", methods=["GET"])
    @app.route("/healthz", methods=["GET"])
    def health():
        return jsonify(ok=True, status="ok"), 200

def _metrics_auth_ok() -> bool:
    auth = request.authorization
    user = os.getenv("METRICS_USER", "")
    pw = os.getenv("METRICS_PASS", "")
    return bool(
        auth and auth.type == "basic" and auth.username == user and auth.password == pw and user and pw
    )

def _body_json() -> Dict[str, Any]:
    data = request.get_json(force=True, silent=False)
    if not isinstance(data, dict):
        raise BadRequest("JSON body must be an object")
    return data

# ───────────────────────── time string normalizer (fraction-safe) ─────────────────────────
import re as _re
_TIME_RE = _re.compile(r"^\s*(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.(?P<f>\d+))?\s*$")
def _normalize_time_hms(time_s: str) -> str:
    """
    Accepts 'HH:MM:SS' or 'HH:MM:SS.frac' and returns canonical string.
    - Hours/minutes are integers.
    - Seconds may carry a fractional part.
    - Allows leap second 'SS==60' (fraction allowed).
    - Disallows 24:00:00.* except exact '24:00:00'.
    """
    m = _TIME_RE.match(time_s or "")
    if not m:
        # allow 'HH:MM' by filling ':00'
        mm_match = _re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", time_s or "")
        if mm_match:
            hh = int(mm_match.group(1)); mm = int(mm_match.group(2))
            if not (0 <= hh <= 24 and 0 <= mm <= 59):
                raise BadRequest("time fields out of range")
            if hh == 24 and mm != 0:
                raise BadRequest("24:00:00 is only allowed exactly")
            return f"{hh:02d}:{mm:02d}:00"
        raise BadRequest("time must be 'HH:MM' or 'HH:MM:SS[.frac]'")

    hh = int(m.group("h")); mm = int(m.group("m")); ss = int(m.group("s"))
    frac = (m.group("f") or "")
    if not (0 <= hh <= 24 and 0 <= mm <= 59 and 0 <= ss <= 60):
        raise BadRequest("time fields out of range")
    if hh == 24:
        if not (mm == 0 and ss == 0 and frac == ""):
            raise BadRequest("24:00:00 is only allowed exactly")
        return "24:00:00"
    frac = "".join(ch for ch in frac if ch.isdigit())
    return f"{hh:02d}:{mm:02d}:{ss:02d}" + (f".{frac}" if frac else "")

# ───────────────────────── timescales computation using core builder ─────────────────────────
def _compute_timescales_payload(date: str, time_s: str, tz: str, dut1_val: float | None) -> Dict[str, Any]:
    """
    Returns {timescales: {...}} with jd_utc, jd_tt, jd_ut1, delta_t, dat, delta_at, dut1, tz_offset_seconds, timezone, warnings, precision.
    """
    # Prefer request DUT1; fall back to environment broadcast if absent
    if dut1_val is None:
        dut1_env = os.getenv("ASTRO_DUT1_BROADCAST", os.getenv("ASTRO_DUT1", "0.0")) or "0.0"
        try:
            dut1_val = float(dut1_env)
        except Exception:
            dut1_val = 0.0

    ts = _ts.build_timescales(date, _normalize_time_hms(time_s), tz, float(dut1_val))
    pkt = ts.to_dict()
    # Provide compatibility alias
    pkt["delta_at"] = pkt.get("dat")
    return {"timescales": pkt}

def _timescales_meta_from_chart(chart: Dict[str, Any]) -> Dict[str, Any]:
    jd_ut = float(chart.get("jd_ut"))
    jd_tt = float(chart.get("jd_tt"))
    jd_ut1 = float(chart.get("jd_ut1"))
    delta_t_sec = (jd_tt - jd_ut) * 86400.0
    dut1_sec = (jd_ut1 - jd_ut) * 86400.0
    return {
        "jd_utc": jd_ut,
        "jd_tt": jd_tt,
        "jd_ut1": jd_ut1,
        "delta_t": delta_t_sec,
        "delta_at": 0.0,
        "dut1": dut1_sec,
    }

# ───────────────────────── system/policy endpoints ─────────────────────────
def _houses_policy_blob() -> Dict[str, Any]:
    return {
        "houses_engine": ENGINE_LABEL,
        "polar": {
            "policy": POLAR_POLICY,
            "soft_limit_deg": POLAR_SOFT_LIMIT_DEG,
            "hard_limit_deg": POLAR_HARD_LIMIT_DEG,
            "numeric_fallback_enabled": os.getenv("ASTRO_HOUSES_NUMERIC_FALLBACK", "1").lower() in ("1", "true", "yes", "on"),
        },
        "declared_systems": list(list_supported_house_systems()),
        "gated_systems": sorted(list(GATED_NOT_IMPLEMENTED)),
    }

def _register_core_api(app: Flask) -> None:
    @app.get("/system-validation")
    def _system_validation():
        return jsonify({"ok": True, "policy": _houses_policy_blob()}), 200

    # timescales
    def _timescales_handler():
        data = _body_json()
        date = data.get("date")
        time_s = data.get("time")
        tz = data.get("tz") or data.get("place_tz") or data.get("timezone")
        dut1 = data.get("dut1", None)
        if not (isinstance(date, str) and isinstance(time_s, str) and isinstance(tz, str)):
            raise BadRequest("Provide 'date', 'time', and 'tz' (IANA)")
        # accept dut1 as string or number
        if dut1 is not None and not isinstance(dut1, (int, float, str)):
            raise BadRequest("dut1 must be a number")
        try:
            dut1_f = float(dut1) if dut1 is not None else None
        except Exception:
            raise BadRequest("dut1 must be parseable as float")
        out = _compute_timescales_payload(date, time_s, tz, dut1_f)
        return jsonify({"ok": True, **out}), 200

    app.add_url_rule("/api/timescales", "timescales", _timescales_handler, methods=["POST"])
    for alias in ("/api/time/timescales", "/time/timescales", "/timescales", "/api/time/convert"):
        app.add_url_rule(alias, f"timescales_alias_{alias}", _timescales_handler, methods=["POST"])

    # chart (lazy import astronomy)
    def _chart_handler():
        global _ASTRONOMY_IMPORT_OK, _ASTRONOMY_IMPORT_ERR
        try:
            # Lazily import here so Gunicorn can boot even if astronomy has an issue
            from app.core.astronomy import compute_chart  # type: ignore
            _ASTRONOMY_IMPORT_OK, _ASTRONOMY_IMPORT_ERR = True, None
        except Exception as e:
            _ASTRONOMY_IMPORT_OK, _ASTRONOMY_IMPORT_ERR = False, repr(e)
            app.logger.exception("Failed to import app.core.astronomy: %s", e)
            return jsonify({
                "ok": False,
                "error": "astronomy_import_error",
                "message": "Failed to import astronomy engine (see logs / __debug/astronomy_import).",
                "detail": _ASTRONOMY_IMPORT_ERR,
            }), 500

        payload = _body_json()

        # backfill timescales if civil provided
        have_all_jd = all(k in payload for k in ("jd_ut", "jd_tt", "jd_ut1"))
        if not have_all_jd:
            date = payload.get("date")
            time_s = payload.get("time")
            tz = payload.get("tz") or payload.get("place_tz") or payload.get("timezone")
            if isinstance(date, str) and isinstance(time_s, str) and isinstance(tz, str):
                # best-effort DUT1
                dut1_in = payload.get("dut1", None)
                dut1_f = float(dut1_in) if isinstance(dut1_in, (int, float, str)) else None
                ts = _compute_timescales_payload(date, time_s, tz, dut1_f)["timescales"]
                payload.update({
                    "jd_ut": ts["jd_utc"], "jd_tt": ts["jd_tt"], "jd_ut1": ts["jd_ut1"]
                })
            else:
                raise BadRequest("Provide either jd_ut/jd_tt/jd_ut1 or civil 'date'+'time'+'tz'")

        payload.setdefault("mode", "tropical")

        chart = compute_chart(payload)
        chart["ok"] = True
        chart.setdefault("meta", {}).setdefault("timescales", _timescales_meta_from_chart(chart))
        return jsonify(chart), 200

    app.add_url_rule("/api/chart", "chart", _chart_handler, methods=["POST"])
    for alias in ("/chart", "/api/compute_chart", "/compute_chart", "/api/astronomy/chart", "/astronomy/chart", "/api/astro/chart"):
        app.add_url_rule(alias, f"chart_alias_{alias}", _chart_handler, methods=["POST"])

# ───────────────────────── app factory ─────────────────────────
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)  # type: ignore

    _configure_logging(app)

    # Optional config
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    if load_config:
        try:
            app.cfg = load_config(cfg_path)  # type: ignore[attr-defined]
        except Exception:
            app.cfg = {}  # type: ignore[attr-defined]

    # Seed metrics
    seeded_routes = (
        "/", "/api/health-check",
        "/api/chart",
        "/api/timescales",
        "/health", "/healthz", "/metrics", "/system-validation"
    )
    for route in seeded_routes:
        MET_REQUESTS.labels(route=route).inc(0)
        REQ_LATENCY.labels(route=route).observe(0.0)
    MET_FALLBACKS.labels(requested="placidus", fallback="equal").inc(0)
    MET_WARNINGS.labels(kind="polar_soft_fallback").inc(0)
    GAUGE_APP_UP.set(1.0)

    @app.before_request
    def _before():
        try:
            p = (request.path or "")
            if p.startswith("/api/") or p in ("/", "/health", "/healthz", "/metrics", "/system-validation"):
                MET_REQUESTS.labels(route=p).inc()
                request._t0 = perf_counter()
        except Exception:
            pass

    @app.after_request
    def _after(resp):
        try:
            p = (request.path or "")
            if (p.startswith("/api/") or p in ("/", "/health", "/healthz", "/system-validation")) and hasattr(request, "_t0"):
                dt = perf_counter() - request._t0
                REQ_LATENCY.labels(route=p).observe(dt)
        except Exception:
            pass
        return resp

    _register_health(app)
    _register_errors(app)
    _register_core_api(app)

    if _routes_bp is not None:
        app.register_blueprint(_routes_bp)
    if _pred_bp is not None:
        app.register_blueprint(_pred_bp, url_prefix="/api")

    # Debug endpoints
    @app.get("/__debug/imports")
    def __debug_imports():
        return jsonify({
            "routes_blueprint_loaded": _routes_bp is not None,
            "routes_import_error": _routes_import_err,
            "predictions_blueprint_loaded": _pred_bp is not None,
            "predictions_import_error": _pred_import_err,
            "houses_backend_ok": _ADV_OK,
            "engine_label": ENGINE_LABEL,
            "blueprints": list(app.blueprints.keys()),
        }), 200

    @app.get("/__debug/routes")
    def __debug_routes():
        rules = []
        for r in app.url_map.iter_rules():
            methods = sorted(m for m in (r.methods or []) if m not in ("HEAD", "OPTIONS"))
            rules.append({"rule": str(r), "endpoint": r.endpoint, "methods": methods})
        rules.sort(key=lambda x: x["rule"])
        return jsonify({"count": len(rules), "rules": rules}), 200

    @app.get("/__debug/ephem")
    def __debug_ephem():
        import importlib.util
        from app.core import ephemeris_adapter as eph
        path = os.getenv("OCP_EPHEMERIS")
        exists = bool(path and os.path.isfile(path))
        size = (os.path.getsize(path) if exists else None)
        skyfield_ok = importlib.util.find_spec("skyfield") is not None
        jplephem_ok = importlib.util.find_spec("jplephem") is not None
        return jsonify({
            "OCP_EPHEMERIS": path,
            "file_exists": exists,
            "file_size_bytes": size,
            "skyfield_importable": skyfield_ok,
            "jplephem_importable": jplephem_ok,
            "diagnostics": eph.ephemeris_diagnostics(),
        }), 200

    @app.get("/__debug/astronomy_import")
    def __debug_astronomy_import():
        return jsonify({
            "import_attempted": _ASTRONOMY_IMPORT_OK is not None,
            "import_ok": _ASTRONOMY_IMPORT_OK,
            "import_error": _ASTRONOMY_IMPORT_ERR,
        }), 200

    @app.get("/favicon.ico")
    def _noop_favicon():
        return ("", 204)


    # /metrics (Basic Auth)
    @app.route("/metrics", methods=["GET"])
    def metrics_endpoint():
        if not _metrics_auth_ok():
            return Response("Unauthorized", 401, {"WWW-Authenticate": 'Basic realm="metrics"'})
        GAUGE_APP_UP.set(1.0)
        try:
            GAUGE_DUT1.set(float(os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", "0.0")) or 0.0))
        except Exception:
            pass
        data = generate_latest(REGISTRY)
        return Response(data, mimetype=CONTENT_TYPE_LATEST)

    app.logger.info(
        "App initialized; houses_engine=%s; routes_loaded=%s, predictions_loaded=%s",
        ENGINE_LABEL, bool(_routes_bp), bool(_pred_bp)
    )
    return app

# ───────────────────────── app instance ─────────────────────────
app = create_app()

# CORS for browser UIs
_allowed_origin = (
    os.environ.get("CORS_ALLOW_ORIGIN")
    or os.environ.get("NETLIFY_ORIGIN")
    or "*"
)
CORS(
    app,
    resources={r"/.*": {"origins": _allowed_origin}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
