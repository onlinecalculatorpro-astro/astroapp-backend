# app/main.py
from __future__ import annotations

import logging
import os
import sys
import traceback
from time import perf_counter
from typing import Any, Dict, Final, Optional

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

# ───────────────────────── Prometheus (safe shim) ─────────────────────────
try:
    from prometheus_client import (  # type: ignore
        Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, REGISTRY, generate_latest,
    )
except Exception:  # pragma: no cover
    class _NoOpMetric:
        def labels(self, **_): return self
        def inc(self, *_a, **_k): pass
        def observe(self, *_a, **_k): pass
        def set(self, *_a, **_k): pass
    def Counter(*_, **__): return _NoOpMetric()  # type: ignore
    def Gauge(*_, **__): return _NoOpMetric()    # type: ignore
    def Histogram(*_, **__): return _NoOpMetric()# type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # type: ignore
    REGISTRY = object()  # type: ignore
    def generate_latest(_reg=None): return b""   # type: ignore

# App-level metrics
MET_REQUESTS: Final = Counter("astro_api_requests_total", "API requests", ["route"])
REQ_LATENCY: Final = Histogram("astro_request_seconds", "API request latency", ["route"])
GAUGE_APP_UP: Final = Gauge("astro_app_up", "1 if app is running")
GAUGE_DUT1: Final = Gauge("astro_dut1_broadcast_seconds", "DUT1 broadcast seconds")

# ───────────────────────── Blueprint imports ─────────────────────────
# ops / diagnostics (no prefix)
_ops_bp = None
_ops_import_err: Optional[str] = None
try:
    from app.api.routes import ops_api as _ops_bp  # type: ignore
except Exception as e_ops_primary:
    _ops_import_err = repr(e_ops_primary)
    # Legacy fallback: routes.py may expose "api"; if so, we can mount at /api.
    try:
        from app.api.routes import api as _legacy_api_bp  # type: ignore
    except Exception:
        _legacy_api_bp = None  # type: ignore

# western (mounted at /api/western)
_western_bp = None
_western_import_err: Optional[str] = None
try:
    from app.api.western_routes import western_api as _western_bp  # type: ignore
except Exception as e:
    _western_import_err = repr(e)
    print("WARNING: western_routes blueprint failed to import:", _western_import_err, file=sys.stderr)
    traceback.print_exc()

# vedic (mounted at /api/vedic) — feature-gated
_vedic_bp = None
_vedic_import_err: Optional[str] = None
_ENABLE_VEDIC = os.getenv("ENABLE_VEDIC_API", "1").lower() in ("1", "true", "yes", "on")
if _ENABLE_VEDIC:
    try:
        from app.api.vedic_routes import vedic_api as _vedic_bp  # type: ignore
    except Exception as e:
        _vedic_import_err = repr(e)
        print("WARNING: vedic_routes blueprint failed to import:", _vedic_import_err, file=sys.stderr)
        traceback.print_exc()

# ───────────────────────── helpers: logging & errors ─────────────────────────
def _configure_logging(app: Flask) -> None:
    """Reuse gunicorn logger if present, else basicConfig."""
    gerr = logging.getLogger("gunicorn.error")
    if gerr.handlers:
        app.logger.handlers = gerr.handlers
        app.logger.setLevel(gerr.level)
    else:
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        app.logger.warning("HTTP %s at %s %s: %s", e.code, request.method, request.path, e.description)
        return jsonify(
            ok=False, error="http_error", code=e.code, name=e.name,
            message=e.description, path=request.path,
        ), e.code

    @app.errorhandler(Exception)
    def _any(e: Exception):
        tb = traceback.format_exc()
        app.logger.error("UNHANDLED %s at %s %s\n%s", type(e).__name__, request.method, request.path, tb)
        return jsonify(ok=False, error="internal_error", type=type(e).__name__, message=str(e), path=request.path), 500

# ───────────────────────── basic auth for /metrics ─────────────────────────
def _metrics_auth_ok() -> bool:
    auth = request.authorization
    user = os.getenv("METRICS_USER", "")
    pw = os.getenv("METRICS_PASS", "")
    return bool(auth and auth.type == "basic" and user and pw and auth.username == user and auth.password == pw)

# ───────────────────────── app factory ─────────────────────────
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)  # type: ignore

    _configure_logging(app)
    _register_error_handlers(app)

    GAUGE_APP_UP.set(1.0)

    # Seed labels so histograms/gauges exist early
    for route in ("/", "/healthz", "/metrics"):
        MET_REQUESTS.labels(route=route).inc(0)
        REQ_LATENCY.labels(route=route).observe(0.0)

    @app.before_request
    def _before_request():
        try:
            MET_REQUESTS.labels(route=(request.path or "")).inc()
            request._t0 = perf_counter()
        except Exception:
            pass

    @app.after_request
    def _after_request(resp: Response):
        # Metrics + Server-Timing header
        try:
            t0 = getattr(request, "_t0", None)
            if t0 is not None:
                dur_s = perf_counter() - t0
                REQ_LATENCY.labels(route=(request.path or "")).observe(dur_s)
                dur_ms = int(dur_s * 1000)
                # Add standard Server-Timing + X-Response-Time headers
                prev = resp.headers.get("Server-Timing")
                this = f"app;dur={dur_ms}"
                resp.headers["Server-Timing"] = (f"{prev}, {this}" if prev else this)
                resp.headers["X-Response-Time"] = f"{dur_ms} ms"
        except Exception:
            pass

        # Add deprecation headers for legacy /api/health
        try:
            if request.path == "/api/health":
                resp.headers.setdefault("Deprecation", "true")
                resp.headers.setdefault("Link", '</healthz>; rel="successor-version"')
        except Exception:
            pass
        return resp

    # ───── Root & Health ─────
    @app.get("/")
    def _root():
        return jsonify(ok=True, service="astro-backend", health="/healthz"), 200

    @app.get("/healthz")
    def _healthz():
        return jsonify(ok=True, status="ok"), 200

    # ───── Metrics (basic auth) ─────
    @app.get("/metrics")
    def metrics_endpoint():
        if not _metrics_auth_ok():
            return Response("Unauthorized", 401, {"WWW-Authenticate": 'Basic realm="metrics"'})
        try:
            GAUGE_DUT1.set(float(os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", "0.0")) or 0.0))
        except Exception:
            pass
        return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)

    # ───── Debug helpers ─────
    @app.get("/__debug/routes")
    def __debug_routes():
        rules: list[Dict[str, Any]] = []
        for r in app.url_map.iter_rules():
            methods = sorted(m for m in (r.methods or []) if m not in ("HEAD", "OPTIONS"))
            rules.append({"rule": str(r), "endpoint": r.endpoint, "methods": methods})
        rules.sort(key=lambda x: x["rule"])
        return jsonify({"count": len(rules), "routes": rules}), 200

    @app.get("/__debug/imports")
    def __debug_imports():
        return jsonify({
            # ops / legacy
            "ops_blueprint_loaded": _ops_bp is not None,
            "ops_import_error": _ops_import_err,
            "legacy_api_present": ('_legacy_api_bp' in globals()) and (globals().get('_legacy_api_bp') is not None),
            # western
            "western_blueprint_loaded": (_western_bp is not None),
            "western_import_error": _western_import_err,
            # vedic
            "vedic_enabled_flag": _ENABLE_VEDIC,
            "vedic_blueprint_loaded": (_vedic_bp is not None),
            "vedic_import_error": _vedic_import_err,
            # registry of mounted blueprints
            "blueprints": list(app.blueprints.keys()),
            "mounts": {
                "ops": "(no prefix)",
                "western": "/api/western",
                "vedic": "/api/vedic" if _ENABLE_VEDIC else "(disabled)",
                "legacy_api": "/api (only if ops_api missing and routes.py exposes 'api')",
            },
        }), 200

    @app.get("/favicon.ico")
    def _noop_favicon():
        return ("", 204)

    # ───── Register blueprints (ROUTE FILES USE RELATIVE PATHS) ─────
    # 1) ops / diagnostics (preferred)
    if _ops_bp is not None:
        app.register_blueprint(_ops_bp)  # no prefix
    else:
        # 1a) legacy compatibility: if routes.py exposes "api" (absolute/relative unknown), mount at /api
        if globals().get('_legacy_api_bp') is not None:
            app.register_blueprint(globals()['_legacy_api_bp'], url_prefix="/api")
        else:
            @app.get("/ops_missing")
            def _ops_missing():
                return jsonify(ok=False, error="ops_blueprint_not_loaded", detail=_ops_import_err), 500

    # 2) western at /api/western
    if _western_bp is not None:
        app.register_blueprint(_western_bp, url_prefix="/api/western")
    else:
        @app.get("/api/western/health")
        def _western_missing():
            return jsonify(ok=False, error="western_blueprint_not_loaded", detail=_western_import_err), 500

    # 3) vedic at /api/vedic (feature-gated)
    if _ENABLE_VEDIC and _vedic_bp is not None:
        app.register_blueprint(_vedic_bp, url_prefix="/api/vedic")
    elif _ENABLE_VEDIC:
        @app.get("/api/vedic/health")
        def _vedic_health_fallback():
            return jsonify(ok=False, error="vedic_blueprint_not_loaded", detail=_vedic_import_err), 500

    app.logger.info(
        "App initialized; ops=%s western=%s vedic_enabled=%s vedic=%s",
        bool(_ops_bp) or bool(globals().get('_legacy_api_bp')),
        bool(_western_bp),
        _ENABLE_VEDIC,
        bool(_vedic_bp),
    )
    return app

# ───────────────────────── app instance ─────────────────────────
app = create_app()

# ───────────────────────── CORS ─────────────────────────
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
