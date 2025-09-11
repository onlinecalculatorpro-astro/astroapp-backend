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

# ───────────────────────── routes blueprint import (Western) ─────────────────────────
_routes_bp = None
_routes_import_err: Optional[str] = None
try:
    from app.api import routes as _routes_mod  # type: ignore
    _routes_bp = _routes_mod.api
except Exception as e:  # pragma: no cover
    _routes_import_err = repr(e)
    print("WARNING: routes blueprint failed to import:", _routes_import_err, file=sys.stderr)
    traceback.print_exc()

# ───────────────────────── Vedic routes blueprint import (optional) ─────────────────────────
_vedic_bp = None
_vedic_import_err: Optional[str] = None
_ENABLE_VEDIC = os.getenv("ENABLE_VEDIC_API", "1").lower() in ("1", "true", "yes", "on")
if _ENABLE_VEDIC:
    try:
        from app.api.vedic_routes import vedic_api as _vedic_bp  # type: ignore
    except Exception as e:  # pragma: no cover
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
        try:
            t0 = getattr(request, "_t0", None)
            if t0 is not None:
                REQ_LATENCY.labels(route=(request.path or "")).observe(perf_counter() - t0)
        except Exception:
            pass
        return resp

    # ───── Health & root ─────
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

    # ───── Debug helpers (minimal, non-sensitive) ─────
    @app.get("/__debug/routes")
    def __debug_routes():
        rules: list[Dict[str, Any]] = []
        for r in app.url_map.iter_rules():
            methods = sorted(m for m in (r.methods or []) if m not in ("HEAD", "OPTIONS"))
            rules.append({"rule": str(r), "endpoint": r.endpoint, "methods": methods})
        rules.sort(key=lambda x: x["rule"])
        return jsonify({"count": len(rules), "rules": rules}), 200

    @app.get("/__debug/imports")
    def __debug_imports():
        return jsonify({
            "routes_blueprint_loaded": _routes_bp is not None,
            "routes_import_error": _routes_import_err,
            "vedic_blueprint_loaded": (_vedic_bp is not None),
            "vedic_import_error": _vedic_import_err,
            "vedic_enabled_flag": _ENABLE_VEDIC,
            "blueprints": list(app.blueprints.keys()),
        }), 200

    @app.get("/favicon.ico")
    def _noop_favicon():
        return ("", 204)

    # ───── Register the core API blueprint (Western; canonical) ─────
    if _routes_bp is not None:
        # routes.py uses absolute '/api/...' paths; no url_prefix needed
        app.register_blueprint(_routes_bp)
    else:
        @app.get("/api/health")
        def _health_fallback():
            return jsonify(ok=False, error="routes_blueprint_not_loaded", detail=_routes_import_err), 500

    # ───── Register the Vedic API blueprint ─────
    if _ENABLE_VEDIC and _vedic_bp is not None:
        # IMPORTANT:
        # Your vedic_routes.py defines ABSOLUTE paths (e.g. '/api/vedic/dasha/vimshottari').
        # Therefore we DO NOT set url_prefix here. If you later convert vedic routes to
        # relative paths (e.g. '/dasha/vimshottari'), change the next line to:
        #     app.register_blueprint(_vedic_bp, url_prefix="/api/vedic")
        app.register_blueprint(_vedic_bp)
    elif _ENABLE_VEDIC and _vedic_bp is None:
        @app.get("/api/vedic/health")
        def _vedic_health_fallback():
            return jsonify(ok=False, error="vedic_blueprint_not_loaded", detail=_vedic_import_err), 500

    app.logger.info(
        "App initialized; routes_loaded=%s vedic_enabled=%s vedic_loaded=%s",
        bool(_routes_bp), _ENABLE_VEDIC, bool(_vedic_bp),
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
