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
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

# ───────────────────────── import API blueprint ─────────────────────────
_routes_import_err: str | None = None
try:
    from app.api import routes as _routes_mod  # type: ignore
    _routes_bp = _routes_mod.api
except Exception as e:  # pragma: no cover
    _routes_bp = None
    _routes_import_err = repr(e)
    print("WARNING: routes blueprint failed to import:", _routes_import_err, file=sys.stderr)
    traceback.print_exc()

# ───────────────────────── Prometheus (safe shim) ─────────────────────────
try:
    from prometheus_client import (  # type: ignore
        Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, REGISTRY, generate_latest,
    )
except Exception:  # pragma: no cover
    class _NoOpMetric:
        def labels(self, **_kwargs): return self
        def inc(self, *_a, **_k): return None
        def observe(self, *_a, **_k): return None
        def set(self, *_a, **_k): return None
    def Counter(_n: str, _h: str, _lbls: list[str] | tuple[str, ...] = ()): return _NoOpMetric()  # type: ignore
    def Gauge(_n: str, _h: str): return _NoOpMetric()  # type: ignore
    def Histogram(_n: str, _h: str, _lbls: list[str] | tuple[str, ...] = ()): return _NoOpMetric()  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # type: ignore
    class _NoRegistry: ...
    REGISTRY = _NoRegistry()  # type: ignore
    def generate_latest(_reg=None): return b""  # type: ignore

# Basic app-level metrics
MET_REQUESTS: Final = Counter("astro_api_requests_total", "API requests", ["route"])
REQ_LATENCY: Final = Histogram("astro_request_seconds", "API request latency", ["route"])
GAUGE_APP_UP: Final = Gauge("astro_app_up", "1 if app is running")
GAUGE_DUT1: Final = Gauge("astro_dut1_broadcast_seconds", "DUT1 broadcast seconds")

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
            ok=False, error="http_error", code=e.code, name=e.name,
            message=e.description, path=request.path,
        ), e.code

    @app.errorhandler(Exception)
    def _any(e: Exception):
        tb = traceback.format_exc()
        app.logger.error("UNHANDLED %s at %s %s\n%s", type(e).__name__, request.method, request.path, tb)
        return jsonify(
            ok=False, error="internal_error", type=type(e).__name__, message=str(e), path=request.path,
        ), 500

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
    _register_errors(app)

    GAUGE_APP_UP.set(1.0)

    # Seed a few paths so histograms have labels
    for route in ("/", "/healthz", "/metrics"):
        MET_REQUESTS.labels(route=route).inc(0)
        REQ_LATENCY.labels(route=route).observe(0.0)

    @app.before_request
    def _before():
        try:
            p = (request.path or "")
            MET_REQUESTS.labels(route=p).inc()
            request._t0 = perf_counter()
        except Exception:
            pass

    @app.after_request
    def _after(resp):
        try:
            p = (request.path or "")
            if hasattr(request, "_t0"):
                dt = perf_counter() - request._t0
                REQ_LATENCY.labels(route=p).observe(dt)
        except Exception:
            pass
        return resp

    # ───── Health (keep only /healthz; root is a tiny noop) ─────
    @app.get("/")
    def _root():
        return jsonify(ok=True, service="astro-backend", health="/healthz"), 200

    @app.get("/healthz")
    def _healthz():
        return jsonify(ok=True, status="ok"), 200

    # ───── /metrics (basic auth) ─────
    @app.get("/metrics")
    def metrics_endpoint():
        if not _metrics_auth_ok():
            return Response("Unauthorized", 401, {"WWW-Authenticate": 'Basic realm="metrics"'})
        try:
            GAUGE_DUT1.set(float(os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", "0.0")) or 0.0))
        except Exception:
            pass
        data = generate_latest(REGISTRY)
        return Response(data, mimetype=CONTENT_TYPE_LATEST)

    # ───── Debug helpers ─────
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
            "blueprints": list(app.blueprints.keys()),
        }), 200

    @app.get("/favicon.ico")
    def _noop_favicon():
        return ("", 204)

    # ───── Register the core API blueprint (all canonical routes live there) ─────
    if _routes_bp is not None:
        # routes.py uses absolute '/api/...' paths; no url_prefix needed
        app.register_blueprint(_routes_bp)
    else:
        # Minimal fallback so health dashboards don’t look totally red
        @app.get("/api/health")
        def _health_fallback():
            return jsonify(ok=False, error="routes_blueprint_not_loaded", detail=_routes_import_err), 500

    app.logger.info("App initialized; routes_loaded=%s", bool(_routes_bp))
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
