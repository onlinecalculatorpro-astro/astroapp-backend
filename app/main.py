# app/main.py
from __future__ import annotations

import logging
import os
from typing import Final
from time import perf_counter

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from app.api.routes import api
from app.api.predictions import predictions_bp          # ✅ NEW: predictions blueprint
from app.utils.config import load_config

# Prometheus metrics (standard mode)
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    REGISTRY,
    generate_latest,
)

# ───────────────────────────── metrics ─────────────────────────────
MET_REQUESTS: Final = Counter("astro_api_requests_total", "API requests", ["route"])
MET_FALLBACKS: Final = Counter(
    "astro_house_fallback_total", "House fallbacks at high latitude", ["requested", "fallback"]
)
MET_WARNINGS: Final = Counter("astro_warning_total", "Non-fatal warnings", ["kind"])
GAUGE_DUT1: Final = Gauge("astro_dut1_broadcast_seconds", "DUT1 broadcast seconds")
GAUGE_APP_UP: Final = Gauge("astro_app_up", "1 if app is running")
REQ_LATENCY: Final = Histogram("astro_request_seconds", "API request latency", ["route"])

# ───────────────────────────── helpers ─────────────────────────────
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
        return jsonify(
            error="http_error",
            code=e.code,
            name=e.name,
            description=e.description,
        ), e.code

    @app.errorhandler(Exception)
    def _any(e: Exception):
        app.logger.exception(e)
        return jsonify(error="internal_error", type=type(e).__name__, message=str(e)), 500


def _register_health(app: Flask) -> None:
    @app.route("/api/health-check", methods=["GET"])
    def api_health():
        return jsonify(ok=True), 200

    @app.route("/health", methods=["GET"])
    @app.route("/healthz", methods=["GET"])
    def health():
        return jsonify(status="ok"), 200


def _metrics_auth_ok() -> bool:
    """Validate Basic Auth credentials for /metrics using env vars."""
    auth = request.authorization
    user = os.getenv("METRICS_USER", "")
    pw = os.getenv("METRICS_PASS", "")
    return bool(
        auth
        and auth.type == "basic"
        and auth.username == user
        and auth.password == pw
        and user
        and pw
    )

# ───────────────────────────── app factory ─────────────────────────────
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # honor reverse proxy headers (Render/Cloudflare)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)  # type: ignore

    _configure_logging(app)

    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    app.cfg = load_config(cfg_path)  # type: ignore[attr-defined]

    # Initialize metric label sets (preseed so time series appear immediately)
    seeded_routes = (
        "/api/health-check",
        "/api/calculate",
        "/api/predictions",   # ✅ NEW
        "/health",
        "/healthz",
        "/metrics",
    )
    for route in seeded_routes:
        MET_REQUESTS.labels(route=route).inc(0)
        REQ_LATENCY.labels(route=route).observe(0.0)
    MET_FALLBACKS.labels(requested="placidus", fallback="equal").inc(0)
    MET_WARNINGS.labels(kind="polar_soft_fallback").inc(0)
    GAUGE_APP_UP.set(1.0)

    # Request counting + timing
    @app.before_request
    def _before():
        try:
            p = (request.path or "")
            if p.startswith("/api/") or p in ("/health", "/healthz", "/metrics"):
                MET_REQUESTS.labels(route=p).inc()
                request._t0 = perf_counter()
        except Exception:
            pass

    @app.after_request
    def _after(resp):
        try:
            p = (request.path or "")
            if (p.startswith("/api/") or p in ("/health", "/healthz")) and hasattr(request, "_t0"):
                dt = perf_counter() - request._t0
                REQ_LATENCY.labels(route=p).observe(dt)
        except Exception:
            pass
        return resp

    # Register components
    app.register_blueprint(api)                                  # existing routes
    app.register_blueprint(predictions_bp, url_prefix="/api")    # ✅ expose /api/predictions
    _register_health(app)
    _register_errors(app)

    # ── /metrics with Basic Auth ─────────────────────────────────────
    @app.route("/metrics", methods=["GET"])
    def metrics_endpoint():
        if not _metrics_auth_ok():
            return Response("Unauthorized", 401, {"WWW-Authenticate": 'Basic realm="metrics"'})

        GAUGE_APP_UP.set(1.0)
        try:
            GAUGE_DUT1.set(float(os.environ.get("ASTRO_DUT1_BROADCAST", "0.0")))
        except Exception:
            pass

        data = generate_latest(REGISTRY)
        return Response(data, mimetype=CONTENT_TYPE_LATEST)

    app.logger.info("App initialized with config path %s", cfg_path)
    return app

# ───────────────────────────── app instance ─────────────────────────────
app = create_app()

# CORS (not used by Prometheus, but useful for browser UIs)
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
