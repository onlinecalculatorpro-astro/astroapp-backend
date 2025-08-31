# app/main.py
from __future__ import annotations

import logging
import os
from typing import Final

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from app.api.routes import api
from app.utils.config import load_config

# Prometheus metrics (standard mode)
from prometheus_client import (
    Counter,
    Gauge,
    CONTENT_TYPE_LATEST,
    REGISTRY,
    generate_latest,
)

# Custom metrics
MET_REQUESTS: Final = Counter(
    "astro_api_requests_total", "API requests", ["route"]
)
MET_FALLBACKS: Final = Counter(
    "astro_house_fallback_total", "House fallbacks at high latitude", ["requested", "fallback"]
)
MET_WARNINGS: Final = Counter(
    "astro_warning_total", "Non-fatal warnings", ["kind"]
)
GAUGE_DUT1: Final = Gauge(
    "astro_dut1_broadcast_seconds", "DUT1 broadcast seconds"
)
GAUGE_APP_UP: Final = Gauge(
    "astro_app_up", "1 if app is running"
)

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

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
    
    _configure_logging(app)

    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    app.cfg = load_config(cfg_path)

    # Initialize metrics
    MET_REQUESTS.labels(route="/api/health-check").inc(0)
    MET_REQUESTS.labels(route="/api/calculate").inc(0)
    MET_REQUESTS.labels(route="/health").inc(0)
    MET_REQUESTS.labels(route="/healthz").inc(0)
    MET_REQUESTS.labels(route="/metrics").inc(0)
    MET_FALLBACKS.labels(requested="placidus", fallback="equal").inc(0)
    MET_WARNINGS.labels(kind="polar_soft_fallback").inc(0)
    GAUGE_APP_UP.set(1.0)

    # Request counting
    @app.before_request
    def _count_requests():
        try:
            p = request.path or ""
            if p.startswith("/api/") or p in ("/health", "/healthz", "/metrics"):
                MET_REQUESTS.labels(route=p).inc()
        except Exception:
            pass

    # Register components
    app.register_blueprint(api)
    _register_health(app)
    _register_errors(app)

    # Metrics endpoint
    @app.route("/metrics", methods=["GET"])
    def metrics_endpoint():
        GAUGE_APP_UP.set(1.0)
        GAUGE_DUT1.set(float(os.environ.get("ASTRO_DUT1_BROADCAST", "0.0")))
        data = generate_latest(REGISTRY)
        return Response(data, mimetype=CONTENT_TYPE_LATEST)

    app.logger.info("App initialized with config path %s", cfg_path)
    return app

app = create_app()

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
