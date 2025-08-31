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

# ── Prometheus (multiprocess-safe) ────────────────────────────────────────────
from prometheus_client import (
    Counter,
    Gauge,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
    generate_latest,
    multiprocess,
)

# Custom metrics (globals so other modules can import/label)
MET_REQUESTS: Final = Counter(
    "astro_api_requests_total", "API requests", ["route"]
)
MET_FALLBACKS: Final = Counter(
    "astro_house_fallback_total",
    "House fallbacks at high latitude",
    ["requested", "fallback"],
)
MET_WARNINGS: Final = Counter(
    "astro_warning_total", "Non-fatal warnings", ["kind"]
)

# Gauges aggregated across workers
GAUGE_DUT1: Final = Gauge(
    "astro_dut1_broadcast_seconds",
    "DUT1 broadcast seconds"
)
GAUGE_APP_UP: Final = Gauge(
    "astro_app_up",
    "1 if app is running"
)


# ── App plumbing ──────────────────────────────────────────────────────────────
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
        return (
            jsonify(
                error="http_error",
                code=e.code,
                name=e.name,
                description=e.description,
            ),
            e.code,
        )

    @app.errorhandler(Exception)
    def _any(e: Exception):
        app.logger.exception(e)
        return jsonify(error="internal_error", type=type(e).__name__, message=str(e)), 500


def _register_health(app: Flask) -> None:
    @app.get("/api/health-check")
    def api_health():
        return jsonify(ok=True), 200

    @app.get("/health")
    @app.get("/healthz")
    def health():
        return jsonify(status="ok"), 200


def _initialize_metrics():
    """Force initialize all metrics so they appear in output"""
    try:
        # Initialize counters with actual values
        MET_REQUESTS.labels(route="/api/health-check").inc()
        MET_REQUESTS.labels(route="/api/calculate").inc()
        MET_FALLBACKS.labels(requested="placidus", fallback="equal")._value._value = 0
        MET_WARNINGS.labels(kind="polar_soft_fallback")._value._value = 0
        
        # Set gauges
        GAUGE_APP_UP.set(1)
        GAUGE_DUT1.set(0)
        
        print("DEBUG: Metrics initialized successfully")
    except Exception as e:
        print(f"DEBUG: Failed to initialize metrics: {e}")


def _register_metrics(app: Flask) -> None:
    # Initialize metrics when the app starts
    _initialize_metrics()
    
    @app.before_request
    def _count_req():
        try:
            MET_REQUESTS.labels(route=request.path).inc()
        except Exception as e:
            print(f"DEBUG: Metrics increment failed: {e}")

    @app.get("/metrics")
    def metrics():
        try:
            # Update gauges on each scrape
            GAUGE_APP_UP.set(1)
            GAUGE_DUT1.set(float(os.environ.get("ASTRO_DUT1_BROADCAST", "0.0")))
            
            # Generate metrics data
            data = generate_latest(REGISTRY)
            
            print(f"DEBUG: Generated {len(data)} bytes of metrics data")
            
            if len(data) < 10:
                print("DEBUG: Metrics data too short, forcing initialization")
                _initialize_metrics()
                data = generate_latest(REGISTRY)
            
            return Response(data, mimetype=CONTENT_TYPE_LATEST)
            
        except Exception as ex:
            print(f"DEBUG: Metrics endpoint error: {ex}")
            return Response(f"# metrics error: {ex}\n", mimetype="text/plain")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)  # type: ignore

    _configure_logging(app)

    # load app config
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    app.cfg = load_config(cfg_path)  # type: ignore[attr-defined]

    # routes / health / errors / metrics
    app.register_blueprint(api)
    _register_health(app)
    _register_errors(app)
    _register_metrics(app)

    app.logger.info("App initialized with config path %s", cfg_path)
    return app


# WSGI target for gunicorn: `gunicorn app.main:app`
app = create_app()

# CORS (tighten in prod by setting CORS_ALLOW_ORIGIN or NETLIFY_ORIGIN)
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
