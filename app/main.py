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

# ── Prometheus (standard mode) ────────────────────────────────────────────
from prometheus_client import (
    Counter,
    Gauge,
    CONTENT_TYPE_LATEST,
    REGISTRY,
    generate_latest,
)

# Custom metrics (globals so other modules can import/label)
MET_REQUESTS: Final = Counter(
    "astro_api_requests_total", "API requests", ["route"]
)
MET_FALLBACKS: Final = Counter(
    "astro_house_fallback_total", "House fallbacks at high latitude", ["requested", "fallback"]
)
MET_WARNINGS: Final = Counter(
    "astro_warning_total", "Non-fatal warnings", ["kind"]
)

# Gauges - standard mode (no multiprocess)
GAUGE_DUT1: Final = Gauge(
    "astro_dut1_broadcast_seconds", "DUT1 broadcast seconds"
)
GAUGE_APP_UP: Final = Gauge(
    "astro_app_up", "1 if app is running"
)

def _initialize_metrics() -> None:
    """Initialize metrics with starter values"""
    try:
        MET_REQUESTS.labels(route="/api/health-check").inc(0)
        MET_REQUESTS.labels(route="/api/calculate").inc(0)
        MET_REQUESTS.labels(route="/health").inc(0)
        MET_REQUESTS.labels(route="/healthz").inc(0)
        MET_REQUESTS.labels(route="/metrics").inc(0)
        MET_FALLBACKS.labels(requested="placidus", fallback="equal").inc(0)
        MET_WARNINGS.labels(kind="polar_soft_fallback").inc(0)
        MET_WARNINGS.labels(kind="polar_reject_strict").inc(0)
        MET_WARNINGS.labels(kind="leap_policy_warn").inc(0)
        GAUGE_APP_UP.set(1.0)
        print("DEBUG: Metrics initialized successfully")
    except Exception as e:
        print(f"DEBUG: Metrics initialization failed: {e}")

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
    @app.get("/api/health-check")
    def api_health():
        return jsonify(ok=True), 200

    @app.get("/health")
    @app.get("/healthz")
    def health():
        return jsonify(status="ok"), 200

def _register_metrics(app: Flask) -> None:
    """Register metrics endpoints and request counting"""
    
    @app.before_request
    def _count_req():
        try:
            p = request.path or ""
            if p.startswith("/api/") or p in ("/health", "/healthz", "/metrics"):
                MET_REQUESTS.labels(route=p).inc()
        except Exception as e:
            app.logger.error(f"Metrics increment failed: {e}")

    @app.route("/metrics", methods=["GET"])
    def metrics():
        """Prometheus metrics endpoint"""
        try:
            app.logger.info("DEBUG: Metrics endpoint called")
            
            # Update gauges on each request
            GAUGE_APP_UP.set(1.0)
            GAUGE_DUT1.set(float(os.environ.get("ASTRO_DUT1_BROADCAST", "0.0")))
            app.logger.info("DEBUG: Gauges updated")
            
            # Check registry state
            collectors = list(REGISTRY._collector_to_names.keys())
            app.logger.info(f"DEBUG: Registry has {len(collectors)} collectors")
            
            # Generate metrics data
            data = generate_latest(REGISTRY)
            app.logger.info(f"DEBUG: Generated {len(data)} bytes of metrics data")
            
            if len(data) < 10:
                app.logger.error("DEBUG: Generated data is too short!")
                return Response("# No metrics data generated\n", mimetype="text/plain")
            
            return Response(data, mimetype=CONTENT_TYPE_LATEST)
            
        except Exception as ex:
            app.logger.exception(f"DEBUG: Metrics endpoint failed: {ex}")
            return Response(f"# metrics error: {ex}\n", mimetype="text/plain")
    
    @app.route("/test-debug", methods=["GET"])
    def test_debug():
        """Test endpoint to verify routing works"""
        app.logger.info("DEBUG: Test debug endpoint called successfully")
        return "Debug endpoint working", 200

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    
    # Trust Render/Proxy headers
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
    
    _configure_logging(app)

    # Load app config
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    app.cfg = load_config(cfg_path)

    # Initialize metrics first
    _initialize_metrics()

    # Register components with error handling
    try:
        app.register_blueprint(api)
        app.logger.info("API blueprint registered successfully")
    except Exception as e:
        app.logger.exception(f"Failed to register API blueprint: {e}")

    try:
        _register_health(app)
        app.logger.info("Health endpoints registered successfully")
    except Exception as e:
        app.logger.exception(f"Failed to register health endpoints: {e}")

    try:
        _register_errors(app)
        app.logger.info("Error handlers registered successfully")
    except Exception as e:
        app.logger.exception(f"Failed to register error handlers: {e}")

    try:
        _register_metrics(app)
        app.logger.info("Metrics endpoints registered successfully")
    except Exception as e:
        app.logger.exception(f"Failed to register metrics endpoints: {e}")
        # Fallback basic metrics endpoint
        @app.route("/metrics")
        def fallback_metrics():
            return "# Fallback metrics endpoint\nastro_app_up 0\n", 200

    app.logger.info("App initialized with config path %s", cfg_path)
    return app

# WSGI target for gunicorn
app = create_app()

# CORS configuration
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
