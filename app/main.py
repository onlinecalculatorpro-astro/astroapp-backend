# app/main.py
from __future__ import annotations

import logging
import os
from typing import Tuple

from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from app.api.routes import api
from app.utils.config import load_config

# --- Prometheus metrics (global so other modules can import and increment) ---
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Requests by path (labelled)
MET_REQUESTS  = Counter("astro_api_requests_total", "API requests", ["route"])
# Domain-specific counters you can increment from house/time code:
MET_FALLBACKS = Counter("astro_house_fallback_total", "House fallbacks at high latitude", ["requested", "fallback"])
MET_WARNINGS  = Counter("astro_warning_total", "Non-fatal warnings", ["kind"])
GAUGE_DUT1    = Gauge("astro_dut1_broadcast_seconds", "DUT1 broadcast seconds")


def _configure_logging(app: Flask) -> None:
    """Adopt Gunicorn's log handlers in production if available."""
    gunicorn_error = logging.getLogger("gunicorn.error")
    if gunicorn_error.handlers:
        app.logger.handlers = gunicorn_error.handlers
        app.logger.setLevel(gunicorn_error.level)
    else:
        # Fallback basic config for local/dev
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def handle_http(e: HTTPException):
        # JSON for all HTTP errors
        payload = {
            "error": "http_error",
            "code": e.code,
            "name": e.name,
            "description": e.description,
        }
        return jsonify(payload), e.code

    @app.errorhandler(Exception)
    def handle_any(e: Exception):
        app.logger.exception(e)
        return jsonify(error="internal_error", type=e.__class__.__name__, message=str(e)), 500


def _register_health_endpoints(app: Flask) -> None:
    @app.get("/api/health-check")
    def api_health_check():
        return jsonify(ok=True), 200

    @app.get("/health")
    @app.get("/healthz")
    def health():
        # Render health checks hit /healthz
        return jsonify(status="ok"), 200


def _register_metrics(app: Flask) -> None:
    """Expose /metrics for Prometheus and count requests."""
    # Count every request by path (never fail requests if metrics hiccup)
    @app.before_request
    def _count_req():
        try:
            MET_REQUESTS.labels(route=request.path).inc()
        except Exception:
            pass

    @app.get("/metrics")
    def metrics():
        try:
            # Optional broadcast value for dashboards
            dut1 = float(os.environ.get("ASTRO_DUT1_BROADCAST", "0.0"))
            GAUGE_DUT1.set(dut1)
            return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
        except Exception as ex:
            # Never break scraping: return minimal text if something odd happens
            return Response(f"# metrics error: {ex}\n", mimetype="text/plain")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Trust X-Forwarded-* headers from Render/Proxies
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    _configure_logging(app)

    # Load YAML/JSON config via helper (path may be overridden by env)
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    app.cfg = load_config(cfg_path)

    # Blueprints
    app.register_blueprint(api)

    # Health + errors + metrics
    _register_health_endpoints(app)
    _register_error_handlers(app)
    _register_metrics(app)

    app.logger.info("App initialized with config path %s", cfg_path)
    return app


# Create the WSGI app object for Gunicorn: `gunicorn app.main:app`
app = create_app()

# ---- CORS (production-safe) -------------------------------------------------
# Prefer a single explicit origin in production. You can set either:
#   CORS_ALLOW_ORIGIN=https://your-frontend.example
#   NETLIFY_ORIGIN=https://your-site.netlify.app
# (CORS_ALLOW_ORIGIN takes precedence.)
_allowed_origin = (
    os.environ.get("CORS_ALLOW_ORIGIN")
    or os.environ.get("NETLIFY_ORIGIN")
    or "*"  # temporary default; tighten in prod
)

# IMPORTANT: use r"/.*" (not r"/*") to avoid `re.PatternError` on Python 3.13+
CORS(
    app,
    resources={r"/.*": {"origins": _allowed_origin}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Useful for local dev; Render/Gunicorn will ignore this
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
