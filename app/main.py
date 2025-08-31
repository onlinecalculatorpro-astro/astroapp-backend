# app/main.py
from __future__ import annotations

import logging
import os
from typing import Final

from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from app.api.routes import api
from app.utils.config import load_config

# ───────────────────────── Prometheus (multiprocess-safe) ─────────────────────
from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    multiprocess,
    REGISTRY,
)

# Request/Domain metrics (globals so other modules can import)
MET_REQUESTS: Final = Counter("astro_api_requests_total", "API requests", ["route"])
MET_FALLBACKS: Final = Counter(
    "astro_house_fallback_total", "House fallbacks at high latitude", ["requested", "fallback"]
)
MET_WARNINGS: Final = Counter("astro_warning_total", "Non-fatal warnings", ["kind"])

# Gauges aggregated across workers
GAUGE_DUT1: Final = Gauge(
    "astro_dut1_broadcast_seconds", "DUT1 broadcast seconds", multiprocess_mode="livesum"
)
# Always-on heartbeat so /metrics is never empty
MET_APP_UP: Final = Gauge(
    "astro_app_up", "1 if Astro backend is running", multiprocess_mode="livesum"
)

# Pre-create some time series (so first scrape shows the metric names)
MET_REQUESTS.labels(route="/api/health-check").inc(0)
MET_REQUESTS.labels(route="/api/calculate").inc(0)
MET_FALLBACKS.labels(requested="placidus", fallback="equal").inc(0)
MET_WARNINGS.labels(kind="polar_soft_fallback").inc(0)
MET_WARNINGS.labels(kind="polar_reject_strict").inc(0)
MET_WARNINGS.labels(kind="leap_policy_warn").inc(0)

# ───────────────────────── Internals ──────────────────────────────────────────
def _configure_logging(app: Flask) -> None:
    """Adopt Gunicorn's log handlers in production if available."""
    gunicorn_error = logging.getLogger("gunicorn.error")
    if gunicorn_error.handlers:
        app.logger.handlers = gunicorn_error.handlers
        app.logger.setLevel(gunicorn_error.level)
    else:
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def handle_http(e: HTTPException):
        return (
            jsonify(
                {
                    "error": "http_error",
                    "code": e.code,
                    "name": e.name,
                    "description": e.description,
                }
            ),
            e.code,
        )

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
        return jsonify(status="ok"), 200


def _register_metrics(app: Flask) -> None:
    """Expose /metrics and count requests (never break requests on metrics issues)."""

    @app.before_request
    def _count_req():
        try:
            MET_REQUESTS.labels(route=request.path).inc()
        except Exception:
            pass  # metrics must never block requests

    @app.get("/metrics")
    def metrics():
        try:
            # Heartbeat + broadcast gauge on every scrape
            MET_APP_UP.set(1.0)
            dut1 = float(os.environ.get("ASTRO_DUT1_BROADCAST", "0.0"))
            GAUGE_DUT1.set(dut1)

            # Aggregate across workers if multiprocess is enabled
            if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry)
                data = generate_latest(registry) or generate_latest(REGISTRY)
            else:
                data = generate_latest(REGISTRY)

            return Response(data, mimetype=CONTENT_TYPE_LATEST)
        except Exception as ex:
            return Response(f"# metrics error: {ex}\n", mimetype="text/plain")


# ───────────────────────── App factory ────────────────────────────────────────
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Trust Render/Proxy headers
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)  # type: ignore

    _configure_logging(app)

    # Config file (env override allowed)
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    app.cfg = load_config(cfg_path)  # type: ignore[attr-defined]

    # Routes
    app.register_blueprint(api)

    # Health, errors, metrics
    _register_health_endpoints(app)
    _register_error_handlers(app)
    _register_metrics(app)

    # Touch a series at boot so a shard exists
    try:
        MET_REQUESTS.labels(route="__boot__").inc()
    except Exception:
        pass

    app.logger.info("App initialized with config path %s", cfg_path)
    return app


# Gunicorn entrypoint: `gunicorn app.main:app`
app = create_app()

# ───────────────────────── CORS (production-safe) ─────────────────────────────
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

# ───────────────────────── Local dev runner ───────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
