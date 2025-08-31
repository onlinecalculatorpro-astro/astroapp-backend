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
    "DUT1 broadcast seconds",
    multiprocess_mode="livesum",
)
GAUGE_APP_UP: Final = Gauge(
    "astro_app_up",
    "1 if app is running",
    multiprocess_mode="livesum",
)

# Seed series so first scrape isn't empty
MET_REQUESTS.labels(route="/api/health-check").inc(1)
MET_REQUESTS.labels(route="/api/calculate").inc(1)
MET_FALLBACKS.labels(requested="placidus", fallback="equal").inc(0)
MET_WARNINGS.labels(kind="polar_soft_fallback").inc(0)
MET_WARNINGS.labels(kind="polar_reject_strict").inc(0)
MET_WARNINGS.labels(kind="leap_policy_warn").inc(0)


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


def _register_metrics(app: Flask) -> None:
    @app.before_request
    def _count_req():
        # never let metrics affect the request path
        try:
            MET_REQUESTS.labels(route=request.path).inc()
        except Exception:
            pass

    @app.get("/metrics")
    def metrics():
        try:
            # heartbeat + broadcast every scrape
            GAUGE_APP_UP.set(1.0)
            GAUGE_DUT1.set(float(os.environ.get("ASTRO_DUT1_BROADCAST", "0.0")))

            # Always use default registry - multiprocess can cause issues
            data = generate_latest(REGISTRY)
            
            if not data:
                # Force some data if registry is empty
                MET_REQUESTS.labels(route="/fallback").inc()
                data = generate_latest(REGISTRY)

            return Response(data, mimetype=CONTENT_TYPE_LATEST)
        except Exception as ex:
            # never fail scraping
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

    # touch a series at boot to ensure a shard exists
    try:
        MET_REQUESTS.labels(route="__boot__").inc()
    except Exception:
        pass

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
