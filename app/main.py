# app/main.py
from __future__ import annotations

import os
import json
import uuid
import time
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, request, g
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from app.api.routes import api
from app.utils.config import load_config
from app.utils.metrics import metrics

# --- Security headers you can tweak if needed ---
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    # HSTS is only respected over HTTPS (Render provides TLS)
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    # If you publish a CSP, start in report-only in case your frontend embeds things
    # "Content-Security-Policy": "default-src 'self'",
}

def create_app():
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["ENV"] = os.environ.get("FLASK_ENV", "production")
    app.config["DEBUG"] = app.config["ENV"] == "development"

    # Load YAML config (plus optional JSONs via env in load_config)
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    app.cfg = load_config(cfg_path)

    # Register API blueprint (mounted at /api by routes.py)
    app.register_blueprint(api)

    # ---- Root landing so Render root isn’t a 404 ----
    @app.get("/")
    def root():
        return jsonify({
            "name": "astroapp-backend",
            "status": "ok",
            "version": getattr(app, "version", None),
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "endpoints": {
                "health": "/health",
                "health_check": "/api/health-check",
                "calculate": "/api/calculate",
                "predictions": "/predictions",
                "rectification_quick": "/rectification/quick",
                "openapi": "/api/openapi",
                "metrics": "/metrics"
            }
        }), 200

    # --- Health endpoints (Render + smoke tests) ---
    @app.get("/api/health-check")
    def api_health_check():
        return jsonify(ok=True), 200

    @app.get("/health")
    @app.get("/healthz")
    def health():
        return jsonify(status="ok"), 200

    # --- Request/response middleware ---
    @app.before_request
    def _before():
        g._t0 = time.time()
        # Correlation / Request ID
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        g.request_id = rid

    @app.after_request
    def _after(resp):
        # add security and correlation headers
        resp.headers.setdefault("X-Request-ID", getattr(g, "request_id", "-"))
        for k, v in SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        # simple latency metric & header
        try:
            dur_ms = int((time.time() - getattr(g, "_t0", time.time())) * 1000)
            resp.headers["X-Response-Time-ms"] = str(dur_ms)
            metrics.observe("http_request_duration_ms", dur_ms, {
                "path": request.path,
                "method": request.method,
                "status": resp.status_code,
            })
        except Exception:
            pass
        return resp

    # --- Error handling: JSON responses + stack traces in logs ---
    @app.errorhandler(HTTPException)
    def handle_http(e: HTTPException):
        # Avoid leaking internals in prod
        payload: dict[str, Any] = {
            "error": "http_error",
            "code": e.code,
            "name": e.name,
            "description": e.description,
            "request_id": getattr(g, "request_id", None),
        }
        return jsonify(payload), e.code

    @app.errorhandler(Exception)
    def handle_any(e: Exception):
        app.logger.exception(e)
        payload: dict[str, Any] = {
            "error": "internal_error",
            "type": e.__class__.__name__,
            "request_id": getattr(g, "request_id", None),
        }
        # show message only in development
        if app.config["DEBUG"]:
            payload["message"] = str(e)
        return jsonify(payload), 500

    return app


app = create_app()

# CORS: lock to your frontend origin (Netlify/Vercel). For local dev, set to "*".
allowed_origin = os.environ.get("NETLIFY_ORIGIN", "https://YOUR_FRONTEND_DOMAIN")
CORS(
    app,
    resources={r"/**": {"origins": [allowed_origin]}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    max_age=600,
)

if __name__ == "__main__":
    # Never use the dev server in a real deployment; run via gunicorn (below).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
