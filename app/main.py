# app/main.py
from __future__ import annotations

import os
from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from app.api.routes import api
from app.api.debug_endpoints import debug_api  # <-- debug/test endpoints
from app.utils.config import load_config


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Load YAML config (plus any optional JSONs via load_config)
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    app.cfg = load_config(cfg_path)

    # Register blueprints
    app.register_blueprint(api)         # main API (paths defined inside routes.py)
    app.register_blueprint(debug_api)   # debug/test endpoints for Postman

    # --- Health endpoints (Render + smoke tests) ---
    @app.get("/api/health-check")
    def api_health_check():
        return jsonify(ok=True), 200

    @app.get("/health")
    @app.get("/healthz")
    def health():
        return jsonify(status="ok"), 200

    # --- Error handling: JSON responses; stack traces go to logs ---
    @app.errorhandler(HTTPException)
    def handle_http(e: HTTPException):
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
    def handle_any(e: Exception):
        app.logger.exception(e)
        return jsonify(error="internal_error", type=e.__class__.__name__, message=str(e)), 500

    return app


app = create_app()

# CORS: start permissive; lock to your Netlify origin after frontend is live
allowed_origin = os.environ.get("NETLIFY_ORIGIN", "*")
CORS(
    app,
    resources={r"/*": {"origins": allowed_origin}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
