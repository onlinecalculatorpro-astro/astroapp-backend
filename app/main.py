# app/main.py
from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from app.api.routes import api
from app.utils.config import load_config
import os


def create_app():
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Load YAML config (plus optional JSONs via env in load_config)
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    app.cfg = load_config(cfg_path)

    # Register API blueprint (mounted at /api by routes.py)
    app.register_blueprint(api)

    # --- Health endpoints (Render + smoke tests) ---
    @app.get("/api/health-check")
    def api_health_check():
        return jsonify(ok=True), 200

    @app.get("/health")
    @app.get("/healthz")
    def health():
        return jsonify(status="ok"), 200

    # --- Error handling: JSON responses + stack traces in logs ---
    @app.errorhandler(HTTPException)
    def handle_http(e: HTTPException):
        return (
            jsonify(error="http_error", code=e.code, name=e.name, description=e.description),
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
