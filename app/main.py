# app/main.py
from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Trust proxy headers (Render/NGINX/etc.)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)  # type: ignore

    # ── Logging ────────────────────────────────────────────────────────────────
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

    # ── Error handlers (always JSON) ──────────────────────────────────────────
    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        app.logger.warning("HTTP %s %s %s: %s", e.code, request.method, request.path, e.description)
        return (
            jsonify(
                ok=False,
                error="http_error",
                code=e.code,
                name=e.name,
                message=e.description,
                path=request.path,
            ),
            e.code,
        )

    @app.errorhandler(Exception)
    def _any(e: Exception):
        app.logger.error(
            "UNHANDLED %s at %s %s\n%s",
            type(e).__name__,
            request.method,
            request.path,
            traceback.format_exc(),
        )
        return jsonify(ok=False, error="internal_error", type=type(e).__name__, message=str(e), path=request.path), 500

    # ── Root & health ─────────────────────────────────────────────────────────
    @app.get("/")
    def _root():
        return jsonify(ok=True, service="astro-backend", health="/healthz"), 200

    @app.get("/healthz")
    def _healthz():
        return jsonify(ok=True, status="ok"), 200

    @app.get("/favicon.ico")
    def _noop_favicon():
        return ("", 204)

    # ── Debug helpers ─────────────────────────────────────────────────────────
    @app.get("/__debug/routes")
    def __debug_routes():
        rules: list[Dict[str, Any]] = []
        for r in app.url_map.iter_rules():
            methods = sorted(m for m in (r.methods or []) if m not in ("HEAD", "OPTIONS"))
            rules.append({"rule": str(r), "endpoint": r.endpoint, "methods": methods})
        rules.sort(key=lambda x: x["rule"])
        return jsonify({"count": len(rules), "rules": rules}), 200

    # Import blueprints with diagnostics (don’t crash app on import errors)
    _ops_bp = None
    _ops_err: Optional[str] = None
    try:
        from app.api.routes import ops_api as _ops_bp  # type: ignore
    except Exception as e:
        _ops_err = repr(e)

    _western_bp = None
    _western_err: Optional[str] = None
    try:
        from app.api.western_routes import western_api as _western_bp  # type: ignore
    except Exception as e:
        _western_err = repr(e)

    _vedic_bp = None
    _vedic_err: Optional[str] = None
    _ENABLE_VEDIC = os.getenv("ENABLE_VEDIC_API", "1").lower() in ("1", "true", "yes", "on")
    try:
        if _ENABLE_VEDIC:
            from app.api.vedic_routes import vedic_api as _vedic_bp  # type: ignore
    except Exception as e:
        _vedic_err = repr(e)

    @app.get("/__debug/imports")
    def __debug_imports():
        return jsonify({
            "ops_blueprint_loaded": _ops_bp is not None,
            "ops_import_error": _ops_err,
            "western_blueprint_loaded": _western_bp is not None,
            "western_import_error": _western_err,
            "vedic_enabled_flag": _ENABLE_VEDIC,
            "vedic_blueprint_loaded": _vedic_bp is not None,
            "vedic_import_error": _vedic_err,
            "blueprints": list(app.blueprints.keys()),
            "mounts": {
                "ops": "(no prefix)",
                "western": "/api/western",
                "vedic": "/api/vedic" if _ENABLE_VEDIC else "(disabled)",
            },
        }), 200

    # ── Register blueprints (all route files must use RELATIVE paths) ─────────
    if _ops_bp is not None:
        app.register_blueprint(_ops_bp)  # no prefix (ops/diagnostics)
    else:
        @app.get("/ops_missing")
        def _ops_missing():
            return jsonify(ok=False, error="ops_blueprint_not_loaded", detail=_ops_err), 500

    if _western_bp is not None:
        app.register_blueprint(_western_bp, url_prefix="/api/western")
    else:
        @app.get("/api/western/health")
        def _western_missing():
            return jsonify(ok=False, error="western_blueprint_not_loaded", detail=_western_err), 500

    if _ENABLE_VEDIC and _vedic_bp is not None:
        app.register_blueprint(_vedic_bp, url_prefix="/api/vedic")
    elif _ENABLE_VEDIC:
        @app.get("/api/vedic/health")
        def _vedic_missing():
            return jsonify(ok=False, error="vedic_blueprint_not_loaded", detail=_vedic_err), 500

    app.logger.info(
        "App initialized; ops=%s western=%s vedic_enabled=%s vedic=%s",
        bool(_ops_bp), bool(_western_bp), _ENABLE_VEDIC, bool(_vedic_bp),
    )
    return app


app = create_app()

# ── CORS ─────────────────────────────────────────────────────────────────────
CORS(
    app,
    resources={r"/.*": {"origins": os.environ.get("CORS_ALLOW_ORIGIN", "*")}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
