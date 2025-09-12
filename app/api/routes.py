# app/api/routes.py
"""
Ops & Diagnostics routes (mounted with NO prefix by main.py)

This file intentionally contains only "ops" endpoints:
- Health checks that won't collide with "/" or "/healthz" in main.py
- Build/version and safe runtime config
- Lightweight diagnostics to complement /__debug/*

Feature APIs live elsewhere:
- Western API:  app/api/western_routes.py   -> mounted at /api/western
- Vedic API:    app/api/vedic_routes.py     -> mounted at /api/vedic
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import os

from flask import Blueprint, jsonify, current_app

ops_api = Blueprint("ops_api", __name__)

# Try to import a version string if your app provides one
try:
    from app.version import VERSION  # type: ignore
except Exception:
    VERSION = os.environ.get("APP_VERSION", "0.0.0")


@ops_api.get("/ops/health")
def ops_health():
    """Independent health endpoint (kept separate from root /healthz)."""
    return jsonify(ok=True, service="astro-backend", scope="ops", status="ok"), 200


@ops_api.get("/api/health")
def api_health_backcompat():
    """Backward-compat health path for older clients that ping /api/health."""
    return jsonify(ok=True, service="astro-backend", scope="api", status="ok"), 200


@ops_api.get("/ops/version")
def ops_version():
    """
    Build/version info. Includes optional git SHA if provided via env.
    """
    git_sha = os.environ.get("GIT_SHA") or os.environ.get("COMMIT_SHA")
    return jsonify({
        "ok": True,
        "service": "astro-backend",
        "version": str(VERSION),
        "git_sha": git_sha,
    }), 200


@ops_api.get("/ops/config")
def ops_config():
    """
    Minimal, safe runtime config snapshot (no secrets).
    Expand as needed, but DO NOT include secrets or sensitive values.
    """
    cfg: Dict[str, Any] = {
        "enable_vedic_api": os.getenv("ENABLE_VEDIC_API", "1"),
        "cors_allow_origin": os.getenv("CORS_ALLOW_ORIGIN", "*"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "env": os.getenv("ENVIRONMENT") or os.getenv("ENV") or "unknown",
    }
    return jsonify(ok=True, config=cfg), 200


@ops_api.get("/ops/diag")
def ops_diag():
    """
    Lightweight diagnostics about blueprint registration & mounts.
    Complements /__debug/imports (which is defined in main.py).
    """
    app = current_app
    mounts = {
        "ops": "(no prefix)",
        "western": "/api/western",
        "vedic": "/api/vedic" if os.getenv("ENABLE_VEDIC_API", "1").lower() in ("1", "true", "yes", "on") else "(disabled)",
    }
    return jsonify({
        "ok": True,
        "blueprints": list(app.blueprints.keys()),
        "mounts": mounts,
        "url_map_count": len(list(app.url_map.iter_rules())),
    }), 200
