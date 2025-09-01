# app/main.py
from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any, Dict, Final, Optional

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

# Optional blueprints (won't crash app if absent)
try:
    from app.api.routes import api as _routes_bp  # type: ignore
except Exception:  # pragma: no cover
    _routes_bp = None

try:
    from app.api.predictions import predictions_bp as _pred_bp  # type: ignore
except Exception:  # pragma: no cover
    _pred_bp = None

# Config loader (optional)
try:
    from app.utils.config import load_config  # type: ignore
except Exception:  # pragma: no cover
    load_config = None  # type: ignore

# Core engines
from app.core.astronomy import compute_chart
from app.core import timescales as _ts

# Prometheus metrics
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    REGISTRY,
    generate_latest,
)

# ───────────────────────────── metrics ─────────────────────────────
MET_REQUESTS: Final = Counter("astro_api_requests_total", "API requests", ["route"])
MET_FALLBACKS: Final = Counter(
    "astro_house_fallback_total", "House fallbacks at high latitude", ["requested", "fallback"]
)
MET_WARNINGS: Final = Counter("astro_warning_total", "Non-fatal warnings", ["kind"])
GAUGE_DUT1: Final = Gauge("astro_dut1_broadcast_seconds", "DUT1 broadcast seconds")
GAUGE_APP_UP: Final = Gauge("astro_app_up", "1 if app is running")
REQ_LATENCY: Final = Histogram("astro_request_seconds", "API request latency", ["route"])


# ───────────────────────────── helpers ─────────────────────────────
def _configure_logging(app: Flask) -> None:
    """Unify logging with Gunicorn or fall back to std logging."""
    gerr = logging.getLogger("gunicorn.error")
    if gerr.handlers:
        app.logger.handlers = gerr.handlers
        app.logger.setLevel(gerr.level)
    else:
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def _register_errors(app: Flask) -> None:
    """Consistent JSON error payloads."""
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
    """Health endpoints for Render/CF probes."""
    @app.route("/", methods=["GET"])
    def root():
        return jsonify(ok=True, service="astro-backend", health="/health"), 200

    @app.route("/api/health-check", methods=["GET"])
    def api_health():
        return jsonify(ok=True), 200

    @app.route("/health", methods=["GET"])
    @app.route("/healthz", methods=["GET"])
    def health():
        return jsonify(status="ok"), 200


def _metrics_auth_ok() -> bool:
    """Basic Auth guard for /metrics, using METRICS_USER + METRICS_PASS env vars."""
    auth = request.authorization
    user = os.getenv("METRICS_USER", "")
    pw = os.getenv("METRICS_PASS", "")
    return bool(
        auth
        and auth.type == "basic"
        and auth.username == user
        and auth.password == pw
        and user
        and pw
    )


def _body_json() -> Dict[str, Any]:
    try:
        data = request.get_json(force=True, silent=False)  # raise if invalid JSON
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data
    except Exception as e:
        raise HTTPException(description=f"Invalid JSON: {e}", response=None, code=400)  # type: ignore[arg-type]


def _compute_timescales_payload(date: str, time_s: str, tz: str) -> Dict[str, Any]:
    """
    Convert civil date/time/tz → {jd_ut, jd_tt, jd_ut1, utc}.
    Uses app.core.timescales; UT1 is UT + DUT1 (env) seconds.
    """
    # Build UTC ISO and JD(UT)
    utc_iso = _ts.to_utc_iso(date, time_s, tz)
    jd_ut = _ts.julian_day_utc(date, time_s, tz)

    # TT from UT via ΔT poly
    # Get a UTC datetime again to extract year/month for ΔT
    from datetime import datetime
    from zoneinfo import ZoneInfo

    dt_utc = datetime.fromisoformat(f"{date}T{time_s}:00").replace(tzinfo=ZoneInfo(tz)).astimezone(ZoneInfo("UTC"))
    jd_tt = _ts.jd_tt_from_utc_jd(jd_ut, dt_utc.year, dt_utc.month)

    # UT1 = UT + DUT1 (seconds) / 86400
    dut1 = float(os.getenv("ASTRO_DUT1_BROADCAST", os.getenv("ASTRO_DUT1", "0.0")))
    jd_ut1 = jd_ut + (dut1 / 86400.0)

    return {
        "utc": utc_iso,
        "jd_ut": float(jd_ut),
        "jd_tt": float(jd_tt),
        "jd_ut1": float(jd_ut1),
        "dut1": float(dut1),
    }


def _register_core_api(app: Flask) -> None:
    """Minimal, stable endpoints used by your browser console tests."""

    # ---- timescales ----
    def _timescales_handler():
        data = _body_json()
        date = data.get("date")
        time_s = data.get("time")
        tz = data.get("tz") or data.get("place_tz")
        if not (isinstance(date, str) and isinstance(time_s, str) and isinstance(tz, str)):
            return jsonify(error="invalid_input", message="Provide 'date', 'time', and 'tz' (IANA)"), 400
        out = _compute_timescales_payload(date, time_s, tz)
        return jsonify(out), 200

    # Canonical
    app.add_url_rule("/api/time/timescales", "timescales", _timescales_handler, methods=["POST"])
    # Friendly aliases (your console code probes many of these)
    for alias in (
        "/api/timescales",
        "/time/timescales",
        "/timescales",
        "/api/time/convert",
    ):
        app.add_url_rule(alias, f"timescales_alias_{alias}", _timescales_handler, methods=["POST"])

    # ---- chart ----
    def _chart_handler():
        payload = _body_json()

        # If caller passed civil inputs but not jd_*, compute them here
        have_all_jd = all(k in payload for k in ("jd_ut", "jd_tt", "jd_ut1"))
        if not have_all_jd:
            date = payload.get("date"); time_s = payload.get("time")
            tz = payload.get("tz") or payload.get("place_tz")
            if isinstance(date, str) and isinstance(time_s, str) and isinstance(tz, str):
                ts = _compute_timescales_payload(date, time_s, tz)
                payload.update({k: ts[k] for k in ("jd_ut", "jd_tt", "jd_ut1")})

        # Default mode if not provided
        payload.setdefault("mode", "tropical")

        res = compute_chart(payload)
        return jsonify(res), 200

    # Canonical
    app.add_url_rule("/api/chart", "chart", _chart_handler, methods=["POST"])
    # Aliases probed by your tests
    for alias in (
        "/chart",
        "/api/compute_chart",
        "/compute_chart",
        "/api/astronomy/chart",
        "/astronomy/chart",
        "/api/astro/chart",
    ):
        app.add_url_rule(alias, f"chart_alias_{alias}", _chart_handler, methods=["POST"])


# ───────────────────────────── app factory ─────────────────────────────
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Trust proxy headers (Render/Cloudflare)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)  # type: ignore

    _configure_logging(app)

    # Config file (optional)
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    if load_config:
        try:
            app.cfg = load_config(cfg_path)  # type: ignore[attr-defined]
        except Exception:
            app.cfg = {}  # type: ignore[attr-defined]

    # Pre-seed metrics
    seeded_routes = (
        "/",
        "/api/health-check",
        "/api/chart",
        "/api/time/timescales",
        "/health",
        "/healthz",
        "/metrics",
    )
    for route in seeded_routes:
        MET_REQUESTS.labels(route=route).inc(0)
        REQ_LATENCY.labels(route=route).observe(0.0)
    MET_FALLBACKS.labels(requested="placidus", fallback="equal").inc(0)
    MET_WARNINGS.labels(kind="polar_soft_fallback").inc(0)
    GAUGE_APP_UP.set(1.0)

    # Request metrics
    @app.before_request
    def _before():
        try:
            p = (request.path or "")
            if p.startswith("/api/") or p in ("/", "/health", "/healthz", "/metrics"):
                MET_REQUESTS.labels(route=p).inc()
                request._t0 = perf_counter()
        except Exception:
            pass

    @app.after_request
    def _after(resp):
        try:
            p = (request.path or "")
            if (p.startswith("/api/") or p in ("/", "/health", "/healthz")) and hasattr(request, "_t0"):
                dt = perf_counter() - request._t0
                REQ_LATENCY.labels(route=p).observe(dt)
        except Exception:
            pass
        return resp

    # Register health/errors first
    _register_health(app)
    _register_errors(app)

    # Core minimal API (timescales + chart)
    _register_core_api(app)

    # Optional feature blueprints
    if _routes_bp is not None:
        app.register_blueprint(_routes_bp)
    if _pred_bp is not None:
        app.register_blueprint(_pred_bp, url_prefix="/api")

    # /metrics (Basic Auth)
    @app.route("/metrics", methods=["GET"])
    def metrics_endpoint():
        if not _metrics_auth_ok():
            return Response("Unauthorized", 401, {"WWW-Authenticate": 'Basic realm="metrics"'})

        GAUGE_APP_UP.set(1.0)
        try:
            GAUGE_DUT1.set(float(os.environ.get("ASTRO_DUT1_BROADCAST", os.environ.get("ASTRO_DUT1", "0.0"))))
        except Exception:
            pass

        data = generate_latest(REGISTRY)
        return Response(data, mimetype=CONTENT_TYPE_LATEST)

    app.logger.info("App initialized; core endpoints ready")
    return app


# ───────────────────────────── app instance ─────────────────────────────
app = create_app()

# CORS for browser UIs (Netlify frontend, etc.)
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
