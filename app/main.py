# app/main.py
from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, Final

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, BadRequest
from werkzeug.middleware.proxy_fix import ProxyFix

# ───────────────────────────── Optional blueprints ─────────────────────────────
# Make blueprint import failures VISIBLE so 404s are easy to debug.
_routes_import_err: str | None = None
_pred_import_err: str | None = None

try:
    from app.api import routes as _routes_mod  # type: ignore
    _routes_bp = _routes_mod.api
except Exception as e:  # pragma: no cover
    _routes_bp = None
    _routes_import_err = repr(e)
    print("WARNING: routes blueprint failed to import:", _routes_import_err, file=sys.stderr)
    traceback.print_exc()

try:
    from app.api.predictions import predictions_bp as _pred_bp  # type: ignore
except Exception as e:  # pragma: no cover
    _pred_bp = None
    _pred_import_err = repr(e)
    print("WARNING: predictions blueprint failed to import:", _pred_import_err, file=sys.stderr)
    traceback.print_exc()

# ───────────────────────────── Optional config loader ─────────────────────────────
try:
    from app.utils.config import load_config  # type: ignore
except Exception as e:  # pragma: no cover
    print("INFO: app.utils.config.load_config not available:", repr(e), file=sys.stderr)
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
    """Consistent JSON error payloads (and visible 500 bodies for DevTools)."""
    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        app.logger.warning("HTTP %s at %s %s: %s", e.code, request.method, request.path, e.description)
        return jsonify(
            error="http_error",
            code=e.code,
            name=e.name,
            message=e.description,
            path=request.path,
        ), e.code

    @app.errorhandler(Exception)
    def _any(e: Exception):
        tb = traceback.format_exc()
        app.logger.error("UNHANDLED %s at %s %s\n%s", type(e).__name__, request.method, request.path, tb)
        return jsonify(
            error="internal_error",
            type=type(e).__name__,
            message=str(e),
            path=request.path,
        ), 500


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
    """Strict JSON object parser with nice 400s."""
    data = request.get_json(force=True, silent=False)
    if not isinstance(data, dict):
        raise BadRequest("JSON body must be an object")
    return data


def _normalize_time_hms(s: str) -> str:
    """Accept 'HH:MM' or 'HH:MM:SS'; normalize to HH:MM:SS."""
    s = s.strip()
    if len(s.split(":")) == 2:
        hh, mm = s.split(":")
        return f"{int(hh):02d}:{int(mm):02d}:00"
    elif len(s.split(":")) == 3:
        hh, mm, ss = s.split(":")
        return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
    raise BadRequest("time must be 'HH:MM' or 'HH:MM:SS'")


def _compute_timescales_payload(date: str, time_s: str, tz: str) -> Dict[str, Any]:
    """
    Convert civil date/time/tz → {jd_ut, jd_tt, jd_ut1, utc}.
    Uses app.core.timescales; UT1 is UT + DUT1 (env) seconds.
    """
    from zoneinfo import ZoneInfo

    time_norm = _normalize_time_hms(time_s)

    # UTC ISO and JD(UT) via your timescale helpers
    utc_iso = _ts.to_utc_iso(date, time_norm, tz)
    jd_ut = _ts.julian_day_utc(date, time_norm, tz)

    # For ΔT polynomial we need UTC year/month
    dt_utc = (
        datetime.fromisoformat(f"{date}T{time_norm}")
        .replace(tzinfo=ZoneInfo(tz))
        .astimezone(ZoneInfo("UTC"))
    )
    jd_tt = _ts.jd_tt_from_utc_jd(jd_ut, dt_utc.year, dt_utc.month)

    # UT1 = UT + DUT1 (seconds)/86400
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
    """Minimal, stable endpoints used by browser console tests."""

    # ---- timescales ----
    def _timescales_handler():
        data = _body_json()
        date = data.get("date")
        time_s = data.get("time")
        tz = data.get("tz") or data.get("place_tz")
        if not (isinstance(date, str) and isinstance(time_s, str) and isinstance(tz, str)):
            raise BadRequest("Provide 'date', 'time', and 'tz' (IANA)")
        out = _compute_timescales_payload(date, time_s, tz)
        return jsonify(out), 200

    app.add_url_rule("/api/time/timescales", "timescales", _timescales_handler, methods=["POST"])
    for alias in ("/api/timescales", "/time/timescales", "/timescales", "/api/time/convert"):
        app.add_url_rule(alias, f"timescales_alias_{alias}", _timescales_handler, methods=["POST"])

    # ---- chart ----
    def _chart_handler():
        payload = _body_json()

        # If caller passed civil inputs but not jd_*, compute them here
        have_all_jd = all(k in payload for k in ("jd_ut", "jd_tt", "jd_ut1"))
        if not have_all_jd:
            date = payload.get("date")
            time_s = payload.get("time")
            tz = payload.get("tz") or payload.get("place_tz")
            if isinstance(date, str) and isinstance(time_s, str) and isinstance(tz, str):
                ts = _compute_timescales_payload(date, time_s, tz)
                payload.update({k: ts[k] for k in ("jd_ut", "jd_tt", "jd_ut1")})
            else:
                raise BadRequest("Provide either jd_ut/jd_tt/jd_ut1 or civil 'date'+'time'+'tz'")

        # Defaults
        payload.setdefault("mode", "tropical")

        res = compute_chart(payload)  # astronomy.py should handle normalization & warnings
        return jsonify(res), 200

    app.add_url_rule("/api/chart", "chart", _chart_handler, methods=["POST"])
    for alias in ("/chart", "/api/compute_chart", "/compute_chart", "/api/astronomy/chart", "/astronomy/chart", "/api/astro/chart"):
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
    seeded_routes = ("/", "/api/health-check", "/api/chart", "/api/time/timescales", "/health", "/healthz", "/metrics")
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

    # Register health/errors first (so probes always work)
    _register_health(app)
    _register_errors(app)

    # Core minimal API (timescales + chart)
    _register_core_api(app)

    # Optional feature blueprints
    if _routes_bp is not None:
        app.register_blueprint(_routes_bp)
    if _pred_bp is not None:
        app.register_blueprint(_pred_bp, url_prefix="/api")

    # --- DEBUG endpoints: verify blueprint load + list routes + list SPK files ---
    @app.get("/__debug/imports")
    def __debug_imports():
        return jsonify({
            "routes_blueprint_loaded": _routes_bp is not None,
            "routes_import_error": _routes_import_err,
            "predictions_blueprint_loaded": _pred_bp is not None,
            "predictions_import_error": _pred_import_err,
            "blueprints": list(app.blueprints.keys()),
        }), 200

    @app.get("/__debug/routes")
    def __debug_routes():
        rules = []
        for r in app.url_map.iter_rules():
            methods = sorted(m for m in (r.methods or []) if m not in ("HEAD", "OPTIONS"))
            rules.append({"rule": str(r), "endpoint": r.endpoint, "methods": methods})
        rules.sort(key=lambda x: x["rule"])
        return jsonify({"count": len(rules), "rules": rules}), 200

    @app.get("/__debug/files")
    def __debug_files():
        spk_dir = "app/data/spk"
        try:
            files = sorted(os.listdir(spk_dir)) if os.path.isdir(spk_dir) else []
        except Exception as e:
            files = [f"<error: {e}>"]
        return jsonify({"spk": files}), 200
    # ---------------------------------------------------------------------------

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

    app.logger.info(
        "App initialized; core endpoints ready; routes_loaded=%s, predictions_loaded=%s",
        bool(_routes_bp), bool(_pred_bp)
    )
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
