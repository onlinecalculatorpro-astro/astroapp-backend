# app/api/routes.py
from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo

from flask import Blueprint, jsonify, request

from app.version import VERSION
from app.utils.config import load_config
from app.utils.metrics import metrics
from app.utils.ratelimit import rate_limit

from app.core.validators import (
    ValidationError,
    parse_chart_payload,
    parse_rectification_payload,
)

# canonical chart call
from app.core.astronomy import compute_chart
# NOTE: predict() removed; use new /api/predictions blueprint
from app.core.rectify import rectification_candidates

# houses: prefer policy façade if present
try:
    from app.core.house import (
        compute_houses_with_policy as _houses_fn,   # type: ignore
        list_supported_house_systems as _list_house_systems,  # type: ignore
    )
    _HOUSES_KIND = "policy"
except Exception:
    from app.core.astronomy import compute_houses as _houses_fn  # type: ignore
    def _list_house_systems() -> list[str]:
        return ["placidus", "koch", "regiomontanus", "campanus", "equal", "porphyry"]
    _HOUSES_KIND = "legacy"

# time kernel
from app.core import time_kernel as _tk

# optional leap-seconds helper
try:
    from app.core import leapseconds as _leaps
except Exception:
    _leaps = None  # type: ignore

api = Blueprint("api", __name__)
DEBUG_VERBOSE = os.getenv("ASTRO_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "on")

# ───────────────────────────── helpers (unchanged) ─────────────────────────────
# ... keep _get_bool, _datetime_to_jd_utc, _find_kernel_callable, _compute_timescales_from_local,
# _sig_accepts, _call_compute_chart, _call_compute_houses, _json_error, _normalize_houses_payload …

# ───────────────────────────── endpoints ─────────────────────────────

@api.get("/api/health")
def health():
    return jsonify({"ok": True, "status": "up", "version": VERSION}), 200

@api.get("/api/houses/systems")
def houses_systems():
    try:
        systems = _list_house_systems()
    except Exception:
        systems = []
    return jsonify({"ok": True, "engine": _HOUSES_KIND, "systems": systems}), 200

@api.post("/api/calculate")
def calculate():
    # same as before
    ...

@api.post("/api/report")
def report():
    # same as before
    ...

# 🚨 REMOVED: legacy @api.post("/predictions") endpoint

@api.post("/rectification/quick")
def rect_quick():
    # same as before
    ...

@api.get("/api/openapi")
def openapi_spec():
    # same as before
    ...

@api.get("/system-validation")
def system_validation():
    # same as before
    ...

@api.get("/api/config")
@metrics.middleware("config")
@rate_limit(1)
def config_info():
    # same as before
    ...
