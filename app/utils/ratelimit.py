# app/utils/ratelimit.py
from __future__ import annotations

"""
Simple, production-friendly token-bucket rate limiter for Flask.

Features
- Per-client buckets with optional per-route scoping
- Pluggable key function (use API key, bearer token, or IP)
- Dynamic request "cost" (e.g., heavier endpoints can cost more tokens)
- Thread-safe (per-process) via RLock
- Standards-ish headers: X-RateLimit-* and Retry-After on 429
- Env toggles:
    ASTRO_RL_DISABLE       -> disable limiter entirely
    ASTRO_RL_ALLOWLIST     -> comma-separated list of client ids/IPs to skip
"""

import math
import os
import time
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import Any, Callable, Dict, Optional

from flask import request, jsonify, make_response

__all__ = ["rate_limit", "ip_key", "endpoint_key", "client_key"]

# ───────────────────────── storage / globals ─────────────────────────
_buckets: Dict[str, "Bucket"] = {}  # in-memory; per-process
_lock = RLock()

_DISABLE_ALL = os.getenv("ASTRO_RL_DISABLE", "0").lower() in ("1", "true", "yes", "on")
_ALLOWLIST = {
    s.strip()
    for s in os.getenv("ASTRO_RL_ALLOWLIST", "").split(",")
    if s.strip()
}

# ───────────────────────── key functions ─────────────────────────
def _first_forwarded_for(req) -> str:
    xff = req.headers.get("X-Forwarded-For", "")
    return (xff.split(",")[0].strip() if xff else "") or (req.remote_addr or "anon")


def ip_key(req) -> str:
    """Bucket per client IP (no route scoping)."""
    return _first_forwarded_for(req)


def endpoint_key(req) -> str:
    """Bucket per client IP + endpoint (route-scoped)."""
    return f"{_first_forwarded_for(req)}:{(req.endpoint or req.path) or '*'}"


def client_key(req) -> str:
    """
    Prefer API credential; fall back to IP. Route-scoped.
    Credential order:
      1) X-API-Key
      2) Authorization: Bearer <token>
      3) IP
    """
    api_key = (req.headers.get("X-API-Key") or "").strip()
    if not api_key:
        auth = (req.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            api_key = auth.split(None, 1)[1]

    ident = api_key or _first_forwarded_for(req)
    route = (req.endpoint or req.path) or "*"
    return f"{ident}:{route}"


# ───────────────────────── bucket / math ─────────────────────────
@dataclass
class Bucket:
    tokens: float       # current tokens
    capacity: float     # burst capacity
    rate: float         # tokens per second
    ts: float           # last refill time (monotonic)
    limit: int          # advertised limit (per minute)
    window: float       # window seconds (informational / headers)


def _now() -> float:
    return time.monotonic()  # immune to wall-clock jumps


def _refill(b: Bucket, now: float) -> None:
    if now > b.ts:
        b.tokens = min(b.capacity, b.tokens + (now - b.ts) * b.rate)
        b.ts = now


def _cleanup(now: float) -> None:
    """
    Opportunistically evict idle, full buckets so memory stays bounded.
    """
    last = getattr(_cleanup, "_last", 0.0)
    if now - last < 30.0:  # at most every 30s
        return
    setattr(_cleanup, "_last", now)

    idle_for = 3 * 60.0  # ~3 minutes by default
    to_del = [
        k for k, b in _buckets.items()
        if b.tokens >= b.capacity and (now - b.ts) > max(idle_for, 3 * b.window)
    ]
    for k in to_del:
        _buckets.pop(k, None)


# ───────────────────────── public decorator ─────────────────────────
def rate_limit(
    max_per_minute: int,
    key_fn: Optional[Callable[[Any], str]] = None,
    *,
    burst: Optional[int] = None,
    window: float = 60.0,
    cost: float = 1.0,
    cost_fn: Optional[Callable[[Any], float]] = None,
):
    """
    Token-bucket rate limiter.

    Args:
        max_per_minute: Allowed steady rate (tokens/min).
        key_fn: function(request) -> str identifying a bucket.
                Defaults to endpoint_key (IP + route).
        burst: Optional bucket capacity (defaults to max_per_minute).
        window: Logical window for headers (seconds). Rate is still per-minute.
        cost: Static token cost per call (can be fractional).
        cost_fn: function(request) -> float for dynamic cost.

    Behavior:
        • On limit, returns 429 JSON:
            {"ok": False, "error": "rate_limited", "details": {"retry_after_seconds": N}}
        • Adds headers on both success and 429:
            X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, X-RateLimit-Policy
            (and Retry-After on 429)
    """
    if max_per_minute <= 0:
        raise ValueError("max_per_minute must be > 0")
    if window <= 0:
        raise ValueError("window must be > 0")

    limit = int(max_per_minute)
    capacity = float(burst if burst is not None else max(limit, 1))
    rate = float(limit) / 60.0  # tokens per second

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if _DISABLE_ALL:
                return f(*args, **kwargs)

            # Don’t rate limit preflight/lightweight methods
            if request.method in ("HEAD", "OPTIONS"):
                return f(*args, **kwargs)

            # Key + allowlist
            kfun = key_fn or endpoint_key
            bucket_key = str(kfun(request))
            # Allowlist matches either full key or its leading identifier (left of ':')
            leading = bucket_key.split(":", 1)[0]
            if bucket_key in _ALLOWLIST or leading in _ALLOWLIST:
                return f(*args, **kwargs)

            now = _now()
            with _lock:
                _cleanup(now)
                b = _buckets.get(bucket_key)
                if b is None:
                    b = Bucket(tokens=capacity, capacity=capacity, rate=rate,
                               ts=now, limit=limit, window=window)
                    _buckets[bucket_key] = b
                else:
                    _refill(b, now)

                # Determine cost
                req_cost = float(cost_fn(request)) if cost_fn else float(cost)
                req_cost = max(0.0, req_cost)

                # Out of tokens?
                if b.tokens + 1e-12 < req_cost:
                    deficit = max(0.0, req_cost - b.tokens)
                    retry_after = max(1, math.ceil(deficit / b.rate))  # seconds
                    headers = {
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(b.limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(retry_after),
                        "X-RateLimit-Policy": f"{b.limit};w={int(b.window)};burst={int(b.capacity)}",
                    }
                    payload = {
                        "ok": False,
                        "error": "rate_limited",
                        "details": {"retry_after_seconds": retry_after},
                    }
                    resp = make_response(jsonify(payload), 429)
                    for k, v in headers.items():
                        resp.headers[k] = v
                    return resp

                # Consume and proceed
                b.tokens -= req_cost
                remaining = max(0, int(b.tokens))
                # Seconds until *next* token (not until full)
                next_token_sec = max(0, math.ceil((1.0 - (b.tokens % 1.0)) / b.rate)) if b.tokens < b.capacity else 0

            # Call the view and attach headers
            rv = f(*args, **kwargs)
            resp = make_response(rv)
            resp.headers.setdefault("X-RateLimit-Limit", str(limit))
            resp.headers["X-RateLimit-Remaining"] = str(remaining)
            resp.headers.setdefault("X-RateLimit-Reset", str(next_token_sec))
            resp.headers.setdefault("X-RateLimit-Policy", f"{limit};w={int(window)};burst={int(capacity)}")
            return resp

        return wrapper

    return decorator
