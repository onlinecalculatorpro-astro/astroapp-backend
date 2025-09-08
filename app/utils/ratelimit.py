# app/utils/ratelimit.py
from __future__ import annotations
"""
Simple, production-friendly token-bucket rate limiter for Flask.

Features
- Per-client buckets with optional per-route scoping
- Pluggable key function (use API key, bearer token, or IP)
- Dynamic request "cost" (e.g., heavier endpoints cost more tokens)
- Thread-safe (per-process) via RLock
- Standards-ish headers on success and 429:
    X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, X-RateLimit-Policy
  Plus: Retry-After (on 429) and X-RateLimit-Bucket (debugging)
- Env toggles:
    ASTRO_RL_DISABLE        -> disable limiter entirely
    ASTRO_RL_ALLOWLIST      -> comma-separated list of client ids/IPs to skip
    ASTRO_RL_JITTER_429_MS  -> optional random jitter (ms) added to Retry-After to reduce thundering herds

Notes
- Storage is in-memory and per-process. If you run multiple workers, rate limiting is enforced per worker.
- "Reset" header indicates seconds until the next token (not until the bucket is full).
"""

import math
import os
import random
import time
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import Any, Callable, Dict, Optional

from flask import request, jsonify, make_response

__all__ = ["rate_limit", "ip_key", "endpoint_key", "client_key"]

# ───────────────────────── configuration ─────────────────────────

_DISABLE_ALL = os.getenv("ASTRO_RL_DISABLE", "0").strip().lower() in {"1", "true", "yes", "on"}

_ALLOWLIST = {
    s.strip()
    for s in os.getenv("ASTRO_RL_ALLOWLIST", "").split(",")
    if s.strip()
}

# Jitter (in milliseconds) to spread retries when many clients hit the cap simultaneously.
# Set to 0 (default) to disable.
_JITTER_429_MS = 0
try:
    _JITTER_429_MS = max(0, int(os.getenv("ASTRO_RL_JITTER_429_MS", "0")))
except Exception:
    _JITTER_429_MS = 0

# ───────────────────────── rate limit helper function ─────────────────────────

def _RL(env_key: str, default_value: int) -> int:
    """
    Rate limit helper function to read from environment with fallback.
    
    Args:
        env_key: Environment variable key (e.g., "ASTRO_RL_DIRECTIONS_PER_MIN")
        default_value: Default rate limit value if env var not set
        
    Returns:
        Integer rate limit value
    """
    try:
        return int(os.getenv(env_key, str(default_value)))
    except (ValueError, TypeError):
        return default_value

# ───────────────────────── storage / globals ─────────────────────────

@dataclass
class Bucket:
    tokens: float        # current tokens available
    capacity: float      # maximum tokens (burst)
    rate: float          # refill rate (tokens/second)
    ts: float            # last refill time (monotonic seconds)
    limit: int           # advertised steady limit (per minute)
    window: float        # informational window for headers (seconds)

_buckets: Dict[str, Bucket] = {}   # in-memory; per-process
_lock = RLock()

# ───────────────────────── key helpers ─────────────────────────

def _first_forwarded_for(req) -> str:
    xff = (req.headers.get("X-Forwarded-For") or "").strip()
    if xff:
        # first hop is the client (strip spaces around commas)
        return xff.split(",")[0].strip()
    return (req.remote_addr or "anon").strip()

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
            api_key = auth.split(None, 1)[1].strip()

    ident = api_key or _first_forwarded_for(req)
    route = (req.endpoint or req.path) or "*"
    return f"{ident}:{route}"

# ───────────────────────── token bucket machinery ─────────────────────────

def _now() -> float:
    # Monotonic time avoids wall-clock jumps impacting refill math.
    return time.monotonic()

def _refill(b: Bucket, now: float) -> None:
    if now <= b.ts:
        return
    # Add tokens proportional to elapsed time, cap at capacity.
    gained = (now - b.ts) * b.rate
    if gained > 0.0:
        b.tokens = min(b.capacity, b.tokens + gained)
        b.ts = now

def _evict_idle(now: float) -> None:
    """
    Opportunistically evict idle, full buckets so memory stays bounded.
    Evict when:
      - bucket is full; and
      - idle for > max(3 minutes, 3 * window)
    """
    last = getattr(_evict_idle, "_last", 0.0)
    if now - last < 30.0:  # run at most every 30s
        return
    setattr(_evict_idle, "_last", now)

    idle_for = 180.0  # 3 minutes
    to_delete = []
    for k, b in _buckets.items():
        if b.tokens >= b.capacity and (now - b.ts) > max(idle_for, 3.0 * b.window):
            to_delete.append(k)
    for k in to_delete:
        _buckets.pop(k, None)

def _headers(limit: int, remaining: int, reset_seconds: int, capacity: int, window: float, bucket_key: str | None = None) -> Dict[str, str]:
    h = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(max(0, reset_seconds)),
        "X-RateLimit-Policy": f"{limit};w={int(window)};burst={int(capacity)}",
    }
    # Helpful for debugging which bucket was used.
    if bucket_key:
        h["X-RateLimit-Bucket"] = bucket_key
    return h

def _retry_after_seconds(deficit_tokens: float, rate_tps: float) -> int:
    """
    Return whole seconds until enough tokens accumulate for the requested cost.
    Guarantee minimum of 1 second when throttled.
    """
    if rate_tps <= 0.0:
        return 1
    seconds = deficit_tokens / rate_tps
    # ceil to the next whole second, but never return 0 when rate limited
    return max(1, int(math.ceil(seconds)))

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
        max_per_minute: Allowed steady rate (tokens/minute).
        key_fn: function(request) -> str that identifies a bucket.
                Defaults to endpoint_key (IP + route).
        burst: Optional bucket capacity (defaults to max_per_minute). Must be >= 1.
        window: Logical window for headers (seconds). Rate is still per-minute.
        cost: Static token cost per call (can be fractional).
        cost_fn: function(request) -> float for dynamic cost (e.g., heavier requests).

    Behavior:
        • When limited, returns HTTP 429 JSON:
            {"ok": False, "error": "rate_limited", "details": {"retry_after_seconds": N}}
        • Always attaches headers:
            X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, X-RateLimit-Policy
          And on 429 also: Retry-After
    """
    if max_per_minute <= 0:
        raise ValueError("max_per_minute must be > 0")
    if window <= 0:
        raise ValueError("window must be > 0")

    limit = int(max_per_minute)
    capacity = float(burst if burst is not None else max(limit, 1))
    capacity = max(1.0, capacity)
    rate = float(limit) / 60.0  # tokens per second

    def decorator(view_fn):
        @wraps(view_fn)
        def wrapped(*args, **kwargs):
            if _DISABLE_ALL:
                return view_fn(*args, **kwargs)

            # Don't rate limit preflight/lightweight methods
            if request.method in {"HEAD", "OPTIONS"}:
                return view_fn(*args, **kwargs)

            # Resolve bucket key & allowlist
            kfun = key_fn or endpoint_key
            bucket_key = str(kfun(request))
            # Allowlist matches either full bucket key or its leading identifier (left of ':')
            leading = bucket_key.split(":", 1)[0]
            if bucket_key in _ALLOWLIST or leading in _ALLOWLIST:
                return view_fn(*args, **kwargs)

            # Determine cost (defensive)
            try:
                req_cost = float(cost_fn(request)) if cost_fn else float(cost)
            except Exception:
                req_cost = float(cost)
            if not math.isfinite(req_cost) or req_cost < 0.0:
                req_cost = 1.0

            now = _now()
            with _lock:
                _evict_idle(now)

                b = _buckets.get(bucket_key)
                if b is None:
                    b = Bucket(tokens=capacity, capacity=capacity, rate=rate, ts=now, limit=limit, window=window)
                    _buckets[bucket_key] = b
                else:
                    _refill(b, now)

                # Enough tokens?
                if b.tokens + 1e-12 < req_cost:
                    deficit = max(0.0, req_cost - b.tokens)
                    base_retry = _retry_after_seconds(deficit, b.rate)

                    # Optional jitter to reduce stampedes (ms -> seconds fraction)
                    if _JITTER_429_MS > 0:
                        jitter_s = random.randint(0, _JITTER_429_MS) / 1000.0
                        retry_after = max(1, int(math.ceil(base_retry + jitter_s)))
                    else:
                        retry_after = base_retry

                    headers = _headers(
                        limit=b.limit,
                        remaining=0,
                        reset_seconds=retry_after,
                        capacity=int(b.capacity),
                        window=b.window,
                        bucket_key=bucket_key,
                    )
                    headers["Retry-After"] = str(retry_after)

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
                # Remaining whole tokens for header reporting
                remaining_whole = int(max(0.0, math.floor(b.tokens)))
                # Seconds until *next* token
                if b.tokens >= b.capacity - 1e-12:
                    next_token_sec = 0
                else:
                    # time to accumulate the fractional part to the next full token
                    fractional = 1.0 - (b.tokens % 1.0)
                    next_token_sec = int(max(0.0, math.ceil(fractional / b.rate)))

            # Call view
            rv = view_fn(*args, **kwargs)
            resp = make_response(rv)

            # Attach headers if not already set by handler
            hdrs = _headers(
                limit=limit,
                remaining=remaining_whole,
                reset_seconds=next_token_sec,
                capacity=int(capacity),
                window=window,
                bucket_key=bucket_key,
            )
            for k, v in hdrs.items():
                resp.headers.setdefault(k, v)
            return resp

        return wrapped

    return decorator
