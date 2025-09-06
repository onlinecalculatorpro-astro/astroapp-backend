from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import Callable, Dict, Optional

from flask import request, jsonify

# In-memory token buckets (per key). Note: per-process only (typical for WSGI).
_buckets: Dict[str, "Bucket"] = {}
_lock = RLock()

# Opt-in env toggles
_DISABLE_ALL = os.getenv("ASTRO_RL_DISABLE", "0").lower() in ("1", "true", "yes", "on")
_ALLOWLIST = {s.strip() for s in os.getenv("ASTRO_RL_ALLOWLIST", "").split(",") if s.strip()}

# Default key: client IP (X-Forwarded-For aware) + endpoint
def _default_key(req) -> str:
    ip = (req.headers.get("X-Forwarded-For", "").split(",")[0].strip()) or (req.remote_addr or "anon")
    return f"{ip}:{(req.endpoint or req.path) or '*'}"

@dataclass
class Bucket:
    tokens: float
    capacity: float          # burst capacity
    rate: float              # tokens per second
    ts: float                # last refill time (monotonic)
    limit: int               # advertised limit (per window)
    window: float            # window seconds (for headers/cleanup)

def _now() -> float:
    # monotonic avoids wall-clock jumps
    return time.monotonic()

def _cleanup(now: float) -> None:
    """Opportunistically evict idle full buckets to cap memory."""
    # Light sweep at most every ~30s
    sweep_every = 30.0
    last = getattr(_cleanup, "_last", 0.0)
    if now - last < sweep_every:
        return
    setattr(_cleanup, "_last", now)

    idle_for = 3 * 60.0  # ≈3 windows by default
    dead: list[str] = []
    for k, b in _buckets.items():
        if b.tokens >= b.capacity and (now - b.ts) > max(idle_for, 3 * b.window):
            dead.append(k)
    for k in dead:
        _buckets.pop(k, None)

def rate_limit(
    max_per_minute: int,
    key_fn: Optional[Callable] = None,
    *,
    burst: Optional[int] = None,
    window: float = 60.0,
    cost: float = 1.0,
    cost_fn: Optional[Callable] = None,
):
    """
    Token-bucket rate limiter (per-process).

    Args:
        max_per_minute: steady rate (tokens/min).
        key_fn: function(request) -> str to partition buckets. Defaults to IP+endpoint.
        burst: bucket capacity (defaults to max_per_minute).
        window: informational window (seconds) for headers. Rate = max_per_minute / 60.
        cost: static token cost per call (can be fractional).
        cost_fn: function(request) -> float to compute dynamic cost.

    Returns:
        Flask decorator producing 429 with JSON + Retry-After when limited.
    """
    limit = int(max_per_minute)
    cap = float(burst if burst is not None else max(limit, 1))
    rate = float(limit) / 60.0  # tokens per second

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if _DISABLE_ALL:
                return f(*args, **kwargs)

            # Skip lightweight methods
            if request.method in ("HEAD", "OPTIONS"):
                return f(*args, **kwargs)

            # Build key and allowlist check
            kfun = key_fn or _default_key
            key = str(kfun(request))
            ip_part = key.split(":", 1)[0]
            if ip_part in _ALLOWLIST:
                return f(*args, **kwargs)

            now = _now()
            with _lock:
                _cleanup(now)
                b = _buckets.get(key)
                if b is None:
                    b = Bucket(tokens=cap, capacity=cap, rate=rate, ts=now, limit=limit, window=window)
                    _buckets[key] = b
                # Refill
                elapsed = max(0.0, now - b.ts)
                b.tokens = min(b.capacity, b.tokens + elapsed * b.rate)
                b.ts = now

                # Determine request cost
                c = float(cost_fn(request)) if cost_fn else float(cost)
                c = max(0.0, c)

                # Not enough tokens? compute Retry-After
                if b.tokens + 1e-12 < c:
                    deficit = max(0.0, c - b.tokens)
                    retry_after = max(1, math.ceil(deficit / b.rate))  # seconds
                    headers = {
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(b.limit),
                        "X-RateLimit-Remaining": str(max(0, math.floor(b.tokens))),
                        "X-RateLimit-Window": str(int(b.window)),
                        "X-RateLimit-Policy": f"token-bucket; window={int(b.window)}s; burst={int(b.capacity)}",
                    }
                    payload = {
                        "ok": False,
                        "error": "rate_limited",
                        "details": {"retry_after_seconds": retry_after, "key": key},
                    }
                    return jsonify(payload), 429, headers

                # Consume and proceed
                b.tokens -= c
                remaining = max(0, math.floor(b.tokens))
                # Approximate seconds to full (informational)
                to_full = 0 if b.tokens >= b.capacity else math.ceil((b.capacity - b.tokens) / b.rate)

            # Set headers on successful responses too
            resp = f(*args, **kwargs)
            try:
                # Flask allows (body, status, headers) tuples; merge if needed
                body, status, headers = (resp if isinstance(resp, tuple) else (resp, None, {}))
            except Exception:
                body, status, headers = resp, None, {}

            # Merge rate headers
            hdrs = dict(headers or {})
            hdrs.setdefault("X-RateLimit-Limit", str(limit))
            hdrs["X-RateLimit-Remaining"] = str(remaining)
            hdrs.setdefault("X-RateLimit-Window", str(int(window)))
            hdrs.setdefault("X-RateLimit-Policy", f"token-bucket; window={int(window)}s; burst={int(cap)}")
            hdrs["X-RateLimit-Reset-After"] = str(int(to_full))

            return (body, status, hdrs) if status is not None else (body, 200, hdrs)

        return wrapper
    return decorator
