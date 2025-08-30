from __future__ import annotations
import time
from functools import wraps
from flask import request, jsonify

_buckets = {}

def rate_limit(max_per_minute: int, key_fn=None):
    window = 60.0
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = key_fn(request) if key_fn else request.remote_addr or "anon"
            bucket = _buckets.get(key, {"ts": now, "tokens": max_per_minute})
            # refill
            elapsed = now - bucket["ts"]
            bucket["tokens"] = min(max_per_minute, bucket["tokens"] + elapsed*(max_per_minute/window))
            bucket["ts"] = now
            if bucket["tokens"] < 1.0:
                retry = int( max(1, 1 + (1.0 - bucket["tokens"]) * (window/max_per_minute)) )
                return jsonify({"error":"rate_limited","retry_after_sec": retry}), 429
            bucket["tokens"] -= 1.0
            _buckets[key] = bucket
            return f(*args, **kwargs)
        return wrapper
    return decorator
