from __future__ import annotations
import time
from typing import Dict, Callable
from functools import wraps
from flask import request

class Metrics:
    def __init__(self):
        self.counters: Dict[str, float] = {}
        self.latency: Dict[str, list] = {}

    def inc(self, name: str, amt: float = 1.0, labels: Dict[str,str] | None = None):
        key = self._key(name, labels)
        self.counters[key] = self.counters.get(key, 0.0) + amt

    def observe(self, name: str, value_ms: float, labels: Dict[str,str] | None = None):
        key = self._key(name, labels)
        self.latency.setdefault(key, []).append(value_ms)
        if len(self.latency[key]) > 1000:
            self.latency[key] = self.latency[key][-1000:]

    def _key(self, name: str, labels: Dict[str,str] | None):
        if not labels:
            return name
        parts = [f'{k}="{v}"' for k,v in sorted(labels.items())]
        return f"{name}{{{','.join(parts)}}}"

    def middleware(self, endpoint_name: str) -> Callable:
        def deco(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                t0 = time.time()
                try:
                    return fn(*args, **kwargs)
                finally:
                    dt = (time.time() - t0) * 1000.0
                    self.inc("astroapp_requests_total", 1.0, {"endpoint": endpoint_name})
                    self.observe("astroapp_request_latency_ms", dt, {"endpoint": endpoint_name})
            return wrapper
        return deco

    def export_prometheus(self) -> str:
        lines = []
        for k, v in self.counters.items():
            lines.append(f"# TYPE {k.split('{')[0]} counter")
            lines.append(f"{k} {v:.0f}")
        for k, samples in self.latency.items():
            if not samples: continue
            avg = sum(samples)/len(samples)
            p95 = sorted(samples)[int(0.95*len(samples))-1] if len(samples) >= 20 else max(samples)
            lines.append(f"# TYPE {k.split('{')[0]} gauge")
            lines.append(f"{k}_avg {avg:.2f}")
            lines.append(f"{k}_p95 {p95:.2f}")
        return "\n".join(lines) + "\n"

metrics = Metrics()
