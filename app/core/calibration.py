from __future__ import annotations
import json, math, os
from typing import Dict

def _sigmoid(x: float) -> float:
    return 1.0/(1.0+math.exp(-x))

def temperature_scale(p: float, T: float, bias: float=0.0) -> float:
    # logit transform with temperature
    eps = 1e-6
    p = min(1-eps, max(eps, p))
    logit = math.log(p/(1-p))
    adj = (logit + bias)/max(T, 1e-6)
    return _sigmoid(adj)

def load_calibrators(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("domains", {})

def apply_calibrator(domain: str, p: float, calibrators: Dict[str, Dict]) -> float:
    c = calibrators.get(domain)
    if not c: return p
    if c.get("method") == "temperature":
        return temperature_scale(p, float(c.get("T",1.0)), float(c.get("bias",0.0)))
    return p
