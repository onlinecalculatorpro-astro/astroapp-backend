from __future__ import annotations
from typing import Dict, Any
from math import isfinite

def combine(dasha: float, transit: float, varga: float, yoga: float) -> float:
    p = 0.40*dasha + 0.35*transit + 0.15*varga + 0.10*yoga
    p = max(0.05, min(0.95, p))
    return p if isfinite(p) else 0.5
