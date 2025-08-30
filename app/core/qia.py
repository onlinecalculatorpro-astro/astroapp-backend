from __future__ import annotations
import math
from typing import Dict

DEFAULT_WEIGHTS = {"dasha":0.40,"transit":0.35,"varga":0.15,"yoga":0.10}

def amplitude_from_evidence(ev: Dict[str, float], a_max: float = 0.8, weights: Dict[str,float] = None) -> float:
    if weights is None: weights = DEFAULT_WEIGHTS
    # simple weighted sum to amplitude, clipped
    a = 0.0
    for k,w in weights.items():
        a += w * float(ev.get(k,0.0))
    return min(a, a_max)

def probability_from_amplitude(a: float, gain: float = 1.0) -> float:
    return min(0.95, max(0.05, (a * gain)))

def qia_adjust(p_raw: float, ev: Dict[str, float]) -> float:
    a = amplitude_from_evidence(ev)
    p_q = probability_from_amplitude(a)
    # combine naive: mean of raw and qia probability
    return (p_raw + p_q) / 2.0
