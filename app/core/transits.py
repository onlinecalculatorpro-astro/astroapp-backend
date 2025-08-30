from __future__ import annotations
from typing import Dict, Any, List
import math

ASPECTS = {
    "conj": 0, "oppo": 180, "trine": 120, "square": 90, "sextile": 60
}

def angular_distance(a: float, b: float) -> float:
    d = (a - b + 540) % 360 - 180
    return abs(d)

def detect_transits(natal: Dict[str, float], transiting: Dict[str, float], orb: float = 4.0) -> Dict[str, float]:
    # Return a simple score proportional to closeness for a subset of triggers
    points = ["Sun","Moon","Asc","MC"]
    planets = ["Jupiter","Saturn","Mars"]
    score = 0.0
    for P in planets:
        p_lon = transiting.get(P)
        if p_lon is None: continue
        for pt in points:
            n_lon = natal.get(pt)
            if n_lon is None: continue
            d = min(angular_distance(p_lon, n_lon),
                    angular_distance(p_lon, (n_lon+120)%360),
                    angular_distance(p_lon, (n_lon+180)%360),
                    angular_distance(p_lon, (n_lon+60)%360))
            if d <= orb:
                score += (orb - d)/orb * 0.35  # scaled
    return min(score, 0.35)
