from __future__ import annotations
from typing import Dict, Any, List

def rectification_candidates(chart_request: Dict[str, Any], window_minutes: int = 90) -> Dict[str, Any]:
    # Deterministic candidates near reported time: -4, 0, +4 minutes
    times = [ -4, 0, 4 ]
    base = sum([ord(c) for c in (chart_request['date'] + chart_request['time'])]) % 100 / 100.0
    scores = [ round(0.65 + 0.2*base - 0.05*abs(m)/4.0, 3) for m in times ]
    best = max(range(len(scores)), key=lambda i: scores[i])
    best_iso = f"{chart_request['date']}T{chart_request['time']}:00Z"
    return {
        "best_time": best_iso,
        "top3_times": [best_iso]*3,
        "composite_scores": scores,
        "confidence_band": "Medium" if max(scores) > 0.72 else "Low",
        "margin_delta": round(abs(scores[best] - sorted(scores)[-2]), 3),
        "features_at_peak": "Transit/Dashā resonance (placeholder)"
    }
