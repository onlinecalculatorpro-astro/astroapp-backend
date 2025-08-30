from __future__ import annotations
from typing import Dict, Any, List

def gaja_kesari(jupiter_sign: int, moon_sign: int) -> bool:
    # Moon and Jupiter in kendra (1,4,7,10) from each other
    d = (jupiter_sign - moon_sign) % 12
    return d in (0,3,6,9)

def rajayoga(lagna_sign: int, lagnadhipati_sign: int, moon_sign: int) -> bool:
    # Simplified: Lagna lord and 10th lord in kendra
    d = (lagnadhipati_sign - lagna_sign) % 12
    return d in (0,3,6,9)

def yoga_scores(flags: Dict[str, bool]) -> float:
    score = 0.0
    if flags.get("gaja_kesari"): score += 0.06
    if flags.get("rajayoga"): score += 0.04
    return score
