from __future__ import annotations
from typing import Dict, Any, List, Tuple

NAKSHATRA_SPAN = 13.3333333333  # 13°20' = 13.333... deg
NAK_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
NAK_YEARS = {"Ketu":7,"Venus":20,"Sun":6,"Moon":10,"Mars":7,"Rahu":18,"Jupiter":16,"Saturn":19,"Mercury":17}

def moon_nakshatra(moon_lon_sidereal: float) -> Tuple[int, float]:
    idx = int((moon_lon_sidereal % 360.0) // NAKSHATRA_SPAN)
    within = (moon_lon_sidereal % NAKSHATRA_SPAN) / NAKSHATRA_SPAN
    return idx, within

def initial_dasha_balance(moon_lon_sidereal: float) -> Tuple[str, float]:
    idx, within = moon_nakshatra(moon_lon_sidereal)
    lord = NAK_ORDER[idx % 9]
    rem_frac = 1.0 - within
    total_years = NAK_YEARS[lord]
    balance_years = total_years * rem_frac
    return lord, balance_years

def mahadasha_sequence(start_lord: str, count: int = 9) -> List[str]:
    i = NAK_ORDER.index(start_lord)
    seq = []
    for k in range(count):
        seq.append(NAK_ORDER[(i + k) % 9])
    return seq

def build_mahadasha_timeline(start_year: float, start_lord: str, years: int = 90) -> List[Dict[str, Any]]:
    # start_year: fractional years remaining for start lord
    out = []
    cur = start_lord
    rem = start_year
    total = 0.0
    while total < years:
        span = rem if total == 0 else NAK_YEARS[cur]
        out.append({"lord": cur, "years": span})
        total += span
        cur = NAK_ORDER[(NAK_ORDER.index(cur)+1)%9]
        rem = NAK_YEARS[cur]
    return out
