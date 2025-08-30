# app/core/leapseconds.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import json, os, math

import erfa  # PyERFA: has dat() for TAI-UTC

@dataclass
class LeapInfo:
    delta_at: float                 # TAI-UTC seconds
    source: str                     # "erfa", "override", "builtin"
    status: str                     # "ok", "stale", "overridden", "erfa-dubious"
    last_known_mjd: float           # last known change MJD in the active table
    erfa_status_code: Optional[int] # erfa.dat status (0 ok, +1 dubious year, etc.)
    notes: Optional[str] = None

# ---- Built-in table (matches ERFA through 2017-01-01) ----
# Format: (MJD, ΔAT seconds) effective from MJD at 00:00 UTC onward.
_BUILTIN_STEPS: List[Tuple[float, float]] = [
    (41317.0, 10.0), (41499.0, 11.0), (41683.0, 12.0), (42048.0, 13.0),
    (42413.0, 14.0), (42778.0, 15.0), (43144.0, 16.0), (43509.0, 17.0),
    (43874.0, 18.0), (44239.0, 19.0), (44786.0, 20.0), (45151.0, 21.0),
    (45516.0, 22.0), (46247.0, 23.0), (47161.0, 24.0), (47892.0, 25.0),
    (48257.0, 26.0), (48804.0, 27.0), (49169.0, 28.0), (49534.0, 29.0),
    (50083.0, 30.0), (50630.0, 31.0), (51179.0, 32.0), (53736.0, 33.0),
    (54832.0, 34.0), (56109.0, 35.0), (57204.0, 36.0), (57754.0, 37.0),  # 2017-01-01
]
_BUILTIN_LAST_MJD = _BUILTIN_STEPS[-1][0]
_BUILTIN_VERSION = "builtin-2017-01-01"

# Optional ops override via env/JSON:
#   ASTRO_DELTA_AT_OVERRIDE_SECS=38.0
#   ASTRO_DELTA_AT_OVERRIDE_FROM_MJD=60350
#   ASTRO_DELTA_AT_JSON=/app/data/leapseconds.json   # [{"mjd":57754.0,"delta_at":37.0}, ...]
def _load_override_table() -> Optional[List[Tuple[float, float]]]:
    path = os.getenv("ASTRO_DELTA_AT_JSON", "").strip()
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        steps: List[Tuple[float, float]] = []
        for row in data:
            steps.append((float(row["mjd"]), float(row["delta_at"])))
        steps.sort(key=lambda t: t[0])
        return steps
    except Exception:
        return None

_OVERRIDE_TABLE = _load_override_table()

def _delta_at_from_steps(mjd: float, steps: List[Tuple[float, float]]) -> Tuple[float, float]:
    last_mjd = steps[0][0]
    delta = steps[0][1]
    for mjd_thr, value in steps:
        if mjd >= mjd_thr:
            last_mjd = mjd_thr
            delta = value
        else:
            break
    return delta, last_mjd

def delta_at(mjd_utc: float) -> LeapInfo:
    """
    Resolve TAI-UTC (ΔAT) with multi-source strategy:
      1) Ops override (env JSON table or pair of env vars) → source="override"
      2) ERFA dat() → source="erfa" (status 0 ok, 1 dubious -> 'erfa-dubious')
      3) Built-in table → source="builtin"
    Adds a 'stale' status if using builtin/override and we're past the next
    possible insertion boundary relative to the table's last step.
    """
    # 1) explicit override pair
    ov_secs = os.getenv("ASTRO_DELTA_AT_OVERRIDE_SECS", "").strip()
    ov_from = os.getenv("ASTRO_DELTA_AT_OVERRIDE_FROM_MJD", "").strip()
    if ov_secs and ov_from:
        try:
            ov_s = float(ov_secs); ov_mjd = float(ov_from)
            if mjd_utc >= ov_mjd:
                return LeapInfo(
                    delta_at=ov_s,
                    source="override",
                    status="overridden",
                    last_known_mjd=ov_mjd,
                    erfa_status_code=None,
                    notes=f"env override from MJD {ov_mjd} (ΔAT={ov_s}s)"
                )
        except ValueError:
            pass

    # 1b) override JSON table
    if _OVERRIDE_TABLE:
        d, last_mjd = _delta_at_from_steps(mjd_utc, _OVERRIDE_TABLE)
        # If override table extends beyond builtin last MJD, mark as override
        if last_mjd >= _BUILTIN_LAST_MJD:
            return LeapInfo(
                delta_at=d, source="override", status="overridden",
                last_known_mjd=last_mjd, erfa_status_code=None,
                notes="override JSON table in use"
            )

    # 2) ERFA dat()
    # erfa.dat takes UTC JD split; returns (delta, status)
    jd = mjd_utc + 2400000.5
    jd1 = math.floor(jd)
    jd2 = jd - jd1
    d, stat = erfa.dat(jd1, jd2)  # type: ignore
    if stat in (0, 1):  # 0 ok, 1 dubious (future/past range)
        return LeapInfo(
            delta_at=float(d), source="erfa",
            status=("ok" if stat == 0 else "erfa-dubious"),
            last_known_mjd=_BUILTIN_LAST_MJD,  # ERFA doesn’t expose last step; use builtin for reference
            erfa_status_code=stat,
            notes=None if stat == 0 else "ERFA marked date as dubious year"
        )

    # 3) Built-in table
    d, last_mjd = _delta_at_from_steps(mjd_utc, _BUILTIN_STEPS)

    # Freshness heuristic: leap seconds can *only* change on Jun 30 / Dec 31.
    # If we are beyond at least one such boundary after last_mjd, mark 'stale'.
    def _is_past_next_boundary(mjd_ref: float, mjd_now: float) -> bool:
        # boundaries are approx every ~182/183 days; simple check:
        return (mjd_now - mjd_ref) >= 183.0

    stale = _is_past_next_boundary(last_mjd, mjd_utc)

    return LeapInfo(
        delta_at=d,
        source="builtin",
        status=("stale" if stale else "ok"),
        last_known_mjd=last_mjd,
        erfa_status_code=stat,
        notes=("builtin table beyond next boundary" if stale else None),
    )
