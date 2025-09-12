# app/core/leapseconds.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import json
import os
import math

import erfa  # PyERFA: provides dj2cal (JD->calendar) and dat(iy,im,id,fd)

__all__ = [
    "LeapInfo",
    "delta_at",            # primary: mjd_utc -> LeapInfo
    "delta_at_from_jd",    # utc JD -> LeapInfo
    "delta_at_from_cal",   # (iy,im,id,fd_utc_day) -> LeapInfo
    "active_table_info",   # metadata about which table is active/known
]

@dataclass(frozen=True)
class LeapInfo:
    delta_at: float                 # TAI-UTC seconds (ΔAT)
    source: str                     # "erfa", "override", "builtin"
    status: str                     # "ok", "stale", "overridden"
    last_known_mjd: float           # last known step MJD in the active/baseline table
    erfa_status_code: Optional[int] # pyERFA does not expose a code; keep None for API continuity
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

# ---- Optional ops override via env/JSON ----
#   ASTRO_DELTA_AT_OVERRIDE_SECS=38.0
#   ASTRO_DELTA_AT_OVERRIDE_FROM_MJD=60350
#   ASTRO_DELTA_AT_JSON=/app/data/leapseconds.json   # [{"mjd":57754.0,"delta_at":37.0}, ...]

def _load_override_table() -> Optional[List[Tuple[float, float]]]:
    path = (os.getenv("ASTRO_DELTA_AT_JSON") or "").strip()
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

def _is_past_next_boundary(mjd_ref: float, mjd_now: float) -> bool:
    # Leap seconds happen at most on Jun 30 / Dec 31 (~every 182/183 days).
    # Simple freshness check: if we are > ~half a year past the last step,
    # mark the static table as "stale".
    return (mjd_now - mjd_ref) >= 183.0

# ---- ERFA helpers ----

def _erfa_delta_at_calendar(iy: int, im: int, iday: int, fd: float) -> float:
    """
    Call pyERFA dat(iy, im, id, fd) → ΔAT seconds.
    pyERFA returns only the value; status codes are not exposed like in the C API.
    """
    return float(erfa.dat(int(iy), int(im), int(iday), float(fd)))  # type: ignore[arg-type]

def _erfa_delta_at_mjd(mjd_utc: float) -> float:
    """
    Convert MJD UTC → calendar + fractional day via dj2cal, then call dat.
    """
    jd = float(mjd_utc) + 2400000.5
    dj1 = math.floor(jd)
    dj2 = jd - dj1
    iy, im, iday, fd = erfa.dj2cal(dj1, dj2)
    return _erfa_delta_at_calendar(iy, im, iday, fd)

# ---- Public API ----

def delta_at(mjd_utc: float) -> LeapInfo:
    """
    Resolve TAI−UTC (ΔAT) for a given UTC MJD using:
      1) Explicit overrides (env pair or JSON table) → source="override"
      2) ERFA dat() via calendar → source="erfa"
      3) Built-in table → source="builtin" (status "stale" if we are past a likely boundary)
    """
    mjd = float(mjd_utc)

    # 1) explicit override pair
    ov_secs = (os.getenv("ASTRO_DELTA_AT_OVERRIDE_SECS") or "").strip()
    ov_from = (os.getenv("ASTRO_DELTA_AT_OVERRIDE_FROM_MJD") or "").strip()
    if ov_secs and ov_from:
        try:
            ov_s = float(ov_secs)
            ov_mjd = float(ov_from)
            if mjd >= ov_mjd:
                return LeapInfo(
                    delta_at=ov_s,
                    source="override",
                    status="overridden",
                    last_known_mjd=ov_mjd,
                    erfa_status_code=None,
                    notes=f"env override from MJD {ov_mjd} (ΔAT={ov_s}s)",
                )
        except ValueError:
            # ignore malformed env override
            pass

    # 1b) override JSON table
    if _OVERRIDE_TABLE:
        d, last_mjd = _delta_at_from_steps(mjd, _OVERRIDE_TABLE)
        # Treat as override regardless of recency; operator intent is explicit.
        return LeapInfo(
            delta_at=d,
            source="override",
            status="overridden",
            last_known_mjd=last_mjd,
            erfa_status_code=None,
            notes="override JSON table in use",
        )

    # 2) ERFA dat() via calendar decomposition
    try:
        d = _erfa_delta_at_mjd(mjd)
        return LeapInfo(
            delta_at=d,
            source="erfa",
            status="ok",
            last_known_mjd=_BUILTIN_LAST_MJD,  # reference for diags
            erfa_status_code=None,             # pyERFA doesn't expose status
            notes=None,
        )
    except Exception as e:
        # Fall through to builtin table if ERFA fails for any reason.
        pass

    # 3) Built-in table fallback
    d, last_mjd = _delta_at_from_steps(mjd, _BUILTIN_STEPS)
    stale = _is_past_next_boundary(last_mjd, mjd)
    return LeapInfo(
        delta_at=d,
        source="builtin",
        status=("stale" if stale else "ok"),
        last_known_mjd=last_mjd,
        erfa_status_code=None,
        notes=("builtin table beyond next boundary" if stale else None),
    )

def delta_at_from_jd(jd_utc: float) -> LeapInfo:
    """Convenience wrapper: UTC JD → LeapInfo."""
    mjd = float(jd_utc) - 2400000.5
    return delta_at(mjd)

def delta_at_from_cal(iy: int, im: int, iday: int, fd_utc: float) -> LeapInfo:
    """
    Convenience wrapper: calendar (UTC) → LeapInfo.
    Builds MJD from calendar via ERFA, then delegates to delta_at.
    """
    # Convert calendar → JD with cal2jd + back to MJD
    dj1, dj2 = erfa.cal2jd(int(iy), int(im), int(iday))
    jd = float(dj1) + float(dj2) + float(fd_utc)
    return delta_at_from_jd(jd)

def active_table_info() -> Dict[str, Any]:
    """
    Return a small diagnostic blob describing which table is active/available.
    """
    info: Dict[str, Any] = {
        "builtin": {
            "version": _BUILTIN_VERSION,
            "last_mjd": _BUILTIN_LAST_MJD,
            "steps_count": len(_BUILTIN_STEPS),
        },
        "override_json": bool(_OVERRIDE_TABLE),
    }
    if _OVERRIDE_TABLE:
        info["override_json_last_mjd"] = _OVERRIDE_TABLE[-1][0]
        info["override_json_steps_count"] = len(_OVERRIDE_TABLE)
    # Env overrides (pair)
    info["override_env_pair"] = bool(
        (os.getenv("ASTRO_DELTA_AT_OVERRIDE_SECS") or "").strip()
        and (os.getenv("ASTRO_DELTA_AT_OVERRIDE_FROM_MJD") or "").strip()
    )
    return info
