# app/core/ephem_singleton.py
# -*- coding: utf-8 -*-
"""
Lazy, thread-safe ephemeris singletons.

Exports (lazy):
- TS:         Skyfield Timescale (module attribute, resolved on first access)
- PLANETS:    Skyfield SPK kernel (e.g., de440s.bsp) (module attribute)
- get_timescale() -> TS
- get_planets()   -> PLANETS

Environment:
- EPHEMERIS_DIR     : directory to cache/load kernels (default: ../ephem)
- EPHEMERIS_KERNEL  : preferred kernel name or path (default: 'de440s.bsp')

Design:
- No heavy imports at module import time.
- First access triggers initialization, guarded by an RLock.
- Clear diagnostics if loading fails.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional, Tuple

# ───────────────────────── Configuration ─────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EPHEM_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "ephem"))
_EPHEM_DIR = os.environ.get("EPHEMERIS_DIR", _DEFAULT_EPHEM_DIR)
_KERNEL_PREF = os.environ.get("EPHEMERIS_KERNEL", "de440s.bsp")
_CANDIDATES = (_KERNEL_PREF, "de440s.bsp", "de421.bsp")

# ───────────────────────── State (lazy) ─────────────────────────
_LOCK = threading.RLock()
_TS: Optional[Any] = None
_PLANETS: Optional[Any] = None
_ERR: Optional[BaseException] = None
_META = {
    "dir": _EPHEM_DIR,
    "kernel": None,      # will be set to the resolved kernel name/path
    "candidates": _CANDIDATES,
}

# ───────────────────────── Internals ─────────────────────────
def _load_ephemeris() -> Tuple[Any, Any]:
    """
    Load TS and PLANETS using skyfield, trying configured candidates.
    Raises the original exception if all candidates fail.
    """
    # Import here to keep module import cheap
    from skyfield.api import Loader

    # Ensure directory exists
    os.makedirs(_EPHEM_DIR, exist_ok=True)
    loader = Loader(_EPHEM_DIR)

    # Timescale (fast; avoid network)
    ts = loader.timescale(builtin=True)

    # Try candidates for the kernel
    last_exc: Optional[BaseException] = None
    for cand in _CANDIDATES:
        try:
            planets = loader(cand)  # name or path
            _META["kernel"] = cand
            return ts, planets
        except BaseException as e:  # capture and keep trying
            last_exc = e

    # If we got here, all candidates failed
    msg = (
        "Failed to initialize Skyfield ephemeris. "
        f"Tried: {', '.join(_CANDIDATES)}. "
        f"EPHEMERIS_DIR={_EPHEM_DIR!r}, EPHEMERIS_KERNEL={_KERNEL_PREF!r}"
    )
    raise RuntimeError(msg) from last_exc

def _ensure_loaded() -> None:
    global _TS, _PLANETS, _ERR
    if _TS is not None and _PLANETS is not None:
        return
    with _LOCK:
        if _TS is not None and _PLANETS is not None:
            return
        try:
            ts, planets = _load_ephemeris()
            _TS, _PLANETS, _ERR = ts, planets, None
        except BaseException as e:
            _TS = _PLANETS = None
            _ERR = e

# ───────────────────────── Public API ─────────────────────────
def get_timescale():
    """Return the Skyfield Timescale singleton (lazy)."""
    _ensure_loaded()
    if _TS is None:
        # Defer raising until access time
        raise RuntimeError(
            f"Skyfield Timescale not available. Cause: {_ERR!r}. "
            f"EPHEMERIS_DIR={_EPHEM_DIR!r} EPHEMERIS_KERNEL={_KERNEL_PREF!r}"
        )
    return _TS

def get_planets():
    """Return the Skyfield SPK kernel singleton (lazy)."""
    _ensure_loaded()
    if _PLANETS is None:
        raise RuntimeError(
            f"Skyfield ephemeris kernel not available. Cause: {_ERR!r}. "
            f"EPHEMERIS_DIR={_EPHEM_DIR!r} EPHEMERIS_KERNEL={_KERNEL_PREF!r}"
        )
    return _PLANETS

# Module-level lazy attributes: TS / PLANETS
def __getattr__(name: str):
    if name == "TS":
        return get_timescale()
    if name == "PLANETS":
        return get_planets()
    raise AttributeError(name)

__all__ = ["TS", "PLANETS", "get_timescale", "get_planets"]
