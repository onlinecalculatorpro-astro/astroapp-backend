# app/core/__init__.py
# -------------------------------------------------------------------
# Core exports + ephemeris singletons preloaded once per process.
# Import TS/PLANETS from here instead of constructing Loader() in
# request paths. This removes per-request kernel I/O and slashes
# latency under load.
# -------------------------------------------------------------------

from .predictive import (
    # Transits
    TransitEngine, TransitEvent, find_transits_in_range,
    # Dasha
    DashaPeriod, vimsottari_dasha, predict_dasha_periods,
    # Varga
    compute_vargas_for_point, compute_vargas,
    # Yogas
    detect_yogas, house_index_for_longitude,
    # Houses & timescales
    compute_houses, timescales_from_civil,
    # Validation
    EvalResult, evaluate_univariate, permutation_pvalue_corr, bh_fdr,
    holdout_replicate, validate_predictions,
    # Features
    feature_transit_proximity, feature_dasha_lords_onehot, feature_yoga_flags,
)

# ---------------- Ephemeris preload (singleton) -------------------
# Preloads Skyfield timescale + planetary kernel exactly once.
# Use:
#   from app.core import TS, PLANETS
#   t = TS.utc(2025, 1, 1)
#   sun = PLANETS["sun"]
# Config:
#   EPHEMERIS_DIR     : override ephemeris directory (default: app/ephem)
#   EPHEMERIS_KERNEL  : "de440s.bsp" (default) or "de421.bsp" (smaller/faster)
# ------------------------------------------------------------------

import os

# Resolve ephemeris directory relative to this package
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EPHEM_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "ephem"))
_EPHEM_DIR = os.environ.get("EPHEMERIS_DIR", _DEFAULT_EPHEM_DIR)

# Ensure directory exists (safe under concurrent startup)
os.makedirs(_EPHEM_DIR, exist_ok=True)

# Choose kernel (allow env override; default de440s.bsp; fallback to de421.bsp)
_KERNEL_ENV = os.environ.get("EPHEMERIS_KERNEL", "de440s.bsp")
_KERNEL_CANDIDATES = [_KERNEL_ENV, "de440s.bsp", "de421.bsp"]

# Lazy module attribute defaults (will be set below)
TS = None       # type: ignore
PLANETS = None  # type: ignore

try:
    from skyfield.api import Loader

    _loader = Loader(_EPHEM_DIR)
    # Preload timescale (builtin=True uses packaged ΔT tables)
    TS = _loader.timescale(builtin=True)

    _kernel_loaded = None
    _kernel_errors = []

    for name in _KernelCandidates if (_KernelCandidates := _KERNEL_CANDIDATES) else []:
        try:
            PLANETS = _loader(name)
            _kernel_loaded = name
            break
        except Exception as e:  # keep trying next candidate
            _kernel_errors.append(f"{name}: {e!r}")

    if PLANETS is None:  # no candidate succeeded
        raise RuntimeError(
            "Failed to load any Skyfield kernel. Tried: "
            + ", ".join(_KERNEL_CANDIDATES)
            + (f" | errors: { '; '.join(_kernel_errors) }" if _kernel_errors else "")
        )

except Exception as e:
    # Defer hard failure until first usage; provide clear guidance
    class _EphemMissing:
        def __getattr__(self, _name):
            raise RuntimeError(
                "Skyfield ephemeris not initialized. "
                "Cause: " + repr(e) + ". "
                "Ensure 'skyfield' is installed and kernel files are accessible. "
                f"EPHEMERIS_DIR={_EPHEM_DIR!r} EPHEMERIS_KERNEL={_KERNEL_ENV!r}"
            )

    TS = _EphemMissing()       # type: ignore
    PLANETS = _EphemMissing()  # type: ignore

def get_timescale():
    """Return process-wide Skyfield timescale singleton."""
    return TS

def get_planets():
    """Return process-wide loaded kernel (e.g., de440s/de421) singleton."""
    return PLANETS
# ------------------------------------------------------------------

__all__ = [
    # predictive exports
    "TransitEngine", "TransitEvent", "find_transits_in_range",
    "DashaPeriod", "vimsottari_dasha", "predict_dasha_periods",
    "compute_vargas_for_point", "compute_vargas",
    "detect_yogas", "house_index_for_longitude",
    "compute_houses", "timescales_from_civil",
    "EvalResult", "evaluate_univariate", "permutation_pvalue_corr",
    "bh_fdr", "holdout_replicate", "validate_predictions",
    "feature_transit_proximity", "feature_dasha_lords_onehot", "feature_yoga_flags",
    # ephemeris singletons + accessors
    "TS", "PLANETS", "get_timescale", "get_planets",
]
