# app/core/__init__.py

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

# --- Ephemeris preload (singleton) -------------------------------
import os
from skyfield.api import Loader

_EPHEM_DIR = os.path.join(os.path.dirname(__file__), "..", "ephem")
_loader = Loader(_EPHEM_DIR)

# These are singletons – import them elsewhere instead of calling Loader() per request
TS = _loader.timescale(builtin=True)
PLANETS = _loader("de440s.bsp")  # switch to "de421.bsp" for lighter, faster load if acceptable

def get_timescale():
    return TS

def get_planets():
    return PLANETS
# -----------------------------------------------------------------

__all__ = [
    "TransitEngine", "TransitEvent", "find_transits_in_range",
    "DashaPeriod", "vimsottari_dasha", "predict_dasha_periods",
    "compute_vargas_for_point", "compute_vargas",
    "detect_yogas", "house_index_for_longitude",
    "compute_houses", "timescales_from_civil",
    "EvalResult", "evaluate_univariate", "permutation_pvalue_corr",
    "bh_fdr", "holdout_replicate", "validate_predictions",
    "feature_transit_proximity", "feature_dasha_lords_onehot", "feature_yoga_flags",
    # expose singletons
    "TS", "PLANETS", "get_timescale", "get_planets",
]
