# app/core/__init__.py
# Core exports + ephemeris singletons (re-exported)

from .predictive import (
    TransitEngine, TransitEvent, find_transits_in_range,
    DashaPeriod, vimsottari_dasha, predict_dasha_periods,
    compute_vargas_for_point, compute_vargas,
    detect_yogas, house_index_for_longitude,
    compute_houses, timescales_from_civil,
    EvalResult, evaluate_univariate, permutation_pvalue_corr, bh_fdr,
    holdout_replicate, validate_predictions,
    feature_transit_proximity, feature_dasha_lords_onehot, feature_yoga_flags,
)

# re-export singletons; no heavy work here → avoids circular init
from .ephem_singleton import TS, PLANETS, get_timescale, get_planets

__all__ = [
    "TransitEngine", "TransitEvent", "find_transits_in_range",
    "DashaPeriod", "vimsottari_dasha", "predict_dasha_periods",
    "compute_vargas_for_point", "compute_vargas",
    "detect_yogas", "house_index_for_longitude",
    "compute_houses", "timescales_from_civil",
    "EvalResult", "evaluate_univariate", "permutation_pvalue_corr",
    "bh_fdr", "holdout_replicate", "validate_predictions",
    "feature_transit_proximity", "feature_dasha_lords_onehot", "feature_yoga_flags",
    "TS", "PLANETS", "get_timescale", "get_planets",
]
