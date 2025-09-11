# app/core/predictive.py (shim)
from app.core.common_predictive import (
    PROF, norm360, wrap180, angdiff, sign_index, is_finite,
    compute_houses, timescales_from_civil,
    pearson_corr, permutation_pvalue_corr, bh_fdr,
    EvalResult, evaluate_univariate, holdout_replicate, validate_predictions
)

from app.core.western_predictive import (
    AspectSpec, AspectKind, MAJOR_ASPECTS, MINOR_ASPECTS,
    antiscia_longitude, contra_antiscia_longitude,
    TransitEvent, TransitEngine, find_transits_in_range
)

from app.core.vedic_predictive import (
    DashaPeriod, vimsottari_dasha, predict_dasha_periods,
    compute_vargas_for_point, compute_vargas,
    detect_yogas, house_index_for_longitude
)

__all__ = [
    # common
    "PROF","norm360","wrap180","angdiff","sign_index","is_finite",
    "compute_houses","timescales_from_civil","pearson_corr","permutation_pvalue_corr",
    "bh_fdr","EvalResult","evaluate_univariate","holdout_replicate","validate_predictions",
    # western
    "AspectSpec","AspectKind","MAJOR_ASPECTS","MINOR_ASPECTS","antiscia_longitude",
    "contra_antiscia_longitude","TransitEvent","TransitEngine","find_transits_in_range",
    # vedic
    "DashaPeriod","vimsottari_dasha","predict_dasha_periods","compute_vargas_for_point",
    "compute_vargas","detect_yogas","house_index_for_longitude",
]
