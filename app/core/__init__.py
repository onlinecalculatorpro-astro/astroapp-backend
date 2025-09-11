# app/core/__init__.py
# -------------------------------------------------------------------
# Lightweight package init. No heavy imports here (avoids circulars).
# Ephemeris singletons live in ephem_singleton and can be imported
# from this package for convenience.
# -------------------------------------------------------------------

from .ephem_singleton import TS, PLANETS, get_timescale, get_planets

__all__ = [
    # Ephemeris singletons + accessors
    "TS", "PLANETS", "get_timescale", "get_planets",
]

# Notes for importers:
# - Import predictive symbols directly from their modules, e.g.:
#     from app.core.predictive import TransitEngine, find_transits_in_range
# - Import validators directly:
#     from app.core.validators import resolve_timescales_from_civil_erfa
# - Do NOT import predictive from app.core (this file) to avoid
#   package import side-effects and circular imports.
