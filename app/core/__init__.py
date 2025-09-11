# app/core/__init__.py
# -*- coding: utf-8 -*-
"""
AstroApp core package — lightweight __init__ (no heavy side effects).

This file intentionally avoids importing runtime-heavy modules (ephemerides,
timescales, prediction engines, etc.) to prevent circular imports and slow
startup. Import those from their concrete modules:

- Ephemeris singleton (TS, PLANETS):   app.core.runtime
- Constants & helpers:                  app.core.constants
- Validators / parsers:                 app.core.validators
- Timescales utilities:                 app.core.timescales
"""

from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = [
    # Core public modules (import these directly in your code)
    "constants",
    "validators",
    "timescales",
]

if TYPE_CHECKING:
    # Type-only hints to keep editors happy without importing at runtime
    from . import constants as constants
    from . import validators as validators
    from . import timescales as timescales
