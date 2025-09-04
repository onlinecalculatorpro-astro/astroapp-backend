# app/version.py
from __future__ import annotations
import os

# Single place to bump the app version (overridable via env for CI/preview)
VERSION = os.getenv("ASTRO_VERSION", "0.1.0")
