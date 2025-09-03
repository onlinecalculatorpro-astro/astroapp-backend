# tests/conftest.py
from __future__ import annotations

"""
Pytest configuration for the AstroApp timescales suite.

- Registers Hypothesis profiles for local dev and CI.
- Freezes the process TZ to UTC (defensive; we always pass IANA zones explicitly).
- Sanity-checks ERFA availability and basic tzdata presence.
- Adds a 'slow' marker (not used by default, but handy if you add heavier tests).
"""

import os
import pytest
from hypothesis import settings, HealthCheck


# ──────────────────────────────────────────────────────────────────────────────
# Hypothesis profiles
# ──────────────────────────────────────────────────────────────────────────────
settings.register_profile(
    "dev",
    settings(
        deadline=None,           # avoid flaky timeouts on slower runners
        max_examples=60,         # fast local runs
        suppress_health_check=[HealthCheck.too_slow],
    ),
)
settings.register_profile(
    "ci",
    settings(
        deadline=None,
        max_examples=120,        # a bit more coverage in CI
        suppress_health_check=[HealthCheck.too_slow],
    ),
)

_profile = (
    "ci"
    if (os.getenv("CI") or os.getenv("GITHUB_ACTIONS"))
    else os.getenv("HYPOTHESIS_PROFILE", "dev")
)
settings.load_profile(_profile)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")


def pytest_report_header(config: pytest.Config) -> str:
    return f"Hypothesis profile: '{_profile}'"


# ──────────────────────────────────────────────────────────────────────────────
# Global fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def freeze_tz_env():
    """
    Ensure the process TZ is UTC so any library that *might* consult TZ
    (even though we pass IANA zones explicitly) is deterministic.
    """
    prev = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = prev


@pytest.fixture(scope="session")
def ensure_erfa():
    """
    Fail early if ERFA/pyERFA isn't importable or missing key functions.
    """
    import erfa  # pyERFA exposes the ERFA namespace as 'erfa'
    assert hasattr(erfa, "dtf2d"), "ERFA.dtf2d not available"
    assert hasattr(erfa, "utctai"), "ERFA.utctai not available"
    assert hasattr(erfa, "taitt"), "ERFA.taitt not available"
    assert hasattr(erfa, "utcut1"), "ERFA.utcut1 not available"
    return erfa


@pytest.fixture(scope="session")
def ensure_tzdata():
    """
    Sanity-check that core IANA zones resolve on this machine.
    If tzdata is missing on a CI runner, you can add 'tzdata' to requirements.
    """
    from zoneinfo import ZoneInfo
    for name in ("UTC", "Asia/Kolkata", "America/New_York"):
        ZoneInfo(name)
