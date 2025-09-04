# tests/test_astronomy.py
from __future__ import annotations

import math
import sys
import types
import importlib
import os

import pytest


# ---------- Helpers to build a fake ephemeris_adapter BEFORE importing astronomy ----------
MAJORS = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
)

BASE_LON = {  # arbitrary deterministic base angles per major
    "Sun": 10.0, "Moon": 20.0, "Mercury": 30.0, "Venus": 40.0, "Mars": 50.0,
    "Jupiter": 60.0, "Saturn": 70.0, "Uranus": 80.0, "Neptune": 90.0, "Pluto": 100.0,
}
BASE_SPD = {  # constant “speeds” (deg/day) to keep tests simple
    "Sun": 0.9856, "Moon": 13.1764, "Mercury": 1.2, "Venus": 1.0, "Mars": 0.5,
    "Jupiter": 0.08, "Saturn": 0.03, "Uranus": 0.01, "Neptune": 0.006, "Pluto": 0.004,
}

# We also support nodes so astronomy can request them as "points".
NODE_LON = {"North Node": 125.0, "South Node": 305.0}

_last_calls = {
    "topocentric": None,
    "latitude": None,
    "longitude": None,
    "elevation_m": None,
}


def _wrap360(x: float) -> float:
    v = float(x) % 360.0
    return 0.0 if abs(v) < 1e-12 else v


def fake_ecliptic_longitudes_and_velocities(
    jd_tt: float,
    names,
    *,
    frame: str = "ecliptic-of-date",
    topocentric: bool = False,
    latitude=None,
    longitude=None,
    elevation_m=None,
):
    # Record last call for assertions in tests
    _last_calls.update(
        {
            "topocentric": bool(topocentric),
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": elevation_m,
        }
    )

    rows = []
    for nm in names:
        if nm in MAJORS:
            lon = _wrap360(BASE_LON[nm] + (jd_tt % 360.0))
            spd = BASE_SPD[nm]
            rows.append(
                {
                    "name": nm,
                    "longitude": lon,
                    "lon": lon,
                    "velocity": spd,
                    "speed": spd,
                    "lat": 0.0,
                }
            )
        elif nm in NODE_LON:
            # Provide nodes too so astronomy can fetch geocentrically
            lon = NODE_LON[nm]
            rows.append({"name": nm, "longitude": lon, "lon": lon, "lat": 0.0})
        else:
            # Unknown body → simply skip (astronomy will warn)
            pass
    return rows


def fake_ecliptic_longitudes(
    jd_tt: float,
    names=None,
    *,
    frame: str = "ecliptic-of-date",
    topocentric: bool = False,
    latitude=None,
    longitude=None,
    elevation_m=None,
):
    # Reuse the same backing function
    if names is None:
        names = list(MAJORS)
    return fake_ecliptic_longitudes_and_velocities(
        jd_tt,
        names,
        frame=frame,
        topocentric=topocentric,
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
    )


def install_fake_adapter():
    """Inject a fake `app.core.ephemeris_adapter` into sys.modules."""
    mod = types.ModuleType("app.core.ephemeris_adapter")
    mod.ecliptic_longitudes_and_velocities = fake_ecliptic_longitudes_and_velocities
    mod.ecliptic_longitudes = fake_ecliptic_longitudes
    mod.current_kernel_name = lambda: "de421(fake)"
    # Optional helpers used by astronomy for diagnostics
    mod.rows_to_maps = lambda rows: {
        "longitudes": {r["name"]: r.get("longitude", r.get("lon")) for r in rows},
        "velocities": {r["name"]: r.get("velocity", r.get("speed")) for r in rows if "velocity" in r or "speed" in r},
    }
    sys.modules["app.core.ephemeris_adapter"] = mod
    # Also ensure package parents exist to avoid import errors
    if "app" not in sys.modules:
        sys.modules["app"] = types.ModuleType("app")
    if "app.core" not in sys.modules:
        sys.modules["app.core"] = types.ModuleType("app.core")


def import_astronomy_fresh():
    """Import the astronomy module after installing the fake adapter."""
    install_fake_adapter()
    # Ensure a clean import (drop any prior cached module)
    sys.modules.pop("app.core.astronomy", None)
    return importlib.import_module("app.core.astronomy")


# ---------- Common constants for tests ----------
JD_TT = 2451545.0  # J2000 TT
# Provide jd_ut and jd_ut1 to bypass timescale resolution (no ERFA needed)
JD_UT = JD_TT - (69.0 / 86400.0)          # rough ΔT fallback used in astronomy
JD_UT1 = JD_UT + (0.1 / 86400.0)          # emulate small DUT1 = +0.1s


def _shortest_delta(a2, a1):
    d = (float(a2) - float(a1) + 540.0) % 360.0 - 180.0
    return -180.0 if d == 180.0 else d


# ============================= Tests =============================

def test_majors_only_basic_longitudes_and_speeds():
    astronomy = import_astronomy_fresh()

    payload = {
        "mode": "tropical",
        "jd_ut": JD_UT,
        "jd_tt": JD_TT,
        "jd_ut1": JD_UT1,
        "bodies": ["Sun", "Moon", "Mars"],
        # geocentric (no topocentric)
    }
    out = astronomy.compute_chart(payload)

    assert out["mode"] == "tropical"
    names = [b["name"] for b in out["bodies"]]
    assert names == ["Sun", "Moon", "Mars"]

    for row in out["bodies"]:
        lon = float(row["longitude_deg"])
        assert 0.0 <= lon < 360.0
        # speeds should be present for majors from our fake adapter
        assert row["speed_deg_per_day"] is not None
        assert math.isfinite(float(row["speed_deg_per_day"]))


def test_nodes_in_bodies_are_moved_to_points_and_default_majors_returned():
    astronomy = import_astronomy_fresh()

    payload = {
        "mode": "tropical",
        "jd_ut": JD_UT,
        "jd_tt": JD_TT,
        "jd_ut1": JD_UT1,
        "bodies": ["North Node", "South Node"],  # will be moved to points
    }
    out = astronomy.compute_chart(payload)

    # Bodies list should fall back to default 10 majors (because caller explicitly provided
    # bodies but they were all nodes).
    assert len(out["bodies"]) == 10
    # Points should include both nodes
    point_names = [p["name"] for p in out.get("points", [])]
    assert "North Node" in point_names and "South Node" in point_names


def test_sidereal_rotation_applies_to_bodies_points_and_angles():
    astronomy = import_astronomy_fresh()

    AY = 15.0  # explicit ayanamsa: subtract 15°
    payload_tropical = {
        "mode": "tropical",
        "jd_ut": JD_UT,
        "jd_tt": JD_TT,
        "jd_ut1": JD_UT1,
        "bodies": ["Sun", "Mars"],
        "points": ["North Node"],
        "latitude": 12.0,
        "longitude": 77.5,
    }
    out_trop = astronomy.compute_chart(payload_tropical)

    payload_sidereal = dict(payload_tropical)
    payload_sidereal["mode"] = "sidereal"
    payload_sidereal["ayanamsa"] = AY
    out_sid = astronomy.compute_chart(payload_sidereal)

    # Bodies longitudes should be shifted by -AY (mod 360)
    trop_lons = {b["name"]: b["longitude_deg"] for b in out_trop["bodies"]}
    sid_lons = {b["name"]: b["longitude_deg"] for b in out_sid["bodies"]}
    for nm in ("Sun", "Mars"):
        d = _shortest_delta(sid_lons[nm], trop_lons[nm])
        assert pytest.approx(d, abs=1e-6) == -AY or pytest.approx(d + 360.0, abs=1e-6) == (360.0 - AY)

    # Points (nodes) should be shifted by -AY too
    tnode = {p["name"]: p["longitude_deg"] for p in out_trop.get("points", [])}
    snode = {p["name"]: p["longitude_deg"] for p in out_sid.get("points", [])}
    d_node = _shortest_delta(snode["North Node"], tnode["North Node"])
    assert pytest.approx(d_node, abs=1e-6) == -AY or pytest.approx(d_node + 360.0, abs=1e-6) == (360.0 - AY)

    # Angles should also be rotated by -AY
    d_asc = _shortest_delta(out_sid["angles"]["asc_deg"], out_trop["angles"]["asc_deg"])
    d_mc = _shortest_delta(out_sid["angles"]["mc_deg"], out_trop["angles"]["mc_deg"])
    assert (pytest.approx(d_asc, abs=1e-6) == -AY or pytest.approx(d_asc + 360.0, abs=1e-6) == (360.0 - AY))
    assert (pytest.approx(d_mc, abs=1e-6) == -AY or pytest.approx(d_mc + 360.0, abs=1e-6) == (360.0 - AY))


def test_topocentric_flow_and_downgrade_near_pole_triggers_warning_and_geocentric_call():
    astronomy = import_astronomy_fresh()

    # Lat very near the hard pole threshold (>= 89.9 by default) triggers downgrade to geocentric
    payload = {
        "mode": "tropical",
        "jd_ut": JD_UT,
        "jd_tt": JD_TT,
        "jd_ut1": JD_UT1,
        "bodies": ["Sun"],
        "topocentric": True,
        "latitude": 90.0,    # beyond hard limit
        "longitude": 0.0,
        "elevation_m": 100.0,
    }
    out = astronomy.compute_chart(payload)

    # Assert warnings mention topocentric downgrade and missing angles geography (since lat/lon removed)
    warnings = out.get("warnings", []) or out["meta"].get("warnings", [])
    joined = " ".join(warnings)
    assert "topocentric_disabled_near_pole(hard)" in joined

    # Our fake adapter records last call — should have been downgraded to geocentric (topocentric=False)
    assert _last_calls["topocentric"] is False

    # Angles may be None due to missing geography after downgrade
    # (astronomy sets latitude/longitude to None when downgrading)
    assert out["angles"]["asc_deg"] is None or out["angles"]["mc_deg"] is None


def test_angles_are_in_0_360_and_present_when_regular_geo_given():
    astronomy = import_astronomy_fresh()

    payload = {
        "mode": "tropical",
        "jd_ut": JD_UT,
        "jd_tt": JD_TT,
        "jd_ut1": JD_UT1,
        "bodies": ["Sun"],
        "latitude": 12.0,
        "longitude": 77.5,
    }
    out = astronomy.compute_chart(payload)

    asc = out["angles"]["asc_deg"]
    mc = out["angles"]["mc_deg"]
    assert asc is not None and mc is not None
    assert 0.0 <= float(asc) < 360.0
    assert 0.0 <= float(mc) < 360.0


def test_output_schema_contains_expected_keys_and_meta():
    astronomy = import_astronomy_fresh()

    payload = {
        "mode": "tropical",
        "jd_ut": JD_UT,
        "jd_tt": JD_TT,
        "jd_ut1": JD_UT1,
        "bodies": ["Sun", "Moon"],
    }
    out = astronomy.compute_chart(payload)

    # top-level keys
    for key in ("mode", "jd_ut", "jd_tt", "jd_ut1", "bodies", "points", "angles", "meta"):
        assert key in out

    # body row shape
    br = out["bodies"][0]
    assert {"name", "longitude_deg", "lon", "speed_deg_per_day", "speed"}.issubset(br.keys())

    # meta shape
    meta = out["meta"]
    assert meta.get("frame") == "ecliptic-of-date"
    assert "source" in meta
    assert meta.get("module") == "astronomy(core)"
