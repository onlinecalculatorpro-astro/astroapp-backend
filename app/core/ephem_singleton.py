# app/core/ephem_singleton.py
import os
from skyfield.api import Loader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EPHEM_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "ephem"))
_EPHEM_DIR = os.environ.get("EPHEMERIS_DIR", _DEFAULT_EPHEM_DIR)
os.makedirs(_EPHEM_DIR, exist_ok=True)

_KERNEL = os.environ.get("EPHEMERIS_KERNEL", "de440s.bsp")
_CANDIDATES = [_KERNEL, "de440s.bsp", "de421.bsp"]

class _EphemMissing:
    def __init__(self, err: Exception):
        self._err = err
    def __getattr__(self, _name):
        raise RuntimeError(
            "Skyfield ephemeris not initialized. "
            f"Cause: {self._err!r}. "
            f"EPHEMERIS_DIR={_EPHEM_DIR!r} EPHEMERIS_KERNEL={_KERNEL!r}"
        )

try:
    _loader = Loader(_EPHEM_DIR)
    TS = _loader.timescale(builtin=True)

    PLANETS = None
    _errors = []
    for name in _CANDIDATES:
        try:
            PLANETS = _loader(name)
            break
        except Exception as e:
            _errors.append(f"{name}: {e!r}")

    if PLANETS is None:
        raise RuntimeError("Failed kernels → " + ", ".join(_errors))
except Exception as e:
    TS = _EphemMissing(e)        # type: ignore
    PLANETS = _EphemMissing(e)   # type: ignore

def get_timescale():
    return TS

def get_planets():
    return PLANETS
