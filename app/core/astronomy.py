# Add these helper functions after your existing _frame_kw function:

def _geo_kwargs_for_sig(sig: inspect.Signature, *, topocentric: bool, lat_q, lon_q, elev_q) -> Dict[str, Any]:
    """Map our geo/topo to whatever parameter names the adapter exposes."""
    kw: Dict[str, Any] = {}
    if "topocentric" in sig.parameters:
        kw["topocentric"] = topocentric
    if topocentric:
        if "latitude" in sig.parameters and lat_q is not None: kw["latitude"] = float(lat_q)
        if "longitude" in sig.parameters and lon_q is not None: kw["longitude"] = float(lon_q)
        if "elevation_m" in sig.parameters and elev_q is not None: kw["elevation_m"] = float(elev_q)
    return kw

def _normalize_adapter_output_to_maps(
    res: Any, names_key: Tuple[str, ...]
) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
    """
    Normalize various adapter return styles into:
      (longitudes_map {name: deg}, speeds_map {name: deg/day or None})
    Robust to adapters that require lower-case body names.
    """
    longitudes: Dict[str, float] = {}
    speeds: Dict[str, Optional[float]] = {}

    want = list(names_key)
    want_lc = [w.lower() for w in want]

    def _merge_numeric_map(src: Dict[Any, Any], *, into: str) -> None:
        src_lc = {str(k).lower(): v for k, v in src.items()}
        for nm, nm_lc in zip(want, want_lc):
            if nm_lc in src_lc and isinstance(src_lc[nm_lc], (int, float)):
                if into == "lon":
                    longitudes[nm] = float(src_lc[nm_lc])
                    speeds.setdefault(nm, None)
                else:
                    speeds[nm] = float(src_lc[nm_lc])

    # Case A: dict formats
    if isinstance(res, dict):
        # A1: {name: deg}
        if all(isinstance(k, (str, int)) and isinstance(v, (int, float)) for k, v in res.items()):
            _merge_numeric_map(res, into="lon")
            return longitudes, speeds
        # A2: {"longitudes": {...}, "velocities": {...}}
        for lon_key in ("longitudes", "longitude", "lon"):
            if lon_key in res and isinstance(res[lon_key], dict):
                _merge_numeric_map(res[lon_key], into="lon")
                break
        for spd_key in ("velocities", "velocity", "speeds", "speed"):
            if spd_key in res and isinstance(res[spd_key], dict):
                _merge_numeric_map(res[spd_key], into="spd")
                break
        if longitudes:
            return longitudes, speeds

    # Case B: list/tuple formats
    if isinstance(res, (list, tuple)):
        for i, nm in enumerate(want[:len(res)]):
            longitudes[nm] = float(res[i])
            speeds[nm] = None
        return longitudes, speeds

    # Anything else → unsupported
    raise AstronomyError("adapter_return_invalid", f"Unsupported adapter return type: {type(res)}")

# Replace your existing _cached_longitudes function with this:

@lru_cache(maxsize=8192)
def _cached_positions(
    jd_tt_q: float,
    names_key: Tuple[str, ...],
    topocentric: bool,
    lat_q: Optional[float],
    lon_q: Optional[float],
    elev_q: Optional[float],
) -> Tuple[Dict[str, float], Dict[str, Optional[float]], str]:
    """
    LRU-cached call to adapter.
    Returns (longitudes_map, speeds_map, source_tag).
    Robust to adapters that require lower-case body names: we try the
    original names first, then retry with lower-case on KeyError/ValueError.
    """
    if eph is None:
        raise AstronomyError("ephemeris_unavailable", f"ephemeris_adapter import failed: {_EPH_IMPORT_ERROR!r}")
    
    fn = _adapter_callable(
        "get_ecliptic_longitudes",
        "ecliptic_longitudes",
        "planetary_longitudes",
    )
    if fn is None:
        raise AstronomyError(
            "adapter_api_mismatch",
            "ephemeris_adapter missing any of: get_ecliptic_longitudes/ecliptic_longitudes/planetary_longitudes",
        )
    
    sig = inspect.signature(fn)
    # Base kwargs: time, frame, topo geometry
    base_kwargs: Dict[str, Any] = {}
    if "jd_tt" in sig.parameters:
        base_kwargs["jd_tt"] = jd_tt_q
    elif "jd" in sig.parameters:
        base_kwargs["jd"] = jd_tt_q
    if "frame" in sig.parameters:
        base_kwargs["frame"] = "ecliptic-of-date"
    
    base_kwargs.update(
        _geo_kwargs_for_sig(
            sig,
            topocentric=topocentric,
            lat_q=lat_q,
            lon_q=lon_q,
            elev_q=elev_q,
        )
    )
    
    # Try with original names, then with lower-case names (some adapters require this)
    attempts: List[Optional[List[str]]] = [None]
    if "names" in sig.parameters:
        attempts = [list(names_key), [n.lower() for n in names_key]]
    
    last_err: Optional[Exception] = None
    res: Any = None
    
    for name_list in attempts:
        kwargs = dict(base_kwargs)
        if name_list is not None:
            kwargs["names"] = name_list
        try:
            try:
                res = fn(**kwargs)
            except TypeError:
                # Fallback for odd signatures that only accept positional jd
                res = fn(jd_tt_q)
            break
        except (KeyError, ValueError) as e:
            # Likely a names-key/format issue; keep last_err and try next attempt
            last_err = e
            continue
        except Exception as e:
            # Unknown failure; if we have another attempt, try it; else bubble up
            last_err = e
            continue
    
    if res is None:
        raise AstronomyError(
            "adapter_failed",
            f"ephemeris adapter failed for names {list(names_key)}; last error: {last_err!r}",
        )
    
    lon_map, spd_map = _normalize_adapter_output_to_maps(res, names_key)
    return lon_map, spd_map, _adapter_source_tag()
