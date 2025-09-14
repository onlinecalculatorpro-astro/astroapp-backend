def _moon_nirayana_deg_at(jd_tt: float, *, ayanamsa_key: str) -> float:
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")

    # ── robust adapter construction across versions ──
    def _mk_ephem():
        # Prefer passing TS to the adapter, not into Config (some builds don't accept 'timescale')
        try:
            cfg = EphemConfig(frame="ecliptic-of-date", planets=PLANETS)  # type: ignore
        except TypeError:
            cfg = EphemConfig(frame="ecliptic-of-date")  # type: ignore
        # Try adapter(timescale=TS), adapter(TS), and last-resort: Config(timescale=TS)
        try:
            return EphemerisAdapter(cfg, timescale=TS)  # type: ignore
        except TypeError:
            try:
                return EphemerisAdapter(cfg, TS)  # type: ignore
            except TypeError:
                try:
                    cfg2 = EphemConfig(frame="ecliptic-of-date", planets=PLANETS, timescale=TS)  # type: ignore
                    return EphemerisAdapter(cfg2)  # type: ignore
                except TypeError:
                    return EphemerisAdapter(cfg)  # type: ignore

    ephem = _mk_ephem()
    rows = ephem.ecliptic_longitudes(float(jd_tt), ["Moon"]).get("results", [])
    if not rows:
        raise RuntimeError("ephemeris returned no Moon longitude")
    moon_trop = float(rows[0]["longitude"])
    ay = float(get_ayanamsa_deg(float(jd_tt), ayanamsa_key))
    return _norm360(moon_trop - ay)
