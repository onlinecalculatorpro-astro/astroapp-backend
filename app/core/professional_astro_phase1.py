# app/core/professional_astro_phase1.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Optional
import logging

log = logging.getLogger(__name__)

# --- Base engine (v2) ---
try:
    from app.core.professional_astro_v2 import ProfessionalAstrologyEngine as BaseEngine
    from app.core.professional_astro_v2 import EngineConfig
except ImportError:  # very defensive fallback so imports never explode
    class BaseEngine:
        def __init__(self): ...
        def get_transit_predictions(self, *a, **k): return []
        def _angles_and_cusps(self, *a, **k): return {}, {}
        def _to_jd_tt(self, dt): return 2451545.0

    @dataclass(frozen=True)
    class EngineConfig:
        zodiac: str = "sidereal"
        ayanamsa: str = "lahiri"
        house_system: str = "placidus"

# --- Ephemeris adapter (outer planets included) ---
try:
    from app.core.ephemeris_adapter import ecliptic_longitudes as eph_lons
    from app.core.ephemeris_adapter import _skyfield_available
except Exception:
    eph_lons = None
    def _skyfield_available(): return False

# -----------------------------------
# Constants / helpers
# -----------------------------------
ALL_BODIES = {
    "Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn",
    "Uranus","Neptune","Pluto","Rahu","Ketu"
}
ALIASES = {
    "uranus": "Uranus",
    "neptune": "Neptune",
    "pluto": "Pluto",
    "true node": "Rahu",
    "mean node": "Rahu",
    "rahu": "Rahu",
    "ketu": "Ketu",
}

ASPECT_ANGLES: Dict[str, float] = {
    "conjunction": 0.0, "sextile": 60.0, "square": 90.0, "trine": 120.0, "opposition": 180.0,
    "semisextile": 30.0, "quintile": 72.0, "sesquiquadrate": 135.0, "biquintile": 144.0, "quincunx": 150.0,
}
DEFAULT_ORBS_MAJOR = {
    "conjunction": 5.0, "sextile": 4.0, "square": 4.0, "trine": 4.0, "opposition": 5.0,
}
DEFAULT_ORBS_MINOR = {
    "semisextile": 1.5, "quintile": 1.5, "sesquiquadrate": 2.0, "biquintile": 1.5, "quincunx": 2.0,
}
DEFAULT_ORBS = {**DEFAULT_ORBS_MAJOR, **DEFAULT_ORBS_MINOR}
OUTER_ORBS = {
    "Uranus":   {**DEFAULT_ORBS, "conjunction": 3.0, "square": 2.5, "trine": 2.5, "opposition": 3.0, "sextile": 2.0},
    "Neptune":  {**DEFAULT_ORBS, "conjunction": 2.5, "square": 2.0, "trine": 2.0, "opposition": 2.5, "sextile": 1.5},
    "Pluto":    {**DEFAULT_ORBS, "conjunction": 2.0, "square": 1.5, "trine": 1.5, "opposition": 2.0, "sextile": 1.0},
}

def _canon(name: str) -> str:
    n = (name or "").strip()
    lower = n.lower()
    if lower in ALIASES:
        return ALIASES[lower]
    return lower[:1].upper() + lower[1:] if n else n

def dt_utc(d: datetime) -> datetime:
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)

def angular_sep(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)

# ===================================================================
#   ProfessionalAstrologyEnginePhase1
# ===================================================================
class ProfessionalAstrologyEnginePhase1(BaseEngine):
    """
    Phase-1:
    - Adds Uranus, Neptune, Pluto via ephemeris_adapter
    - Extended aspects (30°, 72°, 135°, 144°, 150°)
    - Applying/separating via look-ahead sampling
    - Falls back to BaseEngine when ephemeris is unavailable
    """
    name: str = "ProfessionalAstrologyEnginePhase1"
    precision_mode: str = "MAXIMUM"

    def __init__(self):
        super().__init__()
        self._last_diag: Optional[Dict[str, Any]] = None

    # --------------- Public API ---------------
    def get_transit_predictions(
        self,
        birth_dt: datetime,
        lat: float,
        lon: float,
        window: Tuple[datetime, datetime],
        zodiac: str = "sidereal",
        ayanamsa: str = "lahiri",
        house_system: str = "placidus",
        topics: Optional[List[str]] = None,
        max_events: int = 20,
    ) -> List[Dict[str, Any]]:

        if not (_skyfield_available() and eph_lons):
            log.warning("Phase-1 outer-planet features require Skyfield ephemeris; falling back.")
            return super().get_transit_predictions(
                birth_dt, lat, lon, window, zodiac, ayanamsa, house_system, topics, max_events
            )

        cfg = EngineConfig(zodiac=zodiac, ayanamsa=ayanamsa, house_system=house_system)

        jd_birth_tt = self._to_jd_tt(dt_utc(birth_dt))
        natal = self._complete_natal_chart(jd_birth_tt, lat, lon, cfg)

        start_utc = dt_utc(window[0]); end_utc = dt_utc(window[1])

        # diagnostics probe (optional)
        try:
            self._last_diag = self._diag_probe(jd_birth_tt, self._to_jd_tt(start_utc), lat, lon)
        except Exception:
            self._last_diag = None

        events: List[Dict[str, Any]] = []
        events.extend(self._outer_planet_transits(natal, start_utc, end_utc, cfg, lat, lon))

        if topics:
            need = {t.lower() for t in topics}
            events = [e for e in events if e.get("topic","").lower() in need]

        events.sort(key=lambda e: (e.get("score", 0.0), e.get("date", "")), reverse=True)
        return events[:max_events] if max_events else events

    # --------------- Natal compilation ---------------
    def _complete_natal_chart(self, jd_tt: float, lat: float, lon: float, cfg: EngineConfig) -> Dict[str, Any]:
        bodies = eph_lons(jd_tt, lat, lon) if eph_lons else []
        planets: Dict[str, Dict[str, float]] = {}
        for b in bodies or []:
            name = _canon(b.get("name",""))
            if name in ALL_BODIES:
                lonf = float(b.get("lon", 0.0)) % 360.0
                planets[name] = {
                    "lon": lonf,
                    "lat": float(b.get("lat", 0.0)),
                    "speed": float(b.get("speed", 0.0)),
                    "sign": int(lonf // 30),
                    "degree": lonf % 30.0,
                }
        angles, cusps = self._angles_and_cusps_safe(jd_tt, lat, lon, cfg)
        return {"planets": planets, "angles": angles, "cusps": cusps, "jd_tt": jd_tt}

    # --------------- Transits ---------------
    def _outer_planet_transits(
        self, natal: Dict[str, Any], start_dt: datetime, end_dt: datetime, cfg: EngineConfig, lat: float, lon: float
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        natal_planets: Dict[str, Dict[str, float]] = natal.get("planets", {})
        natal_angles: Dict[str, float] = natal.get("angles", {})

        if not natal_planets:
            return out

        step = timedelta(days=7)  # weekly sampling fits slow movers
        t = start_dt
        while t <= end_dt:
            jd_now  = self._to_jd_tt(t)
            jd_prev = self._to_jd_tt(t - step)
            jd_next = self._to_jd_tt(t + step)

            tr_now  = self._planet_lons_extended(jd_now,  lat, lon)
            tr_prev = self._planet_lons_extended(jd_prev, lat, lon)
            tr_next = self._planet_lons_extended(jd_next, lat, lon)

            for outer in ("Uranus","Neptune","Pluto"):
                if outer not in tr_now:
                    continue
                t_lon_now  = tr_now[outer]["lon"]
                t_lon_prev = tr_prev.get(outer, {}).get("lon", t_lon_now)
                t_lon_next = tr_next.get(outer, {}).get("lon", t_lon_now)

                # (A) to natal planets
                for n_name, n_data in natal_planets.items():
                    n_lon = n_data["lon"]
                    asp, orb = self._find_aspect_with_custom_orb(t_lon_now, n_lon, OUTER_ORBS[outer])
                    if asp:
                        dirn = self._trend_direction(t_lon_prev, t_lon_now, t_lon_next, n_lon)
                        strength = self._outer_planet_strength(outer, n_name, asp, orb)
                        if strength >= 0.30:
                            out.append({
                                "topic": self._topic_for_outer(outer, n_name, asp),  # stable helper
                                "date": t.date().isoformat(),
                                "score": round(strength, 3),
                                "narrative": f"{outer} {asp} {n_name} (orb {orb:.2f}°): {self._outer_planet_meaning(outer, n_name, asp)}",
                                "signals": [{
                                    "aspect": asp, "bodies": [outer, n_name],
                                    "orb": round(orb, 2), "strength": strength,
                                    "direction": dirn, "technique": "outer_planet_transit",
                                }],
                            })

                # (B) to angles (if present)
                for angle_name, angle_lon in natal_angles.items():
                    if angle_lon is None:
                        continue
                    asp, orb = self._find_aspect_with_custom_orb(t_lon_now, angle_lon, OUTER_ORBS[outer])
                    if asp:
                        dirn = self._trend_direction(t_lon_prev, t_lon_now, t_lon_next, angle_lon)
                        strength = self._outer_planet_strength(outer, angle_name, asp, orb)
                        if strength >= 0.30:
                            out.append({
                                "topic": self._outer_planet_angle_topic(outer, angle_name, asp),
                                "date": t.date().isoformat(),
                                "score": round(strength, 3),
                                "narrative": f"{outer} {asp} {angle_name} (orb {orb:.2f}°): Major life shift.",
                                "signals": [{
                                    "aspect": asp, "bodies": [outer, angle_name],
                                    "orb": round(orb, 2), "strength": strength,
                                    "direction": dirn, "technique": "outer_planet_transit",
                                }],
                            })
            t += step
        return out

    # --------------- Topic / meaning helpers ---------------
    def _topic_for_outer(self, outer_planet: str, natal_body: str, aspect: str) -> str:
        """Stable helper for topic mapping (used internally)."""
        if outer_planet == "Uranus":
            if natal_body in {"MC","Sun"}: return "career"
            if natal_body == "Venus": return "relationships"
            if natal_body == "Moon": return "relocation"
            return "innovation"
        if outer_planet == "Neptune":
            if natal_body in {"MC","Sun"}: return "career"
            if natal_body in {"Venus","Moon"}: return "relationships"
            if natal_body == "Mercury": return "creativity"
            return "spirituality"
        if outer_planet == "Pluto":
            if natal_body in {"MC","Sun"}: return "career"
            if natal_body == "Mars": return "health"
            if natal_body in {"Venus","Moon"}: return "relationships"
            return "transformation"
        return "general"

    # Back-compat alias so any stale import calling the old name doesn't crash:
    def _outer_planet_topic_mapping(self, outer_planet: str, natal_body: str, aspect: str) -> str:
        return self._topic_for_outer(outer_planet, natal_body, aspect)

    def _outer_planet_angle_topic(self, outer: str, angle: str, aspect: str) -> str:
        return "major_life_changes"

    def _outer_planet_meaning(self, outer: str, natal_body: str, aspect: str) -> str:
        base = {
            "Uranus": "breakthroughs, surprises, new directions",
            "Neptune": "visions, ideals, dissolving of old forms",
            "Pluto": "deep transformation, power shifts, renewal",
        }.get(outer, "significant developments")
        return base

    # --------------- Math / helpers ---------------
    def _planet_lons_extended(self, jd_tt: float, lat: float, lon: float) -> Dict[str, Dict[str, float]]:
        if not eph_lons:
            return {}
        try:
            out: Dict[str, Dict[str, float]] = {}
            for b in eph_lons(jd_tt, lat, lon) or []:
                name = _canon(b.get("name",""))
                if name in ALL_BODIES:
                    out[name] = {
                        "lon": float(b.get("lon", 0.0)) % 360.0,
                        "lat": float(b.get("lat", 0.0)),
                        "speed": float(b.get("speed", 0.0)),
                    }
            return out
        except Exception as e:
            log.warning("eph_lons failed: %s", e)
            return {}

    def _find_aspect_with_custom_orb(self, lon1: float, lon2: float, orb_table: Dict[str, float]) -> Tuple[Optional[str], float]:
        sep = angular_sep(lon1, lon2)
        best_name: Optional[str] = None
        best_orb: float = 999.0
        table = {**DEFAULT_ORBS, **(orb_table or {})}
        for name, angle in ASPECT_ANGLES.items():
            orb = abs(sep - angle)
            max_orb = table.get(name, 0.0)
            if orb <= max_orb and orb < best_orb:
                best_name, best_orb = name, orb
        return (best_name, best_orb) if best_name else (None, 0.0)

    def _outer_planet_strength(self, outer: str, natal_body: str, aspect: str, orb: float) -> float:
        base = {
            "conjunction": 1.00, "opposition": 0.90, "square": 0.80,
            "trine": 0.70, "sextile": 0.60,
            "semisextile": 0.45, "quintile": 0.50, "sesquiquadrate": 0.55, "biquintile": 0.50, "quincunx": 0.55,
        }.get(aspect, 0.50)
        max_orb = OUTER_ORBS.get(outer, DEFAULT_ORBS).get(aspect, 1.5)
        orb_factor = max(0.0, 1.0 - (orb / max_orb))  # tighter = stronger
        target_factor = 1.20 if natal_body in {"Sun","Moon","Asc","MC"} else 1.00
        return base * orb_factor * target_factor

    def _trend_direction(self, lon_prev: float, lon_now: float, lon_next: float, target_lon: float) -> str:
        sep_prev = angular_sep(lon_prev, target_lon)
        sep_now  = angular_sep(lon_now,  target_lon)
        sep_next = angular_sep(lon_next, target_lon)
        if sep_next < sep_now:  return "applying"
        if sep_prev < sep_now:  return "separating"
        return "culminating" if sep_now <= min(sep_prev, sep_next) else "stationary"

    # --------------- Angles / JD fallbacks ---------------
    def _angles_and_cusps_safe(self, jd_tt: float, lat: float, lon: float, cfg: EngineConfig) -> Tuple[Dict[str, float], Dict[int, float]]:
        try:
            angles, cusps = super()._angles_and_cusps(jd_tt, lat, lon, cfg)  # type: ignore
            return angles or {}, cusps or {}
        except Exception:
            return {}, {}

    def _to_jd_tt(self, dt: datetime) -> float:
        try:
            return super()._to_jd_tt(dt)  # type: ignore
        except Exception:
            dt = dt_utc(dt)
            y, m = dt.year, dt.month
            D = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond/1e6)/60.0)/60.0)/24.0
            if m <= 2:
                y -= 1; m += 12
            A = y // 100
            B = 2 - A + (A // 25)
            jd_ut = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + D + B - 1524.5
            return jd_ut + (69.0 / 86400.0)  # rough ΔT → TT

    # --- diagnostics (optional) ---
    def _diag_probe(self, birth_jd_tt: float, start_jd_tt: float, lat: float, lon: float) -> Dict[str, Any]:
        birth = self._planet_lons_extended(birth_jd_tt, lat, lon)
        start = self._planet_lons_extended(start_jd_tt, lat, lon)
        return {
            "has_eph": bool(eph_lons),
            "skyfield_available": bool(_skyfield_available()),
            "birth_keys": sorted(birth.keys()),
            "start_keys": sorted(start.keys()),
            "has_outers_birth": any(k in birth for k in ("Uranus","Neptune","Pluto")),
            "has_outers_start": any(k in start for k in ("Uranus","Neptune","Pluto")),
        }

__all__ = ["ProfessionalAstrologyEnginePhase1"]
