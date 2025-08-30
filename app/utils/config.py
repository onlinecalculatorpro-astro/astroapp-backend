from __future__ import annotations
import os, yaml
from dataclasses import dataclass

@dataclass
class AppConfig:
    mode: str = "sidereal"
    ayanamsa: str = "lahiri"
    ephemeris_kernel: str = "de421"
    rectification_mode: str = "quick"
    rect_window_minutes: int = 90
    rect_step_seconds: int = 120
    cache_lru_mb: int = 64
    cache_sqlite_mb: int = 100
    ttl_houses_days: int = 7
    ttl_rect_days: int = 2
    pro_features_enabled: bool = False
    rate_limits_per_hour: dict | None = None

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rl = data.get('rate_limits', {})
cfg = AppConfig(
        mode=data.get("mode","sidereal"),
        ayanamsa=data.get("ayanamsa","lahiri"),
        ephemeris_kernel=data.get("ephemeris_kernel","de421"),
        rectification_mode=data.get("rectification_mode","quick"),
        rect_window_minutes=int(data.get("rect_window_minutes",90)),
        rect_step_seconds=int(data.get("rect_step_seconds",120)),
        cache_lru_mb=int(data.get("cache_sizes_mb",{}).get("lru",64)),
        cache_sqlite_mb=int(data.get("cache_sizes_mb",{}).get("sqlite",100)),
        ttl_houses_days=int(data.get("ttl_days",{}).get("houses",7)),
        ttl_rect_days=int(data.get("ttl_days",{}).get("rectification_windows",2)),
        pro_features_enabled=bool(data.get("pro_features_enabled", False)),
        rate_limits_per_hour=rl if isinstance(rl, dict) else {},
    )
    # simple env overrides
    cfg.mode = os.getenv("ASTRO_MODE", cfg.mode)
    cfg.ayanamsa = os.getenv("ASTRO_AYANAMSA", cfg.ayanamsa)
    return cfg
