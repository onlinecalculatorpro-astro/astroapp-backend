# app/core/horary.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

__all__ = ["QuestionType", "HoraryInput", "analyze_prasna_enhanced"]

class QuestionType(Enum):
    JOB = "job"
    RELATIONSHIP = "relationship"
    GENERIC = "generic"
    MONEY = "money"
    HEALTH = "health"

@dataclass
class HoraryInput:
    # Required civil + site
    date: str
    time: str
    tz_name: str
    latitude: float
    longitude: float
    # Optional
    place: Optional[str] = None
    zodiac_mode: str = "sidereal"
    ayanamsa: object = "lahiri"
    ayanamsa_deg: Optional[float] = None
    house_system: str = "sripati"
    # KP options
    kp_house_system: str = "placidus"
    kp_ayanamsa: str = "krishnamurti"
    kp_number: Optional[int] = None
    kp_number_mode: str = "anchor_asc"
    # Question context
    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None
    querent_house: int = 1
    quesited_house: Optional[int] = None

def analyze_prasna_enhanced(inp: HoraryInput, method: str = "parashari") -> Dict[str, Any]:
    """
    Minimal stub so API routes work. Replace with real computation later.
    Returns a deterministic structure the routes/tests can inspect.
    """
    method = (method or "parashari").lower()
    if method not in ("parashari", "kp"):
        method = "parashari"

    topic = getattr(inp.question_type, "value", None) if isinstance(inp.question_type, QuestionType) else None
    topic = topic or "generic"

    judgment = f"stub_{method}_analysis_for_{topic}"

    res = {
        "ok": True,
        "method": method,
        "judgment": judgment,
        "context": {
            "question_type": topic,
            "question_text": inp.question_text,
            "querent_house": inp.querent_house,
            "quesited_house": inp.quesited_house,
            "kp_number_used": inp.kp_number if method == "kp" else None,
            "house_system": inp.house_system if method == "parashari" else inp.kp_house_system,
        },
        "chart": {
            "datetime": f"{inp.date}T{inp.time} {inp.tz_name}",
            "site": {"lat": float(inp.latitude), "lon": float(inp.longitude), "place": inp.place},
            "zodiac_mode": inp.zodiac_mode,
            "ayanamsa": inp.ayanamsa,
            "ayanamsa_deg": inp.ayanamsa_deg,
        },
        "meta": {"engine": "horary_stub", "version": 1},
    }
    return res
