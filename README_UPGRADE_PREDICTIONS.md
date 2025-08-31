# AstroApp — Prediction Gold Standard v1.1

*Fixes syntax issues, adds pytest `conftest.py`, and optional Prometheus counters.*

## Register

```python
from app.api.predictions import predictions_bp
app.register_blueprint(predictions_bp, url_prefix="/api")
```

## Test locally

```bash
pip install pytest
pytest -q
```

## cURL

```bash
curl -sS -X POST http://localhost:8000/api/predictions       -H 'Content-Type: application/json'       -d '{
    "birth": {"date":"1990-05-21","time":"14:30","tz":"Asia/Kolkata","lat":12.9716,"lon":77.5946},
    "window": {"days_ahead": 30},
    "preferences": {"zodiac":"sidereal","ayanamsa":"lahiri","house_system":"placidus"},
    "topics": ["career","finance"],
    "max_events": 12
  }' | jq .
```
