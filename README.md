# AstroApp — Backend Skeleton (v1)

This is a **runnable Flask API skeleton** that implements the contracts from your Architecture pack:
- `/api/health`
- `/api/calculate`
- `/api/report`
- `/predictions`
- `/rectification/quick`

> Astronomy and prediction logic are **deterministic placeholders** for now (no heavy ephemeris). They follow the **output schemas** so we can wire the UI and iterate safely on free tiers.

## Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run dev server
export FLASK_APP=app.main:app
flask run -p 5000
# or
python -m app.main

# run tests
pytest -q
```
## Deploy (Render free)
- Use `Procfile` (`web: gunicorn app.main:app`)
- Python version via `runtime.txt`
- Mount a small persistent disk if you want the SQLite cache
