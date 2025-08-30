# AstroApp — Deployment Guide (Render + Netlify)

This guide deploys the **backend** to **Render** and the **frontend** to **Netlify**.
The artifacts were generated on 2025-08-29.

---

## 1) Backend (Render)

**Files**: contents of `AstroApp_backend_production_v5.zip`

**Start command**
```
gunicorn app.main:app
```

**Environment Variables**
```
ASTRO_CONFIG=config/defaults.yaml
ASTRO_CALIBRATORS=config/calibrators.json
ASTRO_HC_THRESHOLDS=config/hc_thresholds.json
```

**Health checks**
- `GET /api/health` → `{"status":"ok"}`
- `GET /api/openapi` → OpenAPI JSON (if bundled)
- `GET /metrics` → Prometheus text (requests + avg/p95 latency)
- `GET /api/config` → sanitized runtime config (mode, ayanamsa, limits, versions)

**Optional (Skyfield)**
If `skyfield` is installed, first boot may download `de421.bsp`. Allow egress or pre-provision the file in the working directory.

**CORS**
Enabled for all origins by default (free-tier demo). Lock this down later if needed.

---

## 2) Frontend (Netlify)

**Files**: contents of `AstroApp_frontend_scaffold_v2.zip`

**Build**
```bash
npm install
npm run build
```

**Environment**
Create `.env` (or Netlify environment var) to point to the backend:
```
VITE_API_BASE=https://<your-render-service>.onrender.com
```

**Publish Directory**
```
dist/
```

**SPA Routing**
A `netlify.toml` is included with a catch-all redirect to `index.html`.

---

## 3) Local smoke test

**Backend**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
# curl http://localhost:5000/api/health
# curl http://localhost:5000/api/config
```

**Frontend**
```bash
npm install
echo "VITE_API_BASE=http://localhost:5000" > .env
npm run dev
# open the printed URL, go to Chart → Results
```

---

## 4) Endpoints Matrix

- `GET /api/health` – health
- `GET /api/openapi` – OpenAPI JSON
- `GET /metrics` – Prometheus
- `GET /system-validation` – SLOs/mode/ayanamsa version
- `GET /api/config` – sanitized runtime config
- `POST /api/calculate` – chart+houses
- `POST /api/report` – chart+houses+narrative
- `POST /predictions` – QIA + Calibrator + HC thresholds + abstention
- `POST /rectification/quick` – quick birth time scan

---

## 5) Notes & Next

- This is a **free-tier** build. Heavy compute paths are optimized to degrade gracefully when ephemeris isn't available.
- When you’re ready, swap in the full astronomy stack and real calibration files without changing API contracts.
