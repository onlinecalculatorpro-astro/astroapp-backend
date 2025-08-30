#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${1:-http://localhost:5000}"

echo "Smoketest against $BASE_URL"

curl -fsS "$BASE_URL/api/health" | tee /dev/stderr | grep -q '"status":"ok"'
curl -fsS "$BASE_URL/api/config" > /dev/null
curl -fsS "$BASE_URL/metrics" | grep -q "astroapp_requests_total"
echo "Health/Config/Metrics OK"

# sample request
JSON='{"date":"1992-11-04","time":"05:25","place_tz":"Asia/Kolkata","latitude":13.0827,"longitude":80.2707,"mode":"sidereal","ayanamsa":"lahiri"}'
curl -fsS -H "Content-Type: application/json" -d "$JSON" "$BASE_URL/api/calculate" | grep -q '"chart"'
curl -fsS -H "Content-Type: application/json" -d "$JSON" "$BASE_URL/api/report" | grep -q '"narrative"'
JSON_PRED='{"date":"1992-11-04","time":"05:25","place_tz":"Asia/Kolkata","latitude":13.0827,"longitude":80.2707,"mode":"sidereal","ayanamsa":"lahiri","horizon":"short"}'
curl -fsS -H "Content-Type: application/json" -d "$JSON_PRED" "$BASE_URL/predictions" | grep -q '"predictions"'
curl -fsS -H "Content-Type: application/json" -d '{"window_minutes":90}' "$BASE_URL/rectification/quick" || true
echo "API endpoints OK"
