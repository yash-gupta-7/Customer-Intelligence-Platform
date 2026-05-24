#!/bin/sh
set -e
WORKERS="${UVICORN_WORKERS:-1}"
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers "$WORKERS" \
  --loop uvloop \
  --log-level info \
  --access-log
