#!/bin/sh
set -eu
if [ "${SERVICE:-api}" = "ui" ]; then
  exec streamlit run app.py --server.address=0.0.0.0 --server.port="${PORT:-8501}" --server.headless=true --browser.gatherUsageStats=false --server.maxUploadSize=5
fi
alembic upgrade head
exec uvicorn qmworkbench.api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
