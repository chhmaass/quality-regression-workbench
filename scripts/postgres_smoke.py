"""Run after alembic upgrade head, against the configured PostgreSQL service."""
from fastapi.testclient import TestClient
import os

from qmworkbench.api import app

with TestClient(app) as client:
    client.headers["X-API-Key"] = os.environ["WORKBENCH_API_KEY"]
    dataset = client.post("/datasets/demo")
    dataset.raise_for_status()
    dataset_id = dataset.json()["id"]
    response = client.post(f"/datasets/{dataset_id}/analyses", json={
        "target": "deviation", "numeric": ["temperature", "throughput"], "compare_elastic": False})
    response.raise_for_status()
    run = response.json()
    restored = client.get(f"/analyses/{run['id']}")
    restored.raise_for_status()
    assert restored.json() == run
    assert client.get(f"/analyses/{run['id']}/report").status_code == 200
    print("PostgreSQL migration / dataset / regression / report round trip OK")
