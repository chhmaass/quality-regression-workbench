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
    listing = client.get("/datasets").json()
    assert any(d["id"] == dataset_id and d["run_count"] >= 1 for d in listing["items"])
    runs = client.get(f"/datasets/{dataset_id}/analyses").json()
    assert any(r["id"] == run["id"] and r["target"] == "deviation" for r in runs["items"])
    assert client.delete(f"/datasets/{dataset_id}?expected_runs=0").status_code == 409
    assert client.delete(f"/analyses/{run['id']}").status_code == 200
    assert client.get(f"/analyses/{run['id']}").status_code == 404
    remaining = client.get(f"/datasets/{dataset_id}/analyses").json()["total"]
    assert client.delete(f"/datasets/{dataset_id}?expected_runs={remaining}").status_code == 200
    assert client.get(f"/datasets/{dataset_id}").status_code == 404
    print("PostgreSQL migration / regression / report / listings / confirmed deletion round trip OK")
