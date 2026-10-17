import pytest
from fastapi.testclient import TestClient

from qmworkbench.api import app
from qmworkbench.storage import engine


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "test.db"))
    monkeypatch.setenv("AUTO_CREATE_TABLES", "1")
    monkeypatch.setenv("WORKBENCH_API_KEY", "test-secret")
    engine.cache_clear()
    with TestClient(app) as client:
        yield client
    engine().dispose()
    engine.cache_clear()


def test_authenticated_round_trip_persistence_and_exports(client):
    assert client.get("/health").status_code == 200
    assert client.post("/datasets/demo").status_code == 401
    client.headers["X-API-Key"] = "test-secret"
    ds = client.post("/datasets/demo").json()
    assert client.post("/datasets/demo").json()["id"] == ds["id"]
    response = client.post(f"/datasets/{ds['id']}/analyses", json={"task": "logistic", "target": "violation", "numeric": ["throughput", "operating_hours"], "categorical": ["machine_type"], "compare_elastic": False})
    assert response.status_code == 200, response.text
    run = response.json()
    assert run["result"]["dataset_digest"] == ds["digest"]
    assert client.get(f"/analyses/{run['id']}").json() == run
    for suffix in ["report", "csv", "config"]:
        assert client.get(f"/analyses/{run['id']}/{suffix}").status_code == 200
    assert client.get(f"/analyses/{run['id']}/csv?model=unknown").status_code == 404
    client.headers.clear()
    assert client.get(f"/analyses/{run['id']}").status_code == 401


def test_validation_errors_are_not_internal_errors(client):
    client.headers["X-API-Key"] = "test-secret"
    ds = client.post("/datasets/demo").json()
    response = client.post(f"/datasets/{ds['id']}/analyses", json={"target": "missing", "numeric": ["throughput"]})
    assert response.status_code == 422
    assert client.post("/datasets", files={"file": ("bad.csv", b"a,a\n1,2", "text/csv")}).status_code == 422


def test_no_secret_no_unprotected_access(client, monkeypatch):
    monkeypatch.delenv("WORKBENCH_API_KEY")
    assert client.post("/datasets/demo").status_code == 503
