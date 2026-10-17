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


def test_target_preview_matches_analysis_and_excludes_unknown_values(client):
    import numpy as np
    from qmworkbench.data import demo_data
    frame = demo_data(120)
    frame.loc[100, "deviation"] = np.nan  # Outside the ten-row display preview.
    frame.loc[101, "lower"] = np.nan
    client.headers["X-API-Key"] = "test-secret"
    ds = client.post("/datasets", files={"file": ("preview.csv", frame.to_csv(index=False).encode(), "text/csv")}).json()
    config = {"task": "logistic", "target_mode": "limits", "target": "deviation", "lower_column": "lower", "upper_column": "upper",
              "numeric": ["temperature", "throughput"], "compare_elastic": False, "imputation": "complete"}
    response = client.post(f"/datasets/{ds['id']}/preview", json=config)
    assert response.status_code == 200, response.text
    preview = response.json()
    run = client.post(f"/datasets/{ds['id']}/analyses", json=config).json()
    r = run["result"]
    assert run["dataset_id"] == ds["id"]
    assert preview["unknown_target"] == r["unknown_target"] == 2
    assert preview["eligible"] == r["n_eligible"]
    assert preview["excluded"] == r["excluded"]
    assert preview["n_train"] == r["n_train"] and preview["n_test"] == r["n_test"]
    assert preview["positive"] + preview["negative"] == 118
    changed_threshold = client.post(f"/datasets/{ds['id']}/preview", json=dict(config, threshold=.8)).json()
    assert changed_threshold == preview  # Classification threshold does not redefine targets.
    client.headers.clear()
    assert client.post(f"/datasets/{ds['id']}/preview", json=config).status_code == 401


def test_preview_rejects_invalid_limits_and_split_before_training(client):
    client.headers["X-API-Key"] = "test-secret"
    ds = client.post("/datasets/demo").json()
    url = f"/datasets/{ds['id']}/preview"
    for config in [
        {"task": "logistic", "target_mode": "limits", "target": "deviation", "numeric": ["throughput"], "lower": 3, "upper": -3},
        {"target": "deviation", "numeric": ["throughput"], "split": "group"},
        {"target": "deviation", "numeric": ["throughput"], "split": "time", "time": "missing"},
    ]:
        assert client.post(url, json=config).status_code == 422
