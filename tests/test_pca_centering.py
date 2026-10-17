import io

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from qmworkbench.analysis import analyze, make_pipeline
from qmworkbench.data import demo_data, demo_pca_data
from qmworkbench.features import FeatureBuilder
from qmworkbench.report import html_report
from qmworkbench.schemas import AnalysisConfig

SENSORS = ["sensor_pressure", "sensor_flow", "sensor_current", "sensor_temperature", "sensor_vibration", "sensor_load"]


def test_center_before_products_and_train_only():
    train = pd.DataFrame({"x": [10., 12., 14., np.nan], "z": [2., 4., 6., 8.]})
    builder = FeatureBuilder(numeric=("x", "z"), interactions=(("x", "z"),), center_numeric=True).fit(train)
    values = builder.transform(pd.DataFrame({"x": [12., 1000.], "z": [5., 100.]}))
    np.testing.assert_allclose(builder.numeric_centers_, [12., 5.])
    np.testing.assert_allclose(values[0], [0., 0., 0., 0.])
    assert values[1,2] == (1000-12)*(100-5)
    np.testing.assert_allclose(builder.numeric_centers_, [12., 5.])


def test_centering_preserves_full_hierarchical_ols_and_design_backtransform():
    frame = demo_data(250)
    cfg = dict(target="deviation", numeric=["temperature", "throughput", "operating_hours"], categorical=["machine_type"],
               interactions=[("temperature", "throughput"), ("temperature", "machine_type")], compare_elastic=False)
    raw,_ = analyze(frame, AnalysisConfig(**cfg, center_numeric=False))
    centered, artifact = analyze(frame, AnalysisConfig(**cfg, center_numeric=True))
    a = [r["prediction"] for r in raw["models"]["reference"]["rows"]]
    b = [r["prediction"] for r in centered["models"]["reference"]["rows"]]
    np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-9)
    pipe = joblib.load(io.BytesIO(artifact))["models"]["reference"][0]
    X = frame.iloc[[r-2 for r in centered["test_source_rows"]]]
    design = pipe.named_steps["features"].transform(X)
    coef = centered["models"]["reference"]["design_coefficients"]["rows"]
    predicted = coef[0]["coefficient"] + design @ np.array([r["coefficient"] for r in coef[1:]])
    np.testing.assert_allclose(predicted,b)


def test_pca_matches_independent_training_transform_and_keeps_categories_separate():
    train = demo_pca_data(200)
    builder = FeatureBuilder(numeric=tuple(SENSORS), categorical=("machine_type",), pca_components=2).fit(train)
    imputer = SimpleImputer(strategy="median").fit(train[SENSORS])
    scaler = StandardScaler().fit(imputer.transform(train[SENSORS]))
    independent = PCA(n_components=2, svd_solver="full").fit(scaler.transform(imputer.transform(train[SENSORS])))
    test = train.iloc[:12].copy()
    test[SENSORS] = test[SENSORS] + 100
    values = builder.transform(test)
    np.testing.assert_allclose(values[:,:2],independent.transform(scaler.transform(imputer.transform(test[SENSORS]))))
    np.testing.assert_allclose(builder.pca_scaler_.mean_,scaler.mean_)
    assert builder.get_feature_names_out().tolist() == ["PC1", "PC2", "machine_type[Typ B]", "sensor_temperature[fehlend]"]
    meta = builder.metadata()["pca"]
    assert set(meta["input_means"]) == set(SENSORS)
    assert 0 < meta["variance"][-1]["cumulative"] <= 1.00000000001
    np.testing.assert_allclose(builder.pca_.components_ @ builder.pca_.components_.T,np.eye(2),atol=1e-12)


def test_full_component_pca_preserves_linear_predictions():
    frame = demo_pca_data(200)
    kwargs = dict(target="deviation", numeric=SENSORS, categorical=["machine_type"], compare_elastic=False)
    direct,_ = analyze(frame, AnalysisConfig(**kwargs, center_numeric=True))
    pca,_ = analyze(frame, AnalysisConfig(**kwargs, feature_mode="pca", pca_components=6))
    np.testing.assert_allclose([r["prediction"] for r in direct["models"]["reference"]["rows"]],
                               [r["prediction"] for r in pca["models"]["reference"]["rows"]], atol=1e-9)


@pytest.mark.parametrize("task,split", [("linear","random"),("linear","time"),("logistic","random"),("logistic","group"),("logistic","time")])
def test_pca_paths_export_persistence_and_odds_units(task, split):
    frame = demo_pca_data(300)
    c = AnalysisConfig(task=task, target="deviation" if task=="linear" else "violation", numeric=SENSORS,
        categorical=["machine_type"], feature_mode="pca", pca_components=2, split=split,
        entity="machine_id" if split=="group" else None, time="timestamp" if split=="time" else None,
        compare_elastic=(split=="random"))
    result,artifact = analyze(frame,c)
    model = result["models"]["reference"]
    assert model["inference"] is None
    assert model["preprocessing"]["first_chain"]["pca"]["n_components"] == 2
    assert model["pca_scores"] and len(model["pca_scores"]) == result["n_test"]
    assert "Komponentengewichte" in html_report(result,"synthetic", "id")
    saved = joblib.load(io.BytesIO(artifact))["models"]["reference"][0]
    X = frame.iloc[[r-2 for r in result["test_source_rows"]]]
    predicted = saved.predict(X) if task=="linear" else saved.predict_proba(X)[:,1]
    np.testing.assert_allclose(predicted,[r["prediction"] for r in model["rows"]])
    if task=="logistic":
        coefficients = model["design_coefficients"]["rows"]
        logits = coefficients[0]["coefficient"] + saved.named_steps["features"].transform(X) @ np.array([r["coefficient"] for r in coefficients[1:]])
        np.testing.assert_allclose(expit(logits),predicted)
        for row in coefficients:
            assert np.isclose(row["odds_ratio"], np.exp(row["coefficient"]))


def test_pca_rejects_unsupported_or_degenerate_designs():
    cfg = dict(target="deviation", numeric=["temperature", "throughput"], feature_mode="pca")
    for extra in [dict(pca_components=3),dict(imputation="mice"),dict(interactions=[("temperature","throughput")]),
                  dict(time="timestamp",time_feature=True)]:
        with pytest.raises(ValidationError):
            AnalysisConfig(**cfg,**extra)
    frame = pd.DataFrame({"x":np.arange(50),"z":2*np.arange(50)})
    with pytest.raises(ValueError,match="redundante"):
        FeatureBuilder(numeric=("x","z"),pca_components=2).fit(frame)
    with pytest.raises(ValueError,match="variierende"):
        FeatureBuilder(numeric=("x","z"),pca_components=1).fit(frame.assign(z=1))


def test_interaction_curves_and_supported_intervals():
    frame = demo_data(200).dropna()
    c = AnalysisConfig(target="deviation", numeric=["temperature","throughput"], center_numeric=True,
                       interactions=[("temperature","throughput")], effect_feature="temperature", effect_modifier="throughput", compare_elastic=False)
    r,_ = analyze(frame,c)
    m = r["models"]["reference"]
    assert len(m["conditional_curve"]["rows"]) == 90
    assert len(set(x["group"] for x in m["conditional_curve"]["rows"])) == 3
    for row in m["inference"]["coefficients"]:
        assert row["ci_lower"] <= row["coefficient"] <= row["ci_upper"]


def test_pca_component_name_does_not_collide_with_user_category():
    frame = demo_pca_data(100).rename(columns={"machine_type":"PC1"})
    builder = FeatureBuilder(numeric=tuple(SENSORS),categorical=("PC1",),pca_components=1).fit(frame)
    assert builder.get_feature_names_out().tolist() == ["PC1", "PC1[Typ B]", "sensor_temperature[fehlend]"]
    assert builder.transform(frame).shape == (100,3)
