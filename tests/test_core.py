import io

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from qmworkbench.analysis import analyze, classification_metrics, get_splits, make_pipeline, regression_metrics
from qmworkbench.data import build_target, demo_data, parse_csv, profile_structure
from qmworkbench.report import html_report, safe_csv
from qmworkbench.schemas import AnalysisConfig


def test_target_boundaries_missing_and_one_sided():
    df = pd.DataFrame({"x": [-3, -2, 0, 2, 3, np.nan]})
    c = AnalysisConfig(task="logistic", target_mode="limits", target="x", lower=-2, upper=2, numeric=["a"])
    y = build_target(df, c)
    assert y.iloc[:5].tolist() == [1, 0, 0, 0, 1]
    assert np.isnan(y.iloc[5])
    strict = c.model_copy(update={"boundary_inclusive": False})
    assert build_target(df, strict).iloc[:5].tolist() == [1, 1, 0, 1, 1]
    upper_only = c.model_copy(update={"lower": None})
    assert build_target(df, upper_only).iloc[:5].tolist() == [0, 0, 0, 0, 1]


def test_per_row_limits_do_not_turn_unknown_into_zero():
    frame = pd.DataFrame({"x": [2, 2, 2], "lo": [-1, np.nan, 3], "hi": [1, 1, 1]})
    c = AnalysisConfig(task="logistic", target_mode="limits", target="x", lower_column="lo", upper_column="hi", numeric=["a"])
    with pytest.raises(ValueError, match="Untergrenze"):
        build_target(frame, c)
    y = build_target(frame.iloc[:2], c)
    assert y.iloc[0] == 1 and np.isnan(y.iloc[1])


def test_deviation_and_binary_labels():
    df = pd.DataFrame({"actual": [501, 498], "nominal": [500, 500], "status": ["ok", "bad"]})
    c = AnalysisConfig(target_mode="deviation", target="actual", nominal="nominal", numeric=["a"])
    assert build_target(df, c).tolist() == [1, -2]
    c = AnalysisConfig(task="logistic", target="status", target_positive="bad", numeric=["a"])
    assert build_target(df, c).tolist() == [0, 1]


def test_metrics_are_hand_calculable_and_null_denominators():
    m = classification_metrics([0, 0, 1, 1], [.1, .9, .2, .8], .5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["accuracy"] == m["precision"] == m["recall"] == .5
    assert classification_metrics([0, 1], [.1, .1], .5)["precision"] is None
    assert classification_metrics([0, 0], [.9, .1], .5)["recall"] is None
    assert regression_metrics([1, 2], [4, 5])["r2"] < 0
    assert regression_metrics([1, 1], [1, 1])["r2"] is None


def test_csv_validation_and_dialects():
    f, digest = parse_csv(b"a;b\n1;2\n3;4\n")
    assert f.shape == (2, 2) and len(digest) == 64
    with pytest.raises(ValueError, match="eindeutig"):
        parse_csv(b"a,a\n1,2\n")
    with pytest.raises(ValueError, match="unendliche"):
        parse_csv(b"a,b\ninf,2\n")
    with pytest.raises(ValueError):
        parse_csv(b"x" * (5 * 1024 * 1024 + 1))


def test_configuration_blocks_unchecked_combinations_and_target_leakage():
    with pytest.raises(ValidationError):
        AnalysisConfig(target="y", numeric=["y"])
    with pytest.raises(ValidationError, match="MICE"):
        AnalysisConfig(target="y", numeric=["x", "z"], imputation="mice", interactions=[("x", "z")])
    with pytest.raises(ValidationError):
        AnalysisConfig(task="logistic", target="y", numeric=["x", "z"], imputation="mice", pooled_inference=True)


def test_group_and_time_splits_prevent_overlap():
    X = pd.DataFrame({"x": np.arange(120), "__entity": np.repeat(np.arange(12), 10), "__raw_time": np.repeat(np.arange(60), 2)})
    y = pd.Series(np.arange(120))
    c = AnalysisConfig(target="y", numeric=["x"], entity="id", split="group")
    tr, te, folds = get_splits(X, y, c)
    assert set(X.iloc[tr].__entity).isdisjoint(X.iloc[te].__entity)
    for a, b in folds:
        assert set(X.iloc[tr].iloc[a].__entity).isdisjoint(X.iloc[tr].iloc[b].__entity)
    c = AnalysisConfig(target="y", numeric=["x"], time="timestamp", split="time")
    tr, te, folds = get_splits(X, y, c)
    assert X.iloc[tr].__raw_time.max() < X.iloc[te].__raw_time.min()
    for a, b in folds:
        assert X.iloc[tr].iloc[a].__raw_time.max() < X.iloc[tr].iloc[b].__raw_time.min()


def test_transformations_only_learn_from_train_and_unknown_category_is_handled():
    c = AnalysisConfig(target="y", numeric=["x"], categorical=["type"], compare_elastic=False)
    tr = pd.DataFrame({"x": [0., 2., np.nan, 4.], "type": ["A", "B", "A", "B"]})
    pipe = make_pipeline(c, False, 1).fit(tr, [0, 1, 2, 3])
    builder = pipe.named_steps["features"]
    assert builder.imputer_.statistics_[0] == 2
    transformed = builder.transform(pd.DataFrame({"x": [np.nan, 1e8], "type": ["C", "C"]}))
    assert transformed[0, 0] == 2
    assert builder.imputer_.statistics_[0] == 2


@pytest.mark.parametrize("task", ["linear", "logistic"])
def test_end_to_end_regression_and_elastic(task):
    df = demo_data(250)
    c = AnalysisConfig(task=task, target="deviation" if task == "linear" else "violation",
        numeric=["temperature", "throughput", "operating_hours"], categorical=["machine_type"])
    result, artifact = analyze(df, c)
    assert set(result["models"]) == {"reference", "elastic_net"}
    assert set(result["train_source_rows"]).isdisjoint(result["test_source_rows"])
    assert len(result["models"]["reference"]["rows"]) == result["n_test"]
    assert joblib.load(io.BytesIO(artifact))["versions"] == result["versions"]
    if task == "linear":
        assert result["models"]["reference"]["metrics"]["rmse"] < 2
    else:
        assert sum(map(sum, result["models"]["reference"]["metrics"]["confusion_matrix"])) == result["n_test"]


def test_selected_interactions_time_square_and_temporal_validation():
    df = demo_data(250)
    c = AnalysisConfig(target="deviation", numeric=["temperature", "throughput"], categorical=["machine_type"],
        time="timestamp", time_feature=True, time_squared=True, split="time",
        interactions=[("temperature", "throughput"), ("__time", "machine_type")], compare_elastic=False)
    r, _ = analyze(df, c)
    names = [x["feature"] for x in r["models"]["reference"]["coefficients"]]
    assert "__time2" in names and "temperature × throughput" in names
    assert any("Extrapolation" in w for w in r["warnings"])


def test_mice_multiple_chains_repeatability_and_training_inference():
    frame = demo_data(150)
    c = AnalysisConfig(target="deviation", numeric=["temperature", "throughput"],
        imputation="mice", imputations=2, mice_iterations=3, pooled_inference=True, compare_elastic=False)
    r, artifact = analyze(frame, c)
    again, _ = analyze(frame, c)
    np.testing.assert_allclose([x["prediction"] for x in r["models"]["reference"]["rows"]],
                               [x["prediction"] for x in again["models"]["reference"]["rows"]])
    assert len(joblib.load(io.BytesIO(artifact))["models"]["reference"]) == 2
    assert len(r["pooled_inference"]["coefficients"]) == 3


def test_redundancy_and_type_errors_are_explicit():
    df = demo_data(100)
    df["constant"] = 1
    with pytest.raises(ValueError, match="redundante"):
        analyze(df, AnalysisConfig(target="deviation", numeric=["throughput", "constant"], compare_elastic=False))
    df["throughput"] = df["throughput"].astype(object)
    df.loc[0, "throughput"] = "oops"
    with pytest.raises(ValueError, match="nichtnumerische"):
        analyze(df, AnalysisConfig(target="deviation", numeric=["throughput"], compare_elastic=False))


def test_report_escapes_user_text_and_csv_formula():
    r, _ = analyze(demo_data(100), AnalysisConfig(target="deviation", numeric=["throughput"], compare_elastic=False))
    report = html_report(r, '<script>alert("x")</script>', "id")
    assert "<script>" not in report and "&lt;script&gt;" in report
    assert "'=1+1" in safe_csv([{"group": "=1+1"}]).decode("utf-8-sig")


def test_structure_is_a_proposal_not_an_unsupported_assertion():
    f = demo_data(100)
    assert "Panel" in profile_structure(f, "machine_id", "timestamp")["proposal"]
    assert profile_structure(f, None, "timestamp")["needs_confirmation"]


def test_conditional_interaction_curve_and_hc3_training_inference():
    df = demo_data(200).dropna().reset_index(drop=True)
    c = AnalysisConfig(target="deviation", numeric=["temperature", "throughput"], categorical=["machine_type"],
        interactions=[("temperature", "machine_type")], effect_feature="temperature", effect_group="machine_type", compare_elastic=False)
    result, _ = analyze(df, c)
    model = result["models"]["reference"]
    assert len(model["conditional_curve"]["rows"]) == 60
    assert "HC3" in model["inference"]["note"]
    assert len(model["inference"]["coefficients"]) == len(model["coefficients"]) + 1


def test_no_naive_inference_after_single_imputation():
    result, _ = analyze(demo_data(200), AnalysisConfig(target="deviation", numeric=["temperature", "throughput"], compare_elastic=False))
    assert result["models"]["reference"]["inference"] is None
