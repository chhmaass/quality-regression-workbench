import io
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit, KFold,
                                    StratifiedKFold, TimeSeriesSplit, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .data import build_target, numeric, profile_structure, time_values
from .features import FeatureBuilder
from .schemas import AnalysisConfig


def classification_metrics(y, probability, threshold):
    pred = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "confusion_axes": "Zeilen: beobachtet 0/1; Spalten: vorhergesagt 0/1",
            "positive_class": 1, "test_positive_count": tp + fn,
            "test_negative_count": tn + fp, "threshold": threshold}


def regression_metrics(y, pred):
    return {"r2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None,
            "rmse": float(np.sqrt(mean_squared_error(y, pred)))}


def make_pipeline(c: AnalysisConfig, regularized: bool, seed: int):
    if c.task == "linear":
        estimator = ElasticNet(max_iter=10000, random_state=seed) if regularized else LinearRegression()
    else:
        estimator = LogisticRegression(penalty="elasticnet" if regularized else None,
            solver="saga" if regularized else "lbfgs", max_iter=5000, random_state=seed)
    return Pipeline([("features", FeatureBuilder(numeric=tuple(c.numeric), categorical=tuple(c.categorical),
        time_feature=c.time_feature, time_squared=c.time_squared, interactions=tuple(c.interactions),
        imputation=c.imputation, seed=seed, iterations=c.mice_iterations,
        center_numeric=c.center_numeric, pca_components=c.pca_components if c.feature_mode == "pca" else None)),
        ("scale", StandardScaler()), ("model", estimator)])


def get_splits(X: pd.DataFrame, y: pd.Series, c: AnalysisConfig):
    idx = np.arange(len(X))
    if c.split == "group":
        if X["__entity"].nunique() < 5:
            raise ValueError("Gruppenvalidierung benötigt mindestens 5 Einheiten.")
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=c.test_fraction, random_state=c.seed).split(X, y, X["__entity"]))
        folds = list(GroupKFold(3).split(X.iloc[train], y.iloc[train], X.iloc[train]["__entity"]))
    elif c.split == "time":
        # Entire timestamps stay on one side, also inside each CV fold.
        times = np.sort(X["__raw_time"].unique())
        if len(times) < 12:
            raise ValueError("Zeitvalidierung benötigt mindestens 12 verschiedene Zeitpunkte.")
        cut = max(1, int(len(times) * (1 - c.test_fraction)))
        train = idx[X["__raw_time"].to_numpy() < times[cut]]
        test = idx[X["__raw_time"].to_numpy() >= times[cut]]
        folds = []
        train_times = np.sort(X.iloc[train]["__raw_time"].unique())
        for a, b in TimeSeriesSplit(3).split(train_times):
            local = X.iloc[train]["__raw_time"]
            folds.append((np.flatnonzero(local.isin(train_times[a])), np.flatnonzero(local.isin(train_times[b]))))
    else:
        train, test = train_test_split(idx, test_size=c.test_fraction, random_state=c.seed,
                                      stratify=y if c.task == "logistic" else None)
        splitter = StratifiedKFold(3, shuffle=True, random_state=c.seed) if c.task == "logistic" else KFold(3, shuffle=True, random_state=c.seed)
        folds = list(splitter.split(X.iloc[train], y.iloc[train]))
    if min(len(train), len(test)) < 8:
        raise ValueError("Trainings- und Testteil benötigen jeweils mindestens 8 Fälle.")
    if c.task == "logistic":
        for part in [train, test]:
            if y.iloc[part].nunique() < 2:
                raise ValueError("Trainings- und Testteil müssen beide Klassen enthalten.")
        for a, b in folds:
            if y.iloc[train].iloc[a].nunique() < 2 or y.iloc[train].iloc[b].nunique() < 2:
                raise ValueError("Ein Validierungsfold enthält nur eine Klasse; mehr Daten oder anderes Design nötig.")
    if c.feature_mode == "pca":
        smallest = min([len(train)] + ([len(a) for a, _ in folds] if c.compare_elastic else []))
        if c.pca_components > min(len(c.numeric), smallest - 1):
            raise ValueError("PCA-Komponentenanzahl übersteigt den kleinsten Trainingsfold; Anzahl reduzieren.")
    return train, test, folds


def pooled_mice_inference(X, y, c):
    """Separate retrospective training-sample inference, not predictive CV estimates."""
    import statsmodels.api as sm
    from statsmodels.imputation.mice import MICE, MICEData
    # Safe aliases avoid executing user-supplied formula text.
    names = [f"x{i}" for i in range(len(c.numeric))]
    frame = X[c.numeric].copy()
    frame.columns = names
    frame["outcome"] = y.to_numpy()
    if frame.shape[0] < max(40, 5 * (len(names) + 1)):
        raise ValueError("Zu wenige Trainingsfälle für den begrenzten MICE-Inferenzpfad.")
    state = np.random.get_state()
    try:
        np.random.seed(c.seed)
        data = MICEData(frame, k_pmm=min(10, len(frame) // 4))
        result = MICE("outcome ~ " + " + ".join(names), sm.OLS, data).fit(n_burnin=c.mice_iterations, n_imputations=c.imputations)
    finally:
        np.random.set_state(state)
    labels = ["Intercept"] + c.numeric
    return {"scope": "Retrospektive additive MICE-Inferenz ausschließlich auf Trainingsfällen; getrennt vom Vorhersageensemble.",
        "assumptions": "Geeignete Imputationsmodelle/MAR vorausgesetzt; keine Kausalitätsaussage.",
        "coefficients": [{"feature": name, "coefficient": float(result.params[i]), "standard_error": float(result.bse[i]),
                          "p_value": float(result.pvalues[i])} for i, name in enumerate(labels)]}


def conditional_curve(models, Xtrain, c):
    if not c.effect_feature:
        return None
    varying = "__raw_time" if c.effect_feature == "__time" else c.effect_feature
    reference = {name: float(Xtrain[name].median()) for name in c.numeric}
    reference.update({name: str(Xtrain[name].fillna("(fehlend)").mode().iloc[0]) for name in c.categorical})
    if c.time_feature:
        reference["__raw_time"] = float(Xtrain["__raw_time"].median())
    values = np.linspace(float(Xtrain[varying].min()), float(Xtrain[varying].max()), 30)
    groups = sorted(Xtrain[c.effect_group].fillna("(fehlend)").unique())[:4] if c.effect_group else ["Referenz"]
    if c.effect_modifier:
        quantiles = Xtrain[c.effect_modifier].quantile([.1, .5, .9]).drop_duplicates().tolist()
        groups = [f"{c.effect_modifier} = {v:.4g}" for v in quantiles]
    curve = []
    for group_index, group in enumerate(groups):
        grid = pd.DataFrame([reference.copy() for _ in values])
        grid[varying] = values
        if c.effect_group:
            grid[c.effect_group] = group
        if c.effect_modifier:
            grid[c.effect_modifier] = quantiles[group_index]
        predictions = [m.predict(grid) if c.task == "linear" else m.predict_proba(grid)[:, 1] for m in models]
        plotted_values = values - (models[0].named_steps["features"].time_origin_ + models[0].named_steps["features"].time_center_) if varying == "__raw_time" else values
        curve.extend({"x": float(x), "prediction": float(pred), "group": str(group)} for x, pred in zip(plotted_values, np.mean(predictions, axis=0)))
    return {"feature": c.effect_feature, "group_feature": c.effect_group or c.effect_modifier, "reference": reference, "rows": curve,
            "note": "Bedingte Modellvorhersage: übrige Features auf Trainingsmedian/-modus fixiert. Kategorien: höchstens vier Gruppen; numerischer Vergleich: Trainingsquantile 10/50/90 %. Zeitachse bei __time um Trainingsmittel zentriert. Keine Kausalwirkung."}


def reference_inference(pipeline, Xtrain, ytrain, c):
    if c.feature_mode == "pca" or c.task != "linear" or c.imputation == "mice" or Xtrain[c.numeric].isna().any().any():
        return None
    import statsmodels.api as sm
    design = pipeline.named_steps["scale"].transform(pipeline.named_steps["features"].transform(Xtrain))
    if c.entity:
        groups = Xtrain["__entity"].to_numpy(dtype=str)
        if len(set(groups)) < 20:
            return {"note": "Keine Cluster-Inferenz bei weniger als 20 Trainingsgruppen in diesem Demonstrator."}
        result = sm.OLS(ytrain.to_numpy(), sm.add_constant(design, has_constant="add")).fit(cov_type="cluster", cov_kwds={"groups": groups})
        covariance = "Nach Einheit geclustert; unabhängige Gruppen und ausreichende Gruppenzahl vorausgesetzt."
    else:
        result = sm.OLS(ytrain.to_numpy(), sm.add_constant(design, has_constant="add")).fit(cov_type="HC3")
        covariance = "HC3; berücksichtigt Heteroskedastizität, keine zeitliche oder gruppeninterne Abhängigkeit."
    names = ["Intercept"] + pipeline.named_steps["features"].get_feature_names_out().tolist()
    return {"note": "Inferenz auf Trainingsfällen und standardisierter Designmatrix. " + covariance,
            "coefficients": [{"feature": name, "coefficient": float(result.params[i]), "standard_error": float(result.bse[i]),
                              "p_value": float(result.pvalues[i]), "ci_lower": float(result.conf_int()[i,0]),
                              "ci_upper": float(result.conf_int()[i,1])} for i, name in enumerate(names)]}


def design_coefficients(models, names, c):
    """Back-transform final design scaling, retaining centered/PC design units."""
    betas = []
    intercepts = []
    for pipeline in models:
        scale, estimator = pipeline.named_steps["scale"], pipeline.named_steps["model"]
        beta = np.ravel(estimator.coef_) / scale.scale_
        betas.append(beta)
        intercepts.append(float(np.ravel(estimator.intercept_)[0] - np.dot(scale.mean_, beta)))
    mean = np.mean(betas, axis=0)
    rows = [{"feature": "Intercept", "coefficient": float(np.mean(intercepts))}]
    rows += [{"feature": str(name), "coefficient": float(value)} for name, value in zip(names, mean)]
    if c.task == "logistic":
        for row in rows:
            value = row["coefficient"]
            row["odds_ratio"] = float(np.exp(value)) if abs(value) < 700 else None
    return {"rows": rows,
            "note": "Koeffizienten in Einheiten der unskalierten Designspalten: numerische Features ggf. um Trainingsmittel zentriert, Dummies 0→1, PC-Scores nach PCA. Bei Interaktionen hängt ein Haupteffekt vom Wert der anderen Merkmale ab. Odds Ratios sind bedingte Odds-Verhältnisse, keine Wahrscheinlichkeitsverhältnisse. Intercept-Exponent = Basis-Odds." +
            (" Bei MICE Mittelwerte über Modelle, keine Koeffizienten oder Odds Ratios eines einzelnen Ensemblemodells; keine gepoolten Intervalle." if len(models) > 1 else "")}


def preprocessing_details(models, c):
    chain_details = []
    for model in models:
        builder = model.named_steps["features"]
        detail = builder.metadata()
        scale = model.named_steps["scale"]
        detail["design_scaling"] = [{"feature": str(name), "mean": float(mean), "scale": float(sd)}
            for name,mean,sd in zip(builder.get_feature_names_out(), scale.mean_, scale.scale_)]
        chain_details.append(detail)
    return {"first_chain": chain_details[0], "chains": chain_details,
            "note": "Alle Bezugswerte ausschließlich auf dem jeweiligen Training gelernt, auch innerhalb der Validierungsfolds. Bei mehreren MICE-Modellen gelten je Modell eigene Bezugswerte."}


def pca_test_scores(model, Xtest, ytest):
    builder = model.named_steps["features"]
    scores = builder.pca_.transform(builder.pca_scaler_.transform(builder._impute(Xtest)))
    step = max(1, int(np.ceil(len(scores) / 2000)))
    return [{"PC1": float(scores[i,0]), "PC2": float(scores[i,1]) if scores.shape[1] > 1 else None,
             "observed": float(ytest.iloc[i])} for i in range(0,len(scores),step)]


def prepare_analysis(df: pd.DataFrame, c: AnalysisConfig):
    """Validate and select eligible rows identically for preview and training."""
    required = set(c.numeric + c.categorical + [c.target])
    required |= {x for x in [c.nominal, c.lower_column, c.upper_column, c.entity, c.time] if x}
    if required - set(df.columns):
        raise ValueError(f"Spalten fehlen: {sorted(required - set(df.columns))}")
    if c.entity and c.entity in c.numeric + c.categorical:
        raise ValueError("Einheit-ID ist Metadatum; individuelle Maschinen-FE sind in v0.1 nicht vorgesehen.")
    if set(c.numeric + c.categorical) & {c.lower_column, c.upper_column}:
        raise ValueError("Grenzwertspalten nicht als direkte Features verwenden.")
    X = df[c.numeric + c.categorical].copy()
    for col in c.numeric:
        X[col] = numeric(X[col])
    for col in c.categorical:
        if (X[col].dropna().astype(str) == "(fehlend)").any():
            raise ValueError("Kategorie '(fehlend)' ist für fehlende Werte reserviert.")
        X[col] = X[col].map(lambda v: np.nan if pd.isna(v) else str(v))
    y = build_target(df, c)
    valid = y.notna()
    if c.entity:
        X["__entity"] = df[c.entity].astype("string")
        if df[c.entity].isna().any():
            raise ValueError("Einheit-ID enthält Fehlwerte.")
    if c.time:
        X["__raw_time"] = time_values(df, c)
        if X["__raw_time"].isna().any():
            raise ValueError("Zeitspalte enthält Fehlwerte.")
    if c.imputation == "complete":
        valid &= X[c.numeric + c.categorical].notna().all(axis=1)
    source_rows = np.flatnonzero(valid.to_numpy())
    X, y = X.loc[valid].copy(), y.loc[valid].copy()
    if len(y) < 50:
        raise ValueError("Mindestens 50 auswertbare Fälle erforderlich.")
    if c.task == "logistic" and (y.nunique() != 2 or y.value_counts().min() < 8):
        raise ValueError("Binäre Regression benötigt mindestens 8 Fälle je Klasse.")
    return X, y, source_rows


def analyze(df: pd.DataFrame, c: AnalysisConfig) -> tuple[dict, bytes]:
    started = time.monotonic()
    X, y, source_rows = prepare_analysis(df, c)
    train, test, folds = get_splits(X, y, c)
    Xtrain, Xtest = X.iloc[train], X.iloc[test]
    ytrain, ytest = y.iloc[train], y.iloc[test]
    messages = ["Beobachtungsdaten belegen keine Kausalität. Grenzen und Features fachlich prüfen.",
                "Wahrscheinlichkeitsschwelle vor der Testbewertung festlegen; nachträgliches Erkunden ist explorativ."]
    if c.entity and c.split == "random":
        messages.append("Wiederholte Einheiten bei Zufallssplit: keine Aussage über Generalisierung auf neue Einheiten oder Zukunft.")
    if c.feature_mode == "pca":
        messages.append("PCA erklärt Featurevarianz, nicht Vorhersagegüte. Komponentenanzahl fachlich vorab festlegen; keine klassische Inferenz für den PCA-Pfad.")
    if c.imputation == "mice":
        messages.append("Vorhersagen sind Mittelwerte mehrerer fold-lokal imputierter Modelle; Koeffizientenstreuung ist kein Rubin-Konfidenzintervall.")
    if c.imputation == "simple":
        messages.append("Einfache Imputation berücksichtigt die Unsicherheit fehlender Werte nicht vollständig.")
    results, artifacts = {}, {}
    scoring = "neg_mean_squared_error" if c.task == "linear" else "neg_log_loss"
    chain_count = c.imputations if c.imputation == "mice" else 1
    if c.imputation == "mice" and len(c.numeric) < 2:
        raise ValueError("MICE benötigt hier mindestens zwei numerische Einflussgrößen.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for label, regularized in [("reference", False)] + ([("elastic_net", True)] if c.compare_elastic else []):
            params = {}
            cv_score = None
            if regularized:
                grid = {"model__alpha": [.01, .1, 1.], "model__l1_ratio": [.2, .8]} if c.task == "linear" else {"model__C": [.1, 1., 10.], "model__l1_ratio": [.2, .8]}
                search = GridSearchCV(make_pipeline(c, True, c.seed), grid, cv=folds, scoring=scoring, error_score="raise", n_jobs=1)
                search.fit(Xtrain, ytrain)
                params, cv_score = search.best_params_, float(search.best_score_)
            models, predictions, coefficients = [], [], []
            unknowns = {}
            for i in range(chain_count):
                pipeline = make_pipeline(c, regularized, c.seed + i).set_params(**params)
                pipeline.fit(Xtrain, ytrain)
                builder = pipeline.named_steps["features"]
                transformed = builder.transform(Xtrain)
                if not regularized:
                    rank = np.linalg.matrix_rank(np.c_[np.ones(len(transformed)), transformed])
                    if rank < transformed.shape[1] + 1:
                        raise ValueError("Referenzmodell nicht eindeutig: konstante/redundante Features oder Interaktionen entfernen.")
                    if len(train) < 3 * (transformed.shape[1] + 1):
                        raise ValueError("Zu viele Modellterme für den Trainingsumfang (mindestens 3 Fälle je Parameter).")
                pred = pipeline.predict(Xtest) if c.task == "linear" else pipeline.predict_proba(Xtest)[:, 1]
                predictions.append(pred)
                coefficients.append(np.ravel(pipeline.named_steps["model"].coef_))
                models.append(pipeline)
                for name, levels in builder.category_levels_.items():
                    count = int((~Xtest[name].fillna("(fehlend)").isin(levels)).sum())
                    if count:
                        unknowns[name] = count
            if unknowns:
                messages.append(f"Unbekannte Testkategorien (als Referenz kodiert; Extrapolation): {unknowns}")
            mean_pred = np.mean(predictions, axis=0)
            metrics = regression_metrics(ytest, mean_pred) if c.task == "linear" else classification_metrics(ytest, mean_pred, c.threshold)
            baseline_pred = np.repeat(float(ytrain.mean()), len(test)) if c.task == "linear" else np.repeat(float(ytrain.mode().iloc[0]), len(test))
            baseline = regression_metrics(ytest, baseline_pred) if c.task == "linear" else classification_metrics(ytest, baseline_pred, .5)
            names = models[0].named_steps["features"].get_feature_names_out()
            results[label] = {"metrics": metrics, "baseline": baseline, "parameters": params,
                "cv_score": cv_score, "cv_scoring": scoring if regularized else None,
                "coefficient_scale": "Koeffizienten pro Standardabweichung der erzeugten Designspalte; Kategorienreferenzen siehe Kodierung.",
                "coefficients": [{"feature": str(name), "coefficient": float(v)} for name, v in zip(names, np.mean(coefficients, axis=0))],
                "intercept": float(np.mean([np.ravel(m.named_steps["model"].intercept_)[0] for m in models])),
                "category_references": {name: levels[0] for name, levels in models[0].named_steps["features"].category_levels_.items()},
                "conditional_curve": conditional_curve(models, Xtrain, c),
                "preprocessing": preprocessing_details(models, c),
                "design_coefficients": design_coefficients(models, names, c),
                "pca_scores": pca_test_scores(models[0], Xtest, ytest) if c.feature_mode == "pca" else None,
                "inference": reference_inference(models[0], Xtrain, ytrain, c) if not regularized else None,
                "rows": [{"source_row": int(source_rows[test[j]] + 2), "observed": float(ytest.iloc[j]), "prediction": float(mean_pred[j]),
                    "residual": float(ytest.iloc[j] - mean_pred[j]) if c.task == "linear" else None,
                    "predicted_positive": bool(mean_pred[j] >= c.threshold) if c.task == "logistic" else None,
                    "imputation_prediction_sd": float(np.std(predictions, axis=0)[j]),
                    "imputed_numeric_count": int(Xtest.iloc[j][c.numeric].isna().sum()),
                    "group": str(df.iloc[source_rows[test[j]]][c.entity]) if c.entity else None,
                    "time": str(df.iloc[source_rows[test[j]]][c.time]) if c.time else None} for j in range(len(test))]}
            artifacts[label] = models
        convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        if convergence:
            raise ValueError("Modell/Validierungsfold konvergiert nicht: Features reduzieren oder Regularisierung prüfen. " + convergence[0])
    if c.time_feature and Xtest["__raw_time"].max() > Xtrain["__raw_time"].max():
        messages.append("Zeitvorhersagen außerhalb des Trainingszeitraums sind Extrapolation; quadratische Trends besonders vorsichtig interpretieren.")
    inference = pooled_mice_inference(Xtrain, ytrain, c) if c.pooled_inference else None
    output = {"config": c.model_dump(), "structure": profile_structure(df, c.entity, c.time),
        "n_input": len(df), "n_eligible": len(y), "unknown_target": int(build_target(df, c).isna().sum()),
        "excluded": int(len(df) - len(y)), "n_train": len(train), "n_test": len(test),
        "train_source_rows": (source_rows[train] + 2).tolist(), "test_source_rows": (source_rows[test] + 2).tolist(),
        "test_scope": "Ausschließlich zurückgehaltene Testfälle; kein automatisches anschließendes Training auf allen Fällen.",
        "models": results, "pooled_inference": inference,
        "warnings": list(dict.fromkeys(messages)), "duration_seconds": round(time.monotonic() - started, 3),
        "versions": {"workbench": __version__, "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}
    buffer = io.BytesIO()
    joblib.dump({"config": c.model_dump(), "models": artifacts, "versions": output["versions"]}, buffer)
    return output, buffer.getvalue()
