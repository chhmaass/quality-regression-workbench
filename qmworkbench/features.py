import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Fold-local imputation, reference encoding and selected interaction terms."""

    def __init__(self, numeric=(), categorical=(), time_feature=False, time_squared=False,
                 interactions=(), imputation="simple", seed=42, iterations=10):
        self.numeric = numeric
        self.categorical = categorical
        self.time_feature = time_feature
        self.time_squared = time_squared
        self.interactions = interactions
        self.imputation = imputation
        self.seed = seed
        self.iterations = iterations

    def fit(self, X, y=None):
        self.numeric_ = list(self.numeric)
        self.category_levels_ = {}
        for name in self.categorical:
            values = X[name].fillna("(fehlend)").astype(str)
            if values.nunique() > 20:
                raise ValueError(f"{name}: maximal 20 Kategorien im Trainingsfold.")
            self.category_levels_[name] = sorted(values.unique())
        if self.numeric_:
            if X[self.numeric_].isna().all().any():
                raise ValueError("Ein numerisches Feature ist im Trainingsfold vollständig leer.")
            self.imputer_ = (IterativeImputer(sample_posterior=True, random_state=self.seed,
                max_iter=self.iterations, keep_empty_features=True) if self.imputation == "mice"
                else SimpleImputer(strategy="median", keep_empty_features=True))
            self.imputer_.fit(X[self.numeric_])
            self.missing_columns_ = [name for name in self.numeric_ if X[name].isna().any()]
        else:
            self.missing_columns_ = []
        self.time_origin_ = float(X["__raw_time"].min()) if self.time_feature else None
        self.time_center_ = float((X["__raw_time"] - self.time_origin_).mean()) if self.time_feature else None
        # Validate term count and rank later against actual fitting data.
        a = self.transform(X.iloc[:2])
        if a.shape[1] > 150:
            raise ValueError("Mehr als 150 erzeugte Features. Kategorien/Interaktionen reduzieren.")
        return self

    def transform(self, X):
        blocks = {}
        if self.numeric_:
            if self.imputation == "mice":
                # One deterministic posterior draw per chain and input batch.
                # Reopening a saved run must not silently change its predictions.
                self.imputer_.random_state_ = np.random.RandomState(self.seed)
            numeric = self.imputer_.transform(X[self.numeric_])
            for j, name in enumerate(self.numeric_):
                blocks[name] = pd.DataFrame({name: numeric[:, j]}, index=X.index)
        for name, levels in self.category_levels_.items():
            values = X[name].fillna("(fehlend)").astype(str)
            blocks[name] = pd.DataFrame({f"{name}[{level}]": (values == level).astype(float)
                                         for level in levels[1:]}, index=X.index)
        if self.time_feature:
            t = X["__raw_time"] - self.time_origin_ - self.time_center_
            blocks["__time"] = pd.DataFrame({"__time": t}, index=X.index)
            if self.time_squared:
                blocks["__time2"] = pd.DataFrame({"__time2": t**2}, index=X.index)
        frames = list(blocks.values())
        for left, right in self.interactions:
            a, b = blocks[left], blocks[right]
            frames.append(pd.DataFrame({f"{x} × {z}": a[x] * b[z] for x in a for z in b}, index=X.index))
        for name in self.missing_columns_:
            frames.append(pd.DataFrame({f"{name}[fehlend]": X[name].isna().astype(float)}, index=X.index))
        out = pd.concat(frames, axis=1)
        self.names_ = out.columns.tolist()
        if out.empty or not np.isfinite(out.to_numpy()).all():
            raise ValueError("Leere oder ungültige Featurematrix.")
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.names_, dtype=object)
