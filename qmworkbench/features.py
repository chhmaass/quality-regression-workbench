import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer, SimpleImputer


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Fold-local imputation, reference encoding and selected interaction terms."""

    def __init__(self, numeric=(), categorical=(), time_feature=False, time_squared=False,
                 interactions=(), imputation="simple", seed=42, iterations=10,
                 center_numeric=False, pca_components=None):
        self.center_numeric = center_numeric
        self.pca_components = pca_components
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
            imputed = self._impute(X)
            self.numeric_centers_ = np.mean(imputed, axis=0) if self.center_numeric else np.zeros(len(self.numeric_))
            if self.pca_components is not None:
                if self.pca_components > min(len(self.numeric_), len(X) - 1):
                    raise ValueError("PCA-Komponentenanzahl übersteigt Featurezahl oder Trainingsumfang eines Folds.")
                self.pca_scaler_ = StandardScaler().fit(imputed)
                if np.any(self.pca_scaler_.var_ <= 0):
                    raise ValueError("PCA benötigt variierende numerische Features in jedem Trainingsfold.")
                self.pca_ = PCA(n_components=self.pca_components, svd_solver="full").fit(self.pca_scaler_.transform(imputed))
                if self.pca_.explained_variance_[-1] <= np.finfo(float).eps * max(1., self.pca_.explained_variance_[0]):
                    raise ValueError("Gewählte PCA enthält eine redundante Komponente; Komponentenanzahl reduzieren.")
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
            values = self._impute(X)
            if self.pca_components is not None:
                scores = self.pca_.transform(self.pca_scaler_.transform(values))
                for j in range(scores.shape[1]):
                    blocks[f"__pca_{j+1}"] = pd.DataFrame({f"PC{j+1}": scores[:, j]}, index=X.index)
            else:
                for j, name in enumerate(self.numeric_):
                    blocks[name] = pd.DataFrame({name: values[:, j] - self.numeric_centers_[j]}, index=X.index)
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

    def _impute(self, X):
        if self.imputation == "mice":
            self.imputer_.random_state_ = np.random.RandomState(self.seed)
        return self.imputer_.transform(X[self.numeric_])

    def metadata(self):
        result = {"mode": "pca" if self.pca_components is not None else "direct",
                  "numeric_centers": {name: float(v) for name, v in zip(self.numeric_, getattr(self, "numeric_centers_", []))},
                  "center_numeric": self.center_numeric,
                  "time_center": float(self.time_origin_ + self.time_center_) if self.time_feature else None,
                  "missing_indicators": self.missing_columns_, "imputation": self.imputation,
                  "imputation_values": {name: float(v) for name,v in zip(self.numeric_, self.imputer_.statistics_)} if self.numeric_ and self.imputation != "mice" else None}
        if self.pca_components is not None:
            ratios = self.pca_.explained_variance_ratio_
            result["pca"] = {"n_components": self.pca_components,
                "variance": [{"component": f"PC{i+1}", "explained": float(v), "cumulative": float(sum(ratios[:i+1]))} for i,v in enumerate(ratios)],
                "weights": [{"feature": name, **{f"PC{i+1}": float(self.pca_.components_[i,j]) for i in range(self.pca_components)}} for j,name in enumerate(self.numeric_)],
                "input_means": {name: float(v) for name,v in zip(self.numeric_, self.pca_scaler_.mean_)},
                "input_scales": {name: float(v) for name,v in zip(self.numeric_, self.pca_scaler_.scale_)},
                "pca_center": self.pca_.mean_.tolist(),
                "note": "PCA nur auf standardisierten, imputierten numerischen Trainingsfeatures; Kategorien und Fehlwertindikatoren separat. Gewichte sind Eigenvektoren, keine Target-Importances. Vorzeichen der Komponenten sind beliebig."}
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.names_, dtype=object)
