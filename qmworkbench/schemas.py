from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    task: Literal["linear", "logistic"] = "linear"
    target_mode: Literal["column", "deviation", "limits"] = "column"
    target: str
    target_positive: str = "1"
    nominal: str | None = None
    lower: float | None = None
    upper: float | None = None
    lower_column: str | None = None
    upper_column: str | None = None
    boundary_inclusive: bool = True
    numeric: list[str] = Field(default_factory=list, max_length=25)
    categorical: list[str] = Field(default_factory=list, max_length=8)
    entity: str | None = None
    time: str | None = None
    time_source: Literal["datetime", "numeric"] = "datetime"
    time_feature: bool = False
    time_squared: bool = False
    center_numeric: bool = False
    feature_mode: Literal["direct", "pca"] = "direct"
    pca_components: int = Field(default=2, ge=1, le=25)
    interactions: list[tuple[str, str]] = Field(default_factory=list, max_length=12)
    imputation: Literal["complete", "simple", "mice"] = "simple"
    imputations: int = Field(default=5, ge=2, le=10)
    mice_iterations: int = Field(default=10, ge=3, le=30)
    pooled_inference: bool = False
    compare_elastic: bool = True
    split: Literal["random", "group", "time"] = "random"
    test_fraction: float = Field(default=0.2, ge=0.15, le=0.35)
    threshold: float = Field(default=0.5, ge=0.01, le=0.99)
    effect_feature: str | None = None
    effect_group: str | None = None
    effect_modifier: str | None = None
    seed: int = Field(default=42, ge=0, le=2147483647)

    @model_validator(mode="after")
    def valid_design(self):
        features = self.numeric + self.categorical
        if len(features) != len(set(features)):
            raise ValueError("Features müssen eindeutige, getrennte Rollen haben.")
        if not features and not self.time_feature:
            raise ValueError("Mindestens ein Feature auswählen.")
        if self.target in features or (self.nominal and self.nominal in features):
            raise ValueError("Target und zur Targetbildung verwendete Spalten nicht als Features verwenden.")
        if self.target_mode == "deviation" and (self.task != "linear" or not self.nominal):
            raise ValueError("Normabweichung benötigt lineares Target und Sollwertspalte.")
        if self.target_mode == "limits":
            if self.task != "logistic":
                raise ValueError("Toleranzverletzung ist ein binäres Target.")
            if all(x is None for x in [self.lower, self.upper, self.lower_column, self.upper_column]):
                raise ValueError("Mindestens eine Toleranzgrenze festlegen.")
            if self.lower is not None and self.lower_column:
                raise ValueError("Untere Grenze als Konstante ODER Spalte wählen.")
            if self.upper is not None and self.upper_column:
                raise ValueError("Obere Grenze als Konstante ODER Spalte wählen.")
            if self.lower is not None and self.upper is not None and self.lower > self.upper:
                raise ValueError("Untere Grenze liegt über oberer Grenze.")
        if self.split == "group" and not self.entity:
            raise ValueError("Gruppentrennung benötigt eine Einheit-ID.")
        if (self.split == "time" or self.time_feature) and not self.time:
            raise ValueError("Zeitaufteilung/-feature benötigt eine Zeitspalte.")
        if self.time_squared and not self.time_feature:
            raise ValueError("Zeitquadrat benötigt den linearen Zeitterm.")
        available = set(features) | ({"__time"} if self.time_feature else set()) | ({"__time2"} if self.time_squared else set())
        if self.effect_feature and self.effect_feature not in set(self.numeric) | ({"__time"} if self.time_feature else set()):
            raise ValueError("Einflussdiagramm benötigt numerisches Feature oder __time.")
        if self.effect_group and (not self.effect_feature or self.effect_group not in self.categorical):
            raise ValueError("Diagrammgruppierung benötigt eine ausgewählte Kategorie und ein numerisches Diagrammfeature.")
        if self.effect_modifier and (not self.effect_feature or self.effect_modifier not in self.numeric or self.effect_modifier == self.effect_feature or self.effect_group):
            raise ValueError("Numerischer Kurvenvergleich benötigt ein anderes numerisches Feature und keine Kategoriengruppierung.")
        if self.feature_mode == "pca":
            if len(self.numeric) < 2 or self.pca_components > len(self.numeric):
                raise ValueError("PCA benötigt mindestens zwei numerische Features; Komponentenanzahl darf deren Anzahl nicht überschreiten.")
            if self.imputation == "mice" or self.interactions or self.time_feature or self.time_squared or self.pooled_inference:
                raise ValueError("PCA unterstützt zunächst einfache Imputation/vollständige Fälle ohne MICE, Interaktionen oder Zeitpolynome. Zeit- und Gruppentests bleiben möglich.")
        pairs = set()
        for a, b in self.interactions:
            pair = tuple(sorted((a, b)))
            if a == b or a not in available or b not in available or pair in pairs:
                raise ValueError("Interaktionen müssen eindeutige Paare ausgewählter Features sein.")
            pairs.add(pair)
        if self.imputation == "mice" and (self.interactions or self.time_squared or self.entity):
            raise ValueError("MICE mit Interaktionen, Zeitquadrat oder Einheitenstruktur ist in v0.1 gesperrt.")
        if self.pooled_inference and (self.task != "linear" or self.imputation != "mice" or self.categorical or self.time_feature or self.split != "random"):
            raise ValueError("Gepoolte MICE-Inferenz unterstützt nur additive numerische lineare Querschnittsmodelle.")
        return self
