import hashlib
import io

import numpy as np
import pandas as pd

from .schemas import AnalysisConfig

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10000


def parse_csv(raw: bytes) -> tuple[pd.DataFrame, str]:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("CSV muss zwischen 1 Byte und 5 MiB groß sein.")
    try:
        text = raw.decode("utf-8-sig")
        import csv
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
        header = next(csv.reader(io.StringIO(text), dialect))
        if any(not x.strip() for x in header) or len(set(header)) != len(header):
            raise ValueError("Spaltennamen müssen ausgefüllt und eindeutig sein.")
        if any(x.startswith("__") for x in header):
            raise ValueError("Spaltennamen mit __ sind für interne Merkmale reserviert.")
        df = pd.read_csv(io.StringIO(text), sep=dialect.delimiter, nrows=MAX_ROWS + 1)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
        raise ValueError("CSV nicht lesbar. UTF-8 und Komma, Semikolon oder Tab verwenden.") from exc
    if len(df) > MAX_ROWS or df.shape[1] > 80 or df.empty:
        raise ValueError("Maximal 10.000 Zeilen / 80 Spalten; mindestens eine Datenzeile.")
    if np.isinf(df.select_dtypes(include="number").to_numpy()).any():
        raise ValueError("CSV enthält unendliche Werte.")
    return df, hashlib.sha256(raw).hexdigest()


def quality(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df), "columns": list(df.columns),
        "missing": {c: int(df[c].isna().sum()) for c in df},
        "types": {c: str(df[c].dtype) for c in df},
        "unique": {c: int(df[c].nunique()) for c in df},
        "duplicate_rows": int(df.duplicated().sum()),
        "warnings": ["Identische Zeilen sind ohne fachlichen Schlüssel nur Dublettenkandidaten; sie werden nicht entfernt."],
        "preview": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
    }


def numeric(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & result.isna()
    if bad.any():
        raise ValueError(f"{series.name}: {int(bad.sum())} nichtnumerische Werte. Dezimalkomma vorher konvertieren.")
    if np.isinf(result.dropna()).any():
        raise ValueError(f"{series.name}: unendliche Werte sind nicht zulässig.")
    return result.astype(float)


def build_target(df: pd.DataFrame, c: AnalysisConfig) -> pd.Series:
    if c.target not in df:
        raise ValueError("Targetspalte nicht vorhanden.")
    if c.target_mode == "column" and c.task == "logistic":
        s = df[c.target]
        values = s.dropna().astype(str).unique().tolist()
        # UI/API strings also accept numeric columns parsed as floats.
        normalize = lambda x: str(x).removesuffix(".0")
        values = list(dict.fromkeys(normalize(x) for x in values))
        pos = normalize(c.target_positive)
        if len(values) != 2 or pos not in values:
            raise ValueError("Binäres Target benötigt genau zwei Klassen und eine vorhandene positive Klasse.")
        return s.map(lambda x: np.nan if pd.isna(x) else float(normalize(x) == pos))
    y = numeric(df[c.target])
    if c.target_mode == "deviation":
        if c.nominal not in df:
            raise ValueError("Sollwertspalte fehlt.")
        return y - numeric(df[c.nominal])
    if c.target_mode == "limits":
        lower = numeric(df[c.lower_column]) if c.lower_column else pd.Series(c.lower, index=df.index, dtype=float)
        upper = numeric(df[c.upper_column]) if c.upper_column else pd.Series(c.upper, index=df.index, dtype=float)
        has_lower = c.lower_column is not None or c.lower is not None
        has_upper = c.upper_column is not None or c.upper is not None
        if ((lower > upper) & lower.notna() & upper.notna()).any():
            raise ValueError("Mindestens eine Zeile hat Untergrenze > Obergrenze.")
        known = y.notna()
        if has_lower:
            known &= lower.notna()
        if has_upper:
            known &= upper.notna()
        violation = pd.Series(False, index=df.index)
        if has_lower:
            violation |= y < lower if c.boundary_inclusive else y <= lower
        if has_upper:
            violation |= y > upper if c.boundary_inclusive else y >= upper
        return violation.astype(float).where(known)
    return y


def time_values(df: pd.DataFrame, c: AnalysisConfig) -> pd.Series:
    if c.time_source == "numeric":
        return numeric(df[c.time])
    t = pd.to_datetime(df[c.time], errors="coerce", utc=True)
    if (df[c.time].notna() & t.isna()).any():
        raise ValueError("Ungültige Zeitangaben; ISO-8601 oder numerischen Counter verwenden.")
    return pd.Series(t.astype("int64") / 86400e9, index=df.index).where(t.notna())


def profile_structure(df: pd.DataFrame, entity: str | None, time: str | None) -> dict:
    if not entity or entity not in df:
        return {"proposal": "Unklar: ohne stabile Einheit-ID kein sicherer Strukturentscheid", "needs_confirmation": True}
    if not time or time not in df:
        repeated = bool(df[entity].dropna().duplicated().any())
        return {"proposal": "Wiederholte Einheiten ohne Zeitzuordnung" if repeated else "Querschnittskandidat", "needs_confirmation": True}
    valid = df[[entity, time]].dropna()
    repeated = valid.groupby(entity)[time].nunique().gt(1).any()
    return {
        "proposal": "Panel / wiederholte Messungen" if repeated else ("Wiederholte Querschnitte (nur bei global stabilen IDs)" if valid[time].nunique() > 1 else "Querschnitt"),
        "duplicate_entity_time": int(valid.duplicated().sum()),
        "needs_confirmation": True,
    }


def demo_data(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    machine = rng.choice([f"M{i:02d}" for i in range(20)], n)
    typ = np.where(np.array([int(x[1:]) for x in machine]) % 2, "Typ B", "Typ A")
    temperature = rng.normal(22, 3, n)
    speed = rng.uniform(30, 100, n)
    counter = np.linspace(0, 60, n)
    deviation = .13 * (temperature - 22) + .018 * (speed - 65) + .45 * (typ == "Typ B") + .012 * counter + .0008 * counter**2 + .003 * (temperature - 22) * (speed - 65) + rng.normal(0, .55, n)
    frame = pd.DataFrame({"inspection_id": np.arange(1, n + 1), "machine_id": machine, "machine_type": typ,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h").astype(str), "temperature": temperature,
        "throughput": speed, "operating_hours": counter, "nominal": 500., "measured": 500 + deviation,
        "deviation": deviation, "lower": -2., "upper": 2., "violation": (abs(deviation) > 2).astype(int)})
    frame.loc[rng.choice(n, max(1, n // 20), replace=False), "temperature"] = np.nan
    return frame


def demo_pca_data(n: int = 600, seed: int = 83) -> pd.DataFrame:
    """Six correlated sensors driven by two independent latent process states."""
    rng = np.random.default_rng(seed)
    load, heat = rng.normal(size=(2, n))
    kind = rng.choice(["Typ A", "Typ B"], n)
    frame = pd.DataFrame({"machine_id": rng.choice([f"M{i:02d}" for i in range(20)], n),
                          "timestamp": pd.date_range("2025-01-01", periods=n, freq="h").astype(str), "machine_type": kind})
    for name, origin, factor in [("sensor_pressure", 50, load), ("sensor_flow", 30, .9*load+.1*heat),
                                  ("sensor_current", 12, 1.1*load), ("sensor_temperature", 22, heat),
                                  ("sensor_vibration", 5, .2*load+.8*heat), ("sensor_load", 70, .8*load+.2*heat)]:
        frame[name] = origin + factor + rng.normal(0, .08, n)
    deviation = .8*load + .7*heat + .3*(kind == "Typ B") + rng.normal(0,.3,n)
    frame["deviation"] = deviation
    frame["violation"] = (abs(deviation) > 1.5).astype(int)
    frame.loc[::23, "sensor_temperature"] = np.nan
    return frame
