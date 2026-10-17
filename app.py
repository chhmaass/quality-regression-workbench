"""Three-step German UI. Analysis and persistence belong to the HTTP backend."""
import json
import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="QM-Regressionswerkstatt", page_icon="📊", layout="wide")
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def request(method, path, **kwargs):
    response = httpx.request(method, API + path, headers={"X-API-Key": st.session_state.get("access_key", "")}, timeout=300, trust_env=False, **kwargs)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ValueError(str(detail))
    return response


def index_or_first(options, desired):
    return options.index(desired) if desired in options else 0


st.title("QM-Regressionswerkstatt")
st.caption("Qualitätsabweichungen erklären · Toleranzverletzungen vorhersagen")
with st.sidebar:
    st.text_input("Zugangsschlüssel", type="password", key="access_key", help="Entspricht WORKBENCH_API_KEY des Backends. Ein gemeinsamer Arbeitsbereich, keine Mehrmandantenlösung.")
    st.caption("Portfolio-Demonstrator. Für die öffentliche Vorführung synthetische Daten verwenden.")
    st.caption("v0.1 · FastAPI / PostgreSQL / scikit-learn / statsmodels")
if not st.session_state.get("access_key"):
    st.info("Zugangsschlüssel eingeben, um Beispieldaten oder eigene CSV-Dateien zu laden.")
    st.stop()

st.subheader("1 · Daten laden")
left, right = st.columns([1, 2])
with left:
    if st.button("Synthetisches QM-Beispiel laden", use_container_width=True):
        try:
            st.session_state.dataset = request("POST", "/datasets/demo").json()
            st.session_state.pop("run", None)
        except (httpx.HTTPError, ValueError) as exc:
            st.error(str(exc))
with right:
    file = st.file_uploader("CSV: UTF-8, Komma/Semikolon/Tab, Dezimalpunkt", type="csv")
    if file and st.button("CSV importieren"):
        try:
            st.session_state.dataset = request("POST", "/datasets", files={"file": (file.name, file.getvalue(), "text/csv")}).json()
            st.session_state.pop("run", None)
        except (httpx.HTTPError, ValueError) as exc:
            st.error(str(exc))

with st.expander("Gespeicherten Analyselauf wieder öffnen"):
    run_id = st.text_input("Lauf-ID")
    if st.button("Lauf laden") and run_id:
        try:
            st.session_state.run = request("GET", f"/analyses/{run_id}").json()
        except (httpx.HTTPError, ValueError) as exc:
            st.error(str(exc))

if "dataset" in st.session_state:
    ds = st.session_state.dataset
    profile = ds["profile"]
    cols = profile["columns"]
    st.write(f"**{ds['name']}** · {profile['rows']} Zeilen · {len(cols)} Spalten")
    with st.expander("Vorschau und Datenqualität"):
        st.dataframe(pd.DataFrame(profile["preview"]), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame({"Spalte": cols, "Typ": [profile["types"][x] for x in cols], "Fehlend": [profile["missing"][x] for x in cols], "Verschieden": [profile["unique"][x] for x in cols]}), hide_index=True)
        st.caption(f"Identische Zeilen: {profile['duplicate_rows']}; keine automatische Entfernung ohne fachlichen Schlüssel.")
    st.subheader("2 · Analyse einstellen")
    a, b = st.columns(2)
    with a:
        task_label = st.radio("Analyse", ["Numerisches Target", "Binäres Target"], horizontal=True)
        task = "linear" if task_label == "Numerisches Target" else "logistic"
        modes = ["Vorhandene Spalte", "Istwert minus Sollwert"] if task == "linear" else ["Vorhandene Spalte", "Aus Toleranzgrenzen"]
        mode_label = st.selectbox("Targetbildung", modes)
        mode = "column" if mode_label == modes[0] else ("deviation" if task == "linear" else "limits")
        target_default = "deviation" if task == "linear" or mode == "limits" else "violation"
        target = st.selectbox("Target / zu prüfender Messwert", cols, index=index_or_first(cols, target_default))
        config = {"task": task, "target_mode": mode, "target": target}
        if mode == "column" and task == "logistic":
            config["target_positive"] = st.text_input("Wert der positiven Klasse", "1")
        if mode == "deviation":
            config["nominal"] = st.selectbox("Sollwertspalte", cols, index=index_or_first(cols, "nominal"))
        if mode == "limits":
            for side, label, default in [("lower", "Untere Grenze", -2.), ("upper", "Obere Grenze", 2.)]:
                kind = st.selectbox(label + " aus", ["Konstante", "Spalte", "Nicht verwenden"], key=side + "_kind")
                if kind == "Konstante":
                    config[side] = st.number_input(label, value=default)
                elif kind == "Spalte":
                    config[side + "_column"] = st.selectbox(label + " – Spalte", cols, index=index_or_first(cols, side))
            config["boundary_inclusive"] = st.checkbox("Wert genau auf der Grenze ist zulässig", value=True)
        eligible = [x for x in cols if x not in {target, config.get("nominal"), config.get("lower_column"), config.get("upper_column")}]
        default_num = [x for x in ["temperature", "throughput", "operating_hours"] if x in eligible]
        config["numeric"] = st.multiselect("Numerische Features", eligible, default=default_num)
        config["categorical"] = st.multiselect("Kategoriale / binäre Features", [x for x in eligible if x not in config["numeric"]], default=["machine_type"] if "machine_type" in eligible and "machine_type" not in config["numeric"] else [])
    with b:
        splits = {"Unabhängiger Querschnitt": "random", "Neue Einheiten": "group", "Zukünftige Zeiträume": "time"}
        config["split"] = splits[st.selectbox("Was soll die Testbewertung abbilden?", list(splits))]
        options = ["Keine"] + cols
        ent = st.selectbox("Einheit-ID (Metadatum, keine Maschinen-FE)", options)
        config["entity"] = None if ent == "Keine" else ent
        ts = st.selectbox("Zeit / Counter-Spalte", options)
        config["time"] = None if ts == "Keine" else ts
        if config["time"]:
            config["time_source"] = "datetime" if st.radio("Zeitformat", ["Zeitstempel", "Numerischer Counter"], horizontal=True) == "Zeitstempel" else "numeric"
            config["time_feature"] = st.checkbox("Zeitcounter als Feature")
            config["time_squared"] = st.checkbox("Zusätzlich Zeitcounter²", disabled=not config["time_feature"])
        config["compare_elastic"] = st.checkbox("Mit Elastic Net vergleichen", value=True)
        if task == "logistic":
            config["threshold"] = st.slider("Wahrscheinlichkeitsschwelle für positive Vorhersage", .01, .99, .5, .01)
            st.caption("Vor der Testbewertung festlegen. Fachliche Toleranzgrenzen bestimmen dagegen das Target.")
        methods = {"Einfach (Median / fehlende Kategorie)": "simple", "Vollständige Fälle": "complete", "MICE (mehrere Imputationen)": "mice"}
        config["imputation"] = methods[st.selectbox("Fehlwertbehandlung", list(methods))]
    with st.expander("Interaktionen und weitere Einstellungen"):
        features = config["numeric"] + config["categorical"] + (["__time"] if config.get("time_feature") else []) + (["__time2"] if config.get("time_squared") else [])
        pairs = [(x, y) for i, x in enumerate(features) for y in features[i + 1:]]
        labels = {f"{x} × {y}": [x, y] for x, y in pairs}
        chosen = st.multiselect("Ausgewählte Interaktionen (maximal 12)", list(labels))
        config["interactions"] = [labels[x] for x in chosen]
        effect = st.selectbox("Optionales Einflussdiagramm", ["Keines"] + config["numeric"] + (["__time"] if config.get("time_feature") else []))
        if effect != "Keines":
            config["effect_feature"] = effect
            group = st.selectbox("Diagramm nach Kategorie aufteilen", ["Keine"] + config["categorical"])
            if group != "Keine":
                config["effect_group"] = group
        config["test_fraction"] = st.slider("Anteil Testdaten", .15, .35, .2, .05)
        config["seed"] = st.number_input("Reproduzierbarer Seed", 0, 2147483647, 42)
        if config["imputation"] == "mice":
            st.info("MICE v0.1: mindestens zwei numerische Features; keine Interaktionen, kein Zeitquadrat und keine Einheit-ID. Formal gepoolte Inferenz nur für den additiven numerischen linearen Querschnittspfad.")
            config["imputations"] = st.slider("Anzahl Imputationen", 2, 10, 5)
            config["mice_iterations"] = st.slider("Iterationen", 3, 30, 10)
            config["pooled_inference"] = st.checkbox("Zusätzliche retrospektive MICE-Inferenz auf Trainingsfällen")
    st.caption("Sicherstellen, dass Features zum Vorhersagezeitpunkt verfügbar sind. Messwert, abgeleiteter Fehlerstatus und Target nicht gegenseitig vorwegnehmen lassen.")
    if st.button("Analyse starten", type="primary"):
        try:
            with st.spinner("Backend prüft Daten und schätzt Modelle …"):
                st.session_state.run = request("POST", f"/datasets/{ds['id']}/analyses", json=config).json()
        except (httpx.HTTPError, ValueError) as exc:
            st.error(str(exc))

if "run" not in st.session_state:
    st.stop()
st.subheader("3 · Ergebnis und Bericht")
run = st.session_state.run
r = run["result"]
st.caption(f"Lauf {run['id']} · {r['n_train']} Trainings- / {r['n_test']} Testfälle · unbekanntes Target: {r['unknown_target']}")
st.info("Ergebnisse gehören zur gespeicherten Konfiguration dieses Laufs. Geänderte Einstellungen werden erst durch eine neue Analyse wirksam.")
model_name = st.selectbox("Ergebnismodell", list(r["models"]))
model = r["models"][model_name]
metrics = model["metrics"]
metric_names = ["r2", "rmse"] if r["config"]["task"] == "linear" else ["accuracy", "precision", "recall"]
for col, key in zip(st.columns(len(metric_names)), metric_names):
    col.metric({"r2": "Test-R²", "rmse": "Test-RMSE", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall"}[key], "nicht bestimmbar" if metrics[key] is None else f"{metrics[key]:.3f}")
rows = pd.DataFrame(model["rows"])
if "confusion_matrix" in metrics:
    st.write("**Konfusionsmatrix · positive Klasse = 1**")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["Beobachtet 0", "Beobachtet 1"], columns=["Vorhergesagt 0", "Vorhergesagt 1"]), use_container_width=True)
    fig = px.histogram(rows, x="prediction", color=rows["observed"].astype(int).astype(str), barmode="overlay", nbins=20,
                       labels={"prediction": "Geschätzte Verletzungswahrscheinlichkeit", "color": "Beobachtet"})
    fig.add_vline(x=r["config"]["threshold"], line_dash="dash")
else:
    show_time = rows["time"].notna().any() and st.checkbox("Zeitverlauf statt Vorhersagevergleich")
    if show_time:
        ordered = rows.copy()
        ordered["plot_time"] = pd.to_numeric(ordered["time"]) if r["config"]["time_source"] == "numeric" else pd.to_datetime(ordered["time"], utc=True)
        ordered = ordered.sort_values("plot_time")
        fig = px.scatter(ordered, x="plot_time", y=["observed", "prediction"], labels={"value": "Target", "plot_time": "Zeit / Counter"})
    else:
        fig = px.scatter(rows, x="observed", y="prediction", labels={"observed": "Beobachtet", "prediction": "Vorhergesagt"})
        bounds = [min(rows.observed.min(), rows.prediction.min()), max(rows.observed.max(), rows.prediction.max())]
        fig.add_scatter(x=bounds, y=bounds, mode="lines", name="Ideale Vorhersage", line={"dash": "dash"})
st.plotly_chart(fig, use_container_width=True)
if model.get("conditional_curve"):
    curve = model["conditional_curve"]
    st.caption(curve["note"])
    st.plotly_chart(px.line(pd.DataFrame(curve["rows"]), x="x", y="prediction", color="group", labels={"x": curve["feature"], "prediction": "Bedingte Modellvorhersage"}), use_container_width=True)
filtered = rows
if rows["group"].notna().any():
    selected = st.multiselect("Tabelle nach Einheit filtern", rows["group"].dropna().unique().tolist())
    if selected:
        filtered = rows[rows["group"].isin(selected)]
st.caption("Kennzahlen beziehen sich auf den gesamten Testteil; Tabellenfilter ändern sie nicht. source_row = logische CSV-Datenzeile einschließlich Kopfzeile.")
st.dataframe(filtered, use_container_width=True, hide_index=True)
with st.expander("Modell, Referenz, Diagnostik und Grenzen"):
    st.write(r["structure"])
    for warning in r["warnings"]:
        st.warning(warning)
    st.write("Referenzvorhersage (Trainingsmittelwert / häufigste Trainingsklasse)", model["baseline"])
    st.write("Gewählte Modellparameter", model["parameters"])
    st.caption(model["coefficient_scale"])
    st.dataframe(pd.DataFrame(model["coefficients"]), hide_index=True)
    st.write("Kategorienreferenzen", model["category_references"])
    if model.get("inference"):
        st.caption(model["inference"]["note"])
        if "coefficients" in model["inference"]:
            st.dataframe(pd.DataFrame(model["inference"]["coefficients"]), hide_index=True)
    if r["config"]["task"] == "linear":
        st.plotly_chart(px.scatter(rows, x="prediction", y="residual", labels={"prediction": "Vorhersage", "residual": "Residuum"}), use_container_width=True)
    if r.get("pooled_inference"):
        st.write(r["pooled_inference"]["scope"])
        st.dataframe(pd.DataFrame(r["pooled_inference"]["coefficients"]), hide_index=True)
    st.json(r["config"])
try:
    html = request("GET", f"/analyses/{run['id']}/report").content
    csv = request("GET", f"/analyses/{run['id']}/csv", params={"model": model_name}).content
    a, b, c = st.columns(3)
    a.download_button("HTML-Bericht", html, "qm-report.html", "text/html")
    b.download_button("Testergebnisse CSV", csv, "qm-results.csv", "text/csv")
    c.download_button("Konfiguration JSON", json.dumps(r["config"], ensure_ascii=False, indent=2), "qm-config.json", "application/json")
except (httpx.HTTPError, ValueError) as exc:
    st.error(str(exc))
