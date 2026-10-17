"""Saved analysis results, separate from editable preparation."""
import json
import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from .ui import request, describe_config, draft_changed, MODEL_LABELS, positive_label


def render():
    if "run" not in st.session_state:
        st.info("Noch kein Ergebnis vorhanden. Bitte zuerst eine Analyse vorbereiten oder dort einen gespeicherten Lauf öffnen.")
        if st.button("Zur Analysevorbereitung"):
            st.switch_page("views/prepare.py")
        return
    st.header("Ergebnis und Bericht")
    if st.button("Konfiguration bearbeiten"):
        st.switch_page("views/prepare.py")
    run = st.session_state.run
    r = run["result"]
    st.caption(f"Lauf {run['id']} · {r['n_train']} Trainings- / {r['n_test']} Testfälle · wegen fehlendem Zielwert ausgeschlossen: {r['unknown_target']} Zeilen")
    st.caption(f"Datensatz: {r['dataset_name']} · Ergebnisse beziehen sich ausschließlich auf die gespeicherte Konfiguration.")
    if draft_changed(run):
        st.warning("Einstellungen oder Datensatz geändert – diese Ergebnisse gehören zum vorherigen Lauf. Für aktuelle Ergebnisse die Analyse erneut starten.")
    with st.expander("Verwendete Analysekonfiguration", expanded=False):
        describe_config(r["config"])
    st.subheader("Modelle auf denselben Testfällen vergleichen")
    comparison = []
    for name, candidate in r["models"].items():
        comparison.append({"Modell": MODEL_LABELS.get(name, name), **candidate["metrics"]})
    comparison.append({"Modell": "Einfache Referenzvorhersage", **r["models"]["reference"]["baseline"]})
    comparison_columns = ["Modell", "r2", "rmse"] if r["config"]["task"] == "linear" else ["Modell", "accuracy", "precision", "recall"]
    st.dataframe(pd.DataFrame(comparison).reindex(columns=comparison_columns).rename(columns={"r2": "Test-R²", "rmse": "Test-RMSE", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall"}), hide_index=True)
    st.caption("Referenzvorhersage: Trainingsmittelwert bei Regression, häufigste Trainingsklasse bei Klassifikation. Höheres R² bzw. höhere Klassifikationskennzahlen und niedrigeres RMSE sind günstiger; betriebliche Eignung fachlich bewerten.")
    model_name = st.selectbox("Ergebnismodell", list(r["models"]), format_func=lambda x: MODEL_LABELS.get(x, x), key="result_model_" + run["id"])
    model = r["models"][model_name]
    metrics = model["metrics"]
    metric_names = ["r2", "rmse"] if r["config"]["task"] == "linear" else ["accuracy", "precision", "recall"]
    for col, key in zip(st.columns(len(metric_names)), metric_names):
        col.metric({"r2": "Test-R²", "rmse": "Test-RMSE", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall"}[key], "nicht bestimmbar" if metrics[key] is None else f"{metrics[key]:.3f}")
    rows = pd.DataFrame(model["rows"])
    if "confusion_matrix" in metrics:
        tn, fp = metrics["confusion_matrix"][0]
        fn, tp = metrics["confusion_matrix"][1]
        st.write(f"Von **{tp + fn} tatsächlichen positiven Fällen** wurden **{tp} erkannt** und **{fn} übersehen**. Bei **{fp} Fällen** wurde fälschlich positiv vorhergesagt (Fehlalarme).")
        st.caption("Accuracy: Anteil korrekter Entscheidungen. Precision: Anteil tatsächlicher positiver Fälle unter den positiven Vorhersagen. Recall: Anteil erkannter Fälle unter allen tatsächlichen positiven Fällen.")
        st.caption(f"Positive Klasse: {positive_label(r['config'])}. Wahrscheinlichkeitsschwelle: {r['config']['threshold']:.0%}.")
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
