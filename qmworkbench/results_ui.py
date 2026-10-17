"""Saved analysis results, separate from editable preparation."""
import json
import httpx
import pandas as pd
import streamlit as st
from . import result_plots
from .report import safe_csv
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
        st.caption(f"Positive Klasse: {positive_label(r['config'])}. Wahrscheinlichkeitsschwelle: {r['config']['threshold']:.0%}. Accuracy: Anteil korrekt; Precision: Anteil tatsächlich positiv unter positiven Vorhersagen; Recall: Anteil erkannter positiver Fälle.")
    prediction_tab, effect_tab, detail_tab = st.tabs(["Vorhersagen", "Einflussgrößen", "Modelldetails"])
    with prediction_tab:
        result_plots.predictions(rows, metrics, r["config"])
        with st.expander("Einzelne Testfälle"):
            filtered = rows
            if rows["group"].notna().any():
                selected = st.multiselect("Tabelle nach Einheit filtern", rows["group"].dropna().unique().tolist())
                if selected:
                    filtered = rows[rows["group"].isin(selected)]
            st.caption("Kennzahlen beziehen sich auf alle Testfälle; Tabellenfilter ändern sie nicht. source_row = logische CSV-Datenzeile einschließlich Kopfzeile.")
            st.dataframe(filtered, use_container_width=True, hide_index=True)
    with effect_tab:
        result_plots.coefficients(model)
        result_plots.pca(model, r["config"]["task"])
        if model.get("design_coefficients"):
            design = model["design_coefficients"]
            st.subheader("Regressionskoeffizienten in Designeinheiten")
            st.caption(design["note"])
            st.dataframe(pd.DataFrame(design["rows"]), hide_index=True)
            st.download_button("Koeffizienten CSV", safe_csv(design["rows"]), "qm-coefficients.csv", "text/csv", key="coefficients_csv")
        else:
            st.caption("Älterer Lauf: unskalierte Designeinheiten und Aufbereitungsmetadaten wurden noch nicht gespeichert. Standardisierte Koeffizienten sind unter Modelldetails verfügbar.")
    with detail_tab:
        st.subheader("Standardisierte Koeffizienten und Intercept")
        st.caption(model["coefficient_scale"])
        st.dataframe(pd.DataFrame([{"feature":"Intercept", "coefficient":model["intercept"]}] + model["coefficients"]), hide_index=True)
        st.write("Kategorienreferenzen", model["category_references"])
        if model.get("preprocessing"):
            prep = model["preprocessing"]
            st.caption(prep["note"])
            first = prep["first_chain"]
            if first.get("mode") == "pca":
                st.write("PCA-Eingangsstandardisierung: Trainingsmittel und Standardabweichungen", {"Mittel":first["pca"]["input_means"], "Standardabweichung":first["pca"]["input_scales"]})
            else:
                st.write("Abgezogene numerische Trainingsbezugswerte", first["numeric_centers"])
            if first.get("time_center") is not None:
                st.write("Zeitbezug: Trainingsmittel des Counters (Zeitstempel in Tagen seit Unix-Epoche)", first["time_center"])
            st.dataframe(pd.DataFrame(first["design_scaling"]), hide_index=True)
            with st.expander("Vollständige Aufbereitungsmetadaten aller Modelle/Imputationen"):
                st.json(prep)
        if model.get("inference"):
            st.subheader("Trainingsinferenz · standardisierte Designmatrix")
            st.caption(model["inference"]["note"])
            if "coefficients" in model["inference"]:
                st.dataframe(pd.DataFrame(model["inference"]["coefficients"]), hide_index=True)
                st.caption("Intervalle: 95 %. Nicht mit der Tabelle in unskalierten Designeinheiten vermischen.")
        else:
            st.info("Keine klassische Koeffizienteninferenz für diesen Modellpfad. Regularisierung, PCA oder fehlende Werte können gewöhnliche Intervalle und p-Werte ungeeignet machen.")
        if r.get("pooled_inference"):
            st.write(r["pooled_inference"]["scope"])
            st.caption("Separater retrospektiver MICE-Pfad auf ursprünglichen numerischen Einheiten, keine Intervalle des Vorhersageensembles.")
            st.dataframe(pd.DataFrame(r["pooled_inference"]["coefficients"]), hide_index=True)
        st.write("Gewählte Modellparameter", model["parameters"])
        st.write("Erkannte Datenstruktur (Vorschlag)", r["structure"])
        for warning in r["warnings"]:
            st.warning(warning)
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
