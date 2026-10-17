"""Result plots built exclusively from saved test predictions and training metadata."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def predictions(rows, metrics, config):
    if config["task"] == "linear":
        fig = px.scatter(rows, x="observed", y="prediction", hover_data=["source_row"],
                         labels={"observed": "Beobachteter Zielwert", "prediction": "Vorhergesagter Zielwert"}, title="Beobachtet und vorhergesagt · Testdaten")
        bounds = [min(rows.observed.min(), rows.prediction.min()), max(rows.observed.max(), rows.prediction.max())]
        fig.add_scatter(x=bounds, y=bounds, mode="lines", name="Ideale Vorhersage", line={"dash": "dash"})
        st.plotly_chart(fig, use_container_width=True)
        residual = px.scatter(rows, x="prediction", y="residual", hover_data=["source_row"], title="Residuen · Testdaten",
                              labels={"prediction": "Vorhersage", "residual": "Beobachtet minus vorhergesagt"})
        residual.add_hline(y=0, line_dash="dash")
        st.plotly_chart(residual, use_container_width=True)
    else:
        values = rows.copy()
        values["Tatsächliche Klasse"] = values.observed.astype(int).astype(str)
        values["Entscheidung"] = np.where(values.observed.astype(bool) == values.predicted_positive.astype(bool), "Korrekt", "Fehlentscheidung")
        fig = px.histogram(values, x="prediction", color="Tatsächliche Klasse", barmode="overlay", nbins=20,
                           title="Vorhergesagte Wahrscheinlichkeit nach tatsächlicher Klasse",
                           labels={"prediction": "Geschätzte Wahrscheinlichkeit für Klasse 1"})
        fig.add_vline(x=config["threshold"], line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(values, x="source_row", y="prediction", color="Entscheidung", symbol="Tatsächliche Klasse",
                         title="Einzelne Testvorhersagen und Fehlentscheidungen",
                         labels={"source_row": "CSV-Datenzeile", "prediction": "Geschätzte Wahrscheinlichkeit"})
        fig.add_hline(y=config["threshold"], line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        matrix = px.imshow(metrics["confusion_matrix"], text_auto=True, x=["Vorhergesagt 0", "Vorhergesagt 1"],
                           y=["Beobachtet 0", "Beobachtet 1"], color_continuous_scale="Blues", title="Konfusionsmatrix · Testdaten")
        matrix.update_layout(coloraxis_showscale=False)
        st.plotly_chart(matrix, use_container_width=True)
    if rows["time"].notna().any() and st.checkbox("Zeitverlauf anzeigen", value=True):
        ordered = rows.copy()
        ordered["Zeit"] = pd.to_numeric(ordered.time) if config["time_source"] == "numeric" else pd.to_datetime(ordered.time, utc=True)
        ordered = ordered.sort_values("Zeit")
        fig = go.Figure()
        fig.add_scatter(x=ordered.Zeit, y=ordered.prediction, mode="markers+lines", name="Vorhersage" if config["task"] == "linear" else "Geschätzte Wahrscheinlichkeit")
        fig.add_scatter(x=ordered.Zeit, y=ordered.observed, mode="markers", name="Tatsächlicher Zielwert")
        if config["task"] == "logistic":
            fig.add_hline(y=config["threshold"], line_dash="dash")
        fig.update_layout(title="Zeitverlauf · ausschließlich Testdaten", xaxis_title="Zeit / Counter", yaxis_title="Zielwert / Vorhersage")
        st.plotly_chart(fig, use_container_width=True)


def coefficients(model):
    frame = pd.DataFrame(model["coefficients"])
    displayed = frame.assign(magnitude=frame.coefficient.abs()).nlargest(25, "magnitude").sort_values("coefficient")
    displayed["Vorzeichen"] = np.where(displayed.coefficient >= 0, "Positiv", "Negativ")
    fig = px.bar(displayed, x="coefficient", y="feature", color="Vorzeichen", orientation="h",
                 title="Standardisierte Regressionskoeffizienten · maximal 25 Terme",
                 labels={"coefficient": "Koeffizient pro Standardabweichung der Designspalte", "feature": "Modellterm"})
    fig.add_vline(x=0, line_color="gray")
    fig.update_layout(height=max(350, len(displayed)*24))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(model["coefficient_scale"] + " Keine allgemeine Feature-Importance oder Kausalwirkung. Bei korrelierten Features und Interaktionen nur gemeinsam interpretieren.")
    if model.get("conditional_curve"):
        curve = model["conditional_curve"]
        st.caption(curve["note"])
        st.plotly_chart(px.line(pd.DataFrame(curve["rows"]), x="x", y="prediction", color="group",
                               title="Bedingte Modellvorhersage", labels={"x":curve["feature"], "prediction":"Modellvorhersage"}), use_container_width=True)
    else:
        st.caption("Einflusskurven können in der Vorbereitung unter erweiterten Einstellungen angefordert werden.")


def pca(model, task):
    info = model.get("preprocessing", {}).get("first_chain", {}).get("pca")
    if not info:
        return
    st.subheader("Hauptkomponenten")
    st.caption(info["note"])
    variance = pd.DataFrame(info["variance"])
    fig = go.Figure()
    fig.add_bar(x=variance.component, y=variance.explained, name="Erklärte Varianz")
    fig.add_scatter(x=variance.component, y=variance.cumulative, mode="lines+markers", name="Kumulierte Varianz")
    fig.update_layout(title="Erklärte numerische Trainingsvarianz", yaxis={"range":[0,1.05], "tickformat":".0%"}, xaxis_title="Komponente")
    st.plotly_chart(fig, use_container_width=True)
    weights = pd.DataFrame(info["weights"]).set_index("feature")
    st.plotly_chart(px.imshow(weights, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Komponentengewichte standardisierter Eingangsgrößen", aspect="auto"), use_container_width=True)
    st.dataframe(weights, use_container_width=True)
    scores = pd.DataFrame(model.get("pca_scores") or [])
    if "PC2" in scores and scores.PC2.notna().any():
        if task == "logistic":
            scores["observed"] = scores.observed.astype(int).astype(str)
        st.plotly_chart(px.scatter(scores, x="PC1", y="PC2", color="observed",
                                  title="Testfälle im Komponentenraum · maximal 2.000 Fälle", labels={"observed":"Tatsächlicher Zielwert"}), use_container_width=True)
