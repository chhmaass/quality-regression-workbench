"""Guided preparation with validation before an analysis is started."""
import httpx
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from .schemas import AnalysisConfig
from .ui import (TASKS, METHODS, SPLITS, control as w, request, use_dataset, open_saved_run,
                 preview, describe_config, draft_changed)


def render():
    st.header("Analyse vorbereiten")
    st.caption("Daten laden → Ziel und Eingangsgrößen festlegen → Einstellungen prüfen → Analyse starten")
    st.subheader("1 · Daten laden und prüfen")
    left, right = st.columns([1, 2])
    with left:
        example = st.selectbox("Beispieldatensatz", ["QM-Prozess", "Korrelierte Sensoren (PCA)"])
        if st.button("Synthetisches QM-Beispiel laden", use_container_width=True):
            try:
                use_dataset(request("POST", "/datasets/demo-pca" if example == "Korrelierte Sensoren (PCA)" else "/datasets/demo").json())
                st.rerun()
            except (httpx.HTTPError, ValueError) as exc:
                st.error(str(exc))
    with right:
        file = st.file_uploader("CSV: UTF-8, Komma/Semikolon/Tab, Dezimalpunkt", type="csv")
        if file and st.button("CSV importieren"):
            try:
                use_dataset(request("POST", "/datasets", files={"file": (file.name, file.getvalue(), "text/csv")}).json())
                st.rerun()
            except (httpx.HTTPError, ValueError) as exc:
                st.error(str(exc))
    with st.expander("Gespeicherten Analyselauf öffnen"):
        run_id = st.text_input("Lauf-ID", key="open_run_id")
        if st.button("Lauf laden"):
            if not run_id.strip():
                st.error("Bitte eine Lauf-ID eingeben.")
            else:
                try:
                    open_saved_run(run_id.strip())
                    st.switch_page("views/results.py")
                except (httpx.HTTPError, ValueError) as exc:
                    st.error(str(exc))
    if "dataset" not in st.session_state:
        st.info("Zum Einstieg das synthetische QM-Beispiel laden oder eine eigene CSV importieren.")
        return
    ds = st.session_state.dataset
    profile = ds["profile"]
    cols = profile["columns"]
    st.write(f"**{ds['name']}** · {profile['rows']} Zeilen · {len(cols)} Spalten")
    with st.expander("Vorschau und Datenqualität"):
        st.dataframe(pd.DataFrame(profile["preview"]), hide_index=True)
        st.dataframe(pd.DataFrame({"Spalte": cols, "Typ": [profile["types"][x] for x in cols],
                                   "Fehlend": [profile["missing"][x] for x in cols],
                                   "Verschieden": [profile["unique"][x] for x in cols]}), hide_index=True)
        st.caption(f"Identische Zeilen: {profile['duplicate_rows']}; keine automatische Entfernung ohne fachlichen Schlüssel.")
    st.subheader("2 · Analyseziel und Eingangsgrößen")
    task = TASKS[w("radio", "task_label", "Was möchten Sie untersuchen?", list(TASKS), horizontal=True)]
    st.caption("Lineare Regression für einen numerischen Zielwert." if task == "linear" else "Logistische Regression für einen binären Fehlerstatus.")
    if st.session_state.get("last_task") not in (None, task):
        for name in ["mode_label", "target"]:
            st.session_state.pop("cfg_" + name, None)
    st.session_state.last_task = task
    modes = ["Vorhandene Spalte", "Istwert minus Sollwert"] if task == "linear" else ["Vorhandene Spalte", "Aus Toleranzgrenzen"]
    mode_label = w("selectbox", "mode_label", "Targetbildung", modes)
    mode = "column" if mode_label == modes[0] else ("deviation" if task == "linear" else "limits")
    if st.session_state.get("last_mode") not in (None, mode_label):
        st.session_state.pop("cfg_target", None)
    st.session_state.last_mode = mode_label
    target_default = "deviation" if task == "linear" or mode == "limits" else "violation"
    target = w("selectbox", "target", "Target / zu prüfender Messwert", cols, index=cols.index(target_default) if target_default in cols else 0)
    c = {"task": task, "target_mode": mode, "target": target}
    if mode == "column" and task == "logistic":
        c["target_positive"] = w("text_input", "target_positive", "Wert der positiven Klasse", "1", help="Zum Beispiel 1 oder fehlerhaft. Dieser Wert wird intern als 1 codiert.")
    if mode == "deviation":
        c["nominal"] = w("selectbox", "nominal", "Sollwertspalte", cols, index=cols.index("nominal") if "nominal" in cols else 0)
    if mode == "limits":
        st.info("Die Toleranzgrenzen definieren den tatsächlichen Fehlerstatus: 1 = verletzt, 0 = zulässig. Beispiel: Messwert measured mit Grenzen 498 und 502; Abweichung deviation mit Grenzen −2 und +2.")
        for side, label, default in [("lower", "Untere Grenze", -2.), ("upper", "Obere Grenze", 2.)]:
            kind = w("selectbox", side + "_kind", label + " aus", ["Konstante", "Spalte", "Nicht verwenden"])
            if kind == "Konstante":
                c[side] = w("number_input", side, label, value=default)
            elif kind == "Spalte":
                c[side + "_column"] = w("selectbox", side + "_column", label + " – Spalte", cols, index=cols.index(side) if side in cols else 0)
        c["boundary_inclusive"] = w("checkbox", "boundary_inclusive", "Wert genau auf der Grenze ist zulässig", value=True)
    eligible = [x for x in cols if x not in {target, c.get("nominal"), c.get("lower_column"), c.get("upper_column")}]
    c["numeric"] = w("multiselect", "numeric", "Numerische Features", eligible,
                      default=[x for x in ["temperature", "throughput", "operating_hours", "sensor_pressure", "sensor_flow", "sensor_current", "sensor_temperature", "sensor_vibration", "sensor_load"] if x in eligible])
    cats = [x for x in eligible if x not in c["numeric"]]
    c["categorical"] = w("multiselect", "categorical", "Kategoriale / binäre Features", cats,
                          default=["machine_type"] if "machine_type" in cats else [])
    st.caption("Nur Eingangsgrößen verwenden, die zum Vorhersagezeitpunkt bekannt sind. IDs dienen normalerweise der Zuordnung. Istwert, Abweichung und Fehlerstatus dürfen sich nicht gegenseitig vorwegnehmen.")
    suspicious = set(c["numeric"] + c["categorical"]) & {"measured", "deviation", "violation"}
    if suspicious:
        st.warning("Mögliche Vorwegnahme des Zielwerts: " + ", ".join(sorted(suspicious)) + ". Fachlich prüfen, ob diese Werte vor der Qualitätsentscheidung verfügbar sind.")
    st.subheader("3 · Bewertung festlegen")
    c["split"] = SPLITS[w("selectbox", "split_label", "Was soll die Testbewertung abbilden?", list(SPLITS))]
    if c["split"] == "random":
        st.caption("Zufällige Testmessungen. Bei wiederholten Messungen derselben Maschine keine Aussage über neue Maschinen oder zukünftige Zeiträume.")
    c["compare_elastic"] = w("checkbox", "compare_elastic", "Mit Elastic Net vergleichen", value=True, help="Regularisierung begrenzt die Modellkomplexität; die Stärke wird innerhalb der Trainingsdaten gewählt.")
    if task == "logistic":
        c["threshold"] = w("slider", "threshold", "Wahrscheinlichkeitsschwelle für positive Vorhersage", .01, .99, .5, .01)
        st.caption("Ab dieser geschätzten Wahrscheinlichkeit wird positiv vorhergesagt. Die Schwelle ändert keine Toleranzgrenzen und keine tatsächlichen Zielwerte. Vor der Testbewertung festlegen.")
    with st.expander("Erweiterte Einstellungen · Fehlwerte, Zeit, Interaktionen"):
        mode_label = w("selectbox", "feature_mode_label", "Feature Engineering", ["Direkte Features", "Hauptkomponenten (PCA)"])
        pca = mode_label == "Hauptkomponenten (PCA)"
        c["feature_mode"] = "pca" if pca else "direct"
        if pca:
            c["pca_components"] = w("number_input", "pca_components", "Anzahl Hauptkomponenten", 1, 25, 2)
            c["center_numeric"] = False
            st.info("PCA standardisiert numerische Trainingsfeatures und bildet daraus Komponenten. Kategorien bleiben separat. Keine Kombination mit MICE, Interaktionen oder Zeitpolynomen; zeitliche und gruppierte Tests bleiben verfügbar.")
        else:
            c["center_numeric"] = w("checkbox", "center_numeric", "Numerische Features vor Interaktionen zentrieren", value=True)
            st.caption("Trainingsmittel abziehen, dann Interaktionen bilden. Zeitcounter wird bereits um seinen Trainingsmittelwert zentriert; danach wird quadriert. Zentrierung allein garantiert keine höhere Modellgüte.")
        c["imputation"] = METHODS[w("selectbox", "imputation_label", "Fehlwertbehandlung", list(METHODS))]
        ent = w("selectbox", "entity", "Einheit-ID (Metadatum, keine Maschinen-FE)", ["Keine"] + cols)
        c["entity"] = None if ent == "Keine" else ent
        ts = w("selectbox", "time", "Zeit / Counter-Spalte", ["Keine"] + cols)
        c["time"] = None if ts == "Keine" else ts
        if c["time"]:
            c["time_source"] = "datetime" if w("radio", "time_format", "Zeitformat", ["Zeitstempel", "Numerischer Counter"], horizontal=True) == "Zeitstempel" else "numeric"
            c["time_feature"] = w("checkbox", "time_feature", "Zeitcounter als Feature", disabled=pca) and not pca
            square = w("checkbox", "time_squared", "Zusätzlich Zeitcounter²", disabled=not c["time_feature"])
            c["time_squared"] = square and c["time_feature"]
        features = c["numeric"] + c["categorical"] + (["__time"] if c.get("time_feature") else []) + (["__time2"] if c.get("time_squared") else [])
        pairs = [(x, y) for i, x in enumerate(features) for y in features[i+1:]]
        labels = {f"{x} × {y}": [x, y] for x, y in pairs}
        chosen = w("multiselect", "interactions", "Ausgewählte Interaktionen (maximal 12)", [] if pca else list(labels), max_selections=12, disabled=pca)
        c["interactions"] = [labels[x] for x in chosen]
        effect = w("selectbox", "effect_feature", "Optionales Einflussdiagramm", ["Keines"] + c["numeric"] + (["__time"] if c.get("time_feature") else []))
        if effect != "Keines":
            c["effect_feature"] = effect
            group = w("selectbox", "effect_group", "Diagramm nach Kategorie aufteilen", ["Keine"] + c["categorical"])
            if group != "Keine":
                c["effect_group"] = group
            else:
                modifier = w("selectbox", "effect_modifier", "Kurven nach numerischem Feature vergleichen", ["Keine"] + [x for x in c["numeric"] if x != effect])
                if modifier != "Keine":
                    c["effect_modifier"] = modifier
        c["test_fraction"] = w("slider", "test_fraction", "Anteil Testdaten", .15, .35, .2, .05)
        c["seed"] = w("number_input", "seed", "Reproduzierbarer Seed", 0, 2147483647, 42)
        if c["imputation"] == "mice":
            st.info("MICE: mindestens zwei numerische Features; keine Interaktionen, kein Zeitquadrat und keine Einheit-ID. Gepoolte Inferenz nur für additive numerische lineare Querschnittsmodelle.")
            c["imputations"] = w("slider", "imputations", "Anzahl Imputationen", 2, 10, 5)
            c["mice_iterations"] = w("slider", "mice_iterations", "Iterationen", 3, 30, 10)
            c["pooled_inference"] = w("checkbox", "pooled_inference", "Zusätzliche retrospektive MICE-Inferenz auf Trainingsfällen")
    st.subheader("4 · Prüfen und starten")
    error = None
    summary = None
    try:
        normalized = AnalysisConfig(**c).model_dump(mode="json")
        st.session_state.draft = {"digest": ds["digest"], "config": normalized}
        describe_config(normalized)
        summary = preview(ds, normalized)
        st.write(f"**Auswertbar: {summary['eligible']} von {summary['rows']} Zeilen** · {summary['n_train']} Training / {summary['n_test']} Test.")
        st.caption(f"Wegen fehlendem Zielwert ausgeschlossen: {summary['unknown_target']} Zeilen. Insgesamt ausgeschlossen (einschließlich gewählter Fehlwertbehandlung): {summary['excluded']}.")
        if task == "logistic":
            known = summary["positive"] + summary["negative"]
            st.write(f"**Target-Vorschau:** {summary['positive']} positive Fälle ({summary['positive']/known:.1%}) · {summary['negative']} negative Fälle. Bezug: alle Zeilen mit bekanntem Zielwert vor weiteren Ausschlüssen.")
    except ValidationError as exc:
        error = " · ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
        st.session_state.draft = {"digest": ds["digest"], "config": c}
    except (httpx.HTTPError, ValueError) as exc:
        error = str(exc)
    if error:
        st.error("Bitte vor dem Start korrigieren: " + error)
    if "run" in st.session_state:
        if draft_changed(st.session_state.run):
            st.warning("Einstellungen oder Datensatz geändert – bisherige Ergebnisse gelten für den vorherigen Lauf. Bitte die Analyse erneut starten.")
        if st.button("Bisheriges Ergebnis öffnen"):
            st.switch_page("views/results.py")
    if st.button("Analyse starten", type="primary", disabled=error is not None or summary is None):
        try:
            with st.spinner("Daten werden geprüft und Modelle geschätzt …"):
                run = request("POST", f"/datasets/{ds['id']}/analyses", json=normalized).json()
            st.session_state.run = run
            st.switch_page("views/results.py")
        except (httpx.HTTPError, ValueError) as exc:
            st.error(str(exc))
