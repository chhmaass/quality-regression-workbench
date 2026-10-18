"""Browse saved datasets and runs, with explicit scoped deletion confirmation."""
import httpx
import pandas as pd
import streamlit as st

from .ui import forget_deleted, open_saved_run, request


def _delete_run(run_id):
    try:
        request("DELETE", f"/analyses/{run_id}")
        forget_deleted(run_id=run_id)
        st.session_state.saved_notice = "Analyselauf gelöscht. Der Datensatz bleibt erhalten."
    except (httpx.HTTPError, ValueError) as exc:
        st.session_state.saved_error = str(exc)


def _delete_dataset(dataset_id, count):
    try:
        request("DELETE", f"/datasets/{dataset_id}", params={"expected_runs": count})
        forget_deleted(dataset_id=dataset_id)
        st.session_state.saved_notice = "Datensatz und zugehörige Analyseläufe gelöscht."
    except (httpx.HTTPError, ValueError) as exc:
        st.session_state.saved_error = str(exc)


def render():
    st.header("Gespeicherte Daten")
    st.caption("Datensätze und Analyseläufe im gemeinsamen Arbeitsbereich. Löschungen sind dauerhaft.")
    if message := st.session_state.pop("saved_notice", None):
        st.success(message)
    if error := st.session_state.pop("saved_error", None):
        st.error(error)
    st.button("Übersicht aktualisieren")  # Each rerun reads fresh metadata, without caching.
    try:
        _render_lists()
    except (httpx.HTTPError, ValueError) as exc:
        st.error(str(exc))
        st.caption("Bei zwischenzeitlich gelöschten Einträgen die Übersicht aktualisieren.")


def _render_lists():
    page = int(st.number_input("Datensatzseite (100 pro Seite)", min_value=1, value=1, step=1, key="saved_dataset_page"))
    datasets = request("GET", "/datasets", params={"limit": 100, "offset": (page - 1) * 100}).json()
    st.subheader(f"Datensätze ({datasets['total']})")
    items = datasets["items"]
    if not items:
        st.info("Keine Datensätze auf dieser Seite. Bei einer höheren Seitennummer bitte Seite 1 wählen.")
        return
    st.dataframe(pd.DataFrame([{"Dateiname": d["name"], "Importdatum (UTC)": d["created_at"],
        "Zeilen": d["rows"], "Spalten": d["columns"], "Analyseläufe": d["run_count"], "Datensatz-ID": d["id"]}
        for d in items]), hide_index=True, use_container_width=True)
    by_id = {d["id"]: d for d in items}
    if st.session_state.get("saved_dataset") not in by_id:
        st.session_state.saved_dataset = None
    dataset_id = st.selectbox("Datensatz auswählen", list(by_id), index=None, placeholder="Datensatz auswählen",
        format_func=lambda id_: f"{by_id[id_]['name']} · {id_}", key="saved_dataset")
    if not dataset_id:
        return
    ds = by_id[dataset_id]
    run_page = int(st.number_input("Laufseite (100 pro Seite)", min_value=1, value=1, step=1, key=f"saved_run_page_{dataset_id}"))
    runs = request("GET", f"/datasets/{dataset_id}/analyses", params={"limit": 100, "offset": (run_page - 1) * 100}).json()
    st.subheader(f"Analyseläufe ({runs['total']})")
    run_by_id = {r["id"]: r for r in runs["items"]}
    task_labels = {"linear": "Lineare Regression", "logistic": "Logistische Regression"}
    if run_by_id:
        st.dataframe(pd.DataFrame([{"Datum (UTC)": r["created_at"], "Analyseart": task_labels.get(r["task"], r["task"]),
            "Target": r["target"], "Lauf-ID": r["id"]} for r in run_by_id.values()]), hide_index=True, use_container_width=True)
        run_key = f"saved_run_{dataset_id}"
        if st.session_state.get(run_key) not in run_by_id:
            st.session_state[run_key] = None
        run_id = st.selectbox("Analyselauf auswählen", list(run_by_id), index=None, placeholder="Analyselauf auswählen",
            format_func=lambda id_: f"{run_by_id[id_]['created_at']} · {task_labels.get(run_by_id[id_]['task'], '')} · {run_by_id[id_]['target']} · {id_}", key=run_key)
        if run_id:
            if st.button("Lauf öffnen"):
                open_saved_run(run_id)
                st.switch_page("views/results.py")
            with st.expander("Ausgewählten Lauf löschen"):
                confirmed = st.checkbox(f"Lauf {run_id} dauerhaft löschen. Der Datensatz bleibt erhalten.", key=f"confirm_run_{run_id}")
                st.button("Lauf endgültig löschen", disabled=not confirmed, on_click=_delete_run, args=(run_id,))
    else:
        st.info("Keine Analyseläufe auf dieser Seite.")
    with st.expander("Ausgewählten Datensatz löschen"):
        st.warning(f"„{ds['name']}“ ({dataset_id}): Der Datensatz und alle {runs['total']} zugehörigen Analyseläufe werden dauerhaft gelöscht, einschließlich gespeicherter Modelle und Ergebnisse.")
        confirmed = st.checkbox(f"Datensatz {dataset_id} mit {runs['total']} Analyseläufen dauerhaft löschen.",
            key=f"confirm_dataset_{dataset_id}_{runs['total']}")
        st.button("Datensatz und zugehörige Läufe endgültig löschen", disabled=not confirmed,
            on_click=_delete_dataset, args=(dataset_id, runs["total"]))
