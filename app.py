"""Entry point shared by the preparation and saved-results pages."""
import streamlit as st

st.set_page_config(page_title="QM-Regressionswerkstatt", page_icon="📊", layout="wide")
# Interrupt widget cleanup when preparation is hidden by page navigation.
for key in list(st.session_state):
    if key.startswith("cfg_") or key in {"access_key", "open_run_id"}:
        st.session_state[key] = st.session_state[key]
page = st.navigation([
    st.Page("views/prepare.py", title="Analyse vorbereiten", icon="🛠️", default=True),
    st.Page("views/results.py", title="Ergebnis und Bericht", icon="📊", url_path="results"),
    st.Page("views/saved.py", title="Gespeicherte Daten", icon="🗂️", url_path="saved"),
])
with st.sidebar:
    st.title("QM-Regressionswerkstatt")
    st.text_input("Zugangsschlüssel", type="password", key="access_key",
                  help="Gemeinsamer Zugang zum Arbeitsbereich. Keine getrennten Benutzerkonten.")
    st.caption("Portfolio-Demonstrator · synthetische Daten für die Vorführung verwenden.")

if not st.session_state.get("access_key"):
    st.info("Zugangsschlüssel in der Seitenleiste eingeben, um Daten und Analysen zu öffnen.")
    st.stop()
page.run()
