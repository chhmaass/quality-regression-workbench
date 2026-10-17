"""Real HTTP backend + Streamlit AppTest for both user flows. No browser needed."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx
from streamlit.testing.v1 import AppTest


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    with tempfile.TemporaryDirectory() as temp:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.update(WORKBENCH_API_KEY="isolated-ui-test-key", DATABASE_URL="sqlite:///" + temp + "/ui.db",
                          API_BASE_URL=f"http://127.0.0.1:{port}")
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "qmworkbench.api:app", "--host", "127.0.0.1", "--port", str(port)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(80):
                try:
                    if httpx.get(os.environ["API_BASE_URL"] + "/health", trust_env=False).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError("API startup failed")
            app = AppTest.from_file("app.py", default_timeout=60).run()
            app.text_input(key="access_key").set_value("isolated-ui-test-key").run()
            next(b for b in app.button if b.label == "Synthetisches QM-Beispiel laden").click().run()
            assert not app.exception and not app.error
            next(b for b in app.button if b.label == "Analyse starten").click().run(timeout=60)
            assert not app.exception and not app.error
            app.switch_page("views/results.py").run()
            assert not app.exception and not app.error and len(app.metric) == 2
            print("Linear UI:", [(m.label, m.value) for m in app.metric])
            app.switch_page("views/prepare.py").run()
            next(r for r in app.radio if r.label == "Was möchten Sie untersuchen?").set_value("Toleranzverletzung vorhersagen").run()
            next(b for b in app.button if b.label == "Analyse starten").click().run(timeout=60)
            assert not app.exception and not app.error
            app.switch_page("views/results.py").run()
            assert not app.exception and not app.error and len(app.metric) == 3
            print("Logistic UI:", [(m.label, m.value) for m in app.metric])
            saved = app.session_state["run"]
            next(b for b in app.button if b.label == "Konfiguration bearbeiten").click().run()
            app.switch_page("views/prepare.py").run()
            assert app.text_input(key="access_key").value == "isolated-ui-test-key"
            assert app.radio(key="cfg_task_label").value == "Toleranzverletzung vorhersagen"
            assert app.multiselect(key="cfg_numeric").value == saved["result"]["config"]["numeric"]
            assert not any("geändert" in w.value for w in app.warning)
            app.slider(key="cfg_threshold").set_value(.7).run()
            assert any("geändert" in w.value for w in app.warning)
            next(b for b in app.button if b.label == "Bisheriges Ergebnis öffnen").click().run()
            app.switch_page("views/results.py").run()
            assert any("vorherigen Lauf" in w.value for w in app.warning)
            assert app.session_state["run"]["result"]["config"]["threshold"] == .5
            assert len(app.metric) == 3 and not app.exception and not app.error
            app.switch_page("views/prepare.py").run()
            assert app.slider(key="cfg_threshold").value == .7
            app.selectbox(key="cfg_mode_label").set_value("Aus Toleranzgrenzen").run()
            app.number_input(key="cfg_lower").set_value(3.).run()
            app.number_input(key="cfg_upper").set_value(-3.).run()
            assert next(b for b in app.button if b.label == "Analyse starten").disabled
            assert any("Untere Grenze" in e.value for e in app.error)

            # A new session opens an immutable saved run and restores its editable settings.
            reopened = AppTest.from_file("app.py", default_timeout=60).run()
            reopened.text_input(key="access_key").set_value("isolated-ui-test-key").run()
            reopened.text_input(key="open_run_id").set_value(saved["id"]).run()
            next(b for b in reopened.button if b.label == "Lauf laden").click().run()
            reopened.switch_page("views/results.py").run()
            assert not reopened.exception and not reopened.error and len(reopened.metric) == 3
            reopened.switch_page("views/prepare.py").run()
            assert reopened.slider(key="cfg_threshold").value == .5
            assert reopened.selectbox(key="cfg_target").value == "violation"
            assert not reopened.exception and not reopened.error
            assert not any("geändert" in w.value for w in reopened.warning)
            print("Navigation, retained settings, stale-results warning, blocked invalid input, saved-run restoration: OK")
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
