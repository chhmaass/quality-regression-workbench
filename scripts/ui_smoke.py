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
            assert len(app.metric) == 2
            print("Linear UI:", [(m.label, m.value) for m in app.metric])
            next(r for r in app.radio if r.label == "Analyse").set_value("Binäres Target").run()
            next(b for b in app.button if b.label == "Analyse starten").click().run(timeout=60)
            assert not app.exception and not app.error
            assert len(app.metric) == 3
            print("Logistic UI:", [(m.label, m.value) for m in app.metric])
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
