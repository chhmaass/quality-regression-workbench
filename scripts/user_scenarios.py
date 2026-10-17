"""Five isolated user scenarios through Streamlit AppTest and a real HTTP API.

CSV import uses multipart HTTP, since AppTest does not support file_uploader.
No production credentials, data, or Railway services are changed.
"""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qmworkbench.data import demo_data


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def main():
    os.chdir(ROOT)
    results = []
    with tempfile.TemporaryDirectory() as temp:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        key = "isolated-scenario-test"
        os.environ.update(WORKBENCH_API_KEY=key, DATABASE_URL=f"sqlite:///{temp}/test.db", API_BASE_URL=base)
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "qmworkbench.api:app", "--port", str(port)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            with httpx.Client(base_url=base, headers={"X-API-Key": key}, trust_env=False, timeout=180) as client:
                for _ in range(100):
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.1)
                else:
                    raise AssertionError("API did not start")

                def new_app(frame=None):
                    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
                    app.text_input(key="access_key").set_value(key).run()
                    if frame is None:
                        widget(app, "button", "Synthetisches QM-Beispiel laden").click().run()
                    else:
                        response = client.post("/datasets", files={"file": ("scenario.csv", frame.to_csv(index=False, sep=";").encode(), "text/csv")})
                        assert response.status_code == 200, response.text
                        app.session_state["dataset"] = response.json()
                        app.run()
                    assert not app.exception and not app.error
                    return app

                def finish(app, name):
                    widget(app, "button", "Analyse starten").click().run(timeout=180)
                    assert not app.exception and not app.error, [e.value for e in app.error]
                    app.switch_page("views/results.py").run()
                    assert not app.exception and not app.error
                    run = app.session_state["run"]
                    r = run["result"]
                    assert set(r["train_source_rows"]).isdisjoint(r["test_source_rows"])
                    assert client.get(f"/analyses/{run['id']}").json()["result"] == r
                    report = client.get(f"/analyses/{run['id']}/report")
                    assert report.status_code == 200 and "<html" in report.text.lower()
                    for model_name, model in r["models"].items():
                        rows = pd.DataFrame(model["rows"])
                        exported = client.get(f"/analyses/{run['id']}/csv", params={"model": model_name})
                        assert exported.status_code == 200
                        assert len(pd.read_csv(io.BytesIO(exported.content))) == r["n_test"]
                        if r["config"]["task"] == "linear":
                            rmse = float(np.sqrt(np.mean((rows.observed - rows.prediction)**2)))
                            assert np.isclose(rmse, model["metrics"]["rmse"])
                        else:
                            predicted = (rows.prediction >= r["config"]["threshold"]).astype(int)
                            actual = rows.observed.astype(int)
                            cm = [[int(((actual == a) & (predicted == b)).sum()) for b in [0, 1]] for a in [0, 1]]
                            assert cm == model["metrics"]["confusion_matrix"]
                            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                            for metric, expected in {"accuracy": (tp+tn)/len(rows), "precision": tp/(tp+fp) if tp+fp else None,
                                                     "recall": tp/(tp+fn) if tp+fn else None}.items():
                                assert model["metrics"][metric] is None if expected is None else np.isclose(model["metrics"][metric], expected)
                    reopened = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
                    reopened.text_input(key="access_key").set_value(key).run()
                    reopened.text_input(key="open_run_id").set_value(run["id"]).run()
                    widget(reopened, "button", "Lauf laden").click().run()
                    reopened.switch_page("views/prepare.py").run()
                    assert not reopened.exception and not reopened.error
                    assert reopened.session_state["draft"]["config"] == r["config"], "Saved settings were not fully restored"
                    results.append({"scenario": name, "status": "PASS", "train": r["n_train"], "test": r["n_test"],
                                    "metrics": {k: v["metrics"] for k, v in r["models"].items()}})
                    return r

                app = new_app()
                r = finish(app, "1 Linear regression + machine type + Elastic Net")
                assert len(app.metric) == 2 and set(r["models"]) == {"reference", "elastic_net"}

                app = new_app()
                widget(app, "radio", "Was möchten Sie untersuchen?").set_value("Toleranzverletzung vorhersagen").run()
                widget(app, "selectbox", "Targetbildung").set_value("Aus Toleranzgrenzen").run()
                widget(app, "slider", "Wahrscheinlichkeitsschwelle für positive Vorhersage").set_value(.65).run()
                r = finish(app, "2 Logistic regression, limits -2/+2, prediction threshold 0.65")
                original = demo_data()
                for row in r["models"]["reference"]["rows"]:
                    assert row["observed"] == float(abs(original.iloc[row["source_row"]-2].deviation) > 2)
                assert len(app.metric) == 3

                frame = demo_data(240, seed=91)
                frame.loc[::7, "throughput"] = np.nan
                app = new_app(frame)
                widget(app, "multiselect", "Numerische Features").set_value(["temperature", "throughput"]).run()
                widget(app, "multiselect", "Kategoriale / binäre Features").set_value([]).run()
                widget(app, "selectbox", "Fehlwertbehandlung").set_value("MICE (mehrere Imputationen)").run()
                widget(app, "slider", "Anzahl Imputationen").set_value(3).run()
                widget(app, "checkbox", "Zusätzliche retrospektive MICE-Inferenz auf Trainingsfällen").check().run()
                r = finish(app, "3 Semicolon CSV, missing values, 3 MICE imputations + pooled inference")
                assert r["pooled_inference"] and r["n_train"] + r["n_test"] == len(frame)

                app = new_app()
                widget(app, "multiselect", "Numerische Features").set_value(["temperature", "throughput"]).run()
                widget(app, "selectbox", "Was soll die Testbewertung abbilden?").set_value("Zukünftige Zeiträume").run()
                widget(app, "selectbox", "Zeit / Counter-Spalte").set_value("timestamp").run()
                widget(app, "checkbox", "Zeitcounter als Feature").check().run()
                widget(app, "checkbox", "Zusätzlich Zeitcounter²").check().run()
                widget(app, "multiselect", "Ausgewählte Interaktionen (maximal 12)").set_value(["temperature × throughput", "machine_type × __time"]).run()
                r = finish(app, "4 Temporal holdout, time squared + two interactions")
                assert max(r["train_source_rows"]) < min(r["test_source_rows"])
                names = [x["feature"] for x in r["models"]["reference"]["coefficients"]]
                assert "__time2" in names and "temperature × throughput" in names
                widget(app, "checkbox", "Zeitverlauf statt Vorhersagevergleich").check().run()
                assert not app.exception and not app.error

                app = new_app()
                widget(app, "radio", "Was möchten Sie untersuchen?").set_value("Toleranzverletzung vorhersagen").run()
                widget(app, "selectbox", "Targetbildung").set_value("Aus Toleranzgrenzen").run()
                widget(app, "number_input", "Untere Grenze").set_value(3.).run()
                widget(app, "number_input", "Obere Grenze").set_value(-3.).run()
                assert widget(app, "button", "Analyse starten").disabled
                assert not app.exception and app.error and "Untere Grenze liegt über oberer Grenze" in app.error[0].value
                invalid_csv = client.post("/datasets", files={"file": ("duplicate.csv", b"a,a\n1,2\n", "text/csv")})
                assert invalid_csv.status_code == 422 and "eindeutig" in invalid_csv.text
                denied = client.post("/datasets/demo", headers={"X-API-Key": "invalid-test-key"})
                assert denied.status_code == 401
                widget(app, "number_input", "Untere Grenze").set_value(-2.).run()
                widget(app, "number_input", "Obere Grenze").set_value(2.).run()
                finish(app, "5 Invalid limits rejected, duplicate CSV rejected, wrong key rejected, corrected input works")
        finally:
            server.terminate()
            server.wait(timeout=10)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
