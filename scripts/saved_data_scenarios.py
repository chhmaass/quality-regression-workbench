"""Management scenarios against the isolated real backend used by ui_smoke."""
import httpx


def exercise(app, base):
    def button(label):
        return next(b for b in app.button if b.label == label)

    def healthy():
        assert not app.exception and not app.error, [e.value for e in app.error]

    with httpx.Client(base_url=base, headers={"X-API-Key": "isolated-ui-test-key"}, trust_env=False) as client:
        saved = app.session_state["run"]
        ds_id = saved["dataset_id"]
        other = client.post("/datasets/demo-pca").json()
        app.switch_page("views/saved.py").run()
        healthy()
        assert app.selectbox(key="saved_dataset").value is None
        app.selectbox(key="saved_dataset").set_value(ds_id).run()
        assert button("Datensatz und zugehörige Läufe endgültig löschen").disabled
        app.selectbox(key=f"saved_run_{ds_id}").set_value(saved["id"]).run()
        assert button("Lauf endgültig löschen").disabled
        button("Lauf öffnen").click().run()
        app.switch_page("views/results.py").run()
        healthy()
        assert len(app.metric) == 3
        app.switch_page("views/prepare.py").run()
        assert app.selectbox(key="cfg_target").value == "violation"
        app.switch_page("views/saved.py").run()
        app.selectbox(key="saved_dataset").set_value(ds_id).run()
        app.selectbox(key=f"saved_run_{ds_id}").set_value(saved["id"]).run()
        app.checkbox(key=f"confirm_run_{saved['id']}").check().run()
        button("Lauf endgültig löschen").click().run()
        healthy()
        assert "run" not in app.session_state
        assert app.session_state['dataset']['id'] == ds_id
        assert client.get(f"/analyses/{saved['id']}").status_code == 404
        assert client.get(f"/datasets/{ds_id}").status_code == 200
        remaining = client.get(f"/datasets/{ds_id}/analyses").json()['total']
        app.checkbox(key=f"confirm_dataset_{ds_id}_{remaining}").check().run()
        app.selectbox(key="saved_dataset").set_value(other['id']).run()
        assert button("Datensatz und zugehörige Läufe endgültig löschen").disabled
        app.selectbox(key="saved_dataset").set_value(ds_id).run()
        app.checkbox(key=f"confirm_dataset_{ds_id}_{remaining}").check().run()
        button("Datensatz und zugehörige Läufe endgültig löschen").click().run()
        healthy()
        assert "dataset" not in app.session_state and "run" not in app.session_state
        assert client.get(f"/datasets/{other['id']}").status_code == 200
        assert client.get(f"/datasets/{ds_id}").status_code == 404
        app.switch_page("views/results.py").run()
        assert len(app.metric) == 0 and not app.exception
        app.switch_page("views/prepare.py").run()
        button("Synthetisches QM-Beispiel laden").click().run()
        button("Analyse starten").click().run(timeout=60)
        app.switch_page("views/results.py").run()
        healthy()
        assert len(app.metric) == 2
        # All remaining data is isolated test data, not the user's database.
        for item in client.get('/datasets').json()['items']:
            assert client.delete(f"/datasets/{item['id']}", params={'expected_runs': item['run_count']}).status_code == 200
        app.switch_page("views/saved.py").run()
        healthy()
        assert any("Keine Datensätze" in i.value for i in app.info)
        print("Saved data: list, open/restore, confirmation isolation, scoped run deletion, dataset cascade, state cleanup, reanalysis and empty overview: OK")
