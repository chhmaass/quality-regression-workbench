from types import SimpleNamespace

import httpx
import pytest

from qmworkbench import ui


def test_copied_key_whitespace_is_removed_before_building_headers(monkeypatch):
    monkeypatch.setattr(ui, "st", SimpleNamespace(session_state={"access_key": "  synthetic-test-key \n"}))
    sent = {}
    def send(method, url, **kwargs):
        sent.update(kwargs)
        return httpx.Response(200, json={"ok": True})
    monkeypatch.setattr(ui.httpx, "request", send)
    assert ui.request("GET", "/health").json() == {"ok": True}
    assert sent["headers"]["X-API-Key"] == "synthetic-test-key"


@pytest.mark.parametrize("token", [" ", "synthetic test key", "synthetic\nkey", "synthetic-ä-key"])
def test_invalid_keys_are_rejected_without_sending_or_echoing(monkeypatch, token):
    monkeypatch.setattr(ui, "st", SimpleNamespace(session_state={"access_key": token}))
    def send(*args, **kwargs):
        pytest.fail("Invalid credentials must not reach the HTTP transport")
    monkeypatch.setattr(ui.httpx, "request", send)
    with pytest.raises(ValueError) as error:
        ui.request("GET", "/health")
    if token.strip():
        assert token not in str(error.value)


@pytest.mark.parametrize("error", [httpx.LocalProtocolError, httpx.ConnectError, httpx.ReadTimeout])
def test_transport_error_does_not_disclose_credentials(monkeypatch, error):
    token = "synthetic-private-test-key"
    monkeypatch.setattr(ui, "st", SimpleNamespace(session_state={"access_key": token}))
    def send(*args, **kwargs):
        raise error("Illegal header value: " + token)
    monkeypatch.setattr(ui.httpx, "request", send)
    with pytest.raises(ValueError) as caught:
        ui.request("GET", "/health")
    assert token not in str(caught.value)
    assert caught.value.__suppress_context__


def test_backend_error_does_not_echo_the_key(monkeypatch):
    token = "synthetic-private-test-key"
    monkeypatch.setattr(ui, "st", SimpleNamespace(session_state={"access_key": token}))
    monkeypatch.setattr(ui.httpx, "request", lambda *args, **kwargs: httpx.Response(400, json={"detail": "Invalid " + token}))
    with pytest.raises(ValueError) as caught:
        ui.request("GET", "/health")
    assert token not in str(caught.value)
