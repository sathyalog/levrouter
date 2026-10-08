"""lev-http client headers: Modal proxy-auth tokens are sent only when both are set."""

from tokentriage.classifiers.lev_http import LevHttpClassifier


def _headers(monkeypatch, key=None, secret=None):
    for name, value in (("TOKENTRIAGE_MODAL_KEY", key), ("TOKENTRIAGE_MODAL_SECRET", secret)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return LevHttpClassifier(base_url="https://x.modal.run", allow_remote=True)._client.headers


def test_modal_headers_sent_when_both_set(monkeypatch):
    h = _headers(monkeypatch, "wk-test", "ws-test")
    assert h["Modal-Key"] == "wk-test" and h["Modal-Secret"] == "ws-test"
    assert h["Authorization"].startswith("Bearer ")


def test_modal_headers_absent_by_default(monkeypatch):
    h = _headers(monkeypatch)
    assert "Modal-Key" not in h and "Modal-Secret" not in h


def test_modal_headers_need_both(monkeypatch):
    h = _headers(monkeypatch, key="wk-test")
    assert "Modal-Key" not in h
