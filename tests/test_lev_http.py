"""lev-http client headers: configured as {header: ENV VAR NAME}, values read from the environment."""

import logging

from levrouter.classifiers.lev_http import LevHttpClassifier


def _headers(monkeypatch, env=None, header_env=None):
    for name in ("MODAL_KEY", "MODAL_SECRET", "OTHER_SECRET", "LEVROUTER_LEV_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    return LevHttpClassifier(base_url="https://x.example.com", allow_remote=True, headers=header_env)._client.headers


def test_configured_headers_are_read_from_the_named_variables(monkeypatch):
    h = _headers(monkeypatch, {"MODAL_KEY": "wk-test", "MODAL_SECRET": "ws-test"},
                 {"Modal-Key": "MODAL_KEY", "Modal-Secret": "MODAL_SECRET"})
    assert h["Modal-Key"] == "wk-test" and h["Modal-Secret"] == "ws-test"
    assert h["Authorization"] == "Bearer local"


def test_no_extra_headers_by_default(monkeypatch):
    h = _headers(monkeypatch)
    assert "Modal-Key" not in h and h["Authorization"] == "Bearer local"


def test_bearer_key_still_works(monkeypatch):
    assert _headers(monkeypatch, {"LEVROUTER_LEV_API_KEY": "hf_abc"})["Authorization"] == "Bearer hf_abc"


def test_authorization_entry_replaces_the_bearer_header(monkeypatch):
    h = _headers(monkeypatch, {"OTHER_SECRET": "Basic abc"}, {"authorization": "OTHER_SECRET"})
    assert h["authorization"] == "Basic abc"
    assert [k for k in h if k.lower() == "authorization"] == ["authorization"]


def test_unset_variable_is_skipped_with_a_warning_naming_the_variable_only(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="levrouter"):
        h = _headers(monkeypatch, {"MODAL_KEY": "wk-secret-value"}, {"Modal-Key": "MODAL_KEY", "Modal-Secret": "MODAL_SECRET"})
    assert "Modal-Key" in h and "Modal-Secret" not in h
    assert "MODAL_SECRET" in caplog.text and "wk-secret-value" not in caplog.text


def test_secret_values_are_not_in_the_classifier_repr(monkeypatch):
    monkeypatch.setenv("MODAL_KEY", "wk-secret-value")
    c = LevHttpClassifier(base_url="https://x.example.com", allow_remote=True, headers={"Modal-Key": "MODAL_KEY"})
    assert "wk-secret-value" not in repr(c)
