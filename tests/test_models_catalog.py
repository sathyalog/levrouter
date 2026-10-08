"""Keeping model data current: catalogue fallback, `tokentriage models refresh`, startup tier check."""

import io
import json
import urllib.request

import pytest

from tokentriage import providers
from tokentriage.__main__ import MODEL_LISTS, main
from tokentriage.config_loader import load_config


@pytest.fixture(autouse=True)
def fresh_catalog():
    providers.openrouter_catalog.cache_clear()
    yield
    providers.openrouter_catalog.cache_clear()


def test_direct_provider_falls_back_to_the_catalogue():
    spec = providers.spec_for("anthropic", "claude-opus-4-8")  # in the OpenRouter catalogue, not the built-in list
    assert spec is not None and spec.id == "claude-opus-4-8" and spec.input_price > 0 and spec.context > 0
    assert providers.spec_for("anthropic", "claude-opus-4-8-20260101") is not None  # dated snapshot
    assert providers.spec_for("anthropic", "gpt-4.1") is None  # never borrows another vendor's model


def test_models_refresh_writes_catalogue_and_available_models(monkeypatch, capsys):
    for env, _, _ in MODEL_LISTS.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    pages = {
        "openrouter.ai": {"data": [{"id": "anthropic/claude-sonnet-6", "pricing": {"prompt": "0.000002", "completion": "0.00001"},
                                    "context_length": 1000000, "architecture": {"input_modalities": ["text"]},
                                    "supported_parameters": ["tools"]}]},
        "api.anthropic.com": {"data": [{"id": "claude-sonnet-6"}, {"id": "claude-haiku-4-5-20251001"}], "has_more": False},
    }
    seen_headers = []

    def fake_urlopen(request, timeout=None):
        seen_headers.append(dict(request.header_items()))
        host = urllib.request.urlparse(request.full_url).hostname
        return io.BytesIO(json.dumps(pages[host]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert main(["models", "refresh"]) == 0

    saved = json.loads(providers.available_models_path().read_text())
    assert saved["providers"] == {"anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-6"]}
    assert "anthropic/claude-sonnet-6" in providers.openrouter_catalog()
    out = capsys.readouterr()
    assert "openai     skipped (OPENAI_API_KEY not set)" in out.out
    assert "sk-ant-secret-value" not in out.out + out.err
    assert any(h.get("X-api-key") == "sk-ant-secret-value" for h in seen_headers)


def test_startup_warns_about_tier_models_the_key_cannot_use(tmp_path, capsys):
    path = providers.available_models_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"providers": {"anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-5-5",
                                                            "claude-opus-5-5"]}}))
    yaml_file = tmp_path / "tokentriage.yaml"
    yaml_file.write_text("router:\n  backend: heuristic\n")
    load_config(str(yaml_file), auto_enable=False)
    out = capsys.readouterr().err
    assert "standard tier uses 'claude-sonnet-5'" in out and "newest similar: 'claude-sonnet-5-5'" in out
    assert "simple tier" not in out and "complex tier" not in out  # dated id and exact id both count as available
