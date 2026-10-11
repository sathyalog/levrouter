import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from levrouter import RouterConfig
from levrouter.classifiers.lev_http import LevHttpClassifier, UnsafeLevEndpoint
from levrouter.features import extract
from levrouter.pii import find, redact
from levrouter.router import build_classifier


@pytest.mark.parametrize(
    "text, token",
    [
        ("mail jane.doe+x@example.co.uk now", "[EMAIL]"),
        ("call +1 415-555-0132", "[PHONE]"),
        ("card 4111 1111 1111 1111", "[CARD]"),
        ("ssn 123-45-6789", "[SSN]"),
        ("aadhaar 2345 6789 0123", "[AADHAAR]"),
        ("pan ABCDE1234F", "[PAN]"),
        ("iban DE89 3704 0044 0532 0130 00", "[IBAN]"),
        ("host 10.0.0.12", "[IP]"),
        ("key sk-ant-api03-abcdefghijklmnopqrstuv", "[SECRET]"),
        ("key sk-proj-abcdefghijklmnopqrstuvwxyz12", "[SECRET]"),
        ("gsk_abcdefghijklmnopqrstuvwx", "[SECRET]"),
        ("hf_abcdefghijklmnopqrstuvwx", "[SECRET]"),
        ("AIzaSyA1234567890abcdefghijklmnopqrstu", "[SECRET]"),
        ("AKIAABCDEFGHIJKLMNOP", "[SECRET]"),
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456", "Bearer [SECRET]"),
        ("password=hunter2hunter2", "password=[SECRET]"),
        ("jwt eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4", "[JWT]"),
    ],
)
def test_redacts(text, token):
    assert token in redact(text)


def test_leaves_ordinary_text_alone():
    text = "Explain the 1990s economy in 3 paragraphs; version 3.14.1 of the API; 2+2=4"
    assert redact(text) == text and find(text) == []


def test_invalid_card_number_is_not_flagged_as_card():
    assert "[CARD]" not in redact("4111 1111 1111 1112")


def test_disabled():
    assert redact("a@b.com", enabled=False) == "a@b.com"


# -- lev-http endpoint guard ---------------------------------------------------


@pytest.mark.parametrize("url", [
    "http://localhost:8000", "http://127.0.0.1:8000", "http://10.1.2.3:8000", "http://lev.internal:8000",
    # Docker Compose and Kubernetes service names
    "http://lev:8000", "http://lev-service:8000", "http://lev_server:8000",
    "http://lev-service.default.svc:8000", "http://lev-service.default.svc.cluster.local:8000",
])
def test_local_lev_urls_allowed(url):
    LevHttpClassifier(url)


@pytest.mark.parametrize("url", [
    "http://lev.example.com:8000", "http://example.com", "http://8.8.8.8:8000",
    # dotless numbers resolve to public addresses (134744072 is 8.8.8.8)
    "http://134744072:8000", "http://0x08080808:8000",
])
def test_public_looking_lev_urls_are_refused(url):
    with pytest.raises(UnsafeLevEndpoint):
        LevHttpClassifier(url)


def test_empty_host_is_refused():
    with pytest.raises(UnsafeLevEndpoint):
        LevHttpClassifier("http:///v1")


def test_remote_lev_url_needs_opt_in_and_https():
    with pytest.raises(UnsafeLevEndpoint, match="not local"):
        LevHttpClassifier("https://lev.example.com")
    with pytest.raises(UnsafeLevEndpoint, match="https"):
        LevHttpClassifier("http://lev.example.com", allow_remote=True)
    LevHttpClassifier("https://lev.example.com", allow_remote=True)


def test_router_refuses_remote_lev_at_startup():
    with pytest.raises(UnsafeLevEndpoint):
        build_classifier(RouterConfig(backend="lev-http", lev_url="https://lev.example.com"))


def test_state_sent_to_lev_http_is_redacted(monkeypatch):
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["body"] = json.loads(request.content)
        sent["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"model": "lev", "usage": {"input_tokens": 1}, "answers": {
            "tier": {"type": "choice", "choice": "simple", "probabilities": {"simple": 0.9, "standard": 0.05, "complex": 0.05}, "confidence": 0.9},
            "needs_reasoning": {"type": "noul", "noul": 0.1},
            "needs_code": {"type": "noul", "noul": 0.0},
        }})

    monkeypatch.setenv("LEVROUTER_LEV_API_KEY", "lev-token-123")
    clf = LevHttpClassifier("http://localhost:8000")
    clf._client = httpx.Client(base_url="http://localhost:8000", transport=httpx.MockTransport(handler),
                               headers=clf._client.headers)
    f = extract([HumanMessage("Refund order for jane@acme.com, card 4111 1111 1111 1111")])
    clf.classify(f, f.state())
    assert "jane@acme.com" not in sent["body"]["state"] and "[EMAIL]" in sent["body"]["state"]
    assert "[CARD]" in sent["body"]["state"]
    assert sent["auth"] == "Bearer lev-token-123"


def test_lev_api_key_not_in_config_repr():
    assert "s3cret" not in repr(RouterConfig(lev_api_key="s3cret"))
