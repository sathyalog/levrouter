"""lev behind `lev serve` (or any TypeSafe /v1/systemone server).

The state sent here contains the user's prompt, so the server is restricted to this
machine or a private network unless allow_remote_lev is set, and then https is required.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import time

from .. import pii
from ..features import RequestFeatures
from ..providers import host_of
from .base import LEV_QUESTIONS, Signals, signals_from_answers


class UnsafeLevEndpoint(ValueError):
    """The lev-http URL would send prompts somewhere this config does not allow."""


log = logging.getLogger("tokentriage")

# A name with no dot, such as a Docker Compose service ("lev") or a Kubernetes one ("lev-service").
_SERVICE_NAME = re.compile(r"[a-z][a-z0-9_-]*")


def _is_private(host: str) -> bool:
    """Judged by the name alone (no DNS lookup): local names, private addresses, and service names."""
    if host in ("localhost",) or host.endswith((".local", ".internal", ".localhost", ".svc")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Public hosts always contain a dot, so a dotless name can only resolve through the local or cluster DNS.
        # It must start with a letter: numeric forms like "134744072" or "0x08080808" resolve to public addresses.
        return bool(_SERVICE_NAME.fullmatch(host))
    return ip.is_loopback or ip.is_private or ip.is_link_local


def check_endpoint(url: str, allow_remote: bool) -> None:
    host = host_of(url) or ""
    if _is_private(host):
        return
    if not allow_remote:
        raise UnsafeLevEndpoint(
            f"lev_url host {host!r} is not local/private; prompts would leave your network. "
            "Set allow_remote_lev=True if this server is yours."
        )
    if not url.lower().startswith("https://"):
        raise UnsafeLevEndpoint(f"remote lev_url must use https, got {url!r}")


def _request_headers(token: str, header_env: dict[str, str] | None) -> dict[str, str]:
    """Bearer token plus the configured headers, whose values come from environment variables.

    `header_env` maps a header name to the NAME of an environment variable, so no secret is ever in a config
    file. A variable that is not set is skipped with a warning that names the variable, never a value.
    A `lev_headers` entry called Authorization replaces the Bearer header.
    """
    headers = {"Authorization": f"Bearer {token}"}
    for header, var in (header_env or {}).items():
        value = os.environ.get(var)
        if not value:
            log.warning("tokentriage: lev_headers: environment variable %s is not set, so header %s is not sent", var, header)
            continue
        for existing in [h for h in headers if h.lower() == header.lower()]:
            del headers[existing]
        headers[header] = value
    return headers


class LevHttpClassifier:
    name = "lev-http"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_s: float = 1.5,
        api_key: str | None = None,
        allow_remote: bool = False,
        redact_input: bool = True,
        headers: dict[str, str] | None = None,
    ):
        import httpx

        check_endpoint(base_url, allow_remote)
        self.redact_input = redact_input
        token = api_key or os.environ.get("TOKENTRIAGE_LEV_API_KEY") or "local"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers=_request_headers(token, headers),
            follow_redirects=False,  # a redirect could carry the prompt to another host
        )

    def warmup(self, timeout_s: float = 120.0) -> None:
        """One throwaway request, so a scale-to-zero server starts before the first real call (it can take ~30 s)."""
        self._client.post(
            "/v1/systemone", json={"state": "hello", "questions": LEV_QUESTIONS}, timeout=timeout_s
        ).raise_for_status()

    def classify(self, features: RequestFeatures, state: str) -> Signals:
        start = time.perf_counter()
        payload = pii.redact(state, self.redact_input)
        resp = self._client.post("/v1/systemone", json={"state": payload, "questions": LEV_QUESTIONS})
        resp.raise_for_status()
        return signals_from_answers(resp.json()["answers"], self.name, (time.perf_counter() - start) * 1000)
