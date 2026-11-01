"""Small, dependency-free HTTP helpers shared by the LLM providers and the web tool."""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

from ultron import __version__
from ultron.errors import ProviderError
from ultron.redact import redact

USER_AGENT = f"ultron/{__version__}"
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
_MAX_RETRY_AFTER = 30.0
_MAX_RESPONSE_BYTES = 20_000_000

_HINTS = {
    400: "the provider rejected the request (details above)",
    401: "authentication failed - check your API key",
    402: "billing problem on the provider account",
    403: "the key is not allowed to use this resource",
    404: "not found - check the model name and base URL",
    413: "request too large - lower max_context_chars",
    429: "rate limited or out of quota",
}


def is_loopback_host(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _LoopbackAwareProxyHandler(urllib.request.ProxyHandler):
    """Honour HTTP(S)_PROXY, except for localhost (Ollama must never go through a proxy)."""

    def proxy_open(self, req, proxy, type):  # signature is fixed by urllib
        if is_loopback_host(urlsplit(req.full_url).hostname or ""):
            return None
        return super().proxy_open(req, proxy, type)


def make_opener(*extra_handlers: Any) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_LoopbackAwareProxyHandler(), *extra_handlers)


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return min(max(float(value), 0.0), _MAX_RETRY_AFTER) if value is not None else None
    except ValueError:
        return None


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Extract the provider's own error message from a JSON error body."""
    try:
        raw = exc.read(4096).decode("utf-8", errors="replace")
    except Exception:  # best effort only: the error body is a convenience, never required
        return ""
    try:
        body = json.loads(raw)
    except ValueError:
        return raw.strip()[:300]
    err = body.get("error") if isinstance(body, dict) else None
    message = err.get("message") if isinstance(err, dict) else (err if isinstance(err, str) else None)
    request_id = body.get("request_id") if isinstance(body, dict) else None
    detail = str(message or raw)[:300]
    return f"{detail} (request_id={request_id})" if request_id else detail


def post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    *,
    timeout: float = 120.0,
    retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    secrets: Iterable[str] = (),
    unreachable_hint: str = "",
) -> dict[str, Any]:
    """POST JSON and return the decoded JSON object, retrying transient failures."""
    secrets = list(secrets)
    if urlsplit(url).scheme not in ("http", "https"):
        raise ProviderError("provider URLs must start with http:// or https://")
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        **headers,
    }
    opener = make_opener()
    host = urlsplit(url).netloc
    attempt = 0
    while True:
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")  # noqa: S310 (scheme checked above)
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise ProviderError("provider response is too large")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ProviderError("provider returned JSON that is not an object")
            return data
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in RETRYABLE_STATUS and attempt < retries:
                delay = _retry_after(exc.headers)
                sleep(delay if delay is not None else backoff * 2**attempt)
                attempt += 1
                continue
            detail = _error_detail(exc)
            hint = _HINTS.get(status, "")
            message = f"{host} returned HTTP {status}" + (f": {detail}" if detail else "")
            if hint and status != 400:
                message += f" [{hint}]"
            raise ProviderError(redact(message, secrets)) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt < retries:
                sleep(backoff * 2**attempt)
                attempt += 1
                continue
            reason = getattr(exc, "reason", exc)
            hint = f" {unreachable_hint}" if unreachable_hint else ""
            raise ProviderError(redact(f"cannot reach {host}: {reason}.{hint}", secrets)) from None
        except ValueError:
            raise ProviderError(f"{host} returned a response that is not valid JSON") from None
