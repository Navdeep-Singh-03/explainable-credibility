"""HTTP layer: timeouts, bounded retries with backoff, Retry-After handling, request budget, size cap.

Security notes
* Redirects are NOT followed (so credentials in headers can never be replayed to another host).
* Responses are size-capped and parsed only as JSON data.
* Error text from servers is sanitised and truncated before it is stored.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Protocol
from urllib.parse import urlencode

from .config import VerifierConfig
from .errors import ProviderError
from .models import FailureKind
from .sanitize import clean_text


@dataclass
class HttpResponse:
    status: int
    headers: Dict[str, str]
    body: bytes


class Transport(Protocol):
    def get(self, url: str, headers: Mapping[str, str], timeout: float, max_bytes: int) -> HttpResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # type: ignore[override]
        return None


class UrllibTransport:
    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_NoRedirect)

    @staticmethod
    def _read(resp: Any, status: int, max_bytes: int) -> HttpResponse:
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ProviderError(FailureKind.BAD_RESPONSE, f"response larger than {max_bytes} bytes")
        return HttpResponse(status, {k.lower(): v for k, v in resp.headers.items()}, data)

    def get(self, url: str, headers: Mapping[str, str], timeout: float, max_bytes: int) -> HttpResponse:
        req = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                return self._read(resp, resp.status, max_bytes)
        except urllib.error.HTTPError as exc:
            try:
                return self._read(exc, exc.code, max_bytes)
            finally:
                exc.close()
        except (socket.timeout, TimeoutError) as exc:
            raise ProviderError(FailureKind.TIMEOUT, "request timed out", retriable=True) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise ProviderError(FailureKind.TIMEOUT, "request timed out", retriable=True) from exc
            raise ProviderError(FailureKind.NETWORK, f"network error: {type(exc.reason).__name__}", retriable=True) from exc
        except OSError as exc:
            raise ProviderError(FailureKind.NETWORK, f"network error: {type(exc).__name__}", retriable=True) from exc


class RunBudget:
    """Per-verify() limits: number of HTTP attempts and a wall-clock deadline."""

    def __init__(self, cfg: VerifierConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg, self._clock = cfg, clock
        self.requests_made = 0
        self.deadline = clock() + cfg.total_timeout_seconds

    def remaining_time(self) -> float:
        return self.deadline - self._clock()

    def exhausted_reason(self) -> Optional[str]:
        if self.requests_made >= self.cfg.max_requests_per_run:
            return f"request limit reached ({self.cfg.max_requests_per_run} per run)"
        if self.remaining_time() <= 0:
            return f"total time limit reached ({self.cfg.total_timeout_seconds}s)"
        return None


class HttpClient:
    def __init__(self, cfg: VerifierConfig, transport: Transport, budget: RunBudget,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg, self.transport, self.budget, self._sleep, self._clock = cfg, transport, budget, sleep, clock
        self._last_request: Optional[float] = None

    # -- helpers ---------------------------------------------------------------------------------
    def _pace(self) -> None:
        if self._last_request is not None and self.cfg.min_request_interval_seconds > 0:
            wait = self.cfg.min_request_interval_seconds - (self._clock() - self._last_request)
            if 0 < wait < self.budget.remaining_time():
                self._sleep(wait)
        self._last_request = self._clock()

    def _wait(self, delay: float) -> bool:
        """Sleep for `delay` if the run deadline allows it; False means: do not retry."""
        if delay >= self.budget.remaining_time():
            return False
        if delay > 0:
            self._sleep(delay)
        return True

    def _backoff(self, attempt: int) -> float:
        return min(self.cfg.backoff_max_seconds, self.cfg.backoff_base_seconds * (2 ** attempt))

    @staticmethod
    def _error_message(resp: HttpResponse) -> str:
        try:
            err = json.loads(resp.body.decode("utf-8", "replace")).get("error", {})
            msg = err.get("message") or err.get("status") or ""
            return clean_text(msg, 160)
        except (ValueError, AttributeError):
            return ""

    def _error_for_status(self, resp: HttpResponse) -> ProviderError:
        s, detail = resp.status, self._error_message(resp)
        suffix = f": {detail}" if detail else ""
        retry_after = None
        ra = resp.headers.get("retry-after", "")
        if ra.strip().isdigit():
            retry_after = float(ra.strip())
        if s == 429:
            return ProviderError(FailureKind.RATE_LIMITED, f"rate limited (HTTP 429){suffix}", retriable=True,
                                 http_status=s, retry_after=retry_after)
        if s in (401, 403):
            low = detail.lower()
            if s == 403 and ("quota" in low or "rate limit" in low):
                return ProviderError(FailureKind.RATE_LIMITED, f"quota exceeded (HTTP 403){suffix}", http_status=s)
            return ProviderError(FailureKind.AUTH, f"authentication/authorisation failed (HTTP {s}){suffix}", http_status=s)
        if s == 408:
            return ProviderError(FailureKind.TIMEOUT, "server reported request timeout (HTTP 408)", retriable=True, http_status=s)
        if s in (500, 502, 503, 504):
            return ProviderError(FailureKind.PROVIDER_ERROR, f"provider server error (HTTP {s}){suffix}", retriable=True,
                                 http_status=s, retry_after=retry_after)
        if 300 <= s < 400:
            return ProviderError(FailureKind.PROVIDER_ERROR, f"unexpected redirect (HTTP {s}); not followed", http_status=s)
        return ProviderError(FailureKind.PROVIDER_ERROR, f"unexpected HTTP status {s}{suffix}", http_status=s)

    # -- public ----------------------------------------------------------------------------------
    def get_json(self, url: str, params: Optional[Mapping[str, Any]] = None,
                 headers: Optional[Mapping[str, str]] = None) -> Any:
        """GET and parse JSON. Secrets belong in `headers`, never in `params`/URL. Raises ProviderError only."""
        full = url + ("?" + urlencode(params) if params else "")
        hdrs = {"Accept": "application/json", "User-Agent": self.cfg.user_agent, **(headers or {})}
        attempt = 0
        while True:
            reason = self.budget.exhausted_reason()
            if reason:
                raise ProviderError(FailureKind.BUDGET_EXHAUSTED, reason)
            self._pace()
            self.budget.requests_made += 1
            timeout = max(0.1, min(self.cfg.timeout_seconds, self.budget.remaining_time()))
            try:
                resp = self.transport.get(full, hdrs, timeout, self.cfg.max_response_bytes)
            except ProviderError as exc:
                if exc.retriable and attempt < self.cfg.max_retries and self._wait(self._backoff(attempt)):
                    attempt += 1
                    continue
                raise
            if 200 <= resp.status < 300:
                try:
                    data = json.loads(resp.body.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ProviderError(FailureKind.BAD_RESPONSE, "response was not valid JSON", http_status=resp.status) from exc
                if not isinstance(data, (dict, list)):
                    raise ProviderError(FailureKind.BAD_RESPONSE, "unexpected JSON shape", http_status=resp.status)
                return data
            err = self._error_for_status(resp)
            if err.retriable and attempt < self.cfg.max_retries:
                delay = err.retry_after if err.retry_after is not None else self._backoff(attempt)
                if delay <= self.cfg.max_retry_after_seconds and self._wait(delay):
                    attempt += 1
                    continue
            raise err
