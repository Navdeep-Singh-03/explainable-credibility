import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_verify import (GoogleFactCheckProvider, NewsVerifier, VerifierConfig)  # noqa: E402
from news_verify.errors import ProviderError  # noqa: E402
from news_verify.http import HttpResponse  # noqa: E402

NOW = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)


class FakeTransport:
    """Scripted transport. Items are HttpResponse, ProviderError (raised) or a callable(url, headers)."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def get(self, url, headers, timeout, max_bytes):
        self.calls.append({"url": url, "headers": dict(headers), "timeout": timeout})
        if not self.script:
            raise AssertionError("unexpected extra HTTP call: " + url)
        item = self.script.pop(0)
        if callable(item):
            item = item(url, headers)
        if isinstance(item, Exception):
            raise item
        return item


def ok(payload, status=200, headers=None):
    return HttpResponse(status, headers or {}, json.dumps(payload).encode())


def review(text, rating, url, publisher="Example Checks", date="2026-09-01T00:00:00Z", title="A fact check", site=None):
    return {"text": text, "claimReview": [{"publisher": {"name": publisher, "site": site or (url.split("/")[2] if url.count("/") >= 2 else "example.org")},
                                           "url": url, "title": title, "reviewDate": date, "textualRating": rating}]}


@pytest.fixture
def cfg():
    return VerifierConfig(min_request_interval_seconds=0, max_retries=2, queries_per_claim=1)


@pytest.fixture
def make_verifier(cfg):
    def _make(*script, config=None, provider=None, **kw):
        t = FakeTransport(*script)
        v = NewsVerifier(provider or GoogleFactCheckProvider(api_key="TEST-KEY-123"), config or cfg, transport=t,
                         now_fn=lambda: NOW, sleep_fn=lambda s: None, **kw)
        return v, t
    return _make


CLAIM = "The government banned all plastic bags nationwide in 2024."
