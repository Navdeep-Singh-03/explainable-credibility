"""Exercises the real urllib transport against a local HTTP server (no external network)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from news_verify import FailureKind, ProviderError, VerifierConfig
from news_verify.http import HttpClient, RunBudget, UrllibTransport
from news_verify.textutil import coverage_relevance


class H(BaseHTTPRequestHandler):
    seen_headers = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        H.seen_headers.append(dict(self.headers))
        if self.path.startswith("/ok"):
            self._send(200, json.dumps({"claims": []}).encode())
        elif self.path.startswith("/redirect"):
            self.send_response(302); self.send_header("Location", "http://127.0.0.1:1/evil"); self.end_headers()
        elif self.path.startswith("/slow"):
            time.sleep(1.0); self._send(200, b"{}")
        elif self.path.startswith("/big"):
            self._send(200, b"[" + b"1," * 50000 + b"1]")
        elif self.path.startswith("/limited"):
            self._send(429, b'{"error":{"message":"slow down"}}', {"Retry-After": "999"})

    def _send(self, code, body, extra=None):
        self.send_response(code); self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)


@pytest.fixture(scope="module")
def base():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def client(**kw):
    cfg = VerifierConfig(min_request_interval_seconds=0, max_retries=kw.pop("max_retries", 0), **kw)
    return HttpClient(cfg, UrllibTransport(), RunBudget(cfg))


def test_real_transport_success_sends_headers(base):
    assert client().get_json(base + "/ok", {"q": "a b"}, headers={"X-Goog-Api-Key": "k"}) == {"claims": []}
    assert H.seen_headers[-1]["X-Goog-Api-Key"] == "k" and "news-verify" in H.seen_headers[-1]["User-Agent"]


def test_real_transport_does_not_follow_redirects(base):
    with pytest.raises(ProviderError) as ei:
        client().get_json(base + "/redirect")
    assert ei.value.kind == FailureKind.PROVIDER_ERROR and "redirect" in ei.value.message


def test_real_transport_timeout_maps_to_timeout_kind(base):
    with pytest.raises(ProviderError) as ei:
        client(timeout_seconds=0.2).get_json(base + "/slow")
    assert ei.value.kind == FailureKind.TIMEOUT


def test_real_transport_size_cap(base):
    with pytest.raises(ProviderError) as ei:
        client(max_response_bytes=1000).get_json(base + "/big")
    assert ei.value.kind == FailureKind.BAD_RESPONSE


def test_real_transport_connection_refused_is_network_error():
    with pytest.raises(ProviderError) as ei:
        client().get_json("http://127.0.0.1:1/x")
    assert ei.value.kind == FailureKind.NETWORK


def test_real_transport_429_long_retry_after_fails_fast(base):
    t0 = time.time()
    with pytest.raises(ProviderError) as ei:
        client(max_retries=3).get_json(base + "/limited")
    assert ei.value.kind == FailureKind.RATE_LIMITED and time.time() - t0 < 2


def test_coverage_relevance_longer_source_not_penalised():
    c = "The government banned all plastic bags nationwide in 2024."
    e = "The ministry confirmed a nationwide ban on all plastic bags took effect in 2024 across the country."
    assert coverage_relevance(c, e)[0] > 0.75
    assert coverage_relevance(c, "Cricket final in Mumbai drew record crowds this season")[0] < 0.2
