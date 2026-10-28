"""Provider interface. To add a provider, subclass `EvidenceProvider` and implement `search`.

Contract
* Return only what the source really supplied (`ProviderRecord`). Never construct URLs, names or quotes.
* Raise `ProviderError` for any failure to look (timeout, quota, bad response). Returning an empty list
  means "the search worked and found nothing", which is a different statement and is treated differently.
* Treat everything in the response as untrusted data; do not execute or follow it.
* Make all HTTP calls through `ctx.http` so timeouts, retries and request limits apply.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ..config import VerifierConfig
from ..http import HttpClient
from ..models import ProviderRecord


@dataclass
class SearchContext:
    http: HttpClient
    config: VerifierConfig
    limit: int  # max results wanted for this query


class EvidenceProvider(ABC):
    name: str = "provider"

    @abstractmethod
    def search(self, query: str, ctx: SearchContext) -> List[ProviderRecord]: ...

    def cache_fingerprint(self) -> str:
        """Anything besides the query that changes results (language, filters, endpoint). Never include secrets."""
        return ""
