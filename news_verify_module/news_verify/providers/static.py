"""Offline provider over a corpus YOU supply (curated reports, saved articles, test fixtures).

It performs simple token-overlap retrieval and returns only records present in the corpus, so it cannot
invent evidence. Corpus items are dicts with: url, source_name, excerpt (required) and optionally
published_date, title, kind, provider_rating, reviewed_claim.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ..models import ProviderRecord
from ..textutil import content_tokens
from .base import EvidenceProvider, SearchContext


class StaticEvidenceProvider(EvidenceProvider):
    name = "static_corpus"

    def __init__(self, corpus: Iterable[Dict[str, Any]]) -> None:
        self._records: List[ProviderRecord] = []
        for item in corpus:
            if not item.get("url") or not item.get("excerpt"):
                continue
            self._records.append(ProviderRecord(
                url=item["url"], source_name=item.get("source_name"), excerpt=item["excerpt"],
                excerpt_origin="static_corpus", kind=item.get("kind", "article"),
                published_date=item.get("published_date"), title=item.get("title"),
                provider_rating=item.get("provider_rating"), reviewed_claim=item.get("reviewed_claim")))

    def cache_fingerprint(self) -> str:
        return f"static:{len(self._records)}"

    def search(self, query: str, ctx: SearchContext) -> List[ProviderRecord]:
        q = set(content_tokens(query))
        scored = []
        for r in self._records:
            overlap = len(q & set(content_tokens((r.reviewed_claim or "") + " " + (r.title or "") + " " + r.excerpt)))
            if overlap:
                scored.append((overlap, r))
        scored.sort(key=lambda t: -t[0])
        return [r for _, r in scored[: ctx.limit]]
