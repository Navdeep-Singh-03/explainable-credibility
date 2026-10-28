"""Google Fact Check Tools API provider (claims:search, v1alpha1).

Returns published fact-check reviews (ClaimReview): publisher, review URL, title, review date, the claim the
reviewer evaluated, and the reviewer's own textual rating. Requires an API key (env var FACTCHECK_API_KEY or
the `api_key` argument). The key is sent in the `X-Goog-Api-Key` header, so it never appears in URLs, logs,
cache keys or error messages.

Scope/limits: one page per query (no pagination); coverage is limited to claims that someone has fact-checked,
so "no results" is common and means nothing about truth.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

from ..errors import ProviderError
from ..models import FailureKind, ProviderRecord
from .base import EvidenceProvider, SearchContext

ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


class GoogleFactCheckProvider(EvidenceProvider):
    name = "google_factcheck"

    def __init__(self, api_key: Optional[str] = None, *, api_key_env: str = "FACTCHECK_API_KEY",
                 language_code: Optional[str] = "en", max_age_days: Optional[int] = None,
                 endpoint: str = ENDPOINT) -> None:
        self._api_key = api_key
        self._api_key_env = api_key_env
        self.language_code, self.max_age_days, self.endpoint = language_code, max_age_days, endpoint

    def _key(self) -> Optional[str]:
        return self._api_key or os.environ.get(self._api_key_env) or None

    def cache_fingerprint(self) -> str:
        return f"{self.endpoint}|lang={self.language_code}|maxAge={self.max_age_days}"

    def search(self, query: str, ctx: SearchContext) -> List[ProviderRecord]:
        key = self._key()
        if not key:
            raise ProviderError(FailureKind.NOT_CONFIGURED,
                                f"no API key: set {self._api_key_env} or pass api_key=")
        params = {"query": query[:300], "pageSize": min(ctx.limit, 100)}
        if self.language_code:
            params["languageCode"] = self.language_code
        if self.max_age_days:
            params["maxAgeDays"] = self.max_age_days
        data = ctx.http.get_json(self.endpoint, params, headers={"X-Goog-Api-Key": key})
        return self.parse_response(data)

    @staticmethod
    def parse_response(data: Any) -> List[ProviderRecord]:
        if not isinstance(data, dict):
            raise ProviderError(FailureKind.BAD_RESPONSE, "expected a JSON object")
        claims = data.get("claims", [])  # absent key = valid empty result
        if not isinstance(claims, list):
            raise ProviderError(FailureKind.BAD_RESPONSE, "'claims' is not a list")
        out: List[ProviderRecord] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            reviewed_text = claim.get("text")
            for rev in claim.get("claimReview") or []:
                if not isinstance(rev, dict) or not rev.get("url") or not isinstance(reviewed_text, str):
                    continue  # cannot identify the source: skip, never guess
                pub = rev.get("publisher") if isinstance(rev.get("publisher"), dict) else {}
                out.append(ProviderRecord(
                    url=rev["url"], source_name=pub.get("name"), source_site=pub.get("site"),
                    excerpt=reviewed_text, excerpt_origin="claim_review.claim_text", kind="claim_review",
                    published_date=rev.get("reviewDate"), title=rev.get("title"),
                    provider_rating=rev.get("textualRating"), reviewed_claim=reviewed_text))
        return out
