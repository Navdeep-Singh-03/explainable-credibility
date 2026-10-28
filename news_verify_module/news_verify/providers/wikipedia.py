"""Retrieve short, attributed Wikipedia extracts as background context (never fact-check verdicts)."""
from __future__ import annotations

from typing import Any, List

from ..errors import ProviderError
from ..models import FailureKind, ProviderRecord
from .base import EvidenceProvider, SearchContext

ENDPOINT = "https://en.wikipedia.org/w/api.php"


class WikipediaContextProvider(EvidenceProvider):
    """Search Wikipedia and return its own introductory extracts with source URLs."""

    name = "wikipedia_context"

    def cache_fingerprint(self) -> str:
        return "mediawiki-generator-search-v1|en.wikipedia.org|intro-3-sentences"

    def search(self, query: str, ctx: SearchContext) -> List[ProviderRecord]:
        data = ctx.http.get_json(
            ENDPOINT,
            {
                "action": "query", "generator": "search", "gsrsearch": query[:250],
                "gsrnamespace": 0, "gsrlimit": min(ctx.limit, 5),
                "prop": "extracts|info", "exintro": 1, "explaintext": 1,
                "exsentences": 3, "inprop": "url", "format": "json", "formatversion": 2,
            },
            headers={"User-Agent": "SignalNewsChecker/1.0 (student research prototype)"},
        )
        return self.parse_response(data)

    @staticmethod
    def parse_response(data: Any) -> List[ProviderRecord]:
        if not isinstance(data, dict):
            raise ProviderError(FailureKind.BAD_RESPONSE, "expected a JSON object from Wikipedia")
        pages = (data.get("query") or {}).get("pages", []) if isinstance(data.get("query") or {}, dict) else []
        if isinstance(pages, dict):
            pages = list(pages.values())
        if not isinstance(pages, list):
            raise ProviderError(FailureKind.BAD_RESPONSE, "Wikipedia returned an invalid pages field")
        records: List[ProviderRecord] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            title, extract, url = page.get("title"), page.get("extract"), page.get("fullurl")
            # Use only the URL actually returned by the API; never synthesize a source link.
            if not isinstance(title, str) or not isinstance(extract, str) or not extract.strip():
                continue
            if not isinstance(url, str) or not url.startswith("https://en.wikipedia.org/wiki/"):
                continue
            records.append(ProviderRecord(
                url=url, source_name="Wikipedia", excerpt=extract.strip(),
                excerpt_origin="mediawiki_api.extract", kind="article", title=title,
                source_site="en.wikipedia.org",
            ))
        return records
