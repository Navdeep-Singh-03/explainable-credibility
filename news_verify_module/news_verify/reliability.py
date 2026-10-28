"""Source reliability: WHO published something, kept strictly separate from WHAT it says.

The seed list below is an illustrative starting point that YOU should review and edit
(`SourceReliabilityRegistry.from_json(path)` or the constructor). Inclusion is not an endorsement and
absence is not a criticism: unknown domains are `UNRATED` (weakly weighted, not distrusted).
Tier describes institutional provenance only; even official sources can be wrong or out of date.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .errors import InvalidConfigError
from .models import SourceTier
from .sanitize import hostname

# Commonly cited fact-checking organisations (verify against e.g. the IFCN signatory list before relying on this).
SEED_FACT_CHECKERS = ["snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "factcheck.afp.com",
                      "leadstories.com", "africacheck.org", "altnews.in", "boomlive.in", "factly.in",
                      "newschecker.in", "vishvasnews.com"]
SEED_REPUTABLE_NEWS = ["reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com"]
SEED_OFFICIAL = ["un.org", "who.int"]
# Government / institutional suffixes: provenance heuristic, not a guarantee of accuracy.
OFFICIAL_SUFFIXES = ("gov", "gov.in", "nic.in", "gov.uk", "mil", "europa.eu", "int")


@dataclass(frozen=True)
class SourceReliability:
    tier: SourceTier
    reason: str


class SourceReliabilityRegistry:
    def __init__(self, entries: Optional[Mapping[str, object]] = None, *, include_seed: bool = True,
                 official_suffixes: Tuple[str, ...] = OFFICIAL_SUFFIXES) -> None:
        self._map: Dict[str, SourceTier] = {}
        if include_seed:
            for d in SEED_FACT_CHECKERS:
                self._map[d] = SourceTier.FACT_CHECKER
            for d in SEED_REPUTABLE_NEWS:
                self._map[d] = SourceTier.REPUTABLE_NEWS
            for d in SEED_OFFICIAL:
                self._map[d] = SourceTier.OFFICIAL
        for dom, tier in (entries or {}).items():
            try:
                self._map[dom.lower().lstrip(".").removeprefix("www.")] = SourceTier(tier)
            except ValueError as exc:
                raise InvalidConfigError(f"unknown tier {tier!r} for {dom!r}") from exc
        self._suffixes = tuple(s.lower().lstrip(".") for s in official_suffixes)

    @classmethod
    def from_json(cls, path: str, **kwargs) -> "SourceReliabilityRegistry":
        """JSON file of {"domain.com": "official|fact_checker|reputable_news|unrated|low_reliability"}."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise InvalidConfigError("reliability file must contain a JSON object")
        return cls(data, **kwargs)

    def assess(self, url_or_host: str) -> SourceReliability:
        host = hostname(url_or_host) if "/" in url_or_host or ":" in url_or_host else url_or_host.lower().removeprefix("www.")
        if not host:
            return SourceReliability(SourceTier.UNRATED, "no host")
        best = None
        for dom, tier in self._map.items():
            if host == dom or host.endswith("." + dom):
                if best is None or len(dom) > len(best[0]):
                    best = (dom, tier)
        if best:
            return SourceReliability(best[1], f"registry match: {best[0]}")
        for suf in self._suffixes:
            if host == suf or host.endswith("." + suf):
                return SourceReliability(SourceTier.OFFICIAL, f"institutional domain suffix .{suf}")
        return SourceReliability(SourceTier.UNRATED, "domain not in registry")
