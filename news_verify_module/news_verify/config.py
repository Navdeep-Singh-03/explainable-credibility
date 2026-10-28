"""All tunables in one validated place. Defaults are conservative starting points, NOT validated optima."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .errors import InvalidConfigError
from .models import SourceTier


def _default_tier_weights() -> Dict[SourceTier, float]:
    return {
        SourceTier.OFFICIAL: 1.0,
        SourceTier.FACT_CHECKER: 1.0,
        SourceTier.REPUTABLE_NEWS: 0.8,
        SourceTier.UNRATED: 0.35,  # unknown source: weak, but not zero and not distrusted
        SourceTier.LOW_RELIABILITY: 0.1,
    }


@dataclass
class VerifierConfig:
    # --- network behaviour ---
    timeout_seconds: float = 10.0  # per HTTP request
    total_timeout_seconds: float = 60.0  # whole verify() call, retries and sleeps included
    max_retries: int = 2  # retries per request for timeouts / network errors / 429 / 5xx
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    max_retry_after_seconds: float = 10.0  # if the server asks for longer, fail fast instead of waiting
    min_request_interval_seconds: float = 0.2  # pacing between requests
    max_response_bytes: int = 2_000_000
    user_agent: str = "news-verify/0.1 (+evidence-retrieval; research use)"

    # --- request limits per verify() call ---
    max_claims: int = 3
    queries_per_claim: int = 2
    max_results_per_query: int = 10
    max_requests_per_run: int = 10  # every HTTP attempt, retries included
    max_evidence_per_claim: int = 15
    max_input_chars: int = 50_000

    # --- caching ---
    cache_ttl_seconds: float = 24 * 3600  # 0 disables caching
    cache_path: Optional[str] = None  # None = in-memory; a file path = SQLite

    # --- evidence assessment ---
    relevance_threshold: float = 0.45
    min_stance_confidence: float = 0.6
    excerpt_max_chars: int = 400
    max_evidence_age_days: Optional[int] = 3 * 365  # None disables the outdated check
    outdated_policy: str = "downweight"  # "downweight" | "insufficient"
    outdated_weight: float = 0.5
    undated_weight: float = 0.8
    dominance_threshold: float = 0.75  # winning side's share of total weight; below = conflicting
    min_decisive_mass: float = 0.6  # winning side's total weight needed to give a verdict
    tier_weights: Dict[SourceTier, float] = field(default_factory=_default_tier_weights)
    allowed_url_schemes: tuple = ("http", "https")

    def __post_init__(self) -> None:
        def positive(name: str, allow_zero: bool = False) -> None:
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0 or (v == 0 and not allow_zero):
                raise InvalidConfigError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} number, got {v!r}")

        for n in ("timeout_seconds", "total_timeout_seconds", "backoff_base_seconds", "backoff_max_seconds",
                  "max_response_bytes", "max_claims", "queries_per_claim", "max_results_per_query",
                  "max_requests_per_run", "max_evidence_per_claim", "max_input_chars", "excerpt_max_chars"):
            positive(n)
        for n in ("max_retries", "max_retry_after_seconds", "min_request_interval_seconds", "cache_ttl_seconds"):
            positive(n, allow_zero=True)
        for n in ("relevance_threshold", "min_stance_confidence", "outdated_weight", "undated_weight",
                  "dominance_threshold", "min_decisive_mass"):
            v = getattr(self, n)
            if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                raise InvalidConfigError(f"{n} must be in [0, 1], got {v!r}")
        if not 0.5 < self.dominance_threshold <= 1.0:
            raise InvalidConfigError("dominance_threshold must be in (0.5, 1]")
        if self.outdated_policy not in ("downweight", "insufficient"):
            raise InvalidConfigError("outdated_policy must be 'downweight' or 'insufficient'")
        if self.max_evidence_age_days is not None and self.max_evidence_age_days <= 0:
            raise InvalidConfigError("max_evidence_age_days must be positive or None")
        missing = set(SourceTier) - set(self.tier_weights)
        if missing:
            raise InvalidConfigError(f"tier_weights missing tiers: {sorted(t.value for t in missing)}")

    def snapshot(self) -> Dict[str, object]:
        """Config for the result's `method` block. Contains no secrets (keys live in providers, not here)."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "tier_weights"}
        d["tier_weights"] = {t.value: w for t, w in self.tier_weights.items()}
        d["allowed_url_schemes"] = list(self.allowed_url_schemes)
        return d
