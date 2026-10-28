"""Data model for the external-verification module.

Everything that leaves the module is a plain dataclass with ``to_dict()`` (JSON-safe).

Two design rules are encoded here:

* ``Stance`` (what the *content* says about the claim) and ``SourceTier`` (how much we trust the *source*)
  are separate fields and are never merged into one number.
* Missing information is ``None`` / an explicit enum value, never a neutral placeholder such as 0.5.
"""
from __future__ import annotations

import dataclasses
import enum
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class _StrEnum(str, enum.Enum):
    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return str(self.value)


class Stance(_StrEnum):
    """What one piece of evidence says about one claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"  # relevant, but does not settle the claim (mixed / unproven / unclear)
    IRRELEVANT = "irrelevant"  # retrieved, but not about this claim; excluded from the verdict


class ClaimStatus(_StrEnum):
    """Overall outcome for one claim. NOTE: this describes the *evidence found*, not ground truth."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient_evidence"


class InsufficientReason(_StrEnum):
    NO_EVIDENCE_FOUND = "no_evidence_found"  # search worked, returned nothing (NOT evidence of falsehood)
    NO_RELEVANT_EVIDENCE = "no_relevant_evidence"  # results came back but none were about this claim
    ONLY_INCONCLUSIVE = "only_inconclusive_evidence"
    CONFLICTING_SOURCES = "conflicting_sources"
    WEAK_EVIDENCE = "weak_evidence"  # a lean exists but source reliability / relevance / age is too low
    OUTDATED_EVIDENCE = "outdated_evidence"
    PROVIDER_FAILURE = "provider_failure"  # search could not be completed, so absence proves nothing


class SourceTier(_StrEnum):
    OFFICIAL = "official"
    FACT_CHECKER = "fact_checker"
    REPUTABLE_NEWS = "reputable_news"
    UNRATED = "unrated"  # unknown source: neither trusted nor distrusted
    LOW_RELIABILITY = "low_reliability"


class QualityLevel(_StrEnum):
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class FailureKind(_StrEnum):
    NOT_CONFIGURED = "not_configured"
    TIMEOUT = "timeout"
    NETWORK = "network_error"
    RATE_LIMITED = "rate_limited"
    AUTH = "auth_error"
    PROVIDER_ERROR = "provider_error"
    BAD_RESPONSE = "bad_response"
    BUDGET_EXHAUSTED = "budget_exhausted"


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k.value if isinstance(k, enum.Enum) else k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    return obj


class _Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class Claim(_Serializable):
    id: str
    text: str
    sentence_index: Optional[int] = None
    features: List[str] = field(default_factory=list)  # why it looked checkable (number, entity, ...)
    flags: List[str] = field(default_factory=list)  # cautions (hedged, needs_context, ...)
    search_queries: List[str] = field(default_factory=list)
    extractor: str = "manual"


@dataclass
class ProviderRecord(_Serializable):
    """What a provider returns: only fields the source actually supplied. Nothing is synthesised."""

    url: str
    source_name: Optional[str]
    excerpt: str  # verbatim text from the provider response (see excerpt_origin)
    excerpt_origin: str  # e.g. "claim_review.claim_text" or "page_text"
    kind: str = "article"  # "claim_review" | "article" | "official_report" | "other"
    published_date: Optional[str] = None  # raw date string from the provider, parsed later
    title: Optional[str] = None
    provider_rating: Optional[str] = None  # the fact-checker's own verdict label, as returned
    reviewed_claim: Optional[str] = None  # the claim the source says it evaluated
    source_site: Optional[str] = None


@dataclass
class Evidence(_Serializable):
    # --- provenance (all required; an item without a valid URL is dropped, never invented) ---
    url: str
    source_name: str
    excerpt: str
    retrieved_at: str  # ISO-8601 UTC; for cache hits this is the ORIGINAL retrieval time
    provider: str
    query: str
    kind: str = "article"
    excerpt_origin: str = "provider_field"
    published_date: Optional[str] = None  # ISO-8601 UTC, or None when the source gave none / it was invalid
    title: Optional[str] = None
    provider_rating: Optional[str] = None
    reviewed_claim: Optional[str] = None
    from_cache: bool = False
    # --- source reliability (independent of what the content says) ---
    source_domain: str = ""
    source_tier: SourceTier = SourceTier.UNRATED
    source_tier_reason: str = ""
    # --- content assessment (independent of how reliable the source is) ---
    relevance: Optional[float] = None
    stance: Optional[Stance] = None  # None only before assessment
    stance_confidence: Optional[float] = None  # heuristic, NOT a calibrated probability
    stance_method: Optional[str] = None
    stance_notes: List[str] = field(default_factory=list)
    outdated: Optional[bool] = None  # None = publication date unknown
    weight: Optional[float] = None  # contribution to the verdict (decisive items only)


@dataclass
class Failure(_Serializable):
    kind: FailureKind
    message: str
    retriable: bool = False
    provider: Optional[str] = None
    query: Optional[str] = None
    http_status: Optional[int] = None


@dataclass
class EvidenceQuality(_Serializable):
    level: QualityLevel
    relevant_count: int = 0
    decisive_count: int = 0
    independent_sources: int = 0  # distinct publishers among decisive evidence
    reliable_independent_sources: int = 0  # ...among the winning side, reliable tier, not outdated
    best_source_tier: Optional[SourceTier] = None
    newest_published_date: Optional[str] = None
    oldest_published_date: Optional[str] = None
    has_conflict: bool = False
    has_outdated: bool = False
    has_undated: bool = False
    retrieval_incomplete: bool = False
    # Unnormalised heuristic weights per side. None when there was no relevant evidence at all:
    # we do not substitute 0.0 or 0.5 for "no information".
    weights: Optional[Dict[str, float]] = None
    rationale: List[str] = field(default_factory=list)


@dataclass
class ClaimResult(_Serializable):
    claim: Claim
    status: ClaimStatus
    insufficient_reason: Optional[InsufficientReason]
    evidence: List[Evidence]
    quality: EvidenceQuality
    retrieval_status: str  # "ok" | "partial" | "failed"
    failures: List[Failure] = field(default_factory=list)
    queries_issued: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class VerificationResult(_Serializable):
    claims: List[ClaimResult]
    provider: str
    generated_at: str
    method: Dict[str, Any]
    requests_made: int = 0
    cache_hits: int = 0
    skipped_sentences: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    interpretation: str = (
        "Statuses describe what retrieved external evidence says about each claim. "
        "'insufficient_evidence' means the question is open, not that the claim is false."
    )
