"""Turn assessed evidence into a per-claim status plus an explicit quality description.

Weighting (a documented heuristic, NOT a probability):
    weight = tier_weight(source) x relevance x stance_confidence x recency_weight
Each publisher counts once per side (its best item), so one outlet publishing five near-identical
articles cannot outvote anyone. Source reliability enters ONLY here, as a weight; it never changes
what an item is judged to say.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import VerifierConfig
from .models import (ClaimStatus, Evidence, EvidenceQuality, Failure, InsufficientReason, QualityLevel, SourceTier,
                     Stance)

RELIABLE = {SourceTier.OFFICIAL, SourceTier.FACT_CHECKER, SourceTier.REPUTABLE_NEWS}
_TIER_ORDER = [SourceTier.OFFICIAL, SourceTier.FACT_CHECKER, SourceTier.REPUTABLE_NEWS, SourceTier.UNRATED,
               SourceTier.LOW_RELIABILITY]


def _recency_weight(e: Evidence, cfg: VerifierConfig) -> float:
    if e.outdated:
        return cfg.outdated_weight
    if e.outdated is None:
        return cfg.undated_weight
    return 1.0


def assess_claim(evidence: List[Evidence], cfg: VerifierConfig,
                 failures: List[Failure]) -> Tuple[ClaimStatus, Optional[InsufficientReason], EvidenceQuality]:
    incomplete = bool(failures)
    relevant = [e for e in evidence if e.stance not in (Stance.IRRELEVANT, None)]
    q = EvidenceQuality(level=QualityLevel.NONE, relevant_count=len(relevant), retrieval_incomplete=incomplete)

    def insufficient(reason: InsufficientReason, msg: str) -> Tuple[ClaimStatus, InsufficientReason, EvidenceQuality]:
        q.rationale.append(msg)
        return ClaimStatus.INSUFFICIENT, reason, q

    if incomplete:
        q.rationale.append(f"retrieval incomplete: {len(failures)} failure(s) "
                           f"({', '.join(sorted({f.kind.value for f in failures}))})")

    if not relevant:
        if incomplete:
            return insufficient(InsufficientReason.PROVIDER_FAILURE,
                                "search could not be completed, so the absence of evidence says nothing about the claim")
        if not evidence:
            return insufficient(InsufficientReason.NO_EVIDENCE_FOUND,
                                "the search ran but returned nothing; this is not evidence that the claim is false")
        return insufficient(InsufficientReason.NO_RELEVANT_EVIDENCE,
                            f"{len(evidence)} result(s) retrieved but none was about this claim")

    # descriptive facts about the relevant evidence
    dates = sorted(e.published_date for e in relevant if e.published_date)
    q.newest_published_date, q.oldest_published_date = (dates[-1], dates[0]) if dates else (None, None)
    q.has_outdated = any(e.outdated for e in relevant)
    q.has_undated = any(e.outdated is None for e in relevant)
    q.best_source_tier = min((e.source_tier for e in relevant), key=_TIER_ORDER.index)

    # weights per side, one best item per publisher
    side: Dict[Stance, Dict[str, float]] = {Stance.SUPPORTS: {}, Stance.CONTRADICTS: {}}
    decisive = [e for e in relevant if e.stance in side]
    q.decisive_count = len(decisive)
    for e in decisive:
        w = cfg.tier_weights[e.source_tier] * (e.relevance or 0.0) * (e.stance_confidence or 0.0) * _recency_weight(e, cfg)
        e.weight = round(w, 4)
        host = e.source_domain or e.url
        side[e.stance][host] = max(side[e.stance].get(host, 0.0), w)
    s_mass, c_mass = sum(side[Stance.SUPPORTS].values()), sum(side[Stance.CONTRADICTS].values())
    q.weights = {"supports": round(s_mass, 4), "contradicts": round(c_mass, 4)}
    q.independent_sources = len({e.source_domain or e.url for e in decisive})
    q.has_conflict = s_mass > 0 and c_mass > 0
    q.level = QualityLevel.WEAK

    total = s_mass + c_mass
    if total == 0:
        return insufficient(InsufficientReason.ONLY_INCONCLUSIVE,
                            "relevant sources were found, but none supports or contradicts the claim (mixed/unproven/unclear)")
    winner = Stance.SUPPORTS if s_mass >= c_mass else Stance.CONTRADICTS
    mw = max(s_mass, c_mass)
    if mw / total < cfg.dominance_threshold:
        return insufficient(InsufficientReason.CONFLICTING_SOURCES,
                            f"sources disagree (supports={s_mass:.2f}, contradicts={c_mass:.2f}); no side dominates")
    win_items = [e for e in decisive if e.stance == winner]
    if mw < cfg.min_decisive_mass:
        return insufficient(InsufficientReason.WEAK_EVIDENCE,
                            f"leaning {winner.value}, but total weight {mw:.2f} < {cfg.min_decisive_mass} "
                            "(low source reliability, relevance, confidence or age)")
    if cfg.outdated_policy == "insufficient" and all(e.outdated for e in win_items):
        return insufficient(InsufficientReason.OUTDATED_EVIDENCE,
                            "all evidence on the leading side is older than the configured age limit")

    # verdict reached: grade the evidence
    fresh_reliable = {e.source_domain or e.url for e in win_items if e.source_tier in RELIABLE and not e.outdated}
    q.reliable_independent_sources = len(fresh_reliable)
    if len(fresh_reliable) >= 2 and not q.has_conflict and not incomplete:
        q.level = QualityLevel.STRONG
        q.rationale.append(f"{len(fresh_reliable)} independent reliable sources agree and nothing contradicts them")
    elif fresh_reliable:
        q.level = QualityLevel.MODERATE
        q.rationale.append(f"{len(fresh_reliable)} reliable, current source(s)")
        if q.has_conflict:
            q.rationale.append("capped below STRONG: opposing evidence also exists")
        if incomplete:
            q.rationale.append("capped below STRONG: some searches failed, so contrary evidence may be missing")
    else:
        q.level = QualityLevel.WEAK
        q.rationale.append("no reliable, current source on the winning side")
    if q.has_outdated:
        q.rationale.append("some evidence is older than the configured age limit and was down-weighted")
    if q.has_undated:
        q.rationale.append("some evidence has no publication date")
    status = ClaimStatus.SUPPORTED if winner == Stance.SUPPORTS else ClaimStatus.CONTRADICTED
    return status, None, q
