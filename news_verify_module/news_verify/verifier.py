"""NewsVerifier: claims -> provider search (cached, budgeted) -> sanitise -> reliability -> relevance -> stance -> verdict."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import __version__
from .aggregate import assess_claim
from .cache import Cache, MemoryCache, NullCache, SqliteCache, make_cache_key
from .claims import ClaimExtractor, RuleBasedClaimExtractor, build_queries
from .classifiers import FactCheckRatingClassifier, StanceClassifier, classify_with_chain
from .config import VerifierConfig
from .errors import InvalidInputError, ProviderError
from .http import HttpClient, RunBudget, Transport, UrllibTransport
from .models import (Claim, ClaimResult, Evidence, Failure, FailureKind, ProviderRecord, SourceTier, Stance,
                     VerificationResult)
from .providers.base import EvidenceProvider, SearchContext
from .reliability import SourceReliabilityRegistry
from .sanitize import canonical_url, clean_text, hostname, iso_now, iso_to_datetime, parse_date, safe_url
from .textutil import coverage_relevance, lexical_relevance

# A failure of these kinds will not fix itself within one run: stop hammering the provider.
_TRIPPING = {FailureKind.NOT_CONFIGURED, FailureKind.AUTH, FailureKind.RATE_LIMITED}


class _Run:
    def __init__(self, budget: RunBudget, http: HttpClient) -> None:
        self.budget, self.http = budget, http
        self.tripped: Optional[ProviderError] = None
        self.cache_hits = 0
        self.warnings: List[str] = []


class NewsVerifier:
    def __init__(self, provider: EvidenceProvider, config: Optional[VerifierConfig] = None, *,
                 extractor: Optional[ClaimExtractor] = None,
                 classifiers: Optional[Sequence[StanceClassifier]] = None,
                 reliability: Optional[SourceReliabilityRegistry] = None,
                 cache: Optional[Cache] = None,
                 transport: Optional[Transport] = None,
                 relevance_fn: Optional[Callable[[str, str], Tuple[float, List[str]]]] = None,
                 now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 sleep_fn: Callable[[float], None] = time.sleep,
                 clock_fn: Callable[[], float] = time.monotonic) -> None:
        self.provider = provider
        self.config = config or VerifierConfig()
        cfg = self.config
        self.extractor = extractor or RuleBasedClaimExtractor(max_claims=cfg.max_claims, queries_per_claim=cfg.queries_per_claim)
        self.classifiers = list(classifiers) if classifiers is not None else [FactCheckRatingClassifier()]
        self.reliability = reliability or SourceReliabilityRegistry()
        if cache is not None:
            self.cache = cache
        elif cfg.cache_ttl_seconds <= 0:
            self.cache = NullCache()
        else:
            self.cache = SqliteCache(cfg.cache_path) if cfg.cache_path else MemoryCache()
        self.transport = transport or UrllibTransport()
        self.relevance_fn = relevance_fn  # None = built-in lexical baselines (see _assess_evidence)
        self._now, self._sleep, self._clock = now_fn, sleep_fn, clock_fn

    # ------------------------------------------------------------------ public API
    def verify_text(self, text: str) -> VerificationResult:
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("text must be a non-empty string")
        if len(text) > self.config.max_input_chars:
            raise InvalidInputError(f"text longer than max_input_chars={self.config.max_input_chars}")
        extraction = self.extractor.extract(text)
        skipped = [{"text": s.text, "reason": s.reason} for s in extraction.skipped]
        result = self._run(extraction.claims, skipped)
        if not extraction.claims:
            result.warnings.append("no checkable claims were extracted; nothing was verified")
        return result

    def verify_claims(self, claims: Sequence[Union[str, Claim]]) -> VerificationResult:
        """Verify claims you extracted yourself (e.g. with an LLM)."""
        built: List[Claim] = []
        for i, c in enumerate(claims, 1):
            if isinstance(c, Claim):
                built.append(c)
            elif isinstance(c, str) and clean_text(c):
                text = clean_text(c, 1000)
                built.append(Claim(id=f"c{i}", text=text, search_queries=build_queries(text, self.config.queries_per_claim)))
            else:
                raise InvalidInputError(f"claim #{i} must be a non-empty string or Claim")
        return self._run(built, [])

    # ------------------------------------------------------------------ internals
    def _run(self, claims: List[Claim], skipped: List[Dict[str, str]]) -> VerificationResult:
        cfg = self.config
        budget = RunBudget(cfg, self._clock)
        run = _Run(budget, HttpClient(cfg, self.transport, budget, self._sleep, self._clock))
        results = [self._verify_one(c, run) for c in claims[: cfg.max_claims]]
        if len(claims) > cfg.max_claims:
            run.warnings.append(f"{len(claims) - cfg.max_claims} claim(s) not checked: over max_claims={cfg.max_claims}")
        return VerificationResult(
            claims=results, provider=self.provider.name, generated_at=iso_now(self._now()),
            method={"module_version": __version__, "extractor": getattr(self.extractor, "name", type(self.extractor).__name__),
                    "classifiers": [getattr(c, "name", type(c).__name__) for c in self.classifiers],
                    "config": cfg.snapshot()},
            requests_made=budget.requests_made, cache_hits=run.cache_hits, skipped_sentences=skipped, warnings=run.warnings)

    def _verify_one(self, claim: Claim, run: _Run) -> ClaimResult:
        cfg = self.config
        queries = (claim.search_queries or build_queries(claim.text, cfg.queries_per_claim))[: cfg.queries_per_claim]
        failures: List[Failure] = []
        evidence: List[Evidence] = []
        seen: set = set()
        ok = issued = 0
        notes: List[str] = []
        dropped = 0
        for q in queries:
            if run.tripped is not None:
                failures.append(Failure(run.tripped.kind, f"skipped: an earlier request in this run failed ({run.tripped.message})",
                                        False, self.provider.name, q, run.tripped.http_status))
                break
            reason = run.budget.exhausted_reason()
            if reason:
                failures.append(Failure(FailureKind.BUDGET_EXHAUSTED, f"skipped: {reason}", False, self.provider.name, q))
                break
            issued += 1
            try:
                records, retrieved_at, from_cache = self._search(q, run)
            except ProviderError as exc:
                failures.append(Failure(exc.kind, clean_text(exc.message, 200), exc.retriable, self.provider.name, q, exc.http_status))
                if exc.kind in _TRIPPING:
                    run.tripped = exc
                continue
            except Exception as exc:  # a buggy third-party provider must not break the structured result
                failures.append(Failure(FailureKind.PROVIDER_ERROR, f"unexpected provider exception: {type(exc).__name__}",
                                        False, self.provider.name, q))
                continue
            ok += 1
            for rec in records:
                ev = self._to_evidence(rec, q, retrieved_at, from_cache)
                if ev is None:
                    dropped += 1
                    continue
                key = canonical_url(ev.url)
                if key in seen:
                    continue
                seen.add(key)
                self._assess_evidence(claim, ev, run)
                evidence.append(ev)
        if dropped:
            notes.append(f"{dropped} result(s) dropped: missing/invalid URL or text, so the source could not be identified")
        evidence.sort(key=lambda e: (e.stance == Stance.IRRELEVANT,
                                     -(cfg.tier_weights[e.source_tier] * (e.relevance or 0.0))))
        if len(evidence) > cfg.max_evidence_per_claim:
            notes.append(f"evidence truncated to {cfg.max_evidence_per_claim} of {len(evidence)} items")
            evidence = evidence[: cfg.max_evidence_per_claim]
        status, reason_, quality = assess_claim(evidence, cfg, failures)
        retrieval = "ok" if not failures else ("partial" if ok else "failed")
        return ClaimResult(claim=claim, status=status, insufficient_reason=reason_, evidence=evidence, quality=quality,
                           retrieval_status=retrieval, failures=failures, queries_issued=issued, notes=notes)

    def _search(self, query: str, run: _Run) -> Tuple[List[ProviderRecord], str, bool]:
        cfg = self.config
        key = make_cache_key(self.provider.name, self.provider.cache_fingerprint(), query, cfg.max_results_per_query)
        try:
            cached = self.cache.get(key)
        except Exception:
            cached = None
            run.warnings.append("cache read failed; continuing without cache")
        if isinstance(cached, dict) and isinstance(cached.get("records"), list) and isinstance(cached.get("retrieved_at"), str):
            try:
                recs = [ProviderRecord(**r) for r in cached["records"]]
                run.cache_hits += 1
                return recs, cached["retrieved_at"], True
            except TypeError:
                pass  # incompatible/tampered entry: treat as a miss
        ctx = SearchContext(http=run.http, config=cfg, limit=cfg.max_results_per_query)
        records = self.provider.search(query, ctx)
        if not isinstance(records, list) or not all(isinstance(r, ProviderRecord) for r in records):
            raise ProviderError(FailureKind.BAD_RESPONSE, "provider returned an invalid result type")
        retrieved_at = iso_now(self._now())
        try:
            self.cache.set(key, {"retrieved_at": retrieved_at, "records": [r.to_dict() for r in records]}, cfg.cache_ttl_seconds)
        except Exception:
            run.warnings.append("cache write failed; results were not cached")
        return records, retrieved_at, False

    def _to_evidence(self, rec: ProviderRecord, query: str, retrieved_at: str, from_cache: bool) -> Optional[Evidence]:
        cfg = self.config
        url = safe_url(rec.url, cfg.allowed_url_schemes)
        excerpt = clean_text(rec.excerpt, cfg.excerpt_max_chars)
        if not url or not excerpt:
            return None
        host = hostname(url)
        name = clean_text(rec.source_name, 120) or host  # host comes from the URL itself, not invented
        site = ""
        if isinstance(rec.source_site, str) and rec.source_site.strip():
            s = rec.source_site.strip()
            site = hostname(s if "://" in s else "https://" + s)
        rel = self.reliability.assess(url)
        if rel.tier == SourceTier.UNRATED and site and site != host:
            alt = self.reliability.assess(site)
            if alt.tier != SourceTier.UNRATED:
                rel = alt
        published = parse_date(rec.published_date, self._now())
        outdated: Optional[bool] = None
        if published and cfg.max_evidence_age_days:
            outdated = (self._now() - iso_to_datetime(published)).days > cfg.max_evidence_age_days
        return Evidence(
            url=url, source_name=name, excerpt=excerpt, retrieved_at=retrieved_at, provider=self.provider.name, query=query,
            kind=clean_text(rec.kind, 30) or "article", excerpt_origin=clean_text(rec.excerpt_origin, 60) or "provider_field",
            published_date=published, title=clean_text(rec.title, 200) or None,
            provider_rating=clean_text(rec.provider_rating, 80) or None,
            reviewed_claim=clean_text(rec.reviewed_claim, cfg.excerpt_max_chars) or None,
            from_cache=from_cache, source_domain=host, source_tier=rel.tier, source_tier_reason=rel.reason, outdated=outdated)

    def _assess_evidence(self, claim: Claim, ev: Evidence, run: _Run) -> None:
        cfg = self.config
        if self.relevance_fn is not None:
            rel, rel_notes = self.relevance_fn(claim.text, ev.reviewed_claim or f"{ev.title or ''} {ev.excerpt}")
        elif ev.reviewed_claim:  # short statement of a claim: symmetric overlap
            rel, rel_notes = lexical_relevance(claim.text, ev.reviewed_claim)
        else:  # longer free text: how much of OUR claim does it cover?
            rel, rel_notes = coverage_relevance(claim.text, f"{ev.title or ''} {ev.excerpt}")
        ev.relevance = rel
        if rel < cfg.relevance_threshold:
            ev.stance, ev.stance_method = Stance.IRRELEVANT, "relevance_threshold"
            ev.stance_notes = [f"relevance {rel:.2f} < threshold {cfg.relevance_threshold}"] + rel_notes
            return
        try:
            res = classify_with_chain(self.classifiers, claim, ev)
        except Exception as exc:
            run.warnings.append(f"classifier error on {ev.source_domain}: {type(exc).__name__}")
            ev.stance, ev.stance_method = Stance.INCONCLUSIVE, "classifier_error"
            ev.stance_notes = [f"classifier raised {type(exc).__name__}; item not used as support or contradiction"]
            return
        ev.stance, ev.stance_confidence, ev.stance_method = res.stance, res.confidence, res.method
        ev.stance_notes = list(res.notes) + rel_notes
        if ev.stance in (Stance.SUPPORTS, Stance.CONTRADICTS) and (res.confidence or 0.0) < cfg.min_stance_confidence:
            ev.stance_notes.append(f"confidence {res.confidence} below min_stance_confidence {cfg.min_stance_confidence}; demoted to inconclusive")
            ev.stance = Stance.INCONCLUSIVE
