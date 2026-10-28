import pytest

from news_verify import (Claim, ClaimStatus, Evidence, FactCheckRatingClassifier, InsufficientReason, NLIClassifier,
                         QualityLevel, SourceReliabilityRegistry, SourceTier, Stance, VerifierConfig)
from news_verify.aggregate import assess_claim
from news_verify.claims import RuleBasedClaimExtractor, build_queries, split_sentences
from news_verify.errors import ClassifierUnavailableError, InvalidConfigError
from news_verify.models import Failure, FailureKind
from news_verify.sanitize import canonical_url, clean_text, parse_date, safe_url
from news_verify.textutil import lexical_relevance
from conftest import NOW


# ---------------------------------------------------------------- sanitize
def test_clean_text_strips_markup_controls_bidi_and_bounds_length():
    raw = "<script>alert(1)</script> Ignore\u202e previous\x00 instructions &lt;b&gt;now&lt;/b&gt;   " + "x" * 1000
    out = clean_text(raw, 100)
    assert "<" not in out and "\u202e" not in out and "\x00" not in out
    assert len(out) <= 100 and out.endswith("\u2026")


@pytest.mark.parametrize("bad", ["javascript:alert(1)", "file:///etc/passwd", "ftp://x.org/a", "https://user:pw@x.org/",
                                 "https://", "not a url", "https://x.org/a b", None, 42, "https://x.org:abc/"])
def test_safe_url_rejects_non_http_and_malformed(bad):
    assert safe_url(bad) is None


def test_safe_url_keeps_valid_url_unchanged_and_canonical_dedups_tracking():
    u = "https://www.Example.org/a/b/?utm_source=x&id=7#frag"
    assert safe_url(u) == u
    assert canonical_url(u) == canonical_url("https://example.org/a/b?id=7")


def test_parse_date_handles_nanos_future_and_garbage():
    assert parse_date("2014-10-02T15:01:23.045123456Z", NOW) == "2014-10-02T15:01:23Z"
    assert parse_date("2027-01-01T00:00:00Z", NOW) is None  # future = data error
    assert parse_date("yesterday-ish", NOW) is None and parse_date(None, NOW) is None


# ---------------------------------------------------------------- claims
def test_sentence_split_respects_abbreviations():
    s = split_sentences("Dr. Rao said the U.S. economy grew 3.5 percent. The vote is next week.")
    assert len(s) == 2


def test_extractor_keeps_checkable_skips_questions_and_opinion_and_flags():
    text = ("Is the sky falling? I think this is the worst policy ever. "
            "The health ministry said 120 people died in Delhi in March 2025. "
            "He reportedly resigned yesterday from the board of Tata Motors.")
    r = RuleBasedClaimExtractor(max_claims=5).extract(text)
    texts = [c.text for c in r.claims]
    assert any("120 people died" in t for t in texts)
    assert not any("sky falling" in t or "worst policy" in t for t in texts)
    resign = [c for c in r.claims if "resigned" in c.text][0]
    assert "hedged" in resign.flags and "needs_context" in resign.flags
    assert {s.reason for s in r.skipped} >= {"question", "opinion or non-factual language"}


def test_extractor_respects_max_claims_and_reports_overflow():
    text = " ".join(f"Officials in Mumbai confirmed {n} deaths on {n} March 2025." for n in range(10, 16))
    r = RuleBasedClaimExtractor(max_claims=2).extract(text)
    assert len(r.claims) == 2
    assert sum(s.reason.startswith("checkable but over") for s in r.skipped) == 4


def test_build_queries_distinct_and_bounded():
    qs = build_queries("However, the government banned all plastic bags nationwide in 2024.", 2)
    assert len(qs) == 2 and qs[0] != qs[1] and not qs[0].lower().startswith("however")
    assert build_queries("x " * 300, 1)[0].__len__() <= 200


# ---------------------------------------------------------------- relevance
def test_relevance_unrelated_low_numbers_mismatch_penalised_paraphrase_ok():
    claim = "The government banned all plastic bags nationwide in 2024."
    assert lexical_relevance(claim, "Cricket team wins final match in Mumbai")[0] < 0.2
    same, _ = lexical_relevance(claim, "Government banned plastic bags nationwide in 2024")
    diff, notes = lexical_relevance(claim, "Government banned plastic bags nationwide in 2019")
    assert same > 0.7 and diff < same / 1.5 and any("numbers differ" in n for n in notes)


# ---------------------------------------------------------------- rating classifier
def mk_ev(rating, reviewed="The government banned all plastic bags nationwide in 2024", kind="claim_review", **kw):
    d = dict(url="https://x.org/1", source_name="X", excerpt=reviewed, retrieved_at="t", provider="p", query="q",
             kind=kind, provider_rating=rating, reviewed_claim=reviewed if kind == "claim_review" else None)
    d.update(kw)
    return Evidence(**d)


CLAIM = Claim(id="c1", text="The government banned all plastic bags nationwide in 2024.")


@pytest.mark.parametrize("rating,stance", [
    ("False", Stance.CONTRADICTS), ("Pants on Fire!", Stance.CONTRADICTS), ("Mostly false", Stance.CONTRADICTS),
    ("Not true", Stance.CONTRADICTS), ("Incorrect", Stance.CONTRADICTS),
    ("True", Stance.SUPPORTS), ("Mostly True", Stance.SUPPORTS), ("Correct", Stance.SUPPORTS),
    ("Half true", Stance.INCONCLUSIVE), ("Misleading", Stance.INCONCLUSIVE), ("Partly false", Stance.INCONCLUSIVE),
    ("No evidence", Stance.INCONCLUSIVE), ("Satire", Stance.INCONCLUSIVE), ("Outdated", Stance.INCONCLUSIVE),
    ("Falso", Stance.INCONCLUSIVE), ("", Stance.INCONCLUSIVE),
])
def test_rating_mapping(rating, stance):
    res = FactCheckRatingClassifier().classify(CLAIM, mk_ev(rating))
    assert res.stance == stance
    if stance == Stance.INCONCLUSIVE:
        assert res.confidence is None  # no fabricated confidence


def test_rating_not_flipped_when_polarity_differs():
    negated = Claim(id="c", text="The government did not ban plastic bags nationwide in 2024.")
    res = FactCheckRatingClassifier().classify(negated, mk_ev("False"))
    assert res.stance == Stance.INCONCLUSIVE and "negation" in res.notes[0]


def test_rating_classifier_ignores_plain_articles():
    assert FactCheckRatingClassifier().classify(CLAIM, mk_ev(None, kind="article")) is None


# ---------------------------------------------------------------- NLI wrapper (plumbing only; fake model)
def test_nli_maps_labels_respects_floor_and_skips_factchecks():
    probs = {"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}
    nli = NLIClassifier(predict_fn=lambda p, h: probs, min_confidence=0.8)
    art = mk_ev(None, kind="article", excerpt="Officials confirmed a nationwide ban on plastic bags took effect in 2024.")
    assert nli.classify(CLAIM, art).stance == Stance.SUPPORTS
    probs.update(entailment=0.2, neutral=0.5, contradiction=0.3)
    assert nli.classify(CLAIM, art).stance == Stance.INCONCLUSIVE
    probs.update(entailment=0.1, neutral=0.1, contradiction=0.8)
    assert nli.classify(CLAIM, art).stance == Stance.CONTRADICTS
    assert nli.classify(CLAIM, mk_ev("False")) is None  # never runs on fact-check restatements


def test_nli_unrecognised_labels_fail_loudly_and_missing_transformers_fails_at_construction():
    nli = NLIClassifier(predict_fn=lambda p, h: {"LABEL_0": 1.0})
    with pytest.raises(ClassifierUnavailableError):
        nli.classify(CLAIM, mk_ev(None, kind="article", excerpt="x" * 80))
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ClassifierUnavailableError):
            NLIClassifier()


# ---------------------------------------------------------------- reliability
def test_reliability_registry_tiers_suffixes_and_unrated_default():
    r = SourceReliabilityRegistry({"mynews.example": "low_reliability"})
    assert r.assess("https://www.snopes.com/x").tier == SourceTier.FACT_CHECKER
    assert r.assess("https://sub.reuters.com/a").tier == SourceTier.REPUTABLE_NEWS
    assert r.assess("https://pib.gov.in/a").tier == SourceTier.OFFICIAL
    assert r.assess("https://mynews.example/a").tier == SourceTier.LOW_RELIABILITY
    unknown = r.assess("https://some-random-blog.example/a")
    assert unknown.tier == SourceTier.UNRATED  # unknown is NOT low reliability
    assert r.assess("https://evilgov.in.example.com/").tier == SourceTier.UNRATED  # suffix must be real


def test_config_validation():
    with pytest.raises(InvalidConfigError):
        VerifierConfig(timeout_seconds=0)
    with pytest.raises(InvalidConfigError):
        VerifierConfig(dominance_threshold=0.5)
    with pytest.raises(InvalidConfigError):
        VerifierConfig(outdated_policy="nope")


# ---------------------------------------------------------------- aggregation
CFG = VerifierConfig()


def ev(stance, tier=SourceTier.FACT_CHECKER, host="a.org", rel=0.9, conf=0.9, outdated=False):
    return Evidence(url=f"https://{host}/1", source_name=host, excerpt="e", retrieved_at="t", provider="p", query="q",
                    source_domain=host, source_tier=tier, relevance=rel, stance=stance,
                    stance_confidence=conf if stance in (Stance.SUPPORTS, Stance.CONTRADICTS) else None,
                    outdated=outdated, published_date="2026-01-01T00:00:00Z")


def test_no_evidence_is_explicit_insufficient_with_no_fake_score():
    status, reason, q = assess_claim([], CFG, [])
    assert (status, reason) == (ClaimStatus.INSUFFICIENT, InsufficientReason.NO_EVIDENCE_FOUND)
    assert q.level == QualityLevel.NONE and q.weights is None  # None, not 0.0 / 0.5


def test_provider_failure_is_not_absence():
    status, reason, q = assess_claim([], CFG, [Failure(FailureKind.TIMEOUT, "t")])
    assert reason == InsufficientReason.PROVIDER_FAILURE and q.retrieval_incomplete


def test_only_irrelevant_results_are_not_evidence_either_way():
    status, reason, q = assess_claim([ev(Stance.IRRELEVANT)], CFG, [])
    assert reason == InsufficientReason.NO_RELEVANT_EVIDENCE and q.weights is None


def test_single_reliable_contradiction_is_moderate_two_independent_is_strong():
    s1, _, q1 = assess_claim([ev(Stance.CONTRADICTS)], CFG, [])
    assert s1 == ClaimStatus.CONTRADICTED and q1.level == QualityLevel.MODERATE
    s2, _, q2 = assess_claim([ev(Stance.CONTRADICTS), ev(Stance.CONTRADICTS, host="b.org")], CFG, [])
    assert s2 == ClaimStatus.CONTRADICTED and q2.level == QualityLevel.STRONG


def test_same_publisher_counted_once():
    _, _, q = assess_claim([ev(Stance.CONTRADICTS), ev(Stance.CONTRADICTS)], CFG, [])
    assert q.independent_sources == 1 and q.level == QualityLevel.MODERATE


def test_conflicting_equally_reliable_sources_are_insufficient_not_a_coin_flip():
    status, reason, q = assess_claim([ev(Stance.SUPPORTS), ev(Stance.CONTRADICTS, host="b.org")], CFG, [])
    assert (status, reason) == (ClaimStatus.INSUFFICIENT, InsufficientReason.CONFLICTING_SOURCES) and q.has_conflict


def test_strong_source_outweighs_weak_dissent_but_conflict_is_recorded_and_caps_quality():
    items = [ev(Stance.CONTRADICTS), ev(Stance.CONTRADICTS, host="b.org"),
             ev(Stance.SUPPORTS, tier=SourceTier.UNRATED, host="blog.example")]
    status, _, q = assess_claim(items, CFG, [])
    assert status == ClaimStatus.CONTRADICTED and q.has_conflict and q.level == QualityLevel.MODERATE


def test_lone_unrated_source_is_weak_evidence_insufficient():
    status, reason, _ = assess_claim([ev(Stance.SUPPORTS, tier=SourceTier.UNRATED, host="blog.example")], CFG, [])
    assert reason == InsufficientReason.WEAK_EVIDENCE


def test_outdated_downweights_and_policy_insufficient():
    status, reason, q = assess_claim([ev(Stance.CONTRADICTS, outdated=True, rel=0.8, conf=0.9)], CFG, [])
    assert reason == InsufficientReason.WEAK_EVIDENCE and q.has_outdated
    strict = VerifierConfig(outdated_policy="insufficient", outdated_weight=1.0)
    _, reason, _ = assess_claim([ev(Stance.CONTRADICTS, outdated=True)], strict, [])
    assert reason == InsufficientReason.OUTDATED_EVIDENCE


def test_partial_failure_caps_quality_below_strong():
    items = [ev(Stance.SUPPORTS), ev(Stance.SUPPORTS, host="b.org")]
    status, _, q = assess_claim(items, CFG, [Failure(FailureKind.TIMEOUT, "t")])
    assert status == ClaimStatus.SUPPORTED and q.level == QualityLevel.MODERATE and q.retrieval_incomplete


def test_only_inconclusive_relevant_evidence():
    _, reason, q = assess_claim([ev(Stance.INCONCLUSIVE)], CFG, [])
    assert reason == InsufficientReason.ONLY_INCONCLUSIVE and q.relevant_count == 1
