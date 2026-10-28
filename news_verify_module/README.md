# External news verification module (`news_verify`)

Self-contained evidence branch of the project. It imports nothing from `fdhn_module` or `resnet_fake`:
**news text in, per-claim evidence assessment out.** Same style as the other modules (documented result with
`to_dict()`, typed errors, offline tests, explicit limitations). Standard library only; `pytest` for tests.

It investigates claims with *external* sources. It does not score writing style, and it never treats
"nothing found" as "false".

## 1. Quick start

```powershell
pip install -r requirements-dev.txt
python -m pytest -q                              # 91 offline tests, no network, no key needed
python examples/verify_text.py --demo            # offline demo with a local corpus

$env:FACTCHECK_API_KEY = "<your Google Fact Check Tools API key>"
python examples/verify_text.py --text "The health ministry said 120 people died in Delhi in March 2025."
python examples/verify_text.py --file article.txt --cache evidence_cache.sqlite --json
```

```python
from news_verify import NewsVerifier, GoogleFactCheckProvider, VerifierConfig

verifier = NewsVerifier(GoogleFactCheckProvider(), VerifierConfig(timeout_seconds=8, cache_path="evidence.sqlite"))
result = verifier.verify_text(article_text)          # extracts claims, then checks each
result = verifier.verify_claims(["Claim one.", "Claim two."])   # or bring your own claims (e.g. from an LLM)
print(result.to_json())
```

Get a key by enabling the *Fact Check Tools API* in a Google Cloud project and creating an API key. The key is sent
in the `X-Goog-Api-Key` header, so it never appears in URLs, cache keys, logs or results.

## 2. Pipeline

```
text -> claim extraction -> queries -> provider.search (cache, limits, retries)
     -> sanitise + validate URL/date -> source reliability  (WHO published it)
     -> relevance -> stance classification                  (WHAT it says)
     -> aggregate -> status + evidence quality + failure status
```

## 3. Result structure

`VerificationResult.claims[]` is a list of `ClaimResult`:

| Field | Meaning |
|---|---|
| `claim` | `id`, `text`, why it looked checkable (`features`), cautions (`flags`: `hedged`, `needs_context`), search queries |
| `status` | `supported` \| `contradicted` \| `insufficient_evidence` |
| `insufficient_reason` | set when insufficient: `no_evidence_found`, `no_relevant_evidence`, `only_inconclusive_evidence`, `conflicting_sources`, `weak_evidence`, `outdated_evidence`, `provider_failure` |
| `evidence[]` | one item per source (below) |
| `quality` | `level` (`none/weak/moderate/strong`), counts, independent sources, `best_source_tier`, newest/oldest date, `has_conflict`, `has_outdated`, `has_undated`, `retrieval_incomplete`, `weights`, `rationale[]` |
| `retrieval_status` | `ok` \| `partial` \| `failed` |
| `failures[]` | `kind` (`not_configured`, `timeout`, `network_error`, `rate_limited`, `auth_error`, `provider_error`, `bad_response`, `budget_exhausted`), message, retriable, query |

Each **evidence item**: `url`, `source_name`, `published_date` (or `null`), `retrieved_at` (original time, also for cache
hits), `excerpt` + `excerpt_origin`, `title`, `provider_rating`, `reviewed_claim`, `from_cache`;
**reliability** `source_tier` + `source_tier_reason`; **content** `relevance`, `stance`
(`supports/contradicts/inconclusive/irrelevant`), `stance_confidence`, `stance_method`, `stance_notes`;
`outdated` (`null` = unknown date) and `weight` (contribution to the verdict).

`status` describes the evidence found, **not ground truth**. `insufficient_evidence` means the question is open.

## 4. How the 12 requirements are met

| # | Requirement | Where / how |
|---|---|---|
| 1 | Verifiable claims | `claims.py`: sentence split, keeps sentences with numbers, dates, named entities, attributions, event verbs; skips questions and opinion; reports skipped sentences and why. Replaceable (`ClaimExtractor`) or bypass with `verify_claims()`. |
| 2 | Configurable provider | `EvidenceProvider` interface. Ships `GoogleFactCheckProvider` (live) and `StaticEvidenceProvider` (your own corpus, offline). |
| 3 | Identifiable, preferably reputable sources | Fact-check reviews carry publisher + review URL. Evidence without a valid URL is dropped. Reliability registry tiers: official / fact-checker / reputable news / unrated / low. |
| 4 | Stored provenance | `Evidence` fields above. Excerpts are verbatim provider fields (cleaned and truncated), never generated. |
| 5 | Support / contradict / insufficient | `ClaimStatus` + `InsufficientReason`. |
| 6 | Documented classifier | `FactCheckRatingClassifier` (rule baseline, section 5) and optional `NLIClassifier`. |
| 7 | Reliability vs content | Separate fields. Reliability only enters aggregation as a weight; it never changes what an item is judged to say. |
| 8 | Absence / failure / irrelevance are not falsity | Distinct reasons: `no_evidence_found`, `provider_failure`, `no_relevant_evidence`. None can yield `contradicted`. Failures are never cached. |
| 9 | No invention | Providers return only what the source supplied. Missing name falls back to the URL's own host; missing date stays `null`. Registry is user-editable. |
| 10 | Conflicts, outdated, outages, limits | Conflict -> `conflicting_sources`; outdated -> flagged and down-weighted (or insufficient by policy); retries with backoff, `Retry-After`, fail-fast on long waits, run-level stop after rate-limit/auth failure, partial results marked `partial` and capped below `strong`. |
| 11 | Timeout, cache, limits | `VerifierConfig`: per-request and total timeouts, `max_requests_per_run` (retries count), `max_claims`, `queries_per_claim`, `max_results_per_query`, pacing, response size cap, TTL cache (memory or SQLite). |
| 12 | Untrusted content | All provider text is stripped of markup/control/bidi characters, bounded, and used only as data. No redirects followed, no embedded links fetched, URLs must be plain http(s) without credentials, cache contents re-validated on load, no `eval`/prompting of retrieved text. |

No evidence means `insufficient_evidence` with `quality.level = none` and `quality.weights = null`. There is no
placeholder score anywhere.

## 5. Classification (documented baseline)

**Relevance first.** An item is only assessed if it is about this claim. Fact-check items: symmetric token overlap
between our claim and the claim the reviewer evaluated. Free-text sources: share of our claim's content words the
source covers. A single shared word is never enough; differing numbers halve the score. Below
`relevance_threshold` the item is kept but marked `irrelevant` and ignored.

**`FactCheckRatingClassifier` (default).** Maps the reviewer's own rating to a stance:

* false / fake / incorrect / hoax / "not true" / pants on fire -> `contradicts` (0.9; "mostly false" 0.75)
* true / correct / accurate / verified -> `supports` (0.9; "mostly true" 0.75)
* misleading, half/partly true, unproven, no evidence, satire, outdated, missing context, unknown or non-English
  labels -> `inconclusive` (deliberately not forced into a side)
* If our claim and the reviewed claim differ in negation, a "False" rating could apply to the *opposite* statement,
  so the item becomes `inconclusive` rather than being flipped.

**`NLIClassifier` (optional, `pip install transformers torch`).** Premise = evidence excerpt, hypothesis = claim,
default model `cross-encoder/nli-deberta-v3-base`, confidence floor 0.8. It runs only on real page/report text and
**never on fact-check items** (a fact-check restates the claim it debunks, which NLI would misread as entailment).
Enable with `NewsVerifier(provider, classifiers=[FactCheckRatingClassifier(), NLIClassifier()])`.

**Verdict weights.** `weight = tier_weight x relevance x stance_confidence x recency_weight`. Each publisher counts once
per side. A side needs share >= `dominance_threshold` (0.75) of total weight, else `conflicting_sources`, and total
weight >= `min_decisive_mass` (0.6), else `weak_evidence`. So one unrated source alone never decides a claim, and two
equally reliable sources that disagree give `insufficient_evidence`. Quality is `strong` only with >= 2 independent
reliable, current sources, no opposing evidence and a complete search.

## 6. Failure handling

| Situation | Behaviour |
|---|---|
| no API key | no request made; `not_configured`; claim `insufficient_evidence / provider_failure` |
| timeout, network error, HTTP 5xx | retried with backoff up to `max_retries`, then `failures[]` entry |
| HTTP 429 | honours `Retry-After` if <= `max_retry_after_seconds`, otherwise fails fast; later claims in the run are skipped, not hammered |
| 401 / 403 | `auth_error`, not retried, run stops calling the provider |
| invalid / oversized / wrong-shape response | `bad_response` |
| request or time budget hit | remaining claims report `budget_exhausted` |
| some queries fail, others work | `retrieval_status = partial`; evidence kept; quality capped below `strong` |
| buggy custom provider | caught; reported as `provider_error` (exception type only, no message) |

## 7. Adding a provider

```python
from news_verify import EvidenceProvider, ProviderRecord, ProviderError, FailureKind

class MyProvider(EvidenceProvider):
    name = "my_provider"
    def cache_fingerprint(self): return "v1|region=in"          # anything that changes results (no secrets)
    def search(self, query, ctx):                                # ctx.http enforces timeout/retries/budget
        data = ctx.http.get_json("https://api.example.com/search", {"q": query, "n": ctx.limit}, headers={"Authorization": "..."})
        return [ProviderRecord(url=h["url"], source_name=h.get("publisher"), excerpt=h["snippet"],
                               excerpt_origin="page_text", kind="article", published_date=h.get("date"), title=h.get("title"))
                for h in data.get("hits", [])]
```

Rules: return only what the source supplied; raise `ProviderError` for any failure to look (an empty list means
"searched, found nothing"); put secrets in headers, not URLs. Free-text providers (`kind="article"`) need an
`NLIClassifier` (or your own `StanceClassifier`) to produce supports/contradicts; otherwise their items are honestly
`inconclusive`. Edit source trust with `SourceReliabilityRegistry({"domain.com": "fact_checker"})` or
`.from_json(path)`.

## 8. Limitations (put these in your report)

1. **Coverage.** The Google provider only knows claims someone has already fact-checked. Most claims return nothing,
   which correctly yields `insufficient_evidence`. It is a verification aid, not a truth oracle.
2. **Rule-based baselines are unvalidated.** Claim extraction, lexical relevance, the rating table and all thresholds
   and tier weights are reasoned defaults, not tuned on labelled data. Lexical relevance misses paraphrases; the
   rating table is English-only. Evaluate on your own labelled claims before reporting accuracy.
3. **NLI is plumbing-tested only.** Tests use a fake model function; the real Hugging Face model was not loaded or
   evaluated in this build (no model-hub access in the build environment). NLI on short snippets is noisy.
4. **Reliability seed list is illustrative.** Review it. Tier reflects institutional provenance, not accuracy;
   official sources can be wrong or stale.
5. **Independence is approximated by domain.** Outlets that copy each other still look independent; fact-checkers
   often cite each other.
6. **Staleness is crude.** Age is measured against the publication date only; a fact-check of a time-sensitive claim
   ("unemployment is 4%") may be old yet still reviewed correctly for its time. The module does not compare the claim's
   own date with the evidence date.
7. **Excerpts for fact-check items are the reviewed claim text plus rating, not the article body.** The module does not
   fetch pages (that would add SSRF and prompt-injection surface); a page-fetching provider can be added through the
   same interface.
8. **Claims without context** ("He resigned yesterday") are flagged `needs_context` and search poorly.
9. **No video/image evidence, no cross-lingual matching.**

## 9. Layout

```
news_verify/models.py        enums + dataclasses (Claim, Evidence, ClaimResult, VerificationResult)
news_verify/config.py        VerifierConfig (validated)
news_verify/claims.py        claim extraction baseline + query building
news_verify/providers/       base interface, google_factcheck.py, static.py
news_verify/http.py          transport, retries, Retry-After, budget, size cap, no redirects
news_verify/cache.py         memory + SQLite caches (successful responses only)
news_verify/sanitize.py      untrusted-text cleaning, URL/date validation
news_verify/reliability.py   source tier registry
news_verify/classifiers.py   rating baseline, optional NLI, chain
news_verify/aggregate.py     verdict + evidence quality
news_verify/verifier.py      NewsVerifier orchestration
examples/verify_text.py      CLI (live or --demo)     tests/   91 tests, offline
```
