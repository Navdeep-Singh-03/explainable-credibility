#!/usr/bin/env python3
"""Verify claims in a news text.

  # live (needs a Google Fact Check Tools API key in FACTCHECK_API_KEY)
  python examples/verify_text.py --text "The health ministry said 120 people died in Delhi in March 2025."
  python examples/verify_text.py --file article.txt --cache evidence_cache.sqlite --json

  # offline demo with a tiny local corpus (no key, no network)
  python examples/verify_text.py --demo
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_verify import (GoogleFactCheckProvider, NewsVerifier, SourceReliabilityRegistry,  # noqa: E402
                         StaticEvidenceProvider, VerifierConfig)

DEMO_TEXT = "The government banned all plastic bags nationwide in 2024. What a strange year."
DEMO_CORPUS = [{
    "url": "https://example.org/demo-fact-check", "source_name": "Demo Corpus Item (not a real source)",
    "kind": "claim_review", "reviewed_claim": "The government banned all plastic bags nationwide in 2024",
    "excerpt": "The government banned all plastic bags nationwide in 2024", "provider_rating": "False",
    "published_date": "2026-09-01T00:00:00Z"}]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text"); src.add_argument("--file"); src.add_argument("--demo", action="store_true")
    p.add_argument("--cache", help="SQLite cache path (default: in-memory)")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--max-requests", type=int, default=10)
    p.add_argument("--json", action="store_true", help="print the full structured result")
    a = p.parse_args()

    text = DEMO_TEXT if a.demo else (a.text if a.text else open(a.file, encoding="utf-8").read())
    cfg = VerifierConfig(timeout_seconds=a.timeout, max_requests_per_run=a.max_requests, cache_path=a.cache)
    provider = StaticEvidenceProvider(DEMO_CORPUS) if a.demo else GoogleFactCheckProvider()
    # Demo only: the fake demo source is registered as a fact-checker so the output shows a verdict. In real use,
    # unknown domains stay 'unrated' and a lone unrated source can never produce a verdict.
    registry = SourceReliabilityRegistry({"example.org": "fact_checker"}) if a.demo else None
    result = NewsVerifier(provider, cfg, reliability=registry).verify_text(text)

    if a.json:
        print(result.to_json())
        return 0
    for r in result.claims:
        print(f"\n[{r.claim.id}] {r.claim.text}")
        reason = f" ({r.insufficient_reason.value})" if r.insufficient_reason else ""
        print(f"  status: {r.status.value}{reason} | quality: {r.quality.level.value} | retrieval: {r.retrieval_status}")
        for f in r.failures:
            print(f"  FAILURE {f.kind.value}: {f.message}")
        for e in r.evidence[:5]:
            print(f"  - [{e.stance.value}, source={e.source_tier.value}] {e.source_name} {e.published_date or 'undated'}\n    {e.url}")
        for line in r.quality.rationale:
            print(f"  note: {line}")
    for w in result.warnings:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
