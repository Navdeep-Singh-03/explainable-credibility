"""Claim extraction.

`RuleBasedClaimExtractor` is a transparent BASELINE: it splits text into sentences and keeps those that look
checkable (numbers, dates, named entities, attributions, event verbs) while skipping questions and opinion.
It cannot resolve pronouns or judge importance. For better extraction implement `ClaimExtractor`
(e.g. with an LLM) or pass your own claim strings to `NewsVerifier.verify_claims()`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Protocol

from .models import Claim
from .sanitize import clean_text
from .textutil import STOPWORDS

_ABBREVIATIONS = ["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "u.s", "u.k",
                  "no", "inc", "ltd", "co", "gov", "sen", "rep", "lt", "col", "gen", "rs", "approx", "dept"]
_SPLIT = re.compile(r"""(?:(?<=[.!?])|(?<=[.!?]["\u201d']))\s+(?=["\u201c'(\[]?[A-Z0-9])""")
_ATTRIB = re.compile(r"\b(said|says|stated|states|announced|announces|confirmed|confirms|claimed|claims|reported|"
                     r"reports|declared|denied|denies|revealed|reveals|warned|warns|according to|told|alleged|alleges)\b", re.I)
_VERBS = re.compile(r"\b(killed|dead|died|dies|arrested|banned|bans|approved|approves|launched|launches|caused|"
                    r"causes?|cures?|increased|increases|decreased|decreases|rose|fell|dropped|won|wins|lost|signed|"
                    r"passed|resigned|elected|cancell?ed|found|shows?|showed|proved?|linked|rejected|ordered|"
                    r"sentenced|charged|convicted|closed|opened|detected|recalled|suspended|released|hit|struck)\b", re.I)
_MONTHS = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_OPINION = re.compile(r"\b(i think|i believe|in my opinion|i feel|we should|you should|should|must|ought to|let's|"
                      r"hopefully|best|worst|beautiful|terrible|amazing|awful|incredible|disgusting|wonderful|"
                      r"click here|subscribe|share this|sign up|follow us)\b", re.I)
_HEDGE = re.compile(r"\b(reportedly|allegedly|rumou?red|may|might|could|possibly|apparently|supposedly|unconfirmed)\b", re.I)
_PRONOUN_START = {"he", "she", "they", "it", "this", "that", "these", "those", "such", "his", "her", "their"}
_LEAD = re.compile(r"^(however|but|and|also|meanwhile|breaking|update|reportedly)\s*[,:\-]?\s+", re.I)


@dataclass
class SkippedSentence:
    text: str
    reason: str


@dataclass
class ExtractionResult:
    claims: List[Claim]
    skipped: List[SkippedSentence] = field(default_factory=list)


class ClaimExtractor(Protocol):
    name: str

    def extract(self, text: str) -> ExtractionResult: ...


def split_sentences(text: str) -> List[str]:
    out: List[str] = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        for a in _ABBREVIATIONS:
            line = re.sub(r"(?<![A-Za-z])" + re.escape(a) + r"\.", a.replace(".", "\x00") + "\x00", line, flags=re.I)
        for s in _SPLIT.split(line):
            s = s.replace("\x00", ".").strip()
            if s:
                out.append(s)
    return out


def build_queries(claim_text: str, n: int = 2, max_chars: int = 200) -> List[str]:
    """Up to n distinct search queries: the claim itself, then a keyword-only variant."""
    q1 = _LEAD.sub("", claim_text.strip()).strip(" \"'\u201c\u201d")
    if len(q1) > max_chars:
        q1 = q1[:max_chars].rsplit(" ", 1)[0]
    queries = [q1] if q1 else []
    if n >= 2:
        seen, kw = set(), []
        for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-%.,]*", q1):
            w = w.strip(".,'")
            if not w or w.lower() in STOPWORDS or (len(w) < 3 and not w.isdigit()) or w.lower() in seen:
                continue
            seen.add(w.lower())
            kw.append(w)
            if len(kw) == 8:
                break
        q2 = " ".join(kw)
        if q2 and q2.lower() != q1.lower():
            queries.append(q2)
    return queries[:n]


class RuleBasedClaimExtractor:
    name = "rule_based_v1"

    def __init__(self, max_claims: int = 3, min_words: int = 5, max_words: int = 80, min_score: int = 2,
                 queries_per_claim: int = 2) -> None:
        self.max_claims, self.min_words, self.max_words = max_claims, min_words, max_words
        self.min_score, self.queries_per_claim = min_score, queries_per_claim

    def extract(self, text: str) -> ExtractionResult:
        text = clean_text(text) if "\n" not in text else "\n".join(clean_text(l) for l in text.split("\n"))
        scored, skipped = [], []
        for idx, sent in enumerate(split_sentences(text)):
            words = sent.split()
            if sent.rstrip().endswith("?"):
                skipped.append(SkippedSentence(sent, "question"))
                continue
            if len(words) < self.min_words:
                skipped.append(SkippedSentence(sent, "too short"))
                continue
            if len(words) > self.max_words:
                skipped.append(SkippedSentence(sent, "too long to treat as one claim"))
                continue
            feats = []
            if re.search(r"\d", sent):
                feats.append("number")
            if _YEAR.search(sent) or _MONTHS.search(sent):
                feats.append("date")
            if any((w[:1].isupper() and w.strip(".,;:\"'()") not in ("I", "A", "The")) for w in words[1:]) or \
                    any(w.isupper() and len(w.strip(".,")) >= 2 for w in words):
                feats.append("named_entity")
            if _ATTRIB.search(sent):
                feats.append("attribution")
            if _VERBS.search(sent):
                feats.append("event_verb")
            if _OPINION.search(sent) and not ({"number", "attribution"} & set(feats)):
                skipped.append(SkippedSentence(sent, "opinion or non-factual language"))
                continue
            if len(feats) < self.min_score:
                skipped.append(SkippedSentence(sent, "not enough checkable features"))
                continue
            scored.append((idx, sent, feats))
        chosen = sorted(sorted(scored, key=lambda t: (-len(t[2]), t[0]))[: self.max_claims], key=lambda t: t[0])
        for idx, sent, _ in sorted(scored, key=lambda t: (-len(t[2]), t[0]))[self.max_claims:]:
            skipped.append(SkippedSentence(sent, "checkable but over max_claims limit"))
        claims = []
        for n, (idx, sent, feats) in enumerate(chosen, 1):
            flags = []
            if _HEDGE.search(sent):
                flags.append("hedged")
            if words_first(sent) in _PRONOUN_START:
                flags.append("needs_context")
            claims.append(Claim(id=f"c{n}", text=sent, sentence_index=idx, features=feats, flags=flags,
                                search_queries=build_queries(sent, self.queries_per_claim), extractor=self.name))
        return ExtractionResult(claims=claims, skipped=skipped)


def words_first(sentence: str) -> str:
    m = re.match(r"\W*([A-Za-z']+)", sentence)
    return m.group(1).lower() if m else ""
