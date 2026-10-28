"""Small, dependency-free text helpers used by claim extraction and the lexical relevance baseline."""
from __future__ import annotations

import re
from typing import List, Set, Tuple

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here hers
    him his how i if in into is it its just me more most my no nor not of off on once only or other our out over own
    same she should so some such than that the their theirs them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you your said says say
    has been new""".split()
)

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NEG = re.compile(
    r"\b(not|no|never|none|nobody|nothing|neither|nor|cannot|without|untrue|isn't|aren't|wasn't|weren't|"
    r"doesn't|don't|didn't|won't|wouldn't|hasn't|haven't|hadn't|can't|couldn't)\b|n't\b",
    re.I,
)


def numbers_in(text: str) -> Set[str]:
    return {m.group(0).replace(",", "").rstrip(".") for m in _NUM.finditer(text)}


def has_negation(text: str) -> bool:
    return bool(_NEG.search(text))


def _stem(t: str) -> str:
    if t.isdigit():
        return t
    if len(t) > 4 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "aeiouls":  # banned -> bann -> ban
        t = t[:-1]
    return t[:6]


def content_tokens(text: str) -> List[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        w = w.replace("'", "")
        if w in STOPWORDS or (len(w) < 2 and not w.isdigit()):
            continue
        out.append(_stem(w))
    return out


def lexical_relevance(claim: str, text: str) -> Tuple[float, List[str]]:
    """Baseline relevance in [0, 1]: blend of Dice overlap and best containment over stemmed content tokens.

    Deliberately simple and explainable. It misses paraphrases; plug in an embedding-based `relevance_fn`
    for better recall. A single shared word can never make a result relevant (>=2 shared tokens required).
    """
    c, e = set(content_tokens(claim)), set(content_tokens(text))
    if not c or not e:
        return 0.0, ["no comparable content words"]
    inter = len(c & e)
    dice = 2 * inter / (len(c) + len(e))
    score = 0.5 * (dice + max(inter / len(c), inter / len(e)))
    notes: List[str] = []
    if inter < 2:
        score = min(score, 0.2)
        notes.append("fewer than 2 shared content words")
    cn, en = numbers_in(claim), numbers_in(text)
    if cn and en and not (cn & en):
        score *= 0.5
        notes.append(f"numbers differ (claim {sorted(cn)} vs source {sorted(en)})")
    return round(min(score, 1.0), 4), notes


def coverage_relevance(claim: str, text: str) -> Tuple[float, List[str]]:
    """Relevance for FREE-TEXT sources (articles, reports), which are longer than the claim.

    Score = fraction of the claim's content words found in the source text. Same guards as `lexical_relevance`.
    Use `lexical_relevance` instead when the text is itself a short statement of a claim (a fact-check's
    reviewed claim), where length mismatch should be penalised.
    """
    c, e = set(content_tokens(claim)), set(content_tokens(text))
    if not c or not e:
        return 0.0, ["no comparable content words"]
    inter = len(c & e)
    score, notes = inter / len(c), []
    if inter < 2:
        score = min(score, 0.2)
        notes.append("fewer than 2 shared content words")
    cn, en = numbers_in(claim), numbers_in(text)
    if cn and en and not (cn & en):
        score *= 0.5
        notes.append(f"numbers differ (claim {sorted(cn)} vs source {sorted(en)})")
    return round(min(score, 1.0), 4), notes
