"""Evidence-stance classification: does THIS evidence support / contradict THIS claim?

Two classifiers ship, chained in order (first one that can judge an item wins):

1. FactCheckRatingClassifier (default, rule-based, English labels)
   Applies ONLY to fact-check reviews (items carrying the reviewer's own rating). It maps the rating to a
   stance, but only after checking that the reviewed claim really is our claim (relevance is done by the
   verifier) and that polarity matches: if one text is negated and the other is not, a "False" rating would
   apply to the opposite statement, so the item is marked INCONCLUSIVE instead of being flipped.
   Ambiguous labels (misleading, partly true, unproven, no evidence, satire, outdated...) are deliberately
   INCONCLUSIVE: they are not forced into support/contradiction. Unrecognised labels (e.g. other languages)
   are INCONCLUSIVE too. Confidences are rule strengths, not calibrated probabilities.

2. NLIClassifier (optional, needs `transformers` + `torch`)
   Natural-language-inference model: premise = evidence excerpt, hypothesis = claim. Used only for items
   whose excerpt is real page/report text. It NEVER runs on fact-check items, because a fact-check restates
   the claim it debunks and an NLI model would call that "entailment".

If no classifier can judge an item, the chain returns INCONCLUSIVE ("no applicable classifier"), never a guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from .errors import ClassifierUnavailableError
from .models import Claim, Evidence, Stance
from .textutil import has_negation


@dataclass
class StanceResult:
    stance: Stance
    confidence: Optional[float]
    method: str
    notes: List[str] = field(default_factory=list)


class StanceClassifier(Protocol):
    name: str

    def classify(self, claim: Claim, evidence: Evidence) -> Optional[StanceResult]:
        """Return None if this classifier does not apply to the item."""


# ---- rating baseline ----------------------------------------------------------------------------------
_MIXED = re.compile(
    r"\b(partly|partially|half|mixture|mixed|some truth|missing context|needs context|lacks context|misleading|"
    r"unproven|unverified|unsupported|no evidence|lacks? evidence|satire|satirical|outdated|exaggerat\w*|cherry|"
    r"disputed|unclear|undetermined|legend|research in progress|unrated|not rated|opinion)\b")
_FALSE = re.compile(
    r"\b(false|fake|incorrect|inaccurate|untrue|wrong|hoax|fabricated|bogus|pants on fire|"
    r"not (?:true|accurate|correct|real|factual)|isn'?t (?:true|accurate|correct|real))\b")
_TRUE = re.compile(r"\b(true|correct|accurate|verified|confirmed|factual)\b")
_MOSTLY = re.compile(r"\b(mostly|largely|generally)\b")


class FactCheckRatingClassifier:
    name = "factcheck_rating_v1"

    def classify(self, claim: Claim, evidence: Evidence) -> Optional[StanceResult]:
        if evidence.kind != "claim_review" and not evidence.provider_rating:
            return None
        rating = (evidence.provider_rating or "").strip()
        if not rating:
            return StanceResult(Stance.INCONCLUSIVE, None, self.name, ["fact-check item without a rating"])
        norm = re.sub(r"[^a-z' ]+", " ", rating.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        label = f"rating {rating!r}"
        if _MIXED.search(norm):
            return StanceResult(Stance.INCONCLUSIVE, None, self.name, [f"{label} is ambiguous/mixed; not forced into a verdict"])
        if _FALSE.search(norm):
            stance, conf = Stance.CONTRADICTS, (0.75 if _MOSTLY.search(norm) else 0.9)
        elif _TRUE.search(norm):
            stance, conf = Stance.SUPPORTS, (0.75 if _MOSTLY.search(norm) else 0.9)
        else:
            return StanceResult(Stance.INCONCLUSIVE, None, self.name, [f"{label} not recognised by the English rule table"])
        reviewed = evidence.reviewed_claim or evidence.excerpt
        if has_negation(claim.text) != has_negation(reviewed):
            return StanceResult(Stance.INCONCLUSIVE, None, self.name,
                                [f"{label} applies to the reviewed claim, but claim and reviewed claim differ in "
                                 "negation, so the rating may apply to the opposite statement; needs NLI/human check"])
        return StanceResult(stance, conf, self.name, [f"{label} mapped to {stance.value}; reviewed claim matched"])


# ---- optional NLI -------------------------------------------------------------------------------------
PredictFn = Callable[[str, str], Dict[str, float]]


def _normalise_labels(scores: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in scores.items():
        kl = str(k).lower()
        if "entail" in kl:
            out["entailment"] = float(v)
        elif "contradict" in kl:
            out["contradiction"] = float(v)
        elif "neutral" in kl:
            out["neutral"] = float(v)
        else:
            raise ClassifierUnavailableError(f"unrecognised NLI label {k!r}; expected entailment/neutral/contradiction")
    return out


def _load_hf_predict_fn(model_name: str, device: Optional[int]) -> PredictFn:
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:
        raise ClassifierUnavailableError("NLIClassifier needs: pip install transformers torch") from exc
    pipe = pipeline("text-classification", model=model_name, top_k=None, device=-1 if device is None else device)

    def predict(premise: str, hypothesis: str) -> Dict[str, float]:
        out = pipe({"text": premise, "text_pair": hypothesis}, truncation=True)
        if out and isinstance(out[0], list):
            out = out[0]
        return {str(d["label"]): float(d["score"]) for d in out}

    return predict


class NLIClassifier:
    """Premise = evidence excerpt (real page/report text), hypothesis = claim."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", *, predict_fn: Optional[PredictFn] = None,
                 min_confidence: float = 0.8, device: Optional[int] = None, min_premise_chars: int = 40) -> None:
        self.name = f"nli:{model_name}" if predict_fn is None else "nli:custom"
        self.min_confidence, self.min_premise_chars = min_confidence, min_premise_chars
        self._predict = predict_fn or _load_hf_predict_fn(model_name, device)  # fails loudly if unavailable

    def classify(self, claim: Claim, evidence: Evidence) -> Optional[StanceResult]:
        if evidence.kind == "claim_review" or evidence.provider_rating:
            return None  # fact-checks restate claims; NLI would misread them
        if len(evidence.excerpt) < self.min_premise_chars:
            return StanceResult(Stance.INCONCLUSIVE, None, self.name, ["excerpt too short for NLI"])
        scores = _normalise_labels(self._predict(evidence.excerpt, claim.text))
        if not scores:
            raise ClassifierUnavailableError("NLI model returned no scores")
        label = max(scores, key=lambda k: scores[k])
        conf = round(scores[label], 4)
        notes = [f"NLI on excerpt only: {', '.join(f'{k}={v:.2f}' for k, v in sorted(scores.items()))}"]
        if label == "neutral" or conf < self.min_confidence:
            return StanceResult(Stance.INCONCLUSIVE, conf, self.name, notes + ["neutral or below confidence floor"])
        return StanceResult(Stance.SUPPORTS if label == "entailment" else Stance.CONTRADICTS, conf, self.name, notes)


def classify_with_chain(classifiers: Sequence[StanceClassifier], claim: Claim, evidence: Evidence) -> StanceResult:
    for c in classifiers:
        res = c.classify(claim, evidence)
        if res is not None:
            return res
    return StanceResult(Stance.INCONCLUSIVE, None, "none", ["no applicable classifier for this kind of evidence"])
