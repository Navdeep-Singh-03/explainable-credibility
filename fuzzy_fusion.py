"""Custom Mamdani-style fuzzy fusion with an explicit missing/conflicting evidence policy.

All score-like inputs must be in [0, 1]. Missing inputs stay missing; no neutral score is
imputed. Membership strengths are fuzzy degrees, never calibrated probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


def _valid(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _low(x: float) -> float:
    return max(0.0, min(1.0, (0.55 - x) / 0.35))


def _medium(x: float) -> float:
    return max(0.0, min((x - 0.25) / 0.25, (0.75 - x) / 0.25, 1.0))


def _high(x: float) -> float:
    return max(0.0, min(1.0, (x - 0.45) / 0.35))


@dataclass
class FusionResult:
    outcome: str
    memberships: dict[str, float]
    defuzzified_value: float
    inputs: dict[str, Optional[float]]
    activated_rules: list[dict[str, object]] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    method: str = "custom Mamdani-style fuzzy rule system; membership values are not probabilities"

    def to_dict(self):
        return asdict(self)


def infer(*, text_fake: Optional[float] = None, image_fake: Optional[float] = None,
          inconsistency: Optional[float] = None, evidence_fake: Optional[float] = None,
          evidence_conflict: bool = False) -> FusionResult:
    """Fuse available model scores and external stance.

    `evidence_fake=0` means current evidence supports the checked claim; `1` means it
    contradicts it. Callers should only provide this when evidence quality is at least
    moderate. `inconsistency` is a semantic mismatch signal, never evidence of falsity.
    """
    xs = {
        "text_fake": _valid(text_fake, "text_fake"),
        "image_fake": _valid(image_fake, "image_fake"),
        "inconsistency": _valid(inconsistency, "inconsistency"),
        "evidence_fake": _valid(evidence_fake, "evidence_fake"),
    }
    known_truth = [v for k, v in xs.items() if k in ("text_fake", "image_fake", "evidence_fake") and v is not None]
    count = len(known_truth)
    rules: list[dict[str, object]] = []
    out = {"Likely Real": 0.0, "Needs Verification": 0.0, "Likely Fake": 0.0}

    def fire(name: str, strength: float, label: str) -> None:
        strength = max(0.0, min(1.0, strength))
        if strength:
            out[label] = max(out[label], strength)
            rules.append({"rule": name, "strength": round(strength, 4), "outcome": label})

    # Corroboration rules: two independent signal families are required for a confident output.
    pairs = [("text_fake", "image_fake"), ("text_fake", "evidence_fake"), ("image_fake", "evidence_fake")]
    for a, b in pairs:
        x, y = xs[a], xs[b]
        if x is None or y is None:
            continue
        fire(f"{a} HIGH AND {b} HIGH -> Likely Fake", min(_high(x), _high(y)), "Likely Fake")
        fire(f"{a} LOW AND {b} LOW -> Likely Real", min(_low(x), _low(y)), "Likely Real")
        fire(f"{a}/{b} disagree -> Needs Verification", min(_high(x), _low(y)), "Needs Verification")
        fire(f"{a}/{b} disagree -> Needs Verification", min(_low(x), _high(y)), "Needs Verification")

    if evidence_conflict:
        fire("conflicting external sources -> Needs Verification", 1.0, "Needs Verification")
    if xs["inconsistency"] is not None:
        fire("high text-image mismatch -> Needs Verification", _high(xs["inconsistency"]), "Needs Verification")
    if count < 2:
        fire("fewer than two truth-related signal families available -> Needs Verification", 1.0, "Needs Verification")
    if any(v is not None and 0.25 < v < 0.75 for v in known_truth):
        fire("medium-strength signal -> Needs Verification", max(_medium(v) for v in known_truth), "Needs Verification")
    if max(out["Likely Real"], out["Likely Fake"]) > 0:
        # A confident conclusion needs multiple inputs and no active uncertainty rule of equal/greater strength.
        pass
    if not rules:
        fire("no decisive rule fired -> Needs Verification", 1.0, "Needs Verification")

    # Mamdani max-aggregation followed by center-of-area defuzzification on a 0..1 output universe.
    # Output sets are trapezoidal shoulders for real/fake and a triangle for verification.
    def triangle(x: float, left: float, center: float, right: float) -> float:
        if x == center:
            return 1.0
        if x < center:
            return max(0.0, (x - left) / (center - left)) if center > left else 0.0
        return max(0.0, (right - x) / (right - center)) if right > center else 0.0

    def output_set(label: str, x: float) -> float:
        if label == "Likely Real":
            return 1.0 if x <= 0.0 else max(0.0, (0.45 - x) / 0.45)
        if label == "Needs Verification":
            return triangle(x, 0.25, 0.5, 0.75)
        return 1.0 if x >= 1.0 else max(0.0, (x - 0.55) / 0.45)

    universe = [i / 100 for i in range(101)]
    aggregate = [max(min(out[label], output_set(label, x)) for label in out) for x in universe]
    area = sum(aggregate)
    centroid = sum(x * degree for x, degree in zip(universe, aggregate)) / area if area else 0.5

    # The discrete outcome uses the strongest aggregated label; uncertainty wins ties.
    confident = max(out["Likely Real"], out["Likely Fake"])
    if out["Needs Verification"] >= confident or confident == 0:
        outcome = "Needs Verification"
    else:
        outcome = "Likely Real" if out["Likely Real"] > out["Likely Fake"] else "Likely Fake"
    explanations = []
    if count < 2:
        explanations.append("Fewer than two truth-related signal families were available; no unsupported signal was imputed.")
    if xs["inconsistency"] is not None and xs["inconsistency"] >= 0.6:
        explanations.append("Text-image mismatch raises uncertainty; semantic mismatch is not proof of falsity.")
    if evidence_conflict:
        explanations.append("Retrieved sources conflict, so external evidence cannot settle the claim.")
    if not explanations:
        explanations.append("Outcome follows the strongest activated corroboration rule; scores and memberships are not probabilities.")
    return FusionResult(outcome, {k: round(v, 4) for k, v in out.items()}, round(centroid, 4), xs, rules, explanations)
