import pytest

from fuzzy_fusion import infer


def test_two_low_signals_support_likely_real():
    result = infer(text_fake=0.05, image_fake=0.1)
    assert result.outcome == "Likely Real"


def test_two_high_signals_support_likely_fake():
    result = infer(text_fake=0.95, image_fake=0.9)
    assert result.outcome == "Likely Fake"


def test_single_signal_stays_uncertain():
    result = infer(text_fake=0.99)
    assert result.outcome == "Needs Verification"
    assert result.inputs["image_fake"] is None


def test_conflict_and_semantic_mismatch_raise_uncertainty():
    result = infer(text_fake=0.9, image_fake=0.9, inconsistency=0.95, evidence_conflict=True)
    assert result.outcome == "Needs Verification"


def test_external_contradiction_corroborates_text_signal():
    assert infer(text_fake=0.9, evidence_fake=0.95).outcome == "Likely Fake"


def test_scores_are_validated():
    with pytest.raises(ValueError):
        infer(text_fake=1.2)


def test_result_includes_center_of_area_defuzzification():
    fake = infer(text_fake=0.95, image_fake=0.9)
    real = infer(text_fake=0.05, image_fake=0.1)
    assert 0.6 < fake.defuzzified_value <= 1.0
    assert 0.0 <= real.defuzzified_value < 0.4
