from pathlib import Path

import pytest

from text_branch import TextBranchUnavailable, TextPredictor


ROOT = Path(__file__).resolve().parents[1]


def test_trained_text_model_returns_documented_score():
    predictor = TextPredictor(ROOT / "text_fake_baseline.joblib")
    result = predictor.predict("A city council approved a new public transit budget.")
    assert 0.0 <= result["fake_score"] <= 1.0
    assert "not a verified probability" in result["score_kind"]
    assert "distant supervision" in result["model"]["label_source"]


def test_empty_text_is_rejected():
    predictor = TextPredictor(ROOT / "text_fake_baseline.joblib")
    with pytest.raises(ValueError):
        predictor.predict("  ")


def test_missing_model_is_explicit():
    with pytest.raises(TextBranchUnavailable):
        TextPredictor(ROOT / "missing-text-model.joblib")
