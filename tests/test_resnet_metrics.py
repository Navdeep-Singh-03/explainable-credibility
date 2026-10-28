import importlib.util
from pathlib import Path

metrics_path = Path(__file__).resolve().parents[1] / "ResNet" / "resnet_fake" / "metrics.py"
spec = importlib.util.spec_from_file_location("resnet_metrics", metrics_path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_report_includes_macro_metrics_and_error_rates():
    result = metrics.classification_report([0, 0, 1, 1], [0.9, 0.1, 0.8, 0.2])
    assert result["macro_f1"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["false_negative_rate"] == 0.5
