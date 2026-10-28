"""Held-out evaluation. Run this ONCE on the test CSV after training is finished."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

from .data import ImageCsvDataset
from .metrics import classification_report
from .predictor import ResNetPredictor


def evaluate(
    weights_path: str,
    csv_path: str,
    root: str,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    threshold: float = 0.5,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Score every image of ``csv_path`` with the trained model and report fake-class metrics.

    The threshold is fixed (default 0.5) and never tuned here; if you want a different one,
    choose it on the VALIDATION csv, not the test csv.
    """
    predictor = ResNetPredictor(weights_path, device=device)
    dataset = ImageCsvDataset(csv_path, root, predictor.transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits = torch.cat([predictor.score_tensor(x) for x, _ in loader])
    scores = torch.sigmoid(logits).tolist()
    report = classification_report(dataset.labels, scores, threshold=threshold)
    report.update({"csv": csv_path, "checkpoint": weights_path})
    return report
