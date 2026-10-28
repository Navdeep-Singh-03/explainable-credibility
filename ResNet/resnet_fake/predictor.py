"""Public inference interface: score ONE image with a trained checkpoint, no retraining.

    predictor = ResNetPredictor("resnet50_fakeddit.pt")
    result = predictor.predict("photo.jpg")      # path, bytes or PIL.Image
    result.fake_score                            # float in [0, 1]; 1 = more likely fake
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch

from .exceptions import (
    InferenceError,
    InvalidInputError,
    ModelLoadError,
    ModelWeightsNotFoundError,
    ResNetFakeError,
)
from .model import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, build_model
from .preprocessing import eval_transform, load_image

SCORE_NOTE = (
    "fake_score = sigmoid(logit) is in [0, 1]; 1 = higher predicted likelihood of a fake/misleading image. "
    "It is a ranking score, not a verified probability, unless score_calibrated is True."
)

MODEL_INFO: Dict[str, Any] = {
    "name": "ResNet-50 binary fake-image classifier (ImageNet transfer learning)",
    "training_data": "Fakeddit multimodal_only samples, 2-way label (distant supervision from subreddit)",
    "weights": "user-trained checkpoint produced by train.py",
}


@dataclass
class ImagePrediction:
    fake_score: float  # final score in [0, 1]; 1 = higher predicted likelihood of fake
    raw_fake_score: float  # before any caller-supplied calibrator
    score_calibrated: bool  # True only if the caller supplied a calibrator
    logit: float
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    score_note: str = SCORE_NOTE
    model_info: Dict[str, Any] = field(default_factory=lambda: dict(MODEL_INFO))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_checkpoint(path: str) -> tuple:
    """Return ``(state_dict, meta)``; accepts our ``{'state_dict','meta'}`` format or a bare state_dict."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)  # refuses arbitrary pickles
    except Exception as exc:
        raise ModelLoadError(f"could not read checkpoint {path!r} ({type(exc).__name__}: {exc})") from exc
    meta: Dict[str, Any] = {}
    if isinstance(obj, Mapping) and "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
        meta = dict(obj.get("meta") or {})
        obj = obj["state_dict"]
    if not isinstance(obj, Mapping) or not obj or not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise ModelLoadError(f"checkpoint {path!r} does not contain a state_dict of tensors")
    return obj, meta


class ResNetPredictor:
    """Loads a trained checkpoint once and scores single images.

    Parameters
    ----------
    weights_path: checkpoint written by ``train.py``.
    device: default CUDA if available else CPU.
    calibrator: optional ``float -> float`` fitted by you on validation data; only sets
        ``score_calibrated=True``, the module cannot judge its quality.
    """

    def __init__(
        self,
        weights_path: Union[str, "os.PathLike[str]", None],
        *,
        device: Optional[Union[str, torch.device]] = None,
        calibrator: Optional[Callable[[float], float]] = None,
    ):
        if calibrator is not None and not callable(calibrator):
            raise InvalidInputError("calibrator must be callable (float -> float) or None")
        self._calibrator = calibrator

        if weights_path is None or str(weights_path) == "":
            raise ModelWeightsNotFoundError("weights_path is required: train the model first with train.py (see README).")
        path = os.fspath(weights_path)
        if not os.path.isfile(path):
            raise ModelWeightsNotFoundError(f"checkpoint not found at {path!r}: train the model first with train.py (see README).")
        self.weights_path = path

        state, self.meta = read_checkpoint(path)
        self.image_size = int(self.meta.get("image_size", IMAGE_SIZE))
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_model(pretrained=False)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ModelLoadError(f"checkpoint {path!r} does not match the ResNet-50 single-logit architecture ({exc})") from exc
        self.model = model.to(self.device).eval()  # eval(): BatchNorm uses running stats -> deterministic
        self.transform = eval_transform(
            self.image_size, tuple(self.meta.get("mean", IMAGENET_MEAN)), tuple(self.meta.get("std", IMAGENET_STD))
        )

    def score_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        """Logits for an already-transformed batch (used by evaluation; BatchNorm is in eval mode so rows are independent)."""
        with torch.inference_mode():
            return self.model(batch.to(self.device)).squeeze(1).float().cpu()

    def predict(self, image: Any) -> ImagePrediction:
        """Score ONE image.

        Raises ``InvalidImageError`` for unusable images and ``InferenceError`` if the forward pass fails.
        """
        img, info = load_image(image)
        try:
            logit = float(self.score_tensor(self.transform(img).unsqueeze(0))[0])
        except ResNetFakeError:
            raise
        except Exception as exc:  # RuntimeError, CUDA OOM, ...
            raise InferenceError(f"ResNet inference failed ({type(exc).__name__}: {exc})") from exc
        if not math.isfinite(logit):
            raise InferenceError("model produced a non-finite logit (NaN/inf)")

        raw = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
        raw = min(1.0, max(0.0, raw))
        final = self._apply_calibrator(raw)

        warnings: List[str] = []
        if info["low_resolution"]:
            warnings.append(f"low-resolution image ({info['original_size'][0]}x{info['original_size'][1]}); score is less reliable")
        if self._calibrator is None:
            warnings.append("score is not calibrated: do not read it as a probability")
        warnings.append("model was trained on Reddit (Fakeddit) images with subreddit-derived labels; it may not transfer to other image sources")
        return ImagePrediction(
            fake_score=final,
            raw_fake_score=raw,
            score_calibrated=self._calibrator is not None,
            logit=logit,
            preprocessing={**info, "model_input_size": self.image_size},
            warnings=warnings,
        )

    def _apply_calibrator(self, raw: float) -> float:
        if self._calibrator is None:
            return raw
        try:
            value = float(self._calibrator(raw))
        except Exception as exc:
            raise InferenceError(f"calibrator failed ({type(exc).__name__}: {exc})") from exc
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise InferenceError(f"calibrator returned {value!r}, which is not in [0, 1]")
        return value
