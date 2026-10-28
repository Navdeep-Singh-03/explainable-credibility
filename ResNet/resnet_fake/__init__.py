"""ResNet-50 fake-image module. Self-contained: imports nothing from other project components.

    from resnet_fake import ResNetPredictor
    result = ResNetPredictor("resnet50_fakeddit.pt").predict("photo.jpg")
    print(result.fake_score)
"""

from .exceptions import (
    InferenceError,
    InvalidImageError,
    InvalidInputError,
    ModelLoadError,
    ModelWeightsNotFoundError,
    ResNetFakeError,
)
from .predictor import ImagePrediction, ResNetPredictor

__all__ = [
    "ResNetPredictor",
    "ImagePrediction",
    "ResNetFakeError",
    "InvalidImageError",
    "InvalidInputError",
    "ModelWeightsNotFoundError",
    "ModelLoadError",
    "InferenceError",
]
__version__ = "0.1.0"
