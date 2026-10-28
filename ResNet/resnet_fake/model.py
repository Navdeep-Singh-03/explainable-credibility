"""ResNet-50 binary fake-image classifier.

Standard torchvision ResNet-50 with the 1000-class head replaced by ONE logit.
Convention used everywhere in this package:  sigmoid(logit) = fake score,
so 1 = higher predicted likelihood that the image is fake/misleading.

Training starts from ImageNet weights (transfer learning) and runs once; the
saved checkpoint is then reused for every new image without retraining.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(pretrained: bool = False) -> nn.Module:
    """ResNet-50 with a single-logit head.

    ``pretrained=True`` downloads the torchvision ImageNet weights (needs internet,
    ~100 MB, training only). Inference never needs it: the checkpoint has every weight.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
