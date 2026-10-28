"""Exception hierarchy for the ResNet-50 image module (independent of the FDHN module)."""

from __future__ import annotations


class ResNetFakeError(Exception):
    """Base class for all errors raised deliberately by this package."""


class InvalidImageError(ResNetFakeError, ValueError):
    """The image is missing, empty, corrupt, unsupported, too small or too large."""


class InvalidInputError(ResNetFakeError, ValueError):
    """Some other argument is invalid (e.g. a bad calibrator)."""


class ModelWeightsNotFoundError(ResNetFakeError, FileNotFoundError):
    """No trained checkpoint exists at the given path (train one with train.py)."""


class ModelLoadError(ResNetFakeError):
    """The checkpoint could not be read or does not match the architecture."""


class InferenceError(ResNetFakeError):
    """The forward pass failed or produced an unusable (non-finite / out-of-range) result."""
