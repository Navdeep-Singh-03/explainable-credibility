"""Typed errors. Everything derives from NewsVerifyError."""
from __future__ import annotations

from typing import Optional

from .models import FailureKind


class NewsVerifyError(Exception):
    """Base class."""


class InvalidConfigError(NewsVerifyError, ValueError):
    pass


class InvalidInputError(NewsVerifyError, ValueError):
    pass


class ClassifierUnavailableError(NewsVerifyError):
    """An optional classifier (e.g. an NLI model) cannot be loaded or used."""


class ProviderError(NewsVerifyError):
    """A provider/network failure. Never means 'the claim is false'; it means 'we could not look'."""

    def __init__(
        self,
        kind: FailureKind,
        message: str,
        *,
        retriable: bool = False,
        http_status: Optional[int] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retriable = retriable
        self.http_status = http_status
        self.retry_after = retry_after
