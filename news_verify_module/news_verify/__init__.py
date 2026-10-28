"""External news-claim verification: retrieve evidence from outside sources and assess it per claim."""
__version__ = "0.1.0"

from .cache import MemoryCache, NullCache, SqliteCache
from .claims import ClaimExtractor, ExtractionResult, RuleBasedClaimExtractor
from .classifiers import FactCheckRatingClassifier, NLIClassifier, StanceClassifier, StanceResult
from .config import VerifierConfig
from .errors import (ClassifierUnavailableError, InvalidConfigError, InvalidInputError, NewsVerifyError,
                     ProviderError)
from .models import (Claim, ClaimResult, ClaimStatus, Evidence, EvidenceQuality, Failure, FailureKind,
                     InsufficientReason, ProviderRecord, QualityLevel, SourceTier, Stance, VerificationResult)
from .providers import (EvidenceProvider, GoogleFactCheckProvider, SearchContext, StaticEvidenceProvider,
                       WikipediaContextProvider)
from .reliability import SourceReliabilityRegistry
from .verifier import NewsVerifier

__all__ = [n for n in dir() if not n.startswith("_")]
