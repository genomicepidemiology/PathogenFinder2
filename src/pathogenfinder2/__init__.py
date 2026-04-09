# Define metadata
__version__ = "0.7.1"

from pathogenfinder2.main import PathogenFinder2
from pathogenfinder2.exceptions import (
    PathogenFinderError,
    ConfigurationError,
    EmbeddingError,
    WeightLoadError,
    PredictionError,
    ExternalToolError,
)
