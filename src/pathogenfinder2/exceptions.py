"""
Custom exception hierarchy for PathogenFinder2.

All library-level errors inherit from :exc:`PathogenFinderError` so callers
can catch the full family with a single ``except PathogenFinderError`` clause
while still being able to handle specific failure modes individually.
"""


class PathogenFinderError(Exception):
    """Base class for all PathogenFinder2 errors."""


class ConfigurationError(PathogenFinderError):
    """Raised when a configuration file is missing required keys or has invalid values."""


class EmbeddingError(PathogenFinderError):
    """Raised when ProtT5 embedding or HDF5 file loading fails."""


class WeightLoadError(PathogenFinderError):
    """Raised when model weight files are missing or incompatible."""


class PredictionError(PathogenFinderError):
    """Raised when the inference pipeline encounters an unrecoverable error."""


class ExternalToolError(PathogenFinderError):
    """Raised when an external tool (Prodigal, Diamond) fails or is not found."""
