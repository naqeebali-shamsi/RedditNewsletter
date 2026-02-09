"""Typed exceptions for the GhostWriter pipeline."""


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class ResearchError(PipelineError):
    """Failed to research topic or facts."""
    pass


class WriterError(PipelineError):
    """Failed to generate draft."""
    pass


class VerificationError(PipelineError):
    """Failed to verify facts."""
    pass


class QualityGateError(PipelineError):
    """Article failed quality gate."""
    pass


class StyleError(PipelineError):
    """Article failed style enforcement."""
    pass
