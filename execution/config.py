"""
Centralized Configuration Management for GhostWriter.

This module provides a single source of truth for all configuration,
eliminating hardcoded paths and centralizing environment variables.

Usage:
    from execution.config import config
    output_path = config.OUTPUT_DIR / "article.md"
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    """Get project root directory (works cross-platform)."""
    # This file is at execution/config.py, so parent.parent is project root
    return Path(__file__).resolve().parent.parent


class PathConfig(BaseModel):
    """Path configuration - all paths derived from PROJECT_ROOT."""
    PROJECT_ROOT: Path = Field(default_factory=_get_project_root)

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def OUTPUT_DIR(self) -> Path:
        """Output directory for drafts and articles."""
        custom = os.getenv("GHOSTWRITER_OUTPUT_DIR")
        path = Path(custom) if custom else self.PROJECT_ROOT / "drafts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def TEMP_DIR(self) -> Path:
        """Temporary directory for intermediate files."""
        custom = os.getenv("GHOSTWRITER_TEMP_DIR")
        path = Path(custom) if custom else self.PROJECT_ROOT / ".tmp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def LOGS_DIR(self) -> Path:
        """Logs directory."""
        path = self.PROJECT_ROOT / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def DIRECTIVES_DIR(self) -> Path:
        """Directives directory."""
        return self.PROJECT_ROOT / "directives"

    @computed_field
    @property
    def EXECUTION_DIR(self) -> Path:
        """Execution scripts directory."""
        return self.PROJECT_ROOT / "execution"


class APIConfig(BaseSettings):
    """API keys and endpoints."""

    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    PERPLEXITY_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GMAIL_CREDENTIALS_PATH: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def has_key(self, provider: str) -> bool:
        """Check if API key is available for provider."""
        key_map = {
            "groq": self.GROQ_API_KEY,
            "google": self.GOOGLE_API_KEY,
            "gemini": self.GOOGLE_API_KEY,
            "perplexity": self.PERPLEXITY_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
            "claude": self.ANTHROPIC_API_KEY,
            "openai": self.OPENAI_API_KEY,
            "gpt": self.OPENAI_API_KEY,
        }
        return bool(key_map.get(provider.lower()))


class QualityConfig(BaseModel):
    """Quality gate configuration."""

    # Score thresholds
    PASS_THRESHOLD: float = 7.0
    ESCALATION_THRESHOLD: float = 6.0
    KILL_PHRASE_MAX_SCORE: float = 4.0

    # Iteration limits
    MAX_ITERATIONS: int = 3
    MAX_REVISION_ATTEMPTS: int = 2

    # Verification requirements
    FACT_VERIFICATION_REQUIRED: bool = True
    MULTI_MODEL_REVIEW_REQUIRED: bool = True
    HUMAN_REVIEW_REQUIRED_FOR_PUBLISH: bool = True

    # Minimum sources
    MIN_VERIFIED_FACTS: int = 3
    MAX_UNVERIFIED_CLAIMS: int = 1

    # Escalation reasons (for logging and UI)
    ESCALATION_REASONS: Dict[str, str] = Field(default={
        "MAX_ITERATIONS": "Maximum revision iterations reached without passing quality gate",
        "FALSE_CLAIMS": "Content contains verified false claims that cannot be auto-corrected",
        "CRITICAL_FAILURE": "Critical editorial failure detected (WSJ showstopper)",
        "LOW_SCORE": "Quality score below acceptable threshold after revisions",
        "VOICE_VIOLATION": "Persistent voice/attribution violations",
        "KILL_PHRASE": "Kill phrase detected that requires human judgment",
        "VERIFICATION_FAILED": "Fact verification system failed - manual review needed"
    })

    # Auto-escalation triggers
    FALSE_CLAIM_AUTO_ESCALATE: bool = True
    WSJ_FAILURE_AUTO_ESCALATE: bool = True
    MIN_SCORE_FOR_AUTO_APPROVE: float = 8.0


class ModelConfig(BaseSettings):
    """Model selection configuration."""

    # Default models by task
    DEFAULT_WRITER_MODEL: str = "llama-3.3-70b-versatile"
    DEFAULT_CRITIC_MODEL: str = "llama-3.3-70b-versatile"
    DEFAULT_EDITOR_MODEL: str = "llama-3.3-70b-versatile"

    # Fast/lightweight model for specialists and simple tasks
    DEFAULT_FAST_MODEL: str = "llama-3.1-8b-instant"

    # Base model used as fallback across agents
    DEFAULT_BASE_MODEL: str = "gemini-2.0-flash-exp"

    # Research models (with web search capability)
    RESEARCH_MODEL_PRIMARY: str = "gemini-2.0-flash"
    RESEARCH_MODEL_FALLBACK: str = "sonar-pro"
    GEMINI_RESEARCH_MODEL: str = "gemini-3-flash-preview"

    # Content evaluation model
    CONTENT_EVAL_MODEL: str = "gemini-1.5-flash"

    # Multi-model panel configuration
    ETHICS_REVIEWER_MODEL: str = "claude-sonnet-4-20250514"
    STRUCTURE_REVIEWER_MODEL: str = "gpt-4o"
    FACT_REVIEWER_MODEL: str = "gemini-2.0-flash"

    # Cost tiers
    TIER_SIMPLE: Dict[str, str] = Field(default={
        "anthropic": "claude-3-haiku-20240307",
        "openai": "gpt-4o-mini",
        "google": "gemini-2.0-flash",
    })
    TIER_MEDIUM: Dict[str, str] = Field(default={
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "google": "gemini-2.0-pro",
    })
    TIER_COMPLEX: Dict[str, str] = Field(default={
        "anthropic": "claude-opus-4-20250514",
        "openai": "gpt-4",
        "google": "gemini-2.0-ultra",
    })

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class VoiceConfig(BaseModel):
    """Voice and style configuration."""

    # Voice types
    VOICE_EXTERNAL: str = "Journalist Observer"
    VOICE_INTERNAL: str = "Practitioner Owner"

    # Publication styles
    STYLE_WSJ: str = "wsj"
    STYLE_BBC: str = "bbc"
    STYLE_CBC: str = "cbc"
    STYLE_CNN: str = "cnn"
    STYLE_MEDIUM: str = "medium"


class GhostWriterConfig(BaseSettings):
    """Main configuration class combining all config sections."""

    paths: PathConfig = Field(default_factory=PathConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    # Application metadata
    APP_NAME: str = "GhostWriter"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", alias="GHOSTWRITER_ENV")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        issues = []
        warnings = []

        # Check for at least one fact verification API
        has_fact_api = any([
            self.api.GOOGLE_API_KEY,
            self.api.PERPLEXITY_API_KEY,
            self.api.ANTHROPIC_API_KEY,
        ])

        if not has_fact_api and self.quality.FACT_VERIFICATION_REQUIRED:
            issues.append(
                "CRITICAL: No fact verification API available. "
                "Set GOOGLE_API_KEY, PERPLEXITY_API_KEY, or ANTHROPIC_API_KEY"
            )

        # Check for writer API
        if not self.api.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY not set - using alternative models")

        # Check paths exist
        if not self.paths.PROJECT_ROOT.exists():
            issues.append(f"Project root not found: {self.paths.PROJECT_ROOT}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "environment": self.ENVIRONMENT,
            "fact_verification_available": has_fact_api,
        }

    def __repr__(self) -> str:
        return (
            f"GhostWriterConfig(\n"
            f"  environment={self.ENVIRONMENT},\n"
            f"  project_root={self.paths.PROJECT_ROOT},\n"
            f"  output_dir={self.paths.OUTPUT_DIR},\n"
            f"  fact_verification_required={self.quality.FACT_VERIFICATION_REQUIRED}\n"
            f")"
        )


# Singleton instance
config = GhostWriterConfig()


# Convenience exports
PROJECT_ROOT = config.paths.PROJECT_ROOT
OUTPUT_DIR = config.paths.OUTPUT_DIR
TEMP_DIR = config.paths.TEMP_DIR
LOGS_DIR = config.paths.LOGS_DIR


def get_config() -> GhostWriterConfig:
    """Get the configuration singleton."""
    return config


def validate_config() -> bool:
    """Validate configuration and print status."""
    result = config.validate()

    if result["issues"]:
        print("Configuration Issues:")
        for issue in result["issues"]:
            print(f"  - {issue}")

    if result["warnings"]:
        print("Configuration Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")

    return result["valid"]


if __name__ == "__main__":
    # Print configuration when run directly
    print(config)
    print()
    validation = config.validate()
    print(f"Valid: {validation['valid']}")
    print(f"Environment: {validation['environment']}")
    print(f"Fact Verification Available: {validation['fact_verification_available']}")
