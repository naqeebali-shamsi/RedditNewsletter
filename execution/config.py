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
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _get_project_root() -> Path:
    """Get project root directory (works cross-platform)."""
    # This file is at execution/config.py, so parent.parent is project root
    return Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """Path configuration - all paths derived from PROJECT_ROOT."""
    PROJECT_ROOT: Path = field(default_factory=_get_project_root)

    @property
    def OUTPUT_DIR(self) -> Path:
        """Output directory for drafts and articles."""
        custom = os.getenv("GHOSTWRITER_OUTPUT_DIR")
        path = Path(custom) if custom else self.PROJECT_ROOT / "drafts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def TEMP_DIR(self) -> Path:
        """Temporary directory for intermediate files."""
        custom = os.getenv("GHOSTWRITER_TEMP_DIR")
        path = Path(custom) if custom else self.PROJECT_ROOT / ".tmp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def LOGS_DIR(self) -> Path:
        """Logs directory."""
        path = self.PROJECT_ROOT / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def DIRECTIVES_DIR(self) -> Path:
        """Directives directory."""
        return self.PROJECT_ROOT / "directives"

    @property
    def EXECUTION_DIR(self) -> Path:
        """Execution scripts directory."""
        return self.PROJECT_ROOT / "execution"


@dataclass
class APIConfig:
    """API keys and endpoints."""

    @property
    def GROQ_API_KEY(self) -> Optional[str]:
        return os.getenv("GROQ_API_KEY")

    @property
    def GOOGLE_API_KEY(self) -> Optional[str]:
        return os.getenv("GOOGLE_API_KEY")

    @property
    def PERPLEXITY_API_KEY(self) -> Optional[str]:
        return os.getenv("PERPLEXITY_API_KEY")

    @property
    def ANTHROPIC_API_KEY(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    @property
    def GMAIL_CREDENTIALS_PATH(self) -> Optional[str]:
        return os.getenv("GMAIL_CREDENTIALS_PATH")

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


@dataclass
class QualityConfig:
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
    ESCALATION_REASONS = {
        "MAX_ITERATIONS": "Maximum revision iterations reached without passing quality gate",
        "FALSE_CLAIMS": "Content contains verified false claims that cannot be auto-corrected",
        "CRITICAL_FAILURE": "Critical editorial failure detected (WSJ showstopper)",
        "LOW_SCORE": "Quality score below acceptable threshold after revisions",
        "VOICE_VIOLATION": "Persistent voice/attribution violations",
        "KILL_PHRASE": "Kill phrase detected that requires human judgment",
        "VERIFICATION_FAILED": "Fact verification system failed - manual review needed"
    }

    # Auto-escalation triggers
    FALSE_CLAIM_AUTO_ESCALATE: bool = True
    WSJ_FAILURE_AUTO_ESCALATE: bool = True
    MIN_SCORE_FOR_AUTO_APPROVE: float = 8.0  # Score required for auto-approval without human review


@dataclass
class ModelConfig:
    """Model selection configuration."""

    # Default models by task
    DEFAULT_WRITER_MODEL: str = field(
        default_factory=lambda: os.getenv("DEFAULT_WRITER_MODEL", "llama-3.3-70b-versatile")
    )
    DEFAULT_CRITIC_MODEL: str = field(
        default_factory=lambda: os.getenv("DEFAULT_CRITIC_MODEL", "llama-3.3-70b-versatile")
    )
    DEFAULT_EDITOR_MODEL: str = field(
        default_factory=lambda: os.getenv("DEFAULT_EDITOR_MODEL", "llama-3.3-70b-versatile")
    )

    # Research models (with web search capability)
    RESEARCH_MODEL_PRIMARY: str = "gemini-2.0-flash"
    RESEARCH_MODEL_FALLBACK: str = "sonar-pro"

    # Multi-model panel configuration
    ETHICS_REVIEWER_MODEL: str = "claude-sonnet-4-20250514"
    STRUCTURE_REVIEWER_MODEL: str = "gpt-4o"
    FACT_REVIEWER_MODEL: str = "gemini-2.0-flash"

    # Cost tiers
    TIER_SIMPLE: Dict[str, str] = field(default_factory=lambda: {
        "anthropic": "claude-3-haiku-20240307",
        "openai": "gpt-4o-mini",
        "google": "gemini-2.0-flash",
    })
    TIER_MEDIUM: Dict[str, str] = field(default_factory=lambda: {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "google": "gemini-2.0-pro",
    })
    TIER_COMPLEX: Dict[str, str] = field(default_factory=lambda: {
        "anthropic": "claude-opus-4-20250514",
        "openai": "gpt-4",
        "google": "gemini-2.0-ultra",
    })


@dataclass
class VoiceConfig:
    """Voice and style configuration."""

    # Voice types
    VOICE_EXTERNAL: str = "Journalist Observer"  # Reporting on others' work
    VOICE_INTERNAL: str = "Practitioner Owner"   # Sharing own experience

    # Publication styles
    STYLE_WSJ: str = "wsj"      # Four Showstoppers, nut graph
    STYLE_BBC: str = "bbc"      # Impartial, assertive
    STYLE_CBC: str = "cbc"      # Intimate, conversational
    STYLE_CNN: str = "cnn"      # Facts First
    STYLE_MEDIUM: str = "medium"  # Engaging, hook-driven


@dataclass
class GhostWriterConfig:
    """Main configuration class combining all config sections."""

    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)

    # Application metadata
    APP_NAME: str = "GhostWriter"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = field(
        default_factory=lambda: os.getenv("GHOSTWRITER_ENV", "development")
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
