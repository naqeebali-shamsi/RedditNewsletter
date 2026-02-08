"""
Circuit Breaker for Content Sources

Prevents repeated calls to failing sources by tracking failure counts
and temporarily disabling sources that exceed the failure threshold.

States:
    closed  - Normal operation, requests pass through
    open    - Source disabled, requests short-circuited
    half-open - Single test request allowed to check recovery
"""

from dataclasses import dataclass, field
from time import time


@dataclass
class SourceCircuit:
    """Circuit breaker state for a single source."""
    source_type: str
    failure_count: int = 0
    failure_threshold: int = 3
    last_failure: float = 0
    cooldown_seconds: float = 300  # 5 minutes
    state: str = "closed"  # closed, open, half-open

    def record_success(self):
        """Reset circuit on successful request."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure = time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def should_attempt(self) -> bool:
        """Check if a request should be attempted."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if time() - self.last_failure > self.cooldown_seconds:
                self.state = "half-open"
                return True
            return False
        return True  # half-open: try once


class AllSourcesFailedError(Exception):
    """Raised when all configured sources fail to fetch content."""
    pass
