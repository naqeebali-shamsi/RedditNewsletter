"""
Circuit Breaker for Content Sources

Wraps pybreaker to provide resilient circuit-breaking for content sources.
Preserves the original SourceCircuit API so all callers work unchanged.

States:
    closed    - Normal operation, requests pass through
    open      - Source disabled, requests short-circuited
    half-open - Single test request allowed to check recovery
"""

import logging
from datetime import datetime, timezone

import pybreaker

_logger = logging.getLogger("ghostwriter.sources.circuit_breaker")

# Map pybreaker internal states to our string state names
_STATE_MAP = {
    pybreaker.STATE_CLOSED: "closed",
    pybreaker.STATE_OPEN: "open",
    pybreaker.STATE_HALF_OPEN: "half-open",
}


class _StateChangeListener(pybreaker.CircuitBreakerListener):
    """Logs circuit breaker state transitions."""

    def state_change(self, cb, old_state, new_state):
        _logger.info(
            "Circuit '%s' state change: %s -> %s",
            cb.name,
            _STATE_MAP.get(old_state.name, old_state.name),
            _STATE_MAP.get(new_state.name, new_state.name),
        )

    def failure(self, cb, exc):
        _logger.debug("Circuit '%s' recorded failure: %s", cb.name, exc)

    def success(self, cb):
        _logger.debug("Circuit '%s' recorded success", cb.name)


class SourceCircuit:
    """Circuit breaker state for a single source.

    Drop-in replacement backed by pybreaker. Preserves the original
    constructor signature and public API (record_success, record_failure,
    should_attempt, state, failure_count).
    """

    def __init__(
        self,
        source_type: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 300,
    ):
        self.source_type = source_type
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self._breaker = pybreaker.CircuitBreaker(
            fail_max=failure_threshold,
            reset_timeout=cooldown_seconds,
            name=source_type,
            listeners=[_StateChangeListener()],
        )

    # -- Public API (unchanged from original) ----------------------------------

    def record_success(self):
        """Reset circuit on successful request."""
        # Call a trivially-succeeding function through the breaker so it
        # registers a success and resets internal counters.
        try:
            self._breaker.call(lambda: None)
        except pybreaker.CircuitBreakerError:
            # If the breaker is open we force-close it, matching original
            # behaviour where record_success() always resets.
            self._breaker.close()

    def record_failure(self):
        """Record a failure and potentially open the circuit."""
        try:
            self._breaker.call(self._raise_failure)
        except (pybreaker.CircuitBreakerError, _SimulatedFailure):
            pass

    def should_attempt(self) -> bool:
        """Check if a request should be attempted."""
        state_name = self._breaker.current_state
        if state_name == pybreaker.STATE_CLOSED:
            return True
        if state_name == pybreaker.STATE_OPEN:
            # Check if cooldown has elapsed by comparing opened_at + timeout
            opened_at = self._breaker._state_storage.opened_at
            if opened_at is not None:
                now = datetime.now(timezone.utc)
                elapsed = (now - opened_at).total_seconds()
                if elapsed >= self._breaker.reset_timeout:
                    # Cooldown expired - transition to half-open
                    self._breaker.half_open()
                    return True
            return False
        # half-open: allow one test request
        return True

    # -- Properties matching original dataclass fields -------------------------

    @property
    def state(self) -> str:
        """Current state as a string: 'closed', 'open', or 'half-open'."""
        return _STATE_MAP.get(self._breaker.current_state, "closed")

    @state.setter
    def state(self, value: str):
        """Allow direct state assignment for backward compatibility."""
        if value == "closed":
            self._breaker.close()
        elif value == "open":
            self._breaker.open()
        elif value == "half-open":
            self._breaker.half_open()

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._breaker.fail_counter

    @failure_count.setter
    def failure_count(self, value: int):
        """Allow direct assignment for backward compatibility.

        Note: Uses pybreaker internal _state_storage._fail_counter.
        Pinned to pybreaker ~= 1.4.
        """
        self._breaker._state_storage._fail_counter = value

    @property
    def last_failure(self) -> float:
        """Timestamp of the last failure (0 if none)."""
        opened_at = getattr(self._breaker._state_storage, 'opened_at', None)
        if opened_at is not None:
            return opened_at.timestamp()
        return 0

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _raise_failure():
        raise _SimulatedFailure("simulated")


class _SimulatedFailure(Exception):
    """Internal exception used to register failures with pybreaker."""
    pass


class AllSourcesFailedError(Exception):
    """Raised when all configured sources fail to fetch content."""
    pass
