"""Errors for persisted WQ safety state."""


class StateIntegrityError(RuntimeError):
    """Existing state cannot be trusted; leave it intact for explicit repair."""
