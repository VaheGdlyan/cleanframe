from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class TelemetryEvent:
    """
    Represents a structured, machine-readable telemetry event emitted
    by the cleanframe data pipeline.
    """

    event_type: str
    timestamp: str  # ISO 8601 string
    run_id: str
    rule_name: str | None
    column: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary representation."""
        return asdict(self)


@runtime_checkable
class TelemetrySink(Protocol):
    """
    Protocol defining an external sink that can receive TelemetryEvents.
    """

    def emit(self, event: TelemetryEvent) -> None:
        """
        Process/route a TelemetryEvent to an external system.

        Args:
            event: The TelemetryEvent object to emit.
        """
        ...
