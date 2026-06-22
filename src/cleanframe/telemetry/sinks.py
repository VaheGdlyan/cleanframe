import json
import logging
from pathlib import Path
from .events import TelemetryEvent


class LocalJsonLinesSink:
    """
    Saves structured telemetry events to a local file in JSON Lines (.jsonl) format.
    """

    def __init__(self, file_path: str | Path) -> None:
        """
        Initialize the sink with a file path.

        Args:
            file_path: Absolute or relative path to the output .jsonl file.
        """
        self.file_path = Path(file_path)

    def emit(self, event: TelemetryEvent) -> None:
        """Serialize the event as a JSON string and append it to the target file."""
        # Ensure parent directories exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        event_data = event.to_dict()
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event_data) + "\n")


class StandardLoggingSink:
    """
    Routes telemetry events to Python's standard logging library.
    """

    def __init__(self, logger_name: str = "cleanframe.telemetry") -> None:
        """
        Initialize the sink with a specific logger name.

        Args:
            logger_name: Name of the logger to retrieve/configure.
        """
        self.logger = logging.getLogger(logger_name)

    def emit(self, event: TelemetryEvent) -> None:
        """Log the event as a formatted string using appropriate severity level."""
        # Determine severity level based on event type
        level = logging.INFO
        if event.event_type in ("target_leakage", "drift_alert", "constraint_violation"):
            level = logging.WARNING

        msg = (
            f"Telemetry Event: [{event.event_type}] "
            f"run={event.run_id} rule={event.rule_name} col={event.column} "
            f"payload={event.payload}"
        )
        self.logger.log(level, msg)
