from typing import Any
from .events import TelemetryEvent


class AuditReport:
    """
    Telemetry report for a cleaning operation.

    Tracks dataset shape, per-rule mutation summaries, and execution time.
    Supports backward-compatible DX while being powered by TelemetryEvents.
    """

    def __init__(
        self,
        initial_shape: tuple[int, int],
        final_shape: tuple[int, int],
        execution_time_ms: float,
        mutations: dict[str, list[str]] | None = None,
        drift_alerts: list[str] | None = None,
        leakage_warnings: list[str] | None = None,
        consistency_warnings: list[str] | None = None,
        events: list[TelemetryEvent] | None = None,
    ) -> None:
        self.initial_shape = initial_shape
        self.final_shape = final_shape
        self.execution_time_ms = execution_time_ms
        self.events = events if events is not None else []

        if events is not None:
            # Dynamically compile attributes from structured events
            self.drift_alerts = [
                e.payload["alert"]
                for e in events
                if e.event_type == "drift_alert" and "alert" in e.payload
            ]
            self.leakage_warnings = [
                e.payload["warning"]
                for e in events
                if e.event_type == "target_leakage" and "warning" in e.payload
            ]
            self.consistency_warnings = [
                e.payload["warning"]
                for e in events
                if e.event_type == "constraint_violation" and "warning" in e.payload
            ]

            self.mutations: dict[str, list[str]] = {}
            for e in events:
                if e.event_type == "rule_mutation":
                    rule = e.rule_name or "Unknown"
                    if rule not in self.mutations:
                        self.mutations[rule] = []
                    summary = e.payload.get("summary", "")
                    if summary and summary not in self.mutations[rule]:
                        self.mutations[rule].append(summary)
        else:
            self.drift_alerts = drift_alerts if drift_alerts is not None else []
            self.leakage_warnings = leakage_warnings if leakage_warnings is not None else []
            self.consistency_warnings = consistency_warnings if consistency_warnings is not None else []
            self.mutations = mutations if mutations is not None else {}

    def to_dict(self) -> dict[str, Any]:
        """Convert the report data to a dictionary."""
        return {
            "initial_shape": self.initial_shape,
            "final_shape": self.final_shape,
            "mutations": self.mutations,
            "execution_time_ms": self.execution_time_ms,
            "drift_alerts": self.drift_alerts,
            "leakage_warnings": self.leakage_warnings,
            "consistency_warnings": self.consistency_warnings,
        }

    def display(self) -> None:
        """Print a formatted terminal summary of the cleaning audit."""
        row_diff = self.final_shape[0] - self.initial_shape[0]
        col_diff = self.final_shape[1] - self.initial_shape[1]

        print("=" * 48)
        print("Audit Report Summary")
        print("=" * 48)
        print(
            f"Initial shape: rows={self.initial_shape[0]}, "
            f"columns={self.initial_shape[1]}"
        )
        print(
            f"Final shape:   rows={self.final_shape[0]}, "
            f"columns={self.final_shape[1]}"
        )
        print(
            f"Rows change:   {row_diff:+d}"
            if row_diff != 0
            else f"Rows change:   {row_diff}"
        )
        print(
            f"Cols change:   {col_diff:+d}"
            if col_diff != 0
            else f"Cols change:   {col_diff}"
        )
        print("-" * 48)
        print(f"Execution time: {self.execution_time_ms:.2f} ms")
        print("-" * 48)
        if self.leakage_warnings:
            print("TARGET LEAKAGE ALERTS:")
            for warn in self.leakage_warnings:
                print(f"  • {warn}")
            print("-" * 48)
        if self.consistency_warnings:
            print("CONSISTENCY ALERTS:")
            for alert in self.consistency_warnings:
                print(f"  • {alert}")
            print("-" * 48)
        if self.drift_alerts:
            print("DRIFT ALERTS:")
            for alert in self.drift_alerts:
                print(f"  • {alert}")
            print("-" * 48)
        print("Mutations Applied:")
        if not self.mutations:
            print("  (No mutations performed)")
        else:
            for rule, actions in self.mutations.items():
                print(f"\n  [{rule}]")
                for action in actions:
                    print(f"    - {action}")
        print("=" * 48)
