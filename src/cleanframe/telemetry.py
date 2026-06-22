class AuditReport:
    """
    Telemetry report for a cleaning operation.

    Tracks dataset shape, per-rule mutation summaries, and execution time.
    """

    def __init__(
        self,
        initial_shape: tuple[int, int],
        final_shape: tuple[int, int],
        mutations: dict[str, list[str]],
        execution_time_ms: float,
        drift_alerts: list[str] | None = None,
        leakage_warnings: list[str] | None = None,
        consistency_warnings: list[str] | None = None,
    ) -> None:
        self.initial_shape = initial_shape
        self.final_shape = final_shape
        self.mutations = mutations
        self.execution_time_ms = execution_time_ms
        self.drift_alerts: list[str] = drift_alerts if drift_alerts is not None else []
        self.leakage_warnings: list[str] = leakage_warnings if leakage_warnings is not None else []
        self.consistency_warnings: list[str] = consistency_warnings if consistency_warnings is not None else []

    def to_dict(self) -> dict[str, object]:
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
        print(f"Rows change:   {row_diff:+d}")
        print(f"Cols change:   {col_diff:+d}")
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

