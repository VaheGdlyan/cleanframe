import narwhals as nw
from typing import Any
from dataclasses import dataclass
from ..base import BaseRule
from ..types import Decision

@dataclass
class ConsistencyConstraint:
    """
    Consistency constraint for cross-column verification.

    Attributes:
        name: Name of the constraint.
        condition: Vectorized Narwhals boolean expression evaluating to True for violations.
        error_msg: Error message to raise or log upon violation.
        action: Enforcement action. Supported values are:
            - "warn": Log the violation in telemetry but do not alter the dataset (default).
            - "drop": Filter out and remove rows violating the condition during transformation.
    """
    name: str
    condition: nw.Expr
    error_msg: str
    action: str = "warn"


class CrossColumnConsistencyRule(BaseRule):
    def __init__(self, constraints: list[ConsistencyConstraint]) -> None:
        self.constraints = constraints

    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        ndf = nw.from_native(df)
        nrows = len(ndf)
        if nrows == 0:
            return []

        decisions: list[Decision] = []
        for constraint in self.constraints:
            mask_df = ndf.select(constraint.condition.alias("violated"))
            violation_count = int(mask_df["violated"].sum())
            if violation_count > 0:
                decisions.append(
                    Decision(
                        rule_name=self.name,
                        column=constraint.name,
                        action="drop" if constraint.action == "drop" else "flag_violation",
                        parameters={
                            "constraint_name": constraint.name,
                            "violation_count": violation_count,
                            "error_msg": constraint.error_msg,
                            "constraint_action": constraint.action,
                        },
                        signal_strength=violation_count / nrows,
                        rationale=(
                            f"{violation_count} rows failed consistency rule '{constraint.name}': "
                            f"{constraint.error_msg}"
                        ),
                        approved=True,
                    )
                )
        return decisions

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        ndf = nw.from_native(df)
        approved_cols = {d.column for d in decisions if d.approved}
        for constraint in self.constraints:
            if constraint.name in approved_cols and constraint.action == "drop":
                # Condition evaluates to True for violations, so keep rows where it is False
                ndf = ndf.filter(~constraint.condition)
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        if not decisions:
            return "No consistency violations detected."
        lines = []
        for d in decisions:
            constraint_name = d.parameters.get("constraint_name", "")
            violation_count = d.parameters.get("violation_count", 0)
            c_action = d.parameters.get("constraint_action", "warn")
            if c_action == "drop":
                lines.append(
                    f"Constraint '{constraint_name}' violated by {violation_count} rows. "
                    f"Dropped {violation_count} rows."
                )
            else:
                lines.append(f"Constraint '{constraint_name}' violated by {violation_count} rows")
        return "\n".join(lines)
