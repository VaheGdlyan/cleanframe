import narwhals as nw
from typing import Any
from dataclasses import dataclass
from ..base import BaseRule
from ..types import Decision

@dataclass
class ConsistencyConstraint:
    name: str
    condition: nw.Expr
    error_msg: str

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
                        action="flag_violation",
                        parameters={
                            "constraint_name": constraint.name,
                            "violation_count": violation_count,
                            "error_msg": constraint.error_msg
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
        return df

    def explain(self, decisions: list[Decision]) -> str:
        if not decisions:
            return "No consistency violations detected."
        lines = []
        for d in decisions:
            constraint_name = d.parameters.get("constraint_name", "")
            violation_count = d.parameters.get("violation_count", 0)
            lines.append(f"Constraint '{constraint_name}' violated by {violation_count} rows")
        return "\n".join(lines)
