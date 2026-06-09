from dataclasses import dataclass
from typing import Any


@dataclass
class Decision:
    """
    Represents a decision made by a data cleaning rule for a specific column.

    Attributes:
        rule_name: Name of the rule that generated this decision.
        column: The column to which the decision applies.
        action: Description of the action to be taken.
        parameters: Parameters for the action, structured as a mapping.
        signal_strength: A numeric value indicating confidence in this decision.
        rationale: Human-readable explanation for the decision.
        approved: Whether this decision is approved for application. Defaults to True.
    """

    rule_name: str
    column: str
    action: str
    parameters: dict[str, Any]
    signal_strength: float
    rationale: str
    approved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "column": self.column,
            "action": self.action,
            "parameters": self.parameters,
            "signal_strength": self.signal_strength,
            "rationale": self.rationale,
            "approved": self.approved,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        return cls(
            rule_name=data["rule_name"],
            column=data["column"],
            action=data["action"],
            parameters=data["parameters"],
            signal_strength=data["signal_strength"],
            rationale=data["rationale"],
            approved=data.get("approved", True),
        )
