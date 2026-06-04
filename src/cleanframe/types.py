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
