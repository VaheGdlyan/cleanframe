from typing import Any
from .types import Decision


class CleaningPlan:
    """
    Represents a plan comprising a set of data cleaning decisions.

    This class offers mechanisms to review, approve, reject, and summarize
    cleaning actions prior to their application.
    """

    def __init__(self, decisions: list[Decision]) -> None:
        """
        Initialize a CleaningPlan with a collection of Decision objects.

        Args:
            decisions: The list of data cleaning decisions to manage.
        """
        self.decisions = decisions

    def approve_all(self) -> None:
        """
        Approve all decisions in the plan.

        Sets the 'approved' flag to True for every stored Decision.
        """
        for decision in self.decisions:
            decision.approved = True

    def reject(self, rule_name: str, column: str) -> None:
        """
        Reject all decisions for the given rule and column.

        Args:
            rule_name: The name of the rule whose decisions should be rejected.
            column: The associated column for which to reject the decisions.
        """
        for decision in self.decisions:
            if decision.rule_name == rule_name and decision.column == column:
                decision.approved = False

    def summary(self) -> list[dict[str, Any]]:
        """
        Return a summary of all decisions in the plan for user inspection.

        Returns:
            A list of dictionaries, each describing a decision and its state.
        """
        return [
            {
                "rule_name": d.rule_name,
                "column": d.column,
                "action": d.action,
                "parameters": d.parameters,
                "signal_strength": d.signal_strength,
                "rationale": d.rationale,
                "approved": d.approved,
            }
            for d in self.decisions
        ]
