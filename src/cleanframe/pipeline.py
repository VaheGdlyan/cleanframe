from typing import Any
from .base import BaseRule
from .plan import CleaningPlan
from .types import Decision

class DataCleaner:
    """
    Core orchestrator for running a series of data cleaning rules.

    Given a list of BaseRule objects, this class coordinates the detection
    of data quality issues by invoking each rule and compiling their decisions
    into a structured CleaningPlan.

    Attributes:
        rules: The sequence of BaseRule objects to apply for detection.
    """

    def __init__(self, rules: list[BaseRule]) -> None:
        """
        Initialize the DataCleaner with an ordered set of cleaning rules.

        Args:
            rules: A list of BaseRule objects defining the cleaning logic.
        """
        self.rules = rules

    def fit(
        self,
        df: Any,
        params_map: dict[str, dict[str, Any]] | None = None,
    ) -> CleaningPlan:
        """
        Apply all rules' detect methods to the dataset and aggregate decisions.

        Args:
            df: The input dataset (e.g., a DataFrame) to analyze.
            params_map: Optional mapping from rule class name to its param dict.

        Returns:
            CleaningPlan: An object containing all collected Decision objects.
        """
        decisions: list[Decision] = []

        for rule in self.rules:
            rule_name = type(rule).__name__
            params = params_map[rule_name] if params_map and rule_name in params_map else {}
            rule_decisions = rule.detect(df, params)
            decisions.extend(rule_decisions)

        return CleaningPlan(decisions)