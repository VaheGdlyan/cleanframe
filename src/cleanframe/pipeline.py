from typing import Any

from .base import BaseRule
from .engine import ExecutionEngine
from .plan import CleaningPlan
from .rules import (
    CardinalityChecker,
    DuplicateHandler,
    NullHandler,
    OutlierHandler,
    SchemaCaster,
)
from .types import Decision


class DataCleaner:
    """
    Core orchestrator for running a series of data cleaning rules.

    If no custom rules are supplied, the default sequence is:
    SchemaCaster, DuplicateHandler, NullHandler, OutlierHandler, CardinalityChecker.
    """

    def __init__(self, rules: list[BaseRule] | None = None) -> None:
        if rules is not None:
            self.rules = rules
        else:
            self.rules = [
                SchemaCaster(),
                DuplicateHandler(),
                NullHandler(),
                OutlierHandler(),
                CardinalityChecker(),
            ]

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
            CleaningPlan containing all collected Decision objects.
        """
        decisions: list[Decision] = []
        for rule in self.rules:
            rule_name = type(rule).__name__
            params = params_map.get(rule_name, {}) if params_map else {}
            decisions.extend(rule.detect(df, params))
        return CleaningPlan(decisions)

    def transform(self, df: Any, plan: CleaningPlan) -> Any:
        """Apply approved decisions in the plan via the execution engine."""
        approved = [d for d in plan.decisions if d.approved]
        engine = ExecutionEngine()
        return engine.execute(df, CleaningPlan(approved))

    def fit_transform(
        self,
        df: Any,
        params_map: dict[str, dict[str, Any]] | None = None,
    ) -> Any:
        """Fit rules and transform the data according to the resulting plan."""
        plan = self.fit(df, params_map)
        return self.transform(df, plan)
