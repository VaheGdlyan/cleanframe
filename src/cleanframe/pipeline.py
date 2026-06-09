from typing import Any
from .base import BaseRule
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
        # 1. Gather all decisions that have been approved
        approved_decisions = [d for d in plan.decisions if d.approved]
        if not approved_decisions:
            return df

        # 2. Sequentially route approved decisions through each registered rule
        current_df = df
        for rule in self.rules:
            # Group decisions belonging to this specific rule class or rule name
            rule_name = rule.__class__.__name__
            rule_decisions = [
                d for d in approved_decisions 
                if getattr(d, "rule", "") == rule_name or getattr(d, "rule_name", "") == rule_name
            ]
            
            # If this rule has approved actions, execute its working Narwhals transformation
            if rule_decisions:
                current_df = rule.transform(current_df, rule_decisions)
                
        return current_df

    def fit_transform(self, df: Any) -> Any:
        # 1. Run detection across all scouts
        plan = self.fit(df)
        
        # 2. Auto-approve every single decision for the pipeline execution
        for decision in plan.decisions:
            decision.approved = True
            
        # 3. Run the bulletproof transformation loop
        return self.transform(df, plan)