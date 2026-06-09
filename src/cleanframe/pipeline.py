import time
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
from .telemetry import AuditReport
from .types import Decision

_BUILTIN_RULES = frozenset({
    "SchemaCaster",
    "DuplicateHandler",
    "NullHandler",
    "OutlierHandler",
    "CardinalityChecker",
})

_DEFAULT_RULES: list[BaseRule] = [
    SchemaCaster(),
    DuplicateHandler(),
    NullHandler(),
    OutlierHandler(),
    CardinalityChecker(),
]


def _dataframe_shape(df: Any) -> tuple[int, int]:
    if hasattr(df, "shape"):
        shape = tuple(df.shape)
        if len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return len(df), len(getattr(df, "columns", []))


def _decision_summary(decision: Decision) -> str:
    action = decision.action
    column = decision.column

    if action == "drop_column":
        return f"Dropped column '{column}'"
    if action == "flag_id":
        return f"Flagged '{column}' as identifier"
    if action == "drop_duplicates":
        return "Removed duplicate rows"
    if action == "clip":
        return f"Clipped outliers in '{column}'"
    if action == "cast":
        return f"Cast '{column}'"
    if action in {"median", "mode"}:
        return f"Imputed nulls in '{column}' ({action})"
    return f"{action.replace('_', ' ').capitalize()} on '{column}'"


def _decisions_for_rule(
    decisions: list[Decision],
    rule_name: str,
) -> list[Decision]:
    return [
        d
        for d in decisions
        if getattr(d, "rule_name", "") == rule_name
        or getattr(d, "rule", "") == rule_name
    ]


def _mutation_entries(rule: BaseRule, decisions: list[Decision]) -> list[str]:
    if type(rule).__name__ in _BUILTIN_RULES:
        return [_decision_summary(d) for d in decisions]
    return [rule.explain(decisions)]


class DataCleaner:
    """
    Core orchestrator for running a series of data cleaning rules.

    Default rule sequence:
    SchemaCaster → DuplicateHandler → NullHandler → OutlierHandler → CardinalityChecker
    """

    def __init__(self, rules: list[BaseRule] | None = None) -> None:
        self.rules: list[BaseRule] = list(rules) if rules is not None else list(_DEFAULT_RULES)
        self.last_report: AuditReport | None = None

    def register_rule(self, rule: BaseRule) -> None:
        """Append a custom rule to the active execution registry."""
        if not isinstance(rule, BaseRule):
            raise TypeError("Registered rule must inherit from BaseRule")
        self.rules.append(rule)

    def fit(
        self,
        df: Any,
        params_map: dict[str, dict[str, Any]] | None = None,
    ) -> CleaningPlan:
        """
        Run each rule's detect method and aggregate the resulting decisions.

        Args:
            df: Input dataset (pandas or polars DataFrame).
            params_map: Optional mapping from rule class name to parameter dict.

        Returns:
            CleaningPlan containing all collected decisions.
        """
        decisions: list[Decision] = []
        for rule in self.rules:
            rule_name = type(rule).__name__
            params = params_map.get(rule_name, {}) if params_map else {}
            detected = rule.detect(df, params)
            if not isinstance(detected, list) or not all(
                isinstance(d, Decision) for d in detected
            ):
                raise TypeError(
                    f"Rule {rule_name}.detect() must return a list of Decision"
                )
            decisions.extend(detected)
        return CleaningPlan(decisions)

    def transform(self, df: Any, plan: CleaningPlan) -> Any:
        start_time = time.perf_counter()
        initial_shape = _dataframe_shape(df)
        mutations: dict[str, list[str]] = {}

        approved = [d for d in plan.decisions if d.approved]
        if not approved:
            self.last_report = AuditReport(
                initial_shape=initial_shape,
                final_shape=initial_shape,
                mutations=mutations,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            return df

        current_df = df
        for rule in self.rules:
            rule_name = type(rule).__name__
            rule_decisions = _decisions_for_rule(approved, rule_name)
            if not rule_decisions:
                continue

            mutations[rule.name] = _mutation_entries(rule, rule_decisions)
            current_df = rule.transform(current_df, rule_decisions)

        self.last_report = AuditReport(
            initial_shape=initial_shape,
            final_shape=_dataframe_shape(current_df),
            mutations=mutations,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )
        return current_df

    def fit_transform(self, df: Any) -> Any:
        plan = self.fit(df)
        for decision in plan.decisions:
            decision.approved = True
        return self.transform(df, plan)
