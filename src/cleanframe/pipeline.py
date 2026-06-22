import math
import time
from typing import Any

import narwhals as nw

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


def _compute_dataframe_stats(df: Any) -> dict[str, dict[str, float]]:
    ndf = nw.from_native(df)
    if ndf.shape[0] == 0:
        return {}

    exprs = []
    for col in ndf.columns:
        exprs.append(nw.col(col).is_null().mean().alias(f"{col}__null_ratio"))
        if ndf[col].dtype.is_numeric():
            exprs.append(nw.col(col).mean().alias(f"{col}__mean"))
            exprs.append(nw.col(col).std().alias(f"{col}__std_dev"))
        else:
            exprs.append(nw.col(col).n_unique().alias(f"{col}__unique_count"))

    if not exprs:
        return {}

    stats_df = ndf.select(*exprs)
    row_values = stats_df.row(0)
    flat_stats = dict(zip(stats_df.columns, row_values, strict=True))

    structured: dict[str, dict[str, float]] = {}
    for key, val in flat_stats.items():
        col, metric = key.split("__", 1)
        if col not in structured:
            structured[col] = {}
        val_float = float(val) if val is not None else 0.0
        if math.isnan(val_float):
            val_float = 0.0
        structured[col][metric] = val_float

    # Get unique categories for categorical columns to track if "new categories appear"
    for col in ndf.columns:
        if not ndf[col].dtype.is_numeric():
            unique_vals = ndf[col].unique().to_list()
            for val in unique_vals:
                if val is not None:
                    structured[col][f"cat:{val}"] = 1.0

    return structured


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
        target_col: str | None = None,
    ) -> CleaningPlan:
        """
        Run each rule's detect method and aggregate the resulting decisions.

        Args:
            df: Input dataset (pandas or polars DataFrame).
            params_map: Optional mapping from rule class name to parameter dict.
            target_col: Optional target variable to check for target leakage.

        Returns:
            CleaningPlan containing all collected decisions.
        """
        ndf = nw.from_native(df)
        if target_col is not None:
            assert target_col in ndf.columns, f"Target column '{target_col}' not found in dataset"

        baseline_stats = _compute_dataframe_stats(df)
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

        leakage_warnings: list[str] = []
        if target_col is not None:
            # 1. Semantic Risk
            target_lower = target_col.lower()
            for col in ndf.columns:
                if col.lower() != target_lower and target_lower in col.lower():
                    leakage_warnings.append(
                        f"HIGH RISK: Column '{col}' semantically leaks target '{target_col}'"
                    )

            # 2. Correlation Risk
            is_target_numeric = ndf[target_col].dtype.is_numeric() or ndf[target_col].dtype.is_boolean()
            if is_target_numeric:
                corr_exprs = []
                corr_cols = []
                for col in ndf.columns:
                    if col != target_col and (ndf[col].dtype.is_numeric() or ndf[col].dtype.is_boolean()):
                        corr_exprs.append(nw.corr(col, target_col).alias(f"{col}__corr"))
                        corr_cols.append(col)

                if corr_exprs:
                    corr_df = ndf.select(*corr_exprs)
                    row_vals = corr_df.row(0)
                    for col, val in zip(corr_cols, row_vals, strict=True):
                        if val is not None:
                            val_float = float(val)
                            if not math.isnan(val_float) and abs(val_float) > 0.85:
                                leakage_warnings.append(
                                    f"HIGH RISK: Column '{col}' has high correlation ({val_float:.4f}) with target '{target_col}'"
                                )

        return CleaningPlan(decisions, baseline_stats, leakage_warnings)

    def transform(self, df: Any, plan: CleaningPlan) -> Any:
        start_time = time.perf_counter()
        initial_shape = _dataframe_shape(df)
        mutations: dict[str, list[str]] = {}

        # Check for distribution drift
        drift_alerts: list[str] = []
        if plan.baseline_stats:
            current_stats = _compute_dataframe_stats(df)
            for col, base_col_stats in plan.baseline_stats.items():
                if col not in current_stats:
                    drift_alerts.append(f"Column '{col}' is missing in the incoming data")
                    continue

                curr_col_stats = current_stats[col]

                # Check null_ratio shift
                base_null = base_col_stats.get("null_ratio", 0.0)
                curr_null = curr_col_stats.get("null_ratio", 0.0)
                if abs(curr_null - base_null) > 0.10:
                    drift_alerts.append(
                        f"Column '{col}': null_ratio shifted by {abs(curr_null - base_null):.1%} "
                        f"(baseline: {base_null:.1%}, current: {curr_null:.1%})"
                    )

                # Check mean shift for numeric columns
                if "mean" in base_col_stats and "mean" in curr_col_stats:
                    base_mean = base_col_stats["mean"]
                    curr_mean = curr_col_stats["mean"]
                    if base_mean != 0.0:
                        mean_shift = abs(curr_mean - base_mean) / abs(base_mean)
                    else:
                        mean_shift = abs(curr_mean)

                    if mean_shift > 0.15:
                        drift_alerts.append(
                            f"Column '{col}': mean shifted by {mean_shift:.1%} "
                            f"(baseline: {base_mean:.4f}, current: {curr_mean:.4f})"
                        )

                # Check for new categories in categorical columns
                if "unique_count" in base_col_stats:
                    base_cats = {k.split("cat:", 1)[1] for k in base_col_stats if k.startswith("cat:")}
                    curr_cats = {k.split("cat:", 1)[1] for k in curr_col_stats if k.startswith("cat:")}
                    new_cats = curr_cats - base_cats
                    if new_cats:
                        drift_alerts.append(
                            f"Column '{col}': new categories detected: {sorted(list(new_cats))}"
                        )

        approved = [d for d in plan.decisions if d.approved]
        leakage_warnings = getattr(plan, "leakage_warnings", [])
        if not approved:
            self.last_report = AuditReport(
                initial_shape=initial_shape,
                final_shape=initial_shape,
                mutations=mutations,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                drift_alerts=drift_alerts,
                leakage_warnings=leakage_warnings,
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
            drift_alerts=drift_alerts,
            leakage_warnings=leakage_warnings,
        )
        return current_df

    def fit_transform(self, df: Any, target_col: str | None = None) -> Any:
        plan = self.fit(df, target_col=target_col)
        for decision in plan.decisions:
            decision.approved = True
        return self.transform(df, plan)
