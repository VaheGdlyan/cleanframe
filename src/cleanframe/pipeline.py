import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import narwhals as nw

from .base import RuleProtocol
from .plan import CleaningPlan
from .registry import discover_plugins
from .rules import (
    CardinalityChecker,
    DuplicateHandler,
    NullHandler,
    OutlierHandler,
    SchemaCaster,
)
from .telemetry import AuditReport, TelemetryEvent, TelemetrySink
from .types import Decision

FrameT = TypeVar("FrameT")

_BUILTIN_RULES = frozenset({
    "SchemaCaster",
    "DuplicateHandler",
    "NullHandler",
    "OutlierHandler",
    "CardinalityChecker",
    "KNNImputationRule",
})

_DEFAULT_RULES: list[RuleProtocol] = [
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
    if action == "knn_impute":
        explanation = decision.parameters.get("explanation", "kNN imputed")
        return f"Imputed nulls in '{column}' ({explanation})"
    if action == "median_fallback":
        explanation = decision.parameters.get("explanation", "median fallback")
        return f"Imputed nulls in '{column}' ({explanation})"
    return f"{action.replace('_', ' ').capitalize()} on '{column}'"


def _decisions_for_rule(
    decisions: list[Decision],
    rule_name: str,
    rule_alt_name: str | None = None,
) -> list[Decision]:
    names = {rule_name}
    if rule_alt_name:
        names.add(rule_alt_name)
    return [
        d
        for d in decisions
        if getattr(d, "rule_name", "") in names
        or getattr(d, "rule", "") in names
    ]


def _mutation_entries(rule: RuleProtocol, decisions: list[Decision]) -> list[str]:
    if type(rule).__name__ in _BUILTIN_RULES:
        return [_decision_summary(d) for d in decisions]
    if hasattr(rule, "explain") and callable(getattr(rule, "explain")):
        return [rule.explain(decisions)]  # type: ignore[attr-defined]
    return [_decision_summary(d) for d in decisions]


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

    def __init__(
        self,
        rules: list[RuleProtocol] | None = None,
        impute_strategy: str = "median",
        sinks: list[TelemetrySink] | None = None,
    ) -> None:
        self.impute_strategy = impute_strategy
        self.sinks: list[TelemetrySink] = list(sinks) if sinks is not None else []
        self.rules: list[RuleProtocol] = []
        self.config_params: dict[str, dict[str, Any]] = {}
        if rules is not None:
            self.rules = list(rules)
        else:
            if impute_strategy == "knn":
                from .rules.knn_imputer import KNNImputationRule
                self.rules = [
                    SchemaCaster(),
                    DuplicateHandler(),
                    KNNImputationRule(),
                    NullHandler(),
                    OutlierHandler(),
                    CardinalityChecker(),
                ]
            else:
                self.rules = list(_DEFAULT_RULES)
            self.rules.extend(discover_plugins())
        self.last_report: AuditReport | None = None

    def _emit(
        self,
        event_type: str,
        rule_name: str | None,
        column: str | None,
        payload: dict[str, Any],
        run_id: str,
    ) -> TelemetryEvent:
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            rule_name=rule_name,
            column=column,
            payload=payload,
        )
        for sink in self.sinks:
            sink.emit(event)
        return event

    def register_rule(self, rule: RuleProtocol) -> None:
        """Append a custom rule to the active execution registry."""
        if not isinstance(rule, RuleProtocol):
            raise TypeError("Registered rule must implement RuleProtocol")
        self.rules.append(rule)

    def fit(
        self,
        df: FrameT,
        params_map: dict[str, dict[str, Any]] | None = None,
        target_col: str | None = None,
    ) -> CleaningPlan[FrameT]:
        """
        Run each rule's detect method and aggregate the resulting decisions.

        Args:
            df: Input dataset (pandas or polars DataFrame).
            params_map: Optional mapping from rule class name to parameter dict.
            target_col: Optional target variable to check for target leakage.

        Returns:
            CleaningPlan containing all collected decisions.
        """
        run_id = str(uuid.uuid4())
        # Override NullHandler strategy if using knn imputer
        if self.impute_strategy == "knn":
            if params_map is None:
                params_map = {}
            if "NullHandler" not in params_map:
                params_map["NullHandler"] = {}
            if "numeric_strategy" not in params_map["NullHandler"]:
                params_map["NullHandler"]["numeric_strategy"] = "none"

        ndf = nw.from_native(df)  # type: ignore[call-overload]
        if target_col is not None:
            assert target_col in ndf.columns, f"Target column '{target_col}' not found in dataset"

        baseline_stats = _compute_dataframe_stats(df)
        decisions: list[Decision] = []
        for rule in self.rules:
            rule_name = type(rule).__name__
            params = dict(self.config_params.get(rule_name, {}))
            if params_map and rule_name in params_map:
                params.update(params_map[rule_name])
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
                    warn_msg = f"HIGH RISK: Column '{col}' semantically leaks target '{target_col}'"
                    leakage_warnings.append(warn_msg)
                    self._emit(
                        event_type="target_leakage",
                        rule_name="TargetLeakageDetector",
                        column=col,
                        payload={"warning": warn_msg},
                        run_id=run_id,
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
                                warn_msg = f"HIGH RISK: Column '{col}' has high correlation ({val_float:.4f}) with target '{target_col}'"
                                leakage_warnings.append(warn_msg)
                                self._emit(
                                    event_type="target_leakage",
                                    rule_name="TargetLeakageDetector",
                                    column=col,
                                    payload={"warning": warn_msg, "correlation": val_float},
                                    run_id=run_id,
                                )

        return CleaningPlan(decisions, baseline_stats, leakage_warnings)

    def transform(self, df: FrameT, plan: CleaningPlan[FrameT]) -> FrameT:
        start_time = time.perf_counter()
        initial_shape = _dataframe_shape(df)
        mutations: dict[str, list[str]] = {}
        run_id = str(uuid.uuid4())
        events: list[TelemetryEvent] = []

        # Emit target leakage warnings from plan
        leakage_warnings = getattr(plan, "leakage_warnings", [])
        for warning in leakage_warnings:
            col = None
            if "Column '" in warning:
                col = warning.split("Column '", 1)[1].split("'", 1)[0]
            event = self._emit(
                event_type="target_leakage",
                rule_name="TargetLeakageDetector",
                column=col,
                payload={"warning": warning},
                run_id=run_id,
            )
            events.append(event)

        # Emit constraint violations (consistency warnings)
        for d in plan.decisions:
            if d.rule_name == "CrossColumnConsistencyRule" and d.action == "flag_violation":
                violation_count = d.parameters.get("violation_count", 0)
                constraint_name = d.parameters.get("constraint_name", "")
                msg = f"CONSTRAINT VIOLATION: {violation_count} rows failed '{constraint_name}' rule"
                event = self._emit(
                    event_type="constraint_violation",
                    rule_name="CrossColumnConsistencyRule",
                    column=d.column,
                    payload={
                        "warning": msg,
                        "constraint_name": constraint_name,
                        "violation_count": violation_count,
                    },
                    run_id=run_id,
                )
                events.append(event)

        # Check for distribution drift
        if plan.baseline_stats:
            current_stats = _compute_dataframe_stats(df)
            for col, base_col_stats in plan.baseline_stats.items():
                if col not in current_stats:
                    msg = f"Column '{col}' is missing in the incoming data"
                    event = self._emit(
                        event_type="drift_alert",
                        rule_name="DistributionDriftDetector",
                        column=col,
                        payload={"alert": msg},
                        run_id=run_id,
                    )
                    events.append(event)
                    continue

                curr_col_stats = current_stats[col]

                # Check null_ratio shift
                base_null = base_col_stats.get("null_ratio", 0.0)
                curr_null = curr_col_stats.get("null_ratio", 0.0)
                if abs(curr_null - base_null) > 0.10:
                    msg = (
                        f"Column '{col}': null_ratio shifted by {abs(curr_null - base_null):.1%} "
                        f"(baseline: {base_null:.1%}, current: {curr_null:.1%})"
                    )
                    event = self._emit(
                        event_type="drift_alert",
                        rule_name="DistributionDriftDetector",
                        column=col,
                        payload={"alert": msg},
                        run_id=run_id,
                    )
                    events.append(event)

                # Check mean shift for numeric columns
                if "mean" in base_col_stats and "mean" in curr_col_stats:
                    base_mean = base_col_stats["mean"]
                    curr_mean = curr_col_stats["mean"]
                    if base_mean != 0.0:
                        mean_shift = abs(curr_mean - base_mean) / abs(base_mean)
                    else:
                        mean_shift = abs(curr_mean)

                    if mean_shift > 0.15:
                        msg = (
                            f"Column '{col}': mean shifted by {mean_shift:.1%} "
                            f"(baseline: {base_mean:.4f}, current: {curr_mean:.4f})"
                        )
                        event = self._emit(
                            event_type="drift_alert",
                            rule_name="DistributionDriftDetector",
                            column=col,
                            payload={"alert": msg},
                            run_id=run_id,
                        )
                        events.append(event)

                # Check for new categories in categorical columns
                if "unique_count" in base_col_stats:
                    base_cats = {k.split("cat:", 1)[1] for k in base_col_stats if k.startswith("cat:")}
                    curr_cats = {k.split("cat:", 1)[1] for k in curr_col_stats if k.startswith("cat:")}
                    new_cats = curr_cats - base_cats
                    if new_cats:
                        msg = f"Column '{col}': new categories detected: {sorted(list(new_cats))}"
                        event = self._emit(
                            event_type="drift_alert",
                            rule_name="DistributionDriftDetector",
                            column=col,
                            payload={"alert": msg},
                            run_id=run_id,
                        )
                        events.append(event)

        approved = [d for d in plan.decisions if d.approved]
        if not approved:
            self.last_report = AuditReport(
                initial_shape=initial_shape,
                final_shape=initial_shape,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                events=events,
            )
            return df

        current_df = df
        for rule in self.rules:
            rule_name = type(rule).__name__
            rule_decisions = _decisions_for_rule(approved, rule_name, getattr(rule, "name", None))
            if not rule_decisions:
                continue

            summaries = _mutation_entries(rule, rule_decisions)
            mutations[rule.name] = summaries
            current_df = rule.transform(current_df, rule_decisions)

            for summary in summaries:
                col = rule_decisions[0].column if rule_decisions else None
                event = self._emit(
                    event_type="rule_mutation",
                    rule_name=rule.name,
                    column=col,
                    payload={"summary": summary},
                    run_id=run_id,
                )
                events.append(event)

        self.last_report = AuditReport(
            initial_shape=initial_shape,
            final_shape=_dataframe_shape(current_df),
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            events=events,
        )
        return current_df

    def fit_transform(self, df: FrameT, target_col: str | None = None) -> FrameT:
        plan = self.fit(df, target_col=target_col)
        for decision in plan.decisions:
            decision.approved = True
        return self.transform(df, plan)

    @classmethod
    def from_config(cls, filepath: str | Path) -> "DataCleaner":
        """
        Load and configure DataCleaner from a TOML file.

        Args:
            filepath: Path to the TOML configuration file.

        Returns:
            A configured DataCleaner instance.
        """
        import tomllib
        from pathlib import Path
        import inspect

        path = Path(filepath)
        with path.open("rb") as f:
            config = tomllib.load(f)

        cleaner_config = config.get("cleaner", {})
        impute_strategy = cleaner_config.get("impute_strategy", "median")

        cleaner = cls(impute_strategy=impute_strategy)

        rules_config = config.get("rules", {})
        new_rules = []
        for rule in cleaner.rules:
            rule_name = type(rule).__name__
            if rule_name in rules_config:
                rule_params = rules_config[rule_name]
                rule_cls = type(rule)

                try:
                    sig = inspect.signature(rule_cls.__init__)
                    valid_init_args = {}
                    for k, v in rule_params.items():
                        if k in sig.parameters:
                            valid_init_args[k] = v
                    new_rule = rule_cls(**valid_init_args)
                except Exception:
                    new_rule = rule_cls()

                new_rules.append(new_rule)
                cleaner.config_params[rule_name] = rule_params
            else:
                new_rules.append(rule)

        cleaner.rules = new_rules
        return cleaner
