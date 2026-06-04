from typing import Any

from ..base import BaseRule
from ..types import Decision
import narwhals as nw


class NullHandler(BaseRule):
    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        ndf = nw.from_native(df)
        cols = list(ndf.columns)
        nrows = int(ndf.shape[0])
        if nrows == 0:
            return []

        num_strategy = params.get("numeric_strategy", "median")
        cat_strategy = params.get("categorical_strategy", "mode")
        result: list[Decision] = []

        null_count_row = ndf.null_count().row(0)
        null_counts = dict(zip(cols, null_count_row, strict=True))
        for col in cols:
            null_count = int(null_counts[col])
            if null_count > 0:
                if ndf[col].dtype.is_numeric():
                    strategy = num_strategy
                else:
                    strategy = cat_strategy
                signal_strength = null_count / nrows
                decision = Decision(
                    rule_name="NullHandler",
                    column=col,
                    action=strategy,
                    parameters={"strategy": strategy},
                    signal_strength=signal_strength,
                    rationale=(
                        f"Column '{col}' has {null_count} null value(s) "
                        f"({signal_strength:.1%} of {nrows} rows); fill with {strategy}."
                    ),
                )
                result.append(decision)
        return result

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        ndf = nw.from_native(df)
        fill_values = {}

        for decision in decisions:
            col = decision.column
            action = decision.action
            if action == "median":
                value = ndf[col].median()
            elif action == "mean":
                value = ndf[col].mean()
            elif action == "mode":
                modes = ndf[col].mode()
                value = modes[0] if len(modes) > 0 else None
            else:
                continue
            fill_values[col] = value

        if fill_values:
            ndf = ndf.with_columns(
                **{col: ndf[col].fill_null(value) for col, value in fill_values.items()}
            )
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        if not decisions:
            return "No missing values detected."

        lines = [
            "Column | Missing Count | Strategy",
            "-------|--------------|---------",
        ]
        for decision in decisions:
            col = decision.column
            missing_count = int(
                decision.signal_strength * 100
            )  # signal_strength * 100 gives percentage
            strategy = decision.parameters.get("strategy", decision.action)
            line = f"{col} | {missing_count}% | {strategy}"
            lines.append(line)
        return "\n".join(lines)
