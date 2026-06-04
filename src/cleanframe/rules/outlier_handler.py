from typing import Any
from ..base import BaseRule
from ..types import Decision
import narwhals as nw


class OutlierHandler(BaseRule):
    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        ndf = nw.from_native(df)
        cols = list(ndf.columns)
        nrows = int(ndf.shape[0])
        if nrows == 0:
            return []

        k = params.get("multiplier", 1.5)
        numeric_cols = [col for col in cols if ndf[col].dtype.is_numeric()]

        results: list[Decision] = []

        for col in numeric_cols:
            series = ndf[col]
            # Compute Q1, Q3 using quantile
            Q1 = series.quantile(0.25, interpolation="linear")
            Q3 = series.quantile(0.75, interpolation="linear")
            IQR = Q3 - Q1
            lower = Q1 - k * IQR
            upper = Q3 + k * IQR

            # Count outliers
            is_outlier = (series < lower) | (series > upper)
            outlier_count = int(is_outlier.sum())
            if outlier_count > 0:
                signal_strength = outlier_count / nrows
                result = Decision(
                    rule_name="OutlierHandler",
                    column=col,
                    action="clip",
                    parameters={"lower_bound": lower, "upper_bound": upper},
                    signal_strength=signal_strength,
                    rationale=(
                        f"Column '{col}': {outlier_count} outlier(s) "
                        f"beyond [{lower}, {upper}] using k={k}."
                    ),
                )
                results.append(result)
        return results

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        ndf = nw.from_native(df)
        for decision in decisions:
            col = decision.column
            params = decision.parameters
            lower = params.get("lower_bound")
            upper = params.get("upper_bound")
            # Only perform if bounds are not None
            if lower is not None and upper is not None:
                series = ndf[col]
                # Vectorized clipping using narwhals expressions
                ndf = ndf.with_columns(
                    **{
                        col: nw.when(series < lower)
                        .then(lower)
                        .when(series > upper)
                        .then(upper)
                        .otherwise(series)
                    }
                )
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        if not decisions:
            return "No outliers detected in any numeric columns."
        lines = [
            "Column | Detected Outliers | Clipping Bounds",
            "-------|-------------------|----------------",
        ]
        for decision in decisions:
            col = decision.column
            n_out = int(decision.signal_strength * 100)  # as % of rows
            bounds = decision.parameters
            lower = bounds.get("lower_bound")
            upper = bounds.get("upper_bound")
            bounds_str = f"[{lower}, {upper}]"
            line = f"{col} | {n_out}% | {bounds_str}"
            lines.append(line)
        return "\n".join(lines)
