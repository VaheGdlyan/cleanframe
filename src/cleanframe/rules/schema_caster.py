from typing import Any
from ..base import BaseRule
from cleanframe.types import Decision
import narwhals as nw

class SchemaCaster(BaseRule):
    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        target_schema = params.get("schema", {})
        # Use narwhals DataFrame for dtype access
        ndf = nw.from_native(df)
        current_dtypes = dict(zip(ndf.columns, map(str, ndf.dtypes)))
        decisions: list[Decision] = []
        for col, target_type in target_schema.items():
            current_type = current_dtypes.get(col)
            if current_type is None:
                continue  # Skip columns missing from DataFrame
            # Normalize for consistent comparison
            if current_type != target_type:
                decisions.append(
                    Decision(
                        rule_name="SchemaCaster",
                        column=col,
                        action="cast",
                        parameters={"target_type": target_type},
                        signal_strength=1.0,
                        rationale=(
                            f"Column '{col}' is {current_type}; cast to {target_type}."
                        ),
                    )
                )
        return decisions

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        if not decisions:
            return df
        ndf = nw.from_native(df)
        for decision in decisions:
            col = decision.column
            target_type = decision.parameters.get("target_type")
            if target_type is not None and col in ndf.columns:
                ndf = ndf.with_columns(
                    ndf[col].cast(target_type).alias(col)
                )
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        if not decisions:
            return "No columns require casting to a new data type."
        lines = [
            "The following columns will be cast to new data types:"
        ]
        for decision in decisions:
            col = decision.column
            target_type = decision.parameters.get("target_type", "Unknown")
            lines.append(f"  • '{col}' → {target_type}")
        return "\n".join(lines)