from typing import Any
from ..base import BaseRule
from cleanframe.types import Decision
import narwhals as nw

class CardinalityChecker(BaseRule):
    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        ndf = nw.from_native(df)
        row_count = len(ndf)
        unique_counts = {col: ndf[col].n_unique() for col in ndf.columns}
        dtypes = dict(zip(ndf.columns, map(str, ndf.dtypes)))
        decisions: list[Decision] = []

        for col in ndf.columns:
            n_unique = unique_counts[col]
            dtype = dtypes.get(col, "")
            # Case 1: Constant/Single-Value column
            if n_unique == 1:
                decisions.append(
                    Decision(
                        rule_name="CardinalityChecker",
                        column=col,
                        action="drop_column",
                        parameters={"unique_count": n_unique, "reason": "Constant column"},
                        signal_strength=1.0,
                        rationale=(
                            f"Column '{col}' has only one unique value; zero information content."
                        ),
                    )
                )
            # Case 2: High-cardinality string/object columns
            elif (
                dtype in {"str", "string", "object"}
                and row_count > 0
                and n_unique / row_count > 0.99
            ):
                decisions.append(
                    Decision(
                        rule_name="CardinalityChecker",
                        column=col,
                        action="flag_id",
                        parameters={"unique_ratio": n_unique / row_count, "reason": "High-cardinality candidate ID"},
                        signal_strength=0.9,
                        rationale=(
                            f"Column '{col}' is likely an identifier: unique ratio {n_unique}/{row_count} ≈ {n_unique / row_count:.2%}."
                        ),
                    )
                )
        return decisions

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        columns_to_drop = [
            d.column
            for d in decisions
            if d.action == "drop_column"
        ]
        if not columns_to_drop:
            return df
        ndf = nw.from_native(df)
        ndf = ndf.drop(*columns_to_drop)
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        dropped = [d.column for d in decisions if d.action == "drop_column"]
        flagged = [d.column for d in decisions if d.action == "flag_id"]

        lines: list[str] = []
        if dropped:
            lines.append("The following columns were dropped for having only a single unique value:")
            for col in dropped:
                lines.append(f"  • '{col}' (constant)")
        if flagged:
            lines.append("The following columns were flagged as likely identifier columns due to their high cardinality (>99% unique):")
            for col in flagged:
                lines.append(f"  • '{col}' (likely ID)")
        if not lines:
            return "No constant or high-cardinality columns detected."
        return "\n".join(lines)