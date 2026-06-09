from typing import Any

import narwhals as nw

from ..base import BaseRule
from ..inference import SemanticSchemaInferrer
from ..types import Decision

_STRING_DTYPES = {"str", "string", "object", "utf8"}


def _is_string_dtype(dtype: Any) -> bool:
    return str(dtype).lower() in _STRING_DTYPES


def _is_datetime_dtype(dtype: Any) -> bool:
    return dtype in (nw.Datetime, nw.Date) or "datetime" in str(dtype).lower()


class SchemaCaster(BaseRule):
    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        ndf = nw.from_native(df)
        current_dtypes = {col: ndf[col].dtype for col in ndf.columns}
        decisions: list[Decision] = []
        decided_cols: set[str] = set()

        inferrer = SemanticSchemaInferrer(df)
        for col in ndf.columns:
            if inferrer.get_semantic_type(col) != "datetime_hint":
                continue
            dtype = current_dtypes[col]
            if not _is_string_dtype(dtype) or _is_datetime_dtype(dtype):
                continue
            decisions.append(
                Decision(
                    rule_name="SchemaCaster",
                    column=col,
                    action="cast",
                    parameters={
                        "target_type": "Datetime",
                        "method": "to_datetime",
                    },
                    signal_strength=1.0,
                    rationale=(
                        f"Column '{col}' is named like a datetime but stored as "
                        f"{dtype}; coerce to Datetime."
                    ),
                )
            )
            decided_cols.add(col)

        target_schema = params.get("schema", {})
        for col, target_type in target_schema.items():
            if col in decided_cols:
                continue
            dtype = current_dtypes.get(col)
            if dtype is None:
                continue
            current_type = str(dtype)
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
            params = decision.parameters
            target_type = params.get("target_type")
            if target_type is None or col not in ndf.columns:
                continue
            if params.get("method") == "to_datetime":
                ndf = ndf.with_columns(ndf[col].str.to_datetime().alias(col))
                continue
            nw_dtype = getattr(nw, str(target_type), None)
            if nw_dtype is None:
                continue
            ndf = ndf.with_columns(ndf[col].cast(nw_dtype).alias(col))
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        if not decisions:
            return "No columns require casting to a new data type."
        lines = ["The following columns will be cast to new data types:"]
        for decision in decisions:
            col = decision.column
            target_type = decision.parameters.get("target_type", "Unknown")
            lines.append(f"  • '{col}' → {target_type}")
        return "\n".join(lines)
