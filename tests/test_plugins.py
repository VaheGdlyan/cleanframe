from typing import Any

import narwhals as nw
import polars as pl

from cleanframe.base import BaseRule
from cleanframe.pipeline import DataCleaner
from cleanframe.types import Decision

_STRING_DTYPES = {"str", "string", "object", "utf8"}


def _is_string_column(ndf: nw.DataFrame, col: str) -> bool:
    return str(ndf[col].dtype).lower() in _STRING_DTYPES


class CustomMockRule(BaseRule):
    """Example third-party rule that uppercases string columns."""

    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        ndf = nw.from_native(df)
        return [
            Decision(
                rule_name=self.name,
                column=col,
                action="to_uppercase",
                parameters={},
                signal_strength=1.0,
                rationale=f"Uppercase string column '{col}'",
                approved=True,
            )
            for col in ndf.columns
            if _is_string_column(ndf, col)
        ]

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        ndf = nw.from_native(df)
        for decision in decisions:
            if decision.action != "to_uppercase":
                continue
            col = decision.column
            ndf = ndf.with_columns(ndf[col].str.to_uppercase().alias(col))
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        cols = [d.column for d in decisions if d.approved]
        return f"Converted columns to uppercase: {', '.join(cols)}"


def test_custom_plugin_execution_and_telemetry():
    """Custom rules registered via register_rule() should run and appear in telemetry."""
    df = pl.DataFrame(
        {
            "name": ["alice", "Bob", "CHARLIE"],
            "score": [1, 2, 3],
        }
    )

    cleaner = DataCleaner()
    cleaner.register_rule(CustomMockRule())

    plan = cleaner.fit(df)
    clean_df = cleaner.transform(df, plan)

    assert clean_df["name"].to_list() == ["ALICE", "BOB", "CHARLIE"]

    report = cleaner.last_report
    assert report is not None
    assert "CustomMockRule" in report.mutations
    assert any(
        "Converted columns to uppercase" in entry
        for entry in report.mutations["CustomMockRule"]
    )
