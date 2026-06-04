from typing import Any
from ..base import BaseRule
from cleanframe.types import Decision
import narwhals as nw

class DuplicateHandler(BaseRule):
    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        subset = params.get("subset", None)
        keep = params.get("keep", "first")
        ndf = nw.from_native(df)
        # Detect duplicated rows according to Narwhals (mimicking pandas.duplicated)
        mask = ndf.duplicated(subset=subset, keep=keep)
        # Narwhals returns mask as Series like pandas/polars
        if hasattr(mask, "sum"):
            num_duplicates = int(mask.sum())
        elif hasattr(mask, "to_numpy"):
            num_duplicates = int(mask.to_numpy().sum())
        else:
            # Fallback
            num_duplicates = int(sum(bool(x) for x in mask))
        if num_duplicates == 0:
            return []
        if subset is None:
            signal_strength = 1.0
            rationale = (
                f"{num_duplicates} full row duplicate(s) detected; will drop duplicate rows keeping the '{keep}' occurrence."
            )
        else:
            signal_strength = 0.8
            rationale = (
                f"{num_duplicates} duplicate(s) detected in subset columns {subset}; will drop duplicates keeping the '{keep}' occurrence."
            )
        return [
            Decision(
                rule_name="DuplicateHandler",
                column="all",
                action="drop_duplicates",
                parameters={"subset": subset, "keep": keep, "num_duplicates": num_duplicates},
                signal_strength=signal_strength,
                rationale=rationale,
            )
        ]

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        # Only process if a deduplication Decision exists
        decision = next(
            (d for d in decisions if d.action == "drop_duplicates" and d.column == "all"), None
        )
        if decision is None:
            return df
        subset = decision.parameters.get("subset", None)
        keep = decision.parameters.get("keep", "first")
        ndf = nw.from_native(df)
        ndf = ndf.unique(subset=subset, keep=keep)
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        decision = next(
            (d for d in decisions if d.action == "drop_duplicates" and d.column == "all"), None
        )
        if decision is None:
            return "No duplicate rows detected; no deduplication applied."
        subset = decision.parameters.get("subset", None)
        keep = decision.parameters.get("keep", "first")
        num_duplicates = decision.parameters.get("num_duplicates", "unknown")
        subset_desc = (
            "entire rows"
            if subset is None
            else f"columns {subset}"
        )
        return (
            f"Dropped {num_duplicates} duplicate row(s) based on {subset_desc}, "
            f"keeping the '{keep}' occurrence of each duplicate."
        )