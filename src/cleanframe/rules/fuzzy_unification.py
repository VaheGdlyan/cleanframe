import narwhals as nw
from typing import Any

from ..base import BaseRule
from ..types import Decision

class FuzzyUnificationRule(BaseRule):
    def __init__(self, threshold: float = 80.0):
        self.threshold = threshold

    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            return []

        ndf = nw.from_native(df)
        decisions: list[Decision] = []
        dtypes = {col: str(ndf[col].dtype) for col in ndf.columns}

        for col in ndf.columns:
            dtype = dtypes.get(col, "")
            if str(dtype).lower() not in {"str", "string", "object", "utf8", "categorical", "category"}:
                continue

            unique_vals = ndf[col].unique().drop_nulls().to_list()
            
            if len(unique_vals) < 2:
                continue

            # Ensure all values are strings for string matching
            unique_vals = [str(x) for x in unique_vals if x is not None]

            clusters = []
            unassigned = set(unique_vals)

            while unassigned:
                target = min(unassigned)
                unassigned.remove(target)
                
                matches = process.extract(
                    target, list(unassigned), scorer=fuzz.token_sort_ratio, score_cutoff=self.threshold
                )
                
                cluster = [target]
                for match_str, score, _ in matches:
                    cluster.append(match_str)
                    unassigned.remove(match_str)
                    
                if len(cluster) > 1:
                    clusters.append(cluster)

            if not clusters:
                continue

            replacement_map = {}
            for cluster in clusters:
                canonical = sorted(cluster)[0]
                for item in cluster:
                    if item != canonical:
                        replacement_map[item] = canonical

            if replacement_map:
                decisions.append(
                    Decision(
                        rule_name=self.name,
                        column=col,
                        action="replace_values",
                        parameters={
                            "mapping": replacement_map,
                            "threshold": self.threshold
                        },
                        signal_strength=1.0,
                        rationale=(
                            f"Unified {len(replacement_map)} similar categorical values into their canonical forms "
                            f"(threshold: {self.threshold})."
                        ),
                    )
                )

        return decisions

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        ndf = nw.from_native(df)
        
        for d in decisions:
            if d.action == "replace_values":
                mapping = d.parameters.get("mapping", {})
                if mapping:
                    ndf = ndf.with_columns(
                        nw.col(d.column).replace_strict(mapping, default=nw.col(d.column))
                    )
        
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        lines = []
        for d in decisions:
            if d.action == "replace_values":
                mapping = d.parameters.get("mapping", {})
                num_replacements = len(mapping)
                lines.append(f"  • '{d.column}': unified {num_replacements} variants.")
        
        if not lines:
            return "No fuzzy unifications performed."
            
        return "Fuzzy unification applied:\n" + "\n".join(lines)
