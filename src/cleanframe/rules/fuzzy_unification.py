import narwhals as nw
from typing import Any

from ..base import BaseRule
from ..types import Decision

class FuzzyUnificationRule(BaseRule):
    def __init__(
        self,
        threshold: float = 80.0,
        exclude_cols: list[str] | None = None,
        pre_lowercase: bool = False,
    ) -> None:
        self.threshold = threshold
        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.pre_lowercase = pre_lowercase

    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            return []

        ndf = nw.from_native(df)
        decisions: list[Decision] = []
        dtypes = {col: str(ndf[col].dtype) for col in ndf.columns}

        for col in ndf.columns:
            if col in self.exclude_cols:
                continue

            dtype = dtypes.get(col, "")
            if str(dtype).lower() not in {"str", "string", "object", "utf8", "categorical", "category"}:
                continue

            # Extract unique values preserving insertion order
            col_list = ndf[col].to_list()
            seen_set = set()
            unique_vals = []
            for x in col_list:
                if x is not None:
                    s_x = str(x)
                    if s_x not in seen_set:
                        seen_set.add(s_x)
                        unique_vals.append(s_x)

            if len(unique_vals) < 2:
                continue

            repr_to_orig: dict[str, list[str]] = {}
            if self.pre_lowercase:
                # Group original values by their lowercase representation
                for val in unique_vals:
                    low = val.lower()
                    if low not in repr_to_orig:
                        repr_to_orig[low] = []
                    repr_to_orig[low].append(val)
            else:
                # Each unique value is its own representation
                repr_to_orig = {val: [val] for val in unique_vals}

            clusters = []
            unassigned = set(repr_to_orig.keys())

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
                    
                if len(cluster) > 1 or len(repr_to_orig[target]) > 1:
                    clusters.append(cluster)

            if not clusters:
                continue

            replacement_map = {}
            for cluster in clusters:
                # Collect all original values belonging to representations in the cluster
                orig_vals = []
                for repr_val in cluster:
                    orig_vals.extend(repr_to_orig[repr_val])

                # Pick the canonical original value (first one that appeared in unique_vals)
                canonical = min(orig_vals, key=lambda x: unique_vals.index(x))
                for item in orig_vals:
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
                            f"(threshold: {self.threshold}, pre_lowercase: {self.pre_lowercase})."
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
