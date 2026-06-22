import math
import warnings
from typing import Any
import numpy as np
import narwhals as nw
from ..base import BaseRule
from ..types import Decision

class KNNImputationRule(BaseRule):
    def __init__(self, k: int = 5, max_rows: int = 10000) -> None:
        self.k = k
        self.max_rows = max_rows

    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        ndf = nw.from_native(df)
        nrows = int(ndf.shape[0])
        if nrows == 0:
            return []

        k = params.get("k", self.k)
        max_rows = params.get("max_rows", self.max_rows)
        decisions: list[Decision] = []

        # Find numeric columns
        numeric_cols = [c for c in ndf.columns if ndf[c].dtype.is_numeric()]
        if not numeric_cols:
            return []

        # Calculate null counts
        null_count_row = ndf.null_count().row(0)
        null_counts = dict(zip(ndf.columns, null_count_row, strict=True))
        cols_with_nulls = [c for c in numeric_cols if null_counts[c] > 0]

        if not cols_with_nulls:
            return []

        # Circuit breaker: check if row count exceeds limit
        if nrows > max_rows:
            warnings.warn(
                f"KNNImputationRule circuit breaker triggered: row count {nrows} "
                f"exceeded max_rows limit of {max_rows}. Falling back to median imputation.",
                RuntimeWarning,
                stacklevel=2
            )
            # Fallback to median imputation
            for col in cols_with_nulls:
                col_median = float(ndf[col].median())
                decisions.append(
                    Decision(
                        rule_name=self.name,
                        column=col,
                        action="median_fallback",
                        parameters={
                            "strategy": "median_fallback",
                            "median_value": col_median,
                            "explanation": f"Median imputed (KNNImputationRule fallback: row count {nrows} exceeded limit {max_rows})"
                        },
                        signal_strength=null_counts[col] / nrows,
                        rationale=(
                            f"Column '{col}' fell back to median imputation because dataset row count "
                            f"({nrows}) exceeds KNNImputationRule max_rows limit ({max_rows})."
                        ),
                    )
                )
            return decisions

        # If complete numeric columns are empty, we cannot calculate distance, so we must fallback to median
        complete_numeric_cols = [c for c in numeric_cols if null_counts[c] == 0]
        if not complete_numeric_cols:
            for col in cols_with_nulls:
                col_median = float(ndf[col].median())
                decisions.append(
                    Decision(
                        rule_name=self.name,
                        column=col,
                        action="median_fallback",
                        parameters={
                            "strategy": "median_fallback",
                            "median_value": col_median,
                            "explanation": "Median imputed (KNNImputationRule fallback: no complete numeric columns available for distance calculation)"
                        },
                        signal_strength=null_counts[col] / nrows,
                        rationale=(
                            f"Column '{col}' fell back to median imputation because there are no complete "
                            f"numeric columns available for distance calculation."
                        ),
                    )
                )
            return decisions

        # Load complete numeric columns as a 2D numpy array
        X_complete = np.array([ndf[c].to_list() for c in complete_numeric_cols]).T

        for col in cols_with_nulls:
            y_vals = ndf[col].to_list()
            missing_indices = [
                i for i, val in enumerate(y_vals)
                if val is None or (isinstance(val, float) and math.isnan(val))
            ]
            non_missing_indices = [
                i for i, val in enumerate(y_vals)
                if val is not None and not (isinstance(val, float) and math.isnan(val))
            ]

            if not non_missing_indices:
                # All values are null, fallback to median (which is nan/None, so maybe default to 0.0 or pass)
                decisions.append(
                    Decision(
                        rule_name=self.name,
                        column=col,
                        action="median_fallback",
                        parameters={
                            "strategy": "median_fallback",
                            "median_value": 0.0,
                            "explanation": "Median imputed to 0.0 (KNNImputationRule fallback: all values in column are missing)"
                        },
                        signal_strength=1.0,
                        rationale=f"Column '{col}' has 100% missing values; defaulted to 0.0.",
                    )
                )
                continue

            imputed_values_map = {}
            actual_k = min(k, len(non_missing_indices))

            for missing_idx in missing_indices:
                x_miss = X_complete[missing_idx]
                X_non_miss = X_complete[non_missing_indices]
                dists = np.sqrt(np.sum((X_non_miss - x_miss) ** 2, axis=1))
                nearest_local_indices = np.argsort(dists)[:actual_k]
                nearest_global_indices = [non_missing_indices[idx] for idx in nearest_local_indices]
                
                # Compute average of target column on nearest neighbors
                neighbor_vals = [y_vals[idx] for idx in nearest_global_indices]
                imputed_val = float(np.mean(neighbor_vals))
                imputed_values_map[str(missing_idx)] = imputed_val  # Store index as string to prevent JSON serialization issues if saved

            decisions.append(
                Decision(
                    rule_name=self.name,
                    column=col,
                    action="knn_impute",
                    parameters={
                        "strategy": "knn",
                        "k": actual_k,
                        "imputed_values": imputed_values_map,
                        "explanation": f"kNN imputed based on {actual_k} nearest neighbors"
                    },
                    signal_strength=len(missing_indices) / nrows,
                    rationale=f"Column '{col}' imputed using {actual_k}-nearest neighbors.",
                )
            )

        return decisions

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        import sys
        ndf = nw.from_native(df)
        
        for d in decisions:
            col = d.column
            action = d.action
            
            if action == "knn_impute":
                imputed_values = d.parameters.get("imputed_values", {})
                if not imputed_values:
                    continue
                
                col_vals = ndf[col].to_list()
                for idx_str, val in imputed_values.items():
                    idx = int(idx_str)
                    col_vals[idx] = val
                
                native_df = ndf.to_native()
                pkg_name = type(native_df).__module__.split('.')[0]
                backend = sys.modules[pkg_name]
                
                new_s = nw.new_series(col, col_vals, backend=backend)
                ndf = ndf.with_columns(new_s)
                
            elif action == "median_fallback":
                median_val = d.parameters["median_value"]
                ndf = ndf.with_columns(nw.col(col).fill_null(median_val))
                
        return ndf.to_native()

    def explain(self, decisions: list[Decision]) -> str:
        if not decisions:
            return "No kNN imputations performed."
        lines = []
        for d in decisions:
            explanation = d.parameters.get("explanation", "")
            lines.append(f"Imputed column '{d.column}': {explanation}")
        return "\n".join(lines)
