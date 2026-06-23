from pathlib import Path
from typing import Any

import duckdb

from .base import BackendProtocol


def _map_to_duckdb_type(target_type: str) -> str:
    target_type_upper = target_type.upper()
    if "DATETIME" in target_type_upper or "TIMESTAMP" in target_type_upper:
        return "TIMESTAMP"
    if "DATE" in target_type_upper:
        return "DATE"
    if "INT" in target_type_upper or "BIGINT" in target_type_upper:
        return "BIGINT"
    if "FLOAT" in target_type_upper or "DOUBLE" in target_type_upper or "REAL" in target_type_upper:
        return "DOUBLE"
    if (
        "STR" in target_type_upper
        or "VARCHAR" in target_type_upper
        or "TEXT" in target_type_upper
        or "UTF8" in target_type_upper
    ):
        return "VARCHAR"
    if "BOOL" in target_type_upper:
        return "BOOLEAN"
    return "VARCHAR"


class DuckDBBackend(BackendProtocol):
    """
    DuckDB-based out-of-core engine allowing out-of-core operations on Parquet/CSV files.
    """

    def __init__(self, memory_limit: str | None = None) -> None:
        self.memory_limit = memory_limit
        self._conn: duckdb.DuckDBPyConnection | None = None

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect()
            if self.memory_limit:
                self._conn.execute(f"PRAGMA memory_limit='{self.memory_limit}'")
        return self._conn

    def read_schema(self, path: str | Path) -> dict[str, str]:
        conn = self.get_connection()
        # Normalise path string
        path_str = str(Path(path).as_posix())
        res = conn.execute(f"DESCRIBE SELECT * FROM '{path_str}'").fetchall()
        return {row[0]: row[1] for row in res}

    def compute_statistics(self, path: str | Path) -> dict[str, dict[str, float]]:
        conn = self.get_connection()
        schema = self.read_schema(path)
        path_str = str(Path(path).as_posix())

        select_exprs = ["COUNT(*) as __total_rows"]
        for col, col_type in schema.items():
            escaped_col = f'"{col}"'
            select_exprs.append(
                f"SUM(CASE WHEN {escaped_col} IS NULL THEN 1 ELSE 0 END) as \"{col}__null_count\""
            )

            is_num = any(
                t in col_type.upper()
                for t in ["INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL"]
            )
            if is_num:
                select_exprs.append(f"AVG({escaped_col}) as \"{col}__mean\"")
                select_exprs.append(f"STDDEV_SAMP({escaped_col}) as \"{col}__std_dev\"")
            else:
                select_exprs.append(f"COUNT(DISTINCT {escaped_col}) as \"{col}__unique_count\"")

        query = f"SELECT {', '.join(select_exprs)} FROM '{path_str}'"
        res = conn.execute(query).fetchone()
        if not res:
            return {}

        col_names = [desc[0] for desc in conn.execute(query).description]
        results_dict = dict(zip(col_names, res, strict=True))

        total_rows = results_dict.get("__total_rows", 0)
        if total_rows == 0:
            return {}

        structured: dict[str, dict[str, float]] = {}
        for col in schema:
            structured[col] = {}
            null_cnt = results_dict.get(f"{col}__null_count", 0)
            structured[col]["null_ratio"] = float(null_cnt) / total_rows

            if f"{col}__mean" in results_dict:
                m = results_dict[f"{col}__mean"]
                structured[col]["mean"] = float(m) if m is not None else 0.0

                s = results_dict[f"{col}__std_dev"]
                structured[col]["std_dev"] = float(s) if s is not None else 0.0

            if f"{col}__unique_count" in results_dict:
                uc = results_dict[f"{col}__unique_count"]
                structured[col]["unique_count"] = float(uc) if uc is not None else 0.0

                escaped_col = f'"{col}"'
                cat_query = f"SELECT DISTINCT {escaped_col} FROM '{path_str}' WHERE {escaped_col} IS NOT NULL LIMIT 100"
                cat_res = conn.execute(cat_query).fetchall()
                for row in cat_res:
                    val = row[0]
                    structured[col][f"cat:{val}"] = 1.0

        return structured

    def sample_to_dataframe(self, path: str | Path, k: int = 10000) -> Any:
        conn = self.get_connection()
        path_str = str(Path(path).as_posix())
        res_count = conn.execute(f"SELECT COUNT(*) FROM '{path_str}'").fetchone()
        total_rows = res_count[0] if res_count is not None else 0

        if total_rows <= k:
            return conn.execute(f"SELECT * FROM '{path_str}'").df()
        else:
            return conn.execute(
                f"SELECT * FROM '{path_str}' USING SAMPLE {k} ROWS (reservoir)"
            ).df()

    def execute_transform(
        self,
        input_path: str | Path,
        output_path: str | Path,
        plan: Any,
    ) -> None:
        conn = self.get_connection()
        schema = self.read_schema(input_path)
        input_path_str = str(Path(input_path).as_posix())
        output_path_str = str(Path(output_path).as_posix())

        dup_decision = None
        for d in plan.decisions:
            if d.approved and d.action == "drop_duplicates":
                dup_decision = d
                break

        needs_row_num = False
        for d in plan.decisions:
            if d.approved and d.action == "knn_impute":
                needs_row_num = True
                break

        from_clause = f"'{input_path_str}'"
        ctes = []

        if needs_row_num:
            ctes.append(
                f"__src_with_rownum AS (SELECT *, row_number() over () - 1 as __row_num FROM '{input_path_str}')"
            )
            from_clause = "__src_with_rownum"

        if dup_decision:
            subset = dup_decision.parameters.get("subset", None)
            if subset:
                escaped_subset = [f'"{c}"' for c in subset]
                ctes.append(
                    f"__src_dedup AS (\n"
                    f"  SELECT *, row_number() over (PARTITION BY {', '.join(escaped_subset)}) as __dup_row_num\n"
                    f"  FROM {from_clause}\n"
                    f")"
                )
                from_clause = "__src_dedup"

        col_exprs = {col: f'"{col}"' for col in schema}

        decisions_by_col: dict[str, list[Any]] = {}
        for d in plan.decisions:
            if not d.approved:
                continue
            col = d.column
            if col not in decisions_by_col:
                decisions_by_col[col] = []
            decisions_by_col[col].append(d)

        for col, col_decisions in decisions_by_col.items():
            if col == "all":
                continue
            expr = f'"{col}"'
            for d in col_decisions:
                if d.action == "cast":
                    target_type = d.parameters.get("target_type")
                    if target_type:
                        db_type = _map_to_duckdb_type(str(target_type))
                        expr = f"CAST({expr} AS {db_type})"
                elif d.action in ("median", "mean", "mode"):
                    agg_func = (
                        "MEDIAN"
                        if d.action == "median"
                        else ("AVG" if d.action == "mean" else "MODE")
                    )
                    val_res = conn.execute(
                        f"SELECT {agg_func}(\"{col}\") FROM '{input_path_str}'"
                    ).fetchone()
                    val = val_res[0] if val_res else None
                    if val is not None:
                        if isinstance(val, str):
                            escaped_val = val.replace("'", "''")
                            val_str = f"'{escaped_val}'"
                        else:
                            val_str = str(val)
                        expr = f"COALESCE({expr}, {val_str})"
                elif d.action == "median_fallback":
                    val = d.parameters.get("median_value")
                    if val is not None:
                        if isinstance(val, str):
                            escaped_val = val.replace("'", "''")
                            val_str = f"'{escaped_val}'"
                        else:
                            val_str = str(val)
                        expr = f"COALESCE({expr}, {val_str})"
                elif d.action == "knn_impute":
                    imputed_values = d.parameters.get("imputed_values", {})
                    if imputed_values:
                        case_expr = "CASE __row_num"
                        for idx_str, val in imputed_values.items():
                            idx = int(idx_str)
                            case_expr += f" WHEN {idx} THEN {val}"
                        case_expr += f" ELSE {expr} END"
                        expr = case_expr
                elif d.action == "clip":
                    lower = d.parameters.get("lower_bound")
                    upper = d.parameters.get("upper_bound")
                    if lower is not None and upper is not None:
                        expr = f"CASE WHEN {expr} < {lower} THEN {lower} WHEN {expr} > {upper} THEN {upper} ELSE {expr} END"
                elif d.action == "replace_values":
                    mapping = d.parameters.get("mapping", {})
                    if mapping:
                        case_expr = f"CASE {expr}"
                        for k, v in mapping.items():
                            k_escaped = str(k).replace("'", "''")
                            v_escaped = str(v).replace("'", "''")
                            case_expr += f" WHEN '{k_escaped}' THEN '{v_escaped}'"
                        case_expr += f" ELSE {expr} END"
                        expr = case_expr
            col_exprs[col] = expr

        select_parts = [f"{expr} AS \"{col}\"" for col, expr in col_exprs.items()]
        select_str = ", ".join(select_parts)

        where_clause = ""
        if dup_decision and dup_decision.parameters.get("subset"):
            where_clause = " WHERE __dup_row_num = 1"

        distinct_str = ""
        if dup_decision and not dup_decision.parameters.get("subset"):
            distinct_str = "DISTINCT "

        cte_str = f"WITH {', '.join(ctes)}\n" if ctes else ""
        query = f"{cte_str}SELECT {distinct_str}{select_str} FROM {from_clause}{where_clause}"

        # Write out
        if output_path_str.lower().endswith(".parquet"):
            copy_query = f"COPY ({query}) TO '{output_path_str}' (FORMAT PARQUET)"
        else:
            copy_query = f"COPY ({query}) TO '{output_path_str}' (FORMAT CSV, HEADER)"

        conn.execute(copy_query)
