from typing import Any, Dict, Optional

def infer_semantic_type(column_name: str) -> str | None:
    """
    Infer the semantic type of a column based on its name using heuristics.

    Args:
        column_name: The name of the column.

    Returns:
        A string representing the inferred semantic type, or None if unknown.
    """
    name = column_name.lower()

    if "email" in name:
        return "email"
    if any(x in name for x in ["price", "revenue", "amount"]):
        return "currency"
    if name == "id" or name.endswith("_id"):
        return "identifier"
    if (
        "date" in name
        or "timestamp" in name
        or "created_at" in name
    ):
        return "datetime_hint"
    return None

class SemanticSchemaInferrer:
    """
    Infers and stores column semantic types based on column names.
    """

    def __init__(self, df: Any) -> None:
        """
        Scan columns and record semantic types for each column, if any.

        Args:
            df: The input dataframe-like object (must have a `.columns` attribute).
        """
        self.semantic_types: Dict[str, str] = {}
        for col in getattr(df, "columns", []):
            stype = infer_semantic_type(col)
            if stype is not None:
                self.semantic_types[col] = stype

    def get_semantic_type(self, column: str) -> Optional[str]:
        """
        Retrieve the semantic type mapping for a specific column, if available.

        Args:
            column: The column name.

        Returns:
            The semantic type if mapped, else None.
        """
        return self.semantic_types.get(column)