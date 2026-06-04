from typing import Any
from .pipeline import DataCleaner
from .plan import CleaningPlan

class CleanFrameAccessor:
    """
    Namespace accessor for CleanFrame library on supported DataFrames.

    Usage:
        df.cf.audit()   # Returns CleaningPlan with suggested actions
        df.cf.clean()   # Returns cleaned DataFrame
    """

    def __init__(self, df: Any) -> None:
        self._df: Any = df

    def audit(self) -> CleaningPlan:
        cleaner = DataCleaner()
        plan: CleaningPlan = cleaner.fit(self._df)
        return plan

    def clean(self) -> Any:
        from .pipeline import DataCleaner
        return DataCleaner().fit_transform(self._df)

# Registration logic for both pandas and polars

try:
    import pandas as pd  # type: ignore[import-untyped]

    pd.api.extensions.register_dataframe_accessor("cf")(CleanFrameAccessor)
except ImportError:
    pass

try:
    import polars as pl

    pl.api.register_dataframe_namespace("cf")(CleanFrameAccessor)
except ImportError:
    pass