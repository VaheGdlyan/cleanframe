from typing import Any

from .pipeline import DataCleaner
from .plan import CleaningPlan
from .telemetry import AuditReport

_CF_REPORT_ATTR = "_cf_report"


class CleanFrameAccessor:
    """
    Namespace accessor for CleanFrame on pandas and polars DataFrames.

    Usage:
        df.cf.audit()   # Returns a CleaningPlan with suggested actions
        df.cf.clean()   # Returns a cleaned DataFrame (with _cf_report attached)
        df.cf.report()  # Prints the audit report from the last clean() call
    """

    def __init__(self, df: Any) -> None:
        self._df = df

    def audit(self) -> CleaningPlan:
        return DataCleaner().fit(self._df)

    def clean(self) -> Any:
        cleaner = DataCleaner()
        clean_df = cleaner.fit_transform(self._df)
        setattr(clean_df, _CF_REPORT_ATTR, cleaner.last_report)
        return clean_df

    def report(self) -> None:
        report: AuditReport | None = getattr(self._df, _CF_REPORT_ATTR, None)
        if report is not None:
            report.display()
            return
        print(
            "No audit report found on this DataFrame. "
            "Run `.cf.clean()` first to generate one."
        )


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
