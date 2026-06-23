from typing import assert_type
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from cleanframe.accessor import CleanFrameAccessor
from cleanframe.plan import CleaningPlan


def test_typing_preservation() -> None:
    df_pd = pd.DataFrame({"a": [1, 2]})
    df_pl = pl.DataFrame({"a": [1, 2]})

    # Check typing preservation using mypy's assert_type
    res_pd = CleanFrameAccessor(df_pd).clean()
    res_pl = CleanFrameAccessor(df_pl).clean()

    # Mypy will assert that these types are correct at typecheck time
    assert_type(res_pd, pd.DataFrame)
    assert_type(res_pl, pl.DataFrame)

    plan_pd = CleanFrameAccessor(df_pd).audit()
    plan_pl = CleanFrameAccessor(df_pl).audit()

    assert_type(plan_pd, CleaningPlan[pd.DataFrame])
    assert_type(plan_pl, CleaningPlan[pl.DataFrame])

