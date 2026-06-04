import pytest
import pandas as pd
from cleanframe.rules import NullHandler, OutlierHandler
from cleanframe.types import Decision

def test_null_handler(messy_dataframe):
    """
    Test NullHandler: it should detect nulls in both columns and fill them appropriately.
    """
    handler = NullHandler()
    decisions = handler.detect(messy_dataframe, params={})
    detected_cols = {d.column for d in decisions}

    # Assert both "age" and "city" are present in detected columns
    assert "age" in detected_cols
    assert "city" in detected_cols
    assert all(isinstance(d, Decision) for d in decisions)
    assert len(decisions) == 2

    # Transform and assert no nulls remain
    cleaned_df = handler.transform(messy_dataframe, decisions)
    # Works with either pandas or polars DataFrame
    if hasattr(cleaned_df, "isnull"):
        assert cleaned_df.isnull().sum().sum() == 0
    elif hasattr(cleaned_df, "null_count"):
        assert cleaned_df.null_count().sum() == 0
    else:
        raise AssertionError("Returned data is neither pandas nor polars DataFrame.")


def test_outlier_handler(messy_dataframe):
    """
    Test OutlierHandler: it should detect and cap the outlier in 'age'.
    """
    handler = OutlierHandler()
    # Detect with default params (multiplier=1.5)
    decisions = handler.detect(messy_dataframe, params={})
    # Ensure at least one decision is for 'age'
    age_decisions = [d for d in decisions if d.column == "age"]
    assert age_decisions, "Should detect outlier(s) in 'age' column"
    assert all(isinstance(d, Decision) for d in decisions)

    # Find the decision for 'age'
    age_decision = age_decisions[0]
    bounds = age_decision.parameters
    lower = bounds.get("lower_bound")
    upper = bounds.get("upper_bound")
    assert lower is not None and upper is not None
    assert age_decision.action == "clip"

    # Transform and verify outlier is clipped
    cleaned_df = handler.transform(messy_dataframe, decisions)
    # Access the cleaned 'age' column as numpy array
    age_series = cleaned_df["age"]
    # For pandas/Polars compatibility:
    if hasattr(age_series, "to_numpy"):
        age_values = age_series.to_numpy()
    else:
        age_values = age_series

    # There should be no value above 'upper'
    assert all(v <= upper for v in age_values if pd.notnull(v))
    # The original 150.0 should now be capped to 'upper'
    original_outlier_idx = messy_dataframe["age"].idxmax()
    clipped_value = cleaned_df.loc[original_outlier_idx, "age"]
    assert clipped_value == upper 