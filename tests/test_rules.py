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
    from cleanframe.rules import SchemaCaster

    def test_schema_caster(messy_dataframe):
        """
        Test SchemaCaster: it should detect and cast 'age' from float to Int64.
        """
        handler = SchemaCaster()
        params = {"schema": {"age": "Int64"}}

        # Detect casting needs
        decisions = handler.detect(messy_dataframe, params)
        assert decisions, "SchemaCaster should generate a Decision for 'age' type mismatch"
        # At least one decision should be for 'age'
        age_decisions = [d for d in decisions if d.column == "age"]
        assert age_decisions, "Should detect type mismatch in 'age' column"
        assert all(isinstance(d, Decision) for d in decisions)
        assert age_decisions[0].action == "cast"

        # Transform and check dtype
        casted_df = handler.transform(messy_dataframe, decisions)
        # For pandas backend
        if hasattr(casted_df, "dtypes"):
            dtype = casted_df.dtypes["age"]
            # Works for pandas nullable integer
            # Allow both 'Int64' and 'int64' (compatibility/strictness may vary)
            assert str(dtype) in ("Int64", "int64")
        # For polars backend
        elif hasattr(casted_df["age"], "dtype"):
            dtype = str(casted_df["age"].dtype)
            # Accept 'Int64', 'Int64(', 'Int64', or equivalent, polars may say 'Int64' or 'Int64(?)'
            assert "Int64" in dtype or "int64" in dtype.lower()
        else:
            raise AssertionError("Returned data is neither pandas nor polars DataFrame.")

            from cleanframe.rules import DuplicateHandler
            import polars as pl

            def test_duplicate_handler():
                # Create a polars DataFrame with one exact duplicate row
                df = pl.DataFrame({
                    "name": ["Alice", "Bob", "Alice", "Carol"],
                    "age": [25, 30, 25, 22]
                })
                # The first and third rows are identical

                handler = DuplicateHandler()
                # Detect duplicates: no subset specified, drop full duplicates
                decisions = handler.detect(df, params={})
                assert isinstance(decisions, list)
                assert len(decisions) == 1
                decision = decisions[0]
                assert decision.action == "drop_duplicates"
                assert decision.column == "all"
                assert decision.parameters.get("num_duplicates", 0) == 1

                # Transform and verify a duplicate row is dropped
                deduped_df = handler.transform(df, decisions)
                # For polars, resulting DataFrame has shape property
                assert hasattr(deduped_df, "shape")
                assert deduped_df.shape[0] == 3

                from cleanframe.rules import DuplicateHandler
                import polars as pl

                def test_duplicate_handler():
                    # Create a polars DataFrame with one exact duplicate row
                    df = pl.DataFrame({
                        "name": ["Alice", "Bob", "Alice", "Carol"],
                        "age": [25, 30, 25, 22]
                    })
                    # The first and third rows are identical

                    handler = DuplicateHandler()                  # Detect duplicates: no subset specified, drop full duplicates


                    from cleanframe.rules import CardinalityChecker
                    import polars as pl

                    def test_cardinality_checker():
                        # Create a small DataFrame
                        df = pl.DataFrame({
                            "constant": [1, 1, 1, 1, 1],
                            "unique_id": ["id1", "id2", "id3", "id4", "id5"],
                            "normal": ["a", "a", "b", "b", "c"],
                        })

                        checker = CardinalityChecker()
                        decisions = checker.detect(df, params={})

                        # Assert a decision drops "constant" column
                        drop_constant = next(
                            (d for d in decisions if d.column == "constant" and d.action == "drop_column"),
                            None)
                        assert drop_constant is not None, "Missing drop_column decision for 'constant'"

                        # Assert a decision flags "unique_id" column
                        flag_unique_id = next(
                            (d for d in decisions if d.column == "unique_id" and d.action == "flag_id"),
                            None)
                        assert flag_unique_id is not None, "Missing flag_id decision for 'unique_id'"

                        # Test transform
                        transformed = checker.transform(df, decisions)
                        # For polars, columns property
                        assert "constant" not in transformed.columns
                        assert "unique_id" in transformed.columns
                        assert "normal" in transformed.columns

     