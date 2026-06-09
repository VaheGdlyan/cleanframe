import pandas as pd
import polars as pl
from cleanframe.pipeline import DataCleaner
from cleanframe.plan import CleaningPlan
from cleanframe.rules import (
    CardinalityChecker,
    DuplicateHandler,
    NullHandler,
    OutlierHandler,
    SchemaCaster,
)
from cleanframe.types import Decision


def test_null_handler(messy_dataframe):
    """NullHandler should detect nulls in both columns and fill them."""
    handler = NullHandler()
    decisions = handler.detect(messy_dataframe, params={})
    detected_cols = {d.column for d in decisions}

    assert "age" in detected_cols
    assert "city" in detected_cols
    assert all(isinstance(d, Decision) for d in decisions)
    assert len(decisions) == 2

    cleaned_df = handler.transform(messy_dataframe, decisions)
    if hasattr(cleaned_df, "isnull"):
        assert cleaned_df.isnull().sum().sum() == 0
    elif hasattr(cleaned_df, "null_count"):
        assert cleaned_df.null_count().sum() == 0
    else:
        raise AssertionError("Returned data is neither pandas nor polars DataFrame.")


def test_outlier_handler_skips_identifier_columns():
    """OutlierHandler should skip columns inferred as identifiers."""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 999_999],
            "age": [25.0, 30.0, 22.0, 150.0],
        }
    )
    handler = OutlierHandler()
    decisions = handler.detect(df, params={})
    detected_cols = {d.column for d in decisions}

    assert "user_id" not in detected_cols
    assert "age" in detected_cols


def test_outlier_handler(messy_dataframe):
    """OutlierHandler should detect and cap the outlier in 'age'."""
    handler = OutlierHandler()
    decisions = handler.detect(messy_dataframe, params={})
    age_decisions = [d for d in decisions if d.column == "age"]
    assert age_decisions, "Should detect outlier(s) in 'age' column"
    assert all(isinstance(d, Decision) for d in decisions)

    age_decision = age_decisions[0]
    bounds = age_decision.parameters
    lower = bounds.get("lower_bound")
    upper = bounds.get("upper_bound")
    assert lower is not None and upper is not None
    assert age_decision.action == "clip"

    cleaned_df = handler.transform(messy_dataframe, decisions)
    age_series = cleaned_df["age"]
    if hasattr(age_series, "to_numpy"):
        age_values = age_series.to_numpy()
    else:
        age_values = age_series

    assert all(v <= upper for v in age_values if pd.notnull(v))
    original_outlier_idx = messy_dataframe["age"].idxmax()
    clipped_value = cleaned_df.loc[original_outlier_idx, "age"]
    assert clipped_value == upper


def test_schema_caster_datetime_hint():
    """SchemaCaster should coerce datetime_hint string columns to Datetime."""
    df = pd.DataFrame(
        {
            "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "name": ["Alice", "Bob", "Carol"],
        }
    )
    handler = SchemaCaster()
    decisions = handler.detect(df, params={})

    created_decisions = [d for d in decisions if d.column == "created_at"]
    assert created_decisions, "Should detect datetime_hint column 'created_at'"
    assert created_decisions[0].parameters["target_type"] == "Datetime"
    assert created_decisions[0].parameters["method"] == "to_datetime"
    assert not any(d.column == "name" for d in decisions)

    casted_df = handler.transform(df, decisions)
    dtype = str(casted_df.dtypes["created_at"])
    assert "datetime" in dtype.lower()


def test_schema_caster():
    """SchemaCaster should detect and cast 'age' from float to Int64."""
    df = pd.DataFrame({"age": [25.0, 30.0, 22.0]})
    handler = SchemaCaster()
    params = {"schema": {"age": "Int64"}}

    decisions = handler.detect(df, params)
    assert decisions, "SchemaCaster should generate a Decision for 'age' type mismatch"
    age_decisions = [d for d in decisions if d.column == "age"]
    assert age_decisions, "Should detect type mismatch in 'age' column"
    assert all(isinstance(d, Decision) for d in decisions)
    assert age_decisions[0].action == "cast"

    casted_df = handler.transform(df, decisions)
    if hasattr(casted_df, "dtypes"):
        dtype = casted_df.dtypes["age"]
        assert str(dtype) in ("Int64", "int64")
    elif hasattr(casted_df["age"], "dtype"):
        dtype = str(casted_df["age"].dtype)
        assert "Int64" in dtype or "int64" in dtype.lower()
    else:
        raise AssertionError("Returned data is neither pandas nor polars DataFrame.")


def test_duplicate_handler():
    """DuplicateHandler should detect and drop one exact duplicate row."""
    df = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice", "Carol"],
            "age": [25, 30, 25, 22],
        }
    )

    handler = DuplicateHandler()
    decisions = handler.detect(df, params={})
    assert isinstance(decisions, list)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.action == "drop_duplicates"
    assert decision.column == "all"
    assert decision.parameters.get("num_duplicates", 0) == 1

    deduped_df = handler.transform(df, decisions)
    assert hasattr(deduped_df, "shape")
    assert deduped_df.shape[0] == 3


def test_cardinality_checker():
    """CardinalityChecker should flag constant and near-unique columns."""
    df = pl.DataFrame(
        {
            "constant": [1, 1, 1, 1, 1],
            "unique_id": ["id1", "id2", "id3", "id4", "id5"],
            "normal": ["a", "a", "b", "b", "c"],
        }
    )

    checker = CardinalityChecker()
    decisions = checker.detect(df, params={})

    drop_constant = next(
        (d for d in decisions if d.column == "constant" and d.action == "drop_column"),
        None,
    )
    assert drop_constant is not None, "Missing drop_column decision for 'constant'"

    flag_unique_id = next(
        (d for d in decisions if d.column == "unique_id" and d.action == "flag_id"),
        None,
    )
    assert flag_unique_id is not None, "Missing flag_id decision for 'unique_id'"

    transformed = checker.transform(df, decisions)
    assert "constant" not in transformed.columns
    assert "unique_id" in transformed.columns
    assert "normal" in transformed.columns


def test_accessor_e2e():
    """The .cf accessor should audit and clean a messy polars DataFrame."""
    df = pl.DataFrame(
        {
            "null_col": [1.0, 2.0, None, 4.0, 5.0],
            "outlier_col": [1.0, 2.0, 1.5, 1000.0, 2.0],
            "constant_col": ["a", "a", "a", "a", "a"],
        }
    )

    plan = df.cf.audit()
    assert plan is not None

    clean_df = df.cf.clean()

    assert "constant_col" not in clean_df.columns

    null_count = clean_df.select(pl.col("null_col").is_null().sum()).item()
    assert null_count == 0

    max_outlier = clean_df.select(pl.col("outlier_col").max()).item()
    assert max_outlier < 500


def test_plan_serialization(tmp_path):
    """CleaningPlan.save/load should round-trip decisions to JSON."""
    decisions = [
        Decision(
            rule_name="CardinalityChecker",
            column="constant_col",
            action="drop_column",
            parameters={},
            signal_strength=0.95,
            rationale="Only one unique value.",
            approved=True,
        ),
        Decision(
            rule_name="NullHandler",
            column="null_col",
            action="impute_median",
            parameters={"strategy": "median"},
            signal_strength=0.85,
            rationale="Contains missing values.",
            approved=False,
        ),
    ]

    plan = CleaningPlan(decisions)
    filepath = tmp_path / "plan.json"
    plan.save(filepath)
    assert filepath.exists()

    loaded_plan = CleaningPlan.load(filepath)
    assert len(loaded_plan.decisions) == len(decisions)

    for original, loaded in zip(decisions, loaded_plan.decisions, strict=True):
        assert loaded.rule_name == original.rule_name
        assert loaded.column == original.column
        assert loaded.action == original.action
        assert loaded.parameters == original.parameters
        assert loaded.signal_strength == original.signal_strength
        assert loaded.rationale == original.rationale
        assert loaded.approved == original.approved


def test_domain_aware_detection():
    """Pipeline should respect semantic types for identifiers and datetime hints."""
    # 1. Create DataFrame
    df = pl.DataFrame({
        "user_id": [101, 102, 103],
        "sale_price": ["12.5", "15.0", "10.2"],
        "signup_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
    })

    # 2. Run DataCleaner .fit()
    cleaner = DataCleaner()
    plan = cleaner.fit(df)

    # 3. Assert "user_id" is inferred as identifier and skipped by outlier engine
    # Find all decisions referring to outlier handling on "user_id"
    outlier_decisions_user_id = [
        d for d in plan.decisions
        if d.rule_name == "OutlierHandler" and d.column == "user_id"
    ]
    # For identifier columns, there should be no outlier decisions
    assert len(outlier_decisions_user_id) == 0, "user_id should be skipped by outlier rule"

    # 4. Assert "signup_date" triggers a datetime parsing suggestion or transformation
    datetime_action_found = False
    for d in plan.decisions:
        if d.column == "signup_date" and d.rule_name in {"SchemaCaster", "NullHandler", "OutlierHandler"}:
            if (
                ("date" in d.action or "datetime" in d.action or "parse" in d.action)
                or ("to_datetime" in d.action or "cast" in d.action)
            ):
                datetime_action_found = True
                break
    assert datetime_action_found, "signup_date should trigger a datetime parsing suggestion or transformation"

