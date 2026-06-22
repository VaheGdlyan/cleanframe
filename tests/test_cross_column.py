import pandas as pd  # type: ignore
import polars as pl
import narwhals as nw
from cleanframe.pipeline import DataCleaner
from cleanframe.rules.cross_column import CrossColumnConsistencyRule, ConsistencyConstraint

def test_cross_column_consistency_pandas() -> None:
    df = pd.DataFrame({
        "age": [12, 25, 15, 30],
        "has_driving_license": [True, True, True, False]
    })
    
    constraint = ConsistencyConstraint(
        name="Underage Driver",
        condition=(nw.col("age") < 16) & nw.col("has_driving_license"),
        error_msg="Cannot have a driving license under age 16"
    )
    
    rule = CrossColumnConsistencyRule(constraints=[constraint])
    cleaner = DataCleaner(rules=[rule])
    
    plan = cleaner.fit(df)
    assert len(plan.decisions) == 1
    d = plan.decisions[0]
    assert d.column == "Underage Driver"
    assert d.action == "flag_violation"
    assert d.parameters["violation_count"] == 2
    
    cleaned_df = cleaner.transform(df, plan)
    pd.testing.assert_frame_equal(df, cleaned_df)
    
    report = cleaner.last_report
    assert report is not None
    assert len(report.consistency_warnings) == 1
    assert "CONSTRAINT VIOLATION: 2 rows failed 'Underage Driver' rule" in report.consistency_warnings[0]

def test_cross_column_consistency_polars() -> None:
    df = pl.DataFrame({
        "age": [12, 25, 15, 30],
        "has_driving_license": [True, True, True, False]
    })
    
    constraint = ConsistencyConstraint(
        name="Underage Driver",
        condition=(nw.col("age") < 16) & nw.col("has_driving_license"),
        error_msg="Cannot have a driving license under age 16"
    )
    
    rule = CrossColumnConsistencyRule(constraints=[constraint])
    cleaner = DataCleaner(rules=[rule])
    
    plan = cleaner.fit(df)
    assert len(plan.decisions) == 1
    d = plan.decisions[0]
    assert d.column == "Underage Driver"
    
    cleaned_df = cleaner.transform(df, plan)
    assert cleaned_df.equals(df)
    
    report = cleaner.last_report
    assert report is not None
    assert len(report.consistency_warnings) == 1
    assert "CONSTRAINT VIOLATION: 2 rows failed 'Underage Driver' rule" in report.consistency_warnings[0]
