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


def test_cross_column_consistency_drop() -> None:
    df = pl.DataFrame({
        "age": [12, 25, 15, 30],
        "role": ["Manager", "Employee", "Manager", "Manager"]
    })

    # Constraint: Cannot be under 18 and hold a Manager title.
    constraint = ConsistencyConstraint(
        name="Underage Manager",
        condition=(nw.col("age") < 18) & (nw.col("role") == "Manager"),
        error_msg="Violation: User is under 18 but has a Manager role.",
        action="drop"
    )

    rule = CrossColumnConsistencyRule(constraints=[constraint])
    cleaner = DataCleaner(rules=[rule])

    # Assert fit generates a valid plan and detects the 2 violating rows
    plan = cleaner.fit(df)
    assert len(plan.decisions) == 1
    d = plan.decisions[0]
    assert d.column == "Underage Manager"
    assert d.action == "drop"
    assert d.parameters["violation_count"] == 2
    assert d.parameters["constraint_action"] == "drop"

    # Transform: violating rows should be dropped
    cleaned_df = cleaner.transform(df, plan)
    
    # Original length was 4, 2 rows had (12, Manager) and (15, Manager) -> both should be dropped.
    assert len(cleaned_df) == 2
    assert list(cleaned_df["age"]) == [25, 30]

    # Verify telemetry report tracks the drop mutation
    report = cleaner.last_report
    assert report is not None
    assert report.initial_shape == (4, 2)
    assert report.final_shape == (2, 2)
    
    # Check that mutations contains the explanation
    assert "CrossColumnConsistencyRule" in report.mutations
    summary = report.mutations["CrossColumnConsistencyRule"][0]
    assert "Constraint 'Underage Manager' violated by 2 rows" in summary
    assert "Dropped 2 rows" in summary

    # Check telemetry events
    mutation_events = [e for e in report.events if e.event_type == "rule_mutation"]
    assert len(mutation_events) == 1
    assert mutation_events[0].payload.get("dropped_rows") == 2

