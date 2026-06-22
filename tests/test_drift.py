import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest
from cleanframe.pipeline import DataCleaner
from cleanframe.plan import CleaningPlan


def test_baseline_stats_computation_pandas():
    """Verify that baseline statistics are computed correctly for a pandas DataFrame."""
    df = pd.DataFrame(
        {
            "num_col": [1.0, 2.0, None, 4.0, 5.0],
            "cat_col": ["a", "b", "a", None, "b"],
        }
    )
    cleaner = DataCleaner()
    plan = cleaner.fit(df)

    assert plan.baseline_stats is not None
    assert "num_col" in plan.baseline_stats
    assert "cat_col" in plan.baseline_stats

    # num_col stats (numeric)
    num_stats = plan.baseline_stats["num_col"]
    assert "null_ratio" in num_stats
    assert num_stats["null_ratio"] == pytest.approx(0.2)
    assert "mean" in num_stats
    # Non-null elements: 1, 2, 4, 5. Mean is 3.0.
    assert num_stats["mean"] == pytest.approx(3.0)
    assert "std_dev" in num_stats
    # Non-null elements: 1, 2, 4, 5. Std is approx 1.82574
    assert num_stats["std_dev"] > 0

    # cat_col stats (categorical)
    cat_stats = plan.baseline_stats["cat_col"]
    assert "null_ratio" in cat_stats
    assert cat_stats["null_ratio"] == pytest.approx(0.2)
    assert "unique_count" in cat_stats
    # Unique values excluding or including nulls?
    # Narwhals n_unique() on 'cat_col' returns number of unique values. Let's see: "a", "b", None. That's 3 unique values.
    assert cat_stats["unique_count"] == 3.0
    assert "cat:a" in cat_stats
    assert "cat:b" in cat_stats
    assert cat_stats["cat:a"] == 1.0
    assert cat_stats["cat:b"] == 1.0


def test_baseline_stats_computation_polars():
    """Verify that baseline statistics are computed correctly for a polars DataFrame."""
    df = pl.DataFrame(
        {
            "num_col": [1.0, 2.0, None, 4.0, 5.0],
            "cat_col": ["a", "b", "a", None, "b"],
        }
    )
    cleaner = DataCleaner()
    plan = cleaner.fit(df)

    assert plan.baseline_stats is not None
    assert "num_col" in plan.baseline_stats
    assert "cat_col" in plan.baseline_stats

    # num_col stats (numeric)
    num_stats = plan.baseline_stats["num_col"]
    assert num_stats["null_ratio"] == pytest.approx(0.2)
    assert num_stats["mean"] == pytest.approx(3.0)
    assert num_stats["std_dev"] > 0

    # cat_col stats (categorical)
    cat_stats = plan.baseline_stats["cat_col"]
    assert cat_stats["null_ratio"] == pytest.approx(0.2)
    assert cat_stats["unique_count"] == 3.0
    assert "cat:a" in cat_stats
    assert "cat:b" in cat_stats


def test_plan_serialization_with_baseline_stats(tmp_path):
    """CleaningPlan save/load should serialize baseline_stats correctly."""
    decisions = []
    baseline_stats = {
        "num_col": {"null_ratio": 0.1, "mean": 5.5, "std_dev": 1.2},
        "cat_col": {"null_ratio": 0.0, "unique_count": 2.0, "cat:x": 1.0, "cat:y": 1.0},
    }
    plan = CleaningPlan(decisions, baseline_stats)

    filepath = tmp_path / "plan.json"
    plan.save(filepath)
    assert filepath.exists()

    loaded_plan = CleaningPlan.load(filepath)
    assert loaded_plan.baseline_stats == baseline_stats


def test_no_drift_on_same_data():
    """Verify that no drift is detected when transform is run on the same dataset."""
    df = pd.DataFrame(
        {
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat_col": ["a", "b", "a", "b", "a"],
        }
    )
    cleaner = DataCleaner()
    plan = cleaner.fit(df)
    
    # Run transform
    cleaner.transform(df, plan)
    report = cleaner.last_report
    assert report is not None
    assert not report.drift_alerts


def test_drift_numeric_mean_shift():
    """Verify drift alerts are generated when numeric mean shifts by > 15%."""
    df_base = pd.DataFrame({"num": [10.0, 10.0, 10.0, 10.0, 10.0]})
    cleaner = DataCleaner()
    plan = cleaner.fit(df_base)

    # Shift mean to 12.0 (+20%) -> Alert expected
    df_shifted = pd.DataFrame({"num": [12.0, 12.0, 12.0, 12.0, 12.0]})
    cleaner.transform(df_shifted, plan)
    report = cleaner.last_report
    assert report is not None
    assert len(report.drift_alerts) == 1
    assert "mean shifted by 20.0%" in report.drift_alerts[0]

    # Shift mean to 11.0 (+10% <= 15%) -> No alert expected
    df_mild_shift = pd.DataFrame({"num": [11.0, 11.0, 11.0, 11.0, 11.0]})
    cleaner.transform(df_mild_shift, plan)
    report = cleaner.last_report
    assert report is not None
    assert not report.drift_alerts


def test_drift_null_ratio_shift():
    """Verify drift alerts are generated when null ratio shifts by > 10%."""
    df_base = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    cleaner = DataCleaner()
    plan = cleaner.fit(df_base)

    # Shift null ratio to 20% (2 nulls) -> Alert expected (diff is 20% > 10%)
    df_shifted = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, None, None]})
    cleaner.transform(df_shifted, plan)
    report = cleaner.last_report
    assert report is not None
    assert any("null_ratio shifted by 20.0%" in alert for alert in report.drift_alerts)

    # Shift null ratio to 5% (not possible in 10 rows, let's use 20 rows. 1 null is 5% <= 10%)
    df_base_20 = pd.DataFrame({"num": list(range(20))})
    plan_20 = cleaner.fit(df_base_20)
    df_mild = pd.DataFrame({"num": [None] + list(range(19))})
    cleaner.transform(df_mild, plan_20)
    report = cleaner.last_report
    assert report is not None
    assert not report.drift_alerts


def test_drift_new_categories():
    """Verify drift alerts are generated when new categories appear in categorical columns."""
    df_base = pd.DataFrame({"cat": ["A", "B", "A", "B"]})
    cleaner = DataCleaner()
    plan = cleaner.fit(df_base)

    # New category "C" appears -> Alert expected
    df_new_cat = pd.DataFrame({"cat": ["A", "B", "C", "A"]})
    cleaner.transform(df_new_cat, plan)
    report = cleaner.last_report
    assert report is not None
    assert len(report.drift_alerts) == 1
    assert "new categories detected" in report.drift_alerts[0]
    assert "C" in report.drift_alerts[0]

    # Subset of categories (only "A") -> No alert expected
    df_subset_cat = pd.DataFrame({"cat": ["A", "A", "A", "A"]})
    cleaner.transform(df_subset_cat, plan)
    report = cleaner.last_report
    assert report is not None
    assert not report.drift_alerts
