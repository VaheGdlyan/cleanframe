import pytest
import pandas as pd  # type: ignore
import polars as pl
from cleanframe.pipeline import DataCleaner

def test_leakage_assertion() -> None:
    df = pd.DataFrame({
        "feature_1": [1, 2, 3],
        "is_fraud": [0, 1, 0]
    })
    cleaner = DataCleaner()
    with pytest.raises(AssertionError) as exc_info:
        cleaner.fit(df, target_col="non_existent")
    assert "Target column 'non_existent' not found in dataset" in str(exc_info.value)

def test_target_leakage_semantic_and_correlation_pandas() -> None:
    df = pd.DataFrame({
        "fraud": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "fraud_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
        "high_corr": [0.99, 0.01, 0.98, 0.02, 0.97, 0.03],
        "low_corr": [0.5, 0.1, 0.8, 0.6, 0.2, 0.7],
    })

    plan = df.cf.audit(target_col="fraud")
    assert len(plan.leakage_warnings) == 2
    
    semantic_warn = [w for w in plan.leakage_warnings if "semantically leaks" in w]
    corr_warn = [w for w in plan.leakage_warnings if "high correlation" in w]
    
    assert len(semantic_warn) == 1
    assert "Column 'fraud_date' semantically leaks target 'fraud'" in semantic_warn[0]
    
    assert len(corr_warn) == 1
    assert "Column 'high_corr' has high correlation" in corr_warn[0]

    # Test clean (accessor)
    cleaned_df = df.cf.clean(target_col="fraud")
    report = getattr(cleaned_df, "_cf_report", None)
    assert report is not None
    assert len(report.leakage_warnings) == 2

def test_target_leakage_polars() -> None:
    df = pl.DataFrame({
        "fraud": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "fraud_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
        "high_corr": [0.99, 0.01, 0.98, 0.02, 0.97, 0.03],
        "low_corr": [0.5, 0.1, 0.8, 0.6, 0.2, 0.7],
    })
    
    plan = df.cf.audit(target_col="fraud")  # type: ignore[attr-defined]
    assert len(plan.leakage_warnings) == 2
    
    semantic_warn = [w for w in plan.leakage_warnings if "semantically leaks" in w]
    corr_warn = [w for w in plan.leakage_warnings if "high correlation" in w]
    
    assert len(semantic_warn) == 1
    assert "Column 'fraud_date' semantically leaks target 'fraud'" in semantic_warn[0]
    
    assert len(corr_warn) == 1
    assert "Column 'high_corr' has high correlation" in corr_warn[0]

def test_no_target_col_skips_leakage() -> None:
    df = pd.DataFrame({
        "fraud": [1.0, 0.0, 1.0],
        "fraud_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "high_corr": [0.99, 0.01, 0.98],
    })
    plan = df.cf.audit()
    assert len(plan.leakage_warnings) == 0
