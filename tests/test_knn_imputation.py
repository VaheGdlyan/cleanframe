import pytest
import pandas as pd  # type: ignore
import polars as pl
from cleanframe.pipeline import DataCleaner
from cleanframe.rules.knn_imputer import KNNImputationRule

def test_knn_imputation_e2e_pandas() -> None:
    df = pd.DataFrame({
        "x": [1.0, 1.1, 5.0],
        "y": [2.0, None, 10.0],
        "z": [3.0, 3.1, 15.0]
    })
    
    cleaner = DataCleaner(impute_strategy="knn")
    plan = cleaner.fit(df, params_map={"KNNImputationRule": {"k": 1}})
    
    knn_decisions = [d for d in plan.decisions if d.rule_name == "KNNImputationRule"]
    assert len(knn_decisions) == 1
    assert knn_decisions[0].column == "y"
    assert knn_decisions[0].action == "knn_impute"
    assert "kNN imputed based on" in knn_decisions[0].parameters["explanation"]
    
    cleaned_df = cleaner.transform(df, plan)
    imputed_val = cleaned_df.loc[1, "y"]
    assert abs(imputed_val - 2.0) < 1e-5

def test_knn_imputation_e2e_polars() -> None:
    df = pl.DataFrame({
        "x": [1.0, 1.1, 5.0],
        "y": [2.0, None, 10.0],
        "z": [3.0, 3.1, 15.0]
    })
    
    cleaner = DataCleaner(impute_strategy="knn")
    plan = cleaner.fit(df, params_map={"KNNImputationRule": {"k": 1}})
    
    knn_decisions = [d for d in plan.decisions if d.rule_name == "KNNImputationRule"]
    assert len(knn_decisions) == 1
    assert knn_decisions[0].column == "y"
    
    cleaned_df = cleaner.transform(df, plan)
    imputed_val = cleaned_df.filter(pl.col("x") == 1.1).select("y").item()
    assert abs(imputed_val - 2.0) < 1e-5

def test_knn_circuit_breaker() -> None:
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, None, 10.0, 4.0, 8.0],
        "z": [3.0, 3.1, 15.0, 5.0, 9.0]
    })
    
    rule = KNNImputationRule(k=2, max_rows=3)
    
    with pytest.warns(RuntimeWarning, match="circuit breaker triggered"):
        decisions = rule.detect(df, {})
        
    assert len(decisions) == 1
    d = decisions[0]
    assert d.column == "y"
    assert d.action == "median_fallback"
    assert "exceeded limit" in d.parameters["explanation"]
    
    cleaned_df = rule.transform(df, decisions)
    assert cleaned_df.loc[1, "y"] == 6.0

def test_knn_no_complete_columns_fallback() -> None:
    df = pd.DataFrame({
        "x": [1.0, None, 3.0],
        "y": [2.0, 4.0, None]
    })
    
    rule = KNNImputationRule(k=2)
    decisions = rule.detect(df, {})
    
    assert len(decisions) == 2
    assert decisions[0].action == "median_fallback"
    assert decisions[1].action == "median_fallback"
    assert "no complete numeric columns" in decisions[0].parameters["explanation"]
    
    cleaned_df = rule.transform(df, decisions)
    assert cleaned_df.loc[1, "x"] == 2.0
    assert cleaned_df.loc[2, "y"] == 3.0
