import pandas as pd  # type: ignore
from cleanframe.rules.fuzzy_unification import FuzzyUnificationRule

def test_fuzzy_unification_detect_and_transform():
    # Arrange
    df = pd.DataFrame({
        "city": ["New York", "new york", "New York City", "Los Angeles", "Los Angeles ", "Chicago"],
        "price": [100, 200, 150, 300, 350, 400]
    })
    
    rule = FuzzyUnificationRule(threshold=80.0)
    
    # Act - Detect
    decisions = rule.detect(df, {})
    
    # Assert - Detect
    assert len(decisions) == 1
    d = decisions[0]
    assert d.column == "city"
    assert d.action == "replace_values"
    mapping = d.parameters["mapping"]
    
    assert len(mapping) > 0
    
    # Act - Transform
    transformed_df = rule.transform(df, decisions)
    
    # Assert - Transform
    n_unique_before = df["city"].nunique()
    n_unique_after = transformed_df["city"].nunique()
    assert n_unique_after < n_unique_before

def test_fuzzy_unification_ignores_nulls_and_numerics():
    df = pd.DataFrame({
        "city": ["Paris", "paris", None, "London"],
        "num": [1, 2, 1, 3]
    })
    rule = FuzzyUnificationRule(threshold=80.0)
    decisions = rule.detect(df, {})
    
    assert len(decisions) == 1
    assert decisions[0].column == "city"
    mapping = decisions[0].parameters["mapping"]
    assert "paris" in mapping or "Paris" in mapping
    
    transformed = rule.transform(df, decisions)
    # n_unique drops nulls by default, so Paris and London -> 2
    assert transformed["city"].nunique() == 2

def test_fuzzy_unification_explain():
    df = pd.DataFrame({
        "city": ["Paris", "paris"]
    })
    rule = FuzzyUnificationRule(threshold=80.0)
    decisions = rule.detect(df, {})
    explanation = rule.explain(decisions)
    assert "Fuzzy unification applied" in explanation
    assert "unified" in explanation
