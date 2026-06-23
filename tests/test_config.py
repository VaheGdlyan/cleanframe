from cleanframe.pipeline import DataCleaner


def test_from_config(tmp_path):
    config_content = """
[cleaner]
impute_strategy = "knn"

[rules.KNNImputationRule]
k = 3
max_rows = 15000

[rules.NullHandler]
numeric_strategy = "mean"
"""
    config_file = tmp_path / "cleanframe.toml"
    config_file.write_text(config_content, encoding="utf-8")

    cleaner = DataCleaner.from_config(config_file)
    assert cleaner.impute_strategy == "knn"

    # Check that rule parameters were correctly applied
    # KNNImputationRule constructor params should be set:
    knn_rules = [r for r in cleaner.rules if type(r).__name__ == "KNNImputationRule"]
    assert len(knn_rules) == 1
    knn_rule = knn_rules[0]
    assert getattr(knn_rule, "k") == 3
    assert getattr(knn_rule, "max_rows") == 15000

    # NullHandler parameters should be in self.config_params
    assert cleaner.config_params["NullHandler"] == {"numeric_strategy": "mean"}
