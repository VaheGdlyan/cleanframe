import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from cleanframe.profiling.reservoir import sample_reservoir
from cleanframe.rules.near_duplicate import NearDuplicateDetector
from cleanframe.types import Decision


def test_reservoir_sampling() -> None:
    # Test with Pandas
    df_pd_large = pd.DataFrame({"a": range(15000)})
    sampled_pd_large = sample_reservoir(df_pd_large, k=10000)
    assert len(sampled_pd_large) == 10000
    assert isinstance(sampled_pd_large, pd.DataFrame)

    df_pd_small = pd.DataFrame({"a": range(50)})
    sampled_pd_small = sample_reservoir(df_pd_small, k=10000)
    assert len(sampled_pd_small) == 50
    assert isinstance(sampled_pd_small, pd.DataFrame)

    # Test with Polars
    df_pl_large = pl.DataFrame({"a": range(15000)})
    sampled_pl_large = sample_reservoir(df_pl_large, k=10000)
    assert len(sampled_pl_large) == 10000
    assert isinstance(sampled_pl_large, pl.DataFrame)

    df_pl_small = pl.DataFrame({"a": range(50)})
    sampled_pl_small = sample_reservoir(df_pl_small, k=10000)
    assert len(sampled_pl_small) == 50
    assert isinstance(sampled_pl_small, pl.DataFrame)


def test_near_duplicate_detector() -> None:
    # Create dataset with hidden near duplicates
    # Row 0 and 1 are near duplicates
    # Row 2 and 3 are near duplicates
    # Row 4 is unique
    data = {
        "name": [
            "Johnathan Doe Senior",
            "Johnathan Doe Senir",  # Typo
            "Alice Montgomery Jr.",
            "Alice Montgomry Jr.",  # Typo
            "Bob Smith",
        ],
        "city": [
            "New York",
            "New York",
            "Los Angeles",
            "Los Angeles",
            "Chicago",
        ],
        "age": [45, 45, 29, 29, 32],
    }

    # Test Pandas
    df_pd = pd.DataFrame(data)
    detector = NearDuplicateDetector(num_perm=32, threshold=0.8)
    decisions_pd = detector.detect(df_pd, {})
    assert len(decisions_pd) == 1
    d_pd = decisions_pd[0]
    assert isinstance(d_pd, Decision)
    assert d_pd.rule_name == "NearDuplicateDetector"
    clusters_pd = d_pd.parameters["clusters"]
    assert len(clusters_pd) == 2
    assert clusters_pd[0] == [0, 1]
    assert clusters_pd[1] == [2, 3]

    # Test Polars
    df_pl = pl.DataFrame(data)
    decisions_pl = detector.detect(df_pl, {})
    assert len(decisions_pl) == 1
    d_pl = decisions_pl[0]
    clusters_pl = d_pl.parameters["clusters"]
    assert len(clusters_pl) == 2
    assert clusters_pl[0] == [0, 1]
    assert clusters_pl[1] == [2, 3]

    # Test with no duplicates
    no_dups_data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [20, 30, 40],
    }
    df_no_dups = pd.DataFrame(no_dups_data)
    decisions_no_dups = detector.detect(df_no_dups, {})
    assert len(decisions_no_dups) == 0

    # Test explain
    explanation = detector.explain(decisions_pd)
    assert "Detected 2 near-duplicate cluster(s)" in explanation
    assert "4 rows" in explanation


def test_near_duplicate_detector_empty() -> None:
    detector = NearDuplicateDetector()
    df_empty = pd.DataFrame(columns=["a", "b"])
    assert len(detector.detect(df_empty, {})) == 0


def test_near_duplicate_detector_transform() -> None:
    detector = NearDuplicateDetector()
    df = pd.DataFrame({"a": [1, 2]})
    res = detector.transform(df, [])
    assert res is df
