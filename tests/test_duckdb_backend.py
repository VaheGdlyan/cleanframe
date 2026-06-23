import tempfile
from pathlib import Path
import pandas as pd  # type: ignore[import-untyped]
import pytest
from cleanframe.backends.duckdb_backend import DuckDBBackend
from cleanframe.pipeline import DataCleaner
from cleanframe.plan import CleaningPlan
from cleanframe.rules import FuzzyUnificationRule


def test_duckdb_backend_out_of_core() -> None:
    # 1. Create a temporary CSV file with various issues
    data = {
        "id": [1, 2, 3, 4, 5, 5],
        "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-05"],
        "age": [30.0, None, 45.0, 50.0, 35.0, 35.0],
        "salary": [50000.0, 60000.0, 55000.0, 1000000.0, 48000.0, 48000.0],  # 1,000,000 is outlier
        "city": ["New York", "New York", "New Yorkk", "London", "London", "London"],  # "New Yorkk" is fuzzy duplicate
    }
    df_raw = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "input.csv"
        output_csv = Path(tmpdir) / "output.csv"
        df_raw.to_csv(input_csv, index=False)

        # 2. Initialize DataCleaner with DuckDBBackend and register FuzzyUnificationRule
        backend = DuckDBBackend(memory_limit="1GB")
        cleaner = DataCleaner(backend=backend)
        cleaner.register_rule(FuzzyUnificationRule())

        # 3. Assert fit() generates a valid CleaningPlan directly from path
        plan = cleaner.fit(input_csv)
        assert isinstance(plan, CleaningPlan)
        assert len(plan.decisions) > 0

        # Verify baseline stats are present and computed correctly on the full file
        assert plan.baseline_stats is not None
        assert "age" in plan.baseline_stats
        assert plan.baseline_stats["age"]["null_ratio"] == pytest.approx(1/6)
        assert plan.baseline_stats["salary"]["mean"] == pytest.approx(210166.67)

        # 4. Assert transform() streams results directly back to disk
        for d in plan.decisions:
            d.approved = True

        out_path = cleaner.transform(input_csv, plan, output_path=output_csv)
        assert Path(out_path).exists()
        assert out_path == output_csv

        # Load transformed file using pandas to verify correct applications of cleaning rules
        df_cleaned = pd.read_csv(output_csv)

        # Verify duplicate rows dropped (was 6 rows, now 5 rows)
        assert len(df_cleaned) == 5

        # Verify age null imputed (with median = 35.0)
        assert df_cleaned["age"].isnull().sum() == 0
        assert df_cleaned.loc[df_cleaned["id"] == 2, "age"].values[0] == 35.0

        # Verify salary outlier clipped (clipped to upper bound = 74125.0)
        assert df_cleaned.loc[df_cleaned["id"] == 4, "salary"].values[0] == 74125.0

        # Verify city fuzzy duplicates unified
        assert df_cleaned.loc[df_cleaned["id"] == 3, "city"].values[0] == "New York"


def test_duckdb_backend_fit_transform() -> None:
    data = {
        "id": [1, 2, 3],
        "age": [20.0, None, 40.0],
    }
    df_raw = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "input.csv"
        output_csv = Path(tmpdir) / "output.csv"
        df_raw.to_csv(input_csv, index=False)

        backend = DuckDBBackend()
        cleaner = DataCleaner(backend=backend)

        res_path = cleaner.fit_transform(input_csv, output_path=output_csv)
        assert Path(res_path).exists()
        assert res_path == output_csv

        df_cleaned = pd.read_csv(output_csv)
        assert df_cleaned["age"].isnull().sum() == 0
        assert df_cleaned.loc[df_cleaned["id"] == 2, "age"].values[0] == 30.0
