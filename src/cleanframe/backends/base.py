from typing import Any, Protocol, runtime_checkable
from pathlib import Path


@runtime_checkable
class BackendProtocol(Protocol):
    def read_schema(self, path: str | Path) -> dict[str, str]:
        """Read the schema of the file at path and return a dict of {column_name: type_str}."""
        ...

    def compute_statistics(self, path: str | Path) -> dict[str, dict[str, float]]:
        """Compute baseline statistics (null_ratio, mean, std_dev, unique_count) for the dataset."""
        ...

    def sample_to_dataframe(self, path: str | Path, k: int = 10000) -> Any:
        """Extract a sample of k rows and return it as a pandas or polars DataFrame."""
        ...

    def execute_transform(
        self,
        input_path: str | Path,
        output_path: str | Path,
        plan: Any,
    ) -> None:
        """Apply the approved decisions in the plan and write the result to output_path."""
        ...
