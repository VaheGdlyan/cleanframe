# Architecture State: `cleanframe`

This document provides a comprehensive overview of the `cleanframe` Phase 1-6 architectural decisions, design patterns, and hardware constraints. It is intended to serve as maximum context for future development and LLM-assisted workflows.

## 1. Executive Summary

`cleanframe` is a **hardware-optimized, backend-agnostic data intelligence framework**. It utilizes Narwhals to seamlessly support both Pandas and Polars DataFrames without code duplication or tight coupling.

**Strict Hardware Constraint:** The framework is explicitly optimized for highly constrained hardware—specifically, 2011-era i3 CPUs and systems with 8GB RAM. To achieve this, it leverages out-of-core streaming execution and algorithmic/mathematical bypasses to prevent memory spikes, completely avoiding naive O(n²) operations or full-memory aggregations.

## 2. Core Design Principles

- **"Do No Harm":** By default, data cleaning rules only act as auditors, raising warnings or cataloging detected anomalies. Explicit mutation actions (e.g., `action="drop"`) must be provided by the user to modify the dataset.
- **Extensibility:** The architecture relies on zero-configuration plugins. Users can inject custom rules using Python's `importlib.metadata` entry points and duck-typing interfaces via `Protocol`, eliminating complex registry classes.
- **Observability:** Telemetry and system metrics are emitted as zero-dependency structured events. Sinks support writing JSON-Lines files (`.jsonl`) or using Python's built-in `logging` module to maintain maximum visibility without external heavy-weight dependencies.

## 3. System Architecture & Subpackages

### The Pipeline (`pipeline.py`)
The `DataCleaner` acts as the execution coordinator. It relies on a memory-efficient, generator-based architecture:
- `fit()`: Yields audit findings without modifying the dataset.
- `transform()`: Applies authorized cleaning logic based on the audit decisions.
- `fit_transform()`: A streamlined pathway that chains detection and transformation sequentially in chunks where applicable.

### The Accessor (`accessor.py`)
`cleanframe` exposes its API natively to DataFrames using a custom Pandas/Polars accessor under the `.cf` namespace. The `CfAccessor` class leverages `typing.Generic[FrameT]` to preserve the exact type of the backend DataFrame (Pandas or Polars) through type-checking systems, ensuring that methods like `.clean()` retain strict backend continuity without type degradation.

### The Rules Engine (`rules/`)
All rules conform to the structural subtyping interface, `RuleProtocol`:

```python
from typing import Any, Protocol, runtime_checkable
from .types import Decision

@runtime_checkable
class RuleProtocol(Protocol):
    @property
    def name(self) -> str: ...

    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]: ...

    def transform(self, df: Any, decisions: list[Decision]) -> Any: ...
```

The system relies on a central `Decision` dataclass to communicate anomalies and the recommended actions. Key rules implemented in the framework include:
- **KNN Imputer:** Nearest-neighbor logic for multivariate missing value imputation.
- **Fuzzy Unification:** Case-insensitive, order-preserving Levenshtein distance string matching.
- **Cross Column Consistency:** Relational constraint validation with `action="drop"` enforcement capabilities.
- **MinHash-LSH:** Near-duplicate record detection built on locality-sensitive hashing to bypass O(n²) comparisons.
- **Target Leakage:** Checks to prevent predictive model contamination.

### The Observability Matrix (`telemetry/`)
System logs, rule decisions, and pipeline performance are emitted as structured dictionaries wrapped in a `TelemetryEvent` dataclass (containing `event_type`, `timestamp`, `run_id`, `rule_name`, `column`, and `payload`). These events are dispatched to a `TelemetrySink` (e.g., `LocalJsonLinesSink`, `StandardLoggingSink`) which ultimately compile the `AuditReport`.

## 4. The Hardware Bypasses (Phase 6 Algorithms)

Given the 8GB RAM / i3 CPU constraint, `cleanframe` features two critical algorithm bypasses:

1. **Reservoir Profiling (`profiling/reservoir.py`):** Uses Vitter's Algorithm R for data sampling. Instead of computing metrics across massive arrays, the framework processes an exact $k$-sized random sample (e.g., $k=10,000$) in a single sequential pass. This caps the profiling memory footprint at ~4MB regardless of the underlying dataset's size (O(k) space complexity).
2. **DuckDB Out-of-Core Execution (`backends/duckdb_backend.py`):** The `DuckDBBackend` enables the framework to clean multi-gigabyte Parquet and CSV files directly from disk. It runs `PRAGMA memory_limit = '4GB'` to hard-cap RAM consumption at the database level and pushes transformations down into lazy SQL CTEs, entirely bypassing Python memory limits.

## 5. Configuration & Developer Experience (DX)

- **Declarative TOML Configuration (`config.py`):** `cleanframe` supports a declarative configuration layer, enabling engineers to define rule pipelines, parameters, and thresholds in `.toml` files for production deployments (parseable via Python 3.11 `tomllib`).
- **Developer Polish:** Rule APIs are optimized for real-world messy data scenarios. For example, the `FuzzyUnificationRule` accepts `exclude_cols` (to shield IDs/emails from being incorrectly clustered) and `pre_lowercase` parameters, allowing case-insensitive matching while preserving the original string's canonical casing.

## 6. Current Status

- **Version:** 1.0.0 (Release Candidate)
- **Quality Gates:** 
  - 51 passing unit tests.
  - 100% strict type coverage (`mypy --strict`).
  - Zero static analysis warnings (`ruff`).
