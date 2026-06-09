# Architecture State: `cleanframe`

## 1. System Overview
- **Name:** `cleanframe` (Agnostic df cleaner, Pandas/Polars via Narwhals).
- **Metrics:** Strict typing (mypy), linted (ruff), 100% test coverage (12 pass via uv).

## 2. 4-Phase Core Architecture
- **P1 (Core):** Narwhals abstraction, `.audit()`/`.clean()` engine, `df.cf` accessor.
- **P2 (Semantic Domains):** Contextual detection via column identity (`_id`, `email`, `price`).
- **P3 (Telemetry Matrix):** `AuditReport` container, shape deltas, mutation log, `df.cf.report()`.
- **P4 (Plugin Matrix):** `BaseRule` polymorphic ABC contract, dynamic `DataCleaner` registry.

## 3. Codebase Map

### Directory Tree
```text
src/
\---cleanframe
    |   accessor.py
    |   base.py
    |   engine.py
    |   inference.py
    |   pipeline.py
    |   plan.py
    |   py.typed
    |   telemetry.py
    |   types.py
    |   __init__.py
    |   
    \---rules
            cardinality_checker.py
            duplicate_handler.py
            null_handler.py
            outlier_handler.py
            schema_caster.py
            __init__.py

tests/
|   conftest.py
|   test_plugins.py
|   test_rules.py
```

### Structural Responsibilities
- `pipeline.py`: Orchestrates the `DataCleaner` workflow, coordinating sequential rule execution and telemetry generation via `fit_transform`.
- `rules/` (Module): Implements concrete `BaseRule` plugins (e.g., `SchemaCaster`, `NullHandler`) for contextual data mutation and validation.
- `telemetry.py`: Defines the `AuditReport` container to track execution latency, dataset shape deltas, and per-rule modification logs.
- `accessor.py`: Registers the `.cf` namespace on Pandas and Polars DataFrames for fluent `.audit()`, `.clean()`, and `.report()` API surfaces.
- `base.py`: Establishes the `BaseRule` polymorphic ABC contract (`detect`, `transform`, `explain`) ensuring unified plugin registry operations.

## 4. Next-Gen Objectives (Phase 5)

### 1. Predictive ML-Driven Missing Data Imputation Layers
Implement a dynamic imputation engine utilizing lightweight predictive models (e.g., k-NN, gradient boosting via Narwhals-compatible wrappers). This layer evaluates feature correlations to infer and interpolate `null` boundaries, surpassing naive mean/median imputation while maintaining the zero-config design contract.

### 2. Fuzzy/Levenshtein String Distance Matching for Categorical Unification
Integrate vectorized fuzzy string matching (via C-optimized libraries like `RapidFuzz`) within a `CategoricalUnificationRule`. This resolves cardinality bloat by detecting Levenshtein distances to map semantically equivalent but typographically divergent tokens (e.g., "New York", "new_york", "NY") to deterministic cluster centroids.

### 3. Production-Grade CI/CD Automation & PyPI Deployment Pipelines
Establish a GitHub Actions matrix pipeline enforcing `mypy`, `ruff`, and `pytest` across supported Python runtimes (3.11+) and DataFrame backend versions (Pandas 2.x, Polars 1.x). Implement automated semantic versioning, changelog compilation (via `Commitizen`), and a zero-touch PyPI/TestPyPI deployment workflow triggered by annotated release tags.
