# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-24

### Added
- **Backend-Agnostic Accessor:** Introduced the `.cf` accessor via Narwhals, enabling exact generic type preservation (`FrameT`) across Pandas and Polars.
- **Out-of-Core Execution Engine:** Implemented the `DuckDBBackend` allowing `PRAGMA memory_limit` enforcement and lazy SQL CTE execution to stream massive Parquet/CSV files.
- **Reservoir Profiling:** Added Vitter's Algorithm R for exact, low-memory ($O(k)$) statistical profiling.
- **Intelligence Rules:** Added `KNNImputationRule`, `FuzzyUnificationRule`, `CrossColumnConsistencyRule`, `NearDuplicateDetector` (MinHash-LSH), and `TargetLeakage` checks.
- **Zero-Config Observability:** Implemented the `TelemetryEvent` and `AuditReport` framework with `LocalJsonLinesSink` and `StandardLoggingSink`.
- **Declarative Configuration:** Added support for parsing TOML pipelines for rigorous production reproducibility.
- **Plugin Ecosystem:** Enabled structural subtyping (`RuleProtocol`) and `importlib.metadata` entry point resolution for zero-config custom rules.
