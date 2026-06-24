# Out-of-Core Execution with DuckDB

Cleanframe was engineered to operate on strictly hardware-constrained environments (such as machines with an i3 CPU and 8GB RAM). 

When datasets exceed your available RAM, loading them entirely into memory using Pandas or Polars will result in a `MemoryError`. To solve this, Cleanframe introduces the **DuckDBBackend**.

## How It Works

The `DuckDBBackend` bypasses Python's memory limits by mapping Cleanframe's logic to database-level operations:
1. **File Streaming:** It reads massive Parquet or CSV files lazily from disk.
2. **Lazy CTEs:** Cleaning decisions and transformations are compiled into Common Table Expressions (CTEs) in SQL rather than eager Python operations.
3. **RAM Hard-Caps:** You can strictly constrain the memory DuckDB is allowed to use.

## Usage

When instantiating the `DataCleaner`, provide a `DuckDBBackend` configured with your desired `memory_limit`. The backend will execute a `PRAGMA memory_limit` command to ensure the engine respects your hardware ceiling.

```python
from cleanframe.pipeline import DataCleaner
from cleanframe.backends import DuckDBBackend

# 1. Initialize the backend with a strict 4GB RAM cap
backend = DuckDBBackend(memory_limit="4GB")

# 2. Attach the backend to the DataCleaner
cleaner = DataCleaner(backend=backend)

# 3. Stream a massive dataset directly from disk
cleaner.fit_transform("s3://massive-bucket/dataset_2026.parquet")
```

Because of this design, the memory footprint remains extremely low, regardless of the physical size of the data file.
