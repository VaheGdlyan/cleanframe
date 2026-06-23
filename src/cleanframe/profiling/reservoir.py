import narwhals as nw
from typing import Any


def sample_reservoir(df: Any, k: int = 10000, seed: int | None = 42) -> Any:
    """
    Extract exactly k random rows from any dataset using uniform sampling.
    If the dataset has fewer than or equal to k rows, returns the dataset as-is.
    """
    ndf = nw.from_native(df)
    n = ndf.shape[0]
    if n <= k:
        return df

    # Use Narwhals-native sample
    sampled = ndf.sample(n=k, seed=seed)
    return nw.to_native(sampled)
