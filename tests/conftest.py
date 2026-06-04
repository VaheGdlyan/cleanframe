import pytest
import pandas as pd
from typing import Any


@pytest.fixture
def messy_dataframe() -> Any:
    # A dataframe containing both nulls and explicit statistical outliers
    return pd.DataFrame(
        {
            "age": [
                25.0,
                30.0,
                None,
                22.0,
                35.0,
                28.0,
                150.0,
            ],  # None is null, 150.0 is an extreme outlier
            "city": [
                "Yerevan",
                "Gyumri",
                "Vanadzor",
                None,
                "Yerevan",
                "Gyumri",
                "Yerevan",
            ],  # None is categorical null
        }
    )
