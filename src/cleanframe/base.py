from abc import ABC, abstractmethod
from typing import Any

from .types import Decision


class BaseRule(ABC):
    """
    Abstract base class for data cleaning rules.

    Subclasses implement stateless detect, transform, and explain methods.
    Custom rules can be registered on DataCleaner via register_rule().
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        """
        Analyze the dataset and return recommended cleaning decisions.

        Args:
            df: Input dataset (pandas or polars DataFrame).
            params: Rule-specific parameters.

        Returns:
            List of Decision objects describing recommended actions.
        """
        ...

    @abstractmethod
    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        """
        Apply approved decisions to the dataset.

        Args:
            df: Input dataset to transform.
            decisions: Approved Decision objects to apply.

        Returns:
            Transformed dataset (same backend as input).
        """
        ...

    @abstractmethod
    def explain(self, decisions: list[Decision]) -> str:
        """
        Return a human-readable summary of the given decisions.

        Args:
            decisions: Decision objects to summarize.

        Returns:
            Summary string for telemetry and reporting.
        """
        ...
