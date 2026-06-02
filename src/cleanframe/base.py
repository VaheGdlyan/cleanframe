from abc import ABC, abstractmethod
from typing import Any
from .types import Decision

class BaseRule(ABC):
    """
    Abstract base class for data cleaning rules.

    All subclasses must implement pure, stateless logic for detection,
    transformation, and explanation of data quality issues and remediations.
    Methods should not cause side effects or rely on internal state;
    all decisions must be determined solely by explicit inputs (data and parameters).

    Rules implementing this base can be composed and run reliably in
    multi-threaded or distributed environments.
    """

    @abstractmethod
    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        """
        Analyze the dataset and identify issues or improvement opportunities.

        Args:
            df: The input dataset (e.g., a DataFrame) to scan.
            params: Rule-specific parameters.

        Returns:
            A list of Decision objects describing the recommended actions.

        This method must be pure and stateless.
        """
        pass

    @abstractmethod
    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        """
        Applies the given decisions to the input dataset, producing a modified copy.

        Args:
            df: The input dataset to transform.
            decisions: The list of Decision objects to apply.

        Returns:
            The transformed dataset (same type as input).

        This method must be pure and stateless.
        """
        pass

    @abstractmethod
    def explain(self, decisions: list[Decision]) -> str:
        """
        Creates a human-readable explanation for a set of decisions.

        Args:
            decisions: The list of Decision objects to explain.

        Returns:
            A string summarizing the rationale and impact of the decisions.

        This method must be pure and stateless.
        """
        pass 
