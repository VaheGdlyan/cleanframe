import importlib.metadata
from typing import Any
from unittest.mock import MagicMock, patch

import narwhals as nw
import polars as pl
import pytest

from cleanframe.base import RuleProtocol
from cleanframe.pipeline import DataCleaner
from cleanframe.types import Decision
from cleanframe.registry import discover_plugins


class DuckTypedRule:
    """A rule class that does NOT inherit from BaseRule but matches RuleProtocol structurally."""

    @property
    def name(self) -> str:
        return "DuckTypedRule"

    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        ndf = nw.from_native(df)
        decisions = []
        if "value" in ndf.columns:
            decisions.append(
                Decision(
                    rule_name=self.name,
                    column="value",
                    action="multiply_by_two",
                    parameters={},
                    signal_strength=1.0,
                    rationale="Multiply all values by two",
                    approved=True,
                )
            )
        return decisions

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        ndf = nw.from_native(df)
        for d in decisions:
            if d.action == "multiply_by_two":
                col = d.column
                ndf = ndf.with_columns((ndf[col] * 2).alias(col))
        return ndf.to_native()


def test_duck_typed_rule_registration():
    """Verify that an object implementing RuleProtocol passes registration."""
    cleaner = DataCleaner(rules=[])
    rule = DuckTypedRule()

    # Assert it implements RuleProtocol using isinstance runtime check
    assert isinstance(rule, RuleProtocol)

    # Register the duck-typed rule
    cleaner.register_rule(rule)
    assert rule in cleaner.rules


def test_invalid_rule_registration():
    """Verify that registering an object that does not match RuleProtocol raises TypeError."""
    cleaner = DataCleaner(rules=[])

    class BadRule:
        pass

    with pytest.raises(TypeError, match="Registered rule must implement RuleProtocol"):
        cleaner.register_rule(BadRule())  # type: ignore[arg-type]


def test_duck_typed_rule_execution():
    """Verify that a registered duck-typed rule executes correctly in the pipeline."""
    df = pl.DataFrame({"value": [1, 2, 3]})
    cleaner = DataCleaner(rules=[])
    cleaner.register_rule(DuckTypedRule())

    plan = cleaner.fit(df)
    res_df = cleaner.transform(df, plan)

    assert res_df["value"].to_list() == [2, 4, 6]

    report = cleaner.last_report
    assert report is not None
    assert "DuckTypedRule" in report.mutations
    # Fallback explain behavior should display decision action since DuckTypedRule doesn't have explain()
    assert any("Multiply by two" in entry for entry in report.mutations["DuckTypedRule"])


def test_plugin_discovery_mocked():
    """Mock importlib.metadata.entry_points to verify automatic discovery of plugins."""
    mock_entry_point = MagicMock(spec=importlib.metadata.EntryPoint)
    mock_entry_point.name = "mock_rule"
    mock_entry_point.load.return_value = DuckTypedRule

    # Mock entry_points to return our mock entry point.
    # We must support both the dict-like entry_points() return value (pre 3.10)
    # and the EntryPoints sequence (Python 3.10+).
    # Since select() is used inside discover_plugins fallback, we mock it comprehensively.
    with patch("importlib.metadata.entry_points") as mock_entry_points:
        # In Python 3.10+, entry_points(group=...) is called.
        # Let's make the mock return a sequence or handle group kwarg.
        mock_eps_sequence = MagicMock()
        mock_eps_sequence.__iter__.return_value = [mock_entry_point]
        mock_entry_points.return_value = mock_eps_sequence

        # Test standalone discovery function
        rules = discover_plugins()
        assert len(rules) == 1
        assert isinstance(rules[0], DuckTypedRule)

        # Verify that DataCleaner.__init__ discovers and loads it
        cleaner = DataCleaner()
        # Find the loaded DuckTypedRule instance
        matching_rules = [r for r in cleaner.rules if isinstance(r, DuckTypedRule)]
        assert len(matching_rules) == 1
