import importlib.metadata
import logging
from .base import RuleProtocol

logger = logging.getLogger(__name__)


def discover_plugins() -> list[RuleProtocol]:
    """
    Discover and load third-party rules registered under the 'cleanframe.rules' entry point group.

    Returns:
        A list of successfully loaded RuleProtocol instances.
    """
    discovered: list[RuleProtocol] = []
    from typing import Any
    eps: Any
    try:
        # In Python 3.10+, entry_points accepts group directly
        eps = importlib.metadata.entry_points(group="cleanframe.rules")
    except (TypeError, ValueError, AttributeError):
        try:
            # Fallback for older python or environment anomalies where group parameter is not supported
            eps = importlib.metadata.entry_points().select(group="cleanframe.rules")
        except Exception:
            eps = []

    for ep in eps:
        try:
            plugin_obj = ep.load()
            if isinstance(plugin_obj, type):
                rule_instance = plugin_obj()
            else:
                rule_instance = plugin_obj

            if isinstance(rule_instance, RuleProtocol):
                discovered.append(rule_instance)
            else:
                logger.warning(
                    f"Discovered rule '{ep.name}' loaded from entry point does not implement RuleProtocol"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load external plugin '{ep.name}' from entry point: {e}"
            )

    return discovered


class RuleRegistry:
    """
    A registry system to manage, discover, and retrieve active rules.
    """

    def __init__(self) -> None:
        self._rules: list[RuleProtocol] = []

    def discover_and_register(self) -> None:
        """Run auto-discovery of plugins and register them."""
        self._rules = discover_plugins()

    @property
    def rules(self) -> list[RuleProtocol]:
        """Return the list of loaded rules."""
        return self._rules
