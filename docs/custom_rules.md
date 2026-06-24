# Custom Rules and Plugin Ecosystem

Cleanframe's architecture is designed for extreme extensibility without requiring you to fork or modify the core repository. We achieve this through Python's `importlib.metadata` entry points and our structural subtyping interface, `RuleProtocol`.

## The `RuleProtocol` Interface

Every rule in Cleanframe acts as an independent auditor and transformer. To build a custom rule, you do not need to inherit from a complex base class. You simply need to satisfy the `RuleProtocol` duck-typing interface:

```python
from typing import Any
from cleanframe.types import Decision

class MyCustomRule:
    @property
    def name(self) -> str:
        return "MyCustomRule"

    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        # Analyze the dataframe (Pandas or Polars via Narwhals)
        # Return a list of Decision objects
        return []

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        # Apply the logic based on approved Decisions
        return df
```

### The "Do No Harm" Principle

When writing your `detect` method, remember Cleanframe's core philosophy: **Do No Harm**.
Your rule should purely identify issues and yield `Decision` objects. If a user wants your rule to actually mutate data, they should pass an explicit configuration flag (e.g., `action="drop"` or `action="impute"`).

## Registering via Entry Points

To inject your custom rule into Cleanframe's ecosystem seamlessly, use Python Entry Points. In your project's `pyproject.toml`, add:

```toml
[project.entry-points."cleanframe.rules"]
my_custom_rule = "my_package.rules:MyCustomRule"
```

Once installed in the same environment, Cleanframe will automatically discover and load `MyCustomRule`, making it available in your pipelines.
