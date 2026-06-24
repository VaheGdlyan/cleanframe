# Telemetry & Observability

Cleanframe believes that data transformations must never be black boxes. To guarantee 100% explainability, every anomaly detected and every modification made is logged as a structured event.

## The `TelemetryEvent`

At the core of the observability matrix is the `TelemetryEvent` dataclass. It captures:
- `event_type`: The nature of the event (e.g., anomaly detected, row dropped).
- `timestamp`: ISO 8601 timestamp.
- `run_id`: A unique UUID tying events to a specific pipeline run.
- `rule_name`: The rule responsible for the event.
- `column`: The specific feature involved.
- `payload`: A JSON-serializable dictionary containing deep context (e.g., fuzzy matching confidence scores, threshold values).

## Sinks

These events are routed to **Sinks**, which are configured in your pipeline. Cleanframe provides two zero-dependency sinks designed for enterprise logging:

### 1. `StandardLoggingSink`
Routes structured events directly into Python's built-in `logging` module. This is ideal when your application runs in a containerized environment (like Kubernetes) and logs are automatically aggregated via standard output streams.

### 2. `LocalJsonLinesSink`
Serializes `TelemetryEvent` objects into highly machine-readable `.jsonl` (JSON-Lines) files. This is perfect for local debugging or for shipping direct audit trails to enterprise tools like Datadog, Splunk, or ELK stacks.

```python
from cleanframe.pipeline import DataCleaner
from cleanframe.telemetry import LocalJsonLinesSink, StandardLoggingSink

cleaner = DataCleaner()

# Add standard terminal logging
cleaner.add_sink(StandardLoggingSink())

# Add JSON-Lines streaming for external aggregators
cleaner.add_sink(LocalJsonLinesSink("audit_trail.jsonl"))
```

By connecting these sinks, your data science operations instantly become fully observable to your wider platform engineering and DevOps teams.
