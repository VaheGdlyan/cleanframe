import json
import logging

import polars as pl
import pytest

from cleanframe.pipeline import DataCleaner
from cleanframe.telemetry import (
    LocalJsonLinesSink,
    StandardLoggingSink,
    TelemetryEvent,
)


class MockSink:
    """Mock sink implementation that records emitted events."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


def test_telemetry_event_model():
    """Verify TelemetryEvent dataclass and serialization."""
    event = TelemetryEvent(
        event_type="test_event",
        timestamp="2026-06-22T22:00:00Z",
        run_id="abc-123",
        rule_name="MockRule",
        column="col1",
        payload={"message": "hello"},
    )
    assert event.event_type == "test_event"
    assert event.timestamp == "2026-06-22T22:00:00Z"
    assert event.run_id == "abc-123"
    assert event.rule_name == "MockRule"
    assert event.column == "col1"
    assert event.payload == {"message": "hello"}

    dct = event.to_dict()
    assert dct["event_type"] == "test_event"
    assert dct["payload"] == {"message": "hello"}


def test_standard_logging_sink(caplog: pytest.LogCaptureFixture):
    """Verify StandardLoggingSink routes events to standard logging."""
    sink = StandardLoggingSink(logger_name="test_observability_logger")
    event = TelemetryEvent(
        event_type="target_leakage",
        timestamp="2026-06-22T22:00:00Z",
        run_id="abc-123",
        rule_name="LeakageDetector",
        column="col1",
        payload={"warning": "some leakage warning"},
    )

    with caplog.at_level(logging.WARNING, logger="test_observability_logger"):
        sink.emit(event)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "target_leakage" in record.message
    assert "some leakage warning" in record.message


def test_local_json_lines_sink(tmp_path):
    """Verify LocalJsonLinesSink correctly format and append JSON to file."""
    output_file = tmp_path / "events.jsonl"
    sink = LocalJsonLinesSink(output_file)

    event1 = TelemetryEvent(
        event_type="event1",
        timestamp="2026-06-22T22:00:00Z",
        run_id="abc-123",
        rule_name=None,
        column=None,
        payload={"val": 1},
    )
    event2 = TelemetryEvent(
        event_type="event2",
        timestamp="2026-06-22T22:01:00Z",
        run_id="abc-123",
        rule_name=None,
        column=None,
        payload={"val": 2},
    )

    sink.emit(event1)
    sink.emit(event2)

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    parsed1 = json.loads(lines[0])
    parsed2 = json.loads(lines[1])

    assert parsed1["event_type"] == "event1"
    assert parsed1["payload"] == {"val": 1}
    assert parsed2["event_type"] == "event2"
    assert parsed2["payload"] == {"val": 2}


def test_datacleaner_pipeline_emits_events():
    """Verify DataCleaner emits target leakage, drift, mutation events to sinks with run_id."""
    df_fit = pl.DataFrame(
        {
            "id_col": [1, 2, 3, 4, 5],
            "target": [1, 0, 1, 0, 1],
            "target_leak": [1.0, 0.0, 1.0, 0.0, 1.0],  # Correlation = 1.0
            "nulls": [1.0, 2.0, None, 4.0, 5.0],
        }
    )

    mock_sink = MockSink()
    cleaner = DataCleaner(sinks=[mock_sink])

    # 1. Fit (Audit) run - detects leakage
    plan = cleaner.fit(df_fit, target_col="target")

    # Verify target_leakage events were emitted
    leakage_events = [e for e in mock_sink.events if e.event_type == "target_leakage"]
    assert len(leakage_events) > 0
    fit_run_id = leakage_events[0].run_id
    assert all(e.run_id == fit_run_id for e in leakage_events)
    # Check payload has warning
    assert "target_leak" in leakage_events[0].column

    # Clear events for the transform run
    mock_sink.events.clear()

    # 2. Transform run - with a plan containing leakage warning & NullHandler mutations
    df_trans = pl.DataFrame(
        {
            "id_col": [1, 2, 3, 4, 5],
            "target": [1, 0, 1, 0, 1],
            "target_leak": [1.0, 0.0, 1.0, 0.0, 1.0],
            # Drift simulation: increase nulls ratio by 20% (from 20% to 40% or 0% to 40%)
            "nulls": [1.0, None, None, 4.0, 5.0],
        }
    )

    _ = cleaner.transform(df_trans, plan)

    # Verify events emitted during transform
    trans_events = mock_sink.events
    assert len(trans_events) > 0

    trans_run_id = trans_events[0].run_id
    # Assert run IDs are unique per run
    assert trans_run_id != fit_run_id
    assert all(e.run_id == trans_run_id for e in trans_events)

    # Check for target_leakage re-emission in transform
    leak_trans = [e for e in trans_events if e.event_type == "target_leakage"]
    assert len(leak_trans) > 0

    # Check for drift alert events
    drift_events = [e for e in trans_events if e.event_type == "drift_alert"]
    assert len(drift_events) > 0
    assert any("nulls" in e.column for e in drift_events)

    # Check for rule mutation events
    mutation_events = [e for e in trans_events if e.event_type == "rule_mutation"]
    assert len(mutation_events) > 0
    assert any("NullHandler" in e.rule_name for e in mutation_events)

    # Verify AuditReport display is powered by these events
    report = cleaner.last_report
    assert report is not None
    assert len(report.drift_alerts) == len(drift_events)
    assert len(report.leakage_warnings) == len(leak_trans)
    assert "NullHandler" in report.mutations
