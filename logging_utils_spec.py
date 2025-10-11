from __future__ import annotations

import json
import logging
from datetime import datetime
from io import StringIO


from logging_utils import JsonFormatter, configure_logging, get_logger


def test_json_formatter_emits_iso_timestamp_and_basic_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="test message",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)
    parsed = json.loads(output)

    assert "ts" in parsed
    datetime.fromisoformat(parsed["ts"])  # Validates ISO format
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "test message"


def test_json_formatter_includes_extra_attributes():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="client",
        level=logging.INFO,
        pathname="client.py",
        lineno=100,
        msg="personalization_start",
        args=(),
        exc_info=None,
    )
    record.client_id = 3
    record.round = 5
    record.macro_f1 = 0.85

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["client_id"] == 3
    assert parsed["round"] == 5
    assert parsed["macro_f1"] == 0.85


def test_json_formatter_handles_non_serializable_extras():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test",
        args=(),
        exc_info=None,
    )

    class NonSerializable:
        pass

    record.weird_obj = NonSerializable()

    output = formatter.format(record)
    parsed = json.loads(output)

    assert "weird_obj" in parsed
    assert isinstance(parsed["weird_obj"], str)


def test_json_formatter_includes_exception_info():
    formatter = JsonFormatter()

    try:
        raise ValueError("test exception")
    except ValueError:
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=True,
        )
        record.exc_info = (ValueError, ValueError("test exception"), None)

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exc_info" in parsed
        assert "ValueError" in parsed["exc_info"]
        assert "test exception" in parsed["exc_info"]


def test_configure_logging_is_idempotent():
    root = logging.getLogger()

    configure_logging()
    count_after_first = len(root.handlers)

    configure_logging()
    count_after_second = len(root.handlers)

    assert count_after_second == count_after_first


def test_configure_logging_respects_level_parameter():
    configure_logging(level="DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG

    configure_logging(level="WARNING")
    assert root.level == logging.WARNING


def test_configure_logging_respects_env_var(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    # Clear existing handlers to test fresh config
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    configure_logging()
    assert root.level == logging.ERROR


def test_configure_logging_defaults_to_info_on_invalid_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    configure_logging()
    assert root.level == logging.INFO


def test_get_logger_returns_configured_logger():
    configure_logging()
    logger = get_logger("test_module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_end_to_end_structured_logging():
    # Clear existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Configure with custom stream
    stream = StringIO()
    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    logger = get_logger("integration_test")
    logger.info("test_event", extra={"client_id": 5, "metric": 0.95})

    output = stream.getvalue()
    parsed = json.loads(output.strip())

    assert parsed["logger"] == "integration_test"
    assert parsed["message"] == "test_event"
    assert parsed["client_id"] == 5
    assert parsed["metric"] == 0.95
