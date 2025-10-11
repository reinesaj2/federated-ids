import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict


# Standard LogRecord attributes that should not be included as extras
STANDARD_FIELDS: frozenset[str] = frozenset(
    [
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    ]
)


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs.

    Emits records as JSON with standard fields and any extras passed via `extra`.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extra attributes (non-standard logging fields)
        for key, value in record.__dict__.items():
            if key not in STANDARD_FIELDS and key not in payload:
                try:
                    json.dumps(value)  # ensure JSON-serializable
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logger with JSON formatter if not already configured.

    Idempotent: safe to call multiple times.
    """
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return

    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)

    # Default level from env, fallback to INFO
    lvl = level or os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        root.setLevel(getattr(logging, str(lvl)))
    except (AttributeError, TypeError, ValueError):
        root.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
