from __future__ import annotations
import contextvars
import logging
from contextlib import contextmanager
from typing import Any, Iterator
import structlog

# Context variable for request tracing across async boundaries
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="SYSTEM")


def inject_request_id(logger: Any, log_method: str, event_dict: dict) -> dict:
    """Structlog processor that injects the current request id into the event dict."""
    try:
        event_dict["request_id"] = request_id_var.get()
    except Exception:
        event_dict["request_id"] = "SYSTEM"
    return event_dict


def setup_logging(log_level: str = "INFO") -> None:
    """Configure standard library logging and structlog for JSON structured logs."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")

    # ✅ FIX: `add_logger_name` relocated to `structlog.stdlib` in modern versions
    from structlog.stdlib import add_logger_name

    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        add_logger_name,  # Uses the correct stdlib-integrated processor
        inject_request_id,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a configured structlog bound logger with the given name."""
    return structlog.get_logger(name)


@contextmanager
def set_request_id(req_id: str) -> Iterator[None]:
    """Context manager to temporarily set the request id for the current context."""
    token = request_id_var.set(req_id)
    try:
        yield
    finally:
        try:
            request_id_var.reset(token)
        except Exception:
            request_id_var.set("SYSTEM")