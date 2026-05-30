"""
Aviation-Grade Structured Logging
Ensures every log entry is tied to a specific request and flight ID for flawless auditing.
"""
# STAGE6_CLEANUP_REVIEW:
# Classification: UNUSED_LOGGING_HELPER_DELETE_CANDIDATE
# Plan lineage: PLAN1_OR_PLAN2_TOOLING
# Runtime status: No active imports/callers found in current caller check.
# Legacy signal: Defines structlog setup/context helpers, while current code calls structlog.get_logger directly.
# Replacement: Direct module-level structlog.get_logger usage or future centralized logging setup if intentionally adopted.
# Action rule: Do not call from new code. Candidate for deletion after final import/caller verification.

from __future__ import annotations
import contextvars
import logging
from contextlib import contextmanager
from typing import Any, Iterator
import structlog

# Context variables for precise tracing across async API calls
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="SYSTEM")
flight_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("flight_id", default="SYSTEM_INIT")


def inject_tracing_vars(logger: Any, log_method: str, event_dict: dict) -> dict:
    """Structlog processor that injects flight_id and request_id into every log."""
    try:
        event_dict["req_id"] = request_id_var.get()
        event_dict["flt_id"] = flight_id_var.get()
    except Exception:
        event_dict["req_id"] = "UNKNOWN"
        event_dict["flt_id"] = "UNKNOWN"
    return event_dict


def setup_logging(log_level: str = "INFO", environment: str = "development") -> None:
    """
    Configure standard library logging and structlog.
    Uses human-readable console logs in dev, and strict JSON in production.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")

    from structlog.stdlib import add_logger_name, add_log_level
    from structlog.processors import TimeStamper, JSONRenderer
    from structlog.dev import ConsoleRenderer

    # المعالجات المشتركة
    processors = [
        TimeStamper(fmt="iso"),
        add_log_level,
        add_logger_name,
        inject_tracing_vars,
    ]

    # اختيار التنسيق بناءً على بيئة التشغيل
    if environment == "production":
        processors.append(JSONRenderer())
    else:
        processors.append(ConsoleRenderer(colors=True))

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
    """Context manager to temporarily set the request id."""
    token = request_id_var.set(req_id)
    try:
        yield
    finally:
        request_id_var.reset(token)


@contextmanager
def set_flight_id(flt_id: str) -> Iterator[None]:
    """Context manager to temporarily set the flight id for tracking the drone's evaluation."""
    token = flight_id_var.set(flt_id)
    try:
        yield
    finally:
        flight_id_var.reset(token)