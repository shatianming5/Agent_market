from __future__ import annotations

import logging
import logging.config
from typing import Dict, Optional

from pythonjsonlogger import jsonlogger

from .config import Settings

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, OTLPSpanExporter
except Exception:  # pragma: no cover - optional dependency missing
    trace = None  # type: ignore[assignment]

_LOGGING_CONFIGURED = False
_TRACING_CONFIGURED = False
_logger = logging.getLogger("agent_market.telemetry")


class JsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter that injects standard log record attributes and ISO8601 timestamp.
    """

    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict) -> None:
        super().add_fields(log_record, record, message_dict)
        if "timestamp" not in log_record:
            log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record.setdefault("level", record.levelname)
        log_record.setdefault("logger", record.name)
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)


def configure_logging(settings: Settings) -> None:
    """
    Configure application logging. Defaults to JSON output at INFO level.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    handlers = ["console"]
    formatters: Dict[str, Dict] = {}
    if settings.log_json:
        formatters["json"] = {
            "format": "%(timestamp)s %(level)s %(logger)s %(message)s",
            "class": "server.telemetry.JsonFormatter",
        }
        selected_formatter = "json"
    else:
        formatters["plain"] = {
            "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            "class": "logging.Formatter",
        }
        selected_formatter = "plain"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": selected_formatter,
            }
        },
        "root": {
            "level": settings.log_level.upper(),
            "handlers": handlers,
        },
    }
    logging.config.dictConfig(logging_config)
    _LOGGING_CONFIGURED = True
    logging.getLogger("uvicorn.error").handlers = []
    logging.getLogger("uvicorn.access").handlers = []


def _parse_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    return headers or {}


def configure_tracing(app, settings: Settings) -> None:
    """
    Configure OpenTelemetry instrumentation when enabled and dependencies available.
    """
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED or not settings.otel_traces_enabled:
        return

    if trace is None:
        _logger.warning("OpenTelemetry dependencies not available; tracing disabled")
        return

    try:
        resource = Resource(attributes={SERVICE_NAME: settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint, headers=_parse_headers(settings.otel_headers))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        RequestsInstrumentor().instrument()
        LoggingInstrumentor().instrument(set_logging_format=True, log_hook=None)
        _TRACING_CONFIGURED = True
        _logger.info("OpenTelemetry tracing enabled via %s", settings.otel_endpoint)
    except Exception as exc:  # pragma: no cover - instrumentation failures
        _logger.warning("Failed to configure OpenTelemetry tracing: %s", exc)


__all__ = ["configure_logging", "configure_tracing"]
