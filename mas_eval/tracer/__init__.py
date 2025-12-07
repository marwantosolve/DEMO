# Tracer module - OpenTelemetry wrapper for MAS tracing
from mas_eval.tracer.otel_tracer import TracerModule
from mas_eval.tracer.span_collector import SpanCollector

__all__ = ["TracerModule", "SpanCollector"]
