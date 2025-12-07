"""
OpenTelemetry Capture Module for Multi-Agent Systems

This module provides a generalized way to capture OpenTelemetry traces from 
any MAS (including Google ADK, LangChain, CrewAI) and save them to JSON.

Based on the beyond-black-box-benchmarking SDK patterns.

Usage:
    from otel_capture import OTelCapture
    
    capture = OTelCapture(output_file="traces.json")
    capture.start()
    
    # ... run your MAS agents ...
    
    capture.stop_and_save()
"""

import json
import os
import threading
import atexit
from datetime import datetime
from typing import List, Dict, Any, Optional, Sequence
from dataclasses import dataclass, field, asdict

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import Resource, SERVICE_NAME


@dataclass
class SerializedSpan:
    """JSON-serializable representation of an OTel span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    start_time: str  # ISO format
    end_time: str    # ISO format  
    duration_ms: float
    status_code: str
    status_description: Optional[str]
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    resource_attributes: Dict[str, Any] = field(default_factory=dict)


class InMemorySpanExporter(SpanExporter):
    """
    Custom SpanExporter that collects spans in-memory instead of sending 
    them to a remote endpoint. Thread-safe for concurrent span generation.
    """
    
    def __init__(self):
        self._spans: List[ReadableSpan] = []
        self._lock = threading.Lock()
        self._shutdown = False
    
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Collect spans in memory."""
        if self._shutdown:
            return SpanExportResult.SUCCESS
        
        with self._lock:
            self._spans.extend(spans)
        
        return SpanExportResult.SUCCESS
    
    def shutdown(self) -> None:
        """Mark exporter as shutdown."""
        self._shutdown = True
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op for in-memory exporter."""
        return True
    
    def get_spans(self) -> List[ReadableSpan]:
        """Get a copy of collected spans."""
        with self._lock:
            return list(self._spans)
    
    def clear(self) -> None:
        """Clear collected spans."""
        with self._lock:
            self._spans.clear()


class OTelCapture:
    """
    Main class for capturing OpenTelemetry traces from Multi-Agent Systems.
    
    Attributes:
        output_file: Path to save captured traces as JSON
        service_name: Name to identify the traced application
        
    Example:
        capture = OTelCapture(output_file="traces.json")
        capture.start()
        
        # Run your agents here
        from google import genai
        client = genai.Client(api_key="...")
        response = client.models.generate_content(...)
        
        capture.stop_and_save()
    """
    
    def __init__(
        self, 
        output_file: str = "otel_traces.json",
        service_name: str = "mas-evaluation",
        auto_save_on_exit: bool = True
    ):
        self.output_file = output_file
        self.service_name = service_name
        self.auto_save_on_exit = auto_save_on_exit
        
        self._exporter: Optional[InMemorySpanExporter] = None
        self._provider: Optional[TracerProvider] = None
        self._started = False
        self._original_provider = None
    
    def start(self) -> None:
        """
        Initialize OpenTelemetry tracing with in-memory collection.
        Call this BEFORE creating any agents or making LLM calls.
        """
        if self._started:
            return
        
        # Store original provider if any
        self._original_provider = trace.get_tracer_provider()
        
        # Create resource with service name
        resource = Resource.create({
            SERVICE_NAME: self.service_name,
            "capture.timestamp": datetime.utcnow().isoformat()
        })
        
        # Create provider and exporter
        self._exporter = InMemorySpanExporter()
        self._provider = TracerProvider(resource=resource)
        
        # Add simple span processor (immediate export, no batching)
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        
        # Set as global tracer provider
        trace.set_tracer_provider(self._provider)
        
        self._started = True
        
        # Register auto-save on exit
        if self.auto_save_on_exit:
            atexit.register(self._auto_save)
        
        print(f"[OTelCapture] Started tracing. Output will be saved to: {self.output_file}")
    
    def get_tracer(self, name: str = "mas-tracer") -> trace.Tracer:
        """Get a tracer for manual span creation."""
        if not self._started:
            self.start()
        return self._provider.get_tracer(name)
    
    def get_raw_spans(self) -> List[ReadableSpan]:
        """Get raw OTel ReadableSpan objects."""
        if not self._exporter:
            return []
        return self._exporter.get_spans()
    
    def get_serialized_spans(self) -> List[Dict[str, Any]]:
        """Get spans as JSON-serializable dictionaries."""
        raw_spans = self.get_raw_spans()
        return [self._serialize_span(span) for span in raw_spans]
    
    def _serialize_span(self, span: ReadableSpan) -> Dict[str, Any]:
        """Convert a ReadableSpan to a SerializedSpan dict."""
        
        # Extract span context
        ctx = span.get_span_context()
        parent_ctx = span.parent
        
        # Convert timestamps (nanoseconds to ISO format)
        start_time = datetime.utcfromtimestamp(span.start_time / 1e9).isoformat() + "Z"
        end_time = datetime.utcfromtimestamp(span.end_time / 1e9).isoformat() + "Z" if span.end_time else None
        duration_ms = (span.end_time - span.start_time) / 1e6 if span.end_time else 0
        
        # Convert attributes (handle non-serializable types)
        attributes = {}
        for key, value in (span.attributes or {}).items():
            attributes[key] = self._make_serializable(value)
        
        # Convert events
        events = []
        for event in (span.events or []):
            events.append({
                "name": event.name,
                "timestamp": datetime.utcfromtimestamp(event.timestamp / 1e9).isoformat() + "Z",
                "attributes": {k: self._make_serializable(v) for k, v in (event.attributes or {}).items()}
            })
        
        # Convert links
        links = []
        for link in (span.links or []):
            links.append({
                "trace_id": format(link.context.trace_id, '032x'),
                "span_id": format(link.context.span_id, '016x'),
                "attributes": {k: self._make_serializable(v) for k, v in (link.attributes or {}).items()}
            })
        
        # Resource attributes
        resource_attrs = {}
        if span.resource:
            for key, value in span.resource.attributes.items():
                resource_attrs[key] = self._make_serializable(value)
        
        serialized = SerializedSpan(
            trace_id=format(ctx.trace_id, '032x'),
            span_id=format(ctx.span_id, '016x'),
            parent_span_id=format(parent_ctx.span_id, '016x') if parent_ctx else None,
            name=span.name,
            kind=span.kind.name if span.kind else "INTERNAL",
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status_code=span.status.status_code.name if span.status else "UNSET",
            status_description=span.status.description if span.status else None,
            attributes=attributes,
            events=events,
            links=links,
            resource_attributes=resource_attrs
        )
        
        return asdict(serialized)
    
    def _make_serializable(self, value: Any) -> Any:
        """Convert a value to JSON-serializable format."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._make_serializable(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._make_serializable(v) for k, v in value.items()}
        else:
            return str(value)
    
    def stop_and_save(self) -> str:
        """
        Stop tracing and save all captured spans to JSON file.
        
        Returns:
            Path to the saved JSON file
        """
        if not self._started:
            print("[OTelCapture] Warning: Tracing was not started")
            return self.output_file
        
        # Force flush any pending spans
        if self._provider:
            self._provider.force_flush()
        
        # Serialize and save
        spans = self.get_serialized_spans()
        
        output = {
            "capture_metadata": {
                "service_name": self.service_name,
                "captured_at": datetime.utcnow().isoformat() + "Z",
                "total_spans": len(spans)
            },
            "spans": spans
        }
        
        # Ensure directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"[OTelCapture] Saved {len(spans)} spans to: {self.output_file}")
        
        return self.output_file
    
    def _auto_save(self) -> None:
        """Auto-save on exit if spans were collected."""
        if self._exporter and len(self._exporter.get_spans()) > 0:
            self.stop_and_save()
    
    def clear(self) -> None:
        """Clear all collected spans."""
        if self._exporter:
            self._exporter.clear()


# Convenience context manager
class OTelCaptureContext:
    """Context manager for easy capture."""
    
    def __init__(self, output_file: str = "otel_traces.json", **kwargs):
        self.capture = OTelCapture(output_file=output_file, **kwargs)
    
    def __enter__(self) -> OTelCapture:
        self.capture.start()
        return self.capture
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.capture.stop_and_save()
        return False


# Decorator for tracing agent functions
def trace_agent(name: str = None, capture: OTelCapture = None):
    """
    Decorator to trace agent functions.
    
    Usage:
        @trace_agent("ResearchAgent")
        def research_agent(query):
            # agent logic
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer("agent-tracer")
            span_name = name or func.__name__
            
            with tracer.start_as_current_span(
                span_name,
                attributes={
                    "agent.function": func.__name__,
                    "agent.module": func.__module__,
                    "traceloop.entity.name": span_name,
                }
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("agent.status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("agent.status", "error")
                    span.set_attribute("agent.error", str(e))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator


# Quick test
if __name__ == "__main__":
    # Simple test
    with OTelCaptureContext("test_traces.json") as capture:
        tracer = capture.get_tracer()
        
        with tracer.start_as_current_span("test-agent") as span:
            span.set_attribute("test.attribute", "hello")
            span.add_event("test-event", {"key": "value"})
        
        with tracer.start_as_current_span("test-llm-call") as span:
            span.set_attribute("llm.model", "gemini-2.0-flash")
            span.set_attribute("llm.request_type", "chat")
    
    print("Test complete! Check test_traces.json")
