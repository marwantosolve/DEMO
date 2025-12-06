"""
Span Collector for MAS Evaluation Framework.

Collects OpenTelemetry spans and converts them to the standardized format.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import threading

from mas_eval.core.types import Span, StepType


class SpanCollector:
    """
    Custom OpenTelemetry span exporter that collects spans for analysis.
    
    Implements the SpanExporter interface to receive spans from the
    OpenTelemetry SDK and store them for later processing.
    """
    
    def __init__(self):
        self._raw_spans: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def export(self, spans) -> bool:
        """
        Export method called by OpenTelemetry SDK.
        
        Thread-safe implementation that minimizes lock contention
        by converting spans outside the critical section.
        
        Args:
            spans: ReadableSpan objects from OTel SDK
        
        Returns:
            True on success
        """
        # Convert outside lock to minimize contention
        converted = [self._convert_otel_span(span) for span in spans]
        
        with self._lock:
            self._raw_spans.extend(converted)
        return True
    
    def _convert_otel_span(self, otel_span) -> Dict[str, Any]:
        """Convert OTel span to dictionary format."""
        attributes = dict(otel_span.attributes) if otel_span.attributes else {}
        
        return {
            "span_id": format(otel_span.context.span_id, '016x'),
            "parent_span_id": format(otel_span.parent.span_id, '016x') if otel_span.parent else None,
            "trace_id": format(otel_span.context.trace_id, '032x'),
            "name": otel_span.name,
            "start_time": otel_span.start_time,
            "end_time": otel_span.end_time,
            "attributes": attributes
        }
    
    def get_raw_spans(self) -> List[Dict[str, Any]]:
        """Get raw span dictionaries."""
        with self._lock:
            return list(self._raw_spans)
    
    def get_spans(self) -> List[Span]:
        """
        Get spans converted to standardized Span objects.
        
        Returns:
            List of Span objects
        """
        with self._lock:
            spans = []
            for raw in self._raw_spans:
                try:
                    span = self._raw_to_span(raw)
                    spans.append(span)
                except Exception:
                    # Skip malformed spans
                    continue
            return spans
    
    def _raw_to_span(self, raw: Dict[str, Any]) -> Span:
        """Convert raw span dict to Span object."""
        attrs = raw.get("attributes", {})
        
        # Parse step type
        step_type_str = attrs.get("step.type", "action")
        try:
            step_type = StepType(step_type_str)
        except ValueError:
            step_type = StepType.ACTION
        
        # Parse timestamps
        start_time = raw.get("start_time")
        if isinstance(start_time, (int, float)):
            start_time = datetime.fromtimestamp(start_time / 1e9)  # nanoseconds
        elif not isinstance(start_time, datetime):
            start_time = datetime.now()
        
        end_time = raw.get("end_time")
        if isinstance(end_time, (int, float)):
            end_time = datetime.fromtimestamp(end_time / 1e9)
        elif not isinstance(end_time, datetime):
            end_time = None
        
        return Span(
            span_id=raw["span_id"],
            parent_span_id=raw.get("parent_span_id"),
            name=raw.get("name", "Unknown"),
            agent_name=attrs.get("agent.name", "Unknown"),
            step_type=step_type,
            content=attrs.get("step.content", ""),
            start_time=start_time,
            end_time=end_time,
            attributes=attrs
        )
    
    def clear(self) -> None:
        """Clear all collected spans."""
        with self._lock:
            self._raw_spans.clear()
    
    def __len__(self) -> int:
        """Return number of collected spans."""
        with self._lock:
            return len(self._raw_spans)
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans."""
        return True
