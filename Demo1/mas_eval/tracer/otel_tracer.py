"""
OpenTelemetry-based tracer module for MAS evaluation.

Provides plug-and-play tracing that works with any Multi-Agent System.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from contextlib import contextmanager
import uuid

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

from mas_eval.core.base import BaseModule
from mas_eval.core.types import Span, StepType, TraceData
from mas_eval.tracer.span_collector import SpanCollector


class TracerModule(BaseModule):
    """
    Plug-and-play OpenTelemetry tracer for Multi-Agent Systems.
    
    Provides a unified interface for tracing agent interactions across
    different MAS frameworks (Google ADK, LangGraph, AutoGen, etc.).
    
    Usage:
        # Basic usage
        tracer = TracerModule(service_name="my-mas")
        
        with tracer.trace_task("Research Report"):
            # Your MAS execution here
            tracer.trace_step("Agent1", "thought", "Analyzing data...")
            tracer.trace_step("Agent1", "action", "Calling API...")
        
        spans = tracer.get_spans()
    """
    
    def __init__(
        self,
        service_name: str = "mas-evaluation",
        adapter: Optional[str] = None,
        console_output: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the TracerModule.
        
        Args:
            service_name: Name of the MAS service for OTel resource
            adapter: Framework adapter ("adk", "langgraph", "autogen", None for generic)
            console_output: Whether to also output spans to console
            config: Additional configuration options
        """
        super().__init__(name="TracerModule", config=config)
        
        self.service_name = service_name
        self.adapter_name = adapter
        self.console_output = console_output
        
        # Initialize span collector
        self._collector = SpanCollector()
        
        # Setup OpenTelemetry
        self._setup_otel()
        
        # Current trace context
        self._current_trace_id: Optional[str] = None
        self._root_context = None
        
        self._initialized = True
    
    def _setup_otel(self) -> None:
        """Setup OpenTelemetry tracer provider."""
        resource = Resource(attributes={
            "service.name": self.service_name,
            "mas.framework": self.adapter_name or "generic"
        })
        
        self._provider = TracerProvider(resource=resource)
        
        # Add span collector exporter
        self._provider.add_span_processor(
            SimpleSpanProcessor(self._collector)
        )
        
        # Optionally add console exporter
        if self.console_output:
            self._provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )
        
        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer("mas_eval.tracer")
    
    @contextmanager
    def trace_task(self, task_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing a complete MAS task.
        
        Args:
            task_name: Name of the task being traced
            metadata: Optional metadata to attach to the trace
        
        Usage:
            with tracer.trace_task("Generate Report"):
                # MAS execution here
                pass
        """
        self._current_trace_id = str(uuid.uuid4())
        
        with self._tracer.start_as_current_span(
            name=f"Task: {task_name}",
            attributes={
                "task.name": task_name,
                "trace.id": self._current_trace_id,
                **(metadata or {})
            }
        ) as root_span:
            self._root_context = trace.get_current_span().get_span_context()
            try:
                yield self
            finally:
                self._root_context = None
    
    def trace_step(
        self,
        agent_name: str,
        step_type: str,
        content: str,
        parent_context: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Trace a single agent step.
        
        Args:
            agent_name: Name of the agent performing the step
            step_type: Type of step ("thought", "action", "observation", "output")
            content: Text content of the step
            parent_context: Optional parent span context for hierarchy
            attributes: Additional attributes to attach
        
        Returns:
            SpanContext for chaining child spans
        """
        with self._tracer.start_as_current_span(
            name=f"{agent_name}.{step_type}",
            context=parent_context,
            attributes={
                "agent.name": agent_name,
                "step.type": step_type,
                "step.content": content[:1000] if content else "",  # Truncate long content
                "timestamp": datetime.now().isoformat(),
                **(attributes or {})
            }
        ) as span:
            return span.get_span_context()
    
    def trace_agent_start(self, agent_name: str, instruction: str = "") -> Any:
        """Trace when an agent starts execution."""
        return self.trace_step(
            agent_name=agent_name,
            step_type="start",
            content=f"Agent started: {instruction[:200]}"
        )
    
    def trace_agent_end(self, agent_name: str, result: str = "") -> Any:
        """Trace when an agent completes execution."""
        return self.trace_step(
            agent_name=agent_name,
            step_type="end",
            content=f"Agent completed: {result[:200]}"
        )
    
    def trace_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Trace a tool/function call by an agent."""
        return self.trace_step(
            agent_name=agent_name,
            step_type="tool_call",
            content=f"Called {tool_name}",
            attributes={"tool.name": tool_name, "tool.args": str(arguments)[:500]}
        )
    
    def trace_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str
    ) -> Any:
        """Trace a message between agents."""
        return self.trace_step(
            agent_name=from_agent,
            step_type="message",
            content=message,
            attributes={"message.to": to_agent}
        )
    
    def get_spans(self) -> List[Span]:
        """
        Get all collected spans as standardized Span objects.
        
        Returns:
            List of Span objects
        """
        return self._collector.get_spans()
    
    def get_raw_spans(self) -> List[Dict[str, Any]]:
        """
        Get raw span data as dictionaries.
        
        Returns:
            List of span dictionaries
        """
        return self._collector.get_raw_spans()
    
    def get_trace_data(self) -> TraceData:
        """
        Get complete trace data.
        
        Returns:
            TraceData object with all spans
        """
        return TraceData(
            trace_id=self._current_trace_id or str(uuid.uuid4()),
            spans=self.get_spans(),
            metadata={"service_name": self.service_name}
        )
    
    def clear(self) -> None:
        """Clear all collected spans."""
        self._collector.clear()
        self._current_trace_id = None
    
    def process(self, data: Any) -> List[Span]:
        """Process method for BaseModule interface."""
        return self.get_spans()
    
    def reset(self) -> None:
        """Reset tracer for new evaluation."""
        self.clear()
    
    # ===== Decorator for automatic tracing =====
    
    def trace(self, agent_name: str, step_type: str = "action"):
        """
        Decorator for automatically tracing function calls.
        
        Usage:
            @tracer.trace("MyAgent", "action")
            def my_function():
                pass
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                self.trace_step(
                    agent_name=agent_name,
                    step_type=step_type,
                    content=f"Executing {func.__name__}"
                )
                result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator


# ===== Factory function for easy creation =====

def create_tracer(
    service_name: str = "mas-evaluation",
    framework: Optional[str] = None,
    **kwargs
) -> TracerModule:
    """
    Factory function to create a configured TracerModule.
    
    Args:
        service_name: Name of the MAS service
        framework: Framework name for adapter selection
        **kwargs: Additional configuration
    
    Returns:
        Configured TracerModule instance
    """
    return TracerModule(
        service_name=service_name,
        adapter=framework,
        **kwargs
    )


# ===== ADK Integration =====

class ADKOTelBridge:
    """
    Bridge to connect ADKTracingCallback with OpenTelemetry TracerModule.
    
    This enables the ADKTracingCallback spans to be exported via OTel
    to external observability platforms.
    
    Usage:
        from mas_eval.adapters import ADKTracingCallback
        from mas_eval.tracer import TracerModule, ADKOTelBridge
        
        tracer = TracerModule(service_name="my-mas")
        callback = ADKTracingCallback()
        bridge = ADKOTelBridge(tracer, callback)
        
        # Run your agents...
        
        # Get unified spans
        spans = bridge.get_spans()
    """
    
    def __init__(self, tracer: TracerModule, adk_callback: Any):
        """
        Initialize the bridge.
        
        Args:
            tracer: TracerModule instance
            adk_callback: ADKTracingCallback instance
        """
        self.tracer = tracer
        self.adk_callback = adk_callback
    
    def sync_to_otel(self) -> int:
        """
        Synchronize ADK callback spans to OpenTelemetry.
        
        Creates OTel spans from the ADK callback's captured spans.
        
        Returns:
            Number of spans synchronized
        """
        adk_spans = self.adk_callback.get_spans()
        synced = 0
        
        for span in adk_spans:
            self.tracer.trace_step(
                agent_name=span.agent_name,
                step_type=span.step_type.value,
                content=span.content,
                attributes={
                    "original_span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    **span.attributes
                }
            )
            synced += 1
        
        return synced
    
    def get_spans(self) -> List[Span]:
        """Get all spans from both ADK callback and OTel tracer."""
        adk_spans = self.adk_callback.get_spans()
        otel_spans = self.tracer.get_spans()
        
        # Merge, preferring ADK spans (they have better structure)
        seen_ids = {s.span_id for s in adk_spans}
        merged = list(adk_spans)
        
        for span in otel_spans:
            if span.span_id not in seen_ids:
                merged.append(span)
        
        return merged
    
    def get_trace_data(self) -> TraceData:
        """Get combined trace data."""
        return TraceData(
            trace_id=self.tracer._current_trace_id or str(uuid.uuid4()),
            spans=self.get_spans(),
            metadata={
                "service_name": self.tracer.service_name,
                "bridge_used": True
            }
        )


# ===== Trace Context Propagation =====

class TraceContextPropagator:
    """
    Utility for propagating trace context across agent boundaries.
    
    Useful for distributed MAS where agents may run in different processes.
    Uses W3C Trace Context format.
    """
    
    @staticmethod
    def inject(context: Optional[Any] = None) -> Dict[str, str]:
        """
        Inject current trace context into headers.
        
        Args:
            context: Optional specific context (uses current if None)
            
        Returns:
            Dictionary with traceparent and tracestate headers
        """
        from opentelemetry.propagate import inject
        from opentelemetry import trace
        
        headers = {}
        
        # Get current span context if no context provided
        if context is None:
            current_span = trace.get_current_span()
            if current_span:
                context = current_span.get_span_context()
        
        inject(headers)
        return headers
    
    @staticmethod
    def extract(headers: Dict[str, str]) -> Optional[Any]:
        """
        Extract trace context from headers.
        
        Args:
            headers: Dictionary containing trace headers
            
        Returns:
            Extracted context or None
        """
        from opentelemetry.propagate import extract
        
        return extract(headers)
    
    @staticmethod
    def create_child_context(parent_headers: Dict[str, str]) -> Any:
        """
        Create a new context as a child of the given parent.
        
        Args:
            parent_headers: Headers from parent trace
            
        Returns:
            New child context
        """
        from opentelemetry import trace
        from opentelemetry.trace import set_span_in_context
        
        parent_ctx = TraceContextPropagator.extract(parent_headers)
        
        tracer = trace.get_tracer("mas_eval.propagation")
        with tracer.start_as_current_span("child_span", context=parent_ctx) as span:
            return span.get_span_context()

