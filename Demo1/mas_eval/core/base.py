"""
Base classes for MAS Evaluation Framework modules.

These abstract classes define the interface that all modules must implement,
enabling the plug-and-play architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import networkx as nx

from mas_eval.core.types import Span, TraceData, EvaluationResult


class BaseModule(ABC):
    """
    Abstract base class for all framework modules.
    
    All modules (Tracer, CRG, Metrics, MAST) inherit from this
    to ensure consistent interface.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the module. Called before first use."""
        self._initialized = True
    
    def is_initialized(self) -> bool:
        """Check if module is initialized."""
        return self._initialized
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process input data and return result.
        
        Each module implements this differently based on its purpose.
        """
        pass
    
    def reset(self) -> None:
        """Reset module state for new evaluation."""
        pass


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    Extend this to create custom metrics that can be plugged into
    the MetricsModule.
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def calculate(self, graph: nx.DiGraph, spans: List[Span]) -> float:
        """
        Calculate the metric value.
        
        Args:
            graph: The Causal Reasoning Graph
            spans: List of spans from the trace
        
        Returns:
            Metric value (typically 0-1 for normalized metrics)
        """
        pass
    
    def get_interpretation(self, value: float) -> str:
        """
        Get human-readable interpretation of the metric value.
        
        Override this to provide custom interpretations.
        """
        if value >= 0.8:
            return "Excellent"
        elif value >= 0.6:
            return "Good"
        elif value >= 0.4:
            return "Fair"
        else:
            return "Poor"


class BaseAdapter(ABC):
    """
    Abstract base class for MAS framework adapters.
    
    Adapters translate framework-specific trace formats into
    the standardized Span format used by mas_eval.
    """
    
    def __init__(self, framework_name: str):
        self.framework_name = framework_name
    
    @abstractmethod
    def convert_to_spans(self, raw_data: Any) -> List[Span]:
        """
        Convert framework-specific trace data to standardized Spans.
        
        Args:
            raw_data: Raw trace data from the MAS framework
        
        Returns:
            List of Span objects
        """
        pass
    
    @abstractmethod
    def inject_tracer(self, agent: Any) -> Any:
        """
        Inject tracing capability into an agent.
        
        Args:
            agent: The agent object to instrument
        
        Returns:
            Instrumented agent
        """
        pass
    
    def get_framework_name(self) -> str:
        """Get the name of the framework this adapter supports."""
        return self.framework_name


class BaseClassifier(ABC):
    """
    Abstract base class for failure classifiers.
    
    Extend this to create different classification strategies
    (zero-shot, few-shot, fine-tuned).
    """
    
    def __init__(self, name: str, model_name: str):
        self.name = name
        self.model_name = model_name
    
    @abstractmethod
    def classify(self, spans: List[Span]) -> List[Dict[str, Any]]:
        """
        Classify failure modes in the trace.
        
        Args:
            spans: List of spans to analyze
        
        Returns:
            List of detected failure modes with confidence scores
        """
        pass
    
    @abstractmethod
    def get_supported_modes(self) -> List[str]:
        """Get list of failure mode codes this classifier can detect."""
        pass
