"""
Base Metrics Module for extensible metric plugins.

Provides the MetricsModule class that manages all metrics and allows
custom metric registration.
"""

from typing import Dict, List, Any, Optional
import networkx as nx

from mas_eval.core.base import BaseModule, BaseMetric
from mas_eval.core.types import Span


class MetricsModule(BaseModule):
    """
    Extensible metrics manager for MAS evaluation.
    
    Manages GEMMAS metrics and allows registration of custom metrics.
    
    Usage:
        metrics = MetricsModule()
        
        # Register custom metric
        metrics.register("my_metric", MyCustomMetric())
        
        # Evaluate
        results = metrics.evaluate(graph, spans)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="MetricsModule", config=config)
        self._metrics: Dict[str, BaseMetric] = {}
        self._load_default_metrics()
        self._initialized = True
    
    def _load_default_metrics(self) -> None:
        """Load default GEMMAS metrics."""
        from mas_eval.metrics.gemmas import IDSMetric, UPRMetric
        
        self._metrics["IDS"] = IDSMetric()
        self._metrics["UPR"] = UPRMetric()
    
    def register(self, name: str, metric: BaseMetric) -> None:
        """
        Register a custom metric.
        
        Args:
            name: Unique name for the metric
            metric: BaseMetric instance
        """
        if not isinstance(metric, BaseMetric):
            raise TypeError(f"Metric must inherit from BaseMetric, got {type(metric)}")
        self._metrics[name] = metric
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a metric.
        
        Args:
            name: Name of metric to remove
        
        Returns:
            True if removed, False if not found
        """
        if name in self._metrics:
            del self._metrics[name]
            return True
        return False
    
    def list_metrics(self) -> List[str]:
        """Get list of registered metric names."""
        return list(self._metrics.keys())
    
    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """Get a metric by name."""
        return self._metrics.get(name)
    
    def evaluate(
        self,
        graph: nx.DiGraph,
        spans: List[Span] = None,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate all or selected metrics.
        
        Args:
            graph: Causal Reasoning Graph
            spans: Optional list of spans
            metrics: Optional list of metric names to evaluate (None = all)
        
        Returns:
            Dictionary mapping metric name to value
        """
        results = {}
        
        target_metrics = metrics or self._metrics.keys()
        
        for name in target_metrics:
            if name not in self._metrics:
                results[name] = None
                continue
            
            try:
                metric = self._metrics[name]
                value = metric.calculate(graph, spans)
                results[name] = value
            except Exception as e:
                results[name] = None
                results[f"{name}_error"] = str(e)
        
        return results
    
    def evaluate_with_interpretation(
        self,
        graph: nx.DiGraph,
        spans: List[Span] = None
    ) -> Dict[str, Any]:
        """
        Evaluate metrics with interpretations.
        
        Returns:
            Dictionary with values and interpretations
        """
        results = {}
        
        for name, metric in self._metrics.items():
            try:
                value = metric.calculate(graph, spans)
                results[name] = {
                    "value": value,
                    "interpretation": metric.get_interpretation(value),
                    "description": metric.description
                }
            except Exception as e:
                results[name] = {
                    "value": None,
                    "error": str(e)
                }
        
        return results
    
    def process(self, data: Any) -> Dict[str, float]:
        """BaseModule interface."""
        if isinstance(data, tuple) and len(data) == 2:
            graph, spans = data
            return self.evaluate(graph, spans)
        elif isinstance(data, nx.DiGraph):
            return self.evaluate(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def reset(self) -> None:
        """Reset to default metrics only."""
        self._metrics.clear()
        self._load_default_metrics()


# ===== Example Custom Metric =====

class AgentActivityMetric(BaseMetric):
    """
    Example custom metric: measures balance of agent activity.
    
    High value = balanced activity across agents
    Low value = one agent dominates
    """
    
    def __init__(self):
        super().__init__(
            name="AgentActivity",
            description="Measures balance of activity across agents"
        )
    
    def calculate(self, graph: nx.DiGraph, spans: List[Span] = None) -> float:
        """Calculate agent activity balance."""
        import numpy as np
        
        # Count nodes per agent
        agent_counts = {}
        for _, data in graph.nodes(data=True):
            agent = data.get('agent', 'Unknown')
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        if len(agent_counts) <= 1:
            return 1.0  # Single agent, perfectly balanced
        
        counts = list(agent_counts.values())
        
        # Calculate normalized entropy as balance measure
        total = sum(counts)
        probs = [c / total for c in counts]
        
        # Shannon entropy normalized by max entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        max_entropy = np.log2(len(counts))
        
        balance = entropy / max_entropy if max_entropy > 0 else 1.0
        return round(balance, 4)
    
    def get_interpretation(self, value: float) -> str:
        if value >= 0.8:
            return "Well-balanced agent activity"
        elif value >= 0.5:
            return "Moderate agent activity balance"
        else:
            return "Imbalanced - one agent dominates"
