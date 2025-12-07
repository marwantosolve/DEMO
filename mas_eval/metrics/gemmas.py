"""
GEMMAS Metrics Implementation.

Implements Information Diversity Score (IDS) and Unnecessary Path Ratio (UPR)
based on the GEMMAS framework for evaluating Multi-Agent Systems.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import networkx as nx

from mas_eval.core.base import BaseMetric
from mas_eval.core.types import Span


class IDSMetric(BaseMetric):
    """
    Information Diversity Score (IDS) Metric.
    
    Measures semantic diversity of agent outputs using embeddings.
    High IDS = High Diversity (Good - agents contribute unique info)
    Low IDS = Repetitive (Bad - agents parrot each other)
    
    Formula: IDS = 1 - Average(Pairwise Similarity)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(
            name="IDS",
            description="Information Diversity Score - measures semantic diversity of agent outputs"
        )
        self.model_name = model_name
        self._encoder = None
    
    def _load_encoder(self):
        """Lazy load the sentence transformer."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for IDS. "
                    "Install with: pip install sentence-transformers"
                )
    
    def calculate(self, graph: nx.DiGraph, spans: List[Span] = None) -> float:
        """
        Calculate Information Diversity Score.
        
        Args:
            graph: Causal Reasoning Graph
            spans: Optional list of spans (not used if graph has content)
        
        Returns:
            IDS score between 0.0 and 1.0
        """
        # Extract content from graph nodes
        contents = []
        for _, data in graph.nodes(data=True):
            content = data.get('content', '')
            if content and len(content.strip()) > 10:
                contents.append(content)
        
        if len(contents) < 2:
            return 1.0  # Perfect diversity if only one content
        
        self._load_encoder()
        
        # Get semantic embeddings
        embeddings = self._encoder.encode(contents)
        
        # Calculate pairwise similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        
        # IDS = 1 - Average Similarity (excluding diagonal)
        n = len(sim_matrix)
        if n <= 1:
            return 1.0
        
        # Sum all similarities except diagonal, then divide by number of pairs
        total_sim = (np.sum(sim_matrix) - n) / (n * (n - 1))
        
        ids_score = 1.0 - total_sim
        return round(max(0.0, min(1.0, ids_score)), 4)
    
    def get_interpretation(self, value: float) -> str:
        """Interpret IDS value."""
        if value >= 0.7:
            return "Excellent diversity - agents contribute unique information"
        elif value >= 0.5:
            return "Good diversity - reasonable information variety"
        elif value >= 0.3:
            return "Fair diversity - some repetition detected"
        else:
            return "Poor diversity - high repetition, agents may be parroting"


class UPRMetric(BaseMetric):
    """
    Unnecessary Path Ratio (UPR) Metric.
    
    Measures inefficiency in the reasoning process.
    Low UPR = Efficient (Good - minimal redundant steps)
    High UPR = Inefficient (Bad - many unnecessary detours)
    
    Formula: UPR = 1 - (Necessary Paths / Total Paths)
    """
    
    def __init__(self, efficiency_threshold: float = 1.2):
        super().__init__(
            name="UPR",
            description="Unnecessary Path Ratio - measures reasoning efficiency"
        )
        self.efficiency_threshold = efficiency_threshold
    
    def calculate(self, graph: nx.DiGraph, spans: List[Span] = None) -> float:
        """
        Calculate Unnecessary Path Ratio.
        
        Args:
            graph: Causal Reasoning Graph
            spans: Not used
        
        Returns:
            UPR score between 0.0 and 1.0
        """
        if graph.number_of_nodes() < 2:
            return 0.0  # No unnecessary paths if minimal graph
        
        # Find root and leaf nodes
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        
        if not roots or not leaves:
            return 0.0
        
        source = roots[0]
        target = leaves[-1]  # Use last leaf as target
        
        try:
            all_paths = list(nx.all_simple_paths(graph, source=source, target=target))
        except nx.NetworkXNoPath:
            return 0.0
        
        total_paths = len(all_paths)
        if total_paths == 0:
            return 0.0
        
        # Find shortest path length
        try:
            shortest_path_len = nx.shortest_path_length(graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return 0.0
        
        # "Necessary" paths are within threshold of shortest path length
        tau = self.efficiency_threshold
        necessary_paths = [
            p for p in all_paths
            if len(p) <= shortest_path_len * tau
        ]
        
        upr = 1.0 - (len(necessary_paths) / total_paths)
        return round(max(0.0, min(1.0, upr)), 4)
    
    def get_interpretation(self, value: float) -> str:
        """Interpret UPR value."""
        if value <= 0.2:
            return "Excellent efficiency - minimal unnecessary paths"
        elif value <= 0.4:
            return "Good efficiency - mostly direct reasoning"
        elif value <= 0.6:
            return "Moderate efficiency - some redundant steps"
        else:
            return "Poor efficiency - many unnecessary detours"


class GEMMAS_Evaluator:
    """
    Complete GEMMAS framework evaluator.
    
    Combines IDS and UPR metrics for comprehensive MAS quality assessment.
    
    Usage:
        evaluator = GEMMAS_Evaluator()
        results = evaluator.evaluate(graph)
        print(evaluator.summary(results))
    """
    
    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize GEMMAS evaluator.
        
        Args:
            encoder_model: Sentence transformer model for IDS
        """
        self.ids_metric = IDSMetric(model_name=encoder_model)
        self.upr_metric = UPRMetric()
        self._custom_metrics: Dict[str, BaseMetric] = {}
    
    def register_metric(self, metric: BaseMetric) -> None:
        """
        Register a custom metric.
        
        Args:
            metric: BaseMetric instance to register
        """
        self._custom_metrics[metric.name] = metric
    
    def evaluate(self, graph: nx.DiGraph, spans: List[Span] = None) -> Dict[str, Any]:
        """
        Perform complete GEMMAS evaluation.
        
        Args:
            graph: Causal Reasoning Graph
            spans: Optional list of spans
        
        Returns:
            Dictionary with all metrics and interpretations
        """
        results = {}
        
        # Core GEMMAS metrics
        ids = self.ids_metric.calculate(graph, spans)
        upr = self.upr_metric.calculate(graph, spans)
        
        results["IDS"] = ids
        results["IDS_interpretation"] = self.ids_metric.get_interpretation(ids)
        results["UPR"] = upr
        results["UPR_interpretation"] = self.upr_metric.get_interpretation(upr)
        
        # Custom metrics
        for name, metric in self._custom_metrics.items():
            try:
                value = metric.calculate(graph, spans)
                results[name] = value
                results[f"{name}_interpretation"] = metric.get_interpretation(value)
            except Exception as e:
                results[name] = None
                results[f"{name}_error"] = str(e)
        
        # Overall quality score
        results["overall_quality"] = self._calculate_overall(ids, upr)
        
        return results
    
    def compute(self, graph: nx.DiGraph, spans: List[Span] = None) -> Dict[str, Any]:
        """
        Alias for evaluate() for backward compatibility.
        
        Args:
            graph: Causal Reasoning Graph
            spans: Optional list of spans
            
        Returns:
            Dictionary with all metrics and interpretations
        """
        return self.evaluate(graph, spans)
    
    def _calculate_overall(self, ids: float, upr: float) -> str:
        """Calculate overall quality assessment."""
        if ids > 0.6 and upr < 0.3:
            return "High"
        elif ids > 0.4 and upr < 0.5:
            return "Medium"
        else:
            return "Low"
    
    def summary(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable summary.
        
        Args:
            results: Results from evaluate()
        
        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 50,
            "GEMMAS EVALUATION RESULTS",
            "=" * 50,
            "",
            f"Information Diversity Score (IDS): {results.get('IDS', 'N/A')}",
            f"  → {results.get('IDS_interpretation', '')}",
            "",
            f"Unnecessary Path Ratio (UPR): {results.get('UPR', 'N/A')}",
            f"  → {results.get('UPR_interpretation', '')}",
            "",
            f"Overall Quality: {results.get('overall_quality', 'N/A')}",
            "=" * 50
        ]
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export evaluator configuration."""
        return {
            "ids_model": self.ids_metric.model_name,
            "upr_threshold": self.upr_metric.efficiency_threshold,
            "custom_metrics": list(self._custom_metrics.keys())
        }


# Backward compatibility alias
GEMMASMetrics = GEMMAS_Evaluator
