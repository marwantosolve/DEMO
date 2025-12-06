"""
MAS Evaluator - Unified API for Multi-Agent System Evaluation.

Combines all evaluation modules into a single, easy-to-use interface.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os

from mas_eval.core.types import Span, TraceData, EvaluationResult, FailureMode
from mas_eval.tracer import TracerModule
from mas_eval.graph import CRGModule, GraphVisualizer
from mas_eval.metrics import MetricsModule, GEMMAS_Evaluator, ThoughtRelevanceMetric
from mas_eval.mast import MASTClassifier, MASTTaxonomy
from mas_eval.suggestions import MASAdvisor


class MASEvaluator:
    """
    Unified Multi-Agent System Evaluator.
    
    Combines tracing, graph analysis, metrics, and MAST classification
    into a single plug-and-play interface.
    
    Usage:
        # Basic usage
        evaluator = MASEvaluator()
        result = evaluator.evaluate(spans)
        print(result.summary())
        
        # Advanced configuration
        evaluator = MASEvaluator(
            enable_mast=True,
            mast_model="gemini-2.0-flash",
            custom_metrics={"balance": AgentActivityMetric()}
        )
    """
    
    def __init__(
        self,
        enable_tracing: bool = True,
        enable_crg: bool = True,
        enable_gemmas: bool = True,
        enable_mast: bool = True,
        mast_model: str = "gemini-2.0-flash",
        mast_confidence: float = 0.5,
        custom_metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the MAS Evaluator.
        
        Args:
            enable_tracing: Enable OpenTelemetry tracing module
            enable_crg: Enable Causal Reasoning Graph building
            enable_gemmas: Enable GEMMAS metrics (IDS/UPR)
            enable_mast: Enable MAST failure classification
            mast_model: Model to use for MAST classification
            mast_confidence: Minimum confidence for MAST failures
            custom_metrics: Dictionary of custom metrics to register
            config: Additional configuration options
        """
        self.config = config or {}
        
        # Initialize modules based on configuration
        self._tracer: Optional[TracerModule] = None
        self._crg: Optional[CRGModule] = None
        self._metrics: Optional[MetricsModule] = None
        self._mast: Optional[MASTClassifier] = None
        
        if enable_tracing:
            self._tracer = TracerModule(
                service_name=self.config.get("service_name", "mas-evaluation")
            )
        
        if enable_crg:
            self._crg = CRGModule()
        
        if enable_gemmas:
            self._metrics = MetricsModule()
            # Register custom metrics
            if custom_metrics:
                for name, metric in custom_metrics.items():
                    self._metrics.register(name, metric)
        
        if enable_mast:
            self._mast = MASTClassifier(
                model=mast_model,
                confidence_threshold=mast_confidence
            )
        
        # Initialize TRS and Advisor
        self._trs = ThoughtRelevanceMetric()
        self._advisor = MASAdvisor()
    
    def evaluate(
        self,
        data: Any,
        include_visualization: bool = False,
        output_dir: Optional[str] = None
    ) -> EvaluationResult:
        """
        Perform complete evaluation on trace data.
        
        Args:
            data: Spans, TraceData, or raw span dictionaries
            include_visualization: Generate CRG visualization
            output_dir: Directory to save outputs (optional)
        
        Returns:
            EvaluationResult with metrics, failures, and graph stats
        """
        # Normalize input
        spans = self._normalize_input(data)
        trace_id = self._get_trace_id(data)
        
        # Build CRG
        graph = None
        graph_stats = {}
        if self._crg:
            graph = self._crg.build(spans)
            graph_stats = self._crg.get_statistics()
            
            if include_visualization and output_dir:
                viz = GraphVisualizer(graph)
                viz_path = os.path.join(output_dir, f"crg_{trace_id}.png")
                viz.plot(output_path=viz_path, show_labels=True)
        
        # Calculate metrics
        metrics = {}
        trs_result = None
        if self._metrics and graph:
            metrics = self._metrics.evaluate(graph, spans)
        
        # Calculate TRS
        if self._trs:
            trs_result = self._trs.calculate(graph, spans)
            metrics["TRS"] = trs_result.get("overall_score", 0.0)
        
        # MAST classification
        failures = []
        if self._mast:
            result = self._mast.classify(spans)
            failures = result.failure_modes
        
        # Build result
        eval_result = EvaluationResult(
            trace_id=trace_id,
            metrics=metrics,
            failures=failures,
            graph_stats=graph_stats,
            timestamp=datetime.now()
        )
        
        # Store TRS result for report generation
        eval_result._trs_result = trs_result
        
        # Generate suggestions
        if self._advisor:
            suggestions = self._advisor.generate_suggestions(eval_result, trs_result)
            eval_result._suggestions = suggestions
        
        # Save outputs if directory specified
        if output_dir:
            self._save_outputs(eval_result, output_dir)
        
        return eval_result
    
    def _normalize_input(self, data: Any) -> List[Span]:
        """Normalize input to list of Span objects."""
        if isinstance(data, TraceData):
            return data.spans
        elif isinstance(data, list):
            if len(data) == 0:
                return []
            if isinstance(data[0], Span):
                return data
            elif isinstance(data[0], dict):
                return [Span.from_dict(d) for d in data]
        raise ValueError(f"Unsupported input type: {type(data)}")
    
    def _get_trace_id(self, data: Any) -> str:
        """Extract or generate trace ID."""
        if isinstance(data, TraceData):
            return data.trace_id
        return f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _save_outputs(self, result: EvaluationResult, output_dir: str) -> None:
        """Save evaluation outputs to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        json_path = os.path.join(output_dir, f"report_{result.trace_id}.json")
        with open(json_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save text summary
        txt_path = os.path.join(output_dir, f"summary_{result.trace_id}.txt")
        with open(txt_path, 'w') as f:
            f.write(result.summary())
    
    # ===== Convenience Methods =====
    
    def get_tracer(self) -> Optional[TracerModule]:
        """Get the tracer module for manual tracing."""
        return self._tracer
    
    def get_crg(self) -> Optional[CRGModule]:
        """Get the CRG module for graph analysis."""
        return self._crg
    
    def get_metrics(self) -> Optional[MetricsModule]:
        """Get the metrics module."""
        return self._metrics
    
    def get_mast(self) -> Optional[MASTClassifier]:
        """Get the MAST classifier."""
        return self._mast
    
    def get_taxonomy(self) -> MASTTaxonomy:
        """Get the MAST taxonomy."""
        return MASTTaxonomy()
    
    def quick_summary(self, result: EvaluationResult) -> str:
        """Generate a quick one-line summary."""
        score = result.get_overall_score()
        failures = len(result.failures)
        return f"Score: {score:.2%} | IDS: {result.metrics.get('IDS', 'N/A')} | UPR: {result.metrics.get('UPR', 'N/A')} | Failures: {failures}"
    
    def to_html(self, result: EvaluationResult, output_path: str) -> None:
        """
        Generate HTML report.
        
        Args:
            result: EvaluationResult to render
            output_path: Path to save HTML file
        """
        html = self._generate_html_report(result)
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _generate_html_report(self, result: EvaluationResult) -> str:
        """Generate HTML report content."""
        score = result.get_overall_score()
        score_color = "#4CAF50" if score >= 0.7 else "#FF9800" if score >= 0.4 else "#F44336"
        
        failures_html = ""
        if result.failures:
            for f in result.failures:
                failures_html += f"""
                <div class="failure">
                    <strong>[{f.code}] {f.name}</strong><br>
                    <small>Category: {f.category} | Confidence: {f.confidence:.2%}</small><br>
                    <p>{f.evidence[:300]}{'...' if len(f.evidence) > 300 else ''}</p>
                </div>
                """
        else:
            failures_html = "<p style='color: #4CAF50;'>✅ No failures detected</p>"
        
        # Generate suggestions HTML
        suggestions_html = ""
        if hasattr(result, '_suggestions') and result._suggestions:
            suggestions_html = self._advisor._format_html(result._suggestions)
        else:
            suggestions_html = "<p style='color: #4CAF50;'>✅ No improvement suggestions</p>"
        
        # Get TRS details
        trs_html = ""
        if hasattr(result, '_trs_result') and result._trs_result:
            trs = result._trs_result
            trs_html = f"""
            <div class="metric">
                <div class="metric-value">{trs.get('overall_score', 'N/A')}</div>
                <div class="metric-label">Thought Relevance Score</div>
            </div>
            """
            
            # Add per-agent TRS breakdown
            if trs.get('agent_scores'):
                trs_html += '<div style="margin-top: 15px;"><strong>Per-Agent Thought Relevance:</strong><ul>'
                for agent, data in trs['agent_scores'].items():
                    color = "#4CAF50" if data['score'] >= 0.7 else "#FF9800" if data['score'] >= 0.4 else "#F44336"
                    trs_html += f'<li><strong>{agent}:</strong> <span style="color:{color}">{data["score"]:.2f}</span> ({data["thought_count"]} thoughts)</li>'
                trs_html += '</ul></div>'
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>MAS Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 30px; }}
        .score {{ font-size: 48px; font-weight: bold; color: {score_color}; }}
        .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: #f9f9f9; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .failure {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
        .stats {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px; }}
        .suggestions {{ margin-top: 20px; }}
        .suggestion {{ border-left: 4px solid; padding: 12px; margin: 10px 0; background: #f8f9fa; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 MAS Evaluation Report</h1>
        <p>Trace ID: {result.trace_id}<br>Generated: {result.timestamp}</p>
        
        <h2>Overall Score</h2>
        <div class="score">{score:.0%}</div>
        
        <h2>Metrics</h2>
        <div class="metric">
            <div class="metric-value">{result.metrics.get('IDS', 'N/A')}</div>
            <div class="metric-label">Information Diversity Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.metrics.get('UPR', 'N/A')}</div>
            <div class="metric-label">Unnecessary Path Ratio</div>
        </div>
        {trs_html}
        
        <h2>MAST Failure Analysis</h2>
        {failures_html}
        
        <h2>💡 Improvement Suggestions</h2>
        {suggestions_html}
        
        <div class="stats">
            <h3>Graph Statistics</h3>
            <p>Nodes: {result.graph_stats.get('num_nodes', 'N/A')} | 
               Edges: {result.graph_stats.get('num_edges', 'N/A')} | 
               Agents: {result.graph_stats.get('num_agents', 'N/A')}</p>
        </div>
    </div>
</body>
</html>
"""


# ===== Factory Functions =====

def create_evaluator(
    preset: str = "full",
    **kwargs
) -> MASEvaluator:
    """
    Create an evaluator with a preset configuration.
    
    Args:
        preset: "full", "metrics_only", "mast_only", "minimal"
        **kwargs: Additional configuration
    
    Returns:
        Configured MASEvaluator
    """
    presets = {
        "full": {
            "enable_tracing": True,
            "enable_crg": True,
            "enable_gemmas": True,
            "enable_mast": True
        },
        "metrics_only": {
            "enable_tracing": False,
            "enable_crg": True,
            "enable_gemmas": True,
            "enable_mast": False
        },
        "mast_only": {
            "enable_tracing": False,
            "enable_crg": False,
            "enable_gemmas": False,
            "enable_mast": True
        },
        "minimal": {
            "enable_tracing": False,
            "enable_crg": True,
            "enable_gemmas": False,
            "enable_mast": False
        }
    }
    
    config = presets.get(preset, presets["full"])
    config.update(kwargs)
    
    return MASEvaluator(**config)
