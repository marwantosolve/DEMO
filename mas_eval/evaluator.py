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
from mas_eval.metrics.thought_relevance import ThoughtQualityMetric
from mas_eval.mast import MASTClassifier, MASTTaxonomy, ClassifierMode
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
            mast_model="gemini-2.5-flash",
            custom_metrics={"balance": AgentActivityMetric()}
        )
    """
    
    def __init__(
        self,
        enable_tracing: bool = True,
        enable_crg: bool = True,
        enable_gemmas: bool = True,
        enable_mast: bool = True,
        mast_model: str = "gemini-2.5-flash",
        mast_mode: str = "few_shot_icl", # zero_shot, few_shot, few_shot_icl, fine_tuned
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
            mast_mode: Classification mode (zero_shot, few_shot_icl, fine_tuned)
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
            # Map string mode to Enum
            try:
                mode_enum = ClassifierMode(mast_mode)
            except ValueError:
                mode_enum = ClassifierMode.FEW_SHOT_ICL
                
            self._mast = MASTClassifier(
                model=mast_model,
                mode=mode_enum,
                confidence_threshold=mast_confidence
            )
        
        # Initialize TRS, Thought Quality, and Advisor
        self._trs = ThoughtRelevanceMetric()
        self._thought_quality = ThoughtQualityMetric()
        self._advisor = MASAdvisor()
        
        # GOT enhancement settings
        self._enable_semantic_edges = self.config.get("enable_semantic_edges", True)
        self._enable_info_flow = self.config.get("enable_info_flow", True)
    
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
            
            # === GOT Enhancements ===
            # Add semantic edges to connect related thoughts
            if self._enable_semantic_edges:
                try:
                    semantic_count = self._crg.add_semantic_edges(
                        similarity_threshold=0.6,
                        cross_agent_only=True
                    )
                    print(f"[GOT] Added {semantic_count} semantic edges")
                except Exception as e:
                    print(f"[GOT] Semantic edges skipped: {e}")
            
            # Add information flow edges
            if self._enable_info_flow:
                try:
                    flow_count = self._crg.add_information_flow_edges()
                    print(f"[GOT] Added {flow_count} information flow edges")
                except Exception as e:
                    print(f"[GOT] Info flow edges skipped: {e}")
            
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
        
        # Calculate comprehensive Thought Quality
        thought_quality_result = None
        if self._thought_quality:
            try:
                thought_quality_result = self._thought_quality.calculate(graph, spans)
                metrics["ThoughtQuality"] = thought_quality_result.get("overall_score", 0.0)
                metrics["Coherence"] = thought_quality_result.get("coherence", 0.0)
                metrics["Depth"] = thought_quality_result.get("depth", 0.0)
                metrics["Actionability"] = thought_quality_result.get("actionability", 0.0)
            except Exception as e:
                print(f"[GOT] Thought quality calculation failed: {e}")
        
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
                <div class="failure-card" data-category="{f.category}">
                    <div class="failure-header">
                        <span class="failure-code">{f.code}</span>
                        <span class="failure-cat">{f.category}</span>
                    </div>
                    <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px;">{f.name}</div>
                    <div style="margin-bottom: 8px; font-size: 14px;">
                        Confidence: <span class="{ 'confidence-high' if f.confidence > 0.8 else 'confidence-med' }">{f.confidence:.0%}</span>
                    </div>
                    <p style="margin: 0; color: #444;">{f.evidence}</p>
                    { f'<div class="evidence">Reasoning: ' + f.reasoning + '</div>' if hasattr(f, 'reasoning') and f.reasoning else '' }
                </div>
                """
        else:
            failures_html = """
            <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 8px;">
                <h3 style="color: #188038; margin: 0;">✅ No Failures Detected</h3>
                <p style="color: #666;">The Multi-Agent System performed within expected parameters.</p>
            </div>
            """
        
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
        
        # Get available failure filters
        categories = set(f.category for f in result.failures)
        filter_buttons = "".join([f'<button onclick="filterFailures(\'{cat}\')">{cat}</button>' for cat in categories])
        filter_buttons = f'<button onclick="filterFailures(\'all\')" class="active">All</button>' + filter_buttons
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>MAS Evaluation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #f8f9fa; color: #333; }}
        .header {{ background: #1a73e8; color: white; padding: 20px 40px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .container {{ max-width: 1200px; margin: 40px auto; padding: 0 20px; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 25px; margin-bottom: 20px; }}
        h1 {{ margin: 0; font-size: 24px; }}
        h2 {{ color: #1a73e8; border-bottom: 2px solid #e8f0fe; padding-bottom: 10px; margin-top: 0; }}
        
        .score-box {{ text-align: center; padding: 20px; }}
        .score-val {{ font-size: 56px; font-weight: bold; color: {score_color}; line-height: 1; }}
        .score-label {{ font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric-item {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .metric-val {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-lbl {{ font-size: 12px; color: #666; margin-top: 5px; }}
        
        .filters {{ margin-bottom: 15px; }}
        .filters button {{ background: #fff; border: 1px solid #ddd; padding: 8px 16px; border-radius: 20px; cursor: pointer; margin-right: 10px; transition: all 0.2s; }}
        .filters button:hover {{ background: #f1f3f4; }}
        .filters button.active {{ background: #1a73e8; color: white; border-color: #1a73e8; }}
        
        .failure-card {{ border-left: 4px solid #ffc107; background: #fff; padding: 20px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
        .failure-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .failure-code {{ font-weight: bold; color: #d93025; background: #fce8e6; padding: 4px 8px; border-radius: 4px; }}
        .failure-cat {{ font-size: 12px; color: #666; background: #f1f3f4; padding: 4px 8px; border-radius: 4px; }}
        .confidence-high {{ color: #188038; font-weight: bold; }}
        .confidence-med {{ color: #f9ab00; font-weight: bold; }}
        .evidence {{ background: #fafafa; padding: 15px; border-radius: 4px; font-style: italic; color: #555; border: 1px solid #eee; margin-top: 10px; }}
        
        .suggestion {{ border-left: 4px solid #1a73e8; background: #f8f9fa; padding: 15px; margin-bottom: 10px; border-radius: 4px; }}
        
        .footer {{ text-align: center; margin-top: 50px; color: #888; font-size: 12px; padding-bottom: 20px; }}
    </style>
    <script>
        function filterFailures(category) {{
            const cards = document.querySelectorAll('.failure-card');
            const buttons = document.querySelectorAll('.filters button');
            
            buttons.forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            
            cards.forEach(card => {{
                if (category === 'all' || card.dataset.category === category) {{
                    card.style.display = 'block';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</head>
<body>
    <div class="header">
        <div style="max-width: 1200px; margin: auto; display: flex; justify-content: space-between; align-items: center;">
            <h1>🔍 MAS Evaluation Report</h1>
            <span style="font-size: 14px; opacity: 0.8;">Trace ID: {result.trace_id}</span>
        </div>
    </div>

    <div class="container">
        <div class="metrics-grid">
            <div class="card score-box">
                <div class="score-val">{score:.0%}</div>
                <div class="score-label">Overall Quality Score</div>
            </div>
            
            <div class="card">
                <h2>Key Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-val">{result.metrics.get('IDS', 'N/A')}</div>
                        <div class="metric-lbl">Diversity (IDS)</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-val">{result.metrics.get('UPR', 'N/A')}</div>
                        <div class="metric-lbl">Efficiency (UPR)</div>
                    </div>
                    {trs_html.replace('<div class="metric">', '').replace('</div>', '') if trs_html else ''}
                </div>
            </div>
        </div>

        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2>MAST Failure Analysis</h2>
                <div class="filters">
                    {filter_buttons}
                </div>
            </div>
            
            <div id="failures-list">
                {failures_html}
            </div>
        </div>
        
        <div class="card">
             <h2>💡 AI Advisor Suggestions</h2>
             {suggestions_html}
        </div>
        
        <div class="footer">
            Generated by MAS Evaluation Framework • {result.timestamp}
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
