"""
Core type definitions for MAS Evaluation Framework.

These types provide a standardized interface for trace data across different
Multi-Agent System frameworks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class StepType(Enum):
    """Types of agent steps captured in traces."""
    THOUGHT = "thought"
    OBSERVATION = "observation"
    ACTION = "action"
    OUTPUT = "output"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE = "message"
    ERROR = "error"


class EdgeType(Enum):
    """Types of edges in the GEMMAS graph (for adjacency matrices)."""
    SPATIAL = "spatial"      # Who talks to whom (S matrix)
    TEMPORAL = "temporal"    # Causal ordering (T matrix)
    SEMANTIC = "semantic"    # Content similarity
    CAUSAL = "causal"        # Parent-child relationship
    INFORMATION_FLOW = "information_flow"  # Content propagation


@dataclass
class Span:
    """
    Represents a single traced step in an agent's execution.
    
    Attributes:
        span_id: Unique identifier for this span
        parent_span_id: ID of parent span (for hierarchy)
        name: Human-readable name of the span
        agent_name: Name of the agent that generated this span
        step_type: Type of step (thought, action, etc.)
        content: Text content of the step
        start_time: When the step started
        end_time: When the step ended
        attributes: Additional key-value metadata
    """
    span_id: str
    name: str
    agent_name: str
    step_type: StepType
    content: str
    start_time: datetime
    end_time: Optional[datetime] = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    # GEMMAS fields for iteration and message tracking
    iteration: int = 0  # Agent's iteration counter
    from_agent: Optional[str] = None  # Source agent for message passing
    to_agent: Optional[str] = None    # Target agent for message passing
    
    def duration_ms(self) -> float:
        """Calculate duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "agent_name": self.agent_name,
            "step_type": self.step_type.value,
            "content": self.content,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "attributes": self.attributes,
            "iteration": self.iteration,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        """Create span from dictionary."""
        return cls(
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            name=data["name"],
            agent_name=data["agent_name"],
            step_type=StepType(data["step_type"]),
            content=data["content"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            attributes=data.get("attributes", {}),
            iteration=data.get("iteration", 0),
            from_agent=data.get("from_agent"),
            to_agent=data.get("to_agent")
        )


@dataclass
class Node:
    """
    Represents a node in the Causal Reasoning Graph.
    
    Attributes:
        node_id: Unique identifier (usually same as span_id)
        agent_name: Name of the agent
        step_type: Type of step
        content: Text content
        timestamp: When this node occurred
        embedding: Optional semantic embedding vector
    """
    node_id: str
    agent_name: str
    step_type: StepType
    content: str
    timestamp: datetime
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GEMMASNode:
    """
    GEMMAS-style node representing (Agent_ID, Iteration, Role).
    
    This aligns with the GEMMAS paper methodology where a node represents
    an agent's specific interaction at a specific time step.
    
    Attributes:
        agent_id: Unique identifier for the agent
        iteration: Which iteration/turn this represents
        role: 'prompt' (input processing) or 'response' (output generation)
        span_id: Link to the original span for full details
        content: Text content (may be truncated)
        timestamp: When this interaction occurred
    """
    agent_id: str
    iteration: int
    role: str  # 'prompt' or 'response'
    span_id: str
    content: str
    timestamp: datetime
    
    def node_key(self) -> str:
        """Generate unique key for this node: (agent_id, iteration, role)."""
        return f"{self.agent_id}_{self.iteration}_{self.role}"
    
    @classmethod
    def from_span(cls, span: "Span", iteration: int = 0) -> "GEMMASNode":
        """Create GEMMASNode from a Span."""
        # Determine role based on step type
        role = "response" if span.step_type in [StepType.OUTPUT, StepType.TOOL_RESULT] else "prompt"
        return cls(
            agent_id=span.agent_name,
            iteration=iteration or span.iteration,
            role=role,
            span_id=span.span_id,
            content=span.content[:500],  # Truncate for memory efficiency
            timestamp=span.start_time
        )


@dataclass
class Edge:
    """
    Represents an edge in the Causal Reasoning Graph.
    
    Attributes:
        source_id: ID of source node
        target_id: ID of target node
        edge_type: Type of relationship (causal, temporal, semantic)
        weight: Optional edge weight
    """
    source_id: str
    target_id: str
    edge_type: str = "causal"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceData:
    """
    Complete trace data from a MAS execution.
    
    Attributes:
        trace_id: Unique identifier for this trace
        spans: List of spans in execution order
        metadata: Additional trace-level metadata
    """
    trace_id: str
    spans: List[Span]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_agents(self) -> List[str]:
        """Get unique list of agents in this trace."""
        return list(set(span.agent_name for span in self.spans))
    
    def get_spans_by_agent(self, agent_name: str) -> List[Span]:
        """Get all spans for a specific agent."""
        return [s for s in self.spans if s.agent_name == agent_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata
        }


@dataclass
class ConfidenceBreakdown:
    """
    Detailed confidence scoring with component breakdown.
    
    Each component contributes to the final probability:
    - evidence_strength (35%): How specific/quoted the evidence is
    - pattern_match_score (25%): Match to MAST taxonomy indicators
    - cross_reference_score (20%): References to specific spans/agents
    - model_confidence (20%): LLM's self-reported confidence
    """
    evidence_strength: float = 0.5       # How specific the evidence is (0-1)
    pattern_match_score: float = 0.5     # Match to taxonomy indicators (0-1)
    cross_reference_score: float = 0.5   # References to spans/agents (0-1)
    model_confidence: float = 0.5        # LLM's self-reported confidence (0-1)
    
    @property
    def final_probability(self) -> float:
        """Compute weighted final probability."""
        weights = {
            'evidence_strength': 0.35,
            'pattern_match_score': 0.25,
            'cross_reference_score': 0.20,
            'model_confidence': 0.20
        }
        return round(
            self.evidence_strength * weights['evidence_strength'] +
            self.pattern_match_score * weights['pattern_match_score'] +
            self.cross_reference_score * weights['cross_reference_score'] +
            self.model_confidence * weights['model_confidence'],
            3
        )
    
    def to_percentage(self) -> str:
        """Return probability as percentage string."""
        return f"{self.final_probability * 100:.1f}%"
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'evidence_strength': self.evidence_strength,
            'pattern_match_score': self.pattern_match_score,
            'cross_reference_score': self.cross_reference_score,
            'model_confidence': self.model_confidence,
            'final_probability': self.final_probability
        }


@dataclass
class FailureMode:
    """
    Represents a detected MAST failure mode with probabilistic confidence.
    
    Attributes:
        code: Failure mode code (e.g., "INTER-3")
        name: Human-readable name
        category: Category (specification, inter_agent, task_verification)
        confidence: Simple confidence score (0-1) for backward compatibility
        evidence: Supporting evidence from trace
        span_ids: IDs of spans involved in failure
        confidence_breakdown: Detailed confidence scoring components
        proof_quotes: Exact quotes from trace as proof
        affected_steps: Step numbers affected (e.g., ["Step 1", "Step 3"])
        reasoning_chain: Chain of reasoning that led to detection
    """
    code: str
    name: str
    category: str
    confidence: float
    evidence: str
    span_ids: List[str] = field(default_factory=list)
    # Enhanced confidence scoring
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    proof_quotes: List[str] = field(default_factory=list)
    affected_steps: List[str] = field(default_factory=list)
    reasoning_chain: str = ""
    
    def get_probability(self) -> float:
        """Get the best available probability score."""
        if self.confidence_breakdown:
            return self.confidence_breakdown.final_probability
        return self.confidence
    
    def get_probability_percentage(self) -> str:
        """Get probability as formatted percentage."""
        return f"{self.get_probability() * 100:.1f}%"


@dataclass
class EvaluationResult:
    """
    Complete evaluation result from MASEvaluator.
    
    Attributes:
        trace_id: ID of evaluated trace
        metrics: Dictionary of metric name -> value
        failures: List of detected failure modes
        graph_stats: Statistics about the CRG
        timestamp: When evaluation was performed
    """
    trace_id: str
    metrics: Dict[str, float]
    failures: List[FailureMode]
    graph_stats: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_overall_score(self) -> float:
        """Calculate overall quality score (0-1)."""
        ids = self.metrics.get("IDS", 0.5)
        upr = self.metrics.get("UPR", 0.5)
        failure_penalty = min(len(self.failures) * 0.1, 0.5)
        
        return max(0, (ids * 0.5 + (1 - upr) * 0.5) - failure_penalty)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "trace_id": self.trace_id,
            "metrics": self.metrics,
            "failures": [
                {
                    "code": f.code,
                    "name": f.name,
                    "category": f.category,
                    "confidence": f.confidence,
                    "evidence": f.evidence
                }
                for f in self.failures
            ],
            "graph_stats": self.graph_stats,
            "overall_score": self.get_overall_score(),
            "timestamp": self.timestamp.isoformat()
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== MAS Evaluation Report ===",
            f"Trace ID: {self.trace_id}",
            f"Overall Score: {self.get_overall_score():.2%}",
            "",
            "Metrics:",
            f"  - IDS (Information Diversity): {self.metrics.get('IDS', 'N/A')}",
            f"  - UPR (Unnecessary Paths): {self.metrics.get('UPR', 'N/A')}",
            "",
            f"Failures Detected: {len(self.failures)}"
        ]
        
        if self.failures:
            lines.append("")
            for f in self.failures:
                lines.append(f"  [{f.code}] {f.name} (conf: {f.confidence:.2f})")
        
        return "\n".join(lines)
