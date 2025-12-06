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
            "attributes": self.attributes
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
            attributes=data.get("attributes", {})
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
class FailureMode:
    """
    Represents a detected MAST failure mode.
    
    Attributes:
        code: Failure mode code (e.g., "INTER-3")
        name: Human-readable name
        category: Category (specification, inter_agent, task_verification)
        confidence: Confidence score (0-1)
        evidence: Supporting evidence from trace
        span_ids: IDs of spans involved in failure
    """
    code: str
    name: str
    category: str
    confidence: float
    evidence: str
    span_ids: List[str] = field(default_factory=list)


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
