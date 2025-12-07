"""
Agentic Analytics Converter

Converts raw OpenTelemetry traces into hierarchical agentic analytics format
with agent sessions, task flows, and flow graphs.

Based on the beyond-black-box-benchmarking paper's innovation for visualizing
and understanding Multi-Agent System behavior.

Usage:
    from agentic_analytics_converter import AgenticAnalyticsConverter
    
    converter = AgenticAnalyticsConverter()
    analytics = converter.convert("otel_traces.json")
    converter.save(analytics, "agentic_analytics.json")
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import re


@dataclass
class AgentInfo:
    """Information about an agent extracted from spans."""
    name: str
    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    llm_model: Optional[str] = None
    tasks_executed: List[str] = field(default_factory=list)
    llm_calls: int = 0
    token_usage: Dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})


@dataclass
class TaskInfo:
    """Information about a task in the task flow."""
    task_id: str
    name: str
    agent: Optional[str] = None
    status: str = "unknown"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: float = 0
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    description: Optional[str] = None
    expected_output: Optional[str] = None
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    span_type: str = "task"  # task, llm_call, tool_call, agent


@dataclass
class FlowGraphNode:
    """Node in the flow graph."""
    id: str
    label: str
    node_type: str  # agent, task, llm, tool
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowGraphEdge:
    """Edge in the flow graph."""
    source: str
    target: str
    edge_type: str  # executes, calls, delegates
    label: Optional[str] = None


class AgenticAnalyticsConverter:
    """
    Converts raw OpenTelemetry spans into agentic analytics format.
    
    The output provides:
    - Session summary with aggregate metrics
    - Agent information with roles, goals, tools
    - Hierarchical task flow with parent-child relationships
    - Flow graph for visualization
    """
    
    # Known span attribute patterns for extracting agentic concepts
    AGENT_PATTERNS = {
        'name': ['crewai.agent.role', 'traceloop.entity.name', 'agent.name'],
        'role': ['crewai.agent.role', 'agent.role'],
        'goal': ['crewai.agent.goal', 'agent.goal'],
        'backstory': ['crewai.agent.backstory'],
        'tools': ['crewai.agent.tools_names', 'agent.tools'],
        'llm': ['crewai.agent.llm', 'llm.model', 'gen_ai.request.model']
    }
    
    TASK_PATTERNS = {
        'description': ['crewai.task.description', 'task.description'],
        'expected_output': ['crewai.task.expected_output'],
        'agent': ['crewai.task.agent', 'crewai.task.agent_role'],
        'input': ['traceloop.entity.input', 'task.input'],
        'output': ['traceloop.entity.output', 'task.output', 'crewai.task.result']
    }
    
    LLM_PATTERNS = {
        'model': ['llm.model', 'gen_ai.request.model', 'llm.request.model'],
        'input_tokens': ['llm.usage.prompt_tokens', 'gen_ai.usage.input_tokens'],
        'output_tokens': ['llm.usage.completion_tokens', 'gen_ai.usage.output_tokens'],
        'request_type': ['llm.request_type', 'gen_ai.operation.name'],
        'prompts': ['llm.prompts', 'gen_ai.prompt'],
        'completions': ['llm.completions', 'gen_ai.completion']
    }
    
    def __init__(self):
        self.spans: List[Dict[str, Any]] = []
        self.span_map: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, AgentInfo] = {}
        self.tasks: List[TaskInfo] = []
        self.nodes: List[FlowGraphNode] = []
        self.edges: List[FlowGraphEdge] = []
    
    def convert(self, input_file: str) -> Dict[str, Any]:
        """
        Convert an OTel traces JSON file to agentic analytics format.
        
        Args:
            input_file: Path to the JSON file from OTelCapture
            
        Returns:
            Agentic analytics dictionary
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        spans = data.get('spans', [])
        return self.convert_from_spans(spans)
    
    def convert_from_spans(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert a list of serialized spans to agentic analytics format.
        
        Args:
            spans: List of span dictionaries
            
        Returns:
            Agentic analytics dictionary
        """
        self._reset()
        self.spans = spans
        
        # Build span map for parent-child lookups
        for span in spans:
            self.span_map[span['span_id']] = span
        
        # Extract information from spans
        self._extract_agents()
        self._build_task_flow()
        self._build_flow_graph()
        
        # Calculate summary metrics
        summary = self._calculate_summary()
        
        return {
            "session_id": self._get_session_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_duration_ms": self._calculate_total_duration(),
            "summary": summary,
            "agents": [asdict(agent) for agent in self.agents.values()],
            "task_flow": [asdict(task) for task in self.tasks],
            "flow_graph": {
                "nodes": [asdict(node) for node in self.nodes],
                "edges": [asdict(edge) for edge in self.edges]
            },
            "raw_span_count": len(spans)
        }
    
    def _reset(self):
        """Reset internal state for new conversion."""
        self.spans = []
        self.span_map = {}
        self.agents = {}
        self.tasks = []
        self.nodes = []
        self.edges = []
    
    def _get_session_id(self) -> str:
        """Get session ID from first span's trace ID."""
        if self.spans:
            return self.spans[0].get('trace_id', 'unknown')
        return 'unknown'
    
    def _calculate_total_duration(self) -> float:
        """Calculate total duration from earliest start to latest end."""
        if not self.spans:
            return 0
        
        start_times = []
        end_times = []
        
        for span in self.spans:
            if span.get('start_time'):
                start_times.append(span['start_time'])
            if span.get('end_time'):
                end_times.append(span['end_time'])
        
        if not start_times or not end_times:
            return 0
        
        # Parse ISO times and calculate difference
        try:
            earliest = min(datetime.fromisoformat(t.rstrip('Z')) for t in start_times)
            latest = max(datetime.fromisoformat(t.rstrip('Z')) for t in end_times)
            return (latest - earliest).total_seconds() * 1000
        except:
            return sum(span.get('duration_ms', 0) for span in self.spans)
    
    def _get_attr(self, attrs: Dict[str, Any], patterns: List[str], default=None) -> Any:
        """Get an attribute value using multiple possible keys."""
        for pattern in patterns:
            if pattern in attrs:
                return attrs[pattern]
        return default
    
    def _extract_agents(self):
        """Extract agent information from spans."""
        for span in self.spans:
            attrs = span.get('attributes', {})
            name = span.get('name', '')
            
            # Check if this is an agent-related span
            is_agent_span = (
                'agent' in name.lower() or
                'Agent' in name or
                any(k.startswith('crewai.agent') for k in attrs.keys()) or
                attrs.get('traceloop.entity.name')
            )
            
            if is_agent_span:
                agent_name = self._get_attr(attrs, self.AGENT_PATTERNS['name']) or name
                
                if agent_name and agent_name not in self.agents:
                    self.agents[agent_name] = AgentInfo(
                        name=agent_name,
                        role=self._get_attr(attrs, self.AGENT_PATTERNS['role']),
                        goal=self._get_attr(attrs, self.AGENT_PATTERNS['goal']),
                        backstory=self._get_attr(attrs, self.AGENT_PATTERNS['backstory']),
                        llm_model=self._get_attr(attrs, self.AGENT_PATTERNS['llm'])
                    )
                    
                    # Extract tools
                    tools = self._get_attr(attrs, self.AGENT_PATTERNS['tools'])
                    if tools:
                        if isinstance(tools, str):
                            try:
                                tools = json.loads(tools)
                            except:
                                tools = [tools]
                        self.agents[agent_name].tools = list(tools) if tools else []
            
            # Track LLM calls per agent
            if self._is_llm_span(attrs):
                agent_name = self._find_parent_agent(span)
                if agent_name and agent_name in self.agents:
                    self.agents[agent_name].llm_calls += 1
                    
                    # Token usage
                    input_tokens = self._get_attr(attrs, self.LLM_PATTERNS['input_tokens'], 0)
                    output_tokens = self._get_attr(attrs, self.LLM_PATTERNS['output_tokens'], 0)
                    
                    if isinstance(input_tokens, (int, float)):
                        self.agents[agent_name].token_usage['input'] += int(input_tokens)
                    if isinstance(output_tokens, (int, float)):
                        self.agents[agent_name].token_usage['output'] += int(output_tokens)
    
    def _is_llm_span(self, attrs: Dict[str, Any]) -> bool:
        """Check if attributes indicate an LLM call span."""
        llm_indicators = [
            'llm.request_type', 'llm.model', 'gen_ai.operation.name',
            'llm.prompts', 'llm.completions', 'gen_ai.request.model'
        ]
        return any(k in attrs for k in llm_indicators)
    
    def _find_parent_agent(self, span: Dict[str, Any]) -> Optional[str]:
        """Find the parent agent for a given span."""
        parent_id = span.get('parent_span_id')
        visited = set()
        
        while parent_id and parent_id not in visited:
            visited.add(parent_id)
            parent = self.span_map.get(parent_id)
            if not parent:
                break
            
            parent_attrs = parent.get('attributes', {})
            agent_name = self._get_attr(parent_attrs, self.AGENT_PATTERNS['name'])
            if agent_name:
                return agent_name
            
            parent_id = parent.get('parent_span_id')
        
        return None
    
    def _build_task_flow(self):
        """Build hierarchical task flow from spans."""
        # Sort spans by start time
        sorted_spans = sorted(
            self.spans, 
            key=lambda s: s.get('start_time', '')
        )
        
        for span in sorted_spans:
            attrs = span.get('attributes', {})
            name = span.get('name', 'unknown')
            
            # Determine span type
            span_type = self._classify_span_type(name, attrs)
            
            # Create task info
            task = TaskInfo(
                task_id=span['span_id'],
                name=name,
                status=span.get('status_code', 'unknown'),
                start_time=span.get('start_time'),
                end_time=span.get('end_time'),
                duration_ms=span.get('duration_ms', 0),
                parent_id=span.get('parent_span_id'),
                span_type=span_type
            )
            
            # Extract task-specific attributes
            task.description = self._get_attr(attrs, self.TASK_PATTERNS['description'])
            task.expected_output = self._get_attr(attrs, self.TASK_PATTERNS['expected_output'])
            task.input_data = self._get_attr(attrs, self.TASK_PATTERNS['input'])
            task.output_data = self._get_attr(attrs, self.TASK_PATTERNS['output'])
            task.agent = self._get_attr(attrs, self.TASK_PATTERNS['agent'])
            
            # If no agent found in attrs, try parent lookup
            if not task.agent and span_type in ['task', 'llm_call', 'tool_call']:
                task.agent = self._find_parent_agent(span)
            
            # Track task execution for agents
            if task.agent and task.agent in self.agents:
                if task.task_id not in self.agents[task.agent].tasks_executed:
                    self.agents[task.agent].tasks_executed.append(task.task_id)
            
            self.tasks.append(task)
        
        # Build parent-child relationships
        task_map = {t.task_id: t for t in self.tasks}
        for task in self.tasks:
            if task.parent_id and task.parent_id in task_map:
                parent = task_map[task.parent_id]
                if task.task_id not in parent.children:
                    parent.children.append(task.task_id)
    
    def _classify_span_type(self, name: str, attrs: Dict[str, Any]) -> str:
        """Classify span into agent, task, llm_call, or tool_call."""
        name_lower = name.lower()
        
        # LLM call indicators
        if self._is_llm_span(attrs):
            return 'llm_call'
        
        # Tool indicators
        if 'tool' in name_lower or any(k.startswith('tool.') for k in attrs.keys()):
            return 'tool_call'
        
        # Agent indicators
        if 'agent' in name_lower or any(k.startswith('crewai.agent') for k in attrs.keys()):
            return 'agent'
        
        # Task indicators
        if 'task' in name_lower or any(k.startswith('crewai.task') for k in attrs.keys()):
            return 'task'
        
        # Crew indicators
        if 'crew' in name_lower or any(k.startswith('crew_') for k in attrs.keys()):
            return 'workflow'
        
        return 'other'
    
    def _build_flow_graph(self):
        """Build flow graph from task flow."""
        added_nodes: Set[str] = set()
        
        for task in self.tasks:
            # Add node if not already added
            if task.task_id not in added_nodes:
                self.nodes.append(FlowGraphNode(
                    id=task.task_id,
                    label=task.name[:50],  # Truncate long names
                    node_type=task.span_type,
                    attributes={
                        "duration_ms": task.duration_ms,
                        "status": task.status
                    }
                ))
                added_nodes.add(task.task_id)
            
            # Add edge from parent to this task
            if task.parent_id:
                edge_type = self._determine_edge_type(task.span_type)
                self.edges.append(FlowGraphEdge(
                    source=task.parent_id,
                    target=task.task_id,
                    edge_type=edge_type
                ))
    
    def _determine_edge_type(self, child_type: str) -> str:
        """Determine edge type based on child span type."""
        type_map = {
            'llm_call': 'calls',
            'tool_call': 'uses',
            'task': 'executes',
            'agent': 'delegates',
            'workflow': 'contains'
        }
        return type_map.get(child_type, 'contains')
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary metrics."""
        total_tokens = sum(
            agent.token_usage['input'] + agent.token_usage['output']
            for agent in self.agents.values()
        )
        
        llm_calls = sum(agent.llm_calls for agent in self.agents.values())
        
        # Count by span type
        type_counts = defaultdict(int)
        for task in self.tasks:
            type_counts[task.span_type] += 1
        
        return {
            "agents": len(self.agents),
            "tasks": type_counts.get('task', 0),
            "llm_calls": llm_calls or type_counts.get('llm_call', 0),
            "tool_calls": type_counts.get('tool_call', 0),
            "total_spans": len(self.spans),
            "total_tokens": total_tokens,
            "span_types": dict(type_counts)
        }
    
    def save(self, analytics: Dict[str, Any], output_file: str) -> str:
        """
        Save agentic analytics to JSON file.
        
        Args:
            analytics: The analytics dictionary from convert()
            output_file: Path to save the JSON file
            
        Returns:
            Path to the saved file
        """
        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        print(f"[AgenticAnalytics] Saved analytics to: {output_file}")
        return output_file


def convert_otel_to_analytics(
    input_file: str, 
    output_file: str = None
) -> Dict[str, Any]:
    """
    Convenience function to convert OTel traces to agentic analytics.
    
    Args:
        input_file: Path to OTel traces JSON
        output_file: Optional path to save analytics (auto-generated if not provided)
        
    Returns:
        Agentic analytics dictionary
    """
    converter = AgenticAnalyticsConverter()
    analytics = converter.convert(input_file)
    
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_analytics.json"
    
    converter.save(analytics, output_file)
    return analytics


# Quick test
if __name__ == "__main__":
    # Create sample OTel data for testing
    sample_spans = [
        {
            "trace_id": "abc123",
            "span_id": "span1",
            "parent_span_id": None,
            "name": "Crew.kickoff",
            "kind": "CLIENT",
            "start_time": "2024-01-01T10:00:00Z",
            "end_time": "2024-01-01T10:00:30Z",
            "duration_ms": 30000,
            "status_code": "OK",
            "status_description": None,
            "attributes": {
                "crew_agents": '[{"role": "Researcher", "goal": "Find info"}]',
                "crew_tasks": '[{"description": "Research topic"}]'
            }
        },
        {
            "trace_id": "abc123",
            "span_id": "span2",
            "parent_span_id": "span1",
            "name": "Agent.execute",
            "kind": "CLIENT",
            "start_time": "2024-01-01T10:00:01Z",
            "end_time": "2024-01-01T10:00:25Z",
            "duration_ms": 24000,
            "status_code": "OK",
            "status_description": None,
            "attributes": {
                "crewai.agent.role": "Researcher",
                "crewai.agent.goal": "Find relevant information",
                "traceloop.entity.name": "Researcher"
            }
        },
        {
            "trace_id": "abc123",
            "span_id": "span3",
            "parent_span_id": "span2",
            "name": "LLM.generate",
            "kind": "CLIENT",
            "start_time": "2024-01-01T10:00:05Z",
            "end_time": "2024-01-01T10:00:10Z",
            "duration_ms": 5000,
            "status_code": "OK",
            "status_description": None,
            "attributes": {
                "llm.model": "gemini-2.0-flash",
                "llm.request_type": "chat",
                "llm.usage.prompt_tokens": 150,
                "llm.usage.completion_tokens": 200
            }
        }
    ]
    
    # Test conversion
    converter = AgenticAnalyticsConverter()
    analytics = converter.convert_from_spans(sample_spans)
    
    print("\n=== Agentic Analytics Output ===")
    print(json.dumps(analytics, indent=2))
    
    # Save to file
    converter.save(analytics, "test_analytics.json")
    print("\nTest complete! Check test_analytics.json")
