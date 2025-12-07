"""
Agent Interaction Graph Module.

Builds a simplified agent-centric graph where:
- Nodes = Agents (N agents → N nodes)
- Edges = Interactions between agents (info exchange, delegation, transfers)

This provides a high-level view of how agents collaborate in the MAS.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import networkx as nx

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from mas_eval.core.types import Span, StepType


class AgentGraphBuilder:
    """
    Builds an Agent-Centric Graph from trace spans.
    
    Unlike CRGModule (which creates node per span), this creates:
    - One node per unique agent
    - Edges based on all interactions between agents
    
    Usage:
        builder = AgentGraphBuilder()
        agent_graph = builder.build(spans)
        
        # Get interaction details
        interactions = builder.get_interaction_summary()
    """
    
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._agent_stats: Dict[str, Dict[str, Any]] = {}
        self._interactions: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    def build(self, spans: List[Span]) -> nx.DiGraph:
        """
        Build agent-centric graph from spans.
        
        Args:
            spans: List of Span objects from trace
            
        Returns:
            NetworkX DiGraph with agents as nodes
        """
        self.graph = nx.DiGraph()
        self._agent_stats = {}
        self._interactions = {}
        
        # Sort spans by time
        sorted_spans = sorted(spans, key=lambda x: x.start_time)
        
        # Phase 1: Collect agent statistics
        self._collect_agent_stats(sorted_spans)
        
        # Phase 2: Create agent nodes
        self._create_agent_nodes()
        
        # Phase 3: Detect and create interaction edges
        self._detect_interactions(sorted_spans)
        self._create_edges()
        
        return self.graph
    
    def _collect_agent_stats(self, spans: List[Span]) -> None:
        """Collect statistics for each agent."""
        for span in spans:
            agent = span.agent_name
            if agent not in self._agent_stats:
                self._agent_stats[agent] = {
                    'span_count': 0,
                    'thought_count': 0,
                    'action_count': 0,
                    'output_count': 0,
                    'tool_count': 0,
                    'first_active': span.start_time,
                    'last_active': span.start_time,
                    'content_samples': []
                }
            
            stats = self._agent_stats[agent]
            stats['span_count'] += 1
            stats['last_active'] = max(stats['last_active'], span.start_time)
            
            # Count by type
            if span.step_type == StepType.THOUGHT:
                stats['thought_count'] += 1
            elif span.step_type == StepType.ACTION:
                stats['action_count'] += 1
            elif span.step_type == StepType.OUTPUT:
                stats['output_count'] += 1
            elif span.step_type in [StepType.TOOL_CALL, StepType.TOOL_RESULT]:
                stats['tool_count'] += 1
            
            # Sample content for summaries
            if len(stats['content_samples']) < 3 and len(span.content) > 20:
                stats['content_samples'].append(span.content[:100])
    
    def _create_agent_nodes(self) -> None:
        """Create one node per agent with aggregated stats."""
        for agent, stats in self._agent_stats.items():
            self.graph.add_node(
                agent,
                agent_name=agent,
                span_count=stats['span_count'],
                thought_count=stats['thought_count'],
                action_count=stats['action_count'],
                output_count=stats['output_count'],
                tool_count=stats['tool_count'],
                first_active=stats['first_active'],
                last_active=stats['last_active'],
                # Node size proportional to activity
                size=stats['span_count'] * 100 + 500
            )
    
    def _detect_interactions(self, spans: List[Span]) -> None:
        """
        Detect all interactions between agents.
        
        Interaction types:
        1. Transfer: Explicit agent transfer (from_agent → to_agent)
        2. Delegation: Parent-child spans across agents
        3. Information Flow: Content reference between agents
        4. Temporal: Sequential activity between agents
        """
        # Build span lookup
        span_map = {s.span_id: s for s in spans}
        
        for span in spans:
            # 1. Transfer detection
            from_agent = span.from_agent or span.attributes.get('from_agent')
            to_agent = span.to_agent or span.attributes.get('to_agent')
            
            if from_agent and to_agent and from_agent != to_agent:
                self._add_interaction(from_agent, to_agent, 'transfer', span.content[:200])
            
            # 2. Delegation detection (parent-child across agents)
            if span.parent_span_id and span.parent_span_id in span_map:
                parent = span_map[span.parent_span_id]
                if parent.agent_name != span.agent_name:
                    self._add_interaction(
                        parent.agent_name, 
                        span.agent_name, 
                        'delegation',
                        f"Delegated: {span.name}"
                    )
        
        # 3. Temporal flow between agents
        prev_span = None
        for span in spans:
            if prev_span and prev_span.agent_name != span.agent_name:
                # Check if there's content similarity (simple heuristic)
                if self._has_content_overlap(prev_span.content, span.content):
                    self._add_interaction(
                        prev_span.agent_name,
                        span.agent_name,
                        'information_flow',
                        f"Content flows from {prev_span.agent_name}"
                    )
                else:
                    # Still mark temporal sequence
                    self._add_interaction(
                        prev_span.agent_name,
                        span.agent_name,
                        'temporal',
                        'Sequential activity'
                    )
            prev_span = span
    
    def _has_content_overlap(self, content1: str, content2: str, threshold: float = 0.2) -> bool:
        """Check if two content strings have significant word overlap."""
        if not content1 or not content2:
            return False
        
        words1 = set(w.lower() for w in content1.split() if len(w) > 4)
        words2 = set(w.lower() for w in content2.split() if len(w) > 4)
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        min_words = min(len(words1), len(words2))
        
        return (overlap / min_words) >= threshold if min_words > 0 else False
    
    def _add_interaction(self, from_agent: str, to_agent: str, 
                         interaction_type: str, detail: str) -> None:
        """Add or update an interaction between agents."""
        key = (from_agent, to_agent)
        
        if key not in self._interactions:
            self._interactions[key] = {
                'count': 0,
                'types': set(),
                'details': []
            }
        
        self._interactions[key]['count'] += 1
        self._interactions[key]['types'].add(interaction_type)
        
        if len(self._interactions[key]['details']) < 5:
            self._interactions[key]['details'].append(detail[:100])
    
    def _create_edges(self) -> None:
        """Create edges from detected interactions."""
        for (from_agent, to_agent), data in self._interactions.items():
            if from_agent in self.graph and to_agent in self.graph:
                self.graph.add_edge(
                    from_agent,
                    to_agent,
                    weight=data['count'],
                    types=list(data['types']),
                    details=data['details'],
                    # Edge width proportional to interaction count
                    width=min(data['count'] * 0.5 + 1, 5)
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent graph."""
        if self.graph.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        n = self.graph.number_of_nodes()
        max_edges = n * (n - 1)  # Directed graph
        
        return {
            "num_agents": n,
            "num_interactions": self.graph.number_of_edges(),
            "agents": list(self.graph.nodes()),
            "collaboration_density": self.graph.number_of_edges() / max_edges if max_edges > 0 else 0,
            "total_interaction_count": sum(d['weight'] for _, _, d in self.graph.edges(data=True)),
            "is_connected": nx.is_weakly_connected(self.graph) if n > 1 else True
        }
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get detailed interaction summary."""
        summary = {
            "agents": {},
            "top_interactions": []
        }
        
        # Agent summaries
        for agent, data in self.graph.nodes(data=True):
            summary["agents"][agent] = {
                "spans": data.get('span_count', 0),
                "thoughts": data.get('thought_count', 0),
                "actions": data.get('action_count', 0),
                "outputs": data.get('output_count', 0),
                "outgoing": self.graph.out_degree(agent),
                "incoming": self.graph.in_degree(agent)
            }
        
        # Top interactions by weight
        edges = [(u, v, d['weight']) for u, v, d in self.graph.edges(data=True)]
        edges.sort(key=lambda x: x[2], reverse=True)
        summary["top_interactions"] = [
            {"from": u, "to": v, "count": w} 
            for u, v, w in edges[:10]
        ]
        
        return summary


class AgentGraphEvaluator:
    """
    Evaluator for Agent Interaction Graphs.
    
    Computes metrics that assess MAS quality based on agent interactions:
    - Centrality: Which agent is most important
    - Collaboration: How well agents work together
    - Balance: Is work evenly distributed
    - Bottlenecks: Are there communication bottlenecks
    
    Usage:
        evaluator = AgentGraphEvaluator()
        metrics = evaluator.evaluate(agent_graph, spans)
        report = evaluator.summary(metrics)
    """
    
    def evaluate(self, graph: nx.DiGraph, spans: List[Span] = None) -> Dict[str, Any]:
        """
        Evaluate the agent graph and compute metrics.
        
        Args:
            graph: Agent-centric graph from AgentGraphBuilder
            spans: Optional original spans for additional analysis
            
        Returns:
            Dictionary of evaluation metrics
        """
        if graph.number_of_nodes() == 0:
            return {"error": "Empty graph", "score": 0}
        
        n = graph.number_of_nodes()
        metrics = {
            "agent_count": n,
            "interaction_count": graph.number_of_edges(),
        }
        
        # 1. Collaboration Density
        max_edges = n * (n - 1)
        metrics["collaboration_density"] = (
            graph.number_of_edges() / max_edges if max_edges > 0 else 0
        )
        
        # 2. Centrality Metrics
        if n > 1:
            try:
                degree_cent = nx.degree_centrality(graph)
                betweenness_cent = nx.betweenness_centrality(graph)
                
                metrics["central_agent"] = max(degree_cent, key=degree_cent.get)
                metrics["central_agent_score"] = max(degree_cent.values())
                
                metrics["bottleneck_agent"] = max(betweenness_cent, key=betweenness_cent.get)
                metrics["bottleneck_score"] = max(betweenness_cent.values())
                
                metrics["degree_centrality"] = degree_cent
                metrics["betweenness_centrality"] = betweenness_cent
            except Exception:
                metrics["central_agent"] = list(graph.nodes())[0]
                metrics["central_agent_score"] = 0.5
                metrics["bottleneck_agent"] = list(graph.nodes())[0]
                metrics["bottleneck_score"] = 0
        else:
            metrics["central_agent"] = list(graph.nodes())[0]
            metrics["central_agent_score"] = 1.0
            metrics["bottleneck_agent"] = list(graph.nodes())[0]
            metrics["bottleneck_score"] = 0
        
        # 3. Load Balance Score
        if spans:
            spans_per_agent = {}
            for span in spans:
                agent = span.agent_name
                spans_per_agent[agent] = spans_per_agent.get(agent, 0) + 1
            
            values = list(spans_per_agent.values())
            if len(values) > 1 and HAS_NUMPY:
                mean_val = np.mean(values)
                std_val = np.std(values)
                # Higher score = more balanced (1 - coefficient of variation)
                metrics["load_balance"] = max(0, 1 - (std_val / mean_val)) if mean_val > 0 else 1.0
            else:
                metrics["load_balance"] = 1.0
            metrics["spans_per_agent"] = spans_per_agent
        else:
            metrics["load_balance"] = 0.5  # Unknown
        
        # 4. Reachability Score
        if n > 1:
            reachable_pairs = 0
            total_pairs = n * (n - 1)
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        if nx.has_path(graph, source, target):
                            reachable_pairs += 1
            metrics["reachability_score"] = reachable_pairs / total_pairs if total_pairs > 0 else 0
        else:
            metrics["reachability_score"] = 1.0
        
        # 5. Information Flow Score (based on edge weights)
        total_weight = sum(d.get('weight', 1) for _, _, d in graph.edges(data=True))
        avg_weight = total_weight / graph.number_of_edges() if graph.number_of_edges() > 0 else 0
        metrics["avg_interaction_strength"] = avg_weight
        metrics["total_interactions"] = total_weight
        
        # 6. Overall MAS Collaboration Score
        metrics["overall_score"] = self._compute_overall_score(metrics)
        
        return metrics
    
    def _compute_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Compute weighted overall collaboration score."""
        weights = {
            'collaboration_density': 0.25,
            'load_balance': 0.25,
            'reachability_score': 0.25,
            # Inverse of bottleneck (lower is better)
            'bottleneck_penalty': 0.25
        }
        
        scores = [
            metrics.get('collaboration_density', 0) * weights['collaboration_density'],
            metrics.get('load_balance', 0.5) * weights['load_balance'],
            metrics.get('reachability_score', 0) * weights['reachability_score'],
            (1 - metrics.get('bottleneck_score', 0)) * weights['bottleneck_penalty']
        ]
        
        return min(1.0, max(0.0, sum(scores)))
    
    def summary(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable summary of metrics."""
        if "error" in metrics:
            return f"❌ Evaluation Error: {metrics['error']}"
        
        lines = [
            "=" * 60,
            "🔗 AGENT INTERACTION GRAPH EVALUATION",
            "=" * 60,
            "",
            f"📊 Graph Structure:",
            f"   Agents: {metrics.get('agent_count', 0)}",
            f"   Interactions: {metrics.get('interaction_count', 0)}",
            f"   Total Interaction Events: {metrics.get('total_interactions', 0):.0f}",
            "",
            f"📈 Collaboration Metrics:",
            f"   Collaboration Density: {metrics.get('collaboration_density', 0):.2%}",
            f"   Load Balance: {metrics.get('load_balance', 0):.2%}",
            f"   Reachability: {metrics.get('reachability_score', 0):.2%}",
            "",
            f"🎯 Key Agents:",
            f"   Most Central: {metrics.get('central_agent', 'N/A')} (score: {metrics.get('central_agent_score', 0):.2f})",
            f"   Potential Bottleneck: {metrics.get('bottleneck_agent', 'N/A')} (score: {metrics.get('bottleneck_score', 0):.2f})",
            "",
            "⭐ Overall Collaboration Score: {:.1%}".format(metrics.get('overall_score', 0)),
            ""
        ]
        
        # Interpretation
        score = metrics.get('overall_score', 0)
        if score >= 0.75:
            lines.append("✅ Excellent: Agents collaborate effectively with balanced workload")
        elif score >= 0.5:
            lines.append("🟡 Good: Reasonable collaboration, some room for improvement")
        elif score >= 0.25:
            lines.append("🟠 Fair: Limited collaboration, consider improving agent connectivity")
        else:
            lines.append("🔴 Poor: Agents are isolated, MAS may not be functioning as expected")
        
        return "\n".join(lines)
