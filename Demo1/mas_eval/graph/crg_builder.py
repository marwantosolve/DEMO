"""
Causal Reasoning Graph (CRG) Module.

Builds and analyzes a Directed Acyclic Graph from agent trace data,
based on the GEMMAS framework for MAS evaluation.
"""

from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
from datetime import datetime

from mas_eval.core.base import BaseModule
from mas_eval.core.types import Span, Node, Edge, TraceData


class CRGModule(BaseModule):
    """
    Causal Reasoning Graph builder and analyzer.
    
    Converts trace spans into a NetworkX DiGraph where:
    - Nodes represent agent states (spans)
    - Edges represent causal dependencies (parent-child) or temporal order
    
    Usage:
        crg = CRGModule()
        graph = crg.build(spans)
        
        # Analyze
        stats = crg.get_statistics()
        critical_path = crg.get_critical_path()
        
        # Visualize
        crg.visualize()
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="CRGModule", config=config)
        self.graph: nx.DiGraph = nx.DiGraph()
        self._initialized = True
    
    def build(self, data: Any) -> nx.DiGraph:
        """
        Build the Causal Reasoning Graph from spans.
        
        Args:
            data: List of Span objects, TraceData, or raw span dicts
        
        Returns:
            NetworkX DiGraph representing the CRG
        """
        # Handle different input formats
        spans = self._normalize_input(data)
        
        # Reset graph
        self.graph = nx.DiGraph()
        
        # Sort by start time to ensure temporal order
        sorted_spans = sorted(spans, key=lambda x: x.start_time)
        
        # Build nodes
        for span in sorted_spans:
            self._add_node(span)
        
        # Build causal edges based on parent-child relationships
        for span in sorted_spans:
            self._add_edges(span)
        
        # Auto-add temporal edges if graph is too disconnected
        # This ensures the graph is always connected for evaluation
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        if num_nodes > 1 and num_edges < (num_nodes - 1):
            # Not fully connected - add temporal edges
            print(f"[CRG] Graph has {num_edges} edges for {num_nodes} nodes - adding temporal edges")
            self.add_temporal_edges(time_threshold_ms=float('inf'))
        
        return self.graph
    
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
    
    def _add_node(self, span: Span) -> None:
        """Add a node to the graph from a span."""
        self.graph.add_node(
            span.span_id,
            agent=span.agent_name,
            step_type=span.step_type.value,
            content=span.content,
            name=span.name,
            timestamp=span.start_time,
            duration_ms=span.duration_ms()
        )
    
    def _add_edges(self, span: Span) -> None:
        """Add edges for a span (causal + temporal)."""
        # Causal edge (parent-child relationship)
        if span.parent_span_id and self.graph.has_node(span.parent_span_id):
            self.graph.add_edge(
                span.parent_span_id,
                span.span_id,
                type="causal",
                weight=1.0
            )
    
    def add_temporal_edges(self, time_threshold_ms: float = 100.0) -> None:
        """
        Add temporal edges between sequential spans.
        
        Args:
            time_threshold_ms: Maximum time gap to consider consecutive
        """
        nodes_by_time = sorted(
            self.graph.nodes(data=True),
            key=lambda x: x[1].get('timestamp', datetime.min)
        )
        
        for i in range(len(nodes_by_time) - 1):
            node1_id, node1_data = nodes_by_time[i]
            node2_id, node2_data = nodes_by_time[i + 1]
            
            # Skip if causal edge exists
            if self.graph.has_edge(node1_id, node2_id):
                continue
            
            # Add temporal edge
            self.graph.add_edge(
                node1_id,
                node2_id,
                type="temporal",
                weight=0.5
            )
    
    def add_semantic_edges(
        self, 
        similarity_threshold: float = 0.7,
        step_types: List[str] = None,
        cross_agent_only: bool = True
    ) -> int:
        """
        Add edges between semantically similar nodes.
        
        This enhances the Graph of Thoughts (GOT) by connecting
        related thoughts across different agents.
        
        Args:
            similarity_threshold: Minimum cosine similarity for edge creation
            step_types: Only consider these step types (default: ["thought"])
            cross_agent_only: Only connect nodes from different agents
            
        Returns:
            Number of semantic edges added
        """
        if step_types is None:
            step_types = ["thought", "output"]
        
        # Collect nodes for semantic analysis
        target_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('step_type') in step_types:
                content = data.get('content', '')
                if content and len(content.strip()) > 10:
                    target_nodes.append((node_id, data))
        
        if len(target_nodes) < 2:
            return 0
        
        # Load sentence transformer
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("[CRG] Warning: sentence-transformers not available for semantic edges")
            return 0
        
        # Get embeddings for all target nodes
        contents = [data.get('content', '')[:1000] for _, data in target_nodes]
        embeddings = encoder.encode(contents)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Add edges for similar pairs
        edges_added = 0
        n = len(target_nodes)
        for i in range(n):
            for j in range(i + 1, n):
                node_i_id, node_i_data = target_nodes[i]
                node_j_id, node_j_data = target_nodes[j]
                
                # Skip if cross_agent_only and same agent
                if cross_agent_only:
                    if node_i_data.get('agent') == node_j_data.get('agent'):
                        continue
                
                # Skip if edge already exists
                if self.graph.has_edge(node_i_id, node_j_id):
                    continue
                if self.graph.has_edge(node_j_id, node_i_id):
                    continue
                
                similarity = sim_matrix[i][j]
                if similarity >= similarity_threshold:
                    self.graph.add_edge(
                        node_i_id,
                        node_j_id,
                        type="semantic",
                        weight=float(similarity),
                        similarity=float(similarity)
                    )
                    edges_added += 1
        
        return edges_added
    
    def add_information_flow_edges(self) -> int:
        """
        Add edges tracking information flow between agents.
        
        Detects when an agent's output content is referenced
        by another agent's thought/action.
        
        Returns:
            Number of information flow edges added
        """
        outputs = []
        other_nodes = []
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get('step_type') == 'output':
                outputs.append((node_id, data))
            elif data.get('step_type') in ['thought', 'action']:
                other_nodes.append((node_id, data))
        
        if not outputs or not other_nodes:
            return 0
        
        edges_added = 0
        
        for out_id, out_data in outputs:
            out_agent = out_data.get('agent', '')
            out_content = out_data.get('content', '').lower()[:500]
            out_time = out_data.get('timestamp', datetime.min)
            
            # Extract key phrases from output (simple heuristic)
            key_phrases = set()
            for word in out_content.split():
                if len(word) > 5 and word.isalpha():
                    key_phrases.add(word)
            
            if len(key_phrases) < 3:
                continue
            
            for node_id, node_data in other_nodes:
                node_agent = node_data.get('agent', '')
                node_content = node_data.get('content', '').lower()[:500]
                node_time = node_data.get('timestamp', datetime.min)
                
                # Skip same agent
                if node_agent == out_agent:
                    continue
                
                # Skip if node is before output
                if node_time <= out_time:
                    continue
                
                # Skip if edge exists
                if self.graph.has_edge(out_id, node_id):
                    continue
                
                # Check for key phrase overlap
                overlap = sum(1 for phrase in key_phrases if phrase in node_content)
                overlap_ratio = overlap / len(key_phrases) if key_phrases else 0
                
                if overlap_ratio >= 0.3:  # 30% key phrase overlap
                    self.graph.add_edge(
                        out_id,
                        node_id,
                        type="information_flow",
                        weight=0.7,
                        overlap_ratio=overlap_ratio
                    )
                    edges_added += 1
        
        return edges_added
    
    def process(self, data: Any) -> nx.DiGraph:
        """BaseModule interface."""
        return self.build(data)
    
    # ===== Analysis Methods =====
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the CRG.
        
        Returns:
            Dictionary with graph statistics
        """
        if self.graph.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_agents": len(self.get_unique_agents()),
            "agents": self.get_unique_agents(),
            "density": nx.density(self.graph),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "num_connected_components": nx.number_weakly_connected_components(self.graph),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }
    
    def get_unique_agents(self) -> List[str]:
        """Get list of unique agents in the graph."""
        agents = set()
        for _, data in self.graph.nodes(data=True):
            agent = data.get('agent', 'Unknown')
            agents.add(agent)
        return sorted(list(agents))
    
    def get_agent_interactions(self) -> Dict[Tuple[str, str], int]:
        """
        Get count of interactions between agent pairs.
        
        Returns:
            Dictionary mapping (agent1, agent2) -> interaction count
        """
        interactions = {}
        
        for source, target in self.graph.edges():
            agent1 = self.graph.nodes[source].get('agent', 'Unknown')
            agent2 = self.graph.nodes[target].get('agent', 'Unknown')
            
            if agent1 != agent2:
                key = (agent1, agent2)
                interactions[key] = interactions.get(key, 0) + 1
        
        return interactions
    
    def get_critical_path(self) -> List[str]:
        """
        Find the critical (longest) path in the graph.
        
        Returns:
            List of node IDs representing the critical path
        """
        if self.graph.number_of_nodes() == 0:
            return []
        
        try:
            # Find longest path in DAG
            longest_path = nx.dag_longest_path(self.graph)
            return longest_path
        except nx.NetworkXUnfeasible:
            # Graph has cycles, return empty
            return []
    
    def get_agent_subgraph(self, agent_name: str) -> nx.DiGraph:
        """
        Get subgraph containing only nodes from a specific agent.
        
        Args:
            agent_name: Name of agent to filter by
        
        Returns:
            Subgraph with only that agent's nodes
        """
        nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('agent') == agent_name
        ]
        return self.graph.subgraph(nodes).copy()
    
    def get_nodes_by_type(self, step_type: str) -> List[str]:
        """Get all node IDs of a specific step type."""
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get('step_type') == step_type
        ]
    
    def get_root_nodes(self) -> List[str]:
        """Get nodes with no incoming edges (root nodes)."""
        return [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
    
    def get_leaf_nodes(self) -> List[str]:
        """Get nodes with no outgoing edges (leaf nodes)."""
        return [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
    
    def get_all_paths(self, max_paths: int = 100) -> List[List[str]]:
        """
        Get all simple paths from root to leaf nodes.
        
        Args:
            max_paths: Maximum number of paths to return
        
        Returns:
            List of paths (each path is a list of node IDs)
        """
        roots = self.get_root_nodes()
        leaves = self.get_leaf_nodes()
        
        all_paths = []
        for root in roots:
            for leaf in leaves:
                try:
                    paths = list(nx.all_simple_paths(
                        self.graph, root, leaf
                    ))
                    all_paths.extend(paths[:max_paths - len(all_paths)])
                    if len(all_paths) >= max_paths:
                        break
                except nx.NetworkXNoPath:
                    continue
            if len(all_paths) >= max_paths:
                break
        
        return all_paths
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format."""
        return {
            "nodes": [
                {"id": n, **d}
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.graph.edges(data=True)
            ],
            "statistics": self.get_statistics()
        }
    
    def reset(self) -> None:
        """Reset the graph."""
        self.graph = nx.DiGraph()
