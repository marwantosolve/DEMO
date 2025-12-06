"""
Graph Visualizer for Causal Reasoning Graphs.

Provides various visualization options for CRG analysis.
"""

from typing import Dict, Any, Optional, List
import networkx as nx


class GraphVisualizer:
    """
    Visualizer for Causal Reasoning Graphs.
    
    Supports multiple visualization backends and styles.
    """
    
    # Agent color palette
    AGENT_COLORS = {
        'Orchestrator': '#FF6B6B',
        'Researcher': '#4ECDC4',
        'Writer': '#96CEB4',
        'Critic': '#FFEAA7',
        'QualityChecker': '#DDA0DD',
        'default': '#CCCCCC'
    }
    
    # Step type colors
    STEP_COLORS = {
        'thought': '#74B9FF',
        'action': '#55EFC4',
        'observation': '#FDCB6E',
        'output': '#A29BFE',
        'error': '#FF7675',
        'default': '#DFE6E9'
    }
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize visualizer with a graph.
        
        Args:
            graph: NetworkX DiGraph to visualize
        """
        self.graph = graph
    
    def plot(
        self,
        figsize: tuple = (14, 10),
        color_by: str = "agent",
        layout: str = "shell",
        show_labels: bool = True,
        output_path: Optional[str] = None,
        title: str = "Causal Reasoning Graph"
    ):
        """
        Plot the graph using matplotlib.
        
        Args:
            figsize: Figure size tuple
            color_by: "agent" or "step_type"
            layout: Layout algorithm ("shell", "spring", "kamada_kawai", "hierarchical")
            show_labels: Whether to show node labels
            output_path: If provided, save to this path
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return
        
        if self.graph.number_of_nodes() == 0:
            print("⚠️ Empty graph, nothing to visualize")
            return
        
        plt.figure(figsize=figsize)
        
        # Get layout
        pos = self._get_layout(layout)
        
        # Get colors
        node_colors = self._get_node_colors(color_by)
        
        # Get labels
        labels = self._get_labels() if show_labels else {}
        
        # Draw graph
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=2000,
            labels=labels,
            font_size=8,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='#888888',
            width=2,
            alpha=0.9
        )
        
        # Add title and legend
        plt.title(title, fontsize=14, fontweight='bold')
        self._add_legend(color_by)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"📊 Graph saved to {output_path}")
        
        plt.show()
        
        # Print statistics
        self._print_stats()
    
    def _get_layout(self, layout: str) -> Dict[str, tuple]:
        """Get node positions based on layout algorithm."""
        if layout == "shell":
            return nx.shell_layout(self.graph)
        elif layout == "spring":
            return nx.spring_layout(self.graph, k=2, iterations=50)
        elif layout == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(self.graph)
            except:
                return nx.spring_layout(self.graph)
        elif layout == "hierarchical":
            try:
                return self._hierarchical_layout()
            except:
                return nx.shell_layout(self.graph)
        else:
            return nx.spring_layout(self.graph)
    
    def _hierarchical_layout(self) -> Dict[str, tuple]:
        """Create a hierarchical layout for DAGs."""
        # Use topological generations for Y position
        try:
            generations = list(nx.topological_generations(self.graph))
        except:
            return nx.spring_layout(self.graph)
        
        pos = {}
        for gen_idx, generation in enumerate(generations):
            for node_idx, node in enumerate(generation):
                x = node_idx - len(generation) / 2
                y = -gen_idx  # Negative to have root at top
                pos[node] = (x, y)
        
        return pos
    
    def _get_node_colors(self, color_by: str) -> List[str]:
        """Get node colors based on coloring strategy."""
        colors = []
        
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            
            if color_by == "agent":
                agent = data.get('agent', 'Unknown')
                # Check for partial matches
                color = self.AGENT_COLORS.get('default')
                for key, val in self.AGENT_COLORS.items():
                    if key.lower() in agent.lower():
                        color = val
                        break
                colors.append(color)
            
            elif color_by == "step_type":
                step_type = data.get('step_type', 'default')
                colors.append(self.STEP_COLORS.get(step_type, self.STEP_COLORS['default']))
            
            else:
                colors.append('#CCCCCC')
        
        return colors
    
    def _get_labels(self) -> Dict[str, str]:
        """Get node labels."""
        labels = {}
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            agent = data.get('agent', 'Unknown')
            step_type = data.get('step_type', '')
            labels[node] = f"{agent}\n({step_type})"
        return labels
    
    def _add_legend(self, color_by: str):
        """Add a legend to the plot."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        if color_by == "agent":
            colors = self.AGENT_COLORS
        else:
            colors = self.STEP_COLORS
        
        patches = [
            mpatches.Patch(color=color, label=name)
            for name, color in colors.items()
            if name != 'default'
        ]
        
        plt.legend(handles=patches, loc='upper left', fontsize=8)
    
    def _print_stats(self):
        """Print graph statistics."""
        print(f"\n📊 Graph Statistics:")
        print(f"   Nodes: {self.graph.number_of_nodes()}")
        print(f"   Edges: {self.graph.number_of_edges()}")
    
    def to_mermaid(self) -> str:
        """
        Export graph to Mermaid diagram format.
        
        Returns:
            Mermaid diagram string
        """
        lines = ["graph TD"]
        
        # Add nodes with styling
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            agent = data.get('agent', 'Unknown')
            step_type = data.get('step_type', '')
            short_id = node[:8]
            lines.append(f'    {short_id}["{agent}<br/>({step_type})"]')
        
        # Add edges
        for source, target in self.graph.edges():
            lines.append(f'    {source[:8]} --> {target[:8]}')
        
        return "\n".join(lines)
    
    def to_html(self, output_path: str):
        """
        Export graph to interactive HTML using pyvis.
        
        Args:
            output_path: Path to save HTML file
        """
        try:
            from pyvis.network import Network
        except ImportError:
            print("pyvis not installed. Run: pip install pyvis")
            return
        
        net = Network(height="600px", width="100%", directed=True)
        
        # Add nodes
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            agent = data.get('agent', 'Unknown')
            step_type = data.get('step_type', '')
            
            color = self.AGENT_COLORS.get('default')
            for key, val in self.AGENT_COLORS.items():
                if key.lower() in agent.lower():
                    color = val
                    break
            
            net.add_node(
                node,
                label=f"{agent}\n({step_type})",
                color=color,
                title=data.get('content', '')[:200]
            )
        
        # Add edges
        for source, target in self.graph.edges():
            net.add_edge(source, target)
        
        net.save_graph(output_path)
        print(f"📊 Interactive graph saved to {output_path}")
