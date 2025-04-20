import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Union
import random

class GraphBuilder:
    def __init__(self, graph: Union[nx.Graph, Dict[Tuple[int, int], float]]):
        """
        Initialize the graph builder with either a NetworkX graph or an edge dictionary.
        
        Args:
            graph: Input graph as NetworkX graph or dictionary of edges with weights
        """
        if isinstance(graph, nx.Graph):
            self.graph = graph
        else:
            self.graph = nx.Graph()
            for (u, v), weight in graph.items():
                self.graph.add_edge(u, v, weight=weight)
    
    def sparsify(self, epsilon: float = 0.1) -> None:
        """
        Sparsify the graph using the Benczur-Karger sparsification algorithm.
        
        Args:
            epsilon: Approximation parameter (default: 0.1)
        """
        # Calculate edge strengths (approximate using random walks)
        edge_strengths = {}
        for u, v in self.graph.edges():
            edge_strengths[(u, v)] = self._estimate_edge_strength(u, v)
        
        # Calculate sampling probabilities
        total_strength = sum(edge_strengths.values())
        sampling_probabilities = {
            edge: min(1, (8 * np.log(self.graph.number_of_nodes()) / (epsilon**2)) * strength / total_strength)
            for edge, strength in edge_strengths.items()
        }
        
        # Create new graph with sampled edges
        new_graph = nx.Graph()
        for (u, v), prob in sampling_probabilities.items():
            if random.random() < prob * 0.5:  # Reduce probability to ensure sparsification
                weight = self.graph[u][v]['weight'] / prob
                new_graph.add_edge(u, v, weight=weight)
        
        # Ensure the graph remains connected
        if not nx.is_connected(new_graph):
            # Add back some edges to maintain connectivity
            components = list(nx.connected_components(new_graph))
            while len(components) > 1:
                # Connect two components with the lightest edge
                min_edge = None
                min_weight = float('inf')
                for u in components[0]:
                    for v in components[1]:
                        if self.graph.has_edge(u, v):
                            weight = self.graph[u][v]['weight']
                            if weight < min_weight:
                                min_weight = weight
                                min_edge = (u, v)
                if min_edge:
                    new_graph.add_edge(min_edge[0], min_edge[1], weight=min_weight)
                components = list(nx.connected_components(new_graph))
        
        self.graph = new_graph
    
    def _estimate_edge_strength(self, u: int, v: int, num_walks: int = 100) -> float:
        """
        Estimate the strength of an edge using random walks.
        
        Args:
            u: First vertex
            v: Second vertex
            num_walks: Number of random walks to perform
            
        Returns:
            Estimated edge strength
        """
        strength = 0
        for _ in range(num_walks):
            # Perform random walk from u
            current = u
            visited = {current}
            while True:
                neighbors = list(self.graph.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                if current == v:
                    strength += 1
                    break
                if current in visited:
                    break
                visited.add(current)
        return strength / num_walks
    
    def get_graph(self) -> nx.Graph:
        """
        Return the current graph.
        
        Returns:
            NetworkX graph object
        """
        return self.graph
    
    def get_edge_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Get all edge weights as a dictionary.
        
        Returns:
            Dictionary mapping edges to their weights
        """
        return {(u, v): data['weight'] for u, v, data in self.graph.edges(data=True)}
    
    def get_vertex_count(self) -> int:
        """
        Get the number of vertices in the graph.
        
        Returns:
            Number of vertices
        """
        return self.graph.number_of_nodes()
    
    def get_edge_count(self) -> int:
        """
        Get the number of edges in the graph.
        
        Returns:
            Number of edges
        """
        return self.graph.number_of_edges() 