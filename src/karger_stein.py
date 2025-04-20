import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple
import random
from .utils import contract_edge

class KargerStein:
    def __init__(self, graph: nx.Graph):
        """
        Initialize the Karger-Stein algorithm with a graph.
        
        Args:
            graph: Input graph as NetworkX graph
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
    
    def find_min_k_cut(self, k: int, iterations: int = 100) -> Dict:
        """
        Find the minimum k-cut using the Karger-Stein algorithm.
        
        Args:
            k: Number of partitions
            iterations: Number of iterations to run (default: 100)
            
        Returns:
            Dictionary containing the cut weight and partitions
        """
        min_cut = float('inf')
        best_partitions = None
        
        for _ in range(iterations):
            self.graph = self.original_graph.copy()
            partitions = self._find_k_cut(k)
            cut_weight = self._calculate_cut_weight(partitions)
            
            if cut_weight < min_cut:
                min_cut = cut_weight
                best_partitions = partitions
        
        return {
            'weight': min_cut,
            'partitions': best_partitions
        }
    
    def find_min_k_cut_recursive(self, k: int, iterations: int = 100) -> Dict:
        """
        Find the minimum k-cut using the recursive Karger-Stein algorithm.
        
        Args:
            k: Number of partitions
            iterations: Number of iterations to run (default: 100)
            
        Returns:
            Dictionary containing the cut weight and partitions
        """
        min_cut = float('inf')
        best_partitions = None
        
        for _ in range(iterations):
            self.graph = self.original_graph.copy()
            partitions = self._karger_stein_recursive(k)
            cut_weight = self._calculate_cut_weight(partitions)
            
            if cut_weight < min_cut:
                min_cut = cut_weight
                best_partitions = partitions
        
        return {
            'weight': min_cut,
            'partitions': best_partitions
        }
    
    def _find_k_cut(self, k: int) -> List[Set]:
        """
        Find a k-cut by contracting edges until k vertices remain.
        
        Args:
            k: Number of partitions
            
        Returns:
            List of sets representing the partitions
        """
        # Create a copy of the graph for contraction
        contracted_graph = self.graph.copy()
        
        # Initialize vertex mapping
        vertex_map = {v: {v} for v in contracted_graph.nodes()}
        
        # Contract edges until we have k vertices
        while contracted_graph.number_of_nodes() > k:
            edge = self._select_random_edge(contracted_graph)
            u, v = edge
            # Update vertex mapping
            vertex_map[u].update(vertex_map[v])
            del vertex_map[v]
            # Contract edge
            contract_edge(contracted_graph, edge)
        
        # Return the partitions
        return list(vertex_map.values())
    
    def _karger_stein_recursive(self, k: int) -> List[Set]:
        """
        Recursive implementation of the Karger-Stein algorithm.
        
        Args:
            k: Number of partitions
            
        Returns:
            List of sets representing the partitions
        """
        if k == 2:
            return self._find_min_cut()
        
        # Create a copy of the graph for contraction
        contracted_graph = self.graph.copy()
        
        # Initialize vertex mapping
        vertex_map = {v: {v} for v in contracted_graph.nodes()}
        
        # Contract edges until we have k vertices
        while contracted_graph.number_of_nodes() > k:
            edge = self._select_random_edge(contracted_graph)
            u, v = edge
            # Update vertex mapping
            vertex_map[u].update(vertex_map[v])
            del vertex_map[v]
            # Contract edge
            contract_edge(contracted_graph, edge)
        
        # Return the partitions
        return list(vertex_map.values())
    
    def _find_min_cut(self) -> List[Set]:
        """
        Find the minimum cut of the graph.
        
        Returns:
            List of two sets representing the partitions
        """
        # Create a copy of the graph for contraction
        contracted_graph = self.graph.copy()
        
        # Initialize vertex mapping
        vertex_map = {v: {v} for v in contracted_graph.nodes()}
        
        # Contract edges until we have 2 vertices
        while contracted_graph.number_of_nodes() > 2:
            edge = self._select_random_edge(contracted_graph)
            u, v = edge
            # Update vertex mapping
            vertex_map[u].update(vertex_map[v])
            del vertex_map[v]
            # Contract edge
            contract_edge(contracted_graph, edge)
        
        # Return the partitions
        return list(vertex_map.values())
    
    def _select_random_edge(self, graph: nx.Graph) -> Tuple[int, int]:
        """
        Select a random edge with probability proportional to its weight.
        
        Args:
            graph: Graph to select edge from
            
        Returns:
            Tuple representing the selected edge
        """
        edges = list(graph.edges(data=True))
        weights = [data['weight'] for _, _, data in edges]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        return edges[np.random.choice(len(edges), p=probabilities)][:2]
    
    def _calculate_cut_weight(self, partitions: List[Set]) -> float:
        """
        Calculate the total weight of edges crossing the partitions.
        
        Args:
            partitions: List of sets representing the partitions
            
        Returns:
            Total weight of the cut
        """
        cut_weight = 0
        # Create a mapping of vertices to their partition index
        vertex_to_partition = {}
        for i, partition in enumerate(partitions):
            for vertex in partition:
                vertex_to_partition[vertex] = i
        
        # Calculate the total weight of edges crossing partitions
        for u, v, data in self.original_graph.edges(data=True):
            if vertex_to_partition[u] != vertex_to_partition[v]:
                cut_weight += data['weight']
        
        return cut_weight 