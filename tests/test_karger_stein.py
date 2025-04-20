import pytest
import networkx as nx
from src.karger_stein import KargerStein
from src.graph_builder import GraphBuilder

def test_karger_stein_initialization():
    # Create a simple graph
    nx_graph = nx.Graph()
    nx_graph.add_edge(0, 1, weight=2.0)
    nx_graph.add_edge(1, 2, weight=3.0)
    
    karger_stein = KargerStein(nx_graph)
    assert karger_stein.graph.number_of_nodes() == 3
    assert karger_stein.graph.number_of_edges() == 2

def test_find_min_k_cut():
    # Create a graph with a known minimum 2-cut
    nx_graph = nx.Graph()
    nx_graph.add_edge(0, 1, weight=1.0)
    nx_graph.add_edge(1, 2, weight=1.0)
    nx_graph.add_edge(2, 3, weight=1.0)
    nx_graph.add_edge(3, 0, weight=1.0)
    nx_graph.add_edge(0, 2, weight=10.0)  # This edge should be in the minimum cut
    
    karger_stein = KargerStein(nx_graph)
    result = karger_stein.find_min_k_cut(k=2, iterations=100)
    
    assert result['weight'] == 10.0  # The minimum cut should be 10.0
    assert len(result['partitions']) == 2  # Should have 2 partitions

def test_find_min_k_cut_recursive():
    # Create a graph with a known minimum 3-cut
    nx_graph = nx.Graph()
    nx_graph.add_edge(0, 1, weight=1.0)
    nx_graph.add_edge(1, 2, weight=1.0)
    nx_graph.add_edge(2, 3, weight=1.0)
    nx_graph.add_edge(3, 0, weight=1.0)
    nx_graph.add_edge(0, 2, weight=10.0)
    nx_graph.add_edge(1, 3, weight=10.0)
    
    karger_stein = KargerStein(nx_graph)
    result = karger_stein.find_min_k_cut_recursive(k=3, iterations=100)
    
    assert len(result['partitions']) == 3  # Should have 3 partitions
    assert result['weight'] > 0  # Cut weight should be positive

def test_edge_selection():
    # Create a graph with weighted edges
    nx_graph = nx.Graph()
    nx_graph.add_edge(0, 1, weight=1.0)
    nx_graph.add_edge(1, 2, weight=2.0)
    nx_graph.add_edge(2, 0, weight=3.0)
    
    karger_stein = KargerStein(nx_graph)
    edge = karger_stein._select_random_edge(nx_graph)
    
    assert len(edge) == 2  # Edge should be a tuple of 2 vertices
    assert edge[0] in nx_graph.nodes()
    assert edge[1] in nx_graph.nodes() 