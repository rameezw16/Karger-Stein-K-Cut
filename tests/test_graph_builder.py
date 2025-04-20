import pytest
import networkx as nx
from src.graph_builder import GraphBuilder

def test_graph_builder_initialization():
    # Test initialization with NetworkX graph
    nx_graph = nx.Graph()
    nx_graph.add_edge(0, 1, weight=2.0)
    nx_graph.add_edge(1, 2, weight=3.0)
    
    builder = GraphBuilder(nx_graph)
    assert builder.get_vertex_count() == 3
    assert builder.get_edge_count() == 2
    
    # Test initialization with edge dictionary
    edge_dict = {(0, 1): 2.0, (1, 2): 3.0}
    builder = GraphBuilder(edge_dict)
    assert builder.get_vertex_count() == 3
    assert builder.get_edge_count() == 2

def test_get_edge_weights():
    nx_graph = nx.Graph()
    nx_graph.add_edge(0, 1, weight=2.0)
    nx_graph.add_edge(1, 2, weight=3.0)
    
    builder = GraphBuilder(nx_graph)
    weights = builder.get_edge_weights()
    
    assert weights == {(0, 1): 2.0, (1, 2): 3.0}

def test_sparsify():
    # Create a dense graph
    nx_graph = nx.complete_graph(5)
    for u, v in nx_graph.edges():
        nx_graph[u][v]['weight'] = 1.0
    
    builder = GraphBuilder(nx_graph)
    builder.sparsify(epsilon=0.1)
    
    # After sparsification, the graph should have fewer edges
    assert builder.get_edge_count() < nx_graph.number_of_edges() 