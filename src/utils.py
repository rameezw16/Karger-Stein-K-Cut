import networkx as nx
import json
from typing import Dict, List, Tuple, Union
import random

def contract_edge(graph: nx.Graph, edge: Tuple[int, int]) -> None:
    """
    Contract an edge in the graph by merging its endpoints.
    
    Args:
        graph: NetworkX graph
        edge: Tuple representing the edge to contract
    """
    u, v = edge
    if u == v:
        return
    
    # Merge edges from v to u
    for neighbor, data in list(graph[v].items()):
        if neighbor != u:
            if graph.has_edge(u, neighbor):
                graph[u][neighbor]['weight'] += data['weight']
            else:
                graph.add_edge(u, neighbor, weight=data['weight'])
    
    # Remove the contracted vertex
    graph.remove_node(v)

def load_graph_from_file(file_path: str) -> nx.Graph:
    """
    Load a graph from a file.
    
    Args:
        file_path: Path to the graph file
        
    Returns:
        NetworkX graph object
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    graph = nx.Graph()
    for edge in data['edges']:
        graph.add_edge(edge['u'], edge['v'], weight=edge['weight'])
    
    return graph

def save_graph_to_file(graph: nx.Graph, file_path: str) -> None:
    """
    Save a graph to a file.
    
    Args:
        graph: NetworkX graph to save
        file_path: Path where to save the graph
    """
    data = {
        'edges': [
            {
                'u': u,
                'v': v,
                'weight': data['weight']
            }
            for u, v, data in graph.edges(data=True)
        ]
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_random_graph(n: int, p: float, weight_range: Tuple[float, float] = (1.0, 10.0)) -> nx.Graph:
    """
    Generate a random weighted graph.
    
    Args:
        n: Number of vertices
        p: Probability of edge creation
        weight_range: Tuple of (min_weight, max_weight)
        
    Returns:
        Randomly generated NetworkX graph
    """
    graph = nx.erdos_renyi_graph(n, p)
    for u, v in graph.edges():
        weight = random.uniform(weight_range[0], weight_range[1])
        graph[u][v]['weight'] = weight
    return graph 