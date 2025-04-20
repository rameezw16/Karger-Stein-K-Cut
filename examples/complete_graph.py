import networkx as nx
from src.karger_stein import KargerStein
import time
import matplotlib.pyplot as plt
import numpy as np

def print_step(step_num: int, description: str):
    """Helper function to print step information"""
    print(f"\n{'='*50}")
    print(f"Step {step_num}: {description}")
    print(f"{'='*50}")

def visualize_graph(graph: nx.Graph, title: str, filename: str = None):
    """Visualize the graph with edge weights."""
    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(graph)  # Use circular layout for complete graphs
    edge_labels = {(u, v): f"{data['weight']:.1f}" for u, v, data in graph.edges(data=True)}
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue')
    
    # Draw edges with weights
    nx.draw_networkx_edges(graph, pos, width=2)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')
    
    plt.title(title)
    plt.axis('off')
    if filename:
        plt.savefig(filename)
    plt.show()

def visualize_partitions(graph: nx.Graph, partitions: list, title: str, filename: str = None):
    """Visualize the graph with partitions highlighted."""
    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(graph)  # Use circular layout for complete graphs
    
    # Define colors for partitions
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    
    # Draw nodes with partition colors
    for i, partition in enumerate(partitions):
        nx.draw_networkx_nodes(graph, pos, nodelist=list(partition),
                             node_color=colors[i % len(colors)],
                             node_size=500, label=f'Partition {i+1}')
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, width=2)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    if filename:
        plt.savefig(filename)
    plt.show()

def visualize_cut_edges(graph: nx.Graph, partitions: list, title: str, filename: str = None):
    """Visualize the graph with cut edges highlighted."""
    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(graph)  # Use circular layout for complete graphs
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightgray')
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(graph, pos, width=1, edge_color='lightgray')
    
    # Draw cut edges in red
    cut_edges = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            for u in partitions[i]:
                for v in partitions[j]:
                    if graph.has_edge(u, v):
                        cut_edges.append((u, v))
    
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, width=3, edge_color='red')
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')
    
    plt.title(title)
    plt.axis('off')
    if filename:
        plt.savefig(filename)
    plt.show()

def main():
    # Step 1: Create a complete weighted graph
    print_step(1, "Creating a complete weighted graph")
    n = 8  # Number of nodes
    G = nx.complete_graph(n)
    
    # Add random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = round(np.random.uniform(1, 10), 1)
    
    print("Graph created with the following properties:")
    print(f"  Number of nodes: {G.number_of_nodes()}")
    print(f"  Number of edges: {G.number_of_edges()}")
    print("\nEdge weights:")
    for u, v, data in G.edges(data=True):
        print(f"  Edge ({u}, {v}): weight = {data['weight']}")

    # Visualize initial graph
    visualize_graph(G, "Complete Graph", "complete_graph.png")

    # Step 2: Initialize the Karger-Stein algorithm
    print_step(2, "Initializing Karger-Stein algorithm")
    k = 3  # Number of desired partitions
    print(f"Setting k = {k} (number of partitions)")
    ks = KargerStein(G, k)
    print("Algorithm initialized with:")
    print(f"  - Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  - Target partitions: {k}")
    print(f"  - Random seed: {ks.seed}")

    # Step 3: Estimate λₖ
    print_step(3, "Estimating λₖ (minimum k-cut weight)")
    start_time = time.time()
    lambda_k = ks._estimate_lambda_k_from_trials(num_samples=5)
    print(f"λₖ estimation completed in {time.time() - start_time:.4f} seconds")
    print(f"Estimated λₖ = {lambda_k}")

    # Step 4: Find minimum k-cut
    print_step(4, "Finding minimum k-cut")
    start_time = time.time()
    result = ks.find_min_k_cut()
    runtime = time.time() - start_time
    
    print("\nResults:")
    print(f"  Minimum cut weight: {result['weight']}")
    print(f"  Runtime: {runtime:.4f} seconds")
    print(f"  Success rate: {result['success_rate']:.2%}")
    print(f"  Number of trials: {result['num_trials']}")
    print(f"  Successful trials: {result['num_successes']}")

    # Step 5: Analyze partitions
    print_step(5, "Analyzing partitions")
    print("\nBest partition found:")
    for i, partition in enumerate(result['partitions']):
        print(f"  Partition {i+1}: {sorted(partition)}")
        print(f"    Size: {len(partition)} nodes")
        print(f"    Nodes: {sorted(partition)}")

    # Visualize partitions
    visualize_partitions(G, result['partitions'], "Complete Graph Partitions", "complete_partitions.png")

    # Step 6: Calculate cut edges
    print_step(6, "Calculating cut edges")
    cut_edges = []
    for i in range(len(result['partitions'])):
        for j in range(i + 1, len(result['partitions'])):
            for u in result['partitions'][i]:
                for v in result['partitions'][j]:
                    if G.has_edge(u, v):
                        weight = G[u][v]['weight']
                        cut_edges.append((u, v, weight))
                        print(f"  Cut edge: ({u}, {v}) with weight {weight}")

    print(f"\nTotal cut weight: {sum(w for _, _, w in cut_edges)}")
    print(f"Number of edges crossing partitions: {len(cut_edges)}")

    # Visualize cut edges
    visualize_cut_edges(G, result['partitions'], "Complete Graph Cut Edges", "complete_cut_edges.png")

    # Step 7: Verify all minimum cuts
    print_step(7, "Verifying all minimum cuts")
    print(f"\nFound {len(result['all_min_cuts'])} distinct minimum cuts:")
    for i, cut in enumerate(result['all_min_cuts']):
        print(f"\nCut {i+1}:")
        for j, partition in enumerate(cut):
            print(f"  Partition {j+1}: {sorted(partition)}")
        
        # Visualize each alternative cut
        visualize_partitions(G, cut, f"Complete Graph Alternative Cut {i+1}", f"complete_alternative_cut_{i+1}.png")

if __name__ == "__main__":
    main() 