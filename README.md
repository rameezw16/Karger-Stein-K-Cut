# Karger-Stein k-Cut Algorithm

A Python implementation of the Karger-Stein algorithm for finding minimum k-cuts in graphs. This implementation provides an efficient randomized algorithm for partitioning graphs into k components while minimizing the total weight of edges crossing between partitions.

## Features

- Efficient implementation of the Karger-Stein algorithm
- Support for weighted graphs
- Multiple optimization strategies:
  - Recursive contraction
  - Nagamochi-Ibaraki sparsification
  - Smart edge selection based on node degrees
- Performance logging and statistics
- Support for finding all minimum k-cuts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Karger-Stein-K-Cut.git
cd Karger-Stein-K-Cut
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Dependencies

- Python 3.8+
- NetworkX
- NumPy

## Running the Code

### Running Examples

1. Create a new Python script (e.g., `example.py`) with the following content:

```python
import networkx as nx
from src.karger_stein import KargerStein

# Create a weighted graph
G = nx.Graph()
G.add_edge(0, 1, weight=2)
G.add_edge(1, 2, weight=3)
G.add_edge(2, 0, weight=1)
G.add_edge(2, 3, weight=4)
G.add_edge(3, 0, weight=2)

# Initialize the Karger-Stein algorithm
k = 2  # Number of desired partitions
ks = KargerStein(G, k)

# Find the minimum k-cut
result = ks.find_min_k_cut()

# Print results
print(f"Minimum cut weight: {result['weight']}")
print("Partitions:")
for i, partition in enumerate(result['partitions']):
    print(f"Partition {i+1}: {partition}")
```

2. Run the script:
```bash
python example.py
```

### Running the Advanced Example

1. Create a new Python script (e.g., `advanced_example.py`) with the following content:

```python
import networkx as nx
from src.karger_stein import KargerStein

# Create a larger weighted graph
G = nx.erdos_renyi_graph(20, 0.3)
for u, v in G.edges():
    G[u][v]['weight'] = 1.0  # Assign unit weights

# Initialize with k=3 partitions
ks = KargerStein(G, k=3)

# Use recursive contraction with Nagamochi-Ibaraki sparsification
result = ks.find_min_k_cut_recursive(k=3)

# Print detailed results
print(f"Minimum cut weight: {result['weight']}")
print(f"Success rate: {result['success_rate']:.2%}")
print("All minimum cuts found:")
for i, cut in enumerate(result['all_min_cuts']):
    print(f"Cut {i+1}:")
    for j, partition in enumerate(cut):
        print(f"  Partition {j+1}: {partition}")
```

2. Run the script:
```bash
python advanced_example.py
```

### Running Tests

To run the test suite:
```bash
python -m pytest tests/
```

## Usage Examples

### Basic Example

```python
import networkx as nx
from src.karger_stein import KargerStein

# Create a weighted graph
G = nx.Graph()
G.add_edge(0, 1, weight=2)
G.add_edge(1, 2, weight=3)
G.add_edge(2, 0, weight=1)
G.add_edge(2, 3, weight=4)
G.add_edge(3, 0, weight=2)

# Initialize the Karger-Stein algorithm
k = 2  # Number of desired partitions
ks = KargerStein(G, k)

# Find the minimum k-cut
result = ks.find_min_k_cut()

# Print results
print(f"Minimum cut weight: {result['weight']}")
print("Partitions:")
for i, partition in enumerate(result['partitions']):
    print(f"Partition {i+1}: {partition}")
```

### Advanced Example with Recursive Contraction

```python
import networkx as nx
from src.karger_stein import KargerStein

# Create a larger weighted graph
G = nx.erdos_renyi_graph(20, 0.3)
for u, v in G.edges():
    G[u][v]['weight'] = 1.0  # Assign unit weights

# Initialize with k=3 partitions
ks = KargerStein(G, k=3)

# Use recursive contraction with Nagamochi-Ibaraki sparsification
result = ks.find_min_k_cut_recursive(k=3)

# Print detailed results
print(f"Minimum cut weight: {result['weight']}")
print(f"Success rate: {result['success_rate']:.2%}")
print("All minimum cuts found:")
for i, cut in enumerate(result['all_min_cuts']):
    print(f"Cut {i+1}:")
    for j, partition in enumerate(cut):
        print(f"  Partition {j+1}: {partition}")
```

## Key Methods

### `KargerStein(graph, k, seed=None)`
- `graph`: NetworkX graph (must be weighted)
- `k`: Number of desired partitions
- `seed`: Random seed for reproducibility

### `find_min_k_cut(num_trials=None)`
Finds the minimum k-cut using repeated random contractions.
- `num_trials`: Number of trials to run (default: n² log n)
- Returns a dictionary with:
  - `weight`: Minimum cut weight
  - `partitions`: Best partition found
  - `all_min_cuts`: All minimum cuts found
  - `success_rate`: Success rate of trials
  - `num_successes`: Number of successful trials
  - `num_trials`: Total number of trials

### `find_min_k_cut_recursive(k, num_trials=None)`
Finds the minimum k-cut using recursive Karger-Stein algorithm with Nagamochi-Ibaraki sparsification.
- `k`: Number of partitions
- `num_trials`: Number of trials to run (default: n² log n)
- Returns the same dictionary structure as `find_min_k_cut`

## Performance Considerations

- The algorithm's success probability is O(1/log n)
- For large graphs, consider using the recursive version with sparsification
- The number of trials can be adjusted based on the desired confidence level
- Edge weights significantly impact the algorithm's performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 