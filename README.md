# Karger-Stein k-Cut Algorithm Implementation

This project implements the Karger-Stein algorithm for solving the minimum k-Cut problem on weighted undirected graphs, based on the 2020 paper "The Karger-Stein Algorithm Is Optimal for k-Cut."

## Project Structure

```
karger_kcut_project/
├── src/                         # Core source code
├── data/                        # Input graph files
├── tests/                       # Unit tests
├── results/                     # Experiment results
└── notebooks/                   # Jupyter notebooks
```

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main entry point is `main.py`. Run it with:
```bash
python main.py --input data/example_graph.txt --k 3
```

## Features

- Implementation of the Karger-Stein algorithm for minimum k-Cut
- Graph sparsification support
- Recursive variant implementation
- Unit tests and benchmarks
- Visualization tools

## Dependencies

- Python 3.8+
- networkx
- matplotlib
- numpy
- pytest (for testing)

## License

MIT License 