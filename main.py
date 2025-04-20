#!/usr/bin/env python3
import argparse
import sys
from src.graph_builder import GraphBuilder
from src.karger_stein import KargerStein
from src.utils import load_graph_from_file

def main():
    parser = argparse.ArgumentParser(description='Karger-Stein k-Cut Algorithm')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input graph file')
    parser.add_argument('--k', type=int, required=True,
                      help='Number of partitions (k)')
    parser.add_argument('--sparsify', action='store_true',
                      help='Enable graph sparsification')
    parser.add_argument('--recursive', action='store_true',
                      help='Use recursive variant')
    parser.add_argument('--iterations', type=int, default=100,
                      help='Number of iterations (default: 100)')
    
    args = parser.parse_args()
    
    try:
        # Load and build graph
        graph = load_graph_from_file(args.input)
        graph_builder = GraphBuilder(graph)
        if args.sparsify:
            graph_builder.sparsify()
        
        # Initialize and run algorithm
        karger_stein = KargerStein(graph_builder.graph)
        if args.recursive:
            min_cut = karger_stein.find_min_k_cut_recursive(args.k, args.iterations)
        else:
            min_cut = karger_stein.find_min_k_cut(args.k, args.iterations)
        
        # Output results
        print(f"Minimum {args.k}-cut found: {min_cut['weight']}")
        print(f"Partitions: {min_cut['partitions']}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 