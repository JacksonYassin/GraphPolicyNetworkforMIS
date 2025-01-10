import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from tqdm import tqdm
import sys
import torch
import os
import argparse

import recursive_tree
import model
import networkx as nx
import numpy as np
import time
from utils import load_benchmark_data
from train import load  # Import the load function to load the model parameters

from heuristic_optimal_solvers import local_search
from heuristic_optimal_solvers import solve_mis_ilp

def test_graph(cmp_buffer, device, graph_file):
    # Read the graph from the graph_file
    G = nx.read_edgelist(graph_file, nodetype=int)
    adj_matrix = nx.to_numpy_array(G)

    # Compute the maximum independent set value using the model
    val_model_buffer = recursive_tree.find_MIS_value(cmp_buffer, adj_matrix, device)

    # Output the maximum independent set size
    print(val_model_buffer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', type=str, required=True, help='Path to the graph.txt file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing the model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model file')
    parser.add_argument('--D', default=32, type=int, required=False, help='Dimension D of the model')
    parser.add_argument('--gnn_depth', default=3, type=int, required=False, help='Number of GNN layers')
    parser.add_argument('--dense_depth', default=4, type=int, required=False, help='Number of dense layers')
    args = parser.parse_args()

    # Set up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize and load the trained model
    cmp_buffer = model.Comparator(args.D, device, num_dense_layers=args.dense_depth, num_gnn_layers=args.gnn_depth)
    model_full_path = os.path.join(args.model_path, args.model_name)
    load(cmp_buffer, model_full_path, device)
    cmp_buffer.eval()

    # Compute and print the maximum independent set value
    test_graph(cmp_buffer, device, args.graph_file)
