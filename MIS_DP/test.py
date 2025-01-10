import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from tqdm import tqdm
import sys

import recursive_tree
import model
import networkx as nx
import numpy as np
import time
from utils import load_benchmark_data
from torch_geometric.utils import to_networkx  # Import to_networkx

from heuristic_optimal_solvers import local_search
from heuristic_optimal_solvers import solve_mis_ilp



def test_dataset(cmp_buffer, device, D, dataset_name, optimum_mips=None, dataset_path=None):

    idx0,idx1 = get_indeces_dataset(dataset_name)
    
    list_G = load_benchmark_data(dataset_name, idxs=(idx0, idx1),dataset_path=dataset_path)

    gur_solver = pywraplp.Solver.CreateSolver('SCIP')

    list_random = []
    list_model_buffer = []
    list_min_degree = []

    list_local_10 = []

    list_ortools_1 = []
    list_ortools_5 = []
    
    list_gurobi_05 = []
    list_gurobi_1 = []
    list_gurobi_5 = []

    list_time_random = []
    list_time_buffer = []
    list_time_min_degree = []

    pbar = tqdm(enumerate(list_G))
    for i, G in pbar:
        num_nodes = G.num_nodes
        num_edges = G.edge_index.size(1)
        pbar.set_description(f'|V|={num_nodes}, |E|={num_edges}')
        
        nx_graph = to_networkx(G, to_undirected=True)  # Convert to NetworkX graph
        
        start = time.time()
        # MIS FOR RANDOM COMPARATOR
        cmp_random = model.Comparator(D, device)
        val_random = recursive_tree.find_MIS_value(cmp_random, nx.to_numpy_array(nx_graph), device)
        end = time.time()
        list_time_random.append(end - start)
        
        start = time.time()
        # MIS FOR MIN_DEGREE
        val_min_degree = recursive_tree.greedy_MIS(nx_graph)
        end = time.time()
        list_time_min_degree.append((end - start))
        
        start = time.time()
        # MIS FOR THE MODEL_BUFFER
        val_model_buffer = recursive_tree.find_MIS_value(cmp_buffer, nx.to_numpy_array(nx_graph), device)
        end = time.time()
        list_time_buffer.append((end - start))
        
        # MIS FOR LOCAL SEARCH (FIX TIME)
        nx_graph = to_networkx(G, to_undirected=True)  # Convert PyTorch Geometric Data to NetworkX graph
        local2_val = local_search(nx_graph, time_limit=10)

        
        # MIS FOR OR AND GUROBI SEARCH (FIX TIME)
        or_1, _ = solve_mis_ilp(nx_graph, time_limit_milliseconds=1000, mode='SCIP')  # Pass NetworkX graph
        or_5, _ = solve_mis_ilp(nx_graph, time_limit_milliseconds=5000, mode='SCIP')  # Pass NetworkX graph

        gur_05, _ = solve_mis_ilp(nx_graph, time_limit_milliseconds=500, solver=gur_solver)  # Pass NetworkX graph
        gur_1, _ = solve_mis_ilp(nx_graph, time_limit_milliseconds=1000, solver=gur_solver)  # Pass NetworkX graph
        gur_5, _ = solve_mis_ilp(nx_graph, time_limit_milliseconds=5000, solver=gur_solver)  # Pass NetworkX graph

        if optimum_mips is not None:

            val_random /= optimum_mips[i]
            val_min_degree /= optimum_mips[i]
            val_model_buffer /= optimum_mips[i]
            
            local1_val /= optimum_mips[i]
            local2_val /= optimum_mips[i]
            
            or_1 /= optimum_mips[i]
            or_5 /= optimum_mips[i]

            gur_05 /= optimum_mips[i]
            gur_1 /= optimum_mips[i]
            gur_5 /= optimum_mips[i]

        list_random.append(val_random)
        list_min_degree.append(val_min_degree)
        list_model_buffer.append(val_model_buffer)
        
        list_local_10.append(local2_val)
        
        list_ortools_1.append(or_1)
        list_ortools_5.append(or_5)
        
        list_gurobi_05.append(gur_05)
        list_gurobi_1.append(gur_1)
        list_gurobi_5.append(gur_5)

    print(f"Mean ratio for random: {(np.array(list_random)).mean()} +/-  {(np.array(list_random)).std()}")
    print(f"Mean ratio for min_degree: {(np.array(list_min_degree)).mean()} +/-  {(np.array(list_min_degree)).std()}")
    print(f"Mean ratio for model_buffer: {(np.array(list_model_buffer)).mean()} +/-  {(np.array(list_model_buffer)).std()}")

    print(f"Mean ratio for local (10s): {(np.array(list_local_10)).mean()} +/-  {(np.array(list_local_10)).std()}")

    print(f"Mean ratio for or_1: {(np.array(list_ortools_1)).mean()} +/-  {(np.array(list_ortools_1)).std()}")
    print(f"Mean ratio for or_5: {(np.array(list_ortools_5)).mean()} +/-  {(np.array(list_ortools_5)).std()}")

    print(f"Mean ratio for gur_05: {(np.array(list_gurobi_05)).mean()} +/-  {(np.array(list_gurobi_05)).std()}")
    print(f"Mean ratio for gur_1: {(np.array(list_gurobi_1)).mean()} +/-  {(np.array(list_gurobi_1)).std()}")
    print(f"Mean ratio for gur_5: {(np.array(list_gurobi_5)).mean()} +/-  {(np.array(list_gurobi_5)).std()}")

    print('random: (' + str(np.array(list_time_random).mean()) + 's/g)')
    print('min_degree: (' + str(np.array(list_time_min_degree).mean()) + 's/g)')
    print('buffer: (' + str(np.array(list_time_buffer).mean()) + 's/g)')

    
def get_indeces_dataset(dataset_name):
    if dataset_name == 'COLLAB':
        idx0 = 4000
        idx1 = 5000
    elif dataset_name == 'TWITTER':
        idx0 = 778
        idx1 = 973
    elif dataset_name == 'RB':
        idx0 = 1600
        idx1 = 2000
    elif dataset_name == 'SPECIAL':
        idx0 = 160
        idx1 = 200
    elif dataset_name == "FACEBOOK":
        idx0 = 0
        idx1 = 9
    else:
        sys.exit('The provided dataset_name is not allowed')

    return idx0,idx1
        
