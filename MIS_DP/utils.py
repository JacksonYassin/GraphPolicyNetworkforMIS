import os
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch
from tqdm import tqdm
import networkx as nx



def load_facebook_graph(directory):
    """
    Loads a graph from the Facebook dataset stored in a directory.

    Parameters:
        directory (str): Path to the directory containing the graph files.

    Returns:
        Data: A PyTorch Geometric Data object for the graph.
    """
    edge_file = None
    feat_file = None
    egofeat_file = None

    # Find the necessary files in the directory
    for file in os.listdir(directory):
        if file.endswith('.edges'):
            edge_file = os.path.join(directory, file)
        elif file.endswith('.feat'):
            feat_file = os.path.join(directory, file)
        elif file.endswith('.egofeat'):
            egofeat_file = os.path.join(directory, file)

    # Check if all required files are found
    if not edge_file or not feat_file or not egofeat_file:
        raise ValueError(f"Missing required files in directory: {directory}")

    # Load edges
    edges = []
    with open(edge_file, 'r') as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            edges.append((src, dst))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Load features for other nodes
    node_features = []
    with open(feat_file, 'r') as f:
        for line in f:
            node_features.append(list(map(float, line.strip().split())))
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Load ego features (for the central node)
    with open(egofeat_file, 'r') as f:
        ego_features = torch.tensor([list(map(float, f.readline().strip().split()))], dtype=torch.float)

    # Pad dimensions if necessary
    max_dim = max(ego_features.size(1), node_features.size(1))
    if ego_features.size(1) < max_dim:
        ego_features = torch.cat([ego_features, torch.zeros((1, max_dim - ego_features.size(1)))], dim=1)
    if node_features.size(1) < max_dim:
        padding = torch.zeros((node_features.size(0), max_dim - node_features.size(1)))
        node_features = torch.cat([node_features, padding], dim=1)

    # Combine ego features and node features
    features = torch.cat([ego_features, node_features], dim=0)

    # Create PyTorch Geometric Data object
    data = Data(x=features, edge_index=edge_index)
    return data



def load_benchmark_data(dataset_name, dataset_path=None, idxs=(0, 100)):
    if dataset_name == "FACEBOOK":
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Expected a directory for dataset_path, got: {dataset_path}")

        # Process each subdirectory (representing a graph)
        graph_dirs = sorted([os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
                             if os.path.isdir(os.path.join(dataset_path, d))])
        graph_dirs = graph_dirs[idxs[0]:idxs[1]]

        list_G_big = []
        for graph_dir in tqdm(graph_dirs):
            graph_data = load_facebook_graph(graph_dir)
            list_G_big.append(graph_data)
        return list_G_big
    else:
        raise Exception(f"The provided dataset_name '{dataset_name}' is not allowed")


def get_path_from_dataset_name(dataset_name):
    """
    Function to get the path from the dataset name or use a custom folder.

    Parameters:
        dataset_name (str): Name of the dataset or custom path.

    Returns:
        str: Path to the dataset or custom folder.
    """
    path = os.path.dirname(os.path.realpath(__file__))

    # Predefined paths
    collab_path = os.path.join(path, 'dataset_buffer', 'collab_graphs.pickle')
    twitter_path = os.path.join(path, 'dataset_buffer', 'TWITTER_SNAP_2.p')
    special_path = os.path.join(path, 'dataset_buffer', 'special_graphs.pickle')
    rb_path = os.path.join(path, 'dataset_buffer', 'rb_graphs.pickle')
    facebook_path = os.path.join(path, 'Facebook')

    # Map dataset names to paths
    predefined_datasets = {
        'TWITTER_SNAP': twitter_path,
        'SPECIAL': special_path,
        'RB': rb_path,
        'COLLAB': collab_path,
        'FACEBOOK': facebook_path
    }

    # Check if the dataset_name matches a predefined dataset
    if dataset_name in predefined_datasets:
        dataset_path = predefined_datasets[dataset_name]
    else:
        # Assume dataset_name is a custom path
        dataset_path = os.path.abspath(dataset_name)

        # Validate if the custom path exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The specified path does not exist: {dataset_path}")

    return dataset_path
