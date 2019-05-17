import numpy as np
import networkx as nx
import scipy.sparse as sp
import itertools
from joblib import Parallel, delayed


def add_features(graph, node_features, edge_features):
    """
    Adds given features to the provided base graph. This function creates a copy of the graph.
    Each edge 'feature' input is a dictionary mapping feature names to a V x D matrix. This matrix
    holds feature values in a padded-adjacency-list format. The pad value is equal to
    to the number of nodes in the graph.
    """
    graph = graph.copy()

    # Simple error handling
    if node_features is None:
        node_features = {}

    if edge_features is None:
        edge_features = {}

    # Add node features
    for name, values in node_features.items():
        values = values.flatten()
        for node in graph.nodes():
            v = {name: float(values[node])}
            graph.add_node(node, **v)

    # Add edge features
    adj_lst, _ = adjacency_list(graph)
    for name, values in edge_features.items():
        for node, lst in enumerate(adj_lst):
            for i, neighbor in enumerate(lst):
                v = {name: float(values[node, i])}
                graph.add_edge(node, neighbor, **v)

    return graph


def max_degrees(graphs, k, unique_neighborhoods=True):
    adj_matrices = [nx.adjacency_matrix(graph) for graph in graphs]

    max_degrees = np.zeros(shape=(k+1,))
    for adj in adj_matrices:
        neighborhoods = random_walk_neighborhoods(adj, k=k, unique_neighborhoods=unique_neighborhoods)
        degrees = [np.max(mat.sum(axis=-1)) for mat in neighborhoods]        
        max_degrees = np.maximum(max_degrees, degrees)

    return max_degrees


def pad_adj_list(adj_lst, max_degree, max_num_nodes,  mask_number):
    padded = []
    for lst in adj_lst:
        pd = np.pad(lst, pad_width=(0, max_degree-len(lst)),
                    mode='constant', constant_values=mask_number)
        padded.append(pd)

    while len(padded) <= max_num_nodes:
        padded.append(np.full(shape=(max_degree, ), fill_value=mask_number))

    # Returns a max_num_nodes x max_degree numpy array
    return np.array(padded)


def neighborhood_adj_lists(neighborhoods, max_degrees, max_num_nodes, mask_number):
    neighborhood_lists = []
    for neighborhood, degree in zip(neighborhoods, max_degrees):
        adj_lst = adj_matrix_to_list(neighborhood)
        adj_lst = pad_adj_list(adj_lst=adj_lst,
                               max_degree=degree,
                               mask_number=mask_number,
                               max_num_nodes=max_num_nodes)
        neighborhood_lists.append(adj_lst)

    return neighborhood_lists


def adj_matrix_to_list(adj_matrix, inverted=False):
    if inverted:
        adj_matrix = adj_matrix.transpose(copy=True)

    rows, cols = adj_matrix.nonzero()

    adj_dict = {}
    for r, c in zip(rows, cols):
        if r not in adj_dict:
            adj_dict[r] = []
        adj_dict[r].append(c)

    # Create adjacency list
    adj_lst = []
    for node in sorted(adj_dict.keys()):
        adj_lst.append(list(sorted(adj_dict[node])))

    return adj_lst


def random_walk_neighborhoods(adj_matrix, k, unique_neighborhoods=True):
    mat = sp.eye(adj_matrix.shape[0], format='csr')
    neighborhoods = [mat]
    agg_mat = mat

    for _ in range(k):
        mat = mat.dot(adj_matrix)
        mat.data[:] = 1

        if unique_neighborhoods:
            # Remove already reached nodes
            mat = mat - agg_mat
            mat.data = np.maximum(mat.data, 0)
            mat.eliminate_zeros()
            mat.data[:] = 1

            agg_mat += mat
            agg_mat.data[:] = 1

        neighborhoods.append(mat)

    return neighborhoods


def adjacency_list(graph):
    adj_lst = list(map(list, iter(graph.adj.values())))
    max_degree = max(map(lambda x: len(x), adj_lst))
    return adj_lst, max_degree


def simple_paths(graph, sources, sinks, max_num_paths):
    cutoff = nx.diameter(graph)

    # Dictionary from (source, sink) to list of paths
    all_paths = {}


    def compute_paths(source, sink, cutoff, max_num_paths):
        paths = list(nx.all_simple_paths(graph, source, sink, cutoff=cutoff))
        paths = sorted(paths, key=len)[:max_num_paths]
        return {(source, sink): paths}

    result = Parallel(n_jobs=-1)(delayed(compute_paths)(source, sink, cutoff, max_num_paths) for source, sink in itertools.product(sources, sinks))

    for paths in result:
        all_paths.update(paths)

    return all_paths
