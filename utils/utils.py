import networkx as nx
import numpy as np
import tensorflow as tf
import math
import json
import gzip
import pickle
import csv
import scipy.sparse as sp
from os.path import exists
from os import remove
from utils.constants import *


def load_params(params_file_path):
    if not exists(params_file_path):
        print('The file {0} does not exist.'.format(params_file_path))
        return

    with open(params_file_path, 'r') as params_file:
        params = json.loads(params_file.read())
    return params


def add_features(graph, demands, flows, proportions, node_weights):
    graph = graph.copy()

    for node in graph.nodes():
        graph.add_node(node, demand=float(demands[node, 0] - demands[node, 1]),
                       node_weight=float(node_weights[node][0]))

    for src, dest in graph.edges():
        flow = float(flows[src, dest])
        prop = float(proportions[src, dest])
        graph.add_edge(src, dest, flow=flow, proportion=prop)

    return graph


def add_features_sparse(graph, demands, flows, proportions, node_weights):
    graph = graph.copy()

    for node in graph.nodes():
        graph.add_node(node, demand=float(demands[node, 0] - demands[node, 1]),
                       node_weight=float(node_weights[node][0]))

    for edge, flow in zip(flows.indices, flows.values):
        if graph.has_edge(*edge):
            graph.add_edge(edge[0], edge[1], flow=float(flow))

    for edge, prop in zip(proportions.indices, proportions.values):
        if graph.has_edge(*edge):
            graph.add_edge(edge[0], edge[1], proportion=float(prop))

    return graph


def features_to_demands(node_features):
    demands = np.zeros(shape=(node_features.shape[0], 1), dtype=float)
    for i in range(node_features.shape[0]):
        demands[i][0] = node_features[i, 0] - node_features[i, 1]
    return demands


def create_demands(graph, min_max_sources, min_max_sinks):
    # Randomly select the number of sources and sinks
    num_sources = np.random.randint(low=min_max_sources[0], high=min_max_sources[1]+1)
    num_sinks = np.random.randint(low=min_max_sinks[0], high=min_max_sinks[1]+1)

    # Randomly select sources and sinks
    source_sink_nodes = np.random.choice(graph.nodes(),
                                         size=num_sources + num_sinks,
                                         replace=False)

    sources = source_sink_nodes[:num_sources]
    sinks = source_sink_nodes[num_sources:]

    source_demands = -softmax(np.random.normal(size=num_sources))
    sink_demands = softmax(np.random.normal(size=num_sinks))

    # Create labels tensor
    demands = np.zeros(shape=(graph.number_of_nodes(), 1), dtype=float)
    for i, node in enumerate(sources):
        demands[node][0] = source_demands[i]

    for i, node in enumerate(sinks):
        demands[node][0] = sink_demands[i]

    return demands


def create_capacities(graph, demands):
    num_edges = graph.number_of_edges()

    abs_demands = np.abs(demands)
    min_capacity = 0.5 * np.min(abs_demands)
    max_capacity = 2.0 * np.max(abs_demands)

    adj_matrix = nx.adjacency_matrix(graph)
    init_capacities = np.random.uniform(low=min_capacity, high=max_capacity, size=(num_edges,))
    capacities = sp.csr_matrix((init_capacities, adj_matrix.indices, adj_matrix.indptr))

    # sources = np.transpose(np.where(demands < 0))
    # sinks = np.transpose(np.where(demands > 0))

    # for source in sources:
    #     source_demand = abs_demands[source[0]][source[1]]
    #     for sink in sinks:
    #         path_gen = nx.all_simple_paths(graph, source=source[0], target=sink[0])
    #         path = list(path_gen.__next__())

    #         # Increase capacity so at least one path can carry the source's flow to the sink.
    #         # This process ensures that a feasible solution exists
    #         for i in range(len(path)-1):
    #             edge = (path[i], path[i+1])
    #             capacities[edge] = max(capacities[edge], source_demand + 1e-3)

    #         edge = (path[i], path[i+1])
    #         capacities[edge] = max(capacities[edge], source_demand + 1e-3)

    return capacities


def create_node_embeddings(graph, num_nodes, neighborhoods):
    """
    Creates node "embeddings" based on the degrees of neighboring vertices and approximate
    centrality measures
    """
    embeddings = np.zeros(shape=(num_nodes, 2 * (len(neighborhoods) - 1) + 2), dtype=float)

    eigen = nx.eigenvector_centrality_numpy(graph)
    pagerank = nx.pagerank_scipy(graph, alpha=0.85)

    out_neighbors = []
    in_neighbors = []
    for i in range(1, len(neighborhoods)):
        out_neighbors.append(neighborhoods[i].sum(axis=1, dtype=float))
        in_neighbors.append(neighborhoods[i].sum(axis=0, dtype=float))

    for i, node in enumerate(graph.nodes()):
        for j in range(len(out_neighbors)):
            embeddings[i][2*j] = out_neighbors[j][i, 0] / graph.number_of_nodes()
            embeddings[i][2*j+1] = in_neighbors[j][0, i] / graph.number_of_nodes()

        embeddings[i][-2] = eigen[node]
        embeddings[i][-1] = pagerank[node]

    return np.array(embeddings)


def create_node_bias(graph):
    bias_mat = np.eye(graph.number_of_nodes(), dtype=float)
    for src, dest in graph.edges():
        bias_mat[src, dest] = 1.0
    return -BIG_NUMBER * (1.0 - bias_mat)


def adj_mat_to_node_bias(adj_mat):
    bias_mat = adj_mat.todense()
    return -BIG_NUMBER * (1.0 - bias_mat)


def gcn_aggregator(adj):
    num_nodes = adj.shape[0]
    adj_hat = adj + np.eye(num_nodes)

    sqrt_degrees = 1.0 / np.sqrt(np.sum(adj_hat, axis=-1))

    degree_mat = np.zeros_like(adj_hat)
    np.fill_diagonal(degree_mat, sqrt_degrees)

    return degree_mat.dot(adj_hat.dot(degree_mat))


def create_obs_indices(dim1, dim2):
    x1 = np.arange(0, dim1)
    x2 = np.arange(0, dim2)

    a, b = np.meshgrid(x1, x2)

    a = np.reshape(a, [-1, 1], order='F')
    b = np.reshape(b, [-1, 1], order='F')
    return np.concatenate([a, b], axis=-1).astype(np.int32)


def create_edge_bias(graph):
    bias_mat = np.eye(graph.number_of_edges(), dtype=float)
    for src, dest in graph.edges():
        for node in graph.predecessors(src):
            bias_mat[src, node] = 1.0

        for node in graph.successors(src):
            bias_mat[src, node] = 1.0

        for node in graph.successors(dest):
            bias_mat[src, node] = 1.0
    return -BIG_NUMBER * (1.0 - bias_mat)


def softmax(arr):
    max_elem = np.max(arr)
    exp_arr = np.exp(arr - max_elem)
    return exp_arr / np.sum(exp_arr)


def sparse_matrix_to_tensor(sparse_mat):
    mat = sparse_mat.tocoo()
    indices = np.mat([mat.row, mat.col]).transpose()
    return tf.SparseTensorValue(indices, mat.data, mat.shape)


def sparse_matrix_to_tensor_multiple(sparse_mat, k):
    mat = sparse_mat.tocoo()
    indices = np.mat([mat.row, mat.col]).transpose()
    num_indices = indices.shape[0]

    expanded_shape = (indices.shape[0] * k, indices.shape[1] + 1)
    indices_expanded = np.zeros(shape=expanded_shape)

    for i in range(k):
        for j in range(num_indices):
            index = i * num_indices + j
            indices_expanded[index][0] = i # Repetition Number
            for t in range(1, indices.shape[1]+1):
                indices_expanded[index][t] = indices[j,t-1]

    data_expanded = np.repeat(mat.data, repeats=k, axis=0)
    return tf.SparseTensorValue(indices_expanded, data_expanded, expanded_shape)


def random_walk_neighborhoods(adj_matrix, k, unique_neighborhoods=True):
    mat = sp.eye(adj_matrix.shape[0])
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


def append_row_to_log(row, log_path):
    with open(log_path, 'a') as log_file:
        log_writer = csv.writer(log_file, delimiter=',', quotechar='|')
        log_writer.writerow(row)


def restore_params(model_path):
    params_path = PARAMS_FILE.format(model_path)
    with gzip.GzipFile(params_path, 'rb') as params_file:
        params_dict = pickle.load(params_file)
    return params_dict


def expand_sparse_matrix(csr_mat, n, m=None):
    """
    Expands the given m x m CSR matrix to size n x n. This function
    will create a new matrix.
    """
    if m is None:
        m = n

    pad_amount = (n + 1) - len(csr_mat.indptr)
    if pad_amount <= 0:
        return csr_mat

    indptr = np.pad(csr_mat.indptr, (0, pad_amount), mode='edge')
    return sp.csr_matrix((csr_mat.data, csr_mat.indices, indptr), shape=(n, m))


def delete_if_exists(file_path):
    if exists(file_path):
        remove(file_path)
