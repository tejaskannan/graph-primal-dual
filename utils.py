import networkx as nx
import numpy as np
import tensorflow as tf
import math
import json
import gzip
import pickle
import csv
from os.path import exists
from constants import *
from scipy.sparse import csr_matrix


def load_params(params_file_path):
    if not exists(params_file_path):
        print('The file {0} does not exist.'.format(params_file_path))
        return

    with open(params_file_path, 'r') as params_file:
        params = json.loads(params_file.read())
    return params


def add_features(graph, demands, flows, proportions):
    graph = graph.copy()

    for node in graph.nodes():
        graph.add_node(node, demand=float(demands[node][0]))

    for src, dest in graph.edges():
        flow = float(flows[src, dest])
        prop = float(proportions[src, dest])
        graph.add_edge(src, dest, flow=flow, proportion=prop)

    return graph


def add_features_sparse(graph, demands, flows, proportions):
    graph = graph.copy()

    for node in graph.nodes():
        graph.add_node(node, demand=float(demands[node][0]))

    for edge, flow in zip(flows.indices, flows.values):
        graph.add_edge(edge[0], edge[1], flow=float(flow))

    for edge, prop in zip(proportions.indices, proportions.values):
        graph.add_edge(edge[0], edge[1], proportion=float(prop))

    return graph


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


def create_node_embeddings(graph):
    """
    Creates node "embeddings" based on a set of (approximate) centrality measures
    """
    embeddings = []

    # Degree Centralities
    in_deg = nx.in_degree_centrality(graph)
    out_deg = nx.out_degree_centrality(graph)

    # Betweenness Centrality
    samples = int(math.log(graph.number_of_nodes()))
    btwn = nx.betweenness_centrality(graph, k=samples, normalized=True)

    # PageRank (Uses power iteration over sparse matrices)
    pagerank =  nx.pagerank_scipy(graph, alpha=0.85, max_iter=50, tol=1e-5)

    # Eigenvector Centrality (uses sparse matrix computations)
    eigen = nx.eigenvector_centrality_numpy(graph, max_iter=50, tol=1e-5)

    for n in graph.nodes():
        embedding = np.array([in_deg[n], out_deg[n], btwn[n], pagerank[n], eigen[n]])
        embeddings.append(embedding)
    return np.array(embeddings)


def create_node_bias(graph):
    bias_mat = np.eye(graph.number_of_nodes(), dtype=float)
    for src, dest in graph.edges():
        bias_mat[src, dest] = 1.0
    return -BIG_NUMBER * (1.0 - bias_mat)


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


def append_row_to_log(row, log_path):
    with open(log_path, 'a') as log_file:
        log_writer = csv.writer(log_file, delimiter=',', quotechar='|')
        log_writer.writerow(row)


def restore_params(model_path):
    params_path = PARAMS_FILE.format(model_path)
    with gzip.GzipFile(params_path, 'rb') as params_file:
        params_dict = pickle.load(params_file)
    return params_dict
