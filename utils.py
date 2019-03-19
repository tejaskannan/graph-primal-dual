import networkx as nx
import numpy as np
import tensorflow as tf
import json
from os.path import exists
from constants import *


def load_params(params_file_path):
    if not exists(params_file_path):
        print('The file {0} does not exist.'.format(params_file_path))
        return

    with open(params_file_path, 'r') as params_file:
        params = json.loads(params_file.read())
    return params


def add_features(graph, labels, capacities):
    graph = graph.copy()

    for i, node in enumerate(graph.nodes()):
        if labels[i][0] == 1:
            graph.add_node(node, source=True)
        elif labels[i][1] == 1:
            graph.add_node(node, sink=True)

    for i, (src, dest) in enumerate(graph.edges()):
        graph.add_edge(src, dest, capacity=capacities[i])

    return graph


def create_tensors(graph, min_max_sources, min_max_sinks):
    # Randomly select the number of sources and sinks
    num_sources = np.random.randint(low=min_max_sources[0], high=min_max_sources[1])
    num_sinks = np.random.randint(low=min_max_sinks[0], high=min_max_sinks[1])

    # Randomly select sources and sinks
    source_sink_nodes = np.random.choice(graph.nodes(),
                                         size=num_sources + num_sinks,
                                         replace=False)

    sources = source_sink_nodes[:num_sources]
    sinks = source_sink_nodes[num_sources:]

    # Create labels tensor
    labels = np.zeros(shape=(graph.number_of_nodes(), 3), dtype=float)
    for node in graph.nodes():
        labels[node][0] = 0
        labels[node][1] = 0
        labels[node][2] = 1

    for node in sources:
        labels[node][0] = 1
        labels[node][1] = 0
        labels[node][2] = 0

    for node in sinks:
        labels[node][0] = 0
        labels[node][1] = 1
        labels[node][2] = 0

    # Create mask to fetch all edges leaving the source
    source_edge_mask = np.zeros(shape=(graph.number_of_edges(), 1), dtype=float)
    for i, (src, dest) in enumerate(graph.edges()):
        if src in sources:
            source_edge_mask[i] = 1

    # Randomize capacities
    capacities = np.random.uniform(size=(graph.number_of_edges(), 1), low=0.1, high=1.0)

    return labels, capacities, source_edge_mask


def create_batch(graph, batch_size, min_max_sources, min_max_sinks):
    node_features = []
    edge_features = []
    source_masks = []
    for i in range(batch_size):
        labels, capacities, source_mask = create_tensors(graph, min_max_sources, min_max_sinks)
        node_features.append(labels)
        edge_features.append(capacities)
        source_masks.append(source_mask)
    return np.array(node_features), np.array(edge_features), np.array(source_masks)


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
