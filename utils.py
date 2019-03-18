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


def add_demands(graph, demands):
    graph = graph.copy()

    for i, node in enumerate(graph.nodes()):
        graph.add_node(node, demand=demands[i])

    return graph


def create_demand_tensor(graph, min_max_sources, min_max_sinks):
    # Randomly select the number of sources and sinks
    num_sources = np.random.randint(low=min_max_sources[0], high=min_max_sources[1])
    num_sinks = np.random.randint(low=min_max_sinks[0], high=min_max_sinks[1])

    # Randomly select sources and sinks
    source_sink_nodes = np.random.choice(graph.nodes(),
                                         size=num_sources + num_sinks,
                                         replace=False)

    sources = source_sink_nodes[:num_sources]
    sinks = source_sink_nodes[num_sources:]

    # Randomly set demand values such that the total source and sink values
    # both sum to one. This property makes the problem feasible.
    source_demands = softmax(np.random.uniform(size=num_sources))
    sink_demands = softmax(np.random.uniform(size=num_sinks))

    # Create demands tensor
    demands = np.zeros(shape=(graph.number_of_nodes(), 1), dtype=float)
    for node, d in zip(sources, source_demands):
        demands[node][0] = -d

    for node, d in zip(sinks, sink_demands):
        demands[node][0] = d

    assert np.sum(demands) < SMALL_NUMBER, 'Demands are not balanced.'

    return demands


def create_node_bias(graph):
    bias_mat = np.eye(graph.number_of_nodes(), dtype=float)
    for src, dest in graph.edges():
        bias_mat[src, dest] = 1.0
    return -BIG_NUMBER * (1.0 - bias_mat)


def softmax(arr):
    max_elem = np.max(arr)
    exp_arr = np.exp(arr - max_elem)
    return exp_arr / np.sum(exp_arr)


def sparse_matrix_to_tensor(sparse_mat):
    mat = sparse_mat.tocoo()
    indices = np.mat([mat.row, mat.col]).transpose()
    return tf.SparseTensorValue(indices, mat.data, mat.shape)
