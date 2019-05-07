import numpy as np
import networkx as nx


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


def adjacency_list(graph):
    adj_lst = list(map(list, iter(graph.adj.values())))
    max_degree = max(map(lambda x: len(x), adj_lst))
    return adj_lst, max_degree
