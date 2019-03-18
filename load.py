import networkx as nx
import numpy as np
from os.path import exists


def load_to_networkx(net_path):
    if not exists(net_path):
        print('The path {0} does not exist.'.format(net_path))
        return

    graph = nx.DiGraph()

    edge_features = {}

    with open(net_path, 'r') as net_file:
        metadata = True

        for line in net_file:
            if line.startswith('~'):
                metadata = False
                continue
            if metadata:
                continue

            edge_elems = line.split('\t')
            init, term, features = _parse_edge_features(edge_elems)

            graph.add_node(init)
            graph.add_node(term)
            graph.add_edge(init, term)

            edge = (init, term)
            edge_features[edge] = features
    nx.set_edge_attributes(graph, edge_features)
    return graph


def _parse_edge_features(edge_elems):
    init = int(_get_index_if_exists(edge_elems, 1, 0))
    term = int(_get_index_if_exists(edge_elems, 2, 0))

    features = {
        'capacity': float(_get_index_if_exists(edge_elems, 3, 0.0)),
        'length': float(_get_index_if_exists(edge_elems, 4, 0.0)),
        'free_flow_time': float(_get_index_if_exists(edge_elems, 5, 0.0)),
        'b': float(_get_index_if_exists(edge_elems, 6, 0.0)),
        'power': float(_get_index_if_exists(edge_elems, 7, 0.0)),
        'speed_limit': float(_get_index_if_exists(edge_elems, 8, 0.0)),
        'toll': float(_get_index_if_exists(edge_elems, 9, 0.0)),
        'link_type': int(_get_index_if_exists(edge_elems, 10, 0.0))
    }

    return init-1, term-1, features


def _get_index_if_exists(array, index, default):
    if len(array) <= index:
        return default
    if len(array[index]) == 0:
        return default
    return array[index]
