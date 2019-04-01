import networkx as nx
import numpy as np
from os.path import exists
from constants import SMALL_NUMBER
from annoy import AnnoyIndex


def load_to_networkx(path):
    graph = nx.DiGraph()

    edge_features = {}

    with open(path, 'r') as net_file:
        metadata = True

        for line in net_file:
            if len(line) == 0:
                continue
            if line.startswith('~'):
                metadata = False
                continue
            if metadata:
                continue

            edge_elems = line.split('\t')
            init, term, features = _parse_edge_features(edge_elems)

            if init == -1 or term == -1:
                continue

            graph.add_node(init)
            graph.add_node(term)
            graph.add_edge(init, term)

            edge = (init, term)
            edge_features[edge] = features
    nx.set_edge_attributes(graph, edge_features)
    return graph


def write_dataset(demands, output_path, mode='a'):
    with open(output_path, mode) as output_file:
        for demand in demands:
            demand_lst = [str(i) + ':' + str(d[0]) for i, d in enumerate(demand) if abs(d) > SMALL_NUMBER]
            output_file.write(' '.join(demand_lst) + '\n')


def read_dataset(demands_path, num_nodes):
    dataset = []
    with open(demands_path, 'r') as demands_file:
        for demand_lst in demands_file:
            demands = np.zeros(shape=(num_nodes, 1), dtype=float)
            demand_values = demand_lst.strip().split(' ')
            for value in demand_values:
                tokens = value.split(':')
                demands[int(tokens[0])][0] = float(tokens[1])
            dataset.append(demands)
    return dataset


def load_embeddings(index_path, embedding_size, num_nodes):
    # Load Annoy index which stores the embedded vectors
    index = AnnoyIndex(embedding_size)
    index.load(index_path)

    embeddings = [index.get_item_vector(i) for i in range(num_nodes)]

    # Unload the index to save memory (loading mmaps the index file)
    index.unload()

    # V x D matrix of embeddings
    return np.array(embeddings)


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
