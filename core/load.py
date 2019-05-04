import networkx as nx
import numpy as np
import pickle
import gzip
import scipy.sparse as sp
from os.path import exists
from utils.constants import SMALL_NUMBER
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


def write_dataset(dataset, output_path):
    """
    dataset is a list of dictionaries of the form { 'dem': [], 'cap': [] }
    """
    with gzip.GzipFile(output_path, 'ab') as output_file:
        for data_point in dataset:

            # Convert demands into node features
            node_features = np.zeros(shape=(data_point['dem'].shape[0], 2))
            for i, demand in enumerate(data_point['dem']):
                if demand[0] > 0:
                    node_features[i][0] = demand[0]
                elif demand[0] < 0:
                    node_features[i][1] = -demand[0]

            compressed_demands = sp.csr_matrix(node_features)
            pickle.dump({'dem': compressed_demands, 'cap': data_point['cap']}, output_file)

        # for demand in demands:
        #     demand_lst = [str(i) + ':' + str(d[0]) for i, d in enumerate(demand) if abs(d) > SMALL_NUMBER]
        #     output_file.write(' '.join(demand_lst) + '\n')


def read_dataset(data_path):
    dataset = []

    with gzip.GzipFile(data_path, 'rb') as data_file:
        try:
            while True:
                data_dict = pickle.load(data_file)
                dataset.append({'dem': data_dict['dem'], 'cap': data_dict['cap']})
        except EOFError:
            pass
        # for demand_lst in demands_file:
        #     demands = np.zeros(shape=(num_nodes, 2), dtype=float)
        #     demand_values = demand_lst.strip().split(' ')
        #     for value in demand_values:
        #         tokens = value.split(':')
        #         demand_val = float(tokens[1])
        #         node = int(tokens[0])

        #         if demand_val > 0:
        #             demands[node][0] = demand_val
        #         elif demand_val < 0:
        #             demands[node][1] = -demand_val

        #     dataset.append(demands)
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
