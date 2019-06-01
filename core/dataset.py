import numpy as np
import math
import networkx as nx
from enum import Enum
from os import path
from bisect import bisect_right
from collections.abc import Iterable
from core.load import read, load_graph
from utils.constants import BIG_NUMBER, WRITE_THRESHOLD
from utils.utils import expand_sparse_matrix, deserialize_dict
from utils.utils import create_node_embeddings, sparse_matrix_to_tensor
from utils.utils import expand_matrix, demands_to_features
from utils.graph_utils import adjacency_list, pad_adj_list, neighborhood_adj_lists
from utils.graph_utils import adj_matrix_to_list, random_walk_neighborhoods
from sklearn.preprocessing import StandardScaler


class Series(Enum):
    TRAIN = 1,
    VALID = 2,
    TEST = 3


class Counters:

    def __init__(self, samples, epoch, sort):
        self.samples = samples
        self.epoch = epoch
        self.sort = sort


class Sample:

    def __init__(self, demands, graph_name, max_num_nodes):
        self.demands = expand_sparse_matrix(demands, n=max_num_nodes, m=1)
        self.graph_name = graph_name


class BatchSample:

    def __init__(self, node_features, demands, graph_data):
        self.node_features = node_features
        self.demands = demands
        self.adj_lst = graph_data.adj_lst
        self.adj_mat = graph_data.adj_mat
        self.inv_adj_lst = graph_data.inv_adj_lst
        self.out_neighborhoods = graph_data.out_neighborhoods
        self.in_neighborhoods = graph_data.in_neighborhoods
        self.embeddings = graph_data.embeddings
        self.num_nodes = graph_data.num_nodes
        self.graph_name = graph_data.graph_name
        self.rev_indices = graph_data.rev_indices
        self.in_indices = graph_data.in_indices
        self.opp_indices = graph_data.opp_indices
        self.common_out_neighbors = graph_data.common_out_neighbors
        self.edge_lengths = graph_data.edge_lengths
        self.normalized_edge_lengths = graph_data.normalized_edge_lengths


class GraphData:

    def __init__(self, graph, graph_name, k, unique_neighborhoods):

        self.graph_name = graph_name
        self.graph = graph

        # Fetch expanded adjacency matrix
        self.adj_mat = nx.adjacency_matrix(graph)

        # Compute adjacency lists
        self.adj_lst, _ = adjacency_list(graph)
        self.inv_adj_lst = adj_matrix_to_list(self.adj_mat, inverted=True)

        # Compute neighborhoods
        self.out_neighborhoods = random_walk_neighborhoods(self.adj_mat, k, unique_neighborhoods)
        self.in_neighborhoods = random_walk_neighborhoods(self.adj_mat.transpose(copy=True), k, unique_neighborhoods)

        # Compute embeddings
        self.embeddings = create_node_embeddings(graph=graph, neighborhoods=self.out_neighborhoods)

        # Save the true number of nodes in this graph
        self.num_nodes = graph.number_of_nodes()

        # Compute neighbors which have common outgoing neighbors
        self.common_out_neighbors = self.common_outgoing_neighbors(graph=graph)

    def common_outgoing_neighbors(self, graph):
        common_out_neighbors = []

        for node in graph.nodes():
            common = []
            for neighbor in graph[node]:
                common += [n for n in graph.predecessors(neighbor) if n != node]
            common_out_neighbors.append(list(set(common)))

        return common_out_neighbors

    def fetch_edge_lengths(self):
        # Fetch and normalize edge lengths from underlying graph
        self.scaler = StandardScaler()
        edge_lengths = np.array([length for (_, _, length) in self.graph.edges.data('length')])
        normalized_lengths = self.scaler.fit_transform(edge_lengths.reshape(-1, 1)).reshape(-1)

        edge_len_dict = {}
        norm_edge_len_dict = {}
        for i, (src, dst) in enumerate(self.graph.edges(keys=False, data=False)):
            edge_len_dict[(src, dst)] = edge_lengths[i]
            norm_edge_len_dict[(src, dst)] = normalized_lengths[i]

        self.edge_lengths = np.zeros_like(self.adj_lst, dtype=float)
        self.normalized_edge_lengths = np.zeros_like(self.adj_lst, dtype=float)
        for node, _ in enumerate(self.adj_lst):
            for j, v in enumerate(self.adj_lst[node]):

                if (node, v) not in edge_len_dict:
                    continue

                self.edge_lengths[node, j] = edge_len_dict[(node, v)]
                self.normalized_edge_lengths[node, j] = norm_edge_len_dict[(node, v)]

    def set_edge_indices(self, adj_lst, inv_adj_lst, max_degree, max_num_nodes):
        dim0 = np.prod(adj_lst.shape)

        # These arrays hold 2D coordinates
        self.in_indices = np.zeros(shape=(dim0, 2))
        self.rev_indices = np.zeros(shape=(dim0, 2))
        self.opp_indices = np.zeros(shape=(dim0, 2))

        index_a = 0
        index_b = 0
        for x in range(adj_lst.shape[0]):
            for y in inv_adj_lst[x]:
                indexof = np.where(adj_lst[y] == x)[0]

                if len(indexof) > 0:
                    self.in_indices[index_a, 0] = y
                    self.in_indices[index_a, 1] = indexof[0]
                else:
                    self.in_indices[index_a, 0] = self.num_nodes
                    self.in_indices[index_a, 1] = max_degree - 1

                index_a += 1

            for y in adj_lst[x]:
                indexof = np.where(inv_adj_lst[y] == x)[0]

                if len(indexof) > 0:
                    self.rev_indices[index_b, 0] = y
                    self.rev_indices[index_b, 1] = indexof[0]
                else:
                    self.rev_indices[index_b, 0] = self.num_nodes
                    self.rev_indices[index_b, 1] = max_degree - 1

                # Reverse Edge
                indexof = np.where(adj_lst[y] == x)[0]

                if len(indexof) > 0:
                    self.opp_indices[index_b, 0] = y
                    self.opp_indices[index_b, 1] = indexof[0]
                else:
                    self.opp_indices[index_b, 0] = self.num_nodes
                    self.opp_indices[index_b, 1] = max_degree - 1

                index_b += 1


class DatasetManager:

    def __init__(self, params):

        self.data_folders = {}
        self.num_samples = {}

        dataset_folder_base = path.join('datasets', params['dataset_name'])
        dataset_params = deserialize_dict(path.join(dataset_folder_base, 'params.pkl.gz'))

        self.data_folders[Series.TRAIN] = path.join(dataset_folder_base, 'train')
        self.num_samples[Series.TRAIN] = dataset_params['train_samples']

        self.data_folders[Series.VALID] = path.join(dataset_folder_base, 'valid')
        self.num_samples[Series.VALID] = dataset_params['valid_samples']

        self.data_folders[Series.TEST] = path.join(dataset_folder_base, 'test')
        self.num_samples[Series.TEST] = dataset_params['test_samples']

        source_sink_dict = deserialize_dict(path.join(dataset_folder_base, 'sources_sinks.pkl.gz'))
        self.sources = source_sink_dict['sources']
        self.sinks = source_sink_dict['sinks']

        self.params = params
        self.dataset = {}
        self.graph_data = None

    def load_graphs(self):
        num_neighborhoods = self.params['num_neighborhoods']
        unique_neighborhoods = self.params['unique_neighborhoods']

        graph = load_graph(graph_name=self.params['graph_name'])
        self.graph_data = GraphData(graph=graph,
                                    graph_name=self.params['graph_name'],
                                    k=num_neighborhoods,
                                    unique_neighborhoods=unique_neighborhoods)

        self.num_nodes = graph.number_of_nodes()

        # Find the maximum outgoing degrees for each neighborhood level
        self.max_out_neighborhood_degrees = [np.max(mat.sum(axis=-1)) for mat in self.graph_data.out_neighborhoods]
        self.max_out_neighborhood_degrees = np.array(self.max_out_neighborhood_degrees).astype(int)

        self.max_in_neighborhood_degrees = [np.max(mat.sum(axis=-1)) for mat in self.graph_data.in_neighborhoods]
        self.max_in_neighborhood_degrees = np.array(self.max_in_neighborhood_degrees).astype(int)

        # Find the maximum outgoing or incoming degree for a single vertex
        max_out_deg = np.max([d for _, d in graph.out_degree()])
        max_in_deg = np.max([d for _, d in graph.in_degree()])
        self.max_degree = int(max(max_out_deg, max_in_deg))

        # Expand graph data to ensure consistent sizing
        self.graph_data.adj_lst = pad_adj_list(adj_lst=self.graph_data.adj_lst,
                                               max_degree=self.max_degree,
                                               max_num_nodes=self.num_nodes,
                                               mask_number=self.graph_data.num_nodes)
        self.graph_data.inv_adj_lst = pad_adj_list(adj_lst=self.graph_data.inv_adj_lst,
                                                   max_degree=self.max_degree,
                                                   max_num_nodes=self.num_nodes,
                                                   mask_number=self.graph_data.num_nodes)

        self.graph_data.set_edge_indices(adj_lst=self.graph_data.adj_lst,
                                         inv_adj_lst=self.graph_data.inv_adj_lst,
                                         max_degree=self.max_degree,
                                         max_num_nodes=self.num_nodes)

        self.graph_data.common_out_neighbors = pad_adj_list(adj_lst=self.graph_data.common_out_neighbors,
                                                            max_degree=self.max_degree**2,
                                                            max_num_nodes=self.num_nodes,
                                                            mask_number=self.graph_data.num_nodes)

        self.graph_data.adj_mat = expand_sparse_matrix(self.graph_data.adj_mat, n=self.num_nodes)

        self.graph_data.embeddings = expand_matrix(self.graph_data.embeddings,
                                                   n=self.num_nodes,
                                                   m=self.graph_data.embeddings.shape[1])

        self.graph_data.out_neighborhoods = neighborhood_adj_lists(neighborhoods=self.graph_data.out_neighborhoods,
                                                                   max_degrees=self.max_out_neighborhood_degrees,
                                                                   max_num_nodes=self.num_nodes,
                                                                   mask_number=self.graph_data.num_nodes)

        self.graph_data.in_neighborhoods = neighborhood_adj_lists(neighborhoods=self.graph_data.in_neighborhoods,
                                                                  max_degrees=self.max_in_neighborhood_degrees,
                                                                  max_num_nodes=self.num_nodes,
                                                                  mask_number=self.graph_data.num_nodes)

        self.graph_data.fetch_edge_lengths()

    def load(self, series):
        assert series is not None

        if series not in self.dataset:
            self.dataset[series] = []

        graph_name = self.params['graph_name']
        folder = self.data_folders[series]
        num_samples = self.num_samples[series]
        num_files = int(math.ceil(num_samples / WRITE_THRESHOLD))

        print('Started loading {0} {1} samples for graph {2}.'.format(num_samples, series.name, graph_name))

        for file_index in range(num_files):
            # Load demands as Sparse CSR matrices to save memory.
            demands = read(folder=folder,
                           file_index=file_index,
                           sources=self.sources,
                           sinks=self.sinks,
                           num_nodes=self.num_nodes)

            self.dataset[series] += [Sample(demands=demand, graph_name=graph_name, max_num_nodes=self.num_nodes)
                                     for demand in demands]

        assert len(self.dataset[series]) == num_samples
        print('Completed loading graph {0} for {1}.'.format(graph_name, series.name))

    def create_batches(self, series, batch_size, shuffle):
        """
        Generator for batches of a single series using uniform shuffling without replacement.
        """
        data = self.dataset[series]
        if shuffle:
            np.random.shuffle(data)

        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]

            batch = []
            for sample in batch_data:
                demands = sample.demands.todense()
                node_features = demands_to_features(demands)
                b = BatchSample(node_features=node_features,
                                demands=demands,
                                graph_data=self.graph_data)
                batch.append(b)

            yield batch

    def get_train_batch(self, batch_size):
        assert self.is_train_initialized, 'Training not yet initialized.'

        self.counters.samples += batch_size

        # Recompute selection probabilities once per epoch
        if self.counters.samples - self.counters.epoch > self.num_train_points:
            self.counters.epoch = self.counters.samples
            curr_epoch = int(self.counters.epoch / self.num_train_points)

            # Update selection
            if curr_epoch > 0:
                self.selection *= self.selection_factor

                # Update probabilities
                self.probs[0] = 1.0
                factor = 1.0 / math.exp(math.log(self.selection) / self.num_train_points)

                for i in range(1, self.num_train_points):
                    self.probs[i] = self.probs[i-1] * factor
                self.probs = self.probs / np.sum(self.probs)

                for i in range(1, self.num_train_points):
                    self.cumulative_probs[i] = self.cumulative_probs[i-1] + self.probs[i]

        # Re-sort data based on losses
        sort_threshold = self.params['batch_params']['sort_freq'] * self.num_train_points
        if self.counters.samples - self.counters.sort > sort_threshold:
            self.counters.sort = self.counters.samples

            # Sort samples based on losses
            samples = list(zip(self.losses, self.indices))
            samples.sort(key=lambda t: t[0], reverse=True)
            losses, indices = zip(*samples)
            self.losses, self.indices = np.array(losses), np.array(indices)

        batch = []
        indices = []
        for i in range(batch_size):
            r = min(np.random.random(), self.cumulative_probs[-1])
            index = bisect_right(self.cumulative_probs, r, lo=0, hi=len(self.cumulative_probs))

            # Prevent any out of bounds errors
            if index >= len(self.cumulative_probs):
                index = len(self.cumulative_probs) - 1

            data_index = self.indices[index]

            sample = self.dataset[Series.TRAIN][data_index]

            demands = sample.demands.todense()
            node_features = demands_to_features(demands)
            b = BatchSample(node_features=node_features,
                            demands=demands,
                            graph_data=self.graph_data)
            batch.append(b)
            indices.append(index)

        return batch, indices

    def report_losses(self, losses, indices):
        if not isinstance(losses, Iterable):
            losses = [losses]

        for loss, index in zip(losses, indices):
            self.losses[index] = loss

    def init(self, num_epochs):
        assert Series.TRAIN in self.dataset

        # Intialize losses
        self.num_train_points = len(self.dataset[Series.TRAIN])
        self.losses = np.full(shape=self.num_train_points, fill_value=BIG_NUMBER)
        self.indices = np.arange(start=0, stop=self.num_train_points, step=1)

        # Initialize counters
        self.counters = Counters(samples=0, epoch=-self.num_train_points, sort=0)

        # Intialize selection pressure
        s_beg = self.params['batch_params']['selection_beg']
        s_end = self.params['batch_params']['selection_end']
        self.selection_factor = math.exp(math.log(s_end / s_beg) / (num_epochs))
        self.selection = s_beg

        # Intialize probabilities
        self.probs = np.full(shape=self.num_train_points, fill_value=1.0/float(self.num_train_points))

        # Initialize cumulative probabilities
        self.cumulative_probs = np.zeros(shape=self.num_train_points, dtype=float)
        self.cumulative_probs[0] = 1.0 / float(self.num_train_points)
        for i in range(1, self.num_train_points):
            self.cumulative_probs[i] = self.cumulative_probs[i-1] + self.probs[i]

        self.is_train_initialized = True

    def num_batches(self, series, batch_size):
        return int(self.num_samples[series] / batch_size)
