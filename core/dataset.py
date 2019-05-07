import numpy as np
import math
import networkx as nx
from enum import Enum
from bisect import bisect_right
from collections.abc import Iterable
from core.load import read_sparse_npz, load_to_networkx
from utils.constants import BIG_NUMBER
from utils.utils import expand_sparse_matrix
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

    def __init__(self, node_features, demands, adj_lst, adj_mat,
                 neighborhoods, embeddings, num_nodes, graph_name):
        self.node_features = node_features
        self.demands = demands
        self.adj_lst = adj_lst
        self.adj_mat = adj_mat
        self.neighborhoods = neighborhoods
        self.embeddings = embeddings
        self.num_nodes = num_nodes
        self.graph_name = graph_name


class GraphData:

    def __init__(self, graph, graph_name, k, unique_neighborhoods):

        self.graph_name = graph_name

        # Fetch expanded adjacency matrix
        self.adj_mat = nx.adjacency_matrix(graph)

        # Compute adjacency list and maximum outgoing degree
        self.adj_lst, _ = adjacency_list(graph)

        # Compute neighborhoods
        self.neighborhoods = random_walk_neighborhoods(self.adj_mat, k, unique_neighborhoods)

        # Compute embeddings
        self.embeddings = create_node_embeddings(graph=graph, neighborhoods=self.neighborhoods)

        # Save the true number of nodes in this graph
        self.num_nodes = graph.number_of_nodes()


class DatasetManager:

    def __init__(self, data_folders, params):
        self.data_folders = data_folders
        self.params = params
        self.dataset = {}
        self.graph_data = {}
        self.scaler = StandardScaler()

    def load_graphs(self, normalize):
        graph_path = 'graphs/{0}.tntp'

        self.graph_data = {}

        num_neighborhoods = self.params['num_neighborhoods']
        unique_neighborhoods = self.params['unique_neighborhoods']

        # Load training and test graphs from TNTP files into networkx
        self.train_graphs = {}
        for graph_name in self.params['train_graph_names']:
            graph = load_to_networkx(path=graph_path.format(graph_name))
            self.train_graphs[graph_name] = graph

            if graph_name in self.graph_data:
                continue
            self.graph_data[graph_name] = GraphData(graph=graph,
                                                    graph_name=graph_name,
                                                    k=num_neighborhoods,
                                                    unique_neighborhoods=unique_neighborhoods)

        self.test_graphs = {}
        for graph_name in self.params['test_graph_names']:
            graph = load_to_networkx(path=graph_path.format(graph_name))
            self.test_graphs[graph_name] = graph

            if graph_name in self.graph_data:
                continue
            self.graph_data[graph_name] = GraphData(graph=graph,
                                                    graph_name=graph_name,
                                                    k=num_neighborhoods,
                                                    unique_neighborhoods=unique_neighborhoods)

        # Normalize embeddings with respect to training data
        if normalize:
            self.normalize_embeddings()

        # Find the maximum number of nodes in all graphs
        num_train_nodes = np.max([g.number_of_nodes() for g in self.train_graphs.values()])
        num_test_nodes = np.max([g.number_of_nodes() for g in self.test_graphs.values()])
        self.max_num_nodes = int(max(num_train_nodes, num_test_nodes))

        # Find the maximum outgoing degrees for each neighborhood level
        max_degrees = np.zeros(shape=(unique_neighborhoods+1,))
        for gd in self.graph_data.values():
            degrees = [np.max(mat.sum(axis=-1)) for mat in gd.neighborhoods]
            max_degrees = np.maximum(max_degrees, degrees)
        self.max_neighborhood_degrees = max_degrees.astype(int)

        # Find the maximum outgoing or incoming degree for a single vertex
        train_out_deg = np.max([max([d for _, d in g.out_degree()]) for g in self.train_graphs.values()])
        train_in_deg = np.max([max([d for _, d in g.in_degree()]) for g in self.train_graphs.values()])

        test_out_deg = np.max([max([d for _, d in g.out_degree()]) for g in self.test_graphs.values()])
        test_in_deg = np.max([max([d for _, d in g.in_degree()]) for g in self.test_graphs.values()])
        self.max_degree = int(max(max(train_out_deg, train_in_deg), max(test_out_deg, test_in_deg)))

        # Expand graph data to ensure consistent sizing
        for gd in self.graph_data.values():
            gd.adj_mat = expand_sparse_matrix(gd.adj_mat, n=self.max_num_nodes)
            gd.embeddings = expand_matrix(gd.embeddings, n=self.max_num_nodes,
                                          m=gd.embeddings.shape[1])
            gd.adj_lst = pad_adj_list(adj_lst=gd.adj_lst,
                                      max_degree=self.max_degree,
                                      max_num_nodes=self.max_num_nodes,
                                      mask_number=gd.num_nodes)
            gd.neighborhoods = neighborhood_adj_lists(neighborhoods=gd.neighborhoods,
                                                      max_degrees=self.max_neighborhood_degrees,
                                                      max_num_nodes=self.max_num_nodes,
                                                      mask_number=gd.num_nodes)

    def load(self, series):
        assert series is not None

        if not isinstance(series, Iterable):
            series = [series]

        for s in series:

            if s not in self.dataset:
                self.dataset[s] = []

            for graph_name, folder in self.data_folders[s].items():

                # Load demands as Sparse CSR matrices to save memory.
                demands = read_sparse_npz(folder=folder)

                self.dataset[s] += [Sample(demands=demand, graph_name=graph_name, max_num_nodes=self.max_num_nodes)
                                    for demand in demands]

    def create_batches(self, series, batch_size, shuffle):
        """
        Returns all batches for a single series using uniform shuffling without replacement.
        """
        data = self.dataset[series]
        if shuffle:
            np.random.shuffle(data)

        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]

            batch = []
            for sample in batch_data:
                gd = self.graph_data[sample.graph_name]

                demands = sample.demands.todense()
                node_features = demands_to_features(demands)
                b = BatchSample(node_features=node_features,
                                demands=demands,
                                adj_lst=gd.adj_lst,
                                adj_mat=gd.adj_mat,
                                neighborhoods=gd.neighborhoods,
                                embeddings=gd.embeddings,
                                num_nodes=gd.num_nodes,
                                graph_name=gd.graph_name)
                batch.append(b)
            batches.append(batch)

        return batches

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

            gd = self.graph_data[sample.graph_name]

            demands = sample.demands.todense()
            node_features = demands_to_features(demands)
            b = BatchSample(node_features=node_features,
                            demands=demands,
                            adj_lst=gd.adj_lst,
                            adj_mat=gd.adj_mat,
                            neighborhoods=gd.neighborhoods,
                            embeddings=gd.embeddings,
                            num_nodes=gd.num_nodes,
                            graph_name=gd.graph_name)
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

    def normalize_embeddings(self):
        embeddings_lst = []
        for graph_name in self.data_folders[Series.TRAIN].keys():
            graph_embeddings = self.graph_data[graph_name].embeddings
            embeddings_lst.append(graph_embeddings)
        embeddings = np.concatenate(embeddings_lst, axis=0)

        self.scaler.fit(embeddings)

        for graph_name in self.graph_data.keys():
            embeddings = self.graph_data[graph_name].embeddings
            transformed_embeddings = self.scaler.transform(embeddings)
            self.graph_data[graph_name].embeddings = transformed_embeddings
