import numpy as np
import math
import networkx as nx
from enum import Enum
from bisect import bisect_right
from collections.abc import Iterable
from core.load import read_dataset
from utils.constants import BIG_NUMBER
from utils.utils import expand_sparse_matrix, random_walk_neighborhoods
from utils.utils import create_node_embeddings, features_to_demands
from utils.utils import sparse_block_diag, dense_block_diag
from sklearn.preprocessing import StandardScaler


class Series(Enum):
    TRAIN = 1,
    VALID = 2,
    TEST = 3


class DataSeries(Enum):
    NODE = 1,
    ADJ = 2,
    NEIGHBORHOOD = 3,
    GRAPH_NAME = 4,
    CAPACITY = 5,
    DEMAND = 6


class Counters:

    def __init__(self, samples, epoch, sort):
        self.samples = samples
        self.epoch = epoch
        self.sort = sort


class Sample:

    def __init__(self, feat_dict, graph_name, num_nodes):
        self.node_features = expand_sparse_matrix(feat_dict['dem'],
                                                  n=num_nodes,
                                                  m=2)
        self.graph_name = graph_name
        self.capacities = expand_sparse_matrix(feat_dict['cap'], n=num_nodes)


class GraphData:

    def __init__(self, adj_matrix, neighborhoods, node_embeddings):
        self.adj_matrix = adj_matrix
        self.neighborhoods = neighborhoods
        self.node_embeddings = node_embeddings


class DatasetManager:

    def __init__(self, file_paths, params):
        self.file_paths = file_paths
        self.params = params
        self.dataset = {}
        self.graph_data = {}
        self.scaler = StandardScaler()

    def load(self, series, graphs, num_nodes, num_neighborhoods, unique_neighborhoods=True):
        assert series is not None

        if not isinstance(series, Iterable):
            series = [series]

        for s in series:

            if s not in self.dataset:
                self.dataset[s] = []

            for graph_name, path in self.file_paths[s].items():
                features = read_dataset(data_path=path)
                self.dataset[s] += [Sample(feat_dict=feat_dict, graph_name=graph_name, num_nodes=num_nodes)
                                    for feat_dict in features]

                # Lazily load graph adjacency matrix, neighborhoods, and node embeddings
                if graph_name not in self.graph_data:
                    adj_matrix = nx.adjacency_matrix(graphs[graph_name])
                    adj_matrix = expand_sparse_matrix(csr_mat=adj_matrix, n=num_nodes)

                    neighborhoods = random_walk_neighborhoods(adj_matrix, k=num_neighborhoods,
                                                              unique_neighborhoods=unique_neighborhoods)
                    embeddings = create_node_embeddings(graph=graphs[graph_name], num_nodes=num_nodes,
                                                        neighborhoods=neighborhoods)

                    self.graph_data[graph_name] = GraphData(adj_matrix, neighborhoods, embeddings)

    def create_shuffled_batches(self, series, batch_size, is_sparse):
        return self.create_batches(series, batch_size, shuffle=True, is_sparse=is_sparse)

    def create_batches(self, series, batch_size, shuffle, is_sparse):
        """
        Returns all batches for a single series using uniform shuffling without replacement.
        """
        data = self.dataset[series]
        if shuffle:
            np.random.shuffle(data)

        node_batches = []
        adj_batches = []
        neighborhoods_batches = []
        demand_batches = []
        graph_name_batches = []
        capacity_batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]

            graph_names = [sample.graph_name for sample in batch]

            # Extract node features and create demand vector
            node_features = [sample.node_features.todense() for sample in batch]
            embeddings = [self.graph_data[name].node_embeddings for name in graph_names]
            demands = [features_to_demands(n) for n in node_features]

            # Put node embeddings and demand values together
            node_features = np.concatenate((node_features, embeddings), axis=-1)

            adj_matrices = [self.graph_data[name].adj_matrix for name in graph_names]
            neighborhoods = [self.graph_data[name].neighborhoods for name in graph_names]
            capacities = [sample.capacities for sample in batch]

            if is_sparse:
                node_features = dense_block_diag(node_features)
                adj_matrices = sparse_block_diag(adj_matrices)
                demands = dense_block_diag(demands)
                capacities = sparse_block_diag(capacities)

                # Convert each neighborhood level into a sparse block matrix
                neighborhood_levels = []
                for i in range(len(neighborhoods[0])):
                    level = [n[i] for n in neighborhoods]
                    neighborhood_levels.append(sparse_block_diag(level))
                neighborhoods = neighborhood_levels

            node_batches.append(node_features)
            adj_batches.append(adj_matrices)
            neighborhoods_batches.append(neighborhoods)
            demand_batches.append(demands)
            graph_name_batches.append(graph_names)
            capacity_batches.append(capacities)

        return {
            DataSeries.NODE: node_batches,
            DataSeries.ADJ: adj_batches,
            DataSeries.NEIGHBORHOOD: neighborhoods_batches,
            DataSeries.DEMAND: demand_batches,
            DataSeries.GRAPH_NAME: graph_name_batches,
            DataSeries.CAPACITY: capacity_batches
        }

    def get_train_batch(self, batch_size, is_sparse):
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
        sort_threshold = self.params['sort_freq'] * self.num_train_points
        if self.counters.samples - self.counters.sort > sort_threshold:
            self.counters.sort = self.counters.samples

            # Sort samples based on losses
            samples = list(zip(self.losses, self.indices))
            samples.sort(key=lambda t: t[0], reverse=True)
            losses, indices = zip(*samples)
            self.losses, self.indices = np.array(losses), np.array(indices)

        node_batch = []
        adj_batch = []
        neighborhood_batch = []
        demand_batch = []
        capacity_batch = []

        indices = []
        for i in range(batch_size):
            r = min(np.random.random(), self.cumulative_probs[-1])
            index = bisect_right(self.cumulative_probs, r, lo=0, hi=len(self.cumulative_probs))

            # Prevent any out of bounds errors
            if index >= len(self.cumulative_probs):
                index = len(self.cumulative_probs) - 1

            data_index = self.indices[index]

            sample = self.dataset[Series.TRAIN][data_index]

            node_features = sample.node_features.todense()
            embeddings = self.graph_data[sample.graph_name].node_embeddings
            demands = features_to_demands(node_features)

            # Put node features and embeddings together
            node_features = np.concatenate((node_features, embeddings), axis=-1)

            node_batch.append(node_features)
            adj_batch.append(self.graph_data[sample.graph_name].adj_matrix)
            neighborhood_batch.append(self.graph_data[sample.graph_name].neighborhoods)
            demand_batch.append(demands)
            capacity_batch.append(sample.capacities)
            indices.append(index)

        if is_sparse:
            node_batch = dense_block_diag(node_batch)
            adj_batch = sparse_block_diag(adj_batch)
            demand_batch = dense_block_diag(demand_batch)
            capacity_batch = sparse_block_diag(capacity_batch)

            # Convert each neighborhood level into a sparse block matrix
            neighborhood_levels = []
            for i in range(len(neighborhood_batch[0])):
                level = [n[i] for n in neighborhood_batch]
                neighborhood_levels.append(sparse_block_diag(level))
            neighborhood_batch = neighborhood_levels

        batch_dict = {
            DataSeries.NODE: node_batch,
            DataSeries.ADJ: adj_batch,
            DataSeries.NEIGHBORHOOD: neighborhood_batch,
            DataSeries.DEMAND: demand_batch,
            DataSeries.CAPACITY: capacity_batch
        }

        return batch_dict, indices

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
        s_beg = self.params['selection_beg']
        s_end = self.params['selection_end']
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
        for graph_name in self.file_paths[Series.TRAIN].keys():
            graph_embeddings = self.graph_data[graph_name].node_embeddings
            embeddings_lst.append(graph_embeddings)
        embeddings = np.concatenate(embeddings_lst, axis=0)

        self.scaler.fit(embeddings)

        for graph_name in self.graph_data.keys():
            node_embeddings = self.graph_data[graph_name].node_embeddings
            transformed_embeddings = self.scaler.transform(node_embeddings)
            self.graph_data[graph_name].node_embeddings = transformed_embeddings
