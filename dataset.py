import numpy as np
import math
from load import read_dataset
from enum import Enum
from collections.abc import Iterable
from constants import BIG_NUMBER
from bisect import bisect_right


class Series(Enum):
    TRAIN = 1,
    VALID = 2,
    TEST = 3


class Counters:

    def __init__(self, samples, epoch, sort):
        self.samples = samples
        self.epoch = epoch
        self.sort = sort


class DatasetManager:

    def __init__(self, train_file, valid_file, test_file, params):
        self.file_paths = {
            Series.TRAIN: train_file,
            Series.VALID: valid_file,
            Series.TEST: test_file
        }
        self.params = params
        self.dataset = {}

    def load_all(self, num_nodes):
        self.load([Series.TRAIN, Series.VALID, Series.TEST], num_nodes)

    def load(self, series, num_nodes):
        assert series is not None

        if not isinstance(series, Iterable):
            series = [series]

        for s in series:
            self.dataset[s] = read_dataset(demands_path=self.file_paths[s], num_nodes=num_nodes)

    def create_shuffled_batches(self, series, batch_size):
        """
        Returns all batches for a single series using uniform shuffling without replacement.
        """
        data = self.dataset[series]
        np.random.shuffle(data)

        node_features = []
        for i in range(0, len(data), batch_size):
            node_features.append(data[i:i+batch_size])
        return np.array(node_features)

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
        sort_threshold = self.params['sort_freq'] * self.num_train_points
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
            batch.append(self.dataset[Series.TRAIN][data_index])
            indices.append(index)

        return batch, indices

    def report_losses(self, losses, indices):
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
