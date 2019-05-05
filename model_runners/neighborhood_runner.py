import numpy as np
import tensorflow as tf
from utils.utils import sparse_matrix_to_tensor, features_to_demands
from utils.constants import BIG_NUMBER, LINE
from core.dataset import DatasetManager, Series, DataSeries
from model_runners.model_runner import ModelRunner
from models.neighborhood_model import NeighborhoodModel


class NeighborhoodRunner(ModelRunner):

    def create_placeholders(self, model, num_nodes, embedding_size, **kwargs):
        num_neighborhoods = kwargs['num_neighborhoods']

        if self.params['sparse']:
            b = self.params['batch_size']
            node_shape = [b * num_nodes, b * (self.num_node_features + embedding_shape)]
            demands_shape = [b * num_nodes, b]
            adj_shape = [None, num_nodes * b]
            neighborhood_shape = [None, num_nodes * b]
            capacity_shape = [None, num_nodes, * b]
        else:
            node_shape = [None, num_nodes, self.num_node_features + embedding_size]
            demands_shape = [None, num_nodes, 1]
            adj_shape = [None, None, num_nodes]
            neighborhood_shape = [None, None, num_nodes]
            capacity_shape = [None, None, num_nodes]


        node_ph = model.create_placeholder(dtype=tf.float32,
                                           shape=node_shape,
                                           name='node-ph',
                                           is_sparse=False)
        demands_ph = model.create_placeholder(dtype=tf.float32,
                                              shape=demands_shape,
                                              name='demands-ph',
                                              is_sparse=False)
        adj_ph = model.create_placeholder(dtype=tf.float32,
                                          shape=adj_shape,
                                          name='adj-ph',
                                          is_sparse=self.params['sparse'])
        dropout_keep_ph = model.create_placeholder(dtype=tf.float32,
                                                   shape=(),
                                                   name='dropout-keep-ph',
                                                   is_sparse=False)
        capacity_ph = model.create_placeholder(dtype=tf.float32,
                                               shape=capacity_shape,
                                               name='capacity-ph',
                                               is_sparse=self.params['sparse'])

        neighborhoods = []
        for i in range(num_neighborhoods+1):
            ph = model.create_placeholder(dtype=tf.float32,
                                          shape=neighborhood_shape,
                                          name='neighborhood-{0}-ph'.format(i),
                                          is_sparse=self.params['sparse'])
            neighborhoods.append(ph)

        return {
            'node_features': node_ph,
            'demands': demands_ph,
            'neighborhoods': neighborhoods,
            'adj': adj_ph,
            'capacities': capacity_ph,
            'num_output_features': num_nodes,
            'dropout_keep_prob': dropout_keep_ph,
            'num_nodes': num_nodes,
            'should_correct_flows': True
        }

    def create_feed_dict(self, placeholders, batch, index, batch_size, data_series):
        node_features = batch[DataSeries.NODE]
        adj = batch[DataSeries.ADJ]
        demands = batch[DataSeries.DEMAND]
        neighborhoods = batch[DataSeries.NEIGHBORHOOD]
        capacities = batch[DataSeries.CAPACITY]
        dropout_keep = self.params['dropout_keep_prob']

        if data_series is not Series.TRAIN:
            node_features = node_features[index]
            adj = adj[index]
            demands = demands[index]
            neighborhoods = neighborhoods[index]
            capacities = capacities[index]
            dropout_keep = 1.0

        if not self.params['sparse']:
            adj = [a.todense() for a in adj]
            capacities = [cap.todense() for cap in capacities]

        feed_dict = {
            placeholders['node_features']: node_features,
            placeholders['demands']: demands,
            placeholders['adj']: adj,
            placeholders['dropout_keep_prob']: dropout_keep,
            placeholders['capacities']: capacities
        }

        # Provide neighborhood matrices
        for j in range(self.params['num_neighborhoods']+1):

            # Get the jth neighborhood for each element in the batch
            if not self.params['sparse']:
                neighborhood = [n[j].todense() for n in neighborhoods]

            neighborhood_ph = placeholders['neighborhoods'][j]
            feed_dict[neighborhood_ph] = neighborhood

        return feed_dict

    def create_model(self, params):
        return NeighborhoodModel(params=params)
