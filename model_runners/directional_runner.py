import numpy as np
import tensorflow as tf
from utils.graph_utils import pad_adj_list, adj_matrix_to_list
from core.dataset import DatasetManager, Series
from model_runners.model_runner import ModelRunner
from models.directional_model import DirectionalModel


class DirectionalRunner(ModelRunner):

    def create_placeholders(self, model, **kwargs):

        # Model parameters
        b = self.params['batch_size']

        embedding_size = kwargs['embedding_size']
        max_num_nodes = kwargs['max_num_nodes'] + 1
        max_degree = kwargs['max_degree']

        # Placeholder shapes
        node_shape = [b, max_num_nodes, self.num_node_features]
        demands_shape = [b, max_num_nodes, 1]
        adj_shape = [b, max_num_nodes, max_degree]
        embedding_shape = [b, max_num_nodes, embedding_size]
        num_nodes_shape = [b, 1]

        node_ph = model.create_placeholder(dtype=tf.float32,
                                           shape=node_shape,
                                           name='node-ph',
                                           is_sparse=False)
        demands_ph = model.create_placeholder(dtype=tf.float32,
                                              shape=demands_shape,
                                              name='demands-ph',
                                              is_sparse=False)
        adj_ph = model.create_placeholder(dtype=tf.int32,
                                          shape=adj_shape,
                                          name='adj-ph',
                                          is_sparse=False)
        inv_adj_ph = model.create_placeholder(dtype=tf.int32,
                                              shape=adj_shape,
                                              name='inv-adj-ph',
                                              is_sparse=False)
        in_indices_ph = model.create_placeholder(dtype=tf.int32,
                                                 shape=[np.prod(adj_shape), 3],
                                                 name='in-indices-ph',
                                                 is_sparse=False)
        rev_indices_ph = model.create_placeholder(dtype=tf.int32,
                                                  shape=[np.prod(adj_shape), 3],
                                                  name='rev-indices-ph',
                                                  is_sparse=False)
        node_embedding_ph = model.create_placeholder(dtype=tf.float32,
                                                     shape=embedding_shape,
                                                     name='node-embedding-ph',
                                                     is_sparse=False)
        dropout_keep_ph = model.create_placeholder(dtype=tf.float32,
                                                   shape=(),
                                                   name='dropout-keep-ph',
                                                   is_sparse=False)
        num_nodes_ph = model.create_placeholder(dtype=tf.int32,
                                                shape=num_nodes_shape,
                                                name='num-nodes-ph',
                                                is_sparse=False)

        return {
            'node_features': node_ph,
            'demands': demands_ph,
            'adj_lst': adj_ph,
            'inv_adj_lst': inv_adj_ph,
            'in_indices': in_indices_ph,
            'rev_indices': rev_indices_ph,
            'dropout_keep_prob': dropout_keep_ph,
            'num_nodes': num_nodes_ph,
            'max_num_nodes': max_num_nodes
        }

    def create_feed_dict(self, placeholders, batch, batch_size, data_series, **kwargs):

        # Padding parameters
        max_degree = kwargs['max_degree']
        max_num_nodes = kwargs['max_num_nodes']

        # Fetch features for each sample in the given batch
        node_features = np.array([sample.node_features for sample in batch])
        demands = np.array([sample.demands for sample in batch])
        adj_lsts = np.array([sample.adj_lst for sample in batch])
        inv_adj_lsts = np.array([sample.inv_adj_lst for sample in batch])
        num_nodes = np.array([sample.num_nodes for sample in batch])
        dropout_keep = self.params['dropout_keep_prob'] if data_series == Series.TRAIN else 1.0

        # 3D indexing used for flow computation and correction
        batch_indices = np.arange(start=0, stop=batch_size)
        batch_indices = np.repeat(batch_indices, adj_lsts.shape[1] * max_degree).reshape((-1, 1))

        rev_indices = np.vstack([sample.rev_indices for sample in batch])
        rev_indices = np.concatenate([batch_indices, rev_indices], axis=1)

        in_indices = np.vstack([sample.in_indices for sample in batch])
        in_indices = np.concatenate([batch_indices, in_indices], axis=1)

        # Add dummy embeddings, features and demands to account for added node
        demands = np.insert(demands, demands.shape[1], 0, axis=1)
        node_features = np.insert(node_features, node_features.shape[1], 0, axis=1)

        feed_dict = {
            placeholders['node_features']: node_features,
            placeholders['demands']: demands,
            placeholders['adj_lst']: adj_lsts,
            placeholders['inv_adj_lst']: inv_adj_lsts,
            placeholders['dropout_keep_prob']: dropout_keep,
            placeholders['num_nodes']: np.reshape(num_nodes, [-1, 1]),
            placeholders['in_indices']: in_indices,
            placeholders['rev_indices']: rev_indices
        }

        return feed_dict

    def create_model(self, params):
        return DirectionalModel(params=params)
