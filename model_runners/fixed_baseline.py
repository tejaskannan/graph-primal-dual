import numpy as np
import tensorflow as tf
from utils.graph_utils import pad_adj_list, adj_matrix_to_list
from model_runners.model_runner import ModelRunner
from models.fixed_model import FixedModel
from utils.constants import BIG_NUMBER, SMALL_NUMBER


class FixedBaseline(ModelRunner):

    def create_placeholders(self, model, **kwargs):

        # Model parameters
        b = self.params['batch_size']
        num_neighborhoods = self.params['num_neighborhoods']

        embedding_size = kwargs['embedding_size']
        max_num_nodes = kwargs['max_num_nodes'] + 1
        max_degree = kwargs['max_degree']

        # Placeholder shapes
        demands_shape = [b, max_num_nodes, 1]
        adj_shape = [b, max_num_nodes, max_degree]
        num_nodes_shape = [b, 1]

        demands_ph = model.create_placeholder(dtype=tf.float32,
                                              shape=demands_shape,
                                              name='demands-ph',
                                              is_sparse=False)
        adj_ph = model.create_placeholder(dtype=tf.int32,
                                          shape=adj_shape,
                                          name='adj-ph',
                                          is_sparse=False)
        in_indices_ph = model.create_placeholder(dtype=tf.int32,
                                                 shape=[np.prod(adj_shape), 3],
                                                 name='in-indices-ph',
                                                 is_sparse=False)
        num_nodes_ph = model.create_placeholder(dtype=tf.int32,
                                                shape=num_nodes_shape,
                                                name='num-nodes-ph',
                                                is_sparse=False)
        flow_proportions_ph = model.create_placeholder(dtype=tf.float32,
                                                       shape=adj_shape,
                                                       name='flow-props-ph',
                                                       is_sparse=False)

        return {
            'demands': demands_ph,
            'adj_lst': adj_ph,
            'in_indices': in_indices_ph,
            'num_nodes': num_nodes_ph,
            'max_num_nodes': max_num_nodes,
            'flow_proportions': flow_proportions_ph
        }

    def create_feed_dict(self, placeholders, batch, batch_size, data_series, **kwargs):

        # Padding parameters
        max_degree = kwargs['max_degree']
        max_num_nodes = kwargs['max_num_nodes']

        # Fetch features for each sample in the given batch
        demands = np.array([sample.demands for sample in batch])
        adj_lsts = np.array([sample.adj_lst for sample in batch])
        num_nodes = np.array([sample.num_nodes for sample in batch]).reshape(-1, 1)

        # 3D indexing used for flow computation and correction
        batch_indices = np.arange(start=0, stop=batch_size)
        batch_indices = np.repeat(batch_indices, adj_lsts.shape[1] * max_degree).reshape((-1, 1))

        in_indices = np.vstack([sample.in_indices for sample in batch])
        in_indices = np.concatenate([batch_indices, in_indices], axis=1)

        # Add dummy embeddings, features and demands to account for added node
        demands = np.insert(demands, demands.shape[1], 0, axis=1)

        if kwargs['name'] == 'random':
            flow_proportions = np.random.uniform(size=adj_lsts.shape, low=-1.0, high=1.0)
        elif kwargs['name'] == 'uniform':
            mask = np.array(adj_lsts < np.expand_dims(num_nodes, axis=-1)).astype(float)
            out_neighbors = np.sum(mask, axis=-1, keepdims=True)
            flow_proportions = mask / np.clip(out_neighbors, a_min=SMALL_NUMBER, a_max=BIG_NUMBER)

        feed_dict = {
            placeholders['demands']: demands,
            placeholders['adj_lst']: adj_lsts,
            placeholders['num_nodes']: num_nodes,
            placeholders['in_indices']: in_indices,
            placeholders['flow_proportions']: flow_proportions
        }

        return feed_dict

    def create_model(self, params):
        return FixedModel(params=params)
