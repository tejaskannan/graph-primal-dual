import numpy as np
import tensorflow as tf
from utils.utils import sparse_matrix_to_tensor, features_to_demands
from utils.graph_utils import neighborhood_adj_lists, pad_adj_list
from utils.graph_utils import adj_matrix_to_list
from utils.utils import neighborhood_batch
from utils.constants import BIG_NUMBER, LINE
from core.dataset import DatasetManager, Series
from model_runners.model_runner import ModelRunner
from models.adj_neighborhood_model import AdjModel


class AdjRunner(ModelRunner):

    def create_placeholders(self, model, **kwargs):

        # Model parameters
        b = self.params['batch_size']
        num_neighborhoods = self.params['num_neighborhoods']

        embedding_size = kwargs['embedding_size']
        max_num_nodes = kwargs['max_num_nodes'] + 1
        max_degree = kwargs['max_degree']
        max_neighborhood_degrees = kwargs['max_neighborhood_degrees']

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
        flow_indices_ph = model.create_placeholder(dtype=tf.int32,
                                                   shape=[np.prod(adj_shape), 3],
                                                   name='flow-indices-ph',
                                                   is_sparse=False)
        out_indices_ph = model.create_placeholder(dtype=tf.int32,
                                                  shape=[np.prod(adj_shape), 3],
                                                  name='out-indices-ph',
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

        neighborhood_phs = []
        for i in range(num_neighborhoods + 1):
            shape = [b, max_num_nodes, max_neighborhood_degrees[i]]
            ph = model.create_placeholder(dtype=tf.int32,
                                          shape=shape,
                                          name='neighborhood-{0}-ph'.format(i),
                                          is_sparse=False)
            neighborhood_phs.append(ph)

        return {
            'node_features': node_ph,
            'demands': demands_ph,
            'node_embeddings': node_embedding_ph,
            'adj_lst': adj_ph,
            'neighborhoods': neighborhood_phs,
            'flow_indices': flow_indices_ph,
            'out_indices': out_indices_ph,
            'dropout_keep_prob': dropout_keep_ph,
            'num_nodes': num_nodes_ph,
            'should_correct_flows': True
        }

    def create_feed_dict(self, placeholders, batch, batch_size, data_series, **kwargs):

        # Padding parameters
        max_degree = kwargs['max_degree']
        max_num_nodes = kwargs['max_num_nodes']
        max_neighborhood_degrees = kwargs['max_neighborhood_degrees']

        # Fetch features for each sample in the given batch
        node_features = np.array([sample.node_features for sample in batch])
        demands = np.array([sample.demands for sample in batch])
        adj_lsts = np.array([sample.adj_lst for sample in batch])
        node_embeddings = np.array([sample.embeddings for sample in batch])
        num_nodes = np.array([sample.num_nodes for sample in batch])
        dropout_keep = self.params['dropout_keep_prob'] if data_series == Series.TRAIN else 1.0

        # Inverted adjacency list used to compute indexes
        inv_adj_lists = [adj_matrix_to_list(sample.adj_mat, inverted=True) for sample in batch]
        inv_adj_tensors = np.array([pad_adj_list(adj_lst, max_degree, max_num_nodes, n)
                                    for adj_lst, n in zip(inv_adj_lists, num_nodes)])

        # 2D indexing used for inflow and flow corrections
        flow_indices = np.zeros(shape=(np.prod(adj_lsts.shape), 3))
        out_indices = np.zeros(shape=(np.prod(adj_lsts.shape), 3))
        
        index_a = 0
        index_b = 0
        for x in range(adj_lsts.shape[0]):
            for y in range(adj_lsts.shape[1]):
                for i, z in enumerate(inv_adj_tensors[x, y]):
                    indexof = np.where(adj_lsts[x, z] == y)[0]

                    flow_indices[index_a, 0] = x

                    if len(indexof) > 0:
                        flow_indices[index_a, 1] = z
                        flow_indices[index_a, 2] = indexof[0]
                    else:
                        flow_indices[index_a, 1] = num_nodes[x]
                        flow_indices[index_a, 2] = max_degree-1

                    index_a += 1

                for i, z in enumerate(adj_lsts[x, y]):
                    # Reverse Edge
                    indexof = np.where(adj_lsts[x, z] == y)[0]

                    out_indices[index_b, 0] = x
                    if len(indexof) > 0:
                        out_indices[index_b, 1] = z
                        out_indices[index_b, 2] = indexof[0]
                    else:
                        out_indices[index_b, 1] = num_nodes[x]
                        out_indices[index_b, 2] = max_degree-1

                    index_b += 1


        # Add dummy embeddings, features and demands to account for added node
        demands = np.insert(demands, demands.shape[1], 0, axis=1)
        node_features = np.insert(node_features, node_features.shape[1], 0, axis=1)
        node_embeddings = np.insert(node_embeddings, node_embeddings.shape[1], 0, axis=1)

        feed_dict = {
            placeholders['node_features']: node_features,
            placeholders['demands']: demands,
            placeholders['adj_lst']: adj_lsts,
            placeholders['node_embeddings']: node_embeddings,
            placeholders['dropout_keep_prob']: dropout_keep,
            placeholders['num_nodes']: np.reshape(num_nodes, [-1, 1]),
            placeholders['flow_indices']: flow_indices,
            placeholders['out_indices']: out_indices
        }

        for i in range(self.params['num_neighborhoods'] + 1):
            neighborhood = [sample.neighborhoods[i] for sample in batch]
            ph = placeholders['neighborhoods'][i]
            feed_dict[ph] = neighborhood

        return feed_dict

    def create_model(self, params):
        return AdjModel(params=params)
