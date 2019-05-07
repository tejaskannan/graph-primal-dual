import numpy as np
import tensorflow as tf
from utils.utils import sparse_matrix_to_tensor, features_to_demands
from utils.utils import neighborhood_adj_lists, pad_adj_list
from utils.utils import neighborhood_batch
from utils.constants import BIG_NUMBER, LINE
from core.dataset import DatasetManager, Series, DataSeries
from model_runners.model_runner import ModelRunner
from models.adj_neighborhood_model import AdjModel


class AdjRunner(ModelRunner):

    def create_placeholders(self, model, num_nodes, embedding_size, **kwargs):
        num_neighborhoods = kwargs['num_neighborhoods']
        max_degree = kwargs['max_degree']
        degrees = kwargs['degrees']

        # Placeholder shapes
        b = self.params['batch_size']
        node_shape = [b, num_nodes+1, self.num_node_features]
        demands_shape = [b, num_nodes+1, 1]
        adj_shape = [b, num_nodes+1, max_degree]
        embedding_shape = [b, num_nodes+1, embedding_size]
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
        for i in range(self.params['num_neighborhoods']+1):
            ph = model.create_placeholder(dtype=tf.int32,
                                          shape=[b, num_nodes+1, degrees[i]],
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
            'num_output_features': num_nodes,
            'dropout_keep_prob': dropout_keep_ph,
            'num_nodes': num_nodes_ph,
            'should_correct_flows': True
        }

    def create_feed_dict(self, placeholders, batch, index, batch_size, data_series):
        node_features = batch[DataSeries.NODE]
        adj = batch[DataSeries.ADJ]
        node_embeddings = batch[DataSeries.EMBEDDING]
        dropout_keep = self.params['dropout_keep_prob']

        # Handles differneces with online batch selection representation
        if data_series is not Series.TRAIN:
            node_features = node_features[index]
            adj = adj[index]
            node_embeddings = node_embeddings[index]
            dropout_keep = 1.0

        max_degree = int(placeholders['adj_lst'].get_shape()[2])

        demands = np.array([features_to_demands(n) for n in node_features])

        neighborhoods, num_nodes = neighborhood_batch(adj_matrices=adj,
                                                      k=self.params['num_neighborhoods']+1,
                                                      unique_neighborhoods=self.params['unique_neighborhoods'])

        n_nodes = np.max(num_nodes)

        adj_lists = [neighborhood_adj_lists(mat, 1, False)[0][1] for mat in adj]
        adj_tensors = np.array([pad_adj_list(adj_lst, max_degree, n_nodes) for adj_lst in adj_lists])

        inv_adj_lists = [neighborhood_adj_lists(mat, 1, False, True)[0][1] for mat in adj]
        inv_adj_tensors = np.array([pad_adj_list(adj_lst, max_degree, n_nodes) for adj_lst in inv_adj_lists])

        # 2D indexing used to extract inflow
        flow_indices = np.zeros(shape=(np.prod(adj_tensors.shape), 3))
        out_indices = np.zeros(shape=(np.prod(adj_tensors.shape), 3))
        
        index_a = 0
        index_b = 0
        for x in range(adj_tensors.shape[0]):
            for y in range(adj_tensors.shape[1]):
                for i, z in enumerate(inv_adj_tensors[x, y]):
                    indexof = np.where(adj_tensors[x, z] == y)[0]

                    flow_indices[index_a, 0] = x

                    if len(indexof) > 0:
                        flow_indices[index_a, 1] = z
                        flow_indices[index_a, 2] = indexof[0]
                    else:
                        flow_indices[index_a, 1] = num_nodes[x]
                        flow_indices[index_a, 2] = max_degree-1

                    index_a += 1

                for i, z in enumerate(adj_tensors[x, y]):
                    # Reverse Edge
                    indexof = np.where(adj_tensors[x, z] == y)[0]

                    out_indices[index_b, 0] = x
                    if len(indexof) > 0:
                        out_indices[index_b, 1] = z
                        out_indices[index_b, 2] = indexof[0]
                    else:
                        out_indices[index_b, 1] = num_nodes[x]
                        out_indices[index_b, 2] = max_degree-1

                    index_b += 1

        # Add dummy embeddings, features and demands
        demands = np.insert(demands, demands.shape[1], 0, axis=1)

        node_features = np.array(node_features)
        node_features = np.insert(node_features, node_features.shape[1], 0, axis=1)

        node_embeddings = np.array(node_embeddings)
        node_embeddings = np.insert(node_embeddings, node_embeddings.shape[1], 0, axis=1)

        feed_dict = {
            placeholders['node_features']: node_features,
            placeholders['demands']: demands,
            placeholders['adj_lst']: adj_tensors,
            placeholders['node_embeddings']: node_embeddings,
            placeholders['dropout_keep_prob']: dropout_keep,
            placeholders['num_nodes']: num_nodes,
            placeholders['flow_indices']: flow_indices,
            placeholders['out_indices']: out_indices
        }

        for i, nbrhood in enumerate(neighborhoods):
            ph = placeholders['neighborhoods'][i]
            feed_dict[ph] = nbrhood

        return feed_dict

    def create_model(self, params):
        return AdjModel(params=params)
