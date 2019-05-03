import tensorflow as tf
from model_runners.model_runner import ModelRunner
from models.gat_model import GATModel
from core.dataset import DataSeries, Series
from utils.utils import features_to_demands, gcn_aggregator


class GCNRunner(ModelRunner):

    def create_placeholders(self, model, num_nodes, embedding_size, **kwargs):

        node_features_ph = model.create_placeholder(dtype=tf.float32,
                                                    shape=[None, num_nodes, self.num_node_features],
                                                    name='node-features-ph',
                                                    is_sparse=False)
        node_embedding_ph = model.create_placeholder(dtype=tf.float32,
                                                     shape=[None, num_nodes, embedding_size],
                                                     name='node-embedding-ph',
                                                     is_sparse=False)
        adj_ph = model.create_placeholder(dtype=tf.float32,
                                          shape=[None, num_nodes, num_nodes],
                                          name='adj-ph',
                                          is_sparse=False)
        node_agg_ph = model.create_placeholder(dtype=tf.float32,
                                               shape=[None, num_nodes, num_nodes],
                                               name='node-agg-ph',
                                               is_sparse=False)
        demands_ph = model.create_placeholder(dtype=tf.float32,
                                              shape=[None, num_nodes, 1],
                                              name='demands-ph',
                                              is_sparse=False)
        dropout_keep_ph = model.create_placeholder(dtype=tf.float32,
                                                   shape=(),
                                                   name='dropout-keep-ph',
                                                   is_sparse=False)
        return {
            'node_embeddings': node_embedding_ph,
            'node_features': node_features_ph,
            'demands': demands_ph,
            'adj': adj_ph,
            'node_agg': node_agg_ph,
            'dropout_keep_prob': dropout_keep_ph,
            'should_correct_flows': True
        }

    def create_feed_dict(self, placeholders, batch, index, batch_size, data_series):
        node_features = batch[DataSeries.NODE]
        adj = batch[DataSeries.ADJ]
        node_embeddings = batch[DataSeries.EMBEDDING]
        dropout_keep = self.params['dropout_keep_prob']

        if data_series is not Series.TRAIN:
            node_features = node_features[index]
            adj = adj[index]
            node_embeddings = node_embeddings[index]
            dropout_keep = 1.0

        if batch_size == 1:
            demands = [features_to_demands(node_features)]
            node_features = [node_features]
            adj = [adj.todense()]
            node_embeddings = [node_embeddings]
        else:
            demands = [features_to_demands(n) for n in node_features]
            adj = [a.todense() for a in adj]

        node_agg = [gcn_aggregator(a) for a in adj]

        feed_dict = {
            placeholders['node_features']: node_features,
            placeholders['demands']: demands,
            placeholders['adj']: adj,
            placeholders['node_agg']: node_agg,
            placeholders['node_embeddings']: node_embeddings,
            placeholders['dropout_keep_prob']: dropout_keep
        }

        return feed_dict

    def create_model(self, params):
        return GATModel(params=params)