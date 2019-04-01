import tensorflow as tf
import numpy as np
from base_model import Model
from layers import MLP, SparseGAT, SparseMinCostFlow
from cost_functions import tf_cost_function


class SparseMCFModel(Model):

    def __init__(self, params, name='sparse-mcf-model'):
        super(SparseMCFModel, self).__init__(params, name)
        self.cost_fn = tf_cost_functions[params['cost_fn']]

    def build(self, **kwargs):

        # V x 1 tensor which contains node features
        demands = kwargs['demands']

        # V x D' tensor which contains pre-computed node embeddings
        node_embeddings = kwargs['node_embeddings']

        # V x V sparse tensor containing the adjacency matrix
        adj = kwargs['adj']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                node_gat = SparseGAT(input_size=self.params['node_encoding'],
                                     output_size=self.params['node_encoding'],
                                     num_heads=self.params['num_heads'],
                                     name='node-gat')
                gru_cell = tf.keras.layers.GRUCell(units=self.params['node_encoding'],
                                                   activation=tf.nn.relu,
                                                   name='node-gru-cell')
                gru_rnn = tf.keras.layers.RNN(cell=gru_cell,
                                              return_state=False,
                                              stateful=True,
                                              name='node-gru-rnn')

                # Stitch together graph and gating layers
                node_encoding = node_embeddings
                for _ in range(self.params['graph_layers']):
                    next_encoding = node_gat(inputs=node_encoding, adj_matrix=adj)
                    node_encoding = gru_rnn(inputs=next_encoding, name='node-gru-update')

                # Compute flow proportions
                decoder = MLP(hidden_sizes=[],
                              output_size=num_output_features,
                              activation=None,
                              name='node-decoder')
                node_concat = tf.concat([node_encoding, demands], axis=1)
                pred_weights = decoder(inputs=node_concat)

                flow_weight_pred = tf.sparse.softmax(adj * pred_weights, name='normalized-weights')
                flow_weight_pred = tf.sparse_to_dense(flow_weight_pred)

                # Compute minimum cost flow from flow weights
                mcf_solver = SparseMCFModel(flow_iters=self.params['flow_iters'])
                self.flow = mcf_solver(inputs=flow_weight_pred, adj=adj, demands=demands)
                self.flow_cost = tf.reduce_sum(self.cost_fn(self.flow))

                self.loss_op = self.flow_cost
                self._build_optimizer_op()
