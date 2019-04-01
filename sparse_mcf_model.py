import tensorflow as tf
import numpy as np
from base_model import Model
from layers import MLP, SparseGAT, SparseMinCostFlow, Gate, MinCostFlow
from cost_functions import tf_cost_functions


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

        num_output_features = kwargs['num_output_features']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                adj_self_loops = tf.sparse.add(adj, tf.sparse.eye(num_rows=tf.shape(adj)[0]))

                # Node encoding
                encoder = MLP(hidden_sizes=[],
                              output_size=self.params['node_encoding'],
                              activation=tf.nn.relu,
                              name='node-encoder')
                node_encoding = encoder(inputs=tf.concat([node_embeddings, demands], axis=1))

                node_gat = SparseGAT(input_size=self.params['node_encoding'],
                                     output_size=self.params['node_encoding'],
                                     num_heads=self.params['num_heads'],
                                     name='node-gat')
                gate = Gate(name='node-gate')

                # Stitch together graph and gating layers
                for _ in range(self.params['graph_layers']):
                    next_encoding = node_gat(inputs=node_encoding, adj_matrix=adj_self_loops)
                    node_encoding = gate(inputs=next_encoding, prev_state=node_encoding)

                # Compute flow proportions
                decoder = MLP(hidden_sizes=[],
                              output_size=num_output_features,
                              activation=None,
                              name='node-decoder')
                pred_weights = decoder(inputs=node_encoding)

                flow_weight_pred = tf.sparse.softmax(adj * pred_weights, name='normalized-weights')

                # Compute minimum cost flow from flow weights
                mcf_solver = SparseMinCostFlow(flow_iters=self.params['flow_iters'])
                flow = mcf_solver(inputs=flow_weight_pred, demands=demands)

                flow_cost = tf.reduce_sum(self.cost_fn.apply(flow))

                self.loss_op = tf.reduce_sum(flow)
                self.output_ops += [flow_cost, flow, flow_weight_pred]
                self.optimizer_op = self._build_optimizer_op()
