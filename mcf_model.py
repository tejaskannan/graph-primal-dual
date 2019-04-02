import tensorflow as tf
import numpy as np
from base_model import Model
from layers import MLP, GAT, Gate, MinCostFlow
from cost_functions import get_cost_function
from constants import BIG_NUMBER


class MCFModel(Model):

    def __init__(self, params, name='sparse-mcf-model'):
        super(MCFModel, self).__init__(params, name)
        self.cost_fn = get_cost_function(name=params['cost_fn'],
                                         constant=params['cost_constant'])

    def build(self, **kwargs):

        # V x 1 tensor which contains node features
        demands = kwargs['demands']

        # V x D' tensor which contains pre-computed node embeddings
        node_embeddings = kwargs['node_embeddings']

        # V x V tensor containing the adjacency matrix
        adj = kwargs['adj']

        # V x V tensor contain the node biases for softmax
        node_bias = kwargs['node_bias']

        num_output_features = kwargs['num_output_features']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                # Node encoding
                encoder = MLP(hidden_sizes=self.params['encoder_hidden'],
                              output_size=self.params['node_encoding'],
                              activation=tf.nn.relu,
                              name='node-encoder')
                node_encoding = encoder(inputs=tf.concat([node_embeddings, demands], axis=1),
                                        dropout_keep_prob=self.params['weight_dropout_keep'])

                node_gat = GAT(input_size=self.params['node_encoding'],
                               output_size=self.params['node_encoding'],
                               num_heads=self.params['num_heads'],
                               dims=2,
                               name='node-gat')
                gate = Gate(name='node-gate')

                # Stitch together graph and gating layers
                for _ in range(self.params['graph_layers']):
                    next_encoding = node_gat(inputs=node_encoding,
                                             bias=node_bias,
                                             weight_dropout_keep=self.params['weight_dropout_keep'],
                                             attn_dropout_keep=self.params['attn_dropout_keep'])
                    node_encoding = gate(inputs=next_encoding, prev_state=node_encoding)

                # Compute flow proportions
                decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                              output_size=num_output_features,
                              activation=None,
                              name='node-decoder')
                pred_weights = decoder(inputs=node_encoding,
                                       dropout_keep_prob=self.params['weight_dropout_keep'])

                bias = -BIG_NUMBER * (1.0 - adj)
                flow_weight_pred = tf.nn.softmax(pred_weights + bias, axis=-1, name='normalized-weights')

                # Compute minimum cost flow from flow weights
                mcf_solver = MinCostFlow(flow_iters=self.params['flow_iters'], dims=2)
                flow = mcf_solver(inputs=flow_weight_pred, demands=demands)
                flow_cost = tf.reduce_sum(self.cost_fn.apply(flow))

                # Compute Dual Problem and associated cost
                dual_decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                                   output_size=1,
                                   activation=None,
                                   name='dual-decoder')
                dual_vars = dual_decoder(inputs=node_encoding,
                                         dropout_keep_prob=self.params['weight_dropout_keep'])

                # Compute dual flows based on dual variables
                dual_diff = dual_vars - tf.transpose(dual_vars, perm=[1, 0])
                dual_flows = adj * tf.nn.relu(self.cost_fn.inv_derivative(dual_diff))

                dual_demand = tf.reduce_sum(dual_vars * demands)
                dual_flow_cost = self.cost_fn.apply(dual_flows) - dual_diff * dual_flows
                dual_cost = tf.reduce_sum(dual_flow_cost) - dual_demand

                self.loss_op = flow_cost - dual_cost
                self.output_ops += [flow_cost, flow, flow_weight_pred, dual_cost, dual_vars, dual_flows, dual_demand, dual_diff]
                self.optimizer_op = self._build_optimizer_op()
