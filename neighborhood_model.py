import tensorflow as tf
import numpy as np
from base_model import Model
from layers import MLP, Neighborhood, SparseMinCostFlow, GRU, MinCostFlow
from layers import AttentionNeighborhood, SparseDualFlow, DualFlow, SparseMax
from cost_functions import get_cost_function
from constants import BIG_NUMBER, SMALL_NUMBER


class NeighborhoodModel(Model):

    def __init__(self, params, name='neighborhood-model'):
        super(NeighborhoodModel, self).__init__(params, name)
        self.cost_fn = get_cost_function(cost_fn=params['cost_fn'])

    def build(self, **kwargs):

        # V x 1 tensor which contains node demands
        demands = kwargs['demands']

        # V x F tensor which contains node features
        node_features = kwargs['node_features']

        # V x D' tensor which contains pre-computed node embeddings
        node_embeddings = kwargs['node_embeddings']

        # List of V x V sparse tensors representing node neighborhoods
        neighborhoods = kwargs['neighborhoods']

        # V x V sparse tensor containing the adjacency matrix
        adj = kwargs['adj']

        dropout_keep_prob = kwargs['dropout_keep_prob']

        num_output_features = kwargs['num_output_features']

        is_sparse = self.params['sparse']
        rank = 2 if is_sparse else 3

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                # Node encoding
                encoder = MLP(hidden_sizes=self.params['encoder_hidden'],
                              output_size=self.params['node_encoding'],
                              activation=tf.nn.tanh,
                              activate_final=True,
                              name='node-encoder')
                node_encoding = encoder(inputs=tf.concat([node_embeddings, node_features], axis=-1),
                                        dropout_keep_prob=dropout_keep_prob)

                node_neighborhood = AttentionNeighborhood(output_size=self.params['node_encoding'],
                                                          is_sparse=is_sparse,
                                                          num_heads=self.params['num_heads'],
                                                          activation=tf.nn.tanh,
                                                          name='node-neighborhood')
                node_gru = GRU(output_size=self.params['node_encoding'],
                               activation=tf.nn.tanh,
                               name='node-gru')

                # Combine message passing steps
                for _ in range(self.params['graph_layers']):
                    next_encoding, attn_coefs = node_neighborhood(inputs=node_encoding,
                                                                  neighborhoods=neighborhoods,
                                                                  dropout_keep_prob=dropout_keep_prob)
                    node_encoding = node_gru(inputs=next_encoding,
                                             state=node_encoding,
                                             dropout_keep_prob=dropout_keep_prob)

                # Compute flow proportions
                decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                              output_size=1,
                              activation=None,
                              name='node-decoder')
                pred_weights = decoder(inputs=node_encoding)

                # Sharpen Weight
                sharpen_init = tf.random.uniform(shape=(1,), dtype=tf.float32)
                sharpen_var = tf.Variable(initial_value=sharpen_init,
                                          trainable=True,
                                          name='sharpen-var')
                sharpen_weight = 1.0 + tf.nn.relu(sharpen_var)

                # Compute minimum cost flow from flow weights
                if is_sparse:
                    pred_weights = adj * tf.transpose(pred_weights, perm=[1, 0])
                    flow_weight_pred = tf.sparse.softmax(pred_weights, name='normalized-weights')
                    mcf_solver = SparseMinCostFlow(flow_iters=self.params['flow_iters'])
                else:
                    pred_weights = adj * tf.transpose(pred_weights, perm=[0, 2, 1])
                    weights = (-BIG_NUMBER * (1.0 - adj)) + pred_weights

                    sparsemax = SparseMax(epsilon=1e-3, name='sparsemax')
                    flow_weight_pred = sparsemax(inputs=weights, mask=adj)

                    #flow_weight_pred = tf.nn.softmax(weights, axis=-1, name='normalized-weights')
                    mcf_solver = MinCostFlow(flow_iters=self.params['flow_iters'])

                flow = mcf_solver(inputs=flow_weight_pred, demands=demands)

                # This operation assumes that the c(0) = 0
                if is_sparse:
                    flow_cost = tf.reduce_sum(self.cost_fn.apply(flow.values))
                else:
                    flow_cost = tf.reduce_sum(self.cost_fn.apply(flow), axis=[1, 2])

                # Compute Dual Problem and associated cost
                dual_decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                                   output_size=1,
                                   activation=None,
                                   name='dual-decoder')
                dual_vars = dual_decoder(inputs=node_encoding)

                # Compute dual flows based on dual variables
                if is_sparse:
                    # Creates a |V| x |V| sparse tensor of dual variables which is
                    # masked by the adjacency matrix. Performing operations in this manner
                    # prevents keeping a |V| x |V| dense tensor.
                    dual_masked = adj * dual_vars
                    dual_transpose = adj * tf.transpose(-1 * dual_vars, perm=[1, 0])
                    dual_diff = tf.sparse.add(dual_masked, dual_transpose)

                    dual_flow_layer = SparseDualFlow(step_size=self.params['dual_step_size'],
                                                     momentum=self.params['dual_momentum'],
                                                     iters=self.params['dual_iters'])
                    dual_flows = dual_flow_layer(inputs=dual_diff, adj=adj, cost_fn=self.cost_fn)

                    dual_demand = tf.reduce_sum(dual_vars * demands)

                    diff_values = dual_flows.values * dual_diff.values
                    dual_flow_cost = self.cost_fn.apply(dual_flows.values) - diff_values
                    dual_cost = tf.reduce_sum(dual_flow_cost) - dual_demand
                else:
                    dual_diff = adj * (dual_vars - tf.transpose(dual_vars, perm=[0, 2, 1]))

                    dual_flow_layer = DualFlow(step_size=self.params['dual_step_size'],
                                               momentum=self.params['dual_momentum'],
                                               iters=self.params['dual_iters'])
                    dual_flows = dual_flow_layer(inputs=dual_diff, adj=adj, cost_fn=self.cost_fn)

                    dual_demand = tf.reduce_sum(dual_vars * demands, axis=[1, 2])
                    dual_flow_cost = self.cost_fn.apply(dual_flows) - dual_diff * dual_flows
                    dual_cost = tf.reduce_sum(dual_flow_cost, axis=[1, 2]) - dual_demand

                self.loss = flow_cost - dual_cost
                self.loss_op = tf.reduce_mean(self.loss)
                self.output_ops += [flow_cost, flow, flow_weight_pred, dual_cost, dual_flows, attn_coefs, pred_weights]
                self.optimizer_op = self._build_optimizer_op()
