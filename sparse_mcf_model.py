import tensorflow as tf
import numpy as np
from base_model import Model
from layers import MLP, SparseGAT, SparseMinCostFlow, Gate, MinCostFlow
from cost_functions import get_cost_function


class SparseMCFModel(Model):

    def __init__(self, params, name='sparse-mcf-model'):
        super(SparseMCFModel, self).__init__(params, name)
        self.cost_fn = get_cost_function(name=params['cost_fn'],
                                         constant=params['cost_constant'])

    def build(self, **kwargs):

        # V x 1 tensor which contains node demands
        demands = kwargs['demands']

        # V x F tensor which contains node features
        node_features = kwargs['node_features']

        # V x D' tensor which contains pre-computed node embeddings
        node_embeddings = kwargs['node_embeddings']

        # V x V sparse tensor containing the adjacency matrix
        adj = kwargs['adj']

        dropout_keep_prob = kwargs['dropout_keep_prob']

        num_output_features = kwargs['num_output_features']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                adj_self_loops = tf.sparse.add(adj, tf.sparse.eye(num_rows=tf.shape(adj)[0]))

                # Node encoding
                encoder = MLP(hidden_sizes=self.params['encoder_hidden'],
                              output_size=self.params['node_encoding'],
                              activation=tf.nn.relu,
                              name='node-encoder')
                node_encoding = encoder(inputs=tf.concat([node_embeddings, node_features], axis=1))

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
                decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                              output_size=num_output_features,
                              activation=None,
                              name='node-decoder')
                pred_weights = decoder(inputs=node_encoding)

                flow_weight_pred = tf.sparse.softmax(adj * pred_weights, name='normalized-weights')

                # Compute minimum cost flow from flow weights
                mcf_solver = SparseMinCostFlow(flow_iters=self.params['flow_iters'])
                flow = mcf_solver(inputs=flow_weight_pred, demands=demands)

                # This operation assumes that the c(0) = 0
                flow_cost = tf.reduce_sum(self.cost_fn.apply(flow.values))

                # Compute Dual Problem and associated cost
                dual_decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                                   output_size=1,
                                   activation=None,
                                   name='dual-decoder')
                dual_vars = dual_decoder(inputs=node_encoding)

                # Compute dual flows based on dual variables
                # This operation is expensive (requires O(|V|) memory)
                dual_diff = dual_vars - tf.transpose(dual_vars, perm=[1, 0])
                dual_flows = adj * tf.nn.relu(self.cost_fn.inv_derivative(dual_diff))

                dual_demand = tf.reduce_sum(dual_vars * demands)
                diff_values = (dual_flows * dual_diff).values
                dual_flow_cost = self.cost_fn.apply(dual_flows.values) - diff_values
                dual_cost = tf.reduce_sum(dual_flow_cost) - dual_demand

                self.loss = flow_cost - dual_cost
                self.loss_op = flow_cost - dual_cost
                self.output_ops += [flow_cost, flow, flow_weight_pred]
                self.optimizer_op = self._build_optimizer_op()
