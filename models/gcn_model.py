import tensorflow as tf
from models.base_model import Model
from core.layers import MLP, MinCostFlow, DualFlow, SparseMax
from utils.constants import BIG_NUMBER
from cost_functions.cost_functions import get_cost_function


class GCNModel(Model):

    def __init__(self, params, name='gat-model'):
        super(GCNModel, self).__init__(params, name)
        self.cost_fn = get_cost_function(cost_fn=params['cost_fn'])

    def build(self, **kwargs):

        # B x V x D tensor
        node_embeddings = kwargs['node_embeddings']

        # B x V x D' tensor
        node_features = kwargs['node_features']

        # B x V x 1 tensor
        demands = kwargs['demands']

        # B x V x V tensor
        adj = kwargs['adj']

        # B x V x V tensor
        node_agg = kwargs['node_agg']

        # Scalar Tensor
        dropout_keep_prob = kwargs['dropout_keep_prob']

        # Boolean
        should_correct_flows = kwargs['should_correct_flows']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                node_concat = tf.concat([node_embeddings, node_features], axis=-1)

                node_encoding_mlp = MLP(hidden_sizes=self.params['encoder_hidden'],
                                        output_size=self.params['node_encoding'],
                                        activation=tf.nn.relu,
                                        activate_final=True,
                                        name='node-encoder')
                node_encoding = node_encoding_mlp(inputs=node_concat,
                                                  dropout_keep_prob=dropout_keep_prob)

                for i in range(self.params['graph_layers']):
                    node_mlp = MLP(hidden_sizes=[],
                               output_size=self.params['node_encoding'],
                               activation=None,
                               name='node-mlp-{0}'.format(i))
                    node_aggregation = tf.matmul(node_agg, node_transform)
                    node_encoding = tf.nn.relu(node_aggregation)

                node_decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                                   output_size=1,
                                   activation=tf.nn.relu,
                                   activate_final=False,
                                   name='node-decoder')
                node_weights = node_decoder(inputs=node_encoding,
                                            dropout_keep_prob=dropout_keep_prob)

                pred_weights = adj * tf.transpose(node_weights, perm=[0, 2, 1])

                bias = -BIG_NUMBER * (1.0 - adj)
                if self.params['use_sparsemax']:
                    sparsemax = SparseMax(epsilon=1e-5, is_sparse=False)

                    # Add self-loops to the bias
                    flow_weight_pred = sparsemax(inputs=pred_weights+bias,
                                                 mask=adj)
                else:
                    flow_weight_pred = tf.nn.softmax(pred_weights+bias, axis=-1)

                mcf_solver = MinCostFlow(flow_iters=self.params['flow_iters'])
                flow = mcf_solver(inputs=flow_weight_pred, demands=demands)

                if should_correct_flows:
                    flow = flow - adj * tf.math.minimum(flow, tf.transpose(flow, perm=[0, 2, 1]))

                flow_cost = tf.reduce_sum(self.cost_fn.apply(flow), axis=[1, 2])

                # Compute Dual Problem and associated cost
                dual_decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                                   output_size=1,
                                   activation=tf.nn.relu,
                                   activate_final=False,
                                   name='dual-decoder')
                dual_vars = dual_decoder(inputs=node_encoding)

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
                self.output_ops += [flow_cost, flow, flow_weight_pred, dual_cost, dual_flows, node_weights]
                self.optimizer_op = self._build_optimizer_op()
