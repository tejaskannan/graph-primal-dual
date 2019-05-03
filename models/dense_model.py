import tensorflow as tf
from models.base_model import Model
from core.layers import MLP, MinCostFlow
from cost_functions.cost_functions import get_cost_function
from utils.constants import BIG_NUMBER


class DenseModel(Model):

    def __init__(self, params, name='dense-model'):
        super(DenseModel, self).__init__(params, name)
        self.cost_fn = get_cost_function(name=params['cost_fn'],
                                         constant=params['cost_constant'])

    def build(self, **kwargs):

        # V x 1 tensor which contains node demands
        demands = kwargs['demands']

        # B x V x F tensor which contains node features
        node_features = kwargs['node_features']

        # V x V tensor containing the adjacency matrix
        adj = kwargs['adj']

        num_output_features = kwargs['num_output_features']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                # Match dimensions
                adj = tf.expand_dims(adj, axis=0)

                # Node encoding
                network = MLP(hidden_sizes=self.params['decoder_hidden'],
                              output_size=num_output_features,
                              activation=tf.nn.relu,
                              activate_final=False,
                              bias_final=False,
                              name='node-encoder')
                pred_weights = network(inputs=node_features)

                # Compute minimum cost flow from flow weights
                weights = (-BIG_NUMBER * (1.0 - adj)) + pred_weights
                flow_weight_pred = tf.nn.softmax(weights, axis=-1, name='normalized-weights')
                mcf_solver = MinCostFlow(flow_iters=self.params['flow_iters'])
                
                flow = mcf_solver(inputs=flow_weight_pred, demands=demands)

                flow_cost = tf.reduce_sum(self.cost_fn.apply(flow), axis=[1, 2])

                self.loss = flow_cost
                self.loss_op = tf.reduce_mean(flow_cost)
                self.output_ops += [flow_cost, flow, flow_weight_pred]
                self.optimizer_op = self._build_optimizer_op()
