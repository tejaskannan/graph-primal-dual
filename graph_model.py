import tensorflow as tf
from base_model import Model
from layers import Dense, GAT, Gate

class MinCostFlowModel(Model):

    def build(self, inputs, **kwargs):
        num_input_features = kwargs['num_input_features']
        num_output_features = kwargs['num_output_features']
        bias = kwargs['bias']
        adj = kwargs['adj']
        demands = kwargs['demands']
        cost_fn = kwargs['cost_fn']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                # Encoder
                node_encoding_layer = Dense(input_size=num_input_features,
                                            output_size=self.params['node_encoding'],
                                            activation=tf.nn.tanh,
                                            name='encoder')
                node_encoding = node_encoding_layer.build(inputs)

                # Process using Graph Attention Layers
                graph_layers = []
                for _ in range(self.params['graph_layers']):
                    gat_layer = GAT(input_size=self.params['node_encoding'],
                                    output_size=self.params['node_encoding'],
                                    num_heads=self.params['num_heads'])
                    gat_output = gat_layer.build(node_encoding, bias=bias)

                    gate_layer = Gate()
                    node_encoding = gate_layer.build(inputs=(node_encoding, gat_output))

                # Decoder
                node_output_layer = Dense(input_size=self.params['node_encoding'],
                                          output_size=num_output_features,
                                          name='decoder')
                node_output = node_output_layer.build(node_encoding)

                # Mask out values on non-existent edges
                node_output = adj * node_output

                # Compute the cost
                self.output_op = cost_fn(node_output)

                # Create loss operation
                self.loss_op = self._build_loss_op(node_output, demands, num_output_features)


    def build_optimizer_ops(self):
        pass

    def run_train_step(self, feed_dict):
        with self._sess.graph.as_default():
            optimizer_ops = self.optimizer_ops()
            op_result = self._sess.run(output_op, feed_dict=feed_dict)
            return op_result

    def _build_loss_op(self, preds, demands, num_output_features):

        # Initialize Dual Variables
        init_beta = tf.random.uniform(shape=(num_output_features, 1), maxval=1.0)
        beta = tf.Variable(initial_value=init_beta,
                           trainable=True,
                           name='beta')

        init_gamma = tf.random.uniform(shape=(num_output_features, num_output_features), maxval=1.0)
        gamma = tf.Variable(initial_value=init_gamma,
                            trainable=True,
                            name='gamma')

        # Compute incoming and outgoing flow values, B x V x 1
        incoming = tf.reduce_sum(preds, axis=1)
        outgoing = tf.reduce_sum(preds, axis=2)

        # Compute dual losses
        demand_dual = tf.reduce_sum(incoming - outgoing - demands, axis=[1, 2])
        positive_dual = tf.reduce_sum(gamma * preds, axis=[1, 2])

        # Compute cost loss
        cost = tf.reduce_sum(preds, axis=[1, 2])

        # Compute average batch loss
        return tf.reduce_mean(cost + demand_dual - positive_dual)

