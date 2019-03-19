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
        is_primal = kwargs['is_primal']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                # Encoder
                node_encoding = tf.layers.dense(inputs=inputs,
                                                units=self.params['node_encoding'],
                                                activation=tf.nn.tanh,
                                                name='encoding')

                # Process using Graph Attention Layers
                graph_layers = []
                for _ in range(self.params['graph_layers']):
                    gat_layer = GAT(input_size=self.params['node_encoding'],
                                    output_size=self.params['node_encoding'],
                                    num_heads=self.params['num_heads'])
                    gat_output = gat_layer.build(node_encoding, bias=bias)

                    # Apply non-linearity
                    gate_output = tf.nn.tanh(gat_output)

                    gate_layer = Gate()
                    node_encoding = gate_layer.build(inputs=(node_encoding, gat_output))

                # Decoder
                node_output = tf.layers.dense(inputs=node_encoding,
                                              units=num_output_features,
                                              activation=tf.nn.relu,
                                              name='decoding')

                # Mask out values on non-existent edges
                node_output = adj * node_output

                # Compute the cost
                output_op = cost_fn(node_output)

                self.output_ops = [output_op, node_output]

                # Initialize Dual Variables. Set to be non-trainable as their updating
                # is handled manually.
                init_beta = tf.random.uniform(shape=(num_output_features, 1), maxval=1.0)
                self.beta = tf.Variable(initial_value=init_beta,
                                        trainable=False,
                                        name='beta')

                init_gamma = tf.random.uniform(shape=(num_output_features, num_output_features), maxval=1.0)
                self.gamma = tf.Variable(initial_value=init_gamma,
                                         trainable=False,
                                         name='gamma')

                # Create loss operation
                self.loss_op = self._build_loss_op(output_op, demands, num_output_features)

                # Create optimizer operation
                if is_primal:
                    self.optimizer_op = self._build_primal_optimizer_op()
                else:
                    self.optimizer_op = self._build_dual_optimizer_op()

    def _build_primal_optimizer_op(self):

        primal_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        dual_vars = [self.beta, self.gamma]

        # Compute gradients for the dual variables
        dual_grads = tf.gradients(ys=self.loss_op,
                                  xs=dual_vars,
                                  name='dual-gradients',
                                  stop_gradients=primal_vars)

        dual_op = self.apply_gradients(gradients=dual_grads,
                                       variables=dual_vars,
                                       multiplier=-1)

        # Gradient projection to enforce dual variable is always positive
        self.gamma = tf.nn.relu(self.gamma)

        # Minimize the primal variables after the dual variable maximization step
        with tf.control_dependencies([dual_op, self.gamma]):
            primal_grads = tf.gradients(ys=self.loss_op,
                                        xs=primal_vars,
                                        name='primal-gradients',
                                        stop_gradients=dual_vars)
            primal_op = self.apply_gradients(gradients=primal_grads, variables=primal_vars)
            return primal_op

    def _build_dual_optimizer_op(self):
        primal_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        dual_vars = [self.beta, self.gamma]

        primal_grads = tf.gradients(ys=self.loss_op,
                                        xs=primal_vars,
                                        name='primal-gradients',
                                        stop_gradients=dual_vars)
        primal_op = self.apply_gradients(gradients=primal_grads, variables=primal_vars)

        # Minimize the primal variables after the dual variable maximization step
        with tf.control_dependencies([primal_op]):

            # Compute gradients for the dual variables
            dual_grads = tf.gradients(ys=self.loss_op,
                                      xs=dual_vars,
                                      name='dual-gradients',
                                      stop_gradients=primal_vars)

            dual_op = self.apply_gradients(gradients=dual_grads,
                                           variables=dual_vars,
                                           multiplier=-1)
        
        # Gradient projection to enforce dual variable is always positive
        self.gamma = tf.nn.relu(self.gamma)

        with tf.control_dependencies([self.gamma]):
            return dual_op

    def run_train_step(self, feed_dict):
        with self._sess.graph.as_default():
            ops = [self.loss_op, self.optimizer_op]
            op_result = self._sess.run([ops], feed_dict=feed_dict)
            return op_result[0][0]

    def _build_loss_op(self, preds, demands, num_output_features):
        # Compute incoming and outgoing flow values, B x V x 1
        incoming = tf.expand_dims(tf.reduce_sum(preds, axis=2), axis=2)
        outgoing = tf.expand_dims(tf.reduce_sum(preds, axis=1), axis=2)

        # Compute dual losses
        demand_diff = incoming - outgoing - demands
        demand_dual = tf.reduce_sum(self.beta * demand_diff, axis=[1, 2])
        positive_dual = tf.reduce_sum(self.gamma * preds, axis=[1, 2])

        # Compute cost loss
        cost = tf.reduce_sum(preds, axis=[1, 2])

        # Compute average batch loss
        return tf.reduce_mean(cost + demand_dual - positive_dual)

