import tensorflow as tf
from base_model import Model
from layers import Dense, GAT, Gate

class MaxFlowModel(Model):

    def build(self, inputs, **kwargs):

        node_input = inputs[0]
        edge_input = inputs[1]

        num_input_features = kwargs['num_input_features']
        num_output_features = kwargs['num_output_features']
        node_bias = kwargs['node_bias']
        edge_bias = kwargs['edge_bias']
        source_mask = kwargs['source_mask']
        adj = kwargs['adj']


        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                # Encoder
                node_encoding = tf.layers.dense(inputs=node_input,
                                                units=self.params['node_encoding'],
                                                activation=tf.nn.tanh,
                                                name='node-encoding')
                edge_encoding = tf.layers.dense(inputs=edge_input,
                                                units=self.params['edge_encoding'],
                                                activation=tf.nn.tanh,
                                                name='edge-encoding')

                # Process using Graph Attention Layers
                graph_layers = []
                for _ in range(self.params['graph_layers']):
                    node_gat_layer = GAT(input_size=self.params['node_encoding'],
                                         output_size=self.params['node_encoding'],
                                         num_heads=self.params['num_heads'],
                                         name='node-gat')
                    node_gat_output = node_gat_layer.build(node_encoding, bias=node_bias)

                    edge_gat_layer = GAT(input_size=self.params['edge_encoding'],
                                         output_size=self.params['edge_encoding'],
                                         num_heads=self.params['num_heads'],
                                         name='edge-gat')
                    edge_gat_output = edge_gat_layer.build(edge_encoding, bias=edge_bias)

                    # Apply non-linearity
                    node_output = tf.nn.tanh(node_gat_output)
                    edge_output = tf.nn.tanh(edge_gat_output)

                    node_gate_layer = Gate(name='node-gate')
                    node_encoding = node_gate_layer.build(inputs=(node_encoding, node_output))

                    edge_gate_layer = Gate(name='edge-gate')
                    edge_encoding = edge_gate_layer.build(inputs=(edge_encoding, edge_output))

                # Decoder
                node_output = tf.layers.dense(inputs=node_encoding,
                                              units=num_output_features,
                                              activation=tf.nn.relu,
                                              name='decoding')

                # Max Flow Decoder Network
                edge_weights = tf.layers.dense(inputs=edge_encoding,
                                               units=1,
                                               activation=tf.math.sigmoid,
                                               name='edge-weight-decode')

                flow_vals = edge_input * edge_weights
                max_flow = tf.reduce_sum(source_mask * flow_vals, axis=[1, 2])

                self.output_ops = [max_flow, flow_vals]

                # Create loss operation
                self.loss_op = tf.reduce_mean(-max_flow)

                self.optimizer_op = self._build_optimizer_op()
