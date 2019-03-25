import tensorflow as tf
from base_model import Model
from layers import GAT, Gate
from constants import BIG_NUMBER, FLOW_THRESHOLD
from cost_functions import tf_cost_functions


class MinCostFlowModel(Model):

    def __init__(self, params, name='max-cost-flow-model'):
        super(MinCostFlowModel, self).__init__(params, name)
        self.cost_fn = tf_cost_functions[params['cost_fn']]

    def build(self, **kwargs):

        # B x V x D tensor which contains node features
        node_input = kwargs['node_input']
        
        # V x V tensor which contains a neighborhood mask for each node
        node_bias = kwargs['node_bias']

        # V x V sparse tensor containing the adjacency matrix
        adj = kwargs['adj']

        num_input_features = kwargs['num_input_features']
        num_output_features = kwargs['num_output_features']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                # Encoder
                node_encoding = tf.layers.dense(inputs=node_input,
                                                units=self.params['node_encoding'],
                                                activation=tf.nn.tanh,
                                                name='node-encoder')

                # Process using Graph Attention Layers
                graph_layers = []
                for _ in range(self.params['graph_layers']):
                    # Graph attention layer
                    node_gat_layer = GAT(input_size=self.params['node_encoding'],
                                         output_size=self.params['node_encoding'],
                                         num_heads=self.params['num_heads'],
                                         name='node-GAT')
                    node_gat_output = node_gat_layer.build(node_encoding, bias=node_bias)

                    # Apply non-linearity
                    node_output = tf.nn.tanh(node_gat_output)

                    # Gate output form attention layer
                    node_gate_layer = Gate(name='node-gate')
                    node_encoding = node_gate_layer.build(inputs=(node_encoding, node_output))

                # Decoder
                pred_weights = tf.layers.dense(inputs=node_encoding,
                                               units=num_output_features,
                                               activation=tf.nn.tanh,
                                               name='node-decoder')

                # Min Cost Flow computation
                # B x |V| x |V| matrix of flow proportions
                identity = tf.eye(num_output_features) * BIG_NUMBER
                flow_weight_pred = tf.nn.softmax(pred_weights + node_bias - identity,
                                                 name='normalized-weights')

                flow = tf.zeros_like(flow_weight_pred, dtype=tf.float32)
                prev_flow = flow + BIG_NUMBER

                def body(flow, prev_flow):
                    masked_flow = tf.transpose(adj * flow, perm=[0, 2, 1], name='mask-flow')
                    inflow = tf.reduce_sum(masked_flow, keepdims=True, axis=2)
                    adjusted_inflow = tf.nn.relu(inflow - node_input, name='adjust-inflow')
                    flow = flow_weight_pred * adjusted_inflow
                    return [flow, prev_flow]

                def cond(flow, prev_flow):
                    return tf.reduce_any(tf.abs(flow - prev_flow) > FLOW_THRESHOLD)

                # Iteratively computes flows from flow proportions
                shape_invariants = [flow.get_shape(), prev_flow.get_shape()]
                flow, pflow = tf.while_loop(cond, body,
                                            loop_vars=[flow, prev_flow],
                                            parallel_iterations=1,
                                            shape_invariants=shape_invariants,
                                            maximum_iterations=self.params['flow_iters'])

                self.flow = flow
                self.flow_cost = tf.reduce_sum(self.cost_fn.apply(flow), axis=[1, 2])
                
                # Compute loss from the dual problem (B x V x 1 tensor)
                duals = tf.layers.dense(inputs=node_encoding,
                                        units=1,
                                        name='flow-duals')

                # B x V x V tensor which contains the pairwise difference between
                # dual variables. This matrix is masked to remove flow values on 
                # non-existent edges.
                dual_diff = duals - tf.transpose(duals, perm=[0, 2, 1])
                dual_flows = adj * self.cost_fn.inv_derivative(dual_diff)

                dual_inflow = tf.reduce_sum(dual_flows, axis=1, keepdims=True)
                dual_outflow = tf.reduce_sum(dual_flows, axis=2, keepdims=True)
                dual_penalty = tf.reduce_sum(duals * (dual_inflow - dual_outflow - node_input),
                                             axis=[1, 2])

                dual_flow_cost = tf.reduce_sum(self.cost_fn.apply(dual_flows), axis=[1, 2])
                self.dual_cost = dual_flow_cost + dual_penalty

                self.loss_op = tf.losses.mean_squared_error(self.flow_cost, self.dual_cost)

                self.output_ops = [self.flow_cost, self.flow, self.dual_cost]
                self.optimizer_op = self._build_optimizer_op()
