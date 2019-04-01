import tensorflow as tf
from base_model import Model
from layers import GAT, Gate, MLP
from constants import BIG_NUMBER, FLOW_THRESHOLD
from cost_functions import tf_cost_functions


class MinCostFlowModel(Model):

    def __init__(self, params, name='max-cost-flow-model'):
        super(MinCostFlowModel, self).__init__(params, name)
        self.cost_fn = tf_cost_functions[params['cost_fn']]

    def build(self, **kwargs):

        # B x V x D tensor which contains node features
        node_input = kwargs['node_input']

        # V x D' tensor which contains pre-computed node embeddings
        node_embeddings = kwargs['node_embeddings']

        # V x V tensor which contains a neighborhood mask for each node
        node_bias = kwargs['node_bias']

        # V x V tensor containing the adjacency matrix
        adj = kwargs['adj']

        num_input_features = kwargs['num_input_features']
        num_output_features = kwargs['num_output_features']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                # Concatenate embeddings and explicit features together
                node_embeddings = tf.expand_dims(node_embeddings, axis=0)
                node_embeddings = tf.tile(node_embeddings, multiples=(tf.shape(node_input)[0], 1, 1))
                node_concat = tf.concat([node_input, node_embeddings], axis=2, name='node-concat')

                # Encoder
                node_encoder = MLP(hidden_sizes=[],
                                   output_size=self.params['node_encoding'],
                                   activation=tf.nn.relu,
                                   name='node-encoder')
                node_encoding = node_encoder(inputs=node_concat)

                # Graph attention layer
                node_gat = GAT(input_size=self.params['node_encoding'],
                               output_size=self.params['node_encoding'],
                               activation=tf.nn.relu,
                               num_heads=self.params['num_heads'],
                               name='node-GAT')

                # Node Gating
                node_gate = Gate(name='node-gate')

                # Process using Graph Attention Layers
                graph_layers = []
                for _ in range(self.params['graph_layers']):
   
                    node_output = node_gat(node_encoding, bias=node_bias)

                    # Gate output from attention layer
                    node_encoding = node_gate(inputs=(node_encoding, node_output))

                # Min Cost Flow computation
                decoder = MLP(hidden_sizes=[],
                              output_size=num_output_features,
                              activation=None,
                              name='node-decoder')
                pred_weights = decoder(inputs=node_encoding)

                # B x |V| x |V| matrix of flow proportions
                identity = tf.eye(num_output_features) * BIG_NUMBER
                flow_weight_pred = tf.nn.softmax(pred_weights + node_bias - identity,
                                                 name='normalized-weights')

                def body(flow, prev_flow):
                    inflow = tf.expand_dims(tf.reduce_sum(adj * flow, axis=1), axis=2)
                    adjusted_inflow = tf.nn.relu(inflow - node_input, name='adjust-inflow')
                    prev_flow = flow
                    flow = flow_weight_pred * adjusted_inflow
                    return [flow, prev_flow]

                def cond(flow, prev_flow):
                    return tf.reduce_any(tf.abs(flow - prev_flow) > FLOW_THRESHOLD)

                # Iteratively computes flows from flow proportions
                flow = tf.zeros_like(flow_weight_pred, dtype=tf.float32)
                prev_flow = flow + BIG_NUMBER
                shape_invariants = [flow.get_shape(), prev_flow.get_shape()]
                flow, pflow = tf.while_loop(cond, body,
                                            loop_vars=[flow, prev_flow],
                                            parallel_iterations=1,
                                            shape_invariants=shape_invariants,
                                            maximum_iterations=self.params['flow_iters'])

                self.flow = flow
                self.flow_cost = tf.reduce_sum(self.cost_fn.apply(flow), axis=[1, 2])
                
                # Compute loss from the dual problem
                dual_alphas = tf.layers.dense(inputs=node_encoding,
                                              units=1,
                                              name='flow-dual-alphas')
                dual_betas = tf.layers.dense(inputs=node_encoding,
                                             units=num_output_features,
                                             activation=tf.nn.relu,
                                             name='flow-dual-betas')

                # B x V x V tensor which contains the pairwise difference between
                # dual variables. This matrix is masked to remove flow values on 
                # non-existent edges.
                dual_diff = tf.transpose(dual_alphas, perm=[0, 2, 1]) - dual_alphas + dual_betas
                dual_flows = adj * self.cost_fn.inv_derivative(dual_diff)

                dual_inflow = tf.reduce_sum(dual_flows, axis=1, keepdims=True)
                dual_outflow = tf.reduce_sum(dual_flows, axis=2, keepdims=True)
                dual_penalty = tf.reduce_sum(dual_alphas * (dual_inflow - dual_outflow - node_input),
                                             axis=[1, 2])
                dual_penalty = dual_penalty - tf.reduce_sum(dual_betas * dual_flows, axis=[1, 2])

                # B x 1 Tensor which contains the dual cost
                dual_flow_cost = tf.reduce_sum(self.cost_fn.apply(dual_flows), axis=[1, 2])
                self.dual_cost = dual_flow_cost + dual_penalty

                loss = tf.square(self.flow_cost - self.dual_cost) + 0.1 * self.flow_cost 
                self.loss_op = tf.reduce_mean(loss)

                self.output_ops = [self.flow_cost, self.flow, self.dual_cost, dual_flows, flow_weight_pred, node_input]
                self.optimizer_op = self._build_optimizer_op()
