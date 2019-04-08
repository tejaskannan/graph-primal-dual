import tensorflow as tf
import numpy as np
from constants import BIG_NUMBER, FLOW_THRESHOLD


class Layer:

    def __init__(self, input_size, output_size, activation, name):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.name = name
        self.initializer = tf.contrib.layers.xavier_initializer()

    def __call__(self, inputs, **kwargs):
        raise NotImplementedError()


class MLP(Layer):
    """
    Multi-layer perceptron with dropout.
    """

    def __init__(self, hidden_sizes, output_size, activation=None,
                 activate_final=False, bias_final=True, name='dense'):
        super(MLP, self).__init__(0, output_size, activation, name)
        self.hidden_sizes = hidden_sizes
        self.activate_final = activate_final
        self.name = name
        self.bias_final = bias_final

    def __call__(self, inputs, **kwargs):
        # Dropout keep probability if passed
        dropout_keep_prob = kwargs['dropout_keep_prob'] if 'dropout_keep_prob' in kwargs else 1.0

        with tf.name_scope(self.name):

            # Hidden layers
            tensors = inputs
            for i, hidden_size in enumerate(self.hidden_sizes):
                hidden = tf.layers.dense(inputs=tensors,
                                         units=hidden_size,
                                         kernel_initializer=self.initializer,
                                         activation=self.activation,
                                         name='{0}-layer-{1}'.format(self.name, i))
                tensors = tf.nn.dropout(x=tensors,
                                        keep_prob=dropout_keep_prob,
                                        name='{0}-layer-{1}-dropout'.format(self.name, i))

            # Output layer
            final_activation = self.activation if self.activate_final else None
            output = tf.layers.dense(inputs=tensors,
                                     units=self.output_size,
                                     kernel_initializer=self.initializer,
                                     activation=final_activation,
                                     use_bias=self.bias_final,
                                     name='{0}-output'.format(self.name))
            output = tf.nn.dropout(x=output,
                                   keep_prob=dropout_keep_prob,
                                   name='{0}-output-dropout'.format(self.name))

            return output


class GRU(Layer):

    def __init__(self, output_size, activation=tf.nn.tanh, name='GRU'):
        super(GRU, self).__init__(0, output_size, activation, name)

    def __call__(self, inputs, **kwargs):
        """
        inputs and state must be 2D tensors
        """
        dropout_keep_prob = kwargs['dropout_keep_prob'] if 'dropout_keep_prob' in kwargs else 1.0
        state = kwargs['state']

        with tf.name_scope(self.name):
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.output_size,
                                              activation=self.activation,
                                              reuse=True,
                                              kernel_initializer=self.initializer,
                                              name='{0}-gru'.format(self.name),
                                              dtype=tf.float32)
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,
                                                     state_keep_prob=dropout_keep_prob)
            new_state = rnn_cell(inputs=inputs, state=state)[1]
        return new_state


class SparseGAT(Layer):
    """
    Sparse Graph Attention Layer from https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_size, output_size, num_heads, activation=tf.nn.relu, name='GAT'):
        super(SparseGAT, self).__init__(input_size, output_size, activation, name)
        self.num_heads = num_heads

    def __call__(self, inputs, **kwargs):
        adj_matrix = kwargs['adj_matrix']
        weight_dropout_keep = kwargs['weight_dropout_keep'] if 'weight_dropout_keep' in kwargs else 1.0
        attn_dropout_keep = kwargs['attn_dropout_keep'] if 'attn_dropout_keep' in kwargs else 1.0

        with tf.name_scope(self.name):
            heads = []
            for i in range(self.num_heads):
                tensor_mlp = MLP(hidden_sizes=[],
                                 output_size=self.output_size,
                                 bias_final=False,
                                 activation=None,
                                 name='{0}-W-{1}'.format(self.name, i))
                tensors = tensor_mlp(inputs=inputs, dropout_keep_prob=weight_dropout_keep)

                attn_mlp = MLP(hidden_sizes=[],
                               output_size=1,
                               bias_final=False,
                               activation=None,
                               name='{0}-a-{1}'.format(self.name, i))
                attn_weights = attn_mlp(inputs=tensors, dropout_keep_prob=attn_dropout_keep)

                masked_1 = adj_matrix * attn_weights
                masked_2 = adj_matrix * tf.transpose(attn_weights, perm=[1, 0])

                sparse_sim_mat = tf.sparse.add(masked_1, masked_2)
                sparse_leaky_relu = tf.SparseTensor(indices=sparse_sim_mat.indices,
                                                    values=tf.nn.leaky_relu(sparse_sim_mat.values),
                                                    dense_shape=sparse_sim_mat.dense_shape)

                attn_coefs = tf.sparse.softmax(sparse_leaky_relu)
                weighted_tensors = tf.sparse.matmul(attn_coefs, tensors)
                attn_head = tf.contrib.layers.bias_add(weighted_tensors, scope='b-{0}'.format(i))
                heads.append(attn_head)

            # Average over all attention heads
            return self.activation((1.0 / self.num_heads) * tf.add_n(heads))


class GAT(Layer):
    """
    Graph Attention Layer from https://arxiv.org/abs/1710.10903
    """
    def __init__(self, input_size, output_size, num_heads, dims=3, activation=tf.nn.relu, name='GAT'):
        super(GAT, self).__init__(input_size, output_size, activation, name)
        self.num_heads = num_heads
        self.dims = dims

    def __call__(self, inputs, **kwargs):
        bias = kwargs['bias']
        weight_dropout_keep = kwargs['weight_dropout_keep'] if 'weight_dropout_keep' in kwargs else 1.0
        attn_dropout_keep = kwargs['attn_dropout_keep'] if 'attn_dropout_keep' in kwargs else 1.0

        with tf.name_scope(self.name):
            heads = []
            for i in range(self.num_heads):
                # Apply weight matrix to the set of inputs, B x V x D' Tensor
                input_mlp = MLP(hidden_sizes=[],
                                output_size=self.output_size,
                                bias_final=False,
                                activation=None,
                                name='{0}-W-{1}'.format(self.name, i))
                transformed_inputs = input_mlp(inputs=inputs, dropout_keep_prob=weight_dropout_keep)

                # Create unnormalized attention weights, B x V x V
                attn_mlp = MLP(hidden_sizes=[],
                               output_size=1,
                               bias_final=False,
                               activation=None,
                               name='{0}-a-{1}'.format(self.name, i))
                attn_weights = attn_mlp(inputs=transformed_inputs, dropout_keep_prob=attn_dropout_keep)

                if self.dims == 3:
                    attn_weights = attn_weights + tf.transpose(attn_weights, [0, 2, 1])
                else:
                    attn_weights = attn_weights + tf.transpose(attn_weights, [1, 0])

                # Compute normalized attention weights, B x V x V
                attention_coefs = tf.nn.softmax(tf.nn.leaky_relu(attn_weights) + bias, axis=-1)

                # Apply attention weights, B x V x F'
                attn_head = tf.matmul(attention_coefs, transformed_inputs)
                attn_head = tf.contrib.layers.bias_add(attn_head, scope='{0}-b-{1}'.format(self.name, i))
                heads.append(attn_head)

            # Average over all attention heads
            return self.activation((1.0 / self.num_heads) * tf.add_n(heads))


class Gate(Layer):
    """
    Skip-connection Gating layer from https://arxiv.org/pdf/1805.10988.pdf
    """
    def __init__(self, name='gate'):
        super(Gate, self).__init__(0, 0, None, name)

    def __call__(self, inputs, **kwargs):

        dropout_keep_prob = kwargs['dropout_keep_prob'] if 'dropout_keep_prob' in kwargs else 1.0
        with tf.name_scope(self.name):
            prev_state = kwargs['prev_state']
            curr_state = inputs

            prev_mlp = MLP(hidden_sizes=[],
                           output_size=1,
                           bias_final=False,
                           name='{0}-W-1'.format(self.name))
            curr_mlp = MLP(hidden_sizes=[],
                           output_size=1,
                           bias_final=False,
                           name='{0}-W-2'.format(self.name))

            prev = prev_mlp(inputs=prev_state, dropout_keep_prob=dropout_keep_prob)
            curr = curr_mlp(inputs=curr_state, dropout_keep_prob=dropout_keep_prob)

            self.z = tf.contrib.layers.bias_add(prev + curr,
                                                activation_fn=tf.math.sigmoid,
                                                scope='{0}-b'.format(self.name))

            # Apply gate
            return self.z * curr_state + (1 - self.z) * prev_state


class SparseNeighborhood(Layer):

    def __init__(self, output_size, activation=tf.nn.tanh, name='sparse-neighborhood'):
        super(SparseNeighborhood, self).__init__(0, output_size, activation, name)
        self.num_neighborhoods = num_neighborhoods

    def __call__(self, inputs, **kwargs):

        # List of 'num_neighborhoods' sparse V x V matrices
        neighborhoods = kwargs['neighborhoods']

        dropout_keep_prob = kwargs['dropout_keep_prob']

        # V x F tensor of node features
        transform_layer = MLP(hidden_sizes=[],
                              output_size=self.output_size,
                              bias_final=False,
                              activation=None,
                              name='{0}-transform'.format(self.name))
        transformed_inputs = transform_layer(inputs=inputs, dropout_keep_prob=dropout_keep_prob)

        # Layer to compute attention weights for each neighborhood
        attn_layer = MLP(hidden_sizes=[],
                 output_size=1,
                 bias_final=False,
                 activation=tf.nn.leaky_relu,
                 activate_final=True,
                 name='{0}-attn-weights'.format(self.name))

        neighborhood_features = []
        neighborhood_attn = []
        for neighborhood_mat in range(neighborhoods):

            # V x F tensor of aggregated node features over the given neighborhood
            neighborhood_sum = tf.sparse.matmul(neighborhood_mat, transformed_inputs)

            # V x 1 tensor of attention weights
            attn_weights = attn_layer(inputs=neighborhood_sum, dropout_keep_prob=dropout_keep_prob)

            neighborhood_features.append(tf.expand_dims(neighborhood_sum, axis=-1))
            neighborhood_attn.append(attn_weights)

        # V x F x K
        neighborhood_concat = tf.concat(neighborhood_features, axis=-1)

        # V x K
        attn_concat = tf.concat(attn_weights, axis=-1)

        # V x K tensor of normalized attention coefficients
        attn_coefs = tf.nn.softmax(attn_concat, axis=-1)

        # V x K x 1 tensor of normalized attention coefficients
        attn_coefs_expanded = tf.expand_dims(attn_coefs, axis=-1)

        # V x F x 1 tensor of weighted neighborhood features
        weighted_features = tf.matmul(neighborhood_concat, attn_coefs_expanded)
        weighted_features = tf.squeeze(weighted_features, axis=-1)
        weighted_features = tf.contrib.layers.bias_add(weighted_features,
                                                       scope='{0}-b'.format(self.name))

        return self.activation(weighted_features)


class MinCostFlow(Layer):

    def __init__(self, flow_iters, dims=3, name='min-cost-flow'):
        super(MinCostFlow, self).__init__(0, 0, None, name)
        self.flow_iters = flow_iters
        self.dims = dims

    def __call__(self, inputs, **kwargs):
        # adj = kwargs['adj']
        demand = kwargs['demands']
        flow_weight_pred = inputs

        def body(flow, prev_flow):
            inflow = tf.expand_dims(tf.reduce_sum(flow, axis=self.dims-2), axis=self.dims-1)
            adjusted_inflow = tf.nn.relu(inflow - demand, name='adjust-inflow')
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
                                    maximum_iterations=self.flow_iters)
        return flow


class SparseMinCostFlow(Layer):

    def __init__(self, flow_iters, name='sparse-min-cost-flow'):
        super(SparseMinCostFlow, self).__init__(0, 0, None, name)
        self.flow_iters = flow_iters

    def __call__(self, inputs, **kwargs):
        demands = kwargs['demands']
        flow_weight_pred = inputs

        def body(flow, prev):
            inflow = tf.sparse_reduce_sum(flow, axis=0)
            inflow = tf.expand_dims(inflow, axis=1)
            adjusted_inflow = tf.nn.relu(inflow - demands)
            new_flow = flow_weight_pred * adjusted_inflow
            return [new_flow, flow.values]

        def cond(flow, prev):
            return tf.reduce_any((flow.values - prev) > FLOW_THRESHOLD)

        flow = flow_weight_pred
        prev = tf.zeros_like(flow_weight_pred.values, dtype=tf.float32)

        flow, prev = tf.while_loop(cond=cond,
                                   body=body,
                                   loop_vars=[flow, prev],
                                   parallel_iterations=1,
                                   maximum_iterations=self.flow_iters,
                                   return_same_structure=True,
                                   name='{0}-flow-calculation'.format(self.name))

        return flow
