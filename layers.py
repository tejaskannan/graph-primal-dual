import tensorflow as tf
import numpy as np
from constants import BIG_NUMBER, FLOW_THRESHOLD


class Layer:

    def __init__(self, output_size, activation, name):
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
        super(MLP, self).__init__(output_size, activation, name)
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
                                        rate=(1.0 - dropout_keep_prob),
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
                                   rate=(1.0 - dropout_keep_prob),
                                   name='{0}-output-dropout'.format(self.name))

            return output


class GRU(Layer):

    def __init__(self, output_size, activation=tf.nn.tanh, name='GRU'):
        super(GRU, self).__init__(output_size, activation, name)

    def __call__(self, inputs, **kwargs):
        dropout_keep_prob = kwargs['dropout_keep_prob'] if 'dropout_keep_prob' in kwargs else 1.0
        state = kwargs['state']

        with tf.name_scope(self.name):

            update_gate = MLP(hidden_sizes=[],
                              output_size=self.output_size,
                              bias_final=False,
                              activation=tf.math.sigmoid,
                              activate_final=True,
                              name='{0}-update-gate'.format(self.name))
            reset_gate = MLP(hidden_sizes=[],
                             output_size=self.output_size,
                             bias_final=False,
                             activation=tf.math.sigmoid,
                             activate_final=True,
                             name='{0}-reset-gate'.format(self.name))
            hidden_gate = MLP(hidden_sizes=[],
                              output_size=self.output_size,
                              bias_final=False,
                              activation=tf.nn.tanh,
                              activate_final=True,
                              name='{0}-hidden-gate'.format(self.name))

            features_concat = tf.concat([inputs, state], axis=-1)

            update_vector = update_gate(inputs=features_concat, dropout_keep_prob=dropout_keep_prob)
            reset_vector = reset_gate(inputs=features_concat, dropout_keep_prob=dropout_keep_prob)

            hidden_concat = tf.concat([inputs, reset_vector * state], axis=-1)
            hidden_vector = hidden_gate(inputs=hidden_concat, dropout_keep_prob=dropout_keep_prob)

            new_state = (1.0 - update_vector) * state + update_vector * hidden_vector

        return self.activation(new_state)


class SparseGAT(Layer):
    """
    Sparse Graph Attention Layer from https://arxiv.org/abs/1710.10903
    """

    def __init__(self, output_size, num_heads, activation=tf.nn.relu, name='GAT'):
        super(SparseGAT, self).__init__(output_size, activation, name)
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

                attn_coefs = tf.sparse.softmax(sparse_sim_mat)
                weighted_tensors = tf.sparse.matmul(attn_coefs, tensors)
                attn_head = tf.contrib.layers.bias_add(weighted_tensors, scope='b-{0}'.format(i))
                heads.append(attn_head)

            # Average over all attention heads
            attn_heads = (1.0 / self.num_heads) * tf.add_n(heads)
            return self.activation(attn_heads)


class GAT(Layer):
    """
    Graph Attention Layer from https://arxiv.org/abs/1710.10903
    """
    def __init__(self, output_size, num_heads, dims=3, activation=tf.nn.relu, name='GAT'):
        super(GAT, self).__init__(output_size, activation, name)
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
                attention_coefs = tf.nn.softmax(attn_weights + bias, axis=-1)

                # Apply attention weights, B x V x F'
                attn_head = tf.matmul(attention_coefs, transformed_inputs)
                attn_head = tf.contrib.layers.bias_add(attn_head, scope='{0}-b-{1}'.format(self.name, i))
                heads.append(attn_head)

            # Average over all attention heads
            attn_heads = (1.0 / self.num_heads) * tf.add_n(heads)
            return self.activation(attn_heads)


class Gate(Layer):
    """
    Skip-connection Gating layer from https://arxiv.org/pdf/1805.10988.pdf
    """
    def __init__(self, name='gate'):
        super(Gate, self).__init__(0, None, name)

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


class SparseMax(Layer):

    def __init__(self, epsilon=0.0, is_sparse=False, name='sparsemax'):
        super(SparseMax, self).__init__(0, None, name)
        self.epsilon = epsilon
        self.is_sparse = is_sparse

    def __call__(self, inputs, **kwargs):
        if self.is_sparse:
            return self.sparse_op(inputs, kwargs['num_rows'])
        return self.dense_op(inputs, kwargs['mask'])

    def dense_op(self, inputs, mask):
        """
        Implementation of sparsemax for tensors of rank 3. The sparsemax
        algorithm is presented in https://arxiv.org/abs/1602.02068. The implementation
        is based on the code for tf.contrib.layers.sparsemax.sparsemax
        https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/sparsemax/python/ops/sparsemax.py
        """

        # Size of the final dimension
        dims = tf.shape(inputs)[-1]

        # The paper calls the inputs 'z', so we use the same conventions here
        z = inputs

        # Sort z vectors
        z_sorted, _ = tf.nn.top_k(z, k=dims, name='{0}-z-sort'.format(self.name))

        # Partial sums based on sorted vectors
        partial_sums = tf.cumsum(z_sorted, axis=-1, name='{0}-cumsum'.format(self.name))

        # Tensor with k values
        k = tf.range(start=1, limit=tf.cast(dims, dtype=z.dtype) + 1,
                     dtype=z.dtype, name='{0}-k'.format(self.name))

        # Tensor of ones and zeros representing which indices are greater
        # than their respective partial sums
        z_threshold = 1.0 + k * z_sorted > partial_sums

        # k(z) value
        k_z = tf.reduce_sum(tf.cast(z_threshold, dtype=tf.int32), axis=-1)

        # 2D matrix of indices
        dim0_indices = tf.range(0, tf.shape(z)[0])
        dim1_indices = tf.range(0, tf.shape(z)[1])

        indices_x, indices_y = tf.meshgrid(dim0_indices, dim1_indices)
        indices_x = tf.reshape(tf.transpose(indices_x), [-1, 1])
        indices_y = tf.reshape(tf.transpose(indices_y), [-1, 1])

        obs_indices = tf.concat([indices_x, indices_y], axis=-1)

        # k(z) indices within the final dimension of the 3D tensor
        indices = tf.concat([obs_indices, tf.reshape(k_z - 1, [-1, 1])], axis=1)

        # Partial sums less than (z)
        tau_sum = tf.gather_nd(partial_sums, indices)
        tau_sum_reshape = tf.reshape(tau_sum, tf.shape(k_z))

        # Threshold value tau(z)
        tau_z = (tau_sum_reshape - 1) / tf.cast(k_z, dtype=z.dtype)
        tau_z = tf.expand_dims(tau_z, axis=-1)

        # Take max of reduced z values with zero
        weights = tf.nn.relu(z - tau_z)

        # Renormalize values to enforce a minimum probability if needed
        if self.epsilon > 0.0:
            weights = mask * tf.clip_by_value(weights, self.epsilon, 1.0)
            weights = weights / tf.norm(weights, ord=1, axis=-1, keepdims=True)

        return weights

    def sparse_op(self, inputs, num_rows, name='sparse-sparsemax'):
        # Fetch individual rows from the sparse tensor
        partitions = tf.cast(inputs.indices[:, 0], dtype=tf.int32)
        rows = tf.dynamic_partition(inputs.values, partitions, num_rows, name='{0}-dyn-part'.format(name))

        def clipped_sparsemax(tensor, epsilon):
            # We need reshape the tensor because the provided sparsemax function requires
            # 2D tensors
            expanded_tensor = tf.expand_dims(tensor, axis=0)
            normalized_tensor = tf.contrib.sparsemax.sparsemax(logits=expanded_tensor,
                                                               name='{0}-sparsemax-op'.format(name))
            # Clip values if necessary
            if epsilon > 0.0:
                clipped = tf.clip_by_value(normalized_tensor, epsilon, 1.0)
                normalized_tensor = clipped / tf.norm(clipped, ord=1, axis=-1, keepdims=True)
            return normalized_tensor

        # Normalize rows using clipped sparsemax and set the value of all empty tensors
        # to -1 for later removal. This trick allows the function to handle zero rows.
        # It may be helpful to translate this operation into tf.while(...) but we leave
        # it as a list comprehension for simplicity.
        normalized = [tf.cond(tf.equal(tf.size(tensor), 0),
                              lambda: tf.constant(-1.0, shape=[1, 1], dtype=tf.float32),
                              lambda: clipped_sparsemax(tensor, self.epsilon)) for tensor in rows]
        concat = tf.squeeze(tf.concat(normalized, axis=1), axis=0)

        # Mask out empty entries (set to -1 from beforehand)
        mask = tf.logical_not(tf.equal(concat, -1.0))
        filtered = tf.boolean_mask(concat, mask)

        return tf.SparseTensor(
            indices=inputs.indices,
            values=filtered,
            dense_shape=inputs.dense_shape
        )


class Neighborhood(Layer):

    def __init__(self, output_size, is_sparse, activation=tf.nn.tanh, name='neighborhood'):
        super(Neighborhood, self).__init__(output_size, activation, name)
        self.is_sparse = is_sparse

    def __call__(self, inputs, **kwargs):

        # List of 'num_neighborhoods' V x V matrices
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
                         activation=None,
                         activate_final=True,
                         name='{0}-attn-weights'.format(self.name))

        neighborhood_features = []
        neighborhood_attn = []
        for neighborhood_mat in neighborhoods:

            # V x F tensor of aggregated node features over the given neighborhood
            if self.is_sparse:
                neighborhood_sum = tf.sparse.matmul(neighborhood_mat, transformed_inputs)
            else:
                neighborhood_sum = tf.matmul(neighborhood_mat, transformed_inputs)

            # V x 1 tensor of attention weights
            node_neighbor_concat = tf.concat([neighborhood_sum, transformed_inputs], axis=-1)
            attn_weights = attn_layer(inputs=node_neighbor_concat, dropout_keep_prob=dropout_keep_prob)

            neighborhood_features.append(tf.expand_dims(neighborhood_sum, axis=-1))
            neighborhood_attn.append(attn_weights)

        # V x F x K
        neighborhood_concat = tf.concat(neighborhood_features, axis=-1)

        # V x K
        attn_concat = tf.concat(neighborhood_attn, axis=-1)

        # V x K tensor of normalized attention coefficients
        attn_coefs = tf.nn.softmax(attn_concat, axis=-1)

        # V x K x 1 tensor of normalized attention coefficients
        attn_coefs_expanded = tf.expand_dims(attn_coefs, axis=-1)

        # V x F x 1 tensor of weighted neighborhood features
        weighted_features = tf.matmul(neighborhood_concat, attn_coefs_expanded)
        weighted_features = tf.squeeze(weighted_features, axis=-1)
        weighted_features = tf.contrib.layers.bias_add(weighted_features,
                                                       scope='{0}-b'.format(self.name))

        return self.activation(weighted_features), attn_coefs


class AttentionNeighborhood(Layer):
    """
    Uses GAT for local neighborhood aggregation before using attention to weight each
    neighborhood individually.
    """

    def __init__(self, output_size, num_heads, is_sparse, activation=tf.nn.tanh, name='neighborhood'):
        super(AttentionNeighborhood, self).__init__(output_size, activation, name)
        self.is_sparse = is_sparse
        self.num_heads = num_heads

    def __call__(self, inputs, **kwargs):

        # List of 'num_neighborhoods' V x V matrices
        neighborhoods = kwargs['neighborhoods']

        dropout_keep_prob = kwargs['dropout_keep_prob']

        if self.is_sparse:
            agg_layer = SparseGAT(output_size=self.output_size,
                                  num_heads=self.num_heads,
                                  activation=self.activation,
                                  name='{0}-sparse-GAT'.format(self.name))
        else:
            agg_layer = GAT(output_size=self.output_size,
                            num_heads=self.num_heads,
                            activation=self.activation,
                            name='{0}-GAT'.format(self.name))

        # Layer to compute attention weights for each aggregated neighborhood
        attn_layer = MLP(hidden_sizes=[],
                         output_size=1,
                         bias_final=False,
                         activation=None,
                         name='{0}-attn-weights'.format(self.name))

        neighborhood_features = []
        neighborhood_attn = []
        for neighborhood_mat in neighborhoods:

            # V x F tensor of aggregated node features over the given neighborhood
            if self.is_sparse:
                neighborhood_agg = agg_layer(inputs=inputs,
                                             adj_matrix=neighborhood_mat,
                                             weight_dropout_keep=dropout_keep_prob,
                                             attn_dropout_keep=dropout_keep_prob)
            else:
                neighborhood_agg = agg_layer(inputs=inputs,
                                             bias=neighborhood_mat,
                                             weight_dropout_keep=dropout_keep_prob,
                                             attn_dropout_keep=dropout_keep_prob)

            # V x 1 tensor of attention weights
            attn_weights = attn_layer(inputs=neighborhood_agg, dropout_keep_prob=dropout_keep_prob)

            neighborhood_features.append(tf.expand_dims(neighborhood_agg, axis=-1))
            neighborhood_attn.append(attn_weights)

        # V x F x K
        neighborhood_concat = tf.concat(neighborhood_features, axis=-1)

        # V x K
        attn_concat = tf.concat(neighborhood_attn, axis=-1)

        # V x K tensor of normalized attention coefficients
        attn_coefs = tf.nn.softmax(attn_concat, axis=-1)

        # V x K x 1 tensor of normalized attention coefficients
        attn_coefs_expanded = tf.expand_dims(attn_coefs, axis=-1)

        # V x F x 1 tensor of weighted neighborhood features
        weighted_features = tf.matmul(neighborhood_concat, attn_coefs_expanded)
        weighted_features = tf.squeeze(weighted_features, axis=-1)
        weighted_features = tf.contrib.layers.bias_add(weighted_features,
                                                       activation_fn=self.activation,
                                                       scope='{0}-bias'.format(self.name))

        return weighted_features, attn_coefs


class MinCostFlow(Layer):

    def __init__(self, flow_iters, dims=3, name='min-cost-flow'):
        super(MinCostFlow, self).__init__(0, None, name)
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
        super(SparseMinCostFlow, self).__init__(0, None, name)
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


class DualFlow(Layer):

    def __init__(self, step_size, momentum, iters, name='dual-flow'):
        super(DualFlow, self).__init__(0, None, name)
        self.step_size = step_size
        self.momentum = momentum
        self.iters = iters

    def __call__(self, inputs, **kwargs):
        dual_diff = inputs
        adj = kwargs['adj']
        cost_fn = kwargs['cost_fn']

        def body(flow, acc, prev_flow):
            momentum_acc = self.momentum * acc
            derivative = cost_fn.derivative(flow - momentum_acc) - dual_diff
            next_acc = momentum_acc + self.step_size * derivative
            next_flow = tf.nn.relu(adj * (flow - next_acc))
            return [next_flow, next_acc, flow]

        def cond(flow, momentum, prev_flow):
            return tf.reduce_any(tf.abs(flow - prev_flow) > FLOW_THRESHOLD)

        dual_flows = dual_diff
        acc = tf.zeros_like(dual_diff, dtype=tf.float32)
        prev_dual_flows = dual_flows + BIG_NUMBER
        shape_invariants = [dual_flows.get_shape(), acc.get_shape(), prev_dual_flows.get_shape()]
        dual_flows, _, _ = tf.while_loop(cond, body,
                                         loop_vars=[dual_flows, acc, prev_dual_flows],
                                         parallel_iterations=1,
                                         shape_invariants=shape_invariants,
                                         maximum_iterations=self.iters,
                                         name=self.name + '-while-loop')

        return dual_flows


class SparseDualFlow(Layer):

    def __init__(self, step_size, momentum, iters, name='sparse-dual-flow'):
        super(SparseDualFlow, self).__init__(0, None, name)
        self.step_size = step_size
        self.momentum = momentum
        self.iters = iters

    def __call__(self, inputs, **kwargs):
        dual_diff = inputs  # |V| x |V| sparse tensor
        cost_fn = kwargs['cost_fn']  # Callable cost function

        def body(flow, acc, prev_flow):
            acc_momentum = self.momentum * acc
            gradient = cost_fn.derivative(flow - acc_momentum) - dual_diff.values

            next_acc = acc_momentum + self.step_size * gradient
            next_flow = tf.nn.relu(flow - next_acc)
            return [next_flow, next_acc, flow]

        def cond(flow, momentum, prev_flow):
            return tf.reduce_any(tf.abs(flow - prev_flow) > FLOW_THRESHOLD)

        # Initialize values for optimization loop
        dual_flows = tf.zeros_like(dual_diff.values, dtype=tf.float32)
        acc = tf.zeros_like(dual_diff.values, dtype=tf.float32)
        prev_dual_flows = dual_flows + BIG_NUMBER
        shape_invariants = [dual_flows.get_shape(), acc.get_shape(), prev_dual_flows.get_shape()]
        dual_flows, _, _ = tf.while_loop(cond, body,
                                         loop_vars=[dual_flows, acc, prev_dual_flows],
                                         parallel_iterations=1,
                                         shape_invariants=shape_invariants,
                                         maximum_iterations=self.iters,
                                         name=self.name + '-while-loop')

        return tf.SparseTensor(indices=dual_diff.indices,
                               values=dual_flows,
                               dense_shape=dual_diff.dense_shape)
