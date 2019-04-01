import tensorflow as tf
import numpy as np

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


class SparseGAT(Layer):
    """
    Sparse Graph Attention Layer from https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_size, output_size, num_heads, activation=tf.nn.relu,
                 weight_dropout_keep=1.0, attn_dropout_keep=1.0, name='GAT'):
        super(SparseGAT, self).__init__(input_size, output_size, activation, name)
        self.num_heads = num_heads
        self.weight_dropout_keep = weight_dropout_keep
        self.attn_dropout_keep = attn_dropout_keep

    def __call__(self, inputs, **kwargs):
        adj_matrix = kwargs['adj_matrix']

        with tf.name_scope(self.name):
            heads = []
            for i in range(self.num_heads):
                tensor_mlp = MLP(hidden_sizes=[],
                                 output_size=self.output_size,
                                 bias_final=False,
                                 activation=None,
                                 name='{0}-W-{1}'.format(self.name, i))
                tensors = tensor_mlp(inputs=inputs, dropout_keep_prob=self.weight_dropout_keep)

                attn_mlp = MLP(hidden_sizes=[],
                               output_size=1,
                               bias_final=False,
                               activation=None,
                               name='{0}-a-{1}'.format(self.name, i))
                attn_weights = attn_mlp(inputs=tensors, dropout_keep_prob=self.attn_dropout_keep)

                masked_1 = neighbor_bias * attn_weights
                masked_2 = neighbor_bias * tf.transpose(attn_weights, perm=[1, 0])

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
    def __init__(self, input_size, output_size, num_heads, activation=tf.nn.relu,
                 weight_dropout_keep=1.0, attn_dropout_keep=1.0, name='GAT'):
        super(GAT, self).__init__(input_size, output_size, activation, name)
        self.num_heads = num_heads
        self.weight_dropout_keep = weight_dropout_keep
        self.attn_dropout_keep = attn_dropout_keep

    def __call__(self, inputs, **kwargs):
        bias = kwargs['bias']

        with tf.name_scope(self.name):
            heads = []
            for i in range(self.num_heads):
                # Apply weight matrix to the set of inputs, B x V x D' Tensor
                input_mlp = MLP(hidden_sizes=[],
                                 output_size=self.output_size,
                                 bias_final=False,
                                 activation=None,
                                 name='{0}-W-{1}'.format(self.name, i))
                transformed_inputs = input_mlp(inputs=inputs, dropout_keep_prob=self.weight_dropout_keep)

                # Create unnormalized attention weights, B x V x V
                attn_mlp = MLP(hidden_sizes=[],
                               output_size=1,
                               bias_final=False,
                               activation=None,
                               name='{0}-a-{1}'.format(self.name, i))
                attn_weights = attn_mlp(inputs=transformed_inputs, dropout_keep_prob=self.attn_dropout_keep)
                attn_weights = attn_weights + tf.transpose(attn_weights, [0, 2, 1])

                # Compute normalized attention weights, B x V x V
                attention_coefs = tf.nn.softmax(tf.nn.leaky_relu(attn_weights) + bias, axis=2)

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
        assert isinstance(inputs, tuple) or isinstance(inputs, list)
        assert len(inputs) == 2

        with tf.name_scope(self.name):
            prev_state = inputs[0]
            curr_state = inputs[1]

            prev = tf.layers.dense(inputs=prev_state,
                                   units=1,
                                   use_bias=False,
                                   name='{0}-W-1'.format(self.name))

            curr = tf.layers.dense(inputs=curr_state,
                                   units=1,
                                   use_bias=False,
                                   name='{0}-W-2'.format(self.name))

            self.z = tf.contrib.layers.bias_add(prev + curr,
                                                activation_fn=tf.math.sigmoid,
                                                scope='{0}-b'.format(self.name))

            # Apply gate
            return self.z * curr_state + (1 - self.z) * prev_state
