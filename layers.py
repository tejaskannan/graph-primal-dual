import tensorflow as tf
import numpy as np


class Layer:
    """
    Base Layer interface. Based on the design of
    layers from Keras (https://keras.io/)
    """

    def __init__(self, input_size, output_size, activation, name):
        self.name = name
        self.activation = activation
        self.input_size = input_size
        self.output_size = output_size

    def build(self, inputs, **kwargs):
        raise NotImplementedError()


class Dense(Layer):
    """
    Fully connected layer.
    """

    def __init__(self, input_size, output_size, activation=None, name='dense'):
        super(Dense, self).__init__(input_size, output_size, activation, name)

    def build(self, inputs, **kwargs):
        with tf.name_scope(self.name):
            init_W = tf.random.uniform(shape=(1, self.input_size, self.output_size), maxval=1.0)
            self.W = tf.Variable(initial_value=init_W,
                                 trainable=True,
                                 name='W')

            init_b = tf.random.uniform(shape=(1, 1, self.output_size), maxval=1.0)
            self.b = tf.Variable(initial_value=init_b,
                                 trainable=True,
                                 name='b')

            output = tf.matmul(inputs, self.W) + self.b
            if self.activation is None:
                return output
            return self.activation(output)


class GAT(Layer):
    """
    Graph Attention Layer from https://arxiv.org/abs/1710.10903
    """
    def __init__(self, input_size, output_size, num_heads, name='GAT'):
        self.num_heads = num_heads
        super(GAT, self).__init__(input_size, output_size, None, name)

    def build(self, inputs, **kwargs):
        assert 'bias' in kwargs
        bias = kwargs['bias']

        heads = []
        for _ in range(self.num_heads):
            # Apply weight matrix to the set of inputs, B x V x F' Tensor
            transformed_inputs = tf.layers.dense(inputs=inputs,
                                                 units=self.output_size,
                                                 use_bias=False,
                                                 name=self.name + '-W')

            # Create unnormalized attention weights, B x V x V
            weights = tf.layers.dense(inputs=transformed_inputs,
                                      units=1,
                                      use_bias=False,
                                      name=self.name + '-a')
            weights = weights + tf.transpose(weights, [0, 2, 1])

            # Compute normalized attention weights, B x V x V
            attention_coefs = tf.nn.softmax(tf.nn.leaky_relu(weights) + bias, axis=2)

            # Apply attention weights, B x V x F'
            head = tf.contrib.layers.bias_add(tf.matmul(attention_coefs, transformed_inputs))
            heads.append(head)

        # Average over all attention heads
        return (1.0 / self.num_heads) * tf.add_n(heads)


class Gate(Layer):
    """
    Skip-connection Gating layer from https://arxiv.org/pdf/1805.10988.pdf
    """

    def __init__(self, name='gate'):
        super(Gate, self).__init__(0, 0, None, name)

    def build(self, inputs, **kwargs):
        assert isinstance(inputs, tuple) or isinstance(inputs, list)
        assert len(inputs) == 2

        prev_state = inputs[0]
        curr_state = inputs[1]
        with tf.name_scope(self.name):

            prev = tf.layers.dense(inputs=prev_state,
                                   units=1,
                                   use_bias=False,
                                   name=self.name + '-W-1')

            curr = tf.layers.dense(inputs=curr_state,
                                   units=1,
                                   use_bias=False,
                                   name=self.name + '-W-2')

            self.z = tf.contrib.layers.bias_add(prev + curr,
                                                activation_fn=tf.math.sigmoid,
                                                scope='b')

            # Apply gate
            return self.z * curr_state + (1 - self.z) * prev_state
