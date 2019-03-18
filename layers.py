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
    Single fully connected layer used for testing.
    """

    def __init__(self, input_size, output_size, activation=None, name='dense'):
        super(Dense, self).__init__(input_size, output_size, activation, name)

    def build(self, inputs, **kwargs):
        with tf.name_scope(self.name):
            init_W = tf.random.uniform(shape=(self.input_size, self.output_size), maxval=1.0)
            self.W = tf.Variable(initial_value=init_W,
                                 trainable=True,
                                 name='W')

            init_b = tf.random.uniform(shape=(1, self.output_size), maxval=1.0)
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

        with tf.name_scope(self.name):
            # Lists to hold weights for each attention head
            self.W = []
            self.a = []
            self.b = []

            heads = []
            for _ in range(self.num_heads):
                # Initialize Trainable Variables
                init_W = tf.random.uniform(shape=(self.input_size, self.output_size), maxval=1.0)
                W = tf.Variable(initial_value=init_W,
                                trainable=True,
                                name='W')
                self.W.append(W)

                init_a = tf.random.uniform(shape=(self.output_size, 1), maxval=1.0)
                a = tf.Variable(initial_value=init_a,
                                trainable=True,
                                name='a')
                self.a.append(a)

                init_b = tf.random.uniform(shape=(1, self.output_size), maxval=1.0)
                b = tf.Variable(initial_value=init_b,
                                trainable=True,
                                name='b')
                self.b.append(b)

                # Apply weight matrix to the set of inputs, B x V x F' Tensor
                transformed_inputs = tf.matmul(inputs, self.W) + self.b

                # Create unnormalized attention weights, B x V x V
                weights = tf.matmul(transformed_inputs, self.a)
                weights = weights + tf.transpose(weights, [0, 2, 1])

                # Compute normalized attention weights, B x V x V
                attention_coefs = tf.nn.softmax(tf.nn.leaky_relu(weights) + bias, axis=2)

                # Apply attention weights, B x V x F'
                heads.append(tf.matmul(attention_coefs, transformed_inputs))

            # Average over all attention heads
            return (1.0 / self.num_heads) * tf.add_n(heads)


class Gate(Layer):
    """
    Skip-connection Gating layer from https://arxiv.org/pdf/1805.10988.pdf
    """

    def __init__(self, input_size, name='gate'):
        super(Gate, self).__init__(input_size, input_size, None, name)

    def build(self, inputs, **kwargs):
        assert isinstance(inputs, tuple) or isinstance(inputs, list)
        assert len(inputs) == 2

        prev_state = inputs
        curr_state = inputs
        with tf.name_scope(self.name):
            # Initialize variables
            init_W_1 = tf.random.uniform(shape=(self.input_size, 1), maxval=1.0)
            self.W_1 = tf.Variable(initial_value=init_W_1,
                                   trainable=True,
                                   name='W-1')

            init_W_2 = tf.random.uniform(shape=(self.input_size, 1), maxval=1.0)
            self.W_2 = tf.Variable(initial_value=init_W_2,
                                   trainable=True,
                                   name='W-2')

            init_b = tf.random.uniform(shape=(self.input_size, 1), maxval=1.0)
            self.b = tf.Variable(initial_value=init_b,
                                 trainable=True,
                                 name='b')

            # Compute gate weight, B x V x 1
            prev = tf.matmul(prev_state, self.W_1)
            curr = tf.matmul(curr_state, self.W_2)
            self.z = tf.math.sigmoid(prev + curr + self.b)

            # Apply gate
            return self.z * curr_state + (1 - self.z) * prev_state
