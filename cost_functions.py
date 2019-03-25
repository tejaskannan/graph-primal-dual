import tensorflow as tf


def tf_exp(x):
    return tf.exp(x) - 1


tf_cost_functions = {
    'tanh': tf.nn.tanh,
    'exp': tf_exp
}
