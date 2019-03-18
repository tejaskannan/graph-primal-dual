import tensorflow as tf
import numpy as np
from layers import Dense
import matplotlib.pyplot as plt

num_samples = 1000000
batch_size = 50
num_train = int(num_samples * 0.8)
num_valid = num_samples - num_train

xs = np.linspace(start=-1.0, stop=1.0, num=num_samples)
np.random.shuffle(xs)

ys = np.square(xs)

x_train = np.array([xs[i:i+batch_size] for i in range(0, num_train, batch_size)])
y_train = np.array([ys[i:i+batch_size] for i in range(0, num_train, batch_size)])

x_valid = np.array([xs[i:i+batch_size] for i in range(num_train, num_samples, batch_size)])
y_valid = np.array([ys[i:i+batch_size] for i in range(num_train, num_samples, batch_size)])

with tf.Session(graph=tf.Graph()) as sess:

    x_placeholder = tf.placeholder(dtype=tf.float32,
                                   shape=[None, 1])
    y_placeholder = tf.placeholder(dtype=tf.float32,
                                   shape=[None, 1])

    hidden_layer_1 = Dense(input_size=1, output_size=8, activation=tf.nn.tanh)
    hidden_layer_2 = Dense(input_size=8, output_size=8, activation=tf.nn.tanh)
    output_layer = Dense(input_size=8, output_size=1)

    hidden_1 = hidden_layer_1.build(x_placeholder)
    hidden_2 = hidden_layer_2.build(hidden_1)
    output = output_layer.build(hidden_2)

    loss_op = tf.reduce_mean(tf.square(y_placeholder - output))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    trainable_vars = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    gradients = tf.gradients(loss_op, trainable_vars)
    gradients = [(grad, var) for grad, var in zip(gradients, trainable_vars)]
    optimizer_op = optimizer.apply_gradients(gradients)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for x, y in zip(x_train, y_train):
        feed_dict = {
            x_placeholder: np.expand_dims(x, axis=1),
            y_placeholder: np.expand_dims(y, axis=1)
        }
        ops = [loss_op, optimizer_op]
        op_result = sess.run(ops, feed_dict=feed_dict)


    for x, y in zip(x_valid, y_valid):
        feed_dict = {
            x_placeholder: np.expand_dims(x, axis=1),
            y_placeholder: np.expand_dims(y, axis=1)
        }
        op_result = sess.run(output, feed_dict=feed_dict)

    plt.plot(x, y, 'bo')
    plt.plot(x, op_result, 'ro')
    plt.show()