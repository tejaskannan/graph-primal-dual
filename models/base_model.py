import tensorflow as tf
import numpy as np
import gzip
import pickle
from os.path import exists
from os import mkdir
from utils.constants import *


class Model:

    def __init__(self, params, name):
        self.name = name
        self.params = params
        self._sess = tf.Session(graph=tf.Graph())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

        # Must be set by a concrete subclass
        self.loss_op = None
        self.loss = None
        self.optimizer_op = None
        self.output_ops = {}

    def init(self):
        with self._sess.graph.as_default():
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def build(self, **kwargs):
        raise NotImplementedError()

    def run_train_step(self, feed_dict):
        with self._sess.graph.as_default():
            ops = [self.loss_op, self.loss, self.optimizer_op]
            op_result = self._sess.run(ops, feed_dict=feed_dict)
            return op_result[0:2]

    def inference(self, feed_dict):
        with self._sess.graph.as_default():
            self.output_ops['loss'] = self.loss_op
            op_results = self._sess.run(self.output_ops, feed_dict=feed_dict)
            return op_results

    def _build_optimizer_op(self):
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.loss_op, trainable_vars)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, self.params['gradient_clip'])
        pruned_gradients = []
        for grad, var in zip(clipped_grad, trainable_vars):
            if grad is not None:
                pruned_gradients.append((grad, var))

        return self.optimizer.apply_gradients(pruned_gradients)

    def create_placeholder(self, dtype, shape, name, is_sparse=False):
        with self._sess.graph.as_default():
            if is_sparse:
                return tf.sparse.placeholder(dtype, shape=shape, name=name)
            return tf.placeholder(dtype, shape=shape, name=name)

    def save(self, output_folder):
        params_path = PARAMS_FILE.format(output_folder)
        with gzip.GzipFile(params_path, 'wb') as out_file:
            pickle.dump(self.params, out_file)

        with self._sess.graph.as_default():
            model_path = MODEL_FILE.format(output_folder, self.name)
            saver = tf.train.Saver()
            saver.save(self._sess, model_path)

    def restore(self, output_folder):
        with self._sess.graph.as_default():
            model_path = MODEL_FILE.format(output_folder, self.name)
            saver = tf.train.Saver()
            saver.restore(self._sess, model_path)
