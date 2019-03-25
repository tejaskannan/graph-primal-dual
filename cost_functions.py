import tensorflow as tf

class CostFunction:

    def apply(self, x):
        raise NotImplementedError()

    def inv_derivative(self, y):
        raise NotImplementedError()


class Square(CostFunction):

    def apply(self, x):
        return tf.square(x)

    def inv_derivative(self, y):
        return 0.5 * y


tf_cost_functions = {
    'square': Square()
}
