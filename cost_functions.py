import tensorflow as tf
from constants import COST_MAX, EXP_MAX, BIG_NUMBER

class CostFunction:

    def __init__(self, constant):
        self.constant = constant

    def apply(self, x):
        raise NotImplementedError()

    def inv_derivative(self, y):
        raise NotImplementedError()

    def clip(self, x):
        return tf.clip_by_value(x, 0, COST_MAX)


class Square(CostFunction):

    def __init__(self, constant):
        super(Square, self).__init__(constant)
        assert constant > 0

    def apply(self, x):
        return self.clip(self.constant * tf.square(x))

    def inv_derivative(self, y):
        return (1.0 / (2.0 * self.constant)) * y


class Exp(CostFunction):

    def __init__(self, constant):
        super(Exp, self).__init__(constant)
        assert constant > 0.0
        assert constant <= EXP_MAX

    def apply(self, x):
        return self.clip(tf.exp(self.constant * x) - 1)

    def inv_derivative(self, y):
        y = tf.clip_by_value(y, self.constant, BIG_NUMBER)
        return self.clip((1.0 / self.constant) * (tf.log(y) - tf.log(self.constant)))


def get_cost_function(name, constant):
    if name == 'square':
        return Square(constant=constant)
    if name == 'exp':
        return Exp(constant=constant)
    return None
