import tensorflow as tf
from constants import COST_MAX, EXP_MAX, BIG_NUMBER, SMALL_NUMBER


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


class Cube(CostFunction):

    def __init__(self, constant):
        super(Cube, self).__init__(constant)
        assert constant > 0

    def apply(self, x):
        return self.clip(tf.pow(x, 3))

    def inv_derivative(self, y):
        x = tf.clip_by_value((1.0 / (3.0 * self.constant)) * y, SMALL_NUMBER, BIG_NUMBER)
        return self.clip(tf.sqrt(x))


class Quad(CostFunction):
    
    def __init__(self, constant):
        super(Quad, self).__init__(constant)
        assert constant > 0.0
        assert constant <= EXP_MAX

    def apply(self, x):
        return self.clip(tf.pow(self.constant * x, 4))

    def inv_derivative(self, y):
        x = tf.clip_by_value((1.0 / (4.0 * self.constant)) * y, SMALL_NUMBER, BIG_NUMBER)
        return self.clip(tf.pow(x, (1.0 / 3.0)))


class Exp(CostFunction):

    def __init__(self, constant):
        super(Exp, self).__init__(constant)
        assert constant > 0.0
        assert constant <= EXP_MAX

    def apply(self, x):
        return self.clip(tf.exp(self.constant * x) - 1)

    def inv_derivative(self, y):
        x = (1.0 / self.constant) * tf.clip_by_value(y, SMALL_NUMBER, BIG_NUMBER)
        return self.clip((1.0 / self.constant) * tf.log(x))


class LinearExp(CostFunction):

    def __init__(self, constant):
        super(LinearExp, self).__init__(constant)
        assert constant > 0.0
        assert constant <= EXP_MAX

    def apply(self, x):
        return self.clip(tf.exp(self.constant * x) - x - 1)

    def inv_derivative(self, y):
        x = (1.0 / self.constant) * tf.clip_by_value(y + 1, SMALL_NUMBER, BIG_NUMBER)
        return self.clip((1.0 / self.constant) * tf.log(x))


def get_cost_function(name, constant):
    if name == 'square':
        return Square(constant=constant)
    if name == 'exp':
        return Exp(constant=constant)
    if name == 'linear_exp':
        return LinearExp(constant=constant)
    if name == 'cube':
        return Cube(constant=constant)
    if name == 'quad':
        return Quad(constant=constant)
    return None
