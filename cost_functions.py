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
        return self.clip(self.constant * tf.pow(x, 3))

    def inv_derivative(self, y):
        x = tf.clip_by_value((1.0 / (3.0 * self.constant)) * y, SMALL_NUMBER, BIG_NUMBER)
        return self.clip(tf.sqrt(x))


class Quad(CostFunction):
    
    def __init__(self, constant):
        super(Quad, self).__init__(constant)
        assert constant > 0.0
        assert constant <= EXP_MAX

    def apply(self, x):
        return self.clip(self.constant * tf.pow(x, 4))

    def inv_derivative(self, y):
        x = tf.clip_by_value((1.0 / (4.0 * self.constant)) * y, SMALL_NUMBER, BIG_NUMBER)
        return self.clip(tf.pow(x, (1.0 / 3.0)))

    def derivative(self, x):
        return 4 * self.constant * tf.pow(x, 3)


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


class PolySin(CostFunction):

    def __init__(self, constant):
        super(PolySin, self).__init__(constant)
        assert constant > 0.0
        assert constant < 10.0

    def apply(self, x):
        return self.clip(x + 0.5 * tf.sin(self.constant * x))

    def inv_derivative(self, y):
        x = tf.clip_by_value(2.0 / self.constant * (y - 1), -1 + SMALL_NUMBER, 1 - SMALL_NUMBER)
        inv = self.clip((1.0 / self.constant) * tf.acos(x))

        # Use second derivative to determine if the critical point is a min or a max
        dx2 = -0.5 * (self.constant**2) * tf.sin(self.constant * inv)
        return tf.where(tf.less_equal(dx2, 0),
                        x=tf.zeros_like(inv, dtype=tf.float32),
                        y=inv)


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
    if name == 'poly_sin':
        return PolySin(constant=constant)
    return None
