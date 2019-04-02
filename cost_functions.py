import tensorflow as tf

class CostFunction:

    def __init__(self, constant):
        self.constant = constant

    def apply(self, x):
        raise NotImplementedError()

    def inv_derivative(self, y):
        raise NotImplementedError()


class Square(CostFunction):

    def __init__(self, constant):
        super(Square, self).__init__(constant)
        assert constant > 0

    def apply(self, x):
        return self.constant * tf.square(x)

    def inv_derivative(self, y):
        return (1.0 / (2.0 * self.constant)) * y


class Exp(CostFunction):

    def __init__(self, constant):
        super(Exp, self).__init__(constant)
        assert constant >= 1.0

    def apply(self, x):
        return tf.exp(self.constant * x) - 1

    def inv_derivative(self, y):
        constant = tf.fill(dims=tf.shape(y), value=self.constant)
        y = tf.where(y > constant, y, constant)
        return (1.0 / self.constant) * (tf.log(y) - tf.log(constant))


def get_cost_function(name, constant):
    if name == 'square':
        return Square(constant=constant)
    if name == 'exp':
        return Exp(constant=constant)
    return None
