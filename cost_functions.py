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


def get_cost_function(name, constant):
    if name == 'square':
        return Square(constant=constant)
    return None
