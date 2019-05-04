import tensorflow as tf
import numpy as np
from utils.constants import COST_MAX, EXP_MAX, BIG_NUMBER, SMALL_NUMBER


class CostFunction:

    def __init__(self, options):
        self.options = options

    def apply(self, x):
        raise NotImplementedError()

    def derivative(self, x):
        raise NotImplementedError()

    def clip(self, x):
        return tf.clip_by_value(x, -COST_MAX, COST_MAX)


class Linear(CostFunction):
    """
    f(x) = ax + b
    """

    def __init__(self, options):
        super(Linear, self).__init__(options)
        self.a = options['a']
        self.b = options['b']

    def apply(self, x):
        return self.clip(self.a * x + self.b)

    def derivative(self, x):
        return self.a


class Quadratic(CostFunction):
    """
    f(x) = ax^2 + bx + c
    """

    def __init__(self, options):
        super(Quadratic, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']

    def apply(self, x):
        return self.clip(tf.square(x) * self.a + x * self.b + self.c)

    def derivative(self, x):
        return self.clip(2.0 * self.a * x + self.b)


class Cubic(CostFunction):
    """
    f(x) = a*x^3 + b*x^2 + c*x + d
    """

    def __init__(self, options):
        super(Cubic, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']
        self.d = options['d']

    def apply(self, x):
        return self.clip(self.a * tf.pow(x, 3) + self.b * tf.square(x) + self.c * x + self.d)

    def derivative(self, x):
        return self.clip(3.0 * self.a * tf.square(x) + 2.0 * self.b * x + self.c)


class Quartic(CostFunction):
    """
    f(x) = a*x^4 + b*x^3 + c*x^2 + d*x + e
    """

    def __init__(self, options):
        super(Quartic, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']
        self.d = options['d']
        self.e = options['e']

    def apply(self, x):
        return self.clip(self.a * tf.pow(x, 4) + self.b * tf.pow(x, 3) +
                         self.c * tf.square(x) + self.d * x + self.e)

    def derivative(self, x):
        return self.clip(4.0 * self.a * tf.pow(x, 3) +
                         3.0 * self.b * tf.square(x) +
                         2.0 * self.c * x + self.d)


class Exp(CostFunction):
    """
    f(x) = e^(a*x) - 1
    """

    def __init__(self, options):
        super(Exp, self).__init__(options)

        self.a = options['a']
        assert self.a > 0.0
        assert self.a <= EXP_MAX

    def apply(self, x):
        return self.clip(tf.exp(self.a * x) - 1)

    def derivative(self, x):
        return self.clip(self.a * tf.exp(self.a * x))


class Log(CostFunction):
    """
    f(x) = a*ln(b*x + 1) + c
    """

    def __init__(self, options):
        super(Log, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']

    def apply(self, x):
        # Clip values so that gradients are well defined
        shifted_x = tf.clip_by_value(self.b * x + 1, SMALL_NUMBER, BIG_NUMBER)
        return self.clip(self.a * tf.log(shifted_x) + self.c)

    def derivative(self, x):
        shifted_x = tf.clip_by_value(self.b * x + 1, SMALL_NUMBER, BIG_NUMBER)
        return self.clip(self.a * self.b * tf.reciprocal(shifted_x))


class LinearSin(CostFunction):
    """
    f(x) = a*x + b*sin(c*x)
    """

    def __init__(self, options):
        super(LinearSin, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']

    def apply(self, x):
        return self.clip(self.a * x + self.b * tf.sin(self.c * x))

    def derivative(self, x):
        return self.a + self.b * self.c * tf.cos(self.c * x)


class Tanh(CostFunction):
    """
    f(x) = a*tanh(b*x) + c
    """

    def __init__(self, options):
        super(Tanh, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']

    def apply(self, x):
        return self.clip(self.a * (tf.nn.tanh(self.b * x)) + self.c)

    def derivative(self, x):
        tanh_squared = tf.square(tf.nn.tanh(self.b * x))
        return self.clip(self.a * self.b * (1 - tanh_squared))


def apply_with_capacities(cost_fn, x, capacities):
    # If the value violated the capacity, an exponential penalty is applied 
    
    # Some constant to produce near-asymptotic behavior
    beta = 20

    # Value at which capacity is seen to be saturated
    t = 0.99 * tf.clip_by_value(capacities, SMALL_NUMBER, BIG_NUMBER)

    # Provided cost function
    below_capacity = cost_fn.apply(x)

    # Cost if capacity is violated
    above_capacity = (cost_fn.derivative(t) / beta) * (tf.exp((x - t) * beta)) + cost_fn.apply(t)
    above_capacity = tf.clip_by_value(above_capacity, 0, BIG_NUMBER)

    return tf.where(x <= t, x=below_capacity, y=above_capacity, name='capacity-cost-wrapper')


def get_cost_function(cost_fn):

    name = cost_fn['name']
    options = cost_fn['options']

    if name == 'linear':
        return Linear(options=options)
    if name == 'quadratic':
        return Quadratic(options=options)
    if name == 'exp':
        return Exp(options=options)
    if name == 'cubic':
        return Cubic(options=options)
    if name == 'quartic':
        return Quartic(options=options)
    if name == 'linear_sin':
        return LinearSin(options=options)
    if name == 'log':
        return Log(options=options)
    return None
