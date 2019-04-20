import tensorflow as tf
import numpy as np
from constants import COST_MAX, EXP_MAX, BIG_NUMBER, SMALL_NUMBER


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

    def __init__(self, options):
        super(Linear, self).__init__(options)
        assert 'a' in options
        assert 'b' in options

    def apply(self, x):
        return self.clip(self.options['a'] * x + self.options['b'])

    def derivative(self, x):
        return self.options['a']


class Quadratic(CostFunction):

    def __init__(self, options):
        super(Quadratic, self).__init__(options)
        assert 'a' in options
        assert 'b' in options
        assert 'c' in options

    def apply(self, x):
        return self.clip(self.options['a'] * tf.square(x) + self.options['b'] * x + self.options['c'])

    def derivative(self, x):
        return self.clip(2.0 * self.options['a'] * x + self.options['b'])


class Cubic(CostFunction):

    def __init__(self, options):
        super(Cubic, self).__init__(options)
        assert 'a' in options
        assert 'b' in options
        assert 'c' in options
        assert 'd' in options

    def apply(self, x):
        return self.clip(self.options['a'] * tf.pow(x, 3) + self.options['b'] * tf.square(x) +\
                         self.options['c'] * x + self.options['d'])

    def derivative(self, x):
        return self.clip(3.0 * self.options['a'] * tf.square(x) + 2.0 * self.options['b'] * x +\
                         self.options['c'])


class Quartic(CostFunction):
    
    def __init__(self, options):
        super(Quartic, self).__init__(options)
        assert 'a' in options
        assert 'b' in options
        assert 'c' in options
        assert 'd' in options
        assert 'e' in options

    def apply(self, x):
        return self.clip(self.options['a'] * tf.pow(x, 4) + self.options['b'] * tf.pow(x, 3) +\
                         self.options['c'] * tf.square(x) + self.options['d'] * x + self.options['e'])

    def derivative(self, x):
        return self.clip(4.0 * self.options['a'] * tf.pow(x, 3) +\
                         3.0 * self.options['b'] * tf.square(x) +\
                         2.0 * self.options['c'] * x + self.options['d'])


class Exp(CostFunction):

    def __init__(self, options):
        super(Exp, self).__init__(options)

        self.constant = options['constant']
        assert self.constant > 0.0
        assert self.constant <= EXP_MAX

    def apply(self, x):
        return self.clip(tf.exp(self.constant * x) - 1)

    def derivative(self, x):
        return self.clip(self.constant * tf.exp(self.constant * x))


class PolySin(CostFunction):

    def __init__(self, options):
        super(PolySin, self).__init__(options)
        assert 'a' in options
        assert 'b' in options
        assert 'c' in options

    def apply(self, x):
        return self.clip(self.options['a'] * x + self.options['b'] * tf.sin(self.options['c'] * x))

    def derivative(self, x):
        return self.options['a'] + self.options['b'] * self.options['c'] * tf.cos(self.options['c'] * x)


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
    if name == 'poly_sin':
        return PolySin(options=options)
    return None
