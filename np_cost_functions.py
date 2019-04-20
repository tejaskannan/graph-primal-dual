import numpy as np
from constants import COST_MAX, EXP_MAX, BIG_NUMBER, SMALL_NUMBER


class CostFunction:

    def __init__(self, options):
        self.options = options

    def __call__(self, x):
        raise NotImplementedError()


class Linear(CostFunction):

    def __init__(self, options):
        super(Linear, self).__init__(options)
        assert 'a' in options
        assert 'b' in options

    def __call__(self, x):
        return np.sum(self.options['a'] * x + self.options['b'])


class Quadratic(CostFunction):

    def __init__(self, options):
        super(Quadratic, self).__init__(options)
        assert 'a' in options
        assert 'b' in options
        assert 'c' in options

    def __call__(self, x):
        return np.sum(self.options['a'] * np.square(x) + self.options['b'] * x + self.options['c'])


class Cubic(CostFunction):

    def __init__(self, options):
        super(Cubic, self).__init__(options)
        assert 'a' in options
        assert 'b' in options
        assert 'c' in options
        assert 'd' in options

    def __call__(self, x):
        return np.sum(self.options['a'] * np.power(x, 3) + self.options['b'] * np.square(x) +\
                      self.options['c'] * x + self.options['d'])


class Quartic(CostFunction):
    
    def __init__(self, options):
        super(Quartic, self).__init__(options)
        assert 'a' in options
        assert 'b' in options
        assert 'c' in options
        assert 'd' in options
        assert 'e' in options

    def __call__(self, x):
        return np.sum(self.options['a'] * np.power(x, 4) + self.options['b'] * np.power(x, 3) +\
                      self.options['c'] * np.square(x) + self.options['d'] * x + self.options['e'])


class Exp(CostFunction):

    def __init__(self, options):
        super(Exp, self).__init__(options)

        self.constant = options['constant']
        assert self.constant > 0.0
        assert self.constant <= EXP_MAX

    def __call__(self, x):
        return np.sum(np.exp(self.constant * x) - 1)


class Log(CostFunction):

    def __init__(self, options):
        super(Log, self).__init__(options)
        assert 'a' in options
        assert 'b' in options
        assert 'c' in options

    def __call__(self, x):
        # Clip values so that gradients are well defined
        return np.sum(self.options['a'] * np.log(self.options['b'] * x + 1) + self.options['c'])


class PolySin(CostFunction):

    def __init__(self, options):
        super(PolySin, self).__init__(options)
        assert 'a' in options
        assert 'b' in options
        assert 'c' in options

    def __call__(self, x):
        return np.sum(self.options['a'] * x + self.options['b'] * np.sin(self.options['c'] * x))


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
    if name == 'log':
        return Log(options=options)
    return None
