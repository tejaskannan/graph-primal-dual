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
        self.a = options['a']
        self.b = options['b']

    def __call__(self, x):
        return np.sum(self.a * x + self.b)


class Quadratic(CostFunction):

    def __init__(self, options):
        super(Quadratic, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']

    def __call__(self, x):
        return np.sum(self.a * np.square(x) + self.b * x + self.c)


class Cubic(CostFunction):

    def __init__(self, options):
        super(Cubic, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']
        self.d = options['d']

    def __call__(self, x):
        return np.sum(self.a * np.power(x, 3) + self.b * np.square(x) + self.c * x + self.d)


class Quartic(CostFunction):

    def __init__(self, options):
        super(Quartic, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']
        self.d = options['d']
        self.e = options['e']

    def __call__(self, x):
        return np.sum(self.a * np.power(x, 4) + self.b * np.power(x, 3) +
                      self.c * np.square(x) + self.d * x + self.e)


class Exp(CostFunction):

    def __init__(self, options):
        super(Exp, self).__init__(options)

        self.a = options['constant']
        assert self.a > 0.0
        assert self.a <= EXP_MAX

    def __call__(self, x):
        return np.sum(np.exp(self.a * x) - 1)


class Log(CostFunction):

    def __init__(self, options):
        super(Log, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']

    def __call__(self, x):
        return np.sum(self.a * np.log(self.b * x + 1) + self.c)


class LinearSin(CostFunction):

    def __init__(self, options):
        super(LinearSin, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']

    def __call__(self, x):
        return np.sum(self.a * x + self.b * np.sin(self.c * x))


class Tanh(CostFunction):

    def __init__(self, options):
        super(Tanh, self).__init__(options)
        self.a = options['a']
        self.b = options['b']
        self.c = options['c']

    def __call__(self, x):
        return np.sum(self.a * np.tanh(self.b * x) + self.c)


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
