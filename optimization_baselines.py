import numpy as np
from scipy import optimize
from np_cost_functions import get_cost_function


class OptimizeBaseline:

    def __init__(self, params):
        self.max_iters = params['flow_iters']
        self.threshold = params['early_stop_threshold']
        self.cost_fn = get_cost_function(params['cost_fn'])

    def optimize(self, graph):
        raise NotImplementedError()

    # Returns the constraint which enforces that solutions are flow
    def _constraint(self, graph, demands, as_dict=False):
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        A = np.zeros(shape=(num_nodes, num_edges), dtype=float)
        for i, node in enumerate(graph.nodes()):
            for j, (src, dest) in enumerate(graph.edges()):
                if node == src:
                    A[i, j] = 1.0
                elif node == dest:
                    A[i, j] = -1.0

        if not as_dict:
            return optimize.LinearConstraint(A, lb=-1 * demands, ub=-1 * demands)
        else:
            return {
                'type': 'eq',
                'fun': lambda x: A.dot(x) + demands
            }


class TrustConstr(OptimizeBaseline):

    def optimize(self, graph, demands):
        options = {
            'maxiter': self.max_iters,
            'factorization_method': 'SVDFactorization'
        }

        initial = np.random.uniform(size=(graph.number_of_edges(),))

        lower_bound = np.zeros(shape=(graph.number_of_edges(),), dtype=float)
        upper_bound = np.full(shape=(graph.number_of_edges(),), fill_value=np.inf)
        bounds = optimize.Bounds(lb=lower_bound, ub=upper_bound)
        constraint = self._constraint(graph, demands)

        flows_per_iter = []

        def callback(x, state):
            flows_per_iter.append(x)
            return False

        hess = optimize.BFGS()
        result = optimize.minimize(fun=self.cost_fn,
                                   x0=initial,
                                   method='trust-constr',
                                   constraints=[constraint],
                                   jac='2-point',
                                   hess=hess,
                                   options=options)

        return np.array(flows_per_iter)


class SLSQP(OptimizeBaseline):

    def optimize(self, graph, demands):
        options = {
            'maxiter': self.max_iters,
            'ftol': self.threshold
        }

        initial = np.zeros(shape=(graph.number_of_edges(),), dtype=float)
        bounds = optimize.Bounds(lb=0, ub=np.inf)
        constraint = self._constraint(graph, demands, as_dict=True)

        flows_per_iter = []

        def callback(x):
            flows_per_iter.append(x)
            return False

        result = optimize.minimize(fun=self.cost_fn,
                                   x0=initial,
                                   bounds=bounds,
                                   method='SLSQP',
                                   constraints=[constraint],
                                   callback=callback,
                                   options=options)

        return np.array(flows_per_iter)
