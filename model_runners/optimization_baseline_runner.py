import numpy as np
import pickle
import gzip
import networkx as nx
from time import time
from models.optimization_models import TrustConstr, SLSQP
from utils.utils import features_to_demands, append_row_to_log
from utils.constants import PARAMS_FILE, WRITE_THRESHOLD
from core.plot import plot_road_flow_graph
from core.dataset import DatasetManager, Series
from os import mkdir
from os.path import exists


class OptimizationBaselineRunner:

    def __init__(self, params, optimizer_name):
        self.params = params

        self.graph_name = params['graph_name']

        cost_fn_name = params['cost_fn']['name']
        self.output_folder = '{0}/{1}-{2}-{3}/'.format(params['output_folder'], optimizer_name, self.graph_name, cost_fn_name)

        self.dataset = DatasetManager(params=params)
        self.dataset.load_graphs()

        assert optimizer_name == 'trust_constr' or optimizer_name == 'slsqp', 'Invalid Optimizer {0}'.format(optimizer_name)
        if optimizer_name == 'trust_constr':
            self.optimizer = TrustConstr(params=params)
        elif optimizer_name == 'slsqp':
            self.optimizer = SLSQP(params=params)

    def optimize(self):
        if not exists(self.output_folder):
            mkdir(self.output_folder)

        test_graph = self.dataset.graph_data.graph

        cost_headers = ['Index', 'Graph', 'Cost', 'Time (sec)']
        costs_path = self.output_folder + 'costs.csv'
        append_row_to_log(cost_headers, costs_path)

        # Save parameters
        params_path = PARAMS_FILE.format(self.output_folder)
        with gzip.GzipFile(params_path, 'wb') as out_file:
            pickle.dump(self.params, out_file)

        # Load test dataset
        self.dataset.load(series=Series.TEST)
        test_samples = [sample for sample in self.dataset.dataset[Series.TEST] if sample.graph_name == self.graph_name]

        step = int(1.0 / self.params['plot_fraction'])
        plot_indices = set(range(0, len(test_samples), step))

        for index, sample in enumerate(test_samples):
            demands = np.array(sample.demands.todense())
            demands = np.reshape(demands, newshape=(demands.shape[0],))

            start = time()
            flows_per_iter, result = self.optimizer.optimize(graph=test_graph, demands=demands)
            end = time()

            flows = flows_per_iter[-1]
            flow_mat = self._flow_matrix(test_graph, flows)

            append_row_to_log([index, self.graph_name, result.fun, end - start], costs_path)

            # Add demands to graph
            flow_graph = test_graph.copy()
            for i, node in enumerate(test_graph.nodes()):
                flow_graph.add_node(node, demand=float(demands[i]))

            for src, dst, key in flow_graph.edges(keys=True, data=False):
                flow_graph.add_edge(src, dst, key=key, flow=flow_mat[src, dst])

            if self.params['plot_flows'] and index in plot_indices:
                file_path = '{0}flows-{1}-{2}.png'.format(self.output_folder, self.graph_name, index)
                plot_road_flow_graph(flow_graph, graph_name=self.params['graph_title'], field='flow', file_path=file_path)

            if (index + 1) % WRITE_THRESHOLD == 0:
                print('Completed {0} instances.'.format(index + 1))

            # nx.write_gexf(flow_graph, '{0}graph-{1}-{2}.gexf'.format(self.output_folder, self.graph_name, index))

    def _flow_matrix(self, graph, flows):
        num_nodes = graph.number_of_nodes()
        flow_mat = np.zeros(shape=(num_nodes, num_nodes), dtype=float)

        for i, (src, dest) in enumerate(graph.edges()):
            flow_mat[src, dest] = flows[i]

        return flow_mat
