import numpy as np
from optimization_baselines import TrustConstr, SLSQP
from datetime import datetime
from time import time
from load import load_to_networkx, read_dataset
from utils import features_to_demands, append_row_to_log
from plot import plot_flow_graph
from os import mkdir
from os.path import exists


class OptimizationBaselineRunner:

    def __init__(self, params, optimizer_name):
        self.params = params
        self.timestamp = datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        graph_names = '-'.join(params['test_graph_names'])

        self.output_folder = '{0}/{1}-{2}-{3}/'.format(params['output_folder'], optimizer_name, graph_names, self.timestamp)

        self.file_paths = {}
        dataset_path = 'datasets/{0}_{1}.txt'
        for dataset_name, graph_name in zip(self.params['test_dataset_names'], self.params['test_graph_names']):
            self.file_paths[graph_name] = dataset_path.format(dataset_name, 'test')

        assert optimizer_name == 'trust_constr' or optimizer_name == 'slsqp', 'Invalid Optimizer {0}'.format(optimizer_name)
        if optimizer_name == 'trust_constr':
            self.optimizer = TrustConstr(params=params)
        elif optimizer_name == 'slsqp':
            self.optimizer = SLSQP(params=params)

    def optimize(self):
        # Load Graphs
        graph_path = 'graphs/{0}.tntp'
        test_graphs = {}
        for graph_name in self.params['test_graph_names']:
            graph = load_to_networkx(path=graph_path.format(graph_name))
            test_graphs[graph_name] = graph

        if not exists(self.output_folder):
            mkdir(self.output_folder)

        cost_headers = ['Index', 'Graph', 'Cost', 'Time (sec)']
        costs_path = self.output_folder + 'costs.csv'
        append_row_to_log(cost_headers, costs_path)

        index = 0
        for graph_name, graph in test_graphs.items():

            test_features = read_dataset(self.file_paths[graph_name], num_nodes=graph.number_of_nodes())
            for features in test_features:
                demands = features_to_demands(features)
                demands = np.reshape(demands, newshape=(demands.shape[0],))
                
                start = time()
                flows_per_iter, result = self.optimizer.optimize(graph=graph, demands=demands)
                end = time()

                flows = flows_per_iter[-1]
                flow_mat = self._flow_matrix(graph, flows)

                append_row_to_log([index, graph_name, result.fun, end - start], costs_path)

                # Add demands to graph
                flow_graph = graph.copy()
                for i, node in enumerate(graph.nodes()):
                    flow_graph.add_node(node, demand=demands[i])

                plot_flow_graph(flow_graph, flow_mat, '{0}flows-{1}-{2}.png'.format(self.output_folder, graph_name, index))

                index += 1

    def _flow_matrix(self, graph, flows):
        num_nodes = graph.number_of_nodes()
        flow_mat = np.zeros(shape=(num_nodes, num_nodes), dtype=float)

        for i, (src, dest) in enumerate(graph.edges()):
            flow_mat[src, dest] = flows[i]

        return flow_mat
