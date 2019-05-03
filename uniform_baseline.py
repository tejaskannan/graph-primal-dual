import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
import os
from time import time
from load import load_to_networkx
from dataset import DatasetManager, Series, DataSeries
from layers import SparseMinCostFlow
from utils import features_to_demands, sparse_subtract
from utils import sparse_scalar_mul, sparse_matrix_to_tensor
from utils import append_row_to_log
from cost_functions import get_cost_function
from constants import SMALL_NUMBER
from plot import plot_flow_graph_sparse


class UniformBaseline:

    def __init__(self, params):
        self.params = params

        assert len(params['test_graph_names']) == 1, 'Uniform Baseline only supports a single graph.'

        self.graph_name = params['test_graph_names'][0]
        cost_fn_name = params['cost_fn']['name']
        self.output_folder = '{0}/uniform-{1}-{2}/'.format(params['output_folder'], self.graph_name, cost_fn_name)

        file_paths = {
            Series.TRAIN: {},
            Series.VALID: {},
            Series.TEST: {}
        }

        dataset_path = 'datasets/{0}_{1}.txt'
        for dataset_name, graph_name in zip(self.params['test_dataset_names'], self.params['test_graph_names']):
            file_paths[Series.TEST][graph_name] = dataset_path.format(dataset_name, 'test')

        self.dataset = DatasetManager(file_paths=file_paths, params=self.params['batch_params'])

    def eval(self):

        # currently only supports running on a single graph
        graph = load_to_networkx('graphs/{0}.tntp'.format(self.graph_name))
        num_nodes = graph.number_of_nodes()

        # Create uniform outgoing proportions
        adj_matrix = nx.adjacency_matrix(graph)
        prop_values = []
        for src, dest in graph.edges():
            prop_values.append(1.0 / float(graph.out_degree(src)))
        prop_mat = sp.csr_matrix((np.array(prop_values), adj_matrix.indices, adj_matrix.indptr))

        prop_tensor = sparse_matrix_to_tensor(prop_mat)

        # Load dataset to get demands for each problem instance
        graphs = {self.graph_name: graph}
        self.dataset.load(series=Series.TEST, num_nodes=num_nodes, graphs=graphs,
                          num_neighborhoods=1, unique_neighborhoods=False)

        batches = self.dataset.create_batches(series=Series.TEST, batch_size=1, shuffle=False)
        num_batches = len(batches[DataSeries.NODE])

        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        log_path = '{0}costs.csv'.format(self.output_folder)
        if os.path.exists(log_path):
            os.remove(log_path)

        append_row_to_log(['Instance', 'Graph Name', 'Cost', 'Time (sec)'], log_path)

        with tf.Session() as sess:

            prop_mat_ph = tf.sparse_placeholder(dtype=tf.float32,
                                                shape=[None, num_nodes],
                                                name='prop-mat-ph')
            demands_ph = tf.placeholder(dtype=tf.float32,
                                        shape=[num_nodes, 1],
                                        name='demands-ph')

            flow_ops = self.solver(prop_mat=prop_mat_ph, demands=demands_ph)

            for i in range(num_batches):
                demands = features_to_demands(batches[DataSeries.NODE][i])

                feed_dict = {
                    prop_mat_ph: prop_tensor,
                    demands_ph: demands
                }

                start = time()
                flow_cost, flows = sess.run(flow_ops, feed_dict=feed_dict)
                elapsed = time() - start

                append_row_to_log([i, self.graph_name, flow_cost, elapsed], log_path)

                flow_graph = self.add_flow_values(graph, flows, demands)
                nx.write_gexf(flow_graph, '{0}{1}-{2}.gexf'.format(self.output_folder, self.graph_name, i))

                if self.params['plot_flows']:
                    file_path = '{0}{1}-{2}.png'.format(self.output_folder, self.graph_name, i)
                    plot_flow_graph_sparse(flow_graph, flows, file_path, use_node_weights=False)

    def solver(self, prop_mat, demands):
        """
        prop_mat is a V x V sparse tensor
        demands is a V x 1 dense tensor
        """
        mcf_solver = SparseMinCostFlow(flow_iters=self.params['flow_iters'])
        flow = mcf_solver(inputs=prop_mat, demands=demands)
        flow_transpose = tf.sparse.transpose(flow, perm=[1, 0])

        # There seems to be a bug when computing gradients for sparse.minimum, so
        # we instead use the alternative formula for minimum below
        # min(a, b) = 0.5 * (a + b - |a - b|)
        flow_add = tf.sparse.add(flow, flow_transpose)
        flow_sub_abs = tf.abs(sparse_subtract(flow, flow_transpose))
        min_flow = sparse_subtract(flow_add, flow_sub_abs)
        flow = tf.sparse.add(flow, sparse_scalar_mul(min_flow, -0.5),
                             threshold=SMALL_NUMBER)

        cost_fn = get_cost_function(self.params['cost_fn'])
        flow_cost = tf.reduce_sum(cost_fn.apply(flow.values))

        return flow_cost, flow

    def add_flow_values(self, graph, flows, demands):
        graph = graph.copy()

        for edge in graph.edges():
            graph.add_edge(edge[0], edge[1], flow=0.0)

        for edge, value in zip(flows.indices, flows.values):
            if graph.has_edge(*edge):
                graph.add_edge(edge[0], edge[1], flow=float(value))

        for node in graph.nodes():
            graph.add_node(node, demand=float(demands[node][0]))

        return graph
