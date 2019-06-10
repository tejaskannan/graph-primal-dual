import tensorflow as tf
import numpy as np
import networkx as nx
from models.dense_model import DenseModel
from utils.utils import create_node_embeddings, features_to_demands
from utils.utils import add_features, append_row_to_log
from utils.constants import BIG_NUMBER, FLOW_THRESHOLD, LINE
from core.plot import plot_flow_graph, plot_costs
from core.dataset import DatasetManager, Series
from os import mkdir
from os.path import exists
from datetime import datetime
from time import time


class DenseBaseline:

    def __init__(self, params):
        self.params = params
        self.timestamp = datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        self.output_folder = '{0}/{1}-dense-{2}/'.format(params['output_folder'], params['graph_name'], self.timestamp)
        self.num_node_features = 7

        train_file = 'datasets/{0}_train.txt'.format(self.params['dataset_name'])
        valid_file = 'datasets/{0}_valid.txt'.format(self.params['dataset_name'])
        test_file = 'datasets/{0}_test.txt'.format(self.params['dataset_name'])
        self.dataset = DatasetManager(train_file, valid_file, test_file, params=self.params['batch_params'])

    def compute_baseline(self):

        # Load input graph
        graph_path = 'graphs/{0}.tntp'.format(self.params['graph_name'])
        graph = load_to_networkx(path=graph_path)

        # Fetch adjacency matrix
        adj_mat = nx.adjacency_matrix(graph).todense()

        num_nodes = graph.number_of_nodes()

        # Initialize model
        model = DenseModel(params=self.params)
        node_ph, demands_ph, adj_ph = self.create_placeholders(model, num_nodes)

        model.build(node_features=node_ph,
                    demands=demands_ph,
                    adj=adj_ph,
                    num_output_features=num_nodes)
        model.init()

        # Create output folder
        if not exists(self.output_folder):
            mkdir(self.output_folder)

        # Load testing set
        self.dataset.load(series=Series.TEST, num_nodes=num_nodes)
        test_batches = self.dataset.create_batches(series=Series.TEST, batch_size=1, shuffle=False)

        num_samples = self.params['batch_size']
        max_iters = self.params['flow_iters']

        times = []
        min_costs = []
        all_costs = []

        # Initialize logging
        log_path = '{0}log.csv'.format(self.output_folder)
        append_row_to_log(['Test Point', 'Min Cost', 'Time (sec)'], log_path)

        for i, node_demands in enumerate(test_batches):
            prev_cost = BIG_NUMBER
            convergence_count = 0
            costs = []

            print('Test Point {0}'.format(i))
            print(LINE)

            # Create demands from features
            demands = features_to_demands(node_demands[0])

            node_features = np.random.uniform(size=(num_samples, num_nodes, self.num_node_features))

            # Insert Demand values into node features
            for k in range(demands.shape[0]):
                node_features[:, k, 0] = demands[k][0]

            # Optimize repeatedly
            j = 0
            elapsed = 0.0
            while j < max_iters:
                feed_dict = {
                    node_ph: node_features,
                    demands_ph: demands,
                    adj_ph: adj_mat
                }

                start = time()
                outputs = model.run_train_step(feed_dict=feed_dict)
                end = time()
                elapsed += (end - start)

                cost = outputs[0]

                costs.append(cost)

                if abs(prev_cost - cost) < FLOW_THRESHOLD:
                    convergence_count += 1

                if convergence_count >= self.params['patience']:
                    break

                j += 1

                if j % 100 == 0:
                    print('Cost after iteration {0}: {1}'.format(j, cost))

            min_cost = np.min(costs)

            print(LINE)
            print('Cost from test point {0}: {1}'.format(i, min_cost))
            print('Total Time: {0}s'.format(elapsed))
            print(LINE)

            times.append(elapsed)
            min_costs.append(min_cost)

            append_row_to_log([i, min_cost, elapsed], log_path)

            # Create flow graphs
            outputs = model.inference(feed_dict=feed_dict)

            flow_cost = outputs[1]
            flows = outputs[2][0]
            flow_proportions = outputs[3][0]

            flow_graph = add_features(graph, demands=node_demands[0], flows=flows,
                                      proportions=flow_proportions)
            plot_flow_graph(flow_graph, flows, '{0}flows-{1}.png'.format(self.output_folder, i))
            plot_flow_graph(flow_graph, flow_proportions, '{0}flow-prop-{1}.png'.format(self.output_folder, i))

            all_costs.append(costs)
            costs = []

        plot_costs(all_costs, '{0}costs.png'.format(self.output_folder))

    def create_placeholders(self, model, num_nodes):
        node_ph = model.create_placeholder(dtype=tf.float32,
                                           shape=[None, num_nodes, self.num_node_features],
                                           name='node-ph',
                                           is_sparse=False)
        demands_ph = model.create_placeholder(dtype=tf.float32,
                                              shape=[num_nodes, 1],
                                              name='demands-ph',
                                              is_sparse=False)
        adj_ph = model.create_placeholder(dtype=tf.float32,
                                          shape=[num_nodes, num_nodes],
                                          name='adj-ph',
                                          is_sparse=False)
        return node_ph, demands_ph, adj_ph
