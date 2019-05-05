import numpy as np
import math
import networkx as nx
import tensorflow as tf
import os
from datetime import datetime
from time import time
from utils.utils import create_demands, append_row_to_log, create_node_embeddings
from utils.utils import add_features_sparse, create_node_bias, restore_params
from utils.utils import sparse_matrix_to_tensor, features_to_demands, random_walk_neighborhoods
from utils.utils import add_features, adj_mat_to_node_bias, delete_if_exists
from utils.constants import BIG_NUMBER, LINE
from core.plot import plot_flow_graph_sparse, plot_flow_graph, plot_weights
from core.load import load_to_networkx, read_dataset
from core.dataset import DatasetManager, Series, DataSeries


PRINT_THRESHOLD = 100


class ModelRunner:

    def __init__(self, params):
        self.params = params
        self.timestamp = datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        graph_names = '-'.join(params['train_graph_names'])
        cost_fn_name = params['cost_fn']['name']
        self.output_folder = '{0}/{1}-{2}-{3}-{4}/'.format(params['output_folder'],
                                                           params['name'],
                                                           graph_names,
                                                           cost_fn_name,
                                                           self.timestamp)
        
        self.num_node_features = 2
        self.embedding_size = 2 * self.params['num_neighborhoods'] + 2

        file_paths = {
            Series.TRAIN: {},
            Series.VALID: {},
            Series.TEST: {}
        }
        dataset_path = 'datasets/{0}_{1}.pkl.gz'
        for dataset_name, graph_name in zip(self.params['train_dataset_names'], self.params['train_graph_names']):
            file_paths[Series.TRAIN][graph_name] = dataset_path.format(dataset_name, 'train')
            file_paths[Series.VALID][graph_name] = dataset_path.format(dataset_name, 'valid')

        for dataset_name, graph_name in zip(self.params['test_dataset_names'], self.params['test_graph_names']):
            file_paths[Series.TEST][graph_name] = dataset_path.format(dataset_name, 'test')

        self.dataset = DatasetManager(file_paths=file_paths, params=self.params['batch_params'])

    def train(self):

        # Load Graphs
        graphs, _, num_nodes = self._load_graphs()

        num_neighborhoods = self.params['num_neighborhoods']

        # Initialize model
        model = self.create_model(params=self.params)

        # Model placeholders
        ph_dict = self.create_placeholders(model=model,
                                           num_nodes=num_nodes,
                                           embedding_size=self.embedding_size,
                                           num_neighborhoods=num_neighborhoods)

        # Create model
        model.build(**ph_dict)
        model.init()

        # Create output folder and initialize logging
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        log_headers = ['Epoch', 'Avg Train Loss', 'Avg Valid Loss']
        log_path = self.output_folder + 'log.csv'
        delete_if_exists(log_path)

        append_row_to_log(log_headers, log_path)

        # Load training and validation datasets
        self.dataset.load(series=Series.TRAIN,
                          graphs=graphs,
                          num_nodes=num_nodes,
                          num_neighborhoods=num_neighborhoods,
                          unique_neighborhoods=self.params['unique_neighborhoods'])
        self.dataset.load(series=Series.VALID,
                          graphs=graphs,
                          num_nodes=num_nodes,
                          num_neighborhoods=num_neighborhoods,
                          unique_neighborhoods=self.params['unique_neighborhoods'])

        self.dataset.init(num_epochs=self.params['epochs'])
        self.dataset.normalize_embeddings()

        # Variables for early stopping
        convergence_count = 0
        prev_loss = BIG_NUMBER

        # Sparse batches must be size 1 due to the lack of support for 3D sparse operations
        # in tensorflow
        batch_size = 1 if self.params['sparse'] else self.params['batch_size']

        for epoch in range(self.params['epochs']):

            print(LINE)
            print('Epoch {0}'.format(epoch))
            print(LINE)

            # Training Batches
            num_train_batches = int(math.ceil(self.dataset.num_train_points / batch_size))
            train_losses = []
            for i in range(num_train_batches):

                batch, indices = self.dataset.get_train_batch(batch_size=batch_size,
                                                              is_sparse=self.params['sparse'])

                feed_dict = self.create_feed_dict(placeholders=ph_dict,
                                                  batch=batch,
                                                  index=i,
                                                  batch_size=batch_size,
                                                  data_series=Series.TRAIN)

                outputs = model.run_train_step(feed_dict=feed_dict)
                avg_loss = outputs[0]
                loss = outputs[1]

                train_losses.append(avg_loss)
                self.dataset.report_losses(loss, indices)

                if not self.params['sparse'] or (i+1) % PRINT_THRESHOLD == 0:
                    if self.params['sparse']:
                        start = (i+1) - PRINT_THRESHOLD
                        avg_loss = np.average(train_losses[start:i+1])
                    print('Average train loss for batch {0}/{1}: {2}'.format(i+1, num_train_batches, avg_loss))

            print(LINE)

            # Validation Batches
            valid_batches = self.dataset.create_shuffled_batches(series=Series.VALID,
                                                                 batch_size=batch_size,
                                                                 is_sparse=self.params['sparse'])
            num_valid_batches = len(valid_batches[DataSeries.NODE])
            valid_losses = []
            for i in range(num_valid_batches):

                feed_dict = self.create_feed_dict(placeholders=ph_dict,
                                                  batch=valid_batches,
                                                  index=i,
                                                  batch_size=batch_size,
                                                  data_series=Series.VALID)

                outputs = model.inference(feed_dict=feed_dict)
                avg_loss = outputs[0]
                valid_losses.append(avg_loss)

                if not self.params['sparse'] or (i+1) % PRINT_THRESHOLD == 0:
                    if self.params['sparse']:
                        start = (i+1) - PRINT_THRESHOLD
                        avg_loss = np.average(valid_losses[start:i+1])
                    print('Average valid loss for batch {0}/{1}: {2}'.format(i+1, num_valid_batches, avg_loss))

            print(LINE)

            avg_train_loss = np.average(train_losses)
            print('Average training loss: {0}'.format(avg_train_loss))

            avg_valid_loss = np.average(valid_losses)
            print('Average validation loss: {0}'.format(avg_valid_loss))

            log_row = [epoch, avg_train_loss, avg_valid_loss]
            append_row_to_log(log_row, log_path)

            # Early Stopping Counters
            if abs(prev_loss - avg_valid_loss) < self.params['early_stop_threshold']:
                convergence_count += 1
            else:
                convergence_count = 0

            if avg_valid_loss < prev_loss:
                print('Saving model...')
                model.save(self.output_folder)
                prev_loss = avg_valid_loss

            if convergence_count >= self.params['patience']:
                print('Early Stopping.')
                break

    def test(self, model_path):
        # Load Graphs
        _, graphs, num_nodes = self._load_graphs()

        num_neighborhoods = self.params['num_neighborhoods']

        # Initialize model
        model = self.create_model(params=self.params)

        # Model placeholders
        ph_dict = self.create_placeholders(model=model,
                                           num_nodes=num_nodes,
                                           embedding_size=self.embedding_size,
                                           num_neighborhoods=num_neighborhoods)

        # Create model
        model.build(**ph_dict)
        model.init()
        model.restore(model_path)

        # Load test data and normalize embeddings
        self.dataset.load(series=Series.TRAIN,
                          num_nodes=num_nodes,
                          graphs=graphs,
                          num_neighborhoods=num_neighborhoods)
        self.dataset.load(series=Series.TEST,
                          num_nodes=num_nodes,
                          graphs=graphs,
                          num_neighborhoods=num_neighborhoods,
                          unique_neighborhoods=self.params['unique_neighborhoods'])
        self.dataset.normalize_embeddings()

        test_batches = self.dataset.create_batches(series=Series.TEST,
                                                   batch_size=self.params['batch_size'],
                                                   shuffle=False,
                                                   is_sparse=self.params['sparse'])
        num_test_batches = len(test_batches[DataSeries.NODE])

        # Iniitalize Testing Log
        log_headers = ['Test Instance', 'Graph', 'Flow Cost', 'Dual Cost', 'Time (sec)']
        log_path = model_path + 'cost.csv'
        delete_if_exists(log_path)
        append_row_to_log(log_headers, log_path)

        for i in range(num_test_batches):

            feed_dict = self.create_feed_dict(placeholders=ph_dict,
                                              batch=test_batches,
                                              index=i,
                                              batch_size=self.params['batch_size'],
                                              data_series=Series.TEST)

            batch_size = len(test_batches[DataSeries.GRAPH_NAME][i])

            start = time()
            outputs = model.inference(feed_dict=feed_dict)
            elapsed = time() - start
            avg_time = elapsed / batch_size

            for j in range(batch_size):
                graph_name = test_batches[DataSeries.GRAPH_NAME][i][j]
                graph = graphs[graph_name]

                flow_cost = outputs[1][j]
                flows = outputs[2][j]
                flow_proportions = outputs[3][j]
                dual_cost = outputs[4][j]
                weights = outputs[6][j]
                node_weights = outputs[7][j]

                demands = test_batches[DataSeries.NODE][i][j]
                if self.params['sparse']:
                    flow_graph = add_features_sparse(graph,
                                                     demands=demands,
                                                     flows=flows,
                                                     proportions=flow_proportions,
                                                     node_weights=node_weights)
                else:
                    flow_graph = add_features(graph,
                                              demands=demands,
                                              flows=flows,
                                              proportions=flow_proportions,
                                              node_weights=node_weights)

                index = i * self.params['batch_size'] + j
                
                # Log Outputs
                append_row_to_log([index, graph_name, flow_cost, dual_cost, avg_time], log_path)

                # Write output graph to Graph XML
                nx.write_gexf(flow_graph, '{0}graph-{1}-{2}.gexf'.format(model_path, graph_name, index))

                if self.params['plot_flows']:
                    if self.params['sparse']:
                        plot_flow_graph_sparse(flow_graph, flows, '{0}flows-{1}-{2}.png'.format(model_path, graph_name, index))
                        plot_flow_graph_sparse(flow_graph, flow_proportions, '{0}flow-prop-{1}-{2}.png'.format(model_path, graph_name, index))
                    else:
                        plot_flow_graph(flow_graph, flows, '{0}flows-{1}-{2}.png'.format(model_path, graph_name, index))
                        plot_flow_graph(flow_graph, flow_proportions, '{0}flow-prop-{1}-{2}.png'.format(model_path, graph_name, index))

                plot_weights(weights, '{0}weights-{1}-{2}.png'.format(model_path, graph_name, index), num_samples=5)

    def create_placeholders(self, model, num_nodes, embedding_size, **kwargs):
        raise NotImplementedError()

    def create_feed_dict(self, placeholders, batch, index, batch_size, data_series):
        raise NotImplementedError()

    def create_model(self, params):
        raise NotImplementedError()

    def _num_neighborhoods(self, graph):
        if 'num_neighborhoods' in self.params:
            return self.params['num_neighborhoods']
        return max(2, int(math.log(graph.number_of_nodes())))

    def _load_graphs(self):
        graph_path = 'graphs/{0}.tntp'

        train_graphs = {}
        for graph_name in self.params['train_graph_names']:
            graph = load_to_networkx(path=graph_path.format(graph_name))
            train_graphs[graph_name] = graph

        test_graphs = {}
        for graph_name in self.params['test_graph_names']:
            graph = load_to_networkx(path=graph_path.format(graph_name))
            test_graphs[graph_name] = graph

        num_train_nodes = np.max([g.number_of_nodes() for g in train_graphs.values()])
        num_test_nodes = np.max([g.number_of_nodes() for g in test_graphs.values()])

        return train_graphs, test_graphs, max(num_train_nodes, num_test_nodes)
