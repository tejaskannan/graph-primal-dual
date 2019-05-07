import numpy as np
import math
import networkx as nx
import tensorflow as tf
import os
from datetime import datetime
from time import time
from utils.utils import append_row_to_log, delete_if_exists
from utils.constants import BIG_NUMBER, LINE
from utils.graph_utils import add_features
from core.plot import plot_flow_graph_adj, plot_weights
from core.load import load_to_networkx, read_dataset
from core.dataset import DatasetManager, Series


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

        folders = {
            Series.TRAIN: {},
            Series.VALID: {},
            Series.TEST: {}
        }
        dataset_folder = 'datasets/{0}/{1}'
        for dataset_name, graph_name in zip(self.params['train_dataset_names'], self.params['train_graph_names']):
            folders[Series.TRAIN][graph_name] = dataset_folder.format(dataset_name, 'train')
            folders[Series.VALID][graph_name] = dataset_folder.format(dataset_name, 'valid')

        for dataset_name, graph_name in zip(self.params['test_dataset_names'], self.params['test_graph_names']):
            folders[Series.TEST][graph_name] = dataset_folder.format(dataset_name, 'test')

        self.dataset = DatasetManager(data_folders=folders, params=self.params)
        self.dataset.load_graphs(normalize=True)

    def train(self):

        # Load Graphs
        graphs = self.dataset.train_graphs

        num_neighborhoods = self.params['num_neighborhoods']

        # Initialize model
        model = self.create_model(params=self.params)

        # Model placeholders
        ph_dict = self.create_placeholders(model=model,
                                           max_num_nodes=self.dataset.max_num_nodes,
                                           embedding_size=self.embedding_size,
                                           num_neighborhoods=num_neighborhoods,
                                           max_degree=self.dataset.max_degree,
                                           max_neighborhood_degrees=self.dataset.max_neighborhood_degrees)

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
        self.dataset.load(series=Series.TRAIN)
        self.dataset.load(series=Series.VALID)

        # Intialize training variables for online batch selection
        self.dataset.init(num_epochs=self.params['epochs'])

        # Variables for early stopping
        convergence_count = 0
        prev_loss = BIG_NUMBER

        # Sparse batches must be size 1 due to the lack of support for 3D sparse operations
        # in tensorflow
        batch_size = self.params['batch_size']

        for epoch in range(self.params['epochs']):

            print(LINE)
            print('Epoch {0}'.format(epoch))
            print(LINE)

            # Training Batches
            num_train_batches = int(math.ceil(self.dataset.num_train_points / batch_size))
            train_losses = []
            for i in range(num_train_batches):

                batch, indices = self.dataset.get_train_batch(batch_size=batch_size)

                feed_dict = self.create_feed_dict(placeholders=ph_dict,
                                                  batch=batch,
                                                  batch_size=batch_size,
                                                  data_series=Series.TRAIN,
                                                  max_degree=self.dataset.max_degree,
                                                  max_num_nodes=self.dataset.max_num_nodes,
                                                  max_neighborhood_degrees=self.dataset.max_neighborhood_degrees)

                outputs = model.run_train_step(feed_dict=feed_dict)
                avg_loss = outputs[0]
                loss = outputs[1]

                train_losses.append(avg_loss)
                self.dataset.report_losses(loss, indices)

                print('Average train loss for batch {0}/{1}: {2}'.format(i+1, num_train_batches, avg_loss))

            print(LINE)

            # Validation Batches
            valid_batches = self.dataset.create_batches(series=Series.VALID,
                                                        batch_size=batch_size,
                                                        shuffle=True)
            num_valid_batches = len(valid_batches)
            valid_losses = []
            for i, batch in enumerate(valid_batches):

                feed_dict = self.create_feed_dict(placeholders=ph_dict,
                                                  batch=batch,
                                                  batch_size=batch_size,
                                                  data_series=Series.VALID,
                                                  max_degree=self.dataset.max_degree,
                                                  max_num_nodes=self.dataset.max_num_nodes,
                                                  max_neighborhood_degrees=self.dataset.max_neighborhood_degrees)

                outputs = model.inference(feed_dict=feed_dict)
                avg_loss = outputs[0]
                valid_losses.append(avg_loss)

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
        graphs = self.dataset.test_graphs

        num_neighborhoods = self.params['num_neighborhoods']

        # Initialize model
        model = self.create_model(params=self.params)

        # Model placeholders
        ph_dict = self.create_placeholders(model=model,
                                           max_num_nodes=self.dataset.max_num_nodes,
                                           embedding_size=self.embedding_size,
                                           num_neighborhoods=num_neighborhoods,
                                           max_degree=self.dataset.max_degree,
                                           max_neighborhood_degrees=self.dataset.max_neighborhood_degrees)

        # Create model
        model.build(**ph_dict)
        model.init()
        model.restore(model_path)

        # Load test data
        self.dataset.load(series=Series.TEST)

        batch_size = self.params['batch_size']
        test_batches = self.dataset.create_batches(series=Series.TEST,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        # Iniitalize Testing Log
        log_headers = ['Test Instance', 'Graph', 'Flow Cost', 'Dual Cost', 'Time (sec)']
        log_path = model_path + 'costs.csv'
        delete_if_exists(log_path)
        append_row_to_log(log_headers, log_path)

        # Compute indices which will be used for plotting. This is done in a deterministic
        # manner to make it easier to compare different runs.
        num_test_batches = len(test_batches)
        num_test_samples = num_test_batches * batch_size
        num_plot_samples = num_test_samples * self.params['plot_fraction']
        step = int(num_test_samples / num_plot_samples)
        plot_indices = set(range(0, num_test_samples, step))

        for i, batch in enumerate(test_batches):

            feed_dict = self.create_feed_dict(placeholders=ph_dict,
                                              batch=batch,
                                              batch_size=batch_size,
                                              data_series=Series.VALID,
                                              max_degree=self.dataset.max_degree,
                                              max_num_nodes=self.dataset.max_num_nodes,
                                              max_neighborhood_degrees=self.dataset.max_neighborhood_degrees)

            start = time()
            outputs = model.inference(feed_dict=feed_dict)
            elapsed = time() - start

            avg_time = elapsed / batch_size

            for j in range(batch_size):

                index = i * batch_size + j

                graph_name = batch[j].graph_name
                graph = graphs[graph_name]

                flow = outputs[1][j]
                flow_cost = outputs[2][j]
                adj_lst = outputs[3][j]
                pred_weights = outputs[4][j]
                dual_cost = outputs[5][j]
                node_weights = outputs[6][j]
                attn_weights = outputs[7][j]

                demands = np.array(batch[j].demands)

                node_features = {
                    'demand': demands,
                    'node_weight': node_weights
                }
                edge_features = {
                    'flow': flow,
                    'flow_proportion': pred_weights 
                }

                flow_graph = add_features(graph=graph,
                                          node_features=node_features,
                                          edge_features=edge_features)

                if self.params['plot_flows'] and index in plot_indices:
                    flow_path = '{0}flows-{1}-{2}.png'.format(model_path, graph_name, index)
                    prop_path = '{0}flow-prop-{1}-{2}.png'.format(model_path, graph_name, index)
                    attn_weight_path = '{0}attn-weights-{1}-{2}.png'.format(model_path, graph_name, index)

                    plot_flow_graph_adj(flow_graph, use_flow_props=False, file_path=flow_path)
                    plot_flow_graph_adj(flow_graph, use_flow_props=True, file_path=prop_path)
                    
                    num_nodes = batch[j].num_nodes
                    plot_weights(weight_matrix=attn_weights,
                                 file_path=attn_weight_path,
                                 num_samples=self.params['plot_weight_samples'],
                                 num_nodes=num_nodes)

                # Log Outputs
                append_row_to_log([index, graph_name, flow_cost, dual_cost, avg_time], log_path)

                # Write output graph to Graph XML
                nx.write_gexf(flow_graph, '{0}graph-{1}-{2}.gexf'.format(model_path, graph_name, index))

    def create_placeholders(self, model, **kwargs):
        raise NotImplementedError()

    def create_feed_dict(self, placeholders, batch, batch_size, data_series, **kwargs):
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
        num_nodes = max(num_train_nodes, num_test_nodes)

        graphs = list(train_graphs.values()) + list(test_graphs.values())
        degrees = max_degrees(graphs, k=self.params['num_neighborhoods'],
                              unique_neighborhoods=self.params['unique_neighborhoods'])

        train_out_deg = np.max([max([d for _, d in g.out_degree()]) for g in train_graphs.values()])
        train_in_deg = np.max([max([d for _, d in g.in_degree()]) for g in train_graphs.values()])

        test_out_deg = np.max([max([d for _, d in g.out_degree()]) for g in test_graphs.values()])
        test_in_deg = np.max([max([d for _, d in g.in_degree()]) for g in test_graphs.values()])
        max_degree = max(max(train_out_deg, train_in_deg), max(test_out_deg, test_in_deg))

        return train_graphs, test_graphs, num_nodes, max_degree, degrees
