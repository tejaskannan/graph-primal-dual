import numpy as np
import networkx as nx
import tensorflow as tf
import os
import math
from datetime import datetime
from time import time
from utils.utils import append_row_to_log, delete_if_exists
from utils.constants import BIG_NUMBER, LINE
from utils.graph_utils import add_features
from core.plot import plot_road_flow_graph
from core.dataset import DatasetManager, Series
from models.optimization_models import SLSQP, TrustConstr


PRINT_THRESHOLD = 100


class ModelRunner:

    def __init__(self, params):
        self.params = params
        self.timestamp = datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        cost_fn_name = params['cost_fn']['name']
        normalizer = 'sparsemax' if params['use_sparsemax'] else 'softmax'
        true_cost = '-true-cost' if params['use_true_cost'] else ''
        self.output_folder = '{0}/{1}-{2}-{3}-{4}-{5}{6}/'.format(params['output_folder'],
                                                           params['name'],
                                                           params['graph_name'],
                                                           cost_fn_name,
                                                           normalizer,
                                                           true_cost,
                                                           self.timestamp)
        
        self.num_node_features = 2
        self.embedding_size = 2*self.params['num_neighborhoods'] + 2

        self.dataset = DatasetManager(params=self.params)
        self.dataset.load_graphs()

    def train(self):
        num_neighborhoods = self.params['num_neighborhoods']

        # Initialize model
        model = self.create_model(params=self.params)

        # Model placeholders
        ph_dict = self.create_placeholders(model=model,
                                           max_num_nodes=self.dataset.num_nodes,
                                           embedding_size=self.embedding_size,
                                           num_neighborhoods=num_neighborhoods,
                                           max_degree=self.dataset.max_degree,
                                           max_out_neighborhood_degrees=self.dataset.max_out_neighborhood_degrees,
                                           max_in_neighborhood_degrees=self.dataset.max_in_neighborhood_degrees)

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

        batch_size = self.params['batch_size']

        # Log training start time
        time_log = self.output_folder + 'time.csv'
        delete_if_exists(time_log)
        start_time = datetime.now()
        append_row_to_log(['Start Time', start_time.strftime('%m-%d-%Y-%H-%M-%S')], time_log)

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
                                                  max_num_nodes=self.dataset.num_nodes,
                                                  max_out_neighborhood_degrees=self.dataset.max_out_neighborhood_degrees,
                                                  max_in_neighborhood_degrees=self.dataset.max_in_neighborhood_degrees)

                outputs = model.run_train_step(feed_dict=feed_dict)
                avg_loss = outputs[0]
                loss = outputs[1]

                # summary = outputs[2]
                # model.train_writer.add_summary(summary, i)

                train_losses.append(avg_loss)
                self.dataset.report_losses(loss, indices)

                print('Average train loss for batch {0}/{1}: {2}'.format(i+1, num_train_batches, avg_loss))

            print(LINE)

            # Validation Batches
            valid_batches = self.dataset.create_batches(series=Series.VALID,
                                                        batch_size=batch_size,
                                                        shuffle=True)
            num_valid_batches = self.dataset.num_batches(series=Series.TEST, batch_size=batch_size)
            valid_losses = []
            for i, batch in enumerate(valid_batches):

                feed_dict = self.create_feed_dict(placeholders=ph_dict,
                                                  batch=batch,
                                                  batch_size=batch_size,
                                                  data_series=Series.VALID,
                                                  max_degree=self.dataset.max_degree,
                                                  max_num_nodes=self.dataset.num_nodes,
                                                  max_out_neighborhood_degrees=self.dataset.max_out_neighborhood_degrees,
                                                  max_in_neighborhood_degrees=self.dataset.max_in_neighborhood_degrees)

                outputs = model.inference(feed_dict=feed_dict)
                avg_loss = outputs['loss']
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

        # Log ending time
        end_time = datetime.now()
        append_row_to_log(['End Time', end_time.strftime('%m-%d-%Y-%H-%M-%S')], time_log)
        append_row_to_log(['Delta', str(end_time - start_time)], time_log)

    def test(self, model_path=None):
        self.params['optimizer']['use_optimizer'] = False

        # Load Graphs
        graph = self.dataset.graph_data.graph

        num_neighborhoods = self.params['num_neighborhoods']

        # Initialize model
        model = self.create_model(params=self.params)

        # Model placeholders
        ph_dict = self.create_placeholders(model=model,
                                           max_num_nodes=self.dataset.num_nodes,
                                           embedding_size=self.embedding_size,
                                           num_neighborhoods=num_neighborhoods,
                                           max_degree=self.dataset.max_degree,
                                           max_out_neighborhood_degrees=self.dataset.max_out_neighborhood_degrees,
                                           max_in_neighborhood_degrees=self.dataset.max_in_neighborhood_degrees)

        # Create model
        model.build(**ph_dict)
        model.init()

        if model_path is not None:
            model.restore(model_path)
        else:
            model_path = self.output_folder
            if not os.path.exists(model_path):
                os.mkdir(model_path)

        # Load test data
        self.dataset.load(series=Series.TEST)

        batch_size = self.params['batch_size']
        test_batches = self.dataset.create_batches(series=Series.TEST,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        # Iniitalize Testing Log
        log_headers = ['Test Instance', 'Graph', 'Flow Cost', 'Dual Cost', 'Time (sec)']

        if self.params['optimizer']['use_optimizer']:
            log_headers += ['Flow with Optimizer', 'Num Iters']
            log_path = model_path + 'costs-{0}.csv'.format(self.params['optimizer']['optimizer_name'])
        else:
            log_path = model_path + 'costs.csv'

        delete_if_exists(log_path)
        append_row_to_log(log_headers, log_path)

        # Compute indices which will be used for plotting. This is done in a deterministic
        # manner to make it easier to compare different runs.
        num_test_batches = self.dataset.num_batches(series=Series.TEST, batch_size=batch_size)
        num_test_samples = num_test_batches * batch_size

        self.params['plot_fraction'] = 0.01
        # self.params['plot_flows'] = False
        step = int(1.0 / self.params['plot_fraction'])
        plot_indices = set(range(0, num_test_samples, step))

        for i, batch in enumerate(test_batches):

            feed_dict = self.create_feed_dict(placeholders=ph_dict,
                                              batch=batch,
                                              batch_size=batch_size,
                                              data_series=Series.TEST,
                                              max_degree=self.dataset.max_degree,
                                              max_num_nodes=self.dataset.num_nodes,
                                              max_out_neighborhood_degrees=self.dataset.max_out_neighborhood_degrees,
                                              max_in_neighborhood_degrees=self.dataset.max_in_neighborhood_degrees,
                                              name=self.params['name'])

            start = time()
            outputs = model.inference(feed_dict=feed_dict)
            elapsed = time() - start

            avg_time = elapsed / batch_size

            for j in range(batch_size):

                index = i * batch_size + j

                graph_name = batch[j].graph_name

                flow = outputs['flow'][j]
                flow_cost = outputs['flow_cost'][j]
                pred_weights = outputs['normalized_weights'][j]
                dual_cost = outputs['dual_cost'][j]

                demands = np.array(batch[j].demands)

                if self.params['optimizer']['use_optimizer']:
                    adj_lst = batch[j].adj_lst
                    initial = np.zeros(shape=(graph.number_of_edges(),))
                    edge_index = 0
                    for i in range(flow.shape[0]):
                        for j in range(flow.shape[1]):
                            if adj_lst[i, j] != batch[j].num_nodes:
                                initial[edge_index] = flow[i, j]
                                edge_index += 1

                    if self.params['optimizer']['optimizer_name'] == 'trust_constr':
                        optimizer_model = TrustConstr(params=self.params)
                    elif self.params['optimizer']['optimizer_name'] == 'slsqp':
                        optimizer_model = SLSQP(params=self.params)

                    optimizer_demands = demands.reshape(-1)
                    flows_per_iter, result = optimizer_model.optimize(graph=graph, demands=optimizer_demands, initial=initial)
                    opt_cost, num_iters = result.fun, result.nit

                node_features = {
                    'demand': demands
                }
                edge_features = {
                    'flow': flow,
                    'flow_proportion': pred_weights 
                }

                flow_graph = add_features(graph=graph,
                                          node_features=node_features,
                                          edge_features=edge_features)

                if self.params['plot_flows'] and index == 400:
                    flow_path = '{0}flows-{1}-{2}'.format(model_path, graph_name, index)
                    prop_path = '{0}flow-prop-{1}-{2}'.format(model_path, graph_name, index)
                    attn_weight_path = '{0}attn-weights-{1}-{2}'.format(model_path, graph_name, index)

                    plot_road_flow_graph(graph=flow_graph, field='flow', graph_name=self.params['graph_title'], file_path=flow_path)
                    plot_road_flow_graph(graph=flow_graph, field='flow_proportion', graph_name=self.params['graph_title'], file_path=prop_path)

                    if 'attn_weights' in outputs:
                        attn_weights = outputs['attn_weights'][j]
                        num_nodes = batch[j].num_nodes
                        plot_weights(weight_matrix=attn_weights,
                                     file_path=attn_weight_path,
                                     num_samples=self.params['plot_weight_samples'],
                                     num_nodes=num_nodes)

                # Log Outputs
                row = [index, graph_name, flow_cost, dual_cost, avg_time]
                if self.params['optimizer']['use_optimizer']:
                    row += [opt_cost, num_iters]
                append_row_to_log(row, log_path)

    def create_placeholders(self, model, **kwargs):
        raise NotImplementedError()

    def create_feed_dict(self, placeholders, batch, batch_size, data_series, **kwargs):
        raise NotImplementedError()

    def create_model(self, params):
        raise NotImplementedError()
