import numpy as np
import networkx as nx
import tensorflow as tf
from mcf_model import MCFModel
from load import load_to_networkx, load_embeddings, read_dataset
from datetime import datetime
from os import mkdir
from os.path import exists
from utils import create_demands, append_row_to_log, create_batches
from utils import add_features, create_node_bias, restore_params
from plot import plot_flow_graph
from constants import BIG_NUMBER, LINE
from dataset import DatasetManager, Series


class MCF:

    def __init__(self, params):
        self.params = params
        self.timestamp = datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        self.output_folder = '{0}/{1}-{2}/'.format(params['output_folder'], params['graph_name'], self.timestamp)
        self.num_node_features = 1

        train_file = 'datasets/{0}_train.txt'.format(self.params['dataset_name'])
        valid_file = 'datasets/{0}_valid.txt'.format(self.params['dataset_name'])
        test_file = 'datasets/{0}_test.txt'.format(self.params['dataset_name'])
        self.dataset = DatasetManager(train_file, valid_file, test_file, params=self.params['batch_params'])

    def train(self):
        # Load graph
        graph_path = 'graphs/{0}.tntp'.format(self.params['graph_name'])
        graph = load_to_networkx(path=graph_path)

        # Create tensors for global graph properties
        adj_mat = nx.adjacency_matrix(graph).todense()
        node_bias = create_node_bias(graph)

        # Graph properties
        num_nodes = graph.number_of_nodes()
        embedding_size = self.params['node_embedding_size']

        # Load pre-trained embeddings
        index_path = 'embeddings/{0}.ann'.format(self.params['graph_name'])
        node_embeddings = load_embeddings(index_path=index_path,
                                          embedding_size=embedding_size,
                                          num_nodes=num_nodes)

        # Initialize model
        model = MCFModel(params=self.params)

        # Model placeholders
        node_ph = model.create_placeholder(dtype=tf.float32,
                                           shape=[None, num_nodes, self.num_node_features],
                                           name='node-ph',
                                           ph_type='dense')
        adj_ph = model.create_placeholder(dtype=tf.float32,
                                          shape=[num_nodes, num_nodes],
                                          name='adj-ph',
                                          ph_type='dense')
        node_embedding_ph = model.create_placeholder(dtype=tf.float32,
                                                     shape=[num_nodes, embedding_size],
                                                     name='node-embedding-ph',
                                                     ph_type='dense')
        node_bias_ph = model.create_placeholder(dtype=tf.float32,
                                                shape=[num_nodes, num_nodes],
                                                name='node-bias-ph',
                                                ph_type='dense')

        # Create model
        model.build(demands=node_ph,
                    node_embeddings=node_embedding_ph,
                    adj=adj_ph,
                    node_bias=node_bias_ph,
                    num_output_features=num_nodes)
        model.init()

        # Create output folder and initialize logging
        if not exists(self.output_folder):
            mkdir(self.output_folder)

        log_headers = ['Epoch', 'Avg Train Loss', 'Avg Valid Loss']
        log_path = self.output_folder + 'log.csv'
        append_row_to_log(log_headers, log_path)

        # Load training and validation datasets
        self.dataset.load(series=Series.TRAIN, num_nodes=num_nodes)
        self.dataset.load(series=Series.VALID, num_nodes=num_nodes)

        # Variables for early stopping
        convergence_count = 0
        prev_loss = BIG_NUMBER

        for epoch in range(self.params['epochs']):

            # Create batches
            train_batches = self.dataset.create_shuffled_batches(series=Series.TRAIN, batch_size=self.params['batch_size'])
            valid_batches = self.dataset.create_shuffled_batches(series=Series.VALID, batch_size=self.params['batch_size'])

            num_train_batches = len(train_batches)
            num_valid_batches = len(valid_batches)

            print(LINE)
            print('Epoch {0}'.format(epoch))
            print(LINE)

            # Training Batches
            train_losses = []
            for i, node_features in enumerate(train_batches):

                feed_dict = {
                    node_ph: node_features,
                    adj_ph: adj_mat,
                    node_embedding_ph: node_embeddings,
                    node_bias_ph: node_bias
                }
                avg_loss = model.run_train_step(feed_dict=feed_dict)
                train_losses.append(avg_loss)

                print('Average train loss for batch {0}/{1}: {2}'.format(i+1, num_train_batches, avg_loss))

            print(LINE)

            # Validation Batches
            valid_losses = []
            for i, node_features in enumerate(valid_batches):

                feed_dict = {
                    node_ph: node_features,
                    adj_ph: adj_mat,
                    node_embedding_ph: node_embeddings,
                    node_bias_ph: node_bias
                }
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

            if avg_valid_loss < prev_loss:
                print('Saving model...')
                model.save(self.output_folder)
                prev_loss = avg_valid_loss

            # Early Stopping
            if abs(prev_loss - avg_valid_loss) < self.params['early_stop_threshold']:
                convergence_count += 1
            else:
                convergence_count = 0

            if convergence_count >= self.params['patience']:
                print('Early Stopping.')
                break

    def test(self, model_path):
        # Load graph
        graph_path = 'graphs/{0}.tntp'.format(self.params['graph_name'])
        graph = load_to_networkx(path=graph_path)

        # Create tensors for global graph properties
        adj_mat = nx.adjacency_matrix(graph).todense()
        node_bias = create_node_bias(graph)

        # Graph properties
        num_nodes = graph.number_of_nodes()
        embedding_size = self.params['node_embedding_size']

        # Load pre-trained embeddings
        index_path = 'embeddings/{0}.ann'.format(self.params['graph_name'])
        node_embeddings = load_embeddings(index_path=index_path,
                                          embedding_size=embedding_size,
                                          num_nodes=num_nodes)

        # Initialize model
        model = MCFModel(params=self.params)

        # Model placeholders
        node_ph = model.create_placeholder(dtype=tf.float32,
                                           shape=[None, num_nodes, self.num_node_features],
                                           name='node-ph',
                                           ph_type='dense')
        adj_ph = model.create_placeholder(dtype=tf.float32,
                                          shape=[num_nodes, num_nodes],
                                          name='adj-ph',
                                          ph_type='dense')
        node_embedding_ph = model.create_placeholder(dtype=tf.float32,
                                                     shape=[num_nodes, embedding_size],
                                                     name='node-embedding-ph',
                                                     ph_type='dense')
        node_bias_ph = model.create_placeholder(dtype=tf.float32,
                                                shape=[num_nodes, num_nodes],
                                                name='node-bias-ph',
                                                ph_type='dense')

        # Create model
        model.build(demands=node_ph,
                    node_embeddings=node_embedding_ph,
                    adj=adj_ph,
                    node_bias=node_bias_ph,
                    num_output_features=num_nodes)
        model.init()
        model.restore(model_path)

        # Load test data
        test_file = 'datasets/{0}_test.txt'.format(self.params['dataset_name'])
        test_dataset = read_dataset(demands_path=test_file, num_nodes=num_nodes)
        test_batches = create_batches(dataset=test_dataset, batch_size=self.params['batch_size'])

        for i, node_features in enumerate(test_batches):

            feed_dict = {
                node_ph: node_features,
                adj_ph: adj_mat,
                node_embedding_ph: node_embeddings,
                node_bias_ph: node_bias
            }
            outputs = model.inference(feed_dict=feed_dict)

            for j in range(len(outputs[1])):
                flow_cost = outputs[1][j]
                flows = outputs[2][j]
                flow_proportions = outputs[3][j]
                flow_graph = add_features(graph, demands=node_features[j], flows=flows,
                                          proportions=flow_proportions)

                # Write output graph to Graph XML
                index = i * self.params['batch_size'] + j
                nx.write_gexf(flow_graph, '{0}graph-{1}.gexf'.format(model_path, index))

                if self.params['plot_flows']:
                    plot_flow_graph(flow_graph, flows, '{0}flows-{1}.png'.format(model_path, index))
                    plot_flow_graph(flow_graph, flow_proportions, '{0}flow-prop-{1}.png'.format(model_path, index))