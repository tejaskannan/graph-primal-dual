import numpy as np
import networkx as nx
import tensorflow as tf
from mcf_model import MCFModel
from load import load_to_networkx, load_embeddings
from datetime import datetime
from os import mkdir
from os.path import exists
from utils import create_demands, append_row_to_log
from utils import add_features, create_node_bias
from plot import plot_flow_graph
from constants import BIG_NUMBER, LINE


class MCF:

    def __init__(self, params):
        self.params = params
        self.timestamp = datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        self.output_folder = '{0}/{1}-{2}/'.format(params['output_folder'], params['graph_name'], self.timestamp)
        self.num_node_features = 1

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
                                           shape=[num_nodes, self.num_node_features],
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

        # Variables for early stopping
        convergence_count = 0
        prev_loss = BIG_NUMBER

        for epoch in range(self.params['epochs']):

            print(LINE)
            print('Epoch {0}'.format(epoch))
            print(LINE)

            # Training Batches
            train_losses = []
            for i in range(self.params['train_samples']):
                node_features = create_demands(graph,
                                               self.params['min_max_sources'],
                                               self.params['min_max_sinks'])
                feed_dict = {
                    node_ph: node_features,
                    adj_ph: adj_mat,
                    node_embedding_ph: node_embeddings,
                    node_bias_ph: node_bias
                }
                avg_loss = model.run_train_step(feed_dict=feed_dict)
                train_losses.append(avg_loss)

            # Validation Batches
            valid_losses = []
            for i in range(self.params['valid_samples']):
                node_features = create_demands(graph,
                                               self.params['min_max_sources'],
                                               self.params['min_max_sinks'])
                feed_dict = {
                    node_ph: node_features,
                    adj_ph: adj_mat,
                    node_embedding_ph: node_embeddings,
                    node_bias_ph: node_bias
                }
                outputs = model.inference(feed_dict=feed_dict)
                avg_loss = outputs[0]
                valid_losses.append(avg_loss)
                print(outputs[6])

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

        # Use random test point
        test_point = create_demands(graph,
                                    self.params['min_max_sources'],
                                    self.params['min_max_sinks'])
        feed_dict = {
            node_ph: test_point,
            adj_ph: adj_mat,
            node_embedding_ph: node_embeddings,
            node_bias_ph: node_bias
        }
        outputs = model.inference(feed_dict=feed_dict)
        flow_cost = outputs[1]

        print('Primal Cost: {0}'.format(flow_cost))
        print('Dual Cost: {0}'.format(outputs[4]))

        flows = outputs[2]
        flow_proportions = outputs[3]
        flow_graph = add_features(graph, demands=test_point, flows=flows, proportions=flow_proportions)

        # Write output graph to Graph XML
        nx.write_gexf(flow_graph, self.output_folder + 'graph.gexf')

        if self.params['plot_flows']:
            plot_flow_graph(flow_graph, flows, self.output_folder + 'flows.png')
