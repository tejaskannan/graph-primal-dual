import argparse
import networkx as nx
import tensorflow as tf
import numpy as np
from utils import load_params, create_node_bias, add_features
from utils import append_row_to_log
from utils import sparse_matrix_to_tensor, create_batch
from load import load_to_networkx
from graph_model import MinCostFlowModel
from plot import plot_flow_graph
from cost_functions import tf_cost_functions
from constants import *
from os import mkdir
from os.path import exists
from datetime import datetime


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Computing Min Cost Flows using Graph Neural Networks.')
    parser.add_argument('--params', type=str, help='Parameters JSON file.')
    args = parser.parse_args()

    # Fetch parameters
    params = load_params(args.params)
    
    # Load graph
    graph_path = 'graphs/{0}.tntp'.format(params['graph_name'])
    if not exists(graph_path):
        print('Unknown graph with name {0}.'.format(params['graph_name']))
        return
    graph = load_to_networkx(graph_path)

    # Create tensors for global graph properties
    adj_mat = nx.adjacency_matrix(graph).todense()
    node_bias = create_node_bias(graph)

    # Graph properties
    num_nodes = graph.number_of_nodes()
    num_node_features = 1

    # Initialize model
    model = MinCostFlowModel(params=params, name='min-cost-flow-model')

    # Create model placeholders
    node_ph = model.create_placeholder(dtype=tf.float32,
                                       shape=[None, num_nodes, num_node_features],
                                       name='node-ph',
                                       ph_type='dense')
    adj_ph = model.create_placeholder(dtype=tf.float32,
                                      shape=[num_nodes, num_nodes],
                                      name='adj-ph',
                                      ph_type='dense')
    node_bias_ph = model.create_placeholder(dtype=tf.float32,
                                            shape=[num_nodes, num_nodes],
                                            name='node-bias-ph',
                                            ph_type='dense')
    node_index_ph = model.create_placeholder(dtype=tf.int32,
                                             shape=[num_nodes],
                                             name='node-index-ph',
                                             ph_type='dense')

    # Create model
    model.build(node_input=node_ph,
                node_index=node_index_ph,
                node_bias=node_bias_ph,
                adj=adj_ph,
                num_input_features=num_node_features,
                num_output_features=num_nodes)
    model.init()

    # Create output folder and initialize logging
    timestamp = datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
    output_folder = '{0}/{1}-{2}/'.format(params['output_folder'], params['graph_name'], timestamp)
    mkdir(output_folder)

    log_headers = ['Epoch', 'Avg Train Loss', 'Avg Valid Loss']
    log_path = output_folder + 'log.csv'
    append_row_to_log(log_headers, log_path)

    for epoch in range(params['epochs']):

        print(LINE)
        print('Epoch {0}'.format(epoch))
        print(LINE)

        num_training_batches = int(params['train_samples'] / params['batch_size']) + 1
        num_validation_batches = int(params['valid_samples'] / params['batch_size']) + 1

        # Training Batches
        train_losses = []
        for i in range(0, params['train_samples'], params['batch_size']):

            # Create random training points
            node_features = create_batch(graph=graph,
                                         batch_size=params['batch_size'],
                                         min_max_sources=params['min_max_sources'],
                                         min_max_sinks=params['min_max_sinks'])

            feed_dict = {
                node_ph: node_features,
                adj_ph: adj_mat,
                node_bias_ph: node_bias
            }
            batch_loss = model.run_train_step(feed_dict=feed_dict)
            avg_batch_loss = batch_loss / params['batch_size']

            batch_num = int(i / params['batch_size']) + 1
            print('Average training loss for batch {0}/{1}: {2}'.format(batch_num, num_training_batches, avg_batch_loss))

            train_losses.append(avg_batch_loss)

        # Validation Batches
        valid_losses = []
        for i in range(0, params['valid_samples'], params['batch_size']):
            # Create random validation points
            node_features = create_batch(graph=graph,
                                         batch_size=params['batch_size'],
                                         min_max_sources=params['min_max_sources'],
                                         min_max_sinks=params['min_max_sinks'])

            feed_dict = {
                node_ph: node_features,
                adj_ph: adj_mat,
                node_bias_ph: node_bias
            }
            outputs = model.inference(feed_dict=feed_dict)
            avg_loss = outputs[0] / params['batch_size']
            valid_losses.append(avg_loss)

        print(LINE)

        avg_train_loss = np.average(train_losses)
        print('Average training loss: {0}'.format(avg_train_loss))

        avg_valid_loss = np.average(valid_losses)
        print('Average validation cost: {0}'.format(avg_valid_loss))

        log_row = [epoch, avg_train_loss, avg_valid_loss]
        append_row_to_log(log_row, log_path)

    # Create random test point
    node_features = create_batch(graph=graph,
                                 batch_size=1,
                                 min_max_sources=params['min_max_sources'],
                                 min_max_sinks=params['min_max_sinks'])
    feed_dict = {
        node_ph: node_features,
        adj_ph: adj_mat,
        node_bias_ph: node_bias
    }
    outputs = model.inference(feed_dict=feed_dict)
    flow_cost = outputs[1][0]
    dual_cost = outputs[3][0]

    print(flow_cost)
    print(dual_cost)

    flows = outputs[2][0]
    demand_graph = add_features(graph, demands=node_features[0], flows=flows)

    # Write output graph to Graph XML
    nx.write_gexf(demand_graph, output_folder + 'graph.gexf')

    if params['plot_flows']:
        plot_flow_graph(demand_graph, flows, output_folder + 'flows.png')

    # Save model weights
    model.save(output_folder)


if __name__ == '__main__':
    main()
