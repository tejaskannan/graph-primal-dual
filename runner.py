import argparse
import networkx as nx
import tensorflow as tf
import numpy as np
from utils import load_params, create_node_bias, add_features
from utils import append_row_to_log, create_demands 
from utils import sparse_matrix_to_tensor, create_batches
from load import load_to_networkx, load_embeddings
from load import write_dataset, read_dataset
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
    parser.add_argument('--train', action='store_true', help='Flag to specify training.')
    parser.add_argument('--generate', action='store_true', help='Flag to specify dataset generation.')
    args = parser.parse_args()

    # Fetch parameters
    params = load_params(args.params)
    
    # Load graph
    graph_path = 'graphs/{0}.tntp'.format(params['graph_name'])
    if not exists(graph_path):
        print('Unknown graph with name {0}.'.format(params['graph_name']))
        return
    graph = load_to_networkx(graph_path)

    if args.train:
        train(graph, params)
    elif args.generate:
        generate(graph, params)


def generate(graph, params):

    train_file = 'datasets/{0}_train.txt'.format(params['dataset_name'])
    valid_file = 'datasets/{0}_valid.txt'.format(params['dataset_name'])

    file_paths = [train_file, valid_file]
    samples = [params['train_samples'], params['valid_samples']]
    for file_path, num_samples in zip(file_paths, samples):
        dataset = []
        for _ in range(num_samples):
            d = create_demands(graph=graph,
                               min_max_sources=params['min_max_sources'],
                               min_max_sinks=params['min_max_sinks'])
            dataset.append(d)
            if len(dataset) == WRITE_THRESHOLD:
                write_dataset(dataset, file_path)
                dataset = []

        # Clean up
        if len(dataset) > 0:
            write_dataset(dataset, file_path)


def train(graph, params):
    # Create tensors for global graph properties
    adj_mat = nx.adjacency_matrix(graph).todense()
    node_bias = create_node_bias(graph)

    # Graph properties
    num_nodes = graph.number_of_nodes()
    num_node_features = 1
    embedding_size = params['node_embedding_size']

    # Load pre-trained embeddings
    index_path = 'embeddings/{0}.ann'.format(params['graph_name'])
    node_embeddings = load_embeddings(index_path=index_path,
                                      embedding_size=embedding_size,
                                      num_nodes=num_nodes)

    # Initialize model
    model = MinCostFlowModel(params=params, name='min-cost-flow-model')

    # Model placeholders
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
    node_embedding_ph = model.create_placeholder(dtype=tf.float32,
                                                 shape=[num_nodes, embedding_size],
                                                 name='node-embedding-ph',
                                                 ph_type='dense')

    # Create model
    model.build(node_input=node_ph,
                node_bias=node_bias_ph,
                node_embeddings=node_embedding_ph,
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


    # Load training and validation datasets
    train_file = 'datasets/{0}_train.txt'.format(params['dataset_name'])
    valid_file = 'datasets/{0}_valid.txt'.format(params['dataset_name'])

    train_dataset = read_dataset(demands_path=train_file, num_nodes=num_nodes)
    valid_dataset = read_dataset(demands_path=valid_file, num_nodes=num_nodes)

    # Variables for early stopping
    convergence_count = 0
    prev_loss = BIG_NUMBER

    for epoch in range(params['epochs']):

        # Shuffle datasets for each epoch
        np.random.shuffle(train_dataset)
        np.random.shuffle(valid_dataset)

        print(LINE)
        print('Epoch {0}'.format(epoch))
        print(LINE)

        # Create batches
        train_batches = create_batches(dataset=train_dataset, batch_size=params['batch_size'])
        valid_batches = create_batches(dataset=valid_dataset, batch_size=params['batch_size'])

        num_train_batches = len(train_batches)
        num_valid_batches = len(valid_batches)

        # Training Batches
        train_losses = []
        for i, node_features in enumerate(train_batches):

            feed_dict = {
                node_ph: node_features,
                adj_ph: adj_mat,
                node_bias_ph: node_bias,
                node_embedding_ph: node_embeddings
            }
            avg_loss = model.run_train_step(feed_dict=feed_dict)
            train_losses.append(avg_loss)

            print('Average training loss for batch {0}/{1}: {2}'.format(i+1, num_train_batches, avg_loss))

        print(LINE)

        # Validation Batches
        valid_losses = []
        for i, node_features in enumerate(valid_batches):
            feed_dict = {
                node_ph: node_features,
                adj_ph: adj_mat,
                node_bias_ph: node_bias,
                node_embedding_ph: node_embeddings
            }
            outputs = model.inference(feed_dict=feed_dict)
            avg_loss = outputs[0]
            valid_losses.append(avg_loss)

            print('Average validation loss for batch {0}/{1}: {2}'.format(i+1, num_valid_batches, avg_loss))

        print(LINE)

        avg_train_loss = np.average(train_losses)
        print('Average training loss: {0}'.format(avg_train_loss))

        avg_valid_loss = np.average(valid_losses)
        print('Average validation loss: {0}'.format(avg_valid_loss))

        log_row = [epoch, avg_train_loss, avg_valid_loss]
        append_row_to_log(log_row, log_path)

        if avg_valid_loss < prev_loss:
            print('Saving model...')
            model.save(output_folder)

        # Early Stopping
        if avg_valid_loss > prev_loss or abs(prev_loss - avg_valid_loss) < params['early_stop_threshold']:
            convergence_count += 1
        else:
            convergence_count = 0

        if convergence_count >= params['patience']:
            print('Early Stopping.')
            break

    # Use random test point
    test_point = np.random.randint(low=0, high=len(valid_dataset)+1)
    feed_dict = {
        node_ph: [valid_dataset[test_point]],
        adj_ph: adj_mat,
        node_bias_ph: node_bias,
        node_embedding_ph: node_embeddings
    }
    outputs = model.inference(feed_dict=feed_dict)
    flow_cost = outputs[1][0]
    dual_cost = outputs[3][0]

    print('Primal Cost: {0}'.format(flow_cost))
    print('Dual Cost: {0}'.format(dual_cost))

    flows = outputs[2][0]
    demand_graph = add_features(graph, demands=node_features[0], flows=flows)

    # Write output graph to Graph XML
    nx.write_gexf(demand_graph, output_folder + 'graph.gexf')

    if params['plot_flows']:
        plot_flow_graph(demand_graph, flows, output_folder + 'flows.png')


if __name__ == '__main__':
    main()
