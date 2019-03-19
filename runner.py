import argparse
import networkx as nx
import tensorflow as tf
import numpy as np
from utils import load_params, create_node_bias, add_demands
from utils import sparse_matrix_to_tensor, create_batch
from load import load_to_networkx
from graph_model import MinCostFlowModel
from plot import plot_flow_graph
from constants import *


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Computing Min Cost Flows using Graph Neural Networks.')
    parser.add_argument('--params', type=str, help='Parameters JSON file.')
    args = parser.parse_args()

    # Fetch parameters
    params = load_params(args.params)
    
    # Load graph
    graph = load_to_networkx(params['graph_path'])

    # Create tensors for global graph properties
    adj_mat = nx.adjacency_matrix(graph).todense()
    node_bias = create_node_bias(graph)

    num_nodes = graph.number_of_nodes()
    num_input_features = 1

    model = MinCostFlowModel(name='min-cost-flow', params=params)

    # Create placeholders
    with model._sess.graph.as_default():
        node_ph = tf.placeholder(dtype=tf.float32,
                                 shape=[None, num_nodes, num_input_features],
                                 name='node-ph')
        adj_ph = tf.placeholder(dtype=tf.float32,
                                shape=[num_nodes, num_nodes],
                                name='adj-ph')
        node_bias_ph = tf.placeholder(dtype=tf.float32,
                                      shape=[num_nodes, num_nodes],
                                      name='node-bias-ph')
        demands_ph = tf.placeholder(dtype=tf.float32,
                                    shape=[None, num_nodes, 1])

    # Create model
    model.build(inputs=node_ph,
                bias=node_bias,
                adj=adj_ph,
                demands=demands_ph,
                num_input_features=num_input_features,
                num_output_features=num_nodes,
                cost_fn=tf.square,
                is_primal=True)
    model.init()

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
            node_features, demands = create_batch(graph=graph,
                                                  batch_size=params['batch_size'],
                                                  min_max_sources=params['min_max_sources'],
                                                  min_max_sinks=params['min_max_sinks'])

            feed_dict = {
                node_ph: node_features,
                adj_ph: adj_mat,
                node_bias_ph: node_bias,
                demands_ph: demands
            }
            avg_batch_loss = model.run_train_step(feed_dict=feed_dict)

            batch_num = int(i / params['batch_size']) + 1
            print('Average training loss for batch {0}/{1}: {2}'.format(batch_num, num_training_batches, avg_batch_loss))

            train_losses.append(avg_batch_loss)

        # Validation Batches
        valid_costs = []
        for i in range(0, params['valid_samples'], params['batch_size']):
             # Create random validation points
            node_features, demands = create_batch(graph=graph,
                                                  batch_size=params['batch_size'],
                                                  min_max_sources=params['min_max_sources'],
                                                  min_max_sinks=params['min_max_sinks'])
            feed_dict = {
                node_ph: node_features,
                adj_ph: adj_mat,
                node_bias_ph: node_bias,
                demands_ph: demands
            }
            costs = model.inference(feed_dict=feed_dict)

            valid_costs.append(costs[0])

        print(LINE)

        avg_train_loss = np.average(train_losses)
        print('Average training loss: {0}'.format(avg_train_loss))

        avg_valid_cost = np.average(valid_costs)
        print('Average validation cost: {0}'.format(avg_valid_cost))

    # Create random test point
    node_features, demands = create_batch(graph=graph,
                                          batch_size=1,
                                          min_max_sources=params['min_max_sources'],
                                          min_max_sinks=params['min_max_sinks'])
    feed_dict = {
        node_ph: node_features,
        adj_ph: adj_mat,
        node_bias_ph: node_bias,
        demands_ph: demands
    }
    outputs = model.inference(feed_dict=feed_dict)
    flows = outputs[1][0]
    demand_graph = add_demands(graph, demands[0])
    plot_flow_graph(demand_graph, flows, 'test.png')


if __name__ == '__main__':
    main()
