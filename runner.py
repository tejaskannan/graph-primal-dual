import argparse
import networkx as nx
import tensorflow as tf
from utils import load_params, create_node_bias
from utils import sparse_matrix_to_tensor, create_demand_tensor
from load import load_to_networkx
from graph_model import MinCostFlowModel


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
    adj_tensor = sparse_matrix_to_tensor(nx.adjacency_matrix(graph))
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
                cost_fn=tf.square)
    model.init()



if __name__ == '__main__':
    main()
