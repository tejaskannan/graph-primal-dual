import argparse
import networkx as nx
import numpy as np
import json
from utils import load_params, restore_params, create_node_embeddings
from utils import append_row_to_log, create_demands, random_walk_neighborhoods
from load import load_to_networkx, load_embeddings
from load import write_dataset
from constants import *
from sparse_mcf import SparseMCF
from neighborhood import NeighborhoodMCF
from dense_baseline import DenseBaseline
from optimization_baseline_runner import OptimizationBaselineRunner
from mcf import MCF
from os import remove
from os.path import exists
from plot import plot_graph


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Computing Min Cost Flows using Graph Neural Networks.')
    parser.add_argument('--params', type=str, help='Parameters JSON file.')
    parser.add_argument('--train', action='store_true', help='Flag to specify training.')
    parser.add_argument('--generate', action='store_true', help='Flag to specify dataset generation.')
    parser.add_argument('--test', action='store_true', help='Flag to specify testing.')
    parser.add_argument('--random-walks', action='store_true')
    parser.add_argument('--dense', action='store_true', help='Flag to specify using dense baseline.')
    parser.add_argument('--slsqp', action='store_true', help='Flag to specify using SLSQP baseline.')
    parser.add_argument('--trust-constr', action='store_true', help='Flag to specify using Trust Constraint baseline.')
    parser.add_argument('--view-params', action='store_true', help='Flag to specify viewing model parameters.')
    parser.add_argument('--graph-stats', action='store_true')
    parser.add_argument('--model', type=str, help='Path to trained model.')
    args = parser.parse_args()

    # Fetch parameters
    if args.params is not None:
        params = load_params(args.params)
    else:
        # Load parameters used to create the given model
        params = restore_params(args.model)

    model_params = params['model'] if 'model' in params else params
    mcf_solver = NeighborhoodMCF(params=model_params)

    if args.train:
        mcf_solver.train()
    elif args.generate:
        generate(params['generate'])
    elif args.test:
        mcf_solver.test(args.model)
    elif args.random_walks:
        random_walks(params['generate']['graph_names'][0])
    elif args.graph_stats:
        graph_stats(params['generate']['graph_names'][0])
    elif args.dense:
        baseline = DenseBaseline(params=model_params)
        baseline.compute_baseline()
    elif args.trust_constr:
        baseline = OptimizationBaselineRunner(params=model_params, optimizer_name='trust_constr')
        baseline.optimize()
    elif args.slsqp:
        baseline = OptimizationBaselineRunner(params=model_params, optimizer_name='slsqp')
        baseline.optimize()
    elif args.view_params:
        print(json.dumps(params, indent=2, sort_keys=True))


def generate(params):
    assert len(params['graph_names']) == len(params['dataset_names'])

    for graph_name, dataset_name in zip(params['graph_names'], params['dataset_names']):
        # Load graph
        graph_path = 'graphs/{0}.tntp'.format(graph_name)
        graph = load_to_networkx(path=graph_path)

        train_file = 'datasets/{0}_train.txt'.format(dataset_name)
        valid_file = 'datasets/{0}_valid.txt'.format(dataset_name)
        test_file = 'datasets/{0}_test.txt'.format(dataset_name)

        file_paths = [train_file, valid_file, test_file]
        samples = [params['train_samples'], params['valid_samples'], params['test_samples']]
        for file_path, num_samples in zip(file_paths, samples):
            dataset = []
            for i in range(num_samples):
                d = create_demands(graph=graph,
                                   min_max_sources=params['min_max_sources'],
                                   min_max_sinks=params['min_max_sinks'])
                dataset.append(d)
                if len(dataset) == WRITE_THRESHOLD:
                    write_dataset(dataset, file_path)
                    print('Wrote {0} samples to {1}'.format(i+1, file_path))
                    dataset = []

            # Clean up
            if len(dataset) > 0:
                write_dataset(dataset, file_path)

            print('Completed {0}.'.format(file_path))


def random_walks(graph_name):
    # Load graph
    graph_path = 'graphs/{0}.tntp'.format(graph_name)
    graph = load_to_networkx(path=graph_path)

    adj = nx.adjacency_matrix(graph).todense()
    mat = np.eye(graph.number_of_nodes())
    total = graph.number_of_nodes()**2

    print('Graph: {0}'.format(graph_name))
    print('Number of Entries: {0}'.format(total))
    for i in range(20):
        nonzero = np.count_nonzero(mat)
        frac = (total - nonzero) / total
        print('Frac of zero entries for walks of length {0}: {1}'.format(i, frac))

        if frac == 0.0:
            break

        mat = mat.dot(adj)


def graph_stats(graph_name):
    # Load graph
    graph_path = 'graphs/{0}.tntp'.format(graph_name)
    graph = load_to_networkx(path=graph_path)

    # Compute average out degree
    total_out_deg = sum(deg for _, deg in graph.out_degree())
    total_in_deg = sum(deg for _, deg in graph.in_degree())
    avg_out_deg = total_out_deg / float(graph.number_of_nodes())
    avg_in_deg = total_in_deg / float(graph.number_of_nodes())

    # Compute average shortest path length
    avg_path_length = nx.average_shortest_path_length(graph)

    # Strongly connected components
    strong_conn_comp = nx.number_strongly_connected_components(graph)

    # Graph diameter
    diameter = nx.diameter(graph)

    # Write stats to output file
    stats_path = 'graphs/{0}_stats.csv'.format(graph_name)
    if exists(stats_path):
        remove(stats_path)

    append_row_to_log(['Graph Name', graph_name], stats_path)
    append_row_to_log(['Num Nodes', graph.number_of_nodes()], stats_path)
    append_row_to_log(['Num (directed) edges', graph.number_of_edges()], stats_path)
    append_row_to_log(['Avg Out Degree', avg_out_deg], stats_path)
    append_row_to_log(['Avg In Degree', avg_in_deg], stats_path)
    append_row_to_log(['Avg Path Length', avg_path_length], stats_path)
    append_row_to_log(['Diameter', diameter], stats_path)
    append_row_to_log(['Strongly Connected Components', strong_conn_comp], stats_path)

    # Save graph visualization using GraphGiz
    plot_graph(graph, file_path='graphs/{0}.png'.format(graph_name))

    # Save graph XML
    nx.write_gexf(graph, path='graphs/{0}.gexf'.format(graph_name))


if __name__ == '__main__':
    main()
