import os
import argparse
import networkx as nx
import numpy as np
import json
import scipy.sparse as sp
from utils.utils import load_params, restore_params, create_node_embeddings
from utils.utils import append_row_to_log, create_demands, file_index
from utils.utils import create_capacities, delete_if_exists, serialize_dict
from utils.graph_utils import random_walk_neighborhoods
from utils.constants import *
from core.load import load_to_networkx, load_embeddings, write_sparse_npz
from core.plot import plot_graph
from model_runners.dense_baseline import DenseBaseline
from model_runners.optimization_baseline_runner import OptimizationBaselineRunner
from model_runners.neighborhood_runner import NeighborhoodRunner
from model_runners.gat_runner import GATRunner
from model_runners.gcn_runner import GCNRunner
from model_runners.uniform_baseline import UniformBaseline
from model_runners.adj_runner import AdjRunner


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
    parser.add_argument('--uniform', action='store_true', help='Flag to specify using the Uniform Weights baseline.')
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

    if args.train:
        mcf_solver = get_model_runner(params=model_params)
        mcf_solver.train()
    elif args.generate:
        generate(params['generate'])
    elif args.test:
        mcf_solver = get_model_runner(params=model_params)
        mcf_solver.test(args.model)
    elif args.random_walks:
        random_walks(params['generate']['graph_names'][0], params['model']['unique_neighborhoods'])
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
    elif args.uniform:
        baseline = UniformBaseline(model_params)
        baseline.eval()
    elif args.view_params:
        print(json.dumps(params, indent=2, sort_keys=True))


def generate(params):
    assert len(params['graph_names']) == len(params['dataset_names'])

    for graph_name, dataset_name in zip(params['graph_names'], params['dataset_names']):
        # Load graph
        graph_path = 'graphs/{0}.tntp'.format(graph_name)
        graph = load_to_networkx(path=graph_path)

        dataset_folder = 'datasets/{0}/'.format(dataset_name)

        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)

        # Save parameters
        serialize_dict(dictionary=params, file_path=dataset_folder + 'params.pkl.gz')

        file_paths = ['train', 'valid', 'test']
        samples = [params['train_samples'], params['valid_samples'], params['test_samples']]
        for file_path, num_samples in zip(file_paths, samples):

            # Create folder to put this data series in
            series_folder = dataset_folder + file_path
            if not os.path.exists(series_folder):
                os.mkdir(series_folder)
            
            # Create new file if there is an existing dataset with this name
            delete_if_exists(file_path)
            
            dataset = []
            for i in range(num_samples):
                d = create_demands(graph=graph,
                                   min_max_sources=params['min_max_sources'],
                                   min_max_sinks=params['min_max_sinks'])

                # cap = create_capacities(graph=graph, demands=d)

                dataset.append(sp.csr_matrix(d))

                if (i+1) % WRITE_THRESHOLD == 0:
                    index, _ = file_index(i)
                    write_sparse_npz(dataset=dataset, folder=series_folder, index=index)
                    print('Completed {0}/{1} samples for {2}.'.format(i+1, num_samples, file_path))
                    dataset = []

            if len(dataset) > 0:
                index, _ = file_index(i)
                write_sparse_npz(dataset=dataset, folder=series_folder, index=index)
            print('Completed {0}.'.format(file_path))


def get_model_runner(params):
    if params['name'] == 'neighborhood':
        return NeighborhoodRunner(params=params)
    elif params['name'] == 'gat':
        return GATRunner(params=params)
    elif params['name'] == 'gcn':
        return GCNRunner(params=params)
    elif params['name'] == 'adj':
        return AdjRunner(params=params)
    raise ValueError('Model with name {0} does not exist.'.format(params['name']))


def random_walks(graph_name, unique_neighborhoods):
    # Load graph
    graph_path = 'graphs/{0}.tntp'.format(graph_name)
    graph = load_to_networkx(path=graph_path)

    adj = nx.adjacency_matrix(graph)
    neighborhoods = random_walk_neighborhoods(adj, k=20, unique_neighborhoods=unique_neighborhoods)

    total = graph.number_of_nodes()**2

    limit = int(unique_neighborhoods)
    count = 0

    print('Graph: {0}'.format(graph_name))
    print('Number of Entries: {0}'.format(total))
    for i, mat in enumerate(neighborhoods):
        nonzero = mat.count_nonzero()
        frac = (total - nonzero) / total
        print('Frac of zero entries for walks of length {0}: {1}'.format(i, frac))

        count += 1

        if frac == limit:
            break

    print(LINE)

    max_degrees = [np.max(mat.sum(axis=-1)) for mat in neighborhoods]        
    for i in range(count):
        print('Maximum out degree at neighborhood level {0}: {1}'.format(i, max_degrees[i]))


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
    if os.path.exists(stats_path):
        os.remove(stats_path)

    append_row_to_log(['Graph Name', graph_name], stats_path)
    append_row_to_log(['Num Nodes', graph.number_of_nodes()], stats_path)
    append_row_to_log(['Num (directed) edges', graph.number_of_edges()], stats_path)
    append_row_to_log(['Avg Out Degree', avg_out_deg], stats_path)
    append_row_to_log(['Avg In Degree', avg_in_deg], stats_path)
    append_row_to_log(['Avg Path Length', avg_path_length], stats_path)
    append_row_to_log(['Diameter', diameter], stats_path)
    append_row_to_log(['Strongly Connected Components', strong_conn_comp], stats_path)

    # Save graph visualization using GraphViz
    # plot_graph(graph, file_path='graphs/{0}.png'.format(graph_name))

    # Save graph XML
    nx.write_gexf(graph, path='graphs/{0}.gexf'.format(graph_name))


if __name__ == '__main__':
    main()
