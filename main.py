import os
import argparse
import networkx as nx
import numpy as np
import json
import scipy.sparse as sp
from annoy import AnnoyIndex
from utils.utils import load_params, restore_params, create_node_embeddings
from utils.utils import append_row_to_log, create_demands, file_index, find_max_sources_sinks
from utils.utils import create_capacities, delete_if_exists, serialize_dict
from utils.graph_utils import random_walk_neighborhoods, simple_paths
from utils.graph_utils import random_sources_sinks, farthest_nodes, farthest_sink_nodes
from utils.constants import *
from core.load import load_embeddings, write, save_graph, load_graph
from core.plot import plot_road_flow_graph
from model_runners.fixed_baseline import FixedBaseline
from model_runners.optimization_baseline import OptimizationBaseline
from model_runners.flow_model_runner import FlowModelRunner


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Computing Min Cost Flows using Graph Neural Networks.')
    parser.add_argument('--params', type=str, help='Parameters JSON file.')
    parser.add_argument('--train', action='store_true', help='Flag to specify training.')
    parser.add_argument('--generate', action='store_true', help='Flag to specify dataset generation.')
    parser.add_argument('--test', action='store_true', help='Flag to specify testing.')
    parser.add_argument('--slsqp', action='store_true', help='Flag to specify using SLSQP baseline.')
    parser.add_argument('--trust-constr', action='store_true', help='Flag to specify using Trust Constraint baseline.')
    parser.add_argument('--fixed', action='store_true', help='Flag to specify using the Fixed Proportions baseline.')
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
        mcf_solver = FlowModelRunner(params=model_params)
        mcf_solver.train()
    elif args.generate:
        generate(params['generate'])
    elif args.test:
        if args.slsqp:
            model_params['optimizer'] = {
                'use_optimizer': True,
                'optimizer_name': 'slsqp'
            }
        elif args.trust_constr:
            model_params['optimizer'] = {
                'use_optimizer': True,
                'optimizer_name': 'trust_constr'
            }
        mcf_solver = FlowModelRunner(params=model_params)
        mcf_solver.test(args.model)
    elif args.random_walks:
        random_walks(params['generate']['graph_names'][0], params['model']['unique_neighborhoods'])
    elif args.graph_stats:
        graph_stats(params['generate']['graph_names'][0])
    elif args.trust_constr:
        baseline = OptimizationBaseline(params=model_params, optimizer_name='trust_constr')
        baseline.optimize()
    elif args.slsqp:
        baseline = OptimizationBaseline(params=model_params, optimizer_name='slsqp')
        baseline.optimize()
    elif args.fixed:
        baseline = FixedBaseline(params=model_params)
        baseline.test(model_path=None)
    elif args.view_params:
        print(json.dumps(params, indent=2, sort_keys=True))


def generate(params):

    graph_name = params['graph_name']
    graph_folder = os.path.join('graphs', graph_name)

    # Fetch graph for the given query and save it if necessary
    if not os.path.exists(graph_folder):
        target = params['osm_query']
        distance = params.get('distance', 1000)
        is_address = params['is_address']
        save_graph(target=target, graph_name=graph_name, distance=distance, is_address=is_address)

    graph = load_graph(graph_name=graph_name)

    dataset_name = params['dataset_name']
    dataset_folder = os.path.join('datasets', dataset_name)

    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)

    # Save parameters
    serialize_dict(dictionary=params, file_path=os.path.join(dataset_folder, 'params.pkl.gz'))

    # Generate and save sources and sinks
    if params['source_sink_strategy'] == 'random':
        sources, sinks = random_sources_sinks(graph, num_sources=params['num_sources'], num_sinks=params['num_sinks'])
    elif params['source_sink_strategy'] == 'farthest_sink':
        sources, sinks = farthest_sink_nodes(graph, num_sources=params['num_sources'], num_sinks=params['num_sinks'])
    else:
        sources, sinks = farthest_nodes(graph, num_sources=params['num_sources'], num_sinks=params['num_sinks'])

    G = graph.copy()
    for source in sources:
        G.add_node(source, demand=-1)
    for sink in sinks:
        G.add_node(sink, demand=1)
    file_path = os.path.join(dataset_folder, 'graph')
    plot_road_flow_graph(graph=G, field='zero', graph_name=graph_name, file_path=file_path)

    source_sink_dict = {
        'sources': sources,
        'sinks': sinks
    }
    serialize_dict(dictionary=source_sink_dict, file_path=os.path.join(dataset_folder, 'sources_sinks.pkl.gz'))

    # Generate and save paths between sources and sinks
    # paths = simple_paths(graph=graph, sources=sources, sinks=sinks, max_num_paths=params['max_num_paths'])
    # serialize_dict(dictionary=paths, file_path=os.path.join(dataset_folder, 'paths.pkl.gz'))

    demands_thresh = 0.1
    source_index = AnnoyIndex(len(sources))
    sink_index = AnnoyIndex(len(sinks))

    file_paths = ['train', 'valid', 'test']
    samples = [params['train_samples'], params['valid_samples'], params['test_samples']]
    for file_path, num_samples in zip(file_paths, samples):

        # Create folder to put this data series in
        series_folder = os.path.join(dataset_folder, file_path)
        if not os.path.exists(series_folder):
            os.mkdir(series_folder)

        dataset = []
        for i in range(num_samples):
            source_demands, sink_demands = create_demands(sources=sources, sinks=sinks)

            # Ensure training demands not too close to either validation or test demands
            if file_path == 'train':
                source_index.add_item(i, source_demands)
                sink_index.add_item(i, sink_demands)
            else:
                _, source_dist = source_index.get_nns_by_vector(source_demands, n=1, include_distances=True)
                _, sink_dist = source_index.get_nns_by_vector(sink_demands, n=1, include_distances=True)

                retries = 0
                while source_dist[0] < demands_thresh and sink_dist[0] < demands_thresh:
                    source_demands, sink_demands = create_demands(sources=sources, sinks=sinks)
                    _, source_dist = source_index.get_nns_by_vector(source_demands, n=1, include_distances=True)
                    _, sink_dist = source_index.get_nns_by_vector(sink_demands, n=1, include_distances=True)

                    assert retries < 1000, 'Retried too many times.'
                    retries += 1

            dataset.append((source_demands, sink_demands))

            # Periodically write to output files
            if (i+1) % WRITE_THRESHOLD == 0:
                index, _ = file_index(i)
                write(dataset=dataset, folder=series_folder, index=index)
                print('Completed {0}/{1} samples for {2}.'.format(i+1, num_samples, file_path))
                dataset = []

        if len(dataset) > 0:
            index, _ = file_index(i)
            write(dataset=dataset, folder=series_folder, index=index)
        print('Completed {0}.'.format(file_path))

        if file_path == 'train':
            source_index.build(10)
            sink_index.build(10)


if __name__ == '__main__':
    main()
