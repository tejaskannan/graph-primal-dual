import osmnx as ox
import networkx as nx
import numpy as np
import scipy.sparse as sp
import matplotlib as plt
from os import path
from os import mkdir
from utils.constants import SMALL_NUMBER
from utils.utils import serialize_dict, deserialize_dict, append_row_to_log
from utils.utils import delete_if_exists
from core.plot import plot_road_graph
from annoy import AnnoyIndex


def save_graph(target, graph_name, distance=1000, is_address=True):

    # Load graph from OSM
    if is_address:
        graph = ox.graph_from_address(address=target, distance=distance, network_type='drive')
    else:
        graph = ox.graph_from_place(query=target, network_type='drive')

    # Project graph
    graph_proj = ox.project_graph(graph)

    folder_path = path.join('graphs', graph_name)
    if not path.exists(folder_path):
        mkdir(folder_path)

    graph_data = graph_proj.graph
    serialize_dict(dictionary=graph_data, file_path=path.join(folder_path, 'graph_data.pkl.gz'))

    # We make the graph strongly connected to ensure that any combination of source / sink
    # constitutes a valid problem
    graph_component = ox.get_largest_component(graph_proj, strongly=True).to_directed()

    # Save pictures of the graph
    plot_road_graph(graph_component, graph_name=graph_name, file_path=path.join(folder_path, 'graph'))

    # Save graph
    ox.save_graphml(graph_component, filename='graph.graphml', folder=folder_path, gephi=True)

    # Save a selection of graph-wide stats.
    stats_file = path.join(folder_path, 'stats.csv')
    delete_if_exists(stats_file)

    n_nodes = graph_component.number_of_nodes()
    n_edges = graph_component.number_of_edges()
    avg_in_deg = np.average([d for _, d in graph_component.in_degree()])
    avg_out_deg = np.average([d for _, d in graph_component.out_degree()])
    diam = nx.diameter(graph_component)

    append_row_to_log(['Number of Nodes', n_nodes], stats_file)
    append_row_to_log(['Number of Edges', n_edges], stats_file)
    append_row_to_log(['Average In Degree', avg_in_deg], stats_file)
    append_row_to_log(['Average Out Degree', avg_out_deg], stats_file)
    append_row_to_log(['Diameter', diam], stats_file)

    cleaned_target = target.replace('\"', '').replace('\'', '').replace(',', '')
    query = '{0}; Dist: {1}'.format(cleaned_target, distance) if is_address else cleaned_target
    append_row_to_log(['Query', query], stats_file)

    return graph_component


def load_graph(graph_name):
    folder_path = path.join('graphs', graph_name)

    graph_data = deserialize_dict(file_path=path.join(folder_path, 'graph_data.pkl.gz'))
    graph = ox.load_graphml(filename='graph.graphml', folder=folder_path)
    # graph = nx.MultiDiGraph(nx.read_graphml(path.join(folder_path, 'graph.graphml'), node_type=int))

    if not nx.is_strongly_connected(graph):
        graph = ox.get_largest_component(graph, strongly=True)

    node_mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
    graph = nx.relabel_nodes(graph, node_mapping)

    # Ensure nodes and edges are ordered consistently
    G = nx.MultiDiGraph()
    for node, data in sorted(graph.nodes(data=True), key=lambda t: t[0]):
        data['demand'] = 0
        G.add_node(node, **data)

    for src, dst, data in sorted(graph.edges(data=True), key=lambda t: (t[0], t[1])):
        # Remove parallel edges and self-loops
        if src == dst or (src in G and dst in G[src]):
            continue

        # Dummy data for compatibility with plotter
        data['zero'] = 0
        G.add_edge(src, dst, key=0, **data)

    G.graph['crs'] = graph_data['crs']
    G.graph['name'] = graph_data['name']
    G = ox.project_graph(G, to_crs=graph_data['crs'])

    return G.to_directed()


def write(dataset, folder, index):
    """
    Serializes the given matrices  as sparse matrices in a set of files. We use a custom function
    here because the scipy.sparse.save_npz function only allows for a single sparse matrix.
    """
    source_data = {str(i): data[0] for i, data in enumerate(dataset)}
    sink_data = {str(i): data[1] for i, data in enumerate(dataset)}

    source_file_path = path.join(folder, 'source-demands-{0}.pkl.gz'.format(index))
    serialize_dict(dictionary=source_data, file_path=source_file_path)

    sink_file_path = path.join(folder, 'sink-demands-{0}.pkl.gz'.format(index))
    serialize_dict(dictionary=sink_data, file_path=sink_file_path)


def read(folder, file_index, sources, sinks, num_nodes):

    def read_dict(folder, name):
        file_path = path.join(folder, name)
        return deserialize_dict(file_path=file_path)

    source_data = read_dict(folder, 'source-demands-{0}.pkl.gz'.format(file_index))
    sink_data = read_dict(folder, 'sink-demands-{0}.pkl.gz'.format(file_index))

    dataset = []
    for i in range(len(source_data)):
        index = str(i)
        source_demands = source_data[index]
        sink_demands = sink_data[index]

        demands = np.zeros(shape=(num_nodes, 1))
        demands[sources, 0] = source_demands
        demands[sinks, 0] = sink_demands

        sp_mat = sp.csr_matrix(demands)
        dataset.append(sp_mat)

    return dataset


def load_embeddings(index_path, embedding_size, num_nodes):
    # Load Annoy index which stores the embedded vectors
    index = AnnoyIndex(embedding_size)
    index.load(index_path)

    embeddings = [index.get_item_vector(i) for i in range(num_nodes)]

    # Unload the index to save memory (loading mmaps the index file)
    index.unload()

    # V x D matrix of embeddings
    return np.array(embeddings)
