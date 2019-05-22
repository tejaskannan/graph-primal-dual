import matplotlib as mpl
mpl.use('pgf')
pgf_with_pdflatex = {
    'pgf.texsystem': 'lualatex',
    'pgf.rcfonts': False
}
mpl.rcParams.update(pgf_with_pdflatex)

import matplotlib.pyplot as plt
import osmnx as ox
from matplotlib import cm
from matplotlib import colors
import numpy as np
import networkx as nx
from utils.constants import *


def plot_road_graph(graph, graph_name, file_path):
    fig, ax = ox.plot_graph(graph,
                            node_size=1,
                            node_color='blue',
                            node_edgecolor='blue',
                            node_zorder=3,
                            show=False,
                            save=False,
                            close=False)
    plt.savefig(file_path + '.pdf', bbox='tight')
    plt.savefig(file_path + '.pgf')


def plot_road_flow_graph(graph, field, graph_name, file_path):
    n_nodes = graph.number_of_nodes()

    edge_cmap = cm.get_cmap(name='YlOrBr')
    node_cmap = cm.get_cmap(name='viridis')

    node_sizes = np.full(shape=(n_nodes,), fill_value=1)
    node_colors = ['gray' for _ in range(n_nodes)]

    demands = [v for (node, v) in graph.nodes.data('demand')]
    node_normalizer = colors.Normalize(vmin=np.min(demands), vmax=np.max(demands))

    for i, (node, demand) in enumerate(graph.nodes.data('demand')):
        if demand > 0:
            node_sizes[i] = 20
            node_colors[i] = node_cmap(node_normalizer(demand))
        elif demand < 0:
            node_sizes[i] = 20
            node_colors[i] = node_cmap(node_normalizer(demand))

    values = [v for (src, dst, v) in graph.edges.data(field)]
    edge_normalizer = colors.Normalize(vmin=np.min(values), vmax=np.max(values))

    edge_colors = [edge_cmap(edge_normalizer(x)) for x in values]

    fig, ax = ox.plot_graph(graph,
                            node_size=node_sizes,
                            node_color=node_colors,
                            node_edgecolor=node_colors,
                            node_zorder=3,
                            edge_color=edge_colors,
                            show=False,
                            save=False,
                            close=False)

    edge_scalar_map = cm.ScalarMappable(norm=edge_normalizer, cmap=edge_cmap)
    edge_scalar_map.set_array(values)

    cax = fig.add_axes([0.05, 0.08, 0.4, 0.02])
    edge_cbar = fig.colorbar(edge_scalar_map, cax=cax, orientation='horizontal')
    edge_cbar.set_label('Flow')

    demands_scalar_map = cm.ScalarMappable(norm=node_normalizer, cmap=node_cmap)
    demands_scalar_map.set_array(demands)

    cax = fig.add_axes([0.55, 0.08, 0.4, 0.02])
    demand_cbar = fig.colorbar(demands_scalar_map, cax=cax, orientation='horizontal')
    demand_cbar.set_label('Demand')

    fig.suptitle('Computed Flows for ' + graph_name, fontsize=12)
    plt.savefig(file_path + '.pdf', bbox='tight')
    plt.savefig(file_path + '.pgf')


def plot_costs(costs_lst, out_path):
    cmap = cm.get_cmap(name='viridis')
    indices = np.linspace(start=0.0, stop=1.0, endpoint=True, num=len(costs_lst))

    for i, costs in enumerate(costs_lst):
        x = np.arange(start=0, stop=len(costs), step=1)
        plt.plot(x, costs, color=cmap(indices[i]))
    plt.xlabel('Iteration')
    plt.ylabel('Projected Cost')
    plt.savefig(out_path)
    plt.close()


def plot_flow_graph_adj(graph, use_flow_props, file_path, use_node_weights=True):
    edge_cmap = cm.get_cmap(name='Reds')
    node_cmap = cm.get_cmap(name='viridis')

    agraph = nx.drawing.nx_agraph.to_agraph(graph)
    agraph.node_attr['style'] = 'filled'
    agraph.graph_attr['pad'] = 2.0
    agraph.graph_attr['overlap'] = 'scalexy'
    agraph.graph_attr['sep'] = 1.0

    if use_node_weights:
        weights = [w for _, w in graph.nodes.data('node_weight')]
        min_weight = min(weights)
        max_weight = max(weights)

    for node, data in graph.nodes(data=True):
        n = agraph.get_node(node)

        demand = data['demand']
        if demand > 0.0:
            n.attr['shape'] = 'diamond'
        elif demand < 0.0:
            n.attr['shape'] = 'square'
        n.attr['label'] = str(round(demand, 2))

        if use_node_weights:
            normalized_weight = (data['node_weight'] - min_weight) / (max_weight - min_weight)
            rgb = node_cmap(normalized_weight)[:3]
            n.attr['fillcolor'] = colors.rgb2hex(rgb)
            n.attr['fontcolor'] = font_color(rgb)
        else:
            n.attr['fillcolor'] = '#BAD7E6'

    flow_label = 'flow_proportion' if use_flow_props else 'flow'
    flow_vals = [v for _, _, v in graph.edges.data(flow_label)]

    max_flow_val = max(flow_vals)
    for src, dst, val in graph.edges.data(flow_label):
        e = agraph.get_edge(src, dst)

        e.attr['color'] = colors.rgb2hex(edge_cmap(val / max_flow_val)[:3])
        if abs(val) > SMALL_NUMBER:
            e.attr['label'] = str(round(val, 2))

    agraph.draw(file_path, prog='neato')


# Flows is a |V| x |V| matrix of flow values
def plot_flow_graph(graph, flows, file_path, use_node_weights=True):
    cmap = cm.get_cmap(name='Reds')
    node_cmap = cm.get_cmap(name='viridis')

    agraph = nx.drawing.nx_agraph.to_agraph(graph)
    agraph.node_attr['style'] = 'filled'
    agraph.graph_attr['pad'] = 2.0
    agraph.graph_attr['overlap'] = 'scalexy'
    agraph.graph_attr['sep'] = 1.0

    if use_node_weights:
        node_weights = [w for _, w in graph.nodes('node_weight')]
        min_weight = np.min(node_weights)
        max_weight = np.max(node_weights)

    for node, data in graph.nodes(data=True):
        n = agraph.get_node(node)
        demand = data['demand']
        if demand > 0.0:
            n.attr['shape'] = 'diamond'
        elif demand < 0.0:
            n.attr['shape'] = 'square'
        n.attr['label'] = str(round(demand, 2))

        if use_node_weights:
            normalized_weight = (data['node_weight'] - min_weight) / (max_weight - min_weight)
            rgb = node_cmap(normalized_weight)[:3]
            n.attr['fillcolor'] = colors.rgb2hex(rgb)
            n.attr['fontcolor'] = font_color(rgb)
        else:
            n.attr['fillcolor'] = '#BAD7E6'

    max_flow_val = np.max(flows)
    min_flow_val = min(0.0, np.min(flows))
    for src, dest in graph.edges():
        flow = flows[src, dest]
        e = agraph.get_edge(src, dest)
        e.attr['color'] = colors.rgb2hex(cmap(flow / max_flow_val)[:3])
        if abs(flow) > SMALL_NUMBER:
            e.attr['label'] = str(round(flow, 2))
            e.attr['labeldistance'] = '3'
    agraph.draw(file_path, prog='neato')


def plot_graph(graph, file_path):
    agraph = nx.drawing.nx_agraph.to_agraph(graph)

    agraph.node_attr['style'] = 'filled'
    agraph.node_attr['fillcolor'] = '#BAD7E6'

    agraph.graph_attr['pad'] = 2.0
    agraph.graph_attr['overlap'] = 'scalexy'
    agraph.graph_attr['sep'] = 1.3

    agraph.draw(file_path, prog='neato')


# Flows is a |V| x |V| sparse tensor value
def plot_flow_graph_sparse(graph, flows, file_path, use_node_weights=True):
    cmap = cm.get_cmap(name='Reds')
    node_cmap = cm.get_cmap(name='viridis')

    agraph = nx.drawing.nx_agraph.to_agraph(graph)
    agraph.node_attr['style'] = 'filled'
    agraph.graph_attr['pad'] = 2.0
    agraph.graph_attr['overlap'] = 'scalexy'
    agraph.graph_attr['sep'] = 1.0

    if use_node_weights:
        node_weights = [w for _, w in graph.nodes('node_weight')]
        min_weight = np.min(node_weights)
        max_weight = np.max(node_weights)

    for node, data in graph.nodes(data=True):
        n = agraph.get_node(node)
        demand = data['demand']
        if demand > 0.0:
            n.attr['shape'] = 'diamond'
        elif demand < 0.0:
            n.attr['shape'] = 'square'
        n.attr['label'] = str(round(demand, 2))

        if use_node_weights:
            normalized_weight = (data['node_weight'] - min_weight) / (max_weight - min_weight)
            rgb = node_cmap(normalized_weight)[:3]
            n.attr['fillcolor'] = colors.rgb2hex(rgb)
            n.attr['fontcolor'] = font_color(rgb)
        else:
            n.attr['fillcolor'] = '#BAD7E6'

    for src, dest in graph.edges():
        e = agraph.get_edge(src, dest)
        e.attr['color'] = colors.rgb2hex(cmap(0.0)[:3])

    max_flow_val = np.max(flows.values)
    min_flow_val = min(0.0, np.min(flows.values))
    for edge, val in zip(flows.indices, flows.values):
        if not graph.has_edge(*edge):
            continue

        e = agraph.get_edge(edge[0], edge[1])
        e.attr['color'] = colors.rgb2hex(cmap(val / max_flow_val)[:3])
        if abs(val) > SMALL_NUMBER:
            e.attr['label'] = str(round(val, 2))
            e.attr['labeldistance'] = '3'
    agraph.draw(file_path, prog='neato')


def plot_weights(weight_matrix, file_path, num_nodes, num_samples=-1):
    if weight_matrix.ndim == 3:
        weight_matrix = weight_matrix[0]

    # Remove dummy nodes used to pad matrices
    weight_matrix = weight_matrix[0:num_nodes,:]

    # Default to using all elements
    num_samples = weight_matrix.shape[0] if num_samples == -1 else num_samples

    step = int(weight_matrix.shape[0] / num_samples) + 1
    indices = list(range(0, weight_matrix.shape[0], step))
    samples = weight_matrix[indices]

    fig, ax = plt.subplots()

    im = ax.imshow(samples, aspect='auto')

    # Labels
    ax.set_yticks(np.arange(len(samples)))
    ax.set_yticklabels(indices)

    ax.set_xticks(np.arange(weight_matrix.shape[1]))
    ax.set_xticklabels(np.arange(weight_matrix.shape[1]))

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Axes Labels
    ax.set_xlabel('Neighborhood Layer')
    ax.set_ylabel('Node Index')
    ax.set_title('Node-Specific Attention Weights for each Neighborhood Layer')

    plt.savefig(file_path)
    plt.close(fig)


def font_color(background_rgb):
    r, g, b = background_rgb

    luma = 255 * (0.2126 * r + 0.7152 * g + 0.0722 * b)

    if luma < 128:
        return '#FFFFFF'
    return '#000000'
