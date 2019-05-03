import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
import networkx as nx
from utils.constants import *


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


def plot_weights(weight_matrix, file_path, num_samples=-1):
    if weight_matrix.ndim == 3:
        weight_matrix = weight_matrix[0]

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

    luma = 255 * (0.2126 * r  + 0.7152 * g + 0.0722 * b)

    if luma < 128:
        return '#FFFFFF'
    return '#000000'
