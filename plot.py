import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
import networkx as nx
from constants import *


def plot_costs(costs_lst, out_path):
    cmap = cm.get_cmap(name='viridis')
    indices = np.linspace(start=0.0, stop=1.0, endpoint=True, num=len(costs_lst))

    for i, costs in enumerate(costs_lst):
        x = np.arange(start=0, stop=len(costs), step=1)
        plt.plot(x, costs, color=cmap(indices[i]))
    plt.xlabel('Iteration')
    plt.ylabel('Projected Cost')
    plt.savefig(out_path)


# Flows is a |V| x |V| matrix of flow values
def plot_flow_graph(graph, flows, file_path):
    cmap = cm.get_cmap(name='Reds')

    agraph = nx.drawing.nx_agraph.to_agraph(graph)
    for node, data in graph.nodes(data=True):
        n = agraph.get_node(node)
        demand = data['demand']
        if demand > 0.0:
            n.attr['color'] = 'green'
        elif demand < 0.0:
            n.attr['color'] = 'blue'
        n.attr['label'] = str(round(demand, 2))

    max_flow_val = np.max(flows)
    min_flow_val = min(0.0, np.min(flows))
    for src, dest in graph.edges():
        flow = flows[src, dest]
        e = agraph.get_edge(src, dest)
        e.attr['color'] = colors.rgb2hex(cmap(flow / max_flow_val)[:3])
        if abs(flow) > SMALL_NUMBER:
            e.attr['label'] = str(round(flow, 2))
    agraph.draw(file_path, prog='dot')


# Flows is a |V| x |V| sparse tensor value
def plot_flow_graph_sparse(graph, flows, file_path):
    cmap = cm.get_cmap(name='Reds')

    agraph = nx.drawing.nx_agraph.to_agraph(graph)
    for node, data in graph.nodes(data=True):
        n = agraph.get_node(node)
        demand = data['demand']
        if demand > 0.0:
            n.attr['color'] = 'green'
        elif demand < 0.0:
            n.attr['color'] = 'blue'
        n.attr['label'] = str(round(demand, 2))

    max_flow_val = np.max(flows.values)
    min_flow_val = min(0.0, np.min(flows.values))
    for edge, val in zip(flows.indices, flows.values):
        e = agraph.get_edge(edge[0], edge[1])
        e.attr['color'] = colors.rgb2hex(cmap(val / max_flow_val)[:3])
        if abs(val) > SMALL_NUMBER:
            e.attr['label'] = str(round(val, 2))
    agraph.draw(file_path, prog='dot')
