import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import networkx as nx
from constants import *


def plot_costs(costs, labels, save_folder):
    cmap = cm.get_cmap(name='viridis')
    color_series = np.linspace(start=0.2, stop=0.8, num=len(costs))

    for i, (series, label, color) in enumerate(zip(costs, labels, color_series)):
        x = list(range(len(series)))
        plt.plot(x, series, color=cmap(color), label=label)

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Estimation per Iteration')
    plt.legend()

    plt.savefig(COSTS_FILE_FORMAT.format(save_folder))


# Flows is a |V| x |V| matrix of flow values
def plot_flow_graph(graph, flows, file_path):
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
        e.attr['color'] = _to_hex(min_flow_val, max_flow_val, flow)
        if abs(flow) > SMALL_NUMBER:
            e.attr['label'] = str(round(flow, 2))
    agraph.draw(file_path, prog='dot')


def _to_hex(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = (value - minimum) / (maximum - minimum + 1e-5)
    g = int(230 * (1 - ratio))
    b = int(230 * (1 - ratio))
    return '#{0:02x}{1:02x}{2:02x}'.format(255, g, b)
