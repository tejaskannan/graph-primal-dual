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


# Flows is a |E| x 1 vector of flow values
def plot_flow_graph(graph, flows, file_path):
    agraph = nx.drawing.nx_agraph.to_agraph(graph)
    for node, data in graph.nodes(data=True):
        n = agraph.get_node(node)
        if 'source' in data and data['source'] == True:
            n.attr['color'] = 'green'
        elif 'sink' in data and data['sink'] == True:
            n.attr['color'] = 'blue'
        n.attr['label'] = str(n)

    max_flow_val = np.max(flows)
    for i, (src, dest, capacity) in enumerate(graph.edges.data('capacity')):
        flow = flows[i][0]
        cap = round(capacity[0], 2)

        e = agraph.get_edge(src, dest)
        e.attr['color'] = _to_hex(0.0, max_flow_val, flow)
        e.attr['label'] = '(' + str(cap) + ', ' + str(round(flow, 2)) + ')'
    agraph.draw(file_path, prog='dot')


def _to_hex(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = (value - minimum) / (maximum - minimum + 1e-5)
    g = int(230 * (1 - ratio))
    b = int(230 * (1 - ratio))
    return '#{0:02x}{1:02x}{2:02x}'.format(255, g, b)
