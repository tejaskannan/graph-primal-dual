import matplotlib as mpl
mpl.use('pgf')
pgf_with_pdflatex = {
    'pgf.texsystem': 'lualatex',
    'pgf.rcfonts': False
}
mpl.rcParams.update(pgf_with_pdflatex)
import os
import os.path
import matplotlib.pyplot as plt
import osmnx as ox
from matplotlib import cm
from matplotlib import colors
from matplotlib.collections import LineCollection
from adjustText import adjust_text
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


def plot_road_flow_graph(graph, field, graph_name, file_path, label_edges=False):
    n_nodes = graph.number_of_nodes()

    edge_cmap = cm.get_cmap(name='coolwarm')
    node_cmap = cm.get_cmap(name='viridis')

    node_sizes = np.full(shape=(n_nodes,), fill_value=1)
    node_colors = ['gray' for _ in range(n_nodes)]

    demands = [v for (node, v) in graph.nodes.data('demand')]
    node_normalizer = colors.Normalize(vmin=np.min(demands), vmax=np.max(demands))

    for i, (node, demand) in enumerate(graph.nodes.data('demand')):
        if demand > 0:
            node_sizes[i] = 4
            node_colors[i] = node_cmap(node_normalizer(demand))
        elif demand < 0:
            node_sizes[i] = 4
            node_colors[i] = node_cmap(node_normalizer(demand))

    values = [v for _, _, v in graph.edges.data(field)]
    # values = list(nx.get_edge_attributes(graph, name=field).values())
    edge_normalizer = colors.Normalize(vmin=0, vmax=0.5)

    edge_colors = [edge_cmap(edge_normalizer(x)) for x in values]

    fig, ax, lines = plot_directed(graph,
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   node_edgecolor=node_colors,
                                   node_zorder=3,
                                   edge_cmap=edge_cmap,
                                   edge_linewidth=0.7,
                                   field=field)

    edge_scalar_map = cm.ScalarMappable(norm=edge_normalizer, cmap=edge_cmap)
    edge_scalar_map.set_array(values)

    cax = fig.add_axes([0.05, 0.08, 0.4, 0.02])
    edge_cbar = fig.colorbar(edge_scalar_map, cax=cax, orientation='horizontal')
    # edge_cbar.set_label(field.replace('_', ' ').capitalize())
    edge_cbar.set_label('Proportion of flow sent along reverse edge')

    demands_scalar_map = cm.ScalarMappable(norm=node_normalizer, cmap=node_cmap)
    demands_scalar_map.set_array(demands)

    cax = fig.add_axes([0.55, 0.08, 0.4, 0.02])
    demand_cbar = fig.colorbar(demands_scalar_map, cax=cax, orientation='horizontal')
    demand_cbar.set_label('Demand')

    # Annotate nodes with demand
    label_xs, label_ys = [], []
    demands_text = []
    legend_text = ['Demands']
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        if abs(data['demand']) > SMALL_NUMBER:
            dem = round(data['demand'], 2)
            text = ax.text(s=str(node), x=data['x'], y=data['y'], fontsize=8)
            demands_text.append(text)
            legend_text.append('{0}: {1}'.format(node, dem))

        x0, x1 = data['x'] - node_sizes[i], data['x'] + node_sizes[i]
        y0, y1 = data['y'] - node_sizes[i], data['y'] + node_sizes[i]
        label_xs += [data['x'], x0, x0, x1, x1]
        label_ys += [data['y'], y0, y1, y0, y1]

    # Annotate edges with the given field value
    if label_edges:
        xs = nx.get_node_attributes(graph, 'x')
        ys = nx.get_node_attributes(graph, 'y')
        for src, dst, data in graph.edges(keys=False, data=True):
            if abs(data[field]) > SMALL_NUMBER:
                val = round(data[field], 2)

                if 'geometry' in data:
                    x_coo, y_coo = data['geometry'].xy
                    x = x_coo[int(len(x_coo) / 2.0)]
                    y = y_coo[int(len(y_coo) / 2.0)]
                else:
                    x = 0.5 * (xs[src] + xs[dst])
                    y = 0.5 * (ys[src] + ys[dst])

                demands_text.append(ax.text(s=str(val), x=x, y=y, fontsize=6, color='gray'))

    samples = 5
    for line in lines:
        for p0, p1 in zip(line[:-1], line[1:]):

            # Interpolate each segment to repel labels from the line
            x0, y0 = p0
            x1, y1 = p1

            if abs(x1 - x0) < SMALL_NUMBER:
                xs = np.full(shape=samples, fill_value=x0)
                ys = np.linspace(start=y0, stop=y1, num=samples, endpoint=False)
            elif abs(y1 - y0) < SMALL_NUMBER:
                xs = np.linspace(start=x0, stop=x1, num=samples, endpoint=False)
                ys = np.full(shape=samples, fill_value=y0)
            else:
                m = (y1 - y0) / (x1 - x0)
                xs = np.linspace(start=x0, stop=x1, num=samples, endpoint=False)
                ys = m * (xs - x0) + y0

            label_xs += list(xs)
            label_ys += list(ys)

    adjust_text(demands_text, x=label_xs, y=label_ys, ax=ax, precision=0.1,
                expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                force_text=(0.01, 0.25), force_points=(0.01, 0.25))
    fig.suptitle('Computed Flows for ' + graph_name, fontsize=14)

    legend_str = '\n'.join(legend_text)
    bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    cax = fig.add_axes([0.02, 0.88, 0.1, 0.1])
    cax.text(x=0.0, y=0.0, s=legend_str, verticalalignment='top', fontsize=10, bbox=bbox_props)

    # Turn off axis labels
    xaxis, yaxis = cax.get_xaxis(), cax.get_yaxis()
    cax.axis('off')
    xaxis.set_visible(False)
    yaxis.set_visible(False)

    plt.savefig(file_path + '.pdf', bbox='tight')

    pgf_folder = file_path + '-pgf'
    if not os.path.exists(pgf_folder):
        os.mkdir(pgf_folder)

    plt.savefig(os.path.join(pgf_folder, 'graph.pgf'))
    plt.close()


def plot_directed(graph, node_size, node_color, node_edgecolor, edge_linewidth, edge_cmap, node_zorder, field):
    """
    Version of OSMNX plot_graph function (https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py#L284)
    which adjusts edges to make directed edges visible.
    """

    node_Xs = [float(x) for _, x in graph.nodes(data='x')]
    node_Ys = [float(y) for _, y in graph.nodes(data='y')]

    # get north, south, east, west values either from bbox parameter or from the
    # spatial extent of the edges' geometries
    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    west, south, east, north = edges.total_bounds

    # if caller did not pass in a fig_width, calculate it proportionately from
    # the fig_height and bounding box aspect ratio
    fig_height = 6
    bbox_aspect_ratio = (north - south) / (east - west)
    fig_width = fig_height / bbox_aspect_ratio

    # create the figure and axis
    bgcolor = 'w'
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
    ax.set_facecolor(bgcolor)

    lines = []
    edge_linewidth = []
    edge_color = []
    i = 0
    for i, (u, v, data) in enumerate(graph.edges(keys=False, data=True)):

        # Skip back edges which have larger flow
        forward_cost = graph[u][v][0][field] if graph[u][v][0][field] > 1e-5 else 1e-5
        if u in graph[v]:
            back_cost = graph[v][u][0][field] if graph[v][u][0][field] > 1e-5 else 1e-5

            if forward_cost < back_cost:
                continue
            elif back_cost >= forward_cost and abs(forward_cost - back_cost) < SMALL_NUMBER and v < u:
                continue

            edge_linewidth.append((forward_cost + back_cost) * 4)

            # Prevent forward-backward comparison if the values are very small
            if forward_cost < 1e-4:
                frac = 0.0
            else:
                frac = (back_cost) / (forward_cost + back_cost)
        else:
            edge_linewidth.append(forward_cost * 4)
            frac = 0.0

        edge_color.append(edge_cmap(frac * 2))
        i += 1

        if 'geometry' in data:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            xs, ys = data['geometry'].xy
            # xs, ys = shift_points(xs, ys, edge_linewidth)
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
            x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']

            # adjust edges to view in both directions
            # xs, ys = shift_points([x1, x2], [y1, y2], edge_linewidth)

        lines.append(list(zip(xs, ys)))

    # edge_linewidth = np.exp(edge_linewidth) - 0.9
    lc = LineCollection(lines, colors=edge_color, linewidths=edge_linewidth, alpha=1.0, zorder=2)
    ax.add_collection(lc)

    # scatter plot the nodes
    ax.scatter(node_Xs, node_Ys, s=node_size, c=node_color, alpha=1.0, edgecolor=node_edgecolor, zorder=node_zorder)

    # # Add arrows to each line
    arrow_style = '->, head_width=0.2, head_length=0.1'
    for i, (line, color) in enumerate(zip(lines, edge_color)):

        mid = int(len(line) / 2)
        start, end = line[mid-1], line[mid]

        ax.annotate('', xy=end, xycoords='data', xytext=start, textcoords='data', zorder=1,
                    arrowprops=dict(arrowstyle=arrow_style, lw=min(edge_linewidth[i], 1.5), connectionstyle='arc3', color=color))

    # set the extent of the figure
    margin = 0.02
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))

    # configure axis appearance
    xaxis = ax.get_xaxis()
    yaxis = ax.get_yaxis()

    xaxis.get_major_formatter().set_useOffset(False)
    yaxis.get_major_formatter().set_useOffset(False)

    # Turn off axis labels
    ax.axis('off')
    ax.margins(0)
    ax.tick_params(which='both', direction='in')
    xaxis.set_visible(False)
    yaxis.set_visible(False)
    fig.canvas.draw()

    # make everything square
    ax.set_aspect('equal')
    fig.canvas.draw()

    return fig, ax, lines


def shift_points(xs, ys, edge_linewidth):
    factor = 4
    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[-1], ys[-1]

    if len(xs) == 2:
        x_mid, y_mid = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        xs.insert(1, x_mid)
        ys.insert(1, y_mid)

    for i in range(1, len(xs) - 1):
        ys[i] += factor * edge_linewidth if x1 > x0 else -factor * edge_linewidth
        xs[i] += factor * edge_linewidth if y1 < y0 else -factor * edge_linewidth
    return xs, ys


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


def plot_weights(weight_matrix, file_path, num_nodes, num_samples=-1):
    if weight_matrix.ndim == 3:
        weight_matrix = weight_matrix[0]

    # Remove dummy nodes used to pad matrices
    weight_matrix = weight_matrix[0:num_nodes, :]

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

    if not file_path.endswith('.png'):
        file_path += '.png'

    plt.savefig(file_path)
    plt.close(fig)


def font_color(background_rgb):
    r, g, b = background_rgb

    luma = 255 * (0.2126 * r + 0.7152 * g + 0.0722 * b)

    if luma < 128:
        return '#FFFFFF'
    return '#000000'
