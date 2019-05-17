import networkx as nx
import numpy as np
from scipy import optimize

SMALL_NUMBER = 1e-7


def normalize(arr):
    return arr / (np.sum(arr) + 1e-7)


def path_flow(sinks, sources, proportions, demands):
    """
    A: adjacency matrix
    sinks: list of sinks in decreasing order of demand. Length m.
    source: list of sources. Length n.
    proportions: n x m x k matrix of proportions to send along each path from source to sink.
    demands: list of V demands (for each vertex)
    """

    rem_demands = demands.copy()

    sink_flows = []

    for sink_index in range(len(sinks)):
        sink = sinks[sink_index]
        
        sink_demand = demands[sink]

        flows = np.zeros(shape=(len(sources),))

        max_iter = 10
        index = 0
        while sink_demand > 1e-7 and index < max_iter:
            flow_props = np.sum(proportions[sink_index,:,:], axis=-1)

            for i, s in enumerate(sources):
                if abs(rem_demands[s]) < 1e-7:
                    flow_props[i] = 0.0
            flow_props = normalize(flow_props)

            for i, s in enumerate(sources):
                added_flow = min(-rem_demands[s], flow_props[i] * sink_demand)

                flows[i] += added_flow

                # reduce source and sink demand by the added flow
                rem_demands[s] += added_flow
                sink_demand -= added_flow

                if (sink_demand < 0):
                    print(sink_demand)

                # re-normalize
                flow_props[i] = 0.0
                flow_props = normalize(flow_props)

                if abs(sink_demand) <= 1e-7:
                    break

            index += 1


        print('FLOWS')
        print(flows)

        demands = rem_demands
        sink_flows.append(flows)

    path_flows = proportions * np.expand_dims(sink_flows, axis=-1)
    return path_flows

def softmax(matrix, axis):
    mat_exp = np.exp(matrix)
    return mat_exp / np.sum(mat_exp, axis=axis, keepdims=True)


def newton_solver(source_demands, sink_demands, proportions):
    n, m = source_demands.shape[0], sink_demands.shape[0]

    # Source demands mapped to sink demands via proportions.
    X = np.kron(np.eye(m), np.reshape(source_demands, [1, -1]))

    y_vec = np.zeros((1,n))
    y_vec[0][0] = 1
    y_vec = np.tile(y_vec, reps=(1,m))

    rows = []
    for _ in range(n):
        rows.append(y_vec)
        y_vec = np.roll(y_vec, shift=1, axis=-1)

    # Outgoing flows for each source
    Y = np.vstack(rows)

    A = np.vstack([X, Y])

    sink_col = np.reshape(sink_demands, [-1, 1])
    source_col = np.reshape(np.ones_like(source_demands), [-1, 1])

    b = np.vstack([sink_col, source_col])
    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    v_prev = v + 10

    p = np.reshape(proportions, [-1, 1])

    d = m + n + m * n
    split = m * n
    mat = np.zeros(shape=(d, d))
    I = np.eye(split)
    zero = np.zeros(shape=(m + n, m + n))

    lhs = np.vstack([I, A])
    rhs = np.vstack([A.T, zero])
    mat = np.hstack([lhs, rhs])

    print(A)

    pad = np.zeros(shape=(d - m*n, 1))

    max_iter = 1000
    step_size = 1
    beta = 0.9
    for i in range(max_iter):
        target = np.vstack([p - v, pad])

        sol, _, _, _ = np.linalg.lstsq(mat, target, rcond=None)
        delta_v = sol[:m*n]

        v_prev = v
        v = v_prev + step_size * delta_v
        step_size *= beta
        i += 1

        if np.linalg.norm(v - v_prev) < SMALL_NUMBER:
            print('Converged in {0} steps.'.format(i))
            break

    return A.dot(v), v, p


n_nodes = 7
max_path_len = 3

num_paths = 2

G = nx.DiGraph()
G.add_nodes_from(list(range(n_nodes)))

edges = [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (3, 6), (4, 5), (4, 6)]
G.add_edges_from(edges)

sources = [0, 1, 2]
sinks = [5, 6]

demands = np.zeros(shape=(n_nodes,))
demands[0] = -0.5
demands[1] = -0.3
demands[2] = -0.2
demands[5] = 0.7
demands[6] = 0.3

proportions = softmax(np.random.uniform(size=(len(sinks), len(sources))), axis=0)
# print(proportions)
#print(path_flow(sinks=sinks, sources=sources, proportions=proportions, demands=demands))
# paths = np.zeros(shape=(len(sources), len(sinks), max_path_len))

print(newton_solver(source_demands=-demands[0:3],
                    sink_demands=demands[5:],
                    proportions=proportions))
