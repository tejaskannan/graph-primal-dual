import tensorflow as tf
import numpy as np
import networkx as nx
from core.load import load_to_networkx
from utils.tf_utils import masked_gather, weighted_sum, rolling_sum, gathered_sum
from utils.flow_utils import destination_attn
from utils.graph_utils import pad_adj_list
from core.layers import AdjGAT, DirectionalGAT
from core.dataset import GraphData


def random_walks(adj_lst, num_paths, k):
    rand_neighbors = tf.transpose(tf.random_shuffle(tf.transpose(adj_lst, perm=[0, 2, 1])), perm=[0, 2, 1])
    
    curr_sample = rand_neighbors[:,:,0:num_paths]

    samples_arr = tf.TensorArray(dtype=tf.int32, size=k, name='path-arr')
    samples_arr = samples_arr.write(0, curr_sample)

    # 2D matrix of indices
    dim0_indices = tf.expand_dims(tf.range(0, tf.shape(rand_neighbors)[0]), axis=-1)

    times = num_paths * tf.shape(rand_neighbors)[1]
    dim0_indices = tf.reshape(tf.tile(dim0_indices, multiples=(1, times)), [-1, 1])

    def body(i, curr_sample, samples_arr):
        rand_neighbors = tf.transpose(tf.random_shuffle(tf.transpose(adj_lst, perm=[0, 2, 1])), perm=[0, 2, 1])

        indices = tf.concat([dim0_indices, tf.reshape(curr_sample, [-1, 1])], axis=-1)
        
        neighbors = tf.gather_nd(rand_neighbors, indices)

        newshape = [tf.shape(curr_sample)[0], tf.shape(curr_sample)[1], tf.shape(curr_sample)[2], -1]
        neighbors = tf.reshape(neighbors, newshape)
        
        neighbor_samples = neighbors[:,:,:,0]
        
        # append to running list of samples
        samples_arr = samples_arr.write(i+1, neighbor_samples)

        return [i+1, neighbor_samples, samples_arr]

    i = tf.constant(0)
    _, c, samples_arr = tf.while_loop(cond=lambda i, c, s: i < k-1,
                                      body=body,
                                      loop_vars=[i, curr_sample, samples_arr],
                                      parallel_iterations=1,
                                      maximum_iterations=k-1,
                                      name='path-generation')

    paths = samples_arr.stack() # K x B x V x N
    paths_tr = tf.transpose(paths, perm=[1, 3, 2, 0]) # B x N x V x K

    return paths_tr

# Extracts features nodes on each path and aggregates via sum
def get_node_features(paths, node_features):
    dim0_indices = tf.expand_dims(tf.range(0, tf.shape(node_features)[0]), axis=-1)

    times = tf.shape(paths)[1] * tf.shape(paths)[2] * tf.shape(paths)[3]
    dim0_indices = tf.reshape(tf.tile(dim0_indices, multiples=(1, times)), [-1, 1])
    indices = tf.concat([dim0_indices, tf.reshape(paths, [-1, 1])], axis=-1)
    path_features = tf.gather_nd(node_features, indices)
    
    newshape = [tf.shape(paths)[0], tf.shape(paths)[1], tf.shape(paths)[2], tf.shape(paths)[3], -1]
    path_features = tf.reshape(path_features, newshape)

    return tf.reduce_sum(path_features, axis=-2)

graph = load_to_networkx('graphs/tiny.tntp')


graph_data = GraphData(graph=graph, graph_name='tiny', k=1, unique_neighborhoods=False)

max_in_degree = max(map(lambda t: t[1], graph.in_degree()))
max_out_degree = max(map(lambda t: t[1], graph.out_degree()))
max_degree = max(max_in_degree, max_out_degree)

print('Max Degree: {0}'.format(max_degree))

graph_data.adj_lst = pad_adj_list(adj_lst=graph_data.adj_lst,
                                  max_degree=max_degree,
                                  max_num_nodes=graph_data.num_nodes,
                                  mask_number=graph_data.num_nodes)
graph_data.set_edge_indices(adj_lst=graph_data.adj_lst,
                            max_degree=max_degree,
                            max_num_nodes=graph_data.num_nodes)

# Create node features
eigen = nx.eigenvector_centrality(graph, max_iter=1000)
out_deg = nx.out_degree_centrality(graph)
in_deg = nx.in_degree_centrality(graph)

node_features = np.zeros(shape=(graph.number_of_nodes()+1, 3))
for node in graph.nodes():
    node_features[node, 0] = node + 1
    node_features[node, 1] = node + 2
    node_features[node, 2] = node + 3

final_node = graph.number_of_nodes()
node_features[final_node, 0] = 0
node_features[final_node, 1] = 0
node_features[final_node, 2] = 0

b = 2

mask_index = np.array([[final_node] for _ in range(b)])

mask = np.zeros_like(graph_data.adj_lst)
for i in range(graph_data.adj_lst.shape[0]):
    for j in range(graph_data.adj_lst.shape[1]):
        index = graph_data.in_indices[i * max_degree + j]
        x, y = int(index[0]), int(index[1])
        mask[i, j] = int(graph_data.adj_lst[x, y] == final_node)

mask = np.array([mask for _ in range(b)]).astype(float)

inv_mask = np.equal(graph_data.inv_adj_lst, final_node).astype(int)

num_samples = 1
path_length = 3

weights = []
for i in range(b):
    w = []
    for j in range(graph.number_of_nodes()+1):
        index = i * graph.number_of_nodes() + j + 1
        w.append(np.arange(start=index, stop=max_degree+index))
    weights.append(w)
weights = np.array(weights)


rev_indices = np.zeros_like(graph_data.in_indices)
index = 0
for x in range(graph_data.adj_lst.shape[0]):
    for y in graph_data.adj_lst[x]:

        indexof = np.where(graph_data.inv_adj_lst[y] == x)[0]
        if len(indexof) > 0:
            rev_indices[index, 0] = y
            rev_indices[index, 1] = indexof[0]
        else:
            rev_indices[index, 0] = graph.number_of_nodes()
            rev_indices[index, 1] = max_degree - 1

        index += 1

# 3D indexing used for flow computation and correction
batch_indices = np.arange(start=0, stop=b)
batch_indices = np.repeat(batch_indices, graph_data.adj_lst.shape[0] * max_degree).reshape((-1, 1))

in_indices = np.vstack([graph_data.in_indices for _ in range(b)])
in_indices = np.concatenate([batch_indices, in_indices], axis=1)

rev_indices = np.vstack([rev_indices for _ in range(b)])
rev_indices = np.concatenate([batch_indices, rev_indices], axis=1)

v = graph_data.adj_lst.shape[0]
d = graph_data.adj_lst.shape[1]
f = 4

initial_states = np.zeros(shape=(v, f))
for i in range(v):
    for j in range(f):
        initial_states[i, j] = i + j + 1

features = np.zeros(shape=(v, d, f))
for i in range(v):
    for j in range(d):
        node = graph_data.adj_lst[i, j]
        for k in range(f):
            features[i, j, k] = initial_states[node, k]

initial_states = np.array([initial_states + i*10 for i in range(b)])
features = np.array([features + i*10 for i in range(b)])

print('Features')
print(features)

print('Init States')
print(initial_states)

with tf.Session() as sess:

    adj_ph = tf.placeholder(dtype=tf.int32, shape=(b,) + graph_data.adj_lst.shape, name='adj-ph')
    # in_indices_ph = tf.placeholder(dtype=tf.int32, shape=in_indices.shape, name='in-ph')
    # rev_indices_ph = tf.placeholder(dtype=tf.int32, shape=rev_indices.shape, name='rev-ph')
    # node_ph = tf.placeholder(dtype=tf.float32, shape=(b,) + node_features.shape, name='node-ph')
    mask_ph = tf.placeholder(dtype=tf.float32, shape=mask.shape, name='mask-ph')
    # weights_ph = tf.placeholder(dtype=tf.float32, shape=weights.shape, name='weights-ph')

    # op = destination_attn(node_weights=weights_ph, in_indices=in_indices_ph, rev_indices=rev_indices_ph, mask=mask_ph)

    # feed_dict = {
    #     in_indices_ph: in_indices,
    #     rev_indices_ph: rev_indices,
    #     node_ph: [node_features, node_features],
    #     weights_ph: weights,
    #     mask_ph: mask
    # }

    features_ph = tf.placeholder(dtype=tf.float32, shape=features.shape, name='features-ph')
    mask_index_ph = tf.placeholder(dtype=tf.int32, shape=mask_index.shape, name='mask-index-ph')
    initial_states_ph = tf.placeholder(dtype=tf.float32, shape=initial_states.shape, name='init-states-ph')

    feed_dict = {
        adj_ph: [graph_data.adj_lst for _ in range(b)],
        features_ph: features,
        mask_index_ph: mask_index,
        initial_states_ph: initial_states,
        mask_ph: mask
    }

    dir_gat = DirectionalGAT(output_size=f, activation=tf.nn.tanh)
    op = dir_gat(inputs=features_ph, adj_lst=adj_ph, mask_index=mask_index_ph, initial_states=initial_states_ph,
                 mask=mask_ph)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    output = sess.run(op, feed_dict=feed_dict)

    print(output)
