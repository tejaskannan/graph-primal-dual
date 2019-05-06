import tensorflow as tf
import numpy as np
import networkx as nx
from core.load import load_to_networkx
from utils.tf_utils import masked_gather, weighted_sum
from utils.utils import neighborhood_adj_lists
from core.layers import AdjGAT


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

graph = load_to_networkx('graphs/berlin-tiergarten.tntp')
adj_lst = list(map(list, iter(graph.adj.values())))

max_degree = max(map(lambda x: len(x), adj_lst))
print('Max Degree: {0}'.format(max_degree))

adj_lst_np = []
for neighbors in adj_lst:
    if len(neighbors) < max_degree:
        #neighbors = np.random.choice(neighbors, size=max_degree, replace=True)
        neighbors = np.pad(neighbors, pad_width=(0, max_degree-len(neighbors)), mode='constant',
                           constant_values=graph.number_of_nodes())
    adj_lst_np.append(neighbors)

adj_lst_np.append(np.full(shape=(max_degree,), fill_value=graph.number_of_nodes()))
adj_lst_np = np.array(adj_lst_np)

# Create node features
eigen = nx.eigenvector_centrality(graph, max_iter=1000)
out_deg = nx.out_degree_centrality(graph)
in_deg = nx.in_degree_centrality(graph)

node_features = np.zeros(shape=(graph.number_of_nodes()+1, 3))
for node in graph.nodes():
    # node_features[node, 0] = eigen[node]
    # node_features[node, 1] = out_deg[node]
    # node_features[node, 2] = in_deg[node]

    node_features[node, 0] = node + 1
    node_features[node, 1] = node + 2
    node_features[node, 2] = node + 3

final_node = graph.number_of_nodes()
node_features[final_node, 0] = 0
node_features[final_node, 1] = 0
node_features[final_node, 2] = 0

mask_index = [[4], [4]]

num_samples = 1
path_length = 3

b = 2
weights = []
for i in range(b):
    w = []
    for j in range(graph.number_of_nodes()+1):
        index = i * graph.number_of_nodes() + j
        w.append(np.arange(start=index, stop=max_degree+index))
    weights.append(w)
weights = np.array(weights)

with tf.Session() as sess:

    adj_ph = tf.placeholder(dtype=tf.int32, shape=(b,) + adj_lst_np.shape, name='adj-ph')
    node_ph = tf.placeholder(dtype=tf.float32, shape=(b,) + node_features.shape, name='node-ph')
    mask_ph = tf.placeholder(dtype=tf.int32, shape=(2, 1), name='mask-ph')
    weights_ph = tf.placeholder(dtype=tf.float32, shape=weights.shape, name='weights-ph')

    # paths = random_walks(adj_ph, num_samples, k=path_length)
    # op = get_node_features(paths, node_ph)

    # op = masked_gather(values=node_ph, indices=adj_ph, mask_index=mask_ph)
    gat = AdjGAT(output_size=3, num_heads=1)
    op = gat(inputs=node_ph, adj_lst=adj_ph, mask_index=mask_ph)
    # op = weighted_sum(values=node_ph,
    #                   indices=adj_ph,
    #                   weights=weights_ph)

    feed_dict = {
        adj_ph: [adj_lst_np, adj_lst_np],
        node_ph: [node_features, node_features],
        #weights_ph: weights
        mask_ph: mask_index
    }

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    output = sess.run(op, feed_dict=feed_dict)

    print(output)
