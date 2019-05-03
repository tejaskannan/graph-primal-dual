import tensorflow as tf
import numpy as np
from utils.utils import sparse_matrix_to_tensor
from scipy.sparse import csr_matrix

def sparsemax(logits, epsilon=1e-3, name='sparsemax'):
    """
    Implementation of sparsemax which supports tensors with rank 3. The sparsemax
    algorithm is presented in https://arxiv.org/abs/1602.02068, and the implementation
    is based on the code for tf.contrib.layers.sparsemax.sparsemax
    https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/sparsemax/python/ops/sparsemax.py
    """

    dims = tf.shape(logits)[-1]

    z = logits

    # Sort z vectors
    z_sorted, _ = tf.nn.top_k(z, k=dims, name='{0}-z-sort'.format(name))

    # Partial sums based on sorted vectors
    partial_sums = tf.cumsum(z_sorted, axis=-1, name='{0}-cumsum'.format(name))

    # Tensor with k values
    k = tf.range(start=1, limit=tf.cast(dims, dtype=logits.dtype) + 1,
                 dtype=logits.dtype, name='{0}-k'.format(name))

    # Tensor of ones and zeros representing which indices are greater 
    # than their respective partial sums
    z_threshold = 1.0 + k * z_sorted > partial_sums

    # k(z) value
    k_z = tf.reduce_sum(tf.cast(z_threshold, dtype=tf.int32), axis=-1)

    dim0_indices = tf.range(0, tf.shape(z)[0])
    dim1_indices = tf.range(0, tf.shape(z)[1])

    indices_x, indices_y = tf.meshgrid(dim0_indices, dim1_indices)
    indices_x = tf.reshape(tf.transpose(indices_x), [-1, 1])
    indices_y = tf.reshape(tf.transpose(indices_y), [-1, 1])

    obs_indices = tf.concat([indices_x, indices_y], axis=-1)

    indices = tf.concat([obs_indices, tf.reshape(k_z - 1, [-1, 1])], axis=1)

    tau_sum = tf.gather_nd(partial_sums, indices)
    tau_sum_reshape = tf.reshape(tau_sum, tf.shape(k_z))

    tau_z = (tau_sum_reshape - 1) / tf.cast(k_z, dtype=logits.dtype)
    tau_z = tf.expand_dims(tau_z, axis=-1)

    weights = tf.clip_by_value(tf.nn.relu(z - tau_z), epsilon, 1.0)
    return weights / tf.norm(weights, ord=1, axis=-1, keepdims=True) 


def sparse_tensor_sparsemax(inputs, num_rows, epsilon=1e-3, name='sparse-sparsemax'):
    # Fetch individual rows from the sparse tensor
    partitions = tf.cast(inputs.indices[:,0], dtype=tf.int32)
    rows = tf.dynamic_partition(inputs.values, partitions, num_rows, name='{0}-dyn-part'.format(name))
    
    def clipped_sparsemax(tensor, epsilon):
        # We need reshape the tensor because the provided sparsemax function requires
        # 2D tensors
        expanded_tensor = tf.expand_dims(tensor, axis=0)
        normalized_tensor = tf.contrib.sparsemax.sparsemax(logits=expanded_tensor,
                                                           name='{0}-sparsemax-op'.format(name))
        # Clip values if necessary
        if epsilon > 0.0:
            clipped = tf.clip_by_value(normalized_tensor, epsilon, 1.0)
            normalized_tensor = clipped / tf.norm(clipped, ord=1, axis=-1, keepdims=True)
        return normalized_tensor

    # Normalize rows using clipped sparsemax and set the value of all empty tensors
    # to -1 for later removal. This trick allows the function to handle zero rows.
    normalized = [tf.cond(tf.equal(tf.size(tensor), 0),
                          lambda: tf.constant(-1.0, shape=[1, 1], dtype=tf.float32),
                          lambda: clipped_sparsemax(tensor, epsilon)) for tensor in rows]
    concat = tf.squeeze(tf.concat(normalized, axis=1), axis=0)
    
    # Mask out empty entries (set to -1 from beforehand)
    mask = tf.logical_not(tf.equal(concat, -1.0))
    filtered = tf.boolean_mask(concat, mask)

    return tf.SparseTensor(
        indices=inputs.indices,
        values=filtered,
        dense_shape=inputs.dense_shape
    )

def neighborhood_lookahead(scores, adj_mat, beta=0.9):
    """
    scores: V x 1 tensor
    adj_mat: V x V tensor
    beta: int scalar
    """
    neighorbood_scores = tf.matmul(adj_mat, scores)

    node_weights = adj_mat * tf.transpose(scores, perm=[1, 0])

    degrees = tf.reduce_sum(adj_mat, axis=-1, keepdims=True)
    neigh_score_mat = adj_mat * tf.transpose(neighorbood_scores, perm=[1, 0])
    back_score_mat = adj_mat * scores
    #degree_mat = adj_mat * (degrees - 1 + 1e-7)

    clipped_deg = tf.transpose(tf.clip_by_value(degrees - 1, 1e-5, 10000), perm=[1, 0])
    neighborhood_weights = (neigh_score_mat - back_score_mat) / clipped_deg
    weights = adj_mat * (beta * node_weights + (1 - beta) * neighborhood_weights)
    return node_weights, adj_mat

def simple_cycle_removal(flows, adj_mat):
    return flows - adj_mat * tf.math.minimum(flows, tf.transpose(flows, perm=[1, 0]))



def create_obs_indices(dim1, dim2):
    x1 = np.arange(0, dim1)
    x2 = np.arange(0, dim2)

    a, b = np.meshgrid(x1, x2)

    a = np.reshape(a.T, [-1, 1], order='C')
    b = np.reshape(b.T, [-1, 1], order='C')
    return np.concatenate([a, b], axis=-1).astype(np.int32)


adj_mat = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]])
flows = np.array([[0, 1.0, 0, 0], [0, 0, 5.0, 0], [0, 4.0, 0, 1.0], [0, 0, 0, 0]])

scores = np.array([10, 2, 3, 4])
scores = np.reshape(scores, [-1, 1])

with tf.Session() as sess:
    # sparse_ph = tf.sparse.placeholder(dtype=tf.float32, shape=[None, arr.shape[1]], name='sparse-ph')
    # sparsemax_op = sparse_tensor_sparsemax(sparse_ph, num_rows=arr.shape[0], epsilon=0.0)

    # arr_ph = tf.placeholder(dtype=tf.float32, shape=arr.shape, name='arr-ph')
    # contrib_sparsemax = tf.contrib.sparsemax.sparsemax(logits=arr_ph)
    # softmax = tf.nn.softmax(arr_ph, axis=-1)

    flows_ph = tf.placeholder(dtype=tf.float32, shape=flows.shape, name='flows-ph')
    adj_ph = tf.placeholder(dtype=tf.float32, shape=adj_mat.shape, name='adj-ph')
    removal_op = simple_cycle_removal(flows=flows_ph, adj_mat=adj_ph)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    output = sess.run(removal_op, feed_dict={ flows_ph: flows, adj_ph: adj_mat })
    print(output)

    # print('Implemented Sparsemax')
    # sparse_tensor = sparse_matrix_to_tensor(sparse_mat)
    # print(sparse_tensor)
    # output = sess.run(sparsemax_op, feed_dict={ sparse_ph: sparse_tensor })
    # print(output)

    # print('Contrib Sparsemax')
    # output = sess.run(contrib_sparsemax, feed_dict={ arr_ph: arr })
    # print(output)

    # print('Softmax')
    # output = sess.run(softmax, feed_dict={ arr_ph: arr })
    # print(output)
