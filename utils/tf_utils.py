import tensorflow as tf
from utils.constants import BIG_NUMBER


def mask_sp_tensor(sp_a, sp_b):
    """
    Removes all elements from sp_a which correspond to indices in sp_b. This function
    assumes that sp_b.indices is a subset of sp_a.indices and that both have the same
    (square) dense shape.
    """

    # Concatenate indices together. The ordering here is important because
    # it ensure that the 'unique' ordering below is the same as that of
    # sp_a.indices.
    concat_indices = tf.concat([sp_a.indices, sp_b.indices], axis=0)

    # Linearize indices into 1D tensor
    linear_mat = tf.cast([[tf.shape(sp_a.values)[0]], [1]], tf.int64)
    linearized = tf.matmul(concat_indices, tf.cast(linear_mat, tf.int64))

    # Get unique indices and values
    values, indices = tf.unique(tf.squeeze(linearized))

    # Assign each index a count of one and then aggregate. This strategy works becuase
    # there are at most 2 copies of a particular index and any singles mean that the
    # index is only present in sp_a (under the assumption of sp_b.indices being a subset
    # of sp_a.indices)
    counts = tf.ones_like(indices, dtype=tf.int64)
    counts = tf.unsorted_segment_sum(counts, indices, num_segments=tf.shape(values)[0])

    # Remove relevant indices
    return tf.sparse.retain(sp_a, counts > 1)


def sparse_subtract(sp_a, sp_b):
    return tf.sparse.add(sp_a, sparse_scalar_mul(sp_b, -1))


def sparse_scalar_mul(sparse_tensor, scalar):
    return tf.SparseTensor(
        indices=sparse_tensor.indices,
        values=scalar * sparse_tensor.values,
        dense_shape=sparse_tensor.dense_shape
    )


def gather_rows(values, indices, name='gather-rows'):
    row_indices = tf.expand_dims(tf.range(start=0, limit=tf.shape(values)[0]), axis=-1)

    index_shape = tf.shape(indices)
    times = index_shape[1] * index_shape[2]
    row_indices = tf.reshape(tf.tile(row_indices, multiples=(1, times)), [-1, 1])

    value_indices = tf.concat([row_indices, tf.reshape(indices, [-1, 1])], axis=-1)

    gathered_values = tf.gather_nd(params=values, indices=value_indices)
    return gathered_values, value_indices


def masked_gather(values, indices, mask_index, name='masked-gather'):
    """
    This function retrieves rows along axis 1 of 'values' corresponding
    to the given indices. All indices which equal to mask_index
    are masked and assigned a large negative number. This function
    returns a B x V x D x F tensor containing the gathered values.

    values is a B x V x F tensor
    indices is a B x V x D tensor
    mask_index is a B x 1 tensor
    """
    gathered_values, value_indices = gather_rows(values, indices)

    index_shape = tf.shape(indices)
    indices_y = tf.reshape(value_indices[:,1], [index_shape[0], -1])

    mask = tf.cast(tf.equal(indices_y, mask_index), tf.float32)
    mask = -BIG_NUMBER * mask

    new_shape = [index_shape[0], index_shape[1], index_shape[2], tf.shape(values)[2]] 
    gathered_values = tf.reshape(gathered_values, new_shape)

    mask = tf.reshape(mask, new_shape[0:3] + [-1])
    masked_values = gathered_values + mask

    return masked_values, mask


def weighted_sum(values, indices, weights, name='weighted-sum'):
    """
    values: B x V x F tensor
    indices: B x V x D tensor
    weights: B x V x 1 tensor

    Returns: B x V x F tensor
    """

    # (B * V * D) x F tensor
    gathered_values, _ = gather_rows(values, indices)

    # B x V x D x F tensor
    index_shape = tf.shape(indices)
    new_shape = [index_shape[0], index_shape[1], index_shape[2], tf.shape(values)[2]] 
    gathered_values = tf.reshape(gathered_values, new_shape)

    # B x V x D x F tensor
    weights = tf.reshape(weights, [index_shape[0], index_shape[1], 1, 1])
    weighted_values = gathered_values * weights

    return tf.reduce_sum(weighted_values, axis=-2)
