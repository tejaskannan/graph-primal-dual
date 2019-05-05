import tensorflow as tf


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


def apply_to_sparse_batch(sparse_tensor, fn, n, b):
    """
    Applies the function to each n x n block in the (n*b) x (n*b) sparse tensor.
    Returns b values, one for each block. Assumes that fn(0) = 0 and each block
    matrix has at least 1 nonzero value.
    """
    indices = tf.cast(sparse_tensor.indices, tf.int32)
    row_partitions = tf.map_fn(lambda x: x[0] // n, indices)
    partitioned_values = tf.dynamic_partition(data=sparse_tensor.values,
                                              partitions=row_partitions,
                                              num_partitions=b)
    return [fn(values) for values in partitioned_values]
