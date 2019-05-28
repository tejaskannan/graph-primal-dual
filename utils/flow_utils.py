import tensorflow as tf
from utils.constants import BIG_NUMBER, FLOW_THRESHOLD, SMALL_NUMBER
from utils.tf_utils import masked_gather


def mcf_solver(pred_weights, demand, in_indices, max_iters, name='mcf-solver'):
    """
    pred_weights: B x (V+1) x D tensor
    demand: B x (V+1) x 1 tensor
    inv_adj_list: B x (V+1) x D tensor

    Returns: B x (V+1) x D tensor containing flow volumes
    """

    def body(flow, prev_flow):
        # Get incoming flows, B * (V+1) * D x 1  tensor
        inflow = tf.gather_nd(flow, in_indices)
        inflow = tf.reshape(inflow, tf.shape(pred_weights))

        total_inflow = tf.reduce_sum(inflow, axis=-1, keepdims=True)

        # Adjust flow by demand, B x (V+1) x 1 tensor
        adjusted_inflow = tf.nn.relu(total_inflow - demand)

        # Determine outgoing flows using computed weights, B x (V+1) x D tensor
        prev_flow = flow
        flow = pred_weights * adjusted_inflow
        return [flow, prev_flow]

    def cond(flow, prev_flow):
        return tf.reduce_any(tf.abs(flow - prev_flow) > FLOW_THRESHOLD)

    # Iteratively computes flows from flow proportions
    flow = tf.zeros_like(pred_weights, dtype=tf.float32)
    prev_flow = flow + BIG_NUMBER
    shape_invariants = [flow.get_shape(), prev_flow.get_shape()]
    flow, pflow = tf.while_loop(cond=cond,
                                body=body,
                                loop_vars=[flow, prev_flow],
                                parallel_iterations=1,
                                shape_invariants=shape_invariants,
                                maximum_iterations=max_iters,
                                name='{0}-while-loop'.format(name))
    return flow, pflow


def directional_mcf_solver(pred_weights, demand, in_indices, num_in_neighbors, max_iters, name='dir-mcf-solver'):
    """
    pred_weights: B x V x D x D tensor
    demand: B x V x 1 tensor
    in_indices: B*V*D x 3 tensor
    num_in_neighbors: B x V x 1 tensor

    Returns: B x V x D tensor containing flow volumes
    """
    in_neighbors = tf.reciprocal(tf.expand_dims(num_in_neighbors, axis=-1) + SMALL_NUMBER)
    demand_expanded = tf.expand_dims(demand, axis=-1)
    demand_tiled = tf.tile(demand, multiples=(1, 1, tf.shape(pred_weights)[2]))

    def body(flow, prev_flow):
        # Get incoming flows, B * V * D x 1  tensor
        inflow = tf.gather_nd(flow, in_indices)
        inflow = tf.reshape(inflow, tf.shape(pred_weights)[0:3])

        # B x V x D x 1
        in_out_flow = tf.expand_dims(inflow, axis=-1)

        # B x V x 1
        total_inflow = tf.reduce_sum(inflow, axis=-1, keepdims=True)
        inflow_tiled = tf.tile(total_inflow, multiples=(1, 1, tf.shape(demand_tiled)[-1]))

        # Adjust flow by demand, B x V x D x D tensor
        adjusted_flow = pred_weights * (in_out_flow - demand_expanded * in_neighbors)

        # Determine outgoing flows using computed weights, B x V x D tensor
        next_flow = tf.where(inflow_tiled > demand_tiled,
                            x=tf.reduce_sum(adjusted_flow, axis=-2),
                            y=tf.zeros_like(prev_flow))

        return [next_flow, flow]

    def cond(flow, prev_flow):
        return tf.reduce_any(tf.abs(flow - prev_flow) > FLOW_THRESHOLD)

    # Iteratively computes flows from flow proportions
    flow = tf.zeros(shape=tf.shape(pred_weights)[0:3], dtype=tf.float32)
    prev_flow = flow + BIG_NUMBER
    shape_invariants = [flow.get_shape(), prev_flow.get_shape()]
    flow, pflow = tf.while_loop(cond=cond,
                                body=body,
                                loop_vars=[flow, prev_flow],
                                parallel_iterations=1,
                                shape_invariants=shape_invariants,
                                maximum_iterations=max_iters,
                                name='{0}-while-loop'.format(name))
    return flow, pflow


def dual_flow(dual_diff, adj_mask, cost_fn, edge_lengths, should_use_edges, step_size, momentum, max_iters, name='dual-flow'):

    def body(flow, acc, prev_flow):
        momentum_acc = momentum * acc
        predicted = tf.nn.relu(adj_mask * (flow - momentum_acc))

        if should_use_edges:
            derivative = cost_fn.derivative(predicted, edge_lengths) + dual_diff
        else:
            derivative = cost_fn.derivative(predicted) + dual_diff

        next_acc = momentum_acc + step_size * derivative
        next_flow = tf.nn.relu(adj_mask * (flow - next_acc))
        return [next_flow, next_acc, flow]

    def cond(flow, momentum, prev_flow):
        return tf.reduce_any(tf.abs(flow - prev_flow) > FLOW_THRESHOLD)

    dual_flows = tf.nn.relu(adj_mask * dual_diff)
    acc = tf.zeros_like(dual_diff, dtype=tf.float32)
    prev_dual_flows = dual_flows + BIG_NUMBER
    shape_invariants = [dual_flows.get_shape(), acc.get_shape(), prev_dual_flows.get_shape()]
    dual_flows, _, pflow = tf.while_loop(cond, body,
                                     loop_vars=[dual_flows, acc, prev_dual_flows],
                                     parallel_iterations=1,
                                     shape_invariants=shape_invariants,
                                     maximum_iterations=max_iters,
                                     name='{0}-while-loop'.format(name))

    return dual_flows, pflow


def destination_attn(node_weights, in_indices, rev_indices, mask, name='dest-attn'):
    """
    node_weights: B x V x D tensor of outgoing node weights
    in_indices: B*V*D x 3 tensor of indices marking incoming neighbors
    mask_index: B x 1 tensor marking the index to mask out

    Returns: B x V x D tensor of weighted node weights based on destination
    """

    # B x V x D tensor of node weights for incoming neighbors
    node_shape = tf.shape(node_weights)

    in_weights = tf.gather_nd(node_weights, in_indices)
    in_weights = (-BIG_NUMBER * mask) + tf.reshape(in_weights, shape=[node_shape[0], node_shape[1], -1])

    # Normalize scores, represents weights augmented by destination attention
    normalized_weights = tf.nn.softmax(in_weights, axis=-1)

    weighted_scores = normalized_weights * in_weights * (1.0 - mask)

    # Re-disribute scores back to the outgoing edges of the original vertices
    gathered_scores = tf.gather_nd(weighted_scores, rev_indices)
    gathered_scores = tf.reshape(gathered_scores, shape=[node_shape[0], node_shape[1], -1])

    return gathered_scores


def correct_proportions(source_demands, sink_demands, proportions, source_dest_mat, source_index_mat, max_iters):
    """
    source_demands: B x K_1 x 1 tensor of source demands
    sink_demands: B x K_2 x 1 tensor of sink demands
    proportions: B x K_2 x K_1 tensor of flow proportions
    source_dest_mat: B x K_2 x K_1*K_2 tensor of source demands for each sink
    source_index_mat: B x K_1 x K_1*K_2 tensor of source indices
    """
    batch_size = tf.shape(source_demands)[0]

    # B x (K1 + K2) x (K1 * K2) tensor
    A = tf.concat([source_dest_mat, source_index_mat], axis=1)

    # B x (K1 + K2) x 1 tensor
    b = tf.concat([sink_demands, tf.ones_like(source_demands)], axis=1)

    # B x (K1 + K2) x 1 tensor
    p = tf.reshape(proportions, [batch_size, -1, 1])

    n, m = tf.shape(source_demands)[1], tf.shape(sink_demands)[1]
    dim = (n * m) + (n + m)

    # B x (K1 * K2) x (K1 * K2) identity matrix
    I = tf.tile(tf.expand_dims(tf.eye(n * m), axis=0), multiples=(batch_size, 1, 1))

    # B x (K1 + K2) x (K1 + K2) zero matrix
    zero = tf.zeros(shape=(batch_size, n + m, n + m))

    # B x (K1 * K2 + K1 + K2) x (K1 * K2)
    lhs = tf.concat([I, A], axis=1)

    # B x (K1 * K2 + K1 + K2) x (K1 + K2)
    A_T = tf.transpose(A, perm=[0, 2, 1])
    rhs = tf.concat([A_T, zero], axis=1)
    
    # B x (K1 * K2 + K1 + K2) x (K1 * K2 + K1 + K2)
    mat = tf.concat([lhs, rhs], axis=2)

    # B x (K1 + K2) x 1
    zero_pad = tf.zeros(shape=(batch_size, n + m, 1))

    # B x (K1 * K2) x 1 tensor
    v = tf.linalg.lstsq(matrix=A, rhs=b, fast=False, name='v-init')
    v_prev = v + BIG_NUMBER

    # Use constant step size for simplicity
    step_size = tf.constant(1.0, dtype=tf.float32)

    def body(v, v_prev, step_size):
        # B x (K1 * K2 + K1 + K2) x 1
        target = tf.concat([p - v, zero_pad], axis=1)

        # B x (K1 * K2 + K1 + K2) x 1
        sol = tf.linalg.lstsq(matrix=mat, rhs=target, fast=False, name='v-step')

        # B x (K1 * K2) x 1
        delta_v = sol[:,:n*m,:]

        next_v = v + step_size * delta_v
        return [next_v, v, step_size * beta]

    def cond(v, v_prev, step_size):
        return tf.reduce_any(tf.square(tf.norm(v - v_prev, axis=1)) > FLOW_THRESHOLD)

    props, _, _ = tf.while_loop(cond, body,
                                loop_vars=[v, v_prev, step_size],
                                shape_invariants=[v.get_shape(), v_prev.get_shape(), step_size.get_shape()],
                                parallel_iterations=1,
                                name='newton-correction',
                                maximum_iterations=max_iters)

    return tf.reshape(props, [batch_size, n, m])