import tensorflow as tf
from utils.constants import BIG_NUMBER, FLOW_THRESHOLD
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


def dual_flow(dual_diff, adj_mask, cost_fn, step_size, momentum, max_iters, name='dual-flow'):

    def body(flow, acc, prev_flow):
        momentum_acc = momentum * acc
        predicted = tf.nn.relu(adj_mask * (flow - momentum_acc))
        derivative = cost_fn.derivative(predicted) + dual_diff
        next_acc = momentum_acc + step_size * derivative
        next_flow = tf.nn.relu(adj_mask * (flow - next_acc))
        return [next_flow, next_acc, flow]

    def cond(flow, momentum, prev_flow):
        return tf.reduce_any(tf.abs(flow - prev_flow) > FLOW_THRESHOLD)

    dual_flows = dual_diff
    acc = tf.zeros_like(dual_diff, dtype=tf.float32)
    prev_dual_flows = dual_flows + BIG_NUMBER
    shape_invariants = [dual_flows.get_shape(), acc.get_shape(), prev_dual_flows.get_shape()]
    dual_flows, _, _ = tf.while_loop(cond, body,
                                     loop_vars=[dual_flows, acc, prev_dual_flows],
                                     parallel_iterations=1,
                                     shape_invariants=shape_invariants,
                                     maximum_iterations=max_iters,
                                     name='{0}-while-loop'.format(name))

    return dual_flows


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

    gathered_scores = tf.gather_nd(weighted_scores, rev_indices)
    gathered_scores = tf.reshape(gathered_scores, shape=[node_shape[0], node_shape[1], -1])

    return weighted_scores, gathered_scores
