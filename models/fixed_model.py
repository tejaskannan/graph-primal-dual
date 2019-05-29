import tensorflow as tf
from models.base_model import Model
from utils.constants import BIG_NUMBER, SMALL_NUMBER, FLOW_THRESHOLD
from cost_functions.cost_functions import get_cost_function
from utils.flow_utils import mcf_solver
from core.layers import SparseMax

class FixedModel(Model):

    def __init__(self, params, name='fixed-model'):
        super(FixedModel, self).__init__(params, name)
        self.cost_fn = get_cost_function(cost_fn=params['cost_fn'])
        self.should_use_edges = params['cost_fn']['use_edges']

    def build(self, **kwargs):

        # B x V x D tensor of flow proportions
        flow_proportions = kwargs['flow_proportions']

        # B x V x D tensor containing the padded adjacency list
        adj_lst = kwargs['adj_lst']

        # B x V x 1 tensor of node demands
        demands = kwargs['demands']

        # B x 1 tensor containing the number of nodes in each sample
        num_nodes = kwargs['num_nodes']

        # B*V*D x 3 tensor containing 3D indices used to compute inflow
        in_indices = kwargs['in_indices']

        with self._sess.graph.as_default():        
            mask_indices = tf.expand_dims(num_nodes, axis=-1)
            mask = tf.cast(tf.equal(adj_lst, mask_indices), tf.float32)
            adj_mask = 1.0 - mask

            # Mask out nonexistent neighbors before normalization, B x V x D
            masked_proportions = (-BIG_NUMBER * mask) + flow_proportions
            if self.params['use_sparsemax']:
                sparsemax = SparseMax(epsilon=SMALL_NUMBER)
                normalized_weights = sparsemax(inputs=masked_proportions, mask=adj_mask)
            else:
                normalized_weights = tf.nn.softmax(masked_proportions, axis=-1)

            flow, pflow = mcf_solver(pred_weights=normalized_weights,
                                     demand=demands,
                                     in_indices=in_indices,
                                     max_iters=self.params['flow_iters'])

            flow_cost = tf.reduce_sum(self.cost_fn.apply(flow), axis=[1, 2])
            self.loss = flow_cost
            self.loss_op = tf.reduce_mean(self.loss)

            # Named outputs
            self.output_ops['flow'] = flow
            self.output_ops['flow_cost'] = flow_cost
            self.output_ops['normalized_weights'] = normalized_weights
            self.output_ops['dual_cost'] = tf.zeros_like(flow_cost)
