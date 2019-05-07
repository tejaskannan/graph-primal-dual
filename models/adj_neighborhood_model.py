import tensorflow as tf
import numpy as np
from models.base_model import Model
from core.layers import MLP, AdjGAT, GRU, AttentionNeighborhood
from core.layers import DualFlow, SparseMax
from utils.constants import BIG_NUMBER, SMALL_NUMBER
from utils.tf_utils import masked_gather
from utils.flow_utils import mcf_solver, dual_flow
from cost_functions.cost_functions import get_cost_function, apply_with_capacities


class AdjModel(Model):

    def __init__(self, params, name='neighborhood-model'):
        super(AdjModel, self).__init__(params, name)
        self.cost_fn = get_cost_function(cost_fn=params['cost_fn'])

    def build(self, **kwargs):

        # B x V x 1 tensor which contains node demands
        demands = kwargs['demands']

        # B x V x F tensor which contains node features
        node_features = kwargs['node_features']

        # B x V x E tensor which contains pre-computed node embeddings
        node_embeddings = kwargs['node_embeddings']

        # B x V x D tensor containing the padded adjacency list
        adj_lst = kwargs['adj_lst']

        # List of B x V x D tensors containing padded adjacency lists for k neighborhood levels
        neighborhoods = kwargs['neighborhoods']

        # B*V*D x 3 tensor containing 3D indices used to compute inflow
        flow_indices = kwargs['flow_indices']

        # B*V*D x 3 tensor containing 2D indices of outgoing neighbors
        out_indices = kwargs['out_indices']

        # B x 1
        num_nodes = kwargs['num_nodes'] 

        # Float
        dropout_keep_prob = kwargs['dropout_keep_prob']

        # Integer
        num_output_features = kwargs['num_output_features']

        # Boolean
        should_correct_flows = kwargs['should_correct_flows']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                # Node encoding
                encoder = MLP(hidden_sizes=self.params['encoder_hidden'],
                              output_size=self.params['node_encoding'],
                              activation=tf.nn.tanh,
                              activate_final=True,
                              name='node-encoder')
                node_encoding = encoder(inputs=tf.concat([node_embeddings, node_features], axis=-1),
                                        dropout_keep_prob=dropout_keep_prob)

                node_neighborhood = AttentionNeighborhood(output_size=self.params['node_encoding'],
                                                          num_heads=self.params['num_heads'],
                                                          activation=tf.nn.tanh,
                                                          is_sparse=False,
                                                          use_adj_lists=True,
                                                          name='node-neighborhood')

                # node_neighborhood = AdjGAT(output_size=self.params['node_encoding'],
                #                            num_heads=self.params['num_heads'],
                #                            activation=tf.nn.tanh,
                #                            name='node-gat')
                node_gru = GRU(output_size=self.params['node_encoding'],
                               activation=tf.nn.tanh,
                               name='node-gru')

                # Combine message passing steps
                for _ in range(self.params['graph_layers']):
                    # next_encoding = node_neighborhood(inputs=node_encoding,
                    #                                   adj_lst=adj_lst,
                    #                                   mask_index=num_nodes)
                    next_encoding, attn_weights = node_neighborhood(inputs=node_encoding,
                                                                    neighborhoods=neighborhoods,
                                                                    mask_index=num_nodes,
                                                                    dropout_keep_prob=dropout_keep_prob)
                    node_encoding = node_gru(inputs=next_encoding,
                                             state=node_encoding,
                                             dropout_keep_prob=dropout_keep_prob)

                # Compute flow proportions
                decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                              output_size=1,
                              activation=tf.nn.tanh,
                              activate_final=False,
                              name='node-decoder')

                # B x V x 1
                node_weights = decoder(inputs=node_encoding)

                pred_weights, _ = masked_gather(values=node_weights,
                                                indices=adj_lst,
                                                mask_index=num_nodes)
                pred_weights = tf.squeeze(pred_weights, axis=-1)

                # Mask to remove nonexistent edges
                mask_indices = tf.expand_dims(num_nodes, axis=-1)
                mask = 1.0 - tf.cast(tf.equal(adj_lst, mask_indices), tf.float32)

                # Normalize weights for outgoing neighbors
                if self.params['use_sparsemax']:
                    sparsemax = SparseMax(epsilon=1e-5)
                    normalized_weights = sparsemax(inputs=pred_weights, mask=mask)
                else:
                    normalized_weights = tf.nn.softmax(pred_weights, axis=-1)

                flow, pflow = mcf_solver(pred_weights=normalized_weights,
                                         demand=demands,
                                         flow_indices=flow_indices,
                                         max_iters=self.params['flow_iters'])

                if should_correct_flows:
                    rev_flow = tf.reshape(tf.gather_nd(flow, out_indices), tf.shape(pred_weights))
                    flow_diff = tf.minimum(flow, rev_flow)
                    flow = tf.nn.relu(flow - flow_diff)

                flow_cost = tf.reduce_sum(self.cost_fn.apply(flow), axis=[1, 2])

                # Compute Dual Problem and associated cost
                dual_decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                                   output_size=1,
                                   activation=tf.nn.tanh,
                                   activate_final=False,
                                   name='dual-decoder')
                dual_vars = dual_decoder(inputs=node_encoding)

                # B x (V + 1) x D tensor of repeated dual variables
                dual = mask * dual_vars

                # Need to compute transpose (via a masked gather)
                dual_tr, _ = masked_gather(values=dual_vars,
                                           indices=adj_lst,
                                           mask_index=num_nodes,
                                           set_zero=True)
                dual_tr = tf.squeeze(dual_tr, axis=-1)
                dual_diff = dual - dual_tr

                # B x V x D
                dual_flows = dual_flow(dual_diff=dual_diff,
                                       adj_mask=mask,
                                       cost_fn=self.cost_fn,
                                       step_size=self.params['dual_step_size'],
                                       momentum=self.params['dual_momentum'],
                                       max_iters=self.params['dual_iters'])

                dual_demand = tf.reduce_sum(dual_vars * demands, axis=[1, 2])
                dual_flow_cost = self.cost_fn.apply(dual_flows) - dual_diff * dual_flows
                dual_cost = tf.reduce_sum(dual_flow_cost, axis=[1, 2]) - dual_demand

                self.loss = flow_cost - dual_cost
                self.loss_op = tf.reduce_mean(self.loss)
                self.output_ops += [flow, flow_cost, adj_lst, normalized_weights, dual_cost]
                self.optimizer_op = self._build_optimizer_op()
