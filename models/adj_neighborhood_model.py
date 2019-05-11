import tensorflow as tf
from models.base_model import Model
from core.layers import MLP, AdjGAT, GRU, AttentionNeighborhood, SparseMax
from utils.constants import BIG_NUMBER, SMALL_NUMBER
from utils.tf_utils import masked_gather
from utils.flow_utils import mcf_solver, dual_flow
from cost_functions.cost_functions import get_cost_function


class AdjModel(Model):

    def __init__(self, params, name='neighborhood-model'):
        super(AdjModel, self).__init__(params, name)
        self.cost_fn = get_cost_function(cost_fn=params['cost_fn'])

    def build(self, **kwargs):

        # B x V x 1 tensor which contains node demands
        demands = kwargs['demands']

        # B x V x F tensor which contains node features
        node_features = kwargs['node_features']

        # B x V x D tensor containing the padded adjacency list
        adj_lst = kwargs['adj_lst']

        # List of B x V x D tensors containing padded adjacency lists for k neighborhood levels
        neighborhoods = kwargs['neighborhoods']

        # B*V*D x 3 tensor containing 3D indices used to compute inflow
        in_indices = kwargs['in_indices']

        # B*V*D x 3 tensor containing 2D indices of outgoing neighbors
        rev_indices = kwargs['rev_indices']

        # B x 1
        num_nodes = kwargs['num_nodes']

        # Floating point number between 0 and 1
        dropout_keep_prob = kwargs['dropout_keep_prob']

        # Boolean
        should_correct_flows = kwargs['should_correct_flows']

        # Scalar Int
        max_num_nodes = kwargs['max_num_nodes']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                node_indices = tf.range(start=0, limit=max_num_nodes)
                node_indices = tf.tile(tf.expand_dims(node_indices, axis=0),
                                       multiples=(tf.shape(num_nodes)[0], 1))

                node_embedding_init = tf.random.normal(shape=(max_num_nodes, self.params['node_embedding_size']))
                node_embedding_var = tf.Variable(node_embedding_init,
                                                 trainable=True,
                                                 name='node-embedding-var')
                node_embeddings = tf.nn.embedding_lookup(params=node_embedding_var,
                                                         ids=node_indices,
                                                         max_norm=1,
                                                         name='node-embedding-lookup')

                # Node encoding, B x V x K
                encoder = MLP(hidden_sizes=[],
                              output_size=self.params['node_encoding'],
                              activation=None,
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

                node_gru = GRU(output_size=self.params['node_encoding'],
                               activation=tf.nn.tanh,
                               name='node-gru')

                # Combine message passing steps
                for _ in range(self.params['graph_layers']):
                    next_encoding, attn_weights = node_neighborhood(inputs=node_encoding,
                                                                    neighborhoods=neighborhoods,
                                                                    mask_index=num_nodes,
                                                                    dropout_keep_prob=dropout_keep_prob)
                    node_encoding = node_gru(inputs=next_encoding,
                                             state=node_encoding,
                                             dropout_keep_prob=dropout_keep_prob)

                # Neighbor States, B x V x D x K
                neighbor_states, _ = masked_gather(values=node_encoding,
                                                indices=adj_lst,
                                                mask_index=num_nodes,
                                                set_zero=True)

                # Mask to remove nonexistent edges, B x V x D
                mask_indices = tf.expand_dims(num_nodes, axis=-1)
                mask = tf.cast(tf.equal(adj_lst, mask_indices), tf.float32)

                # Current States tiled across neighbors, B x V x D x K
                tiled_states = tf.tile(tf.expand_dims(node_encoding, axis=-2),
                                       multiples=(1, 1, tf.shape(neighbor_states)[2], 1))
                tiled_states = tf.expand_dims(1.0 - mask, axis=-1) * tiled_states

                concat_states = tf.concat([tiled_states, neighbor_states], axis=-1)

                # Compute flow proportions
                decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                              output_size=1,
                              activation=tf.nn.tanh,
                              activate_final=False,
                              name='node-decoder')

                # B x V x D x 1
                node_weights = decoder(inputs=concat_states)

                # B x V x D
                node_weights = tf.squeeze(node_weights, axis=-1)

                # Mask out nonexistent neighbors for normalization, B x V x D
                pred_weights = (-BIG_NUMBER * mask) + node_weights

                # Normalize weights for outgoing neighbors
                if self.params['use_sparsemax']:
                    sparsemax = SparseMax(epsilon=1e-5)
                    normalized_weights = sparsemax(inputs=pred_weights, mask=(1.0 - mask))
                else:
                    normalized_weights = tf.nn.softmax(pred_weights, axis=-1)

                flow, pflow = mcf_solver(pred_weights=normalized_weights,
                                         demand=demands,
                                         in_indices=in_indices,
                                         max_iters=self.params['flow_iters'])

                if should_correct_flows:
                    rev_flow = tf.reshape(tf.gather_nd(flow, rev_indices), tf.shape(pred_weights))
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

                # alpha_j - alpha_i
                dual_diff = dual_tr - dual

                # B x V x D
                dual_flows = dual_flow(dual_diff=dual_diff,
                                       adj_mask=mask,
                                       cost_fn=self.cost_fn,
                                       step_size=self.params['dual_step_size'],
                                       momentum=self.params['dual_momentum'],
                                       max_iters=self.params['dual_iters'])

                dual_demand = tf.reduce_sum(dual_vars * demands, axis=[1, 2])
                dual_flow_cost = self.cost_fn.apply(dual_flows) + dual_diff * dual_flows
                dual_cost = tf.reduce_sum(dual_flow_cost, axis=[1, 2]) - dual_demand

                self.loss = flow_cost - dual_cost
                self.loss_op = tf.reduce_mean(self.loss)
                self.output_ops += [flow, flow_cost, adj_lst, normalized_weights, dual_cost, attn_weights, concat_states, pred_weights]
                self.optimizer_op = self._build_optimizer_op()
