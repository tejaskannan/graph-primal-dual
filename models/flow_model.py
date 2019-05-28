import tensorflow as tf
from models.base_model import Model
from core.layers import MLP, GRU, SparseMax
from utils.constants import BIG_NUMBER, SMALL_NUMBER, FLOW_THRESHOLD
from utils.tf_utils import masked_gather
from utils.flow_utils import mcf_solver, dual_flow, destination_attn
from cost_functions.cost_functions import get_cost_function
from models.aggregators import Neighborhood, GAT, GGNN


class FlowModel(Model):

    def __init__(self, params, name='flow-model'):
        super(FlowModel, self).__init__(params, name)
        self.cost_fn = get_cost_function(cost_fn=params['cost_fn'])
        self.should_use_edges = params['cost_fn']['use_edges']

    def build(self, **kwargs):

        # B x V x 1 tensor which contains node demands
        demands = kwargs['demands']

        # B x V x F tensor which contains node features
        node_features = kwargs['node_features']

        # B x V x D tensor containing the padded adjacency list
        adj_lst = kwargs['adj_lst']

        # B x V x D tensor containing padded inverse adjacency list
        inv_adj_lst = kwargs['inv_adj_lst']

        # B x V x D tensor of edge lengths
        edge_lengths = kwargs['edge_lengths']

        # B x V x D tensor of normalized edge lengths
        norm_edge_lengths = kwargs['norm_edge_lengths']

        # B x V x 2D tensor of commmon outgoing neighbors
        common_neighbors = kwargs['common_neighbors']

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

                # Select specific node aggregator
                if self.params['name'] == 'neighborhood':
                    node_aggregator = Neighborhood(output_size=self.params['node_encoding'],
                                                   num_heads=self.params['num_heads'],
                                                   activation=tf.nn.tanh,
                                                   name='neighborhood-aggregator')
                elif self.params['name'] == 'gat':
                    node_aggregator = GAT(output_size=self.params['node_encoding'],
                                          num_heads=self.params['num_heads'],
                                          activation=tf.nn.tanh,
                                          use_gru_gate=False,
                                          name='GAT-aggregator')
                elif self.params['name'] == 'gated-gat':
                    node_aggregator = GAT(output_size=self.params['node_encoding'],
                                          num_heads=self.params['num_heads'],
                                          activation=tf.nn.tanh,
                                          use_gru_gate=True,
                                          name='GRU-GAT-aggregator')
                elif self.params['name'] == 'ggnn':
                    node_aggregator = GGNN(output_size=self.params['node_encoding'],
                                           activation=tf.nn.tanh,
                                           name='GGNN-aggregator')
                else:
                    raise ValueError('Model with name {0} does not exist.'.format(self.params['name']))

                # Combine message passing steps
                for _ in range(self.params['graph_layers']):
                    node_encoding = node_aggregator(node_states=node_encoding,
                                                    adj_lst=adj_lst,
                                                    inv_adj_lst=inv_adj_lst,
                                                    node_indices=node_indices,
                                                    dropout_keep_prob=dropout_keep_prob,
                                                    neighborhoods=neighborhoods,
                                                    num_nodes=num_nodes)

                # Neighbor States, B x V x D x K
                neighbor_states, _ = masked_gather(values=node_encoding,
                                                   indices=adj_lst,
                                                   mask_index=num_nodes,
                                                   set_zero=True)

                # Mask to remove nonexistent edges, B x V x D
                mask_indices = tf.expand_dims(num_nodes, axis=-1)
                mask = tf.cast(tf.equal(adj_lst, mask_indices), tf.float32)
                adj_mask = 1.0 - mask

                # Current States tiled across neighbors, B x V x D x K
                tiled_states = tf.tile(tf.expand_dims(node_encoding, axis=-2),
                                       multiples=(1, 1, tf.shape(neighbor_states)[2], 1))
                tiled_states = tf.expand_dims(adj_mask, axis=-1) * tiled_states

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

                # # B x V x D, node weights augmented by destinations
                # inv_mask = tf.cast(tf.equal(inv_adj_lst, mask_indices), tf.float32)
                # node_weights = destination_attn(node_weights=node_weights,
                #                                 in_indices=in_indices,
                #                                 rev_indices=rev_indices,
                #                                 mask=inv_mask)

                # Mask out nonexistent neighbors before normalization, B x V x D
                pred_weights = (-BIG_NUMBER * mask) + node_weights

                # Normalize weights for outgoing neighbors
                if self.params['use_sparsemax']:
                    sparsemax = SparseMax(epsilon=SMALL_NUMBER)
                    normalized_weights = sparsemax(inputs=pred_weights, mask=adj_mask)
                else:
                    normalized_weights = tf.nn.softmax(pred_weights, axis=-1)

                normalized_weights = tf.debugging.check_numerics(normalized_weights, 'Normalized Weights has Inf or NaN.')

                flow, pflow = mcf_solver(pred_weights=normalized_weights,
                                         demand=demands,
                                         in_indices=in_indices,
                                         max_iters=self.params['flow_iters'])

                flow = tf.debugging.check_numerics(flow, 'Flow has Inf or NaN.')

                if self.should_use_edges:
                    flow_cost = tf.reduce_sum(self.cost_fn.apply(flow, edge_lengths), axis=[1, 2])
                else:
                    flow_cost = tf.reduce_sum(self.cost_fn.apply(flow), axis=[1, 2])

                flow_cost = tf.debugging.check_numerics(flow_cost, 'Flow Cost has Inf or NaN.')

                # Compute Dual Problem and associated cost
                dual_decoder = MLP(hidden_sizes=self.params['decoder_hidden'],
                                   output_size=1,
                                   activation=tf.nn.tanh,
                                   activate_final=False,
                                   name='dual-decoder')

                node_encoding = tf.debugging.check_numerics(node_encoding, 'Node Encoding has Inf or Nan.')

                dual_vars = dual_decoder(inputs=node_encoding)

                # dual_vars = tf.debugging.check_numerics(dual_vars, 'Dual Variables have Inf or Nan.')

                # B x (V + 1) x D tensor of repeated dual variables
                dual = adj_mask * dual_vars

                # Need to compute transpose (via a masked gather)
                dual_tr, _ = masked_gather(values=dual_vars,
                                           indices=adj_lst,
                                           mask_index=num_nodes,
                                           set_zero=True)
                dual_tr = tf.squeeze(dual_tr, axis=-1)

                # alpha_j - alpha_i
                dual_diff = dual_tr - dual

                # B x V x D
                dual_flows, dual_idx = dual_flow(dual_diff=dual_diff,
                                                 adj_mask=adj_mask,
                                                 cost_fn=self.cost_fn,
                                                 edge_lengths=edge_lengths,
                                                 should_use_edges=self.should_use_edges,
                                                 step_size=self.params['dual_step_size'],
                                                 momentum=self.params['dual_momentum'],
                                                 max_iters=self.params['dual_iters'])

                dual_flows = tf.debugging.check_numerics(dual_flows, 'Dual Flows have Inf or NaN.')

                dual_demand = tf.reduce_sum(dual_vars * demands, axis=[1, 2])

                if self.should_use_edges:
                    dual_flow_cost = self.cost_fn.apply(dual_flows, edge_lengths)
                else:
                    dual_flow_cost = self.cost_fn.apply(dual_flows)

                dual_flow_cost += dual_diff * dual_flows
                dual_cost = tf.reduce_sum(dual_flow_cost, axis=[1, 2]) - dual_demand

                dual_cost = tf.debugging.check_numerics(dual_cost, 'Dual Cost has Inf or NaN.')

                # tf.summary.histogram('Duality Gap', flow_cost - dual_cost)

                # Add a regularizer to penalize incentivize the model to produce a tree
                # nonzero_prop = tf.cast(normalized_weights > SMALL_NUMBER, dtype=tf.int32)
                # nonzero_reg = tf.expand_dims(tf.reduce_sum(nonzero_prop, axis=[1, 2]), axis=-1) / num_nodes
                # nonzero_reg = tf.cast(tf.squeeze(nonzero_reg, axis=-1), dtype=tf.float32)

                self.loss = (flow_cost - dual_cost)
                self.loss_op = tf.reduce_mean(self.loss)

                # Named outputs
                self.output_ops['flow'] = flow
                self.output_ops['flow_cost'] = flow_cost
                self.output_ops['normalized_weights'] = normalized_weights
                self.output_ops['dual_cost'] = dual_cost
                self.output_ops['pred_weights'] = pred_weights
                self.output_ops['dual_flow'] = dual_flows
                self.output_ops['dual_idx'] = dual_idx

                self.optimizer_op = self._build_optimizer_op()
                # self.train_writer = tf.summary.FileWriter('./logs/train', self._sess.graph)
