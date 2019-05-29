import tensorflow as tf
from core.layers import GRU, MLP, AttentionNeighborhood, AdjGAT
from utils.tf_utils import masked_gather


class Aggregator:

    def __init__(self, output_size, activation, name):
        self.output_size = output_size
        self.activation = activation
        self.name = name

    def __call__(self, node_states, dropout_keep_prob, **kwargs):
        raise NotImplementedError()


class Neighborhood(Aggregator):

    def __init__(self, output_size, num_heads, activation, name):
        super(Neighborhood, self).__init__(output_size, activation, name)
        self.num_heads = num_heads

        self.out_neighborhood_agg = AttentionNeighborhood(output_size=output_size,
                                                          num_heads=num_heads,
                                                          activation=activation,
                                                          is_sparse=False,
                                                          use_adj_lists=True,
                                                          name='{0}-node-out-neighborhood'.format(name))

        self.in_neighborhood_agg = AttentionNeighborhood(output_size=output_size,
                                                         num_heads=num_heads,
                                                         activation=activation,
                                                         is_sparse=False,
                                                         use_adj_lists=True,
                                                         name='{0}-node-in-neighborhood'.format(name))

        self.combiner = MLP(hidden_sizes=[],
                            output_size=output_size,
                            activation=activation,
                            activate_final=True,
                            bias_final=True,
                            name='{0}-combiner'.format(name))

        self.node_gru = GRU(output_size=output_size,
                            activation=activation,
                            name='{0}-node-gru'.format(name))

    def __call__(self, node_states, dropout_keep_prob, **kwargs):
        out_neighborhoods = kwargs['out_neighborhoods']
        in_neighborhoods = kwargs['in_neighborhoods']
        num_nodes = kwargs['num_nodes']

        out_encoding, _ = self.out_neighborhood_agg(inputs=node_states,
                                                    neighborhoods=out_neighborhoods,
                                                    mask_index=num_nodes,
                                                    dropout_keep_prob=dropout_keep_prob)

        in_encoding, _ = self.in_neighborhood_agg(inputs=node_states,
                                                  neighborhoods=in_neighborhoods,
                                                  mask_index=num_nodes,
                                                  dropout_keep_prob=dropout_keep_prob)

        next_encoding = self.combiner(inputs=tf.concat([out_encoding, in_encoding], axis=-1),
                                      dropout_keep_prob=dropout_keep_prob)

        node_encoding = self.node_gru(inputs=next_encoding,
                                      state=node_states,
                                      dropout_keep_prob=dropout_keep_prob)

        return node_encoding


class GAT(Aggregator):

    def __init__(self, output_size, num_heads, activation, use_gru_gate, name):
        super(GAT, self).__init__(output_size, activation, name)
        self.num_heads = num_heads

        self.out_node_gat = AdjGAT(output_size=output_size,
                                   num_heads=num_heads,
                                   activation=activation,
                                   name='{0}-out-gat'.format(name))

        self.in_node_gat = AdjGAT(output_size=output_size,
                                  num_heads=num_heads,
                                  activation=activation,
                                  name='{0}-in-gat'.format(name))

        self.combiner = MLP(hidden_sizes=[],
                            output_size=output_size,
                            activation=activation,
                            activate_final=True,
                            bias_final=True,
                            name='{0}-combiner'.format(name))

        self.node_gru = GRU(output_size=output_size,
                            activation=activation,
                            name='{0}-node-gat-gru'.format(name))

        self.use_gru_gate = use_gru_gate


    def __call__(self, node_states, dropout_keep_prob, **kwargs):
        adj_lst = kwargs['adj_lst']
        inv_adj_lst = kwargs['inv_adj_lst']
        num_nodes = kwargs['num_nodes']

        # B x V x 1
        node_indices = tf.expand_dims(kwargs['node_indices'], axis=-1)

        out_neighbor_indices = tf.concat([adj_lst, node_indices], axis=-1)
        out_encoding = self.out_node_gat(inputs=node_states,
                                     adj_lst=out_neighbor_indices,
                                     mask_index=num_nodes,
                                     weight_dropout_keep=dropout_keep_prob,
                                     attn_dropout_keep=dropout_keep_prob)

        in_neighbor_indices = tf.concat([inv_adj_lst, node_indices], axis=-1)
        in_encoding = self.in_node_gat(inputs=node_states,
                                       adj_lst=in_neighbor_indices,
                                       mask_index=num_nodes,
                                       weight_dropout_keep=dropout_keep_prob,
                                       attn_dropout_keep=dropout_keep_prob)

        # Combine encodings from forward and backward edges
        next_encoding = self.combiner(inputs=tf.concat([out_encoding, in_encoding], axis=-1),
                                      dropout_keep_prob=dropout_keep_prob)

        if self.use_gru_gate:
            return self.node_gru(inputs=next_encoding,
                                 state=node_states,
                                 dropout_keep_prob=dropout_keep_prob)

        return next_encoding


class GGNN(Aggregator):

    def __init__(self, output_size, activation, name):
        super(GGNN, self).__init__(output_size, activation, name)

        self.node_gru = GRU(output_size=output_size,
                            activation=activation,
                            name='{0}-node-gat-gru'.format(name))


    def __call__(self, node_states, dropout_keep_prob, **kwargs):
        adj_lst = kwargs['adj_lst']
        inv_adj_lst = kwargs['inv_adj_lst']
        num_nodes = kwargs['num_nodes']

        # B x V x D x K
        out_neighbor_states, _ = masked_gather(values=node_states,
                                               indices=adj_lst,
                                               mask_index=num_nodes,
                                               set_zero=True,
                                               name='{0}-out-masked-gather'.format(self.name))

        # B x V x D x K
        in_neighbor_states, _ = masked_gather(values=node_states,
                                              indices=adj_lst,
                                              mask_index=num_nodes,
                                              set_zero=True,
                                              name='{0}-in-masked-gather'.format(self.name))

        combined_states = tf.reduce_sum(out_neighbor_states, axis=-2) + \
                          tf.reduce_sum(in_neighbor_states, axis=-2) + \
                          node_states

        return self.node_gru(inputs=combined_states,
                             state=node_states,
                             dropout_keep_prob=dropout_keep_prob)
