#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: model.py
# =====================================

import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention
import numpy as np

from model_utils import positional_encoding, EncoderLayer

tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class MLPNet(Model):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(MLPNet, self).__init__(name=kwargs['name'])
        self.first_ = Dense(num_hidden_units,
                            activation=hidden_activation,
                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                            dtype=tf.float32)
        self.hidden = Sequential([Dense(num_hidden_units,
                                        activation=hidden_activation,
                                        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                        dtype=tf.float32) for _ in range(num_hidden_layers-1)])
        output_activation = kwargs['output_activation'] if kwargs.get('output_activation') else 'linear'
        if kwargs.get('output_bias'):
            self.outputs = Dense(output_dim,
                                 activation=output_activation,
                                 kernel_initializer=tf.keras.initializers.Orthogonal(1.),
                                 bias_initializer=tf.keras.initializers.Constant(kwargs.get('output_bias')),
                                 dtype=tf.float32)
        else:
            self.outputs = Dense(output_dim,
                                 activation=output_activation,
                                 kernel_initializer=tf.keras.initializers.Orthogonal(1.),
                                 bias_initializer=tf.keras.initializers.Constant(0.),
                                 dtype=tf.float32)
        self.build(input_shape=(None, input_dim))

    def call(self, x, **kwargs):
        x = self.first_(x)
        x = self.hidden(x)
        x = self.outputs(x)
        return x


class AlphaModel(Model):
    def __init__(self, **kwargs):
        super(AlphaModel, self).__init__(name=kwargs['name'])
        self.log_alpha = tf.Variable(0., dtype=tf.float32)

class LamModel(Model):
    def __init__(self, **kwargs):
        super(LamModel, self).__init__(name=kwargs['name'])
        self.var = tf.Variable(-10., dtype=tf.float32)


class AttnNet(Model):
    def __init__(self, ego_dim, con_dim, max_seq_len,
                 num_attn_layers, d_model, d_ff, num_heads, dropout, **kwargs):
        super(AttnNet, self).__init__(name=kwargs['name'])

#        obs_dim = kwargs.get('obs_dim')
#        assert obs_dim == ego_dim + con_dim * (max_seq_len - 1), print(obs_dim, ego_dim, con_dim, max_seq_len)
        self.ego_dim = ego_dim
        self.con_dim = con_dim
        self.max_seq_len = max_seq_len

        self.num_layers = num_attn_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.ego_embedding = Sequential([Dense(units=d_ff,
                                               kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                               activation='elu',
                                               dtype=tf.float32),
                                         Dense(d_model)])
        self.cons_embedding = Sequential([Dense(units=d_ff,
                                               kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                               activation='elu',
                                               dtype=tf.float32),
                                          Dense(d_model)])

        self.pe = positional_encoding(max_seq_len, d_model)
        self.dropout = Dropout(self.dropout_rate)

        self.attn_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout)
                            for _ in range(self.num_layers-1)]
        self.out_attn = MultiHeadAttention(1, d_model, 
                                           kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                           dropout=dropout)
        self.build(input_shape=[(None, 1, ego_dim), (None, max_seq_len-1, con_dim),
                                (None, max_seq_len, max_seq_len), (None, max_seq_len, max_seq_len)])


    def call(self, input, **kwargs):
        '''
        return
        :x [B, T, d_model]
        :weights [B, 1, T, T]
        '''
        training = kwargs.get('training')
        x_ego, x_cons, padding_mask, mu_mask = input[0], input[1], input[2], input[3]
        assert x_ego.shape[2] == self.ego_dim
        assert x_cons.shape[2] == self.con_dim
        assert x_cons.shape[1] == self.max_seq_len-1

        x1 = self.ego_embedding(x_ego)
        x2 = self.cons_embedding(x_cons)
        x = tf.concat([x1, x2], axis=1)
        assert x.shape[1] == self.max_seq_len
        x += self.pe[:, :self.max_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers-1):
            x = self.attn_layers[i](x, training, padding_mask)

        output_mask = tf.minimum(padding_mask, mu_mask)
        x, attn_weights = self.out_attn(x, x, attention_mask=output_mask,
                                        return_attention_scores=True, training=training)

        return x, attn_weights


def test_alpha():
    import numpy as np
    alpha_model = AlphaModel(name='alpha')
    print(alpha_model.trainable_weights)
    print(len(alpha_model.trainable_weights))
    print(alpha_model.get_weights())
    print(alpha_model.log_alpha)
    b = alpha_model.log_alpha
    alpha_model.set_weights(np.array([3]))
    print(b)

    with tf.GradientTape() as tape:
        b = 3.*alpha_model.log_alpha
    print(tape.gradient(b, alpha_model.trainable_weights[0]))

def test_lam():
    import numpy as np




    with tf.GradientTape() as tape:
        lam_model = LamModel(name='lam')
        print(lam_model.trainable_weights)
        print(len(lam_model.trainable_weights))
        print(lam_model.get_weights())
        print(lam_model.log_lam)
        b = lam_model.log_lam
        lam_model.set_weights(np.array([3]))
        print(b)
        c = 3.*lam_model.log_lam
    print(tape.gradient(c, lam_model.trainable_weights[0]))


def test_attrib():
    import numpy as np

    a = Variable(0, name='d')

    p = MLPNet(2, 2, 128, 1, name='ttt')
    print(hasattr(p, 'get_weights'))
    print(hasattr(p, 'trainable_weights'))
    print(hasattr(a, 'get_weights'))
    print(hasattr(a, 'trainable_weights'))
    print(type(a))
    print(type(p))
    # print(a.name)
    # print(p.name)
    # p.build((None, 2))
    p.summary()
    # inp = np.random.random([10, 2])
    # out = p.forward(inp)
    # print(p.get_weights())
    # print(p.trainable_weights)


def test_clone():
    p = MLPNet(2, 2, 128, 1, name='ttt')
    print(p._is_graph_network)
    s = tf.keras.models.clone_model(p)
    print(s)


def test_out():
    import numpy as np
    Qs = tuple(MLPNet(8, 2, 128, 1, name='Q' + str(i)) for i in range(2))
    inp = np.random.random((128, 8))
    out = [Q(inp) for Q in Qs]
    print(out)


def test_memory():
    import time
    Q = MLPNet(8, 2, 128, 1)
    time.sleep(111111)


def test_memory2():
    import time
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(30,), activation='relu'),
                                 tf.keras.layers.Dense(20, activation='relu'),
                                 tf.keras.layers.Dense(20, activation='relu'),
                                 tf.keras.layers.Dense(10, activation='relu')])
    time.sleep(10000)


if __name__ == '__main__':
    test_lam()
