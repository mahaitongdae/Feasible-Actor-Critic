#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/6/3
# @Author  : Dongjie Yu (Tsinghua Univ.)
# @FileName: model_utils.py
# =====================================

import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization

import numpy as np
import math

tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


def get_angles(pos, i, d_model, max_len=1000):
    angle_rates = 1 / np.power(max_len, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    p = np.ones((position, 1))
    p[0] = 0
    angle_rads = get_angles(p,
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def pointwise_feedforward(d_model, d_ff):
    return Sequential([
        Dense(d_ff, activation='elu'),
        Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_heads, d_model,
                                      kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                      dropout=dropout)
        self.ffn = pointwise_feedforward(d_model, d_ff)

        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        ## post-LN
        # attn_output = self.mha(x, x, x, mask, training=training)  # (batch_size, input_seq_len, d_model)
        # out1 = self.ln1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        # ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        # ffn_output = self.dropout(ffn_output, training=training)
        # out2 = self.ln2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        ## pre-LN
        x = self.ln1(x)
        out1 = x + self.mha(x, x, x, mask, training=training)

        out1 = self.ln2(out1)
        out2 = self.dropout(out1 + self.ffn(out1), training=training)

        return out2
