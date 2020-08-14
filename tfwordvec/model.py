from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation, Embedding, Dense
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tfmiss.keras.layers import AdaptiveEmbedding, AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax, L2Scale
from .layer import Reduction


def build_model(h_params, unit_vocab, label_vocab):
    top_labels, _ = label_vocab.split_by_frequency(h_params.label_freq)
    num_labels = len(top_labels)

    inputs = _encoder_inputs(h_params)
    encoder = _build_encoder(h_params, unit_vocab)
    outputs = encoder(inputs)

    if 'sm' != h_params.model_head:
        # Add labels input after encoder call to bypass "ignoring input" warning
        inputs['labels'] = Input(name='labels', shape=(), dtype=tf.int32)

    if 'ss' == h_params.model_head:
        head = SampledSofmax(num_labels, h_params.neg_samp)
        outputs = head([outputs, inputs['labels']])
    elif 'nce' == h_params.model_head:
        head = NoiseContrastiveEstimation(num_labels, h_params.neg_samp)
        outputs = head([outputs, inputs['labels']])
    elif 'asm' == h_params.model_head:
        head = AdaptiveSoftmax(num_labels, h_params.asm_cutoff, h_params.asm_factor, h_params.asm_drop)
        outputs = head([outputs, inputs['labels']])
    else:  # 'sm' == h_params.model_head:
        outputs = Dense(num_labels, name='logits')(outputs)
        outputs = Activation('softmax', dtype=tf.float32)(outputs)

    model = Model(inputs=list(inputs.values()), outputs=outputs)

    return model, encoder


def _build_encoder(h_params, unit_vocab):
    inputs = _encoder_inputs(h_params)

    vectorization, embed_dim = _encoder_vectorization(h_params, unit_vocab)
    outputs = vectorization(inputs['inputs'])
    outputs = _encoder_embedding(h_params, embed_dim)(outputs)

    if 'ngram' == h_params.input_unit:
        outputs = Reduction(h_params.ngram_comb, name='ngrams')(outputs)

    if 'cbowpos' == h_params.vect_model:
        positions = Embedding(h_params.window_size * 2, h_params.embed_size)(inputs['positions'])
        outputs = tf.keras.layers.multiply([outputs, positions], name='aligns')

    if h_params.vect_model in {'cbow', 'cbowpos'}:
        outputs = Reduction('mean', name='contexts')(outputs)

    if h_params.l2_scale >= 1.:
        outputs = L2Scale(h_params.l2_scale)(outputs)

    return Model(inputs=list(inputs.values()), outputs=outputs, name='encoder')


def _encoder_inputs(h_params):
    shape_dims = (h_params.vect_model in {'cbow', 'cbowpos'}) + int('ngram' == h_params.input_unit)
    inputs = {
        'inputs': Input(
            name='inputs',
            shape=(None,) * shape_dims,
            dtype=tf.string,
            ragged='skipgram' != h_params.vect_model)
    }

    if 'cbowpos' == h_params.vect_model:
        inputs['positions'] = Input(name='positions', shape=(None,), dtype=tf.int32, ragged=True)

    return inputs


def _encoder_vectorization(h_params, unit_vocab):
    unit_top, _ = unit_vocab.split_by_frequency(h_params.unit_freq)
    unit_keys = unit_top.tokens()
    vectorization = StringLookup(vocabulary=unit_keys, mask_token=None, name='lookup')

    return vectorization, len(unit_keys) + 1  # + [UNK]


def _encoder_embedding(h_params, input_dim):
    if 'adapt' == h_params.embed_type:
        return AdaptiveEmbedding(h_params.aemb_cutoff, input_dim, h_params.embed_size, h_params.aemb_factor)
    else:
        with tf.device('/CPU:0'):
            return Embedding(input_dim, h_params.embed_size)
