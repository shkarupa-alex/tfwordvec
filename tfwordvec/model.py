from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding, Dense, Multiply
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.layers.preprocessing.reduction import Reduction
from tfmiss.keras.layers import AdaptiveEmbedding, AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax, L2Scale


def build_model(h_params, unit_vocab, label_vocab):
    top_labels, _ = label_vocab.split_by_frequency(h_params.label_freq)
    num_labels = len(top_labels)

    inputs = _encoder_inputs(h_params)
    if 'sm' != h_params.model_head:
        inputs['labels'] = Input(shape=(None,), dtype=tf.int64)

    encoder = _build_encoder(h_params, unit_vocab)
    outputs = encoder(inputs)

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
        head = Dense(num_labels, activation='softmax')
        outputs = head(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model, encoder


def _build_encoder(h_params, unit_vocab):
    inputs = _encoder_inputs(h_params)

    outputs = _encoder_vectorization(h_params, unit_vocab)(inputs['inputs'])
    outputs = _encoder_embedding(h_params, 999)(outputs)

    if 'ngram' == h_params.input_unit:
        outputs = Reduction(h_params.ngram_comb)(outputs)

    if 'cbowpos' == h_params.vect_model:
        positions = Embedding(h_params.window_size * 2 - 1, h_params.embed_size)(inputs['positions'])
        multiply = Multiply()
        outputs = multiply([outputs, positions])

    if h_params.vect_model in {'cbow', 'cbowpos'}:
        outputs = Reduction('mean')(outputs)

    if h_params.l2_scale > 0.:
        outputs = L2Scale(h_params.l2_scale)(outputs)

    return Model(inputs=inputs, outputs=outputs)


def _encoder_inputs(h_params):
    shape_dims = 'skipgram' != h_params.vect_model + 'ngram' == h_params.input_unit
    inputs = {
        'inputs': Input(
            shape=(None,) * shape_dims,
            dtype=tf.string,
            ragged='skipgram' != h_params.vect_model)
    }

    if 'cbowpos' == h_params.vect_model:
        inputs['positions'] = Input(shape=(None,), dtype=tf.int32, ragged=True)

    return inputs


def _encoder_vectorization(h_params, unit_vocab):
    unit_top, _ = unit_vocab.split_by_freq(h_params.unit_freq)
    unit_keys = unit_top.tokens()
    vectorization = TextVectorization(
        max_tokens=len(unit_keys), standardize=None, split=None, pad_to_max_tokens=False)
    vectorization.set_vocabulary(unit_keys)

    return vectorization


def _encoder_embedding(h_params, input_dim):
    if 'adapt' == h_params.embed_type:
        return AdaptiveEmbedding(h_params.aemb_cutoff, input_dim, h_params.embed_size, h_params.aemb_factor)
    else:
        with tf.device('/CPU:0'):
            return Embedding(input_dim, h_params.embed_size)
