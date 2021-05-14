from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation, Dense, Embedding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tfmiss.keras.layers import AdaptiveEmbedding, AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax
from tfmiss.keras.layers import CharNgams, L2Scale, Reduction
from .input import UNK_MARK, RESERVED
from .layer import MapFlat


def build_model(h_params, unit_vocab, label_vocab):
    top_labels, _ = label_vocab.split_by_frequency(h_params.label_freq)
    num_labels = len(top_labels)

    inputs = _context_inputs(h_params)
    encoder = _build_encoder(h_params, unit_vocab)
    logits = encoder(inputs)

    if 'sm' != h_params.model_head:
        # Add labels input after encoder call to bypass "ignoring input" warning
        inputs['labels'] = Input(name='labels', shape=(), dtype=tf.int32)

    if 'ss' == h_params.model_head:
        head = SampledSofmax(num_labels, h_params.neg_samp)
        probs = head([logits, inputs['labels']])
    elif 'nce' == h_params.model_head:
        head = NoiseContrastiveEstimation(num_labels, h_params.neg_samp)
        probs = head([logits, inputs['labels']])
    elif 'asm' == h_params.model_head:
        head = AdaptiveSoftmax(num_labels, h_params.asm_cutoff, h_params.asm_factor, h_params.asm_drop)
        probs = head([logits, inputs['labels']])
    else:  # 'sm' == h_params.model_head:
        probs = Dense(num_labels, name='logits')(logits)
        probs = Activation('softmax', dtype=tf.float32)(probs)

    model = Model(inputs=list(inputs.values()), outputs=probs, name='trainer')

    return model


def _build_encoder(h_params, unit_vocab):
    inputs = _context_inputs(h_params)

    encoder = _unit_encoder(h_params, unit_vocab)
    embeddings = MapFlat(encoder, name='unit_encoding')(inputs['units'])

    if 'cbowpos' == h_params.vect_model:
        positions = Embedding(
            input_dim=h_params.window_size * 2,
            output_dim=h_params.embed_size,
            name='position_embedding')(inputs['positions'])
        embeddings = tf.keras.layers.multiply([embeddings, positions], name='position_encoding')

    if h_params.vect_model in {'cbow', 'cbowpos'}:
        embeddings = Reduction('mean', name='context_reduction')(embeddings)

    return Model(inputs=list(inputs.values()), outputs=embeddings, name='context_encoder')


def _context_inputs(h_params):
    has_context = int(h_params.vect_model in {'cbow', 'cbowpos'})
    inputs = {
        'units': Input(
            name='units',
            shape=(None,) * int(has_context),
            dtype=tf.string,
            ragged=has_context)
    }

    if 'cbowpos' == h_params.vect_model:
        inputs['positions'] = Input(name='positions', shape=(None,), dtype=tf.int32, ragged=True)

    return inputs


def _unit_encoder(h_params, unit_vocab):
    inputs = Input(name='units', shape=(), dtype=tf.string)

    units = inputs
    if 'ngram' == h_params.input_unit:
        units = CharNgams(
            minn=h_params.ngram_minn,
            maxn=h_params.ngram_maxn,
            itself=h_params.ngram_self,
            reserved=RESERVED,
            name='ngram_expansion')(units)

    unit_top, _ = unit_vocab.split_by_frequency(h_params.unit_freq)
    unit_keys = unit_top.tokens()
    lookup = StringLookup(
        vocabulary=unit_keys,
        mask_token=None,
        oov_token=UNK_MARK,
        name='unit_lookup')
    indexes = lookup(units)

    if 'adapt' == h_params.embed_type:
        embed = AdaptiveEmbedding(
            cutoff=h_params.aemb_cutoff,
            input_dim=lookup.vocabulary_size(),
            output_dim=h_params.embed_size,
            factor=h_params.aemb_factor,
            name='unit_embedding')
    else:
        with tf.device('/CPU:0'):
            embed = Embedding(
                input_dim=lookup.vocabulary_size(),
                output_dim=h_params.embed_size,
                name='unit_embedding')
    embeddings = embed(indexes)

    if 'ngram' == h_params.input_unit:
        embeddings = Reduction(h_params.ngram_comb, name='ngram_reduction')(embeddings)

    if h_params.l2_scale > 1.:
        embeddings = L2Scale(h_params.l2_scale, name='scale')(embeddings)

    return Model(inputs=inputs, outputs=embeddings, name='unit_encoder')
