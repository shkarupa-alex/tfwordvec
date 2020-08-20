from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tfmiss.preprocessing import skip_gram, cont_bow, down_sample
from tfmiss.text import normalize_unicode, zero_digits, split_chars, lower_case
from tfmiss.training import estimate_bucket_pipeline

BOS_MARK = '[BOS]'
EOS_MARK = '[EOS]'
UNK_MARK = '[UNK]'
RESERVED = [BOS_MARK, EOS_MARK, UNK_MARK]


def train_dataset(src_path, h_params, label_vocab):
    label_table, label_last = _label_lookup(label_vocab, h_params)

    def _pre_transform(sentences):
        units = _transform_split(sentences, h_params)
        units = down_sample(
            source=units,
            freq_vocab=label_vocab,
            threshold=h_params.samp_thold,
            min_freq=h_params.label_freq)
        features, labels = _transform_model(units, h_params)

        labels = label_table.lookup(labels)
        features['filters'] = tf.not_equal(labels, label_last)

        return features, labels

    def _post_transform(features, labels):
        features.pop('filters', None)
        features.pop('lengths', None)

        if 'sm' == h_params.model_head:
            return features, labels

        features['labels'] = labels

        return features

    dataset = _raw_dataset(src_path, h_params)
    dataset = dataset.map(_pre_transform, tf.data.experimental.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.filter(lambda features, *args: features['filters'])

    if h_params.vect_model in {'cbow', 'cbowpos'} and h_params.bucket_cbow:
        buck_bounds = list(range(2, h_params.window_size * 2 + 2))
        buck_bounds, batch_sizes, _ = estimate_bucket_pipeline(buck_bounds, h_params.batch_size, safe=False)
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            lambda features, *args: features['lengths'],
            buck_bounds,
            batch_sizes,
            no_padding=True
        ))
    else:
        dataset = dataset.batch(h_params.batch_size)

    dataset = dataset.map(_post_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(100)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def vocab_dataset(src_path, h_params):
    def _transform(sentences):
        return _transform_split(sentences, h_params)

    dataset = _raw_dataset(src_path, h_params)
    dataset = dataset.map(_transform, tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def _raw_dataset(src_path, h_params):
    wild_card = os.path.join(src_path, '*.txt.gz')
    fileset = tf.data.Dataset.list_files(wild_card)

    dataset = fileset.interleave(
        lambda gz_file: _line_datset(gz_file, h_params),
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(h_params.batch_size)

    return dataset


def _line_datset(gz_file, h_params):
    dataset = tf.data.TextLineDataset(gz_file, 'GZIP', None, tf.data.experimental.AUTOTUNE)

    if 'char' == h_params.input_unit:
        dataset = dataset.batch(2)
        dataset = dataset.map(
            lambda rows: tf.strings.reduce_join(rows, separator='\n'),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def _transform_split(sentences, h_params):
    sentences = normalize_unicode(sentences, 'NFKC')
    if 'char' != h_params.input_unit:
        sentences = tf.strings.regex_replace(sentences, r'\s+', ' ')
    sentences = tf.strings.strip(sentences)
    if h_params.lower_case:
        sentences = lower_case(sentences)
    if h_params.zero_digits:
        sentences = zero_digits(sentences)

    if 'char' == h_params.input_unit:
        units = split_chars(sentences)
    else:
        units = tf.strings.split(sentences, sep=' ')

    bos = tf.fill([units.nrows(), 1], BOS_MARK)
    eos = tf.fill([units.nrows(), 1], EOS_MARK)
    units = tf.concat([bos, units, eos], axis=1)

    return units


def _transform_model(units, h_params):
    features = {}
    if h_params.vect_model in {'cbow', 'cbowpos'}:
        targets, contexts, positions = cont_bow(units, h_params.window_size)

        if h_params.bucket_cbow:
            features['lengths'] = tf.cast(contexts.row_lengths(), tf.int32)

        if 'cbowpos' == h_params.vect_model:
            positions = tf.ragged.map_flat_values(
                lambda flat: tf.where(
                    tf.greater(flat, 0),
                    flat - 1,
                    flat
                ) + h_params.window_size,
                positions)
            features['positions'] = positions
    else:
        targets, contexts = skip_gram(units, h_params.window_size)

    features['units'] = contexts

    return features, targets


def _label_lookup(label_vocab, h_params):
    if not isinstance(label_vocab, Vocabulary):
        raise ValueError('Wrong label vocabulary type')

    top, _ = label_vocab.split_by_frequency(h_params.label_freq)
    keys = top.tokens() + [UNK_MARK]
    values = tf.range(len(keys), dtype=tf.int64)
    last = len(keys) - 1
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
        keys=keys, values=values, key_dtype=tf.string), last)

    return table, last
