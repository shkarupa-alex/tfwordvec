import os
import tensorflow as tf
from nlpvocab import Vocabulary
from .ops import skip_gram, cont_bow, down_sample, normalize_unicode, zero_digits, split_chars, lower_case
from tfmiss.training import estimate_bucket_pipeline
from .config import BOS_MARK, EOS_MARK, UNK_MARK, InputUnit, VectModel, ModelHead
from .model import _unit_embedder


def train_dataset(src_path, config, unit_vocab, label_vocab):
    unit_embedder = _unit_embedder(config, unit_vocab, with_prep=False)
    label_table, label_last = _label_lookup(label_vocab, config)

    def _pre_transform(sentences):
        units = _transform_split(sentences, config)
        units = down_sample(
            source=units,
            freq_vocab=label_vocab,
            threshold=config.samp_thold,
            min_freq=config.label_freq)
        features, labels = _transform_model(units, config)

        labels = label_table.lookup(labels)
        features['filters'] = tf.not_equal(labels, label_last)

        return features, labels

    def _post_transform(features, labels):
        features.pop('filters', None)
        features.pop('lengths', None)

        features['units'] = unit_embedder.preprocess(features['units'])

        if ModelHead.SOFTMAX == config.model_head:
            return features, labels

        features['labels'] = labels

        return features

    dataset = _raw_dataset(src_path, config)
    dataset = dataset.map(_pre_transform, tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.unbatch()
    dataset = dataset.filter(lambda features, *args: features['filters'])

    if config.vect_model in {VectModel.CBOW, VectModel.CBOWPOS} and config.bucket_cbow:
        buck_bounds = list(range(2, config.window_size * 2 + 2))
        buck_bounds, batch_sizes, _ = estimate_bucket_pipeline(buck_bounds, config.batch_size, safe=False)
        dataset = dataset.bucket_by_sequence_length(
            lambda features, *args: features['lengths'],
            buck_bounds,
            batch_sizes,
            no_padding=True)
    else:
        dataset = dataset.batch(config.batch_size)

    dataset = dataset.map(_post_transform, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def vocab_dataset(src_path, config):
    def _transform(sentences):
        return _transform_split(sentences, config)

    dataset = _raw_dataset(src_path, config)
    dataset = dataset.map(_transform, tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _raw_dataset(src_path, config):
    wild_card = os.path.join(src_path, '*.txt.gz')
    fileset = tf.data.Dataset.list_files(wild_card)

    dataset = fileset.interleave(
        lambda gz_file: _line_datset(gz_file, config),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.filter(lambda sentence: tf.strings.length(tf.strings.strip(sentence)) > 0)
    dataset = dataset.batch(config.batch_size)

    return dataset


def _line_datset(gz_file, config):
    dataset = tf.data.TextLineDataset(gz_file, 'GZIP', None, tf.data.AUTOTUNE)

    if InputUnit.CHAR == config.input_unit:
        dataset = dataset.batch(2)
        dataset = dataset.map(
            lambda rows: tf.strings.reduce_join(rows, separator='\n'),
            num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def _transform_split(sentences, config):
    sentences = normalize_unicode(sentences, 'NFKC')
    if InputUnit.CHAR != config.input_unit:
        sentences = tf.strings.regex_replace(sentences, r'\s+', ' ')
    sentences = tf.strings.strip(sentences)

    if config.lower_case:
        sentences = lower_case(sentences)
    if config.zero_digits:
        sentences = zero_digits(sentences)

    if InputUnit.CHAR == config.input_unit:
        units = split_chars(sentences)
    else:
        units = tf.strings.split(sentences, sep=' ')

    # Same preprocessing should be in CbowContext
    bos = tf.fill([units.nrows(), 1], BOS_MARK)
    eos = tf.fill([units.nrows(), 1], EOS_MARK)
    units = tf.concat([bos, units, eos], axis=1)

    return units


def _transform_model(units, config):
    features = {}
    if config.vect_model in {VectModel.CBOW, VectModel.CBOWPOS}:
        targets, contexts, positions = cont_bow(units, config.window_size)

        if config.bucket_cbow:
            features['lengths'] = tf.cast(contexts.row_lengths(), 'int32')

        if VectModel.CBOWPOS == config.vect_model:
            positions = tf.ragged.map_flat_values(
                lambda flat: tf.where(
                    tf.greater(flat, 0),
                    flat - 1,
                    flat
                ) + config.window_size,
                positions)
            features['positions'] = positions
    else:
        targets, contexts = skip_gram(units, config.window_size)

    features['units'] = contexts

    return features, targets


def _label_lookup(label_vocab, config):
    assert isinstance(label_vocab, Vocabulary), 'Wrong label vocabulary type'

    top, _ = label_vocab.split_by_frequency(config.label_freq)
    keys = top.tokens() + [UNK_MARK]
    values = tf.range(len(keys), dtype='int64')
    last = len(keys) - 1
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
        keys=keys, values=values, key_dtype='string'), last)

    return table, last
