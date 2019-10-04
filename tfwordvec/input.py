from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.preprocessing import skip_gram, cont_bow, down_sample
from tfmiss.text import zero_digits, lower_case, split_chars, wrap_with, char_ngrams
from tfmiss.training import estimate_bucket_pipeline


def normalize_text(strings):
    pass


def transform_train_input(sentences, params, freq_tokens, freq_values):
    if params.zero_digits:
        sentences = zero_digits(sentences)

    multi_char = 'char' != params.model_unit
    if multi_char:
        items = tf.strings.split(sentences, sep=' ')
    else:  # chars
        _sentences = tf.strings.join(sentences, '\n')
        items = split_chars(_sentences)

    if freq_tokens is not None and freq_values is not None:
        items = down_sample(items, freq_tokens, freq_values, threshold=params.sample_thold, min_freq=params.min_freq)

    if 'skipgram' == params.vector_model:
        target, context, position = cont_bow(items, params.window_size)
        position = tf.strings.as_string(position)
        length = context.row_lengths()

        features = {
            'source': context,
            'position': position,
            'length': length
        }
        labels = target
    else:
        target, context = skip_gram(items, params.window_size)
        features = {'source': target}
        labels = context

    if 'lower' == params.label_case:
        labels = lower_case(labels)

    if 'extract' == params.feat_case:
        cases = extract_case(context, title=multi_char, mixed=multi_char)
        features.update(cases)

    if 'same' != params.feat_case:
        features['source'] = lower_case(features['source'])

    if 'ngram' == params.model_unit:
        features['source'] = wrap_with(features['source'], '<', '>')
        features['source'] = char_ngrams(features['source'], params.ngram_minn, params.ngram_maxn, params.ngram_itself)

    return features, labels


def train_input_fn(wildcard, params, freq_vocab=None):
    def _transform_input(sentences):
        if freq_vocab is None:
            freq_tokens, freq_values = None, None
        else:
            # Required for downsampling. Should not contain <UNK> token.
            freq_tokens, freq_values = zip(*freq_vocab.most_common())

        return transform_train_input(sentences, params, freq_tokens, freq_values)

    dataset = tf.data.Dataset.list_files(wildcard)
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename, compression_type='GZIP'),
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.map(_transform_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.unbatch())

    if freq_vocab is not None:  # training
        dataset = dataset.shuffle(params.batch_size * 1000)
        dataset = dataset.repeat(params.num_repeats)

    if 'cbow' == params.vector_model and params.bucket_cbow:
        buck_bounds = list(range(2, params.window_size * 2 + 2))
        buck_bounds, batch_sizes, _ = estimate_bucket_pipeline(buck_bounds, params.batch_size, safe=False)

        def _seq_len(features, labels):
            return features['length']

        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            _seq_len,
            buck_bounds,
            batch_sizes
        ))
    else:
        dataset = dataset.batch(params.batch_size)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
