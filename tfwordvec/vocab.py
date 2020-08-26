from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from .input import vocab_dataset
from .hparam import build_hparams
from .layer import ExpandNgams


def extract_vocab(data_path, h_params):
    if not tf.executing_eagerly():
        raise EnvironmentError('Eager mode should be enabled by default')

    label_vocab = Vocabulary()
    dataset = vocab_dataset(data_path, h_params)

    for labels in dataset:
        labels = _unicode_tensor(labels.flat_values)
        label_vocab.update(labels)

    unit_vocab = Vocabulary()
    if 'ngram' == h_params.input_unit:
        labels = tf.constant(label_vocab.tokens(), dtype=tf.string)
        ngrams = ExpandNgams(
            ngram_minn=h_params.ngram_minn,
            ngram_maxn=h_params.ngram_maxn,
            ngram_self=h_params.ngram_self)(labels)
        for label, ngram in zip(label_vocab.tokens(), ngrams):
            ngram = _unicode_tensor(ngram).reshape([-1])
            for n in ngram:
                unit_vocab[n] += label_vocab[label]
    else:
        unit_vocab.update(label_vocab)

    return unit_vocab, label_vocab


def vocab_names(data_path, h_params, format=Vocabulary.FORMAT_BINARY_PICKLE):
    model_name = h_params.vect_model
    if 'cbowpos' == h_params.vect_model:
        model_name = 'cbow'

    ext = 'pkl' if Vocabulary.FORMAT_BINARY_PICKLE == format else 'tsv'

    unit_vocab = 'vocab_{}_{}_unit.{}'.format(model_name, h_params.input_unit, ext)
    label_vocab = 'vocab_{}_{}_label.{}'.format(model_name, h_params.input_unit, ext)

    return os.path.join(data_path, unit_vocab), os.path.join(data_path, label_vocab)


def _unicode_tensor(tensor):
    return np.char.decode(tensor.numpy().astype('S'), 'utf-8')


def main():
    parser = argparse.ArgumentParser(description='Extract vocabulary from dataset')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON file with hyper parameters')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to train .txt.gz files')

    argv, _ = parser.parse_known_args()
    if not os.path.exists(argv.data_path) or not os.path.isdir(argv.data_path):
        raise ValueError('Wrong train dataset path')

    tf.get_logger().setLevel(logging.INFO)

    h_params = build_hparams(json.loads(argv.hyper_params.read()))

    tf.get_logger().info('Estimating {} and label vocabularies'.format(h_params.input_unit))
    unit_vocab, label_vocab = extract_vocab(argv.data_path, h_params)

    tf.get_logger().info('Saving vocabularies to {}'.format(argv.data_path))
    unit_pkl, label_pkl = vocab_names(argv.data_path, h_params)
    unit_vocab.save(unit_pkl)
    label_vocab.save(label_pkl)

    unit_tsv, label_tsv = vocab_names(argv.data_path, h_params, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    unit_vocab.save(unit_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    label_vocab.save(label_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)

    if 'dense' == h_params.embed_type:
        unit1k_vocab, _ = unit_vocab.split_by_size(1000)
        unit1k_tsv = unit_tsv[:-4] + '1k' + unit_tsv[-4:]
        unit1k_vocab.save(unit1k_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)

    tf.get_logger().info('Vocabularies saved to {}'.format(argv.data_path))
