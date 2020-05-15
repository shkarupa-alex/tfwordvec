from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from .input import vocab_dataset
from .hparam import build_hparams


def extract_vocab(src_path, h_params):
    if not tf.executing_eagerly():
        raise EnvironmentError('Eager mode should be enabled by default')

    tf.get_logger().info('Window size will be set to 1 during vocabularies estimation')
    h_params.window_size = 1

    unit_vocab, label_vocab = Vocabulary(), Vocabulary()
    dataset = vocab_dataset(src_path, h_params)

    for features, labels in dataset:
        units = features['inputs']
        if isinstance(units, tf.RaggedTensor):
            units = units.flat_values
        units = _unicode_tensor(units).reshape([-1])
        unit_vocab.update(units)

        labels = _unicode_tensor(labels).reshape([-1])
        label_vocab.update(labels)

    return unit_vocab, label_vocab


def main():
    parser = argparse.ArgumentParser(description='Extract vocabulary from dataset')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON file with hyper parameters')
    parser.add_argument(
        'src_path',
        type=str,
        help='Path to train .txt.gz files')

    argv, _ = parser.parse_known_args()
    if not os.path.exists(argv.src_path) or not os.path.isdir(argv.src_path):
        raise ValueError('Wrong train dataset path')

    h_params = build_hparams(json.loads(argv.hyper_params.read()))

    tf.get_logger().info('Estimating {} and label vocabularies'.format(h_params.input_unit))
    unit_vocab, label_vocab = extract_vocab(argv.src_path, h_params)

    tf.get_logger().info('Saving vocabularies to {}'.format(argv.src_path))
    unit_pkl, label_pkl = vocab_names(argv.src_path, h_params)
    unit_vocab.save(unit_pkl)
    label_vocab.save(label_pkl)

    unit_tsv, label_tsv = vocab_names(argv.src_path, h_params, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    unit_vocab.save(unit_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    label_vocab.save(label_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)

    unit1k_vocab, _ = unit_vocab.split_by_size(1000)
    unit1k_tsv = unit_tsv[:-4] + '1k' + unit_tsv[-4:]
    unit1k_vocab.save(unit1k_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)


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
