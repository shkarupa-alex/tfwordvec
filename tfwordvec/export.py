from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import tensorflow as tf
from gensim.models.keyedvectors import Vocab
from gensim.models.utils_any2vec import _save_word2vec_format
from nlpvocab import Vocabulary
from tensorflow_hub import KerasLayer
from .hparam import build_hparams
from .input import RESERVED
from .vocab import vocab_names


def export_vectors(data_path, params_path, model_path):
    with open(params_path, 'r') as f:
        h_params = build_hparams(json.loads(f.read()))

    unit_path, _ = vocab_names(data_path, h_params)
    unit_vocab = Vocabulary.load(unit_path)
    unit_top, _ = unit_vocab.split_by_frequency(h_params.unit_freq)

    units = RESERVED + [u for u in unit_top.tokens() if u not in RESERVED]
    if 'ngram' == h_params.input_unit:
        units = [u[1:-1] for u in units if u.startswith('<') and u.endswith('>')]

    embed = KerasLayer(os.path.join(model_path, 'unit_encoder'))
    vectors = embed(units).numpy()
    vocab = {u: Vocab(index=i, count=len(units) - i) for i, u in enumerate(units)}

    _save_word2vec_format(os.path.join(model_path, 'unit_encoder.bin'), vocab, vectors, binary=True)
    tf.get_logger().info('Unit vectors saved to {}'.format(os.path.join(model_path, 'unit_encoder.bin')))


def main():
    parser = argparse.ArgumentParser(description='Word2Vec encoder exporter')
    parser.add_argument(
        'params_path',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparameters file')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to source .txt.gz documents with one sentence per line')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to save model')

    argv, _ = parser.parse_known_args()
    if not os.path.exists(argv.data_path) or not os.path.isdir(argv.data_path):
        raise IOError('Wrong data path')
    if not os.path.exists(argv.model_path) or not os.path.isdir(argv.model_path):
        raise IOError('Wrong model path')

    params_path = argv.params_path.name
    argv.params_path.close()

    tf.get_logger().setLevel(logging.INFO)

    export_vectors(argv.data_path, params_path, argv.model_path)
