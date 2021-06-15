import argparse
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


def export_vectors(data_path, h_params, model_path):
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
    assert os.path.exists(argv.data_path) and os.path.isdir(argv.data_path), 'Wrong data path'
    assert os.path.exists(argv.model_path) and os.path.isdir(argv.model_path), 'Wrong model path'

    unit_path = os.path.join(argv.model_path, 'unit_encoder')
    assert os.path.exists(unit_path) and os.path.isdir(unit_path), \
        'Unit encoder not found. Did you export model to TFHub?'

    params_path = argv.params_path.name
    argv.params_path.close()
    h_params = build_hparams(params_path)

    tf.get_logger().setLevel(logging.INFO)

    export_vectors(argv.data_path, h_params, argv.model_path)
