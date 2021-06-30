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


def export_vectors(vocab_path, params_path, model_path):
    h_params = build_hparams(params_path)

    unit_vocab = Vocabulary.load(vocab_path)
    unit_top, _ = unit_vocab.split_by_frequency(h_params.unit_freq)

    units = RESERVED + [u for u in unit_top.tokens() if u not in RESERVED]
    embed = KerasLayer(os.path.join(model_path, 'unit_encoder'))
    vectors = embed(units).numpy()
    vocab = {u: Vocab(index=i, count=len(units) - i) for i, u in enumerate(units)}

    _save_word2vec_format(os.path.join(model_path, 'unit_encoder.bin'), vocab, vectors, binary=True)
    tf.get_logger().info('Unit vectors saved to {}'.format(os.path.join(model_path, 'unit_encoder.bin')))


def main():
    parser = argparse.ArgumentParser(description='Word2Vec encoder exporter')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparameters file')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to save model')
    parser.add_argument(
        'vocab_path',
        type=argparse.FileType('rb'),
        help='Path to vocabulary')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.model_path) and os.path.isdir(argv.model_path), 'Wrong model path'

    unit_path = os.path.join(argv.model_path, 'unit_encoder')
    assert os.path.exists(unit_path) and os.path.isdir(unit_path), \
        'Unit encoder not found. Did you export model to TFHub?'

    params_path = argv.hyper_params.name
    argv.hyper_params.close()

    vocab_path = argv.vocab_path.name
    argv.vocab_path.close()

    tf.get_logger().setLevel(logging.INFO)

    export_vectors(vocab_path, params_path, argv.model_path)
