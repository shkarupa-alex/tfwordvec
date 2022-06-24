import argparse
import logging
import math
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tensorflow_hub import KerasLayer
from .config import RESERVED, build_config


def export_vectors(vocab_path, params_path, model_path):
    config = build_config(params_path)

    unit_vocab = Vocabulary.load(vocab_path)
    unit_top, _ = unit_vocab.split_by_frequency(config.unit_freq)

    units = RESERVED + [u for u in unit_top.tokens() if u not in RESERVED]
    embed = KerasLayer(os.path.join(model_path, 'unit_encoder'))
    vectors = embed(units[:1]).numpy()

    with open(os.path.join(model_path, 'unit_encoder.bin'), 'wb') as fout:
        fout.write(f'{len(units)} {vectors.shape[1]}\n'.encode('utf-8'))

        step = 1000
        for i in range(math.ceil(len(units) / step)):
            _units = units[i * step:(i + 1) * step]
            _vectors = embed(_units).numpy()

            # store in sorted order: most frequent words at the top
            for j in range(len(_units)):
                word = _units[j]
                vector = _vectors[j].astype('float32')
                fout.write(f'{word} '.encode('utf-8') + vector.tobytes())

    tf.get_logger().info('Unit vectors saved to {}'.format(os.path.join(model_path, 'unit_encoder.bin')))


def main():
    parser = argparse.ArgumentParser(description='Word2Vec encoder exporter')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='YAML-encoded model hyperparameters file')
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
