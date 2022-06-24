import argparse
import logging
import os
import tensorflow as tf
from keras import layers, models
from nlpvocab import Vocabulary
from .config import VectModel, build_config
from .layer import CbowContext
from .model import _unit_encoder, _model_encoder
from .vocab import _vocab_names


def export_encoder(data_path, params_path, model_path):
    config = build_config(params_path)

    unit_path, _ = _vocab_names(data_path, config)
    unit_vocab = Vocabulary.load(unit_path)

    model = models.load_model(os.path.join(model_path, 'last'))

    unit_weights = model.get_layer('context_encoder').get_layer('unit_encoder').get_weights()
    unit_encoder = _unit_encoder(config, unit_vocab, with_prep=True)

    unit_inputs = layers.Input(name='units', shape=(), dtype='string')
    unit_outputs = unit_encoder(unit_inputs)
    unit_encoder.set_weights(unit_weights)
    unit_model = models.Model(inputs=unit_inputs, outputs=unit_outputs)

    unit_model.save(os.path.join(model_path, 'unit_encoder'), include_optimizer=False)
    tf.get_logger().info('Unit encoder saved to {}'.format(os.path.join(model_path, 'unit_encoder')))

    if config.vect_model in {VectModel.CBOW, VectModel.CBOWPOS}:
        context_weights = model.get_layer('context_encoder').get_weights()
        context_encoder = _model_encoder(config, unit_vocab, with_prep=True)

        context_inputs = layers.Input(name='units', shape=(None,), ragged=True, dtype='string')
        context_outputs = CbowContext(layer=context_encoder, window=config.window_size,
                                      position=VectModel.CBOWPOS == config.vect_model)(context_inputs)
        context_encoder.set_weights(context_weights)
        context_model = models.Model(inputs=context_inputs, outputs=context_outputs)

        context_model.save(os.path.join(model_path, 'context_encoder'), include_optimizer=False)
        tf.get_logger().info('Context encoder saved to {}'.format(os.path.join(model_path, 'context_encoder')))

    tf.get_logger().info('Export finished')


def main():
    parser = argparse.ArgumentParser(description='Word2Vec encoder exporter')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='YAML-encoded model hyperparameters file')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to source .txt.gz documents with one sentence per line')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to save model')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.model_path) and os.path.isdir(argv.model_path), 'Wrong model path'

    params_path = argv.hyper_params.name
    argv.hyper_params.close()

    tf.get_logger().setLevel(logging.INFO)

    export_encoder(argv.data_path, params_path, argv.model_path)
