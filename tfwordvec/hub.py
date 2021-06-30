import argparse
import logging
import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from .hparam import build_hparams
from .layer import CbowContext


def export_encoder(params_path, model_path):
    h_params = build_hparams(params_path)

    model = tf.keras.models.load_model(os.path.join(model_path, 'train'))
    save_options = tf.saved_model.SaveOptions(namespace_whitelist=['Miss'])

    unit_encoder = model.get_layer('context_encoder').get_layer('unit_encoding').layer
    unit_inputs = Input(name='units', shape=(), dtype=tf.string)
    unit_outputs = unit_encoder(unit_inputs)
    unit_model = Model(inputs=unit_inputs, outputs=unit_outputs)
    unit_model.save(os.path.join(model_path, 'unit_encoder'), options=save_options, include_optimizer=False)
    tf.get_logger().info('Unit encoder saved to {}'.format(os.path.join(model_path, 'unit_encoder')))

    if h_params.vect_model in {'cbow', 'cbowpos'}:
        context_encoder = model.get_layer('context_encoder')
        context_inputs = Input(name='units', shape=(None,), ragged=True, dtype=tf.string)
        context_outputs = CbowContext(layer=context_encoder, window=h_params.window_size,
                                      position='cbowpos' == h_params.vect_model)(context_inputs)
        context_model = Model(inputs=context_inputs, outputs=context_outputs)
        context_model.save(os.path.join(model_path, 'context_encoder'), options=save_options, include_optimizer=False)
        tf.get_logger().info('Context encoder saved to {}'.format(os.path.join(model_path, 'context_encoder')))

    tf.get_logger().info('Export finished')


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

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.model_path) and os.path.isdir(argv.model_path), 'Wrong model path'

    params_path = argv.hyper_params.name
    argv.hyper_params.close()

    tf.get_logger().setLevel(logging.INFO)

    export_encoder(params_path, argv.model_path)
