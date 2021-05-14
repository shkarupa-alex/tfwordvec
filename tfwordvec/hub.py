from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from .hparam import build_hparams
from .layer import NormalizeUnits, CbowContext


def export_encoder(params_path, model_path):
    with open(params_path, 'r') as f:
        h_params = build_hparams(json.loads(f.read()))

    model = tf.keras.models.load_model(os.path.join(model_path, 'train'))

    unit_encoder = model.get_layer('context_encoder').get_layer('unit_encoding').layer
    unit_inputs = Input(name='units', shape=(), dtype=tf.string)
    unit_outputs = NormalizeUnits(
        lower=h_params.lower_case,
        zero=h_params.zero_digits)(unit_inputs)
    unit_outputs = unit_encoder(unit_outputs)
    unit_model = Model(inputs=unit_inputs, outputs=unit_outputs)
    save_options = tf.saved_model.SaveOptions(namespace_whitelist=['Miss'])
    # TODO tf.saved_model.save
    unit_model.save(os.path.join(model_path, 'unit_encoder'), options=save_options)
    tf.get_logger().info('Unit encoder saved to {}'.format(os.path.join(model_path, 'unit_encoder')))

    if h_params.vect_model in {'cbow', 'cbowpos'}:
        tf.get_logger().info('Context encoder could not be exported for now')
        context_encoder = model.get_layer('context_encoder')
        context_inputs = Input(name='units', shape=(None,), ragged=True, dtype=tf.string)
        context_outputs = NormalizeUnits(
            lower=h_params.lower_case,
            zero=h_params.zero_digits)(context_inputs)
        context_outputs = CbowContext(
            layer=context_encoder,
            window=h_params.window_size,
            position='cbowpos' == h_params.vect_model)(context_outputs)
        context_model = Model(inputs=context_inputs, outputs=context_outputs)
        context_model.save(os.path.join(model_path, 'context_encoder'), options=save_options)
        # TODO tf.saved_model.save(context_model, os.path.join(model_path, 'context_encoder'))
        tf.get_logger().info('Context encoder saved to {}'.format(os.path.join(model_path, 'context_encoder')))

    tf.get_logger().info('Export finished')


def main():
    parser = argparse.ArgumentParser(description='Word2Vec encoder exporter')
    parser.add_argument(
        'params_path',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparameters file')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to save model')

    argv, _ = parser.parse_known_args()
    if not os.path.exists(argv.model_path) or not os.path.isdir(argv.model_path):
        raise IOError('Wrong model path')

    params_path = argv.params_path.name
    argv.params_path.close()

    tf.get_logger().setLevel(logging.INFO)

    export_encoder(params_path, argv.model_path)
