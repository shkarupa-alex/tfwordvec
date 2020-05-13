from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from .hparam import build_hparams
from .input import train_dataset
from .model import build_model
from .vocab import vocab_names


def train_model(data_path, params_path, model_path):
    with open(params_path, 'r') as f:
        h_params = build_hparams(json.loads(f.read()))

    unit_path, label_path = vocab_names(data_path, h_params)
    unit_vocab = Vocabulary.load(unit_path)
    label_vocab = Vocabulary.load(label_path)

    dataset = train_dataset(data_path, h_params, label_vocab)
    model, encoder = build_model(h_params, unit_vocab, label_vocab)
    callbacks = [
        tf.keras.callbacks.TensorBoard(os.path.join(model_path, 'logs'), profile_batch='20, 30'),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor='loss', verbose=True)
    ]

    model.compile(
        optimizer=h_params.train_optim,  # TODO lr
        loss='sparse_categorical_crossentropy' if 'sm' == h_params.model_head else None,
        callbacks=callbacks,
        run_eagerly=False,
    )
    model.fit(dataset, epochs=h_params.num_epochs)

    tf.saved_model.save(encoder, os.path.join(model_path, 'export'))


def main():
    parser = argparse.ArgumentParser(description='Word2Vec model trainer')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to source .txt.gz documents with one sentence per line')
    parser.add_argument(
        'params_path',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparameters file')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to save model')

    argv, _ = parser.parse_known_args()
    if not os.path.exists(argv.data_path) or not os.path.isdir(argv.data_path):
        raise IOError('Wrong data path')

    params_path = argv.hyper_params.name
    argv.hyper_params.close()

    tf.get_logger().setLevel(logging.INFO)

    train_model(argv.data_path, params_path, argv.model_path)
