import argparse
import logging
import os
import shutil
import tensorflow as tf
from nlpvocab import Vocabulary
from keras import backend, callbacks, optimizers
from keras.mixed_precision import policy as mixed_precision
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from tfmiss.keras.callbacks import LRFinder
from .config import ModelHead, build_config
from .input import train_dataset
from .model import build_model
from .vocab import _vocab_names


def train_model(data_path, params_path, model_path, findlr_steps=0):
    backup_path = os.path.join(model_path, 'train')
    weights_path = os.path.join(model_path, 'weights')
    export_path = os.path.join(model_path, 'last')

    config = build_config(params_path)

    if config.use_jit:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        tf.config.optimizer.set_jit('autoclustering')

    if config.mixed_fp16:
        mixed_precision.set_global_policy('mixed_float16')

    if findlr_steps:
        lr_finder = LRFinder(findlr_steps)
        call_backs = [lr_finder]
    else:
        lr_finder = None
        call_backs = [
            callbacks.TensorBoard(
                os.path.join(model_path, 'logs'),
                update_freq=1000,
                profile_batch='140, 160'),
            callbacks.BackupAndRestore(backup_path)
        ]

    if 'ranger' == config.train_optim.lower():
        optimizer = Lookahead(RectifiedAdam(config.learn_rate))
    else:
        optimizer = optimizers.get(config.train_optim)
        backend.set_value(optimizer.lr, config.learn_rate)

    unit_path, label_path = _vocab_names(data_path, config)
    unit_vocab = Vocabulary.load(unit_path)
    label_vocab = Vocabulary.load(label_path)
    dataset = train_dataset(data_path, config, unit_vocab, label_vocab)

    model = build_model(config, unit_vocab, label_vocab)
    if not os.path.exists(backup_path) and os.path.exists(f'{weights_path}.index'):
        model.load_weights(weights_path)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy' if ModelHead.SOFTMAX == config.model_head else None)
    model.summary()

    model.fit(
        dataset,
        epochs=1 if findlr_steps > 0 else config.num_epochs,
        callbacks=call_backs,
        steps_per_epoch=findlr_steps if findlr_steps > 0 else None
    )

    if findlr_steps > 0:
        best_lr, _ = lr_finder.plot()
        tf.get_logger().info('Best lr should be near: {}'.format(best_lr))
    else:
        model.save(export_path)
        shutil.rmtree(backup_path)

        model.save_weights(os.path.join(model_path, 'weights'))


def main():
    parser = argparse.ArgumentParser(description='Word2Vec model trainer')
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
    parser.add_argument(
        '--findlr_steps',
        type=int,
        default=0,
        help='Run model with LRFinder callback')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.data_path) and os.path.isdir(argv.data_path), 'Wrong data path'

    params_path = argv.hyper_params.name
    argv.hyper_params.close()

    tf.get_logger().setLevel(logging.INFO)

    train_model(argv.data_path, params_path, argv.model_path, argv.findlr_steps)
