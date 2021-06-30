import argparse
import logging
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from tfmiss.keras.callbacks import LRFinder
from .hparam import build_hparams
from .input import train_dataset
from .model import build_model
from .vocab import _vocab_names


def train_model(data_path, params_path, model_path, findlr_steps=0):
    h_params = build_hparams(params_path)

    if h_params.use_jit:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        tf.config.optimizer.set_jit(True)

    if h_params.mixed_fp16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    if findlr_steps:
        lr_finder = LRFinder(findlr_steps)
        callbacks = [lr_finder]
    else:
        lr_finder = None
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                os.path.join(model_path, 'logs'),
                update_freq=1000,
                profile_batch='140, 160'),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_path, 'train'),
                monitor='loss',
                verbose=True,
                options=tf.saved_model.SaveOptions(namespace_whitelist=['Addons', 'Miss']))
        ]

    if 'ranger' == h_params.train_optim.lower():
        optimizer = Lookahead(RectifiedAdam(h_params.learn_rate))
    else:
        optimizer = tf.keras.optimizers.get(h_params.train_optim)
        tf.keras.backend.set_value(optimizer.lr, h_params.learn_rate)

    unit_path, label_path = _vocab_names(data_path, h_params)
    unit_vocab = Vocabulary.load(unit_path)
    label_vocab = Vocabulary.load(label_path)
    dataset = train_dataset(data_path, h_params, label_vocab)

    if os.path.isdir(os.path.join(model_path, 'train')):
        tf.get_logger().info('Previous training found. Loading pretrained model.')
        model = tf.keras.models.load_model(os.path.join(model_path, 'train'))  # TODO: check
    else:
        tf.get_logger().info('No previous training found. Creating model from scratch.')
        model = build_model(h_params, unit_vocab, label_vocab)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy' if 'sm' == h_params.model_head else None
        )

    model.summary()
    model.fit(
        dataset,
        epochs=1 if findlr_steps > 0 else h_params.num_epochs,
        callbacks=callbacks,
        steps_per_epoch=findlr_steps if findlr_steps > 0 else None
    )

    if findlr_steps > 0:
        best_lr, _ = lr_finder.plot()
        tf.get_logger().info('Best lr should be near: {}'.format(best_lr))


def main():
    parser = argparse.ArgumentParser(description='Word2Vec model trainer')
    parser.add_argument(
        'hyper_params',
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
