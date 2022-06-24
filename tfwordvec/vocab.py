import argparse
import logging
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from .input import vocab_dataset
from .model import _unit_embedder
from .config import RESERVED, VectModel, build_config


def extract_vocab(data_path, config):
    dataset = vocab_dataset(data_path, config)

    label_vocab = Vocabulary()
    for labels in dataset:
        label_vocab.update(labels.flat_values.numpy())
    label_vocab = Vocabulary({w.decode('utf-8'): f for w, f in label_vocab.most_common()})
    assert 0 == label_vocab[''], 'Empty label occured'

    embedder = _unit_embedder(config, RESERVED, with_prep=False)
    unit_vocab = embedder.vocab(label_vocab)

    return unit_vocab, label_vocab


def _vocab_names(data_path, config, fmt=Vocabulary.FORMAT_BINARY_PICKLE):
    model_name = config.vect_model.value
    if VectModel.CBOWPOS == config.vect_model:
        model_name = VectModel.CBOW.value

    ext = 'pkl' if Vocabulary.FORMAT_BINARY_PICKLE == fmt else 'tsv'

    unit_vocab = 'vocab_{}_{}_unit.{}'.format(model_name, config.input_unit.value, ext)
    label_vocab = 'vocab_{}_{}_label.{}'.format(model_name, config.input_unit.value, ext)

    return os.path.join(data_path, unit_vocab), os.path.join(data_path, label_vocab)


def main():
    parser = argparse.ArgumentParser(description='Extract vocabulary from dataset')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='YAML-encoded model hyperparameters file')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to train .txt.gz files')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.data_path) and os.path.isdir(argv.data_path), 'Wrong train dataset path'

    tf.get_logger().setLevel(logging.INFO)

    params_path = argv.hyper_params.name
    argv.hyper_params.close()
    config = build_config(params_path)

    tf.get_logger().info('Estimating {} and label vocabularies'.format(config.input_unit))
    unit_vocab, label_vocab = extract_vocab(argv.data_path, config)

    tf.get_logger().info('Saving vocabularies to {}'.format(argv.data_path))
    unit_pkl, label_pkl = _vocab_names(argv.data_path, config)
    unit_vocab.save(unit_pkl)
    label_vocab.save(label_pkl)

    unit_tsv, label_tsv = _vocab_names(argv.data_path, config, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    unit_vocab.save(unit_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    label_vocab.save(label_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)

    tf.get_logger().info('Vocabularies saved to {}'.format(argv.data_path))
