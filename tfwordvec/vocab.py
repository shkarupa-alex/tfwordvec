import argparse
import logging
import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tfmiss.keras.layers import CharNgams
from .input import RESERVED, vocab_dataset
from .hparam import build_hparams


def extract_vocab(data_path, h_params):
    label_vocab = Vocabulary()
    dataset = vocab_dataset(data_path, h_params)

    for labels in dataset:
        labels = _unicode_tensor(labels.flat_values)
        label_vocab.update(labels)

    assert 0 == label_vocab[''], 'Empty label occured'

    unit_vocab = Vocabulary()
    if 'ngram' == h_params.input_unit:
        labels = tf.constant(label_vocab.tokens(), dtype=tf.string)
        ngrams = CharNgams(
            minn=h_params.ngram_minn,
            maxn=h_params.ngram_maxn,
            itself=h_params.ngram_self,
            reserved=RESERVED)(labels)
        for label, ngram in zip(label_vocab.tokens(), ngrams):
            ngram = _unicode_tensor(ngram).reshape([-1])
            for n in ngram:
                unit_vocab[n] += label_vocab[label]
    else:
        unit_vocab.update(label_vocab)

    return unit_vocab, label_vocab


def vocab_names(data_path, h_params, fmt=Vocabulary.FORMAT_BINARY_PICKLE):
    model_name = h_params.vect_model
    if 'cbowpos' == h_params.vect_model:
        model_name = 'cbow'

    ext = 'pkl' if Vocabulary.FORMAT_BINARY_PICKLE == fmt else 'tsv'

    unit_vocab = 'vocab_{}_{}_unit.{}'.format(model_name, h_params.input_unit, ext)
    label_vocab = 'vocab_{}_{}_label.{}'.format(model_name, h_params.input_unit, ext)

    return os.path.join(data_path, unit_vocab), os.path.join(data_path, label_vocab)


def _unicode_tensor(tensor):
    return np.char.decode(tensor.numpy().astype('S'), 'utf-8')


def main():
    parser = argparse.ArgumentParser(description='Extract vocabulary from dataset')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON file with hyper parameters')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to train .txt.gz files')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.data_path) and os.path.isdir(argv.data_path), 'Wrong train dataset path'

    tf.get_logger().setLevel(logging.INFO)

    params_path = argv.hyper_params.name
    argv.hyper_params.close()
    h_params = build_hparams(params_path)

    tf.get_logger().info('Estimating {} and label vocabularies'.format(h_params.input_unit))
    unit_vocab, label_vocab = extract_vocab(argv.data_path, h_params)

    tf.get_logger().info('Saving vocabularies to {}'.format(argv.data_path))
    unit_pkl, label_pkl = vocab_names(argv.data_path, h_params)
    unit_vocab.save(unit_pkl)
    label_vocab.save(label_pkl)

    unit_tsv, label_tsv = vocab_names(argv.data_path, h_params, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    unit_vocab.save(unit_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)
    label_vocab.save(label_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)

    if 'dense' == h_params.embed_type:
        unit1k_vocab, _ = unit_vocab.split_by_size(999)
        unit1k_vocab['[UNK]'] = unit1k_vocab[unit1k_vocab.tokens()[0]] + 1
        unit1k_tsv = unit_tsv[:-4] + '1k' + unit_tsv[-4:]
        unit1k_vocab.save(unit1k_tsv, Vocabulary.FORMAT_TSV_WITH_HEADERS)

    tf.get_logger().info('Vocabularies saved to {}'.format(argv.data_path))
