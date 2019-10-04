from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
from nlpvocab import Vocabulary
from tfwordvec.word2vec.estimator import _TRAIN_OPTIMIZER, _LEARNING_RATE, Word2VecEstimator
from tfwordvec.word2vec.input import skipgram_input_fn, cbow_input_fn
from tfwordvec.word2vec.model import LossFunction


def word2vec():
    argv = parse_cli_args(description='Word2Vec estimator')

    logging.basicConfig(level=logging.INFO)
    logging.info('Training word2vec with options: {}'.format(argv))

    freq_vocab = Vocabulary.load(argv.freq_vocab)

    estimator = Word2VecEstimator(
        freq_vocab=freq_vocab,
        min_freq=argv.min_freq,
        embed_size=argv.embedding_size,
        loss_func=argv.train_loss,
        sampled_count=argv.sampled_count,
        train_opt=argv.train_optimizer,
        learn_rate=argv.learning_rate,
        model_dir=argv.model_path
    )

    file_pattern = os.path.join(argv.train_path, '*.txt.gz')
    input_fn = cbow_input_fn if argv.train_model == 'cbow' else skipgram_input_fn

    estimator.train(input_fn=lambda: input_fn(
        file_pattern=file_pattern,
        batch_size=argv.batch_size,
        freq_vocab=freq_vocab,
        sample_threshold=argv.sample_threshold,
        min_freq=argv.min_freq,
        window_size=argv.window_size,
        repeats_count=argv.repeats_count,
        cycle_length=argv.cycle_length,
        threads_count=argv.threads_count
    ))


def parse_cli_args(description=None):
    parser = argparse.ArgumentParser(description=description)

    # Paths
    parser.add_argument(
        'train_path',
        type=str,
        help='Path to source text documents. Should be gzipped.')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to save model')

    # Vocabulary
    parser.add_argument(
        'freq_vocab',
        action=argparse.FileType('rb'),
        help='Frequency vocabulary (nlpvocab.Vocabulary in binary format)')
    parser.add_argument(
        '--min_freq',
        type=int,
        default=5,
        help='Treat words that appears less times as unique')

    # Input
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Training batch size')
    parser.add_argument(
        '--sample_threshold',
        type=float,
        default=1e-3,
        help='Downsampling threshold. Useful range is (0, 1e-5)')
    parser.add_argument(
        '--window_size',
        type=int,
        default=5,
        help='Number of context words to take into account')
    parser.add_argument(
        '--repeats_count',
        type=int,
        default=1,
        help='Number of dataset repeats')
    parser.add_argument(
        '--threads_count',
        type=int,
        default=12,
        help='Number of threads for data preprocessing')
    parser.add_argument(
        '--cycle_length',
        type=int,
        default=None,
        help='Number of input files to process in parallel. By default 1/4 of threads count')

    # Training
    parser.add_argument(
        'embedding_size',
        type=int,
        default=100,
        help='Size of embedding vector. Common values are 100-1000')
    parser.add_argument(
        '--train_model',
        choices=['cbow', 'skipgram'],
        default='cbow',
        help='Model: continuous bag of words vs skip gram')
    parser.add_argument(
        '--train_loss',
        choices=[LossFunction.NOISE_CONTRASTIVE_ESTIMATION, LossFunction.SAMPLED_SOFTMAX],
        default=LossFunction.NOISE_CONTRASTIVE_ESTIMATION,
        help='How to combine embedding results for each entry')
    parser.add_argument(
        '--sampled_count',
        type=int,
        default=5,
        help='Number of negative examples. Common values are 3 - 10 (0 = not used)')
    parser.add_argument(
        'train_optimizer',
        choices=['Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'],
        default=_TRAIN_OPTIMIZER,
        help='Training optimizer')
    parser.add_argument(
        'learning_rate',
        type=float,
        default=_LEARNING_RATE,
        help='Training optimizer')
    argv, _ = parser.parse_known_args()

    assert os.path.exists(argv.train_path) and os.path.isdir(argv.train_path)
    assert not os.path.exists(argv.model_path) or os.path.isdir(argv.model_path)

    freq_vocab_name = argv.freq_vocab.name
    argv.freq_vocab.close()
    argv.freq_vocab = freq_vocab_name
    del freq_vocab_name
    assert argv.freq_vocab.endswith('.pkl')
    assert argv.min_freq > 1

    assert argv.batch_size > 0
    assert 0.0 <= argv.sample_threshold <= 1.0
    assert argv.window_size > 0
    assert argv.repeats_count > 0
    assert argv.threads_count > 0
    if argv.cycle_length is None:
        argv.cycle_length = max(1, argv.threads_count // 4)
    else:
        assert argv.cycle_length > 0

    assert argv.embedding_size > 0
    assert argv.sampled_count > 0
    assert argv.learning_rate > 0.0

    return argv