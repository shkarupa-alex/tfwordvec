from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from ..hparam import build_hparams
from ..input import train_dataset, vocab_dataset


class TestTrainDataset(tf.test.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.char_vocab = Vocabulary.load(
            os.path.join(self.data_dir, 'char_vocab.tsv'),
            format=Vocabulary.FORMAT_TSV_WITH_HEADERS)
        self.word_vocab = Vocabulary.load(
            os.path.join(self.data_dir, 'word_vocab.tsv'),
            format=Vocabulary.FORMAT_TSV_WITH_HEADERS)

    def test_char_skipgram_sm_lower(self):
        h_params = build_hparams({
            'input_unit': 'char',
            'vect_model': 'skipgram',
            'model_head': 'sm',
            'batch_size': 8
        })
        dataset = train_dataset(self.data_dir, h_params, self.char_vocab)

        for row in dataset.take(1):
            self.assertLen(row, 2)
            features, labels = row

            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['units'])

            self.assertTrue(tf.is_tensor(features['units']))
            self.assertListEqual(features['units'].shape.as_list(), [8])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [8])
            self.assertEqual(labels.dtype, tf.int64)

    def test_char_cbow_ss_cased(self):
        h_params = build_hparams({
            'input_unit': 'char',
            'vect_model': 'cbow',
            'model_head': 'ss',
            'lower_case': False,
            'batch_size': 6
        })
        dataset = train_dataset(self.data_dir, h_params, self.char_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [8])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_char_cbowpos_nce_cased(self):
        h_params = build_hparams({
            'input_unit': 'char',
            'vect_model': 'cbowpos',
            'model_head': 'nce',
            'lower_case': False,
            'batch_size': 10
        })
        dataset = train_dataset(self.data_dir, h_params, self.char_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [8])
            self.assertEqual(features['labels'].dtype, tf.int64)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [8, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

    def test_word_skipgram_asm_lower(self):
        h_params = build_hparams({
            'input_unit': 'word',
            'vect_model': 'skipgram',
            'model_head': 'asm',
            'batch_size': 4
        })
        dataset = train_dataset(self.data_dir, h_params, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertTrue(tf.is_tensor(features['units']))
            self.assertListEqual(features['units'].shape.as_list(), [4])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [4])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_word_cbow_sm_cased_nobuck(self):
        h_params = build_hparams({
            'input_unit': 'word',
            'vect_model': 'cbow',
            'model_head': 'sm',
            'lower_case': False,
            'batch_size': 6,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, h_params, self.word_vocab)

        for row in dataset.take(1):
            self.assertLen(row, 2)
            features, labels = row

            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [6, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6])
            self.assertEqual(labels.dtype, tf.int64)

    def test_word_cbowpos_ss_cased_nobuck(self):
        h_params = build_hparams({
            'input_unit': 'word',
            'vect_model': 'cbowpos',
            'model_head': 'ss',
            'lower_case': False,
            'batch_size': 4,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, h_params, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [4, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [4])
            self.assertEqual(features['labels'].dtype, tf.int64)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [4, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

    def test_ngram_skipgram_nce_lower(self):
        h_params = build_hparams({
            'input_unit': 'ngram',
            'vect_model': 'skipgram',
            'model_head': 'nce',
            'batch_size': 8
        })
        dataset = train_dataset(self.data_dir, h_params, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [8])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_ngram_cbow_asm_cased(self):
        h_params = build_hparams({
            'input_unit': 'ngram',
            'vect_model': 'cbow',
            'model_head': 'asm',
            'lower_case': False,
            'batch_size': 2
        })
        dataset = train_dataset(self.data_dir, h_params, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [8])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_ngram_cbowpos_sm_cased(self):
        h_params = build_hparams({
            'input_unit': 'ngram',
            'vect_model': 'cbowpos',
            'model_head': 'sm',
            'lower_case': False,
            'batch_size': 2
        })
        dataset = train_dataset(self.data_dir, h_params, self.word_vocab)

        for row in dataset.take(1):
            self.assertLen(row, 2)
            features, labels = row

            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [8, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [8])
            self.assertEqual(labels.dtype, tf.int64)


class TestVocabDataset(tf.test.TestCase):
    def setUp(self):
        np.random.seed(1)
        tf.random.set_seed(2)
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test_char_skipgram_sm_lower(self):
        h_params = build_hparams({
            'input_unit': 'char',
            'vect_model': 'skipgram',
            'model_head': 'sm',
            'batch_size': 6
        })
        dataset = vocab_dataset(self.data_dir, h_params)

        for features, labels in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['units'])

            self.assertTrue(tf.is_tensor(features['units']))
            self.assertListEqual(features['units'].shape.as_list(), [6])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6])
            self.assertEqual(labels.dtype, tf.string)

    def test_char_cbow_ss_cased(self):
        h_params = build_hparams({
            'input_unit': 'char',
            'vect_model': 'cbow',
            'model_head': 'ss',
            'lower_case': False,
            'batch_size': 6
        })
        dataset = vocab_dataset(self.data_dir, h_params)

        for features, labels in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [8])
            self.assertEqual(labels.dtype, tf.string)

    def test_char_cbowpos_nce_cased(self):
        h_params = build_hparams({
            'input_unit': 'char',
            'vect_model': 'cbowpos',
            'model_head': 'nce',
            'lower_case': False,
            'batch_size': 10
        })
        dataset = vocab_dataset(self.data_dir, h_params)

        for features, labels in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [8, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [8])
            self.assertEqual(labels.dtype, tf.string)

    def test_word_skipgram_asm_lower(self):
        h_params = build_hparams({
            'input_unit': 'word',
            'vect_model': 'skipgram',
            'model_head': 'asm',
            'batch_size': 4
        })
        dataset = vocab_dataset(self.data_dir, h_params)

        for features, labels in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['units'])

            self.assertTrue(tf.is_tensor(features['units']))
            self.assertListEqual(features['units'].shape.as_list(), [4])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [4])
            self.assertEqual(labels.dtype, tf.string)

    def test_word_cbow_sm_cased_nobuck(self):
        h_params = build_hparams({
            'input_unit': 'word',
            'vect_model': 'cbow',
            'model_head': 'sm',
            'lower_case': False,
            'batch_size': 6,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, h_params)

        for features, labels in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [6, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6])
            self.assertEqual(labels.dtype, tf.string)

    def test_word_cbowpos_ss_cased_nobuck(self):
        h_params = build_hparams({
            'input_unit': 'word',
            'vect_model': 'cbowpos',
            'model_head': 'ss',
            'lower_case': False,
            'batch_size': 4,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, h_params)

        for features, labels in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [4, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [4, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [4])
            self.assertEqual(labels.dtype, tf.string)

    def test_ngram_skipgram_nce_lower(self):
        h_params = build_hparams({
            'input_unit': 'ngram',
            'vect_model': 'skipgram',
            'model_head': 'nce',
            'batch_size': 8
        })
        dataset = vocab_dataset(self.data_dir, h_params)

        for features, labels in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [8])
            self.assertEqual(labels.dtype, tf.string)

    def test_ngram_cbow_asm_cased(self):
        h_params = build_hparams({
            'input_unit': 'ngram',
            'vect_model': 'cbow',
            'model_head': 'asm',
            'lower_case': False,
            'batch_size': 2
        })
        dataset = vocab_dataset(self.data_dir, h_params)

        for features, labels in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [8])
            self.assertEqual(labels.dtype, tf.string)

    def test_ngram_cbowpos_sm_cased(self):
        h_params = build_hparams({
            'input_unit': 'ngram',
            'vect_model': 'cbowpos',
            'model_head': 'sm',
            'lower_case': False,
            'batch_size': 2
        })
        dataset = vocab_dataset(self.data_dir, h_params)

        for features, labels in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [8, None, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [8, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [8])
            self.assertEqual(labels.dtype, tf.string)


if __name__ == "__main__":
    tf.test.main()
