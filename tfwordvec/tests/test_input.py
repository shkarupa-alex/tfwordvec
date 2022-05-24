import numpy as np
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from ..config import InputUnit, VectModel, ModelHead, build_config
from ..input import train_dataset, vocab_dataset


class TestTrainDataset(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(1)
        tf.random.set_seed(1)
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.char_vocab = Vocabulary.load(
            os.path.join(self.data_dir, 'char_vocab.tsv'),
            format=Vocabulary.FORMAT_TSV_WITH_HEADERS)
        self.word_vocab = Vocabulary.load(
            os.path.join(self.data_dir, 'word_vocab.tsv'),
            format=Vocabulary.FORMAT_TSV_WITH_HEADERS)

    def test_char_skipgram_sm(self):
        config = build_config({
            'input_unit': InputUnit.CHAR,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 8,
            'samp_thold': 1e-1
        })
        dataset = train_dataset(self.data_dir, config, self.char_vocab)

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

    def test_char_cbow_ss(self):
        config = build_config({
            'input_unit': InputUnit.CHAR,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.SAMPLED,
            'batch_size': 6,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.char_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [6, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [6])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_char_cbowpos_nce(self):
        config = build_config({
            'input_unit': InputUnit.CHAR,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.NCE,
            'batch_size': 10,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.char_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [10, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [10])
            self.assertEqual(features['labels'].dtype, tf.int64)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [10, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

    def test_word_skipgram_asm(self):
        config = build_config({
            'input_unit': InputUnit.WORD,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.ADAPTIVE,
            'batch_size': 4,
            'samp_thold': 1e-1
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertTrue(tf.is_tensor(features['units']))
            self.assertListEqual(features['units'].shape.as_list(), [4])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [4])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_word_cbow_sm(self):
        config = build_config({
            'input_unit': InputUnit.WORD,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 4,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for row in dataset.take(1):
            self.assertLen(row, 2)
            features, labels = row

            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [4, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [4])
            self.assertEqual(labels.dtype, tf.int64)

    def test_word_cbowpos_ss(self):
        config = build_config({
            'input_unit': InputUnit.WORD,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.SAMPLED,
            'batch_size': 4,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

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

    def test_ngram_skipgram_nce(self):
        config = build_config({
            'input_unit': InputUnit.NGRAM,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.NCE,
            'batch_size': 8,
            'samp_thold': 1e-1
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.Tensor)
            self.assertListEqual(features['units'].shape.as_list(), [8])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [8])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_ngram_cbow_asm(self):
        config = build_config({
            'input_unit': InputUnit.NGRAM,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.ADAPTIVE,
            'batch_size': 2,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [2, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [2])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_ngram_cbowpos_sm(self):
        config = build_config({
            'input_unit': InputUnit.NGRAM,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 2,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for row in dataset.take(1):
            self.assertLen(row, 2)
            features, labels = row

            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [2, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [2, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [2])
            self.assertEqual(labels.dtype, tf.int64)

    def test_bpe_skipgram_nce(self):
        config = build_config({
            'input_unit': InputUnit.BPE,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.NCE,
            'batch_size': 8,
            'samp_thold': 1e-1
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.Tensor)
            self.assertListEqual(features['units'].shape.as_list(), [8])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [8])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_bpe_cbow_asm(self):
        config = build_config({
            'input_unit': InputUnit.BPE,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.ADAPTIVE,
            'batch_size': 2,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [2, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [2])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_bpe_cbowpos_sm(self):
        config = build_config({
            'input_unit': InputUnit.BPE,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 2,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for row in dataset.take(1):
            self.assertLen(row, 2)
            features, labels = row

            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [2, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [2, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [2])
            self.assertEqual(labels.dtype, tf.int64)

    def test_cnn_skipgram_nce(self):
        config = build_config({
            'input_unit': InputUnit.CNN,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.NCE,
            'batch_size': 8,
            'samp_thold': 1e-1
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.Tensor)
            self.assertListEqual(features['units'].shape.as_list(), [8])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [8])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_cnn_cbow_asm(self):
        config = build_config({
            'input_unit': InputUnit.CNN,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.ADAPTIVE,
            'batch_size': 2,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for features in dataset.take(1):
            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['labels', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [2, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertTrue(tf.is_tensor(features['labels']))
            self.assertListEqual(features['labels'].shape.as_list(), [2])
            self.assertEqual(features['labels'].dtype, tf.int64)

    def test_cnn_cbowpos_sm(self):
        config = build_config({
            'input_unit': InputUnit.CNN,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 2,
            'samp_thold': 1e-1,
            'bucket_cbow': False
        })
        dataset = train_dataset(self.data_dir, config, self.word_vocab)

        for row in dataset.take(1):
            self.assertLen(row, 2)
            features, labels = row

            self.assertIsInstance(features, dict)
            self.assertEqual(sorted(features.keys()), ['positions', 'units'])

            self.assertIsInstance(features['units'], tf.RaggedTensor)
            self.assertListEqual(features['units'].shape.as_list(), [2, None])
            self.assertEqual(features['units'].dtype, tf.string)

            self.assertIsInstance(features['positions'], tf.RaggedTensor)
            self.assertListEqual(features['positions'].shape.as_list(), [2, None])
            self.assertEqual(features['positions'].dtype, tf.int32)

            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [2])
            self.assertEqual(labels.dtype, tf.int64)


class TestVocabDataset(tf.test.TestCase):
    def setUp(self):
        np.random.seed(1)
        tf.random.set_seed(2)
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test_char_skipgram_sm(self):
        config = build_config({
            'input_unit': InputUnit.CHAR,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 6
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_char_cbow_ss(self):
        config = build_config({
            'input_unit': InputUnit.CHAR,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.SAMPLED,
            'batch_size': 6,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_char_cbowpos_nce(self):
        config = build_config({
            'input_unit': InputUnit.CHAR,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.NCE,
            'batch_size': 10,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [10, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_word_skipgram_asm(self):
        config = build_config({
            'input_unit': InputUnit.WORD,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.ADAPTIVE,
            'batch_size': 4
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [4, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_word_cbow_sm(self):
        config = build_config({
            'input_unit': InputUnit.WORD,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 6,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_word_cbowpos_ss(self):
        config = build_config({
            'input_unit': InputUnit.WORD,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.SAMPLED,
            'batch_size': 4,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [4, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_ngram_skipgram_nce(self):
        config = build_config({
            'input_unit': InputUnit.NGRAM,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.NCE,
            'batch_size': 8
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [8, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_ngram_cbow_asm(self):
        config = build_config({
            'input_unit': InputUnit.NGRAM,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.ADAPTIVE,
            'batch_size': 2,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [2, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_ngram_cbowpos_sm(self):
        config = build_config({
            'input_unit': InputUnit.NGRAM,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 2,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [2, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_bpe_skipgram_sm(self):
        config = build_config({
            'input_unit': InputUnit.BPE,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 6
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_bpe_cbow_ss(self):
        config = build_config({
            'input_unit': InputUnit.BPE,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.SAMPLED,
            'batch_size': 6,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_bpe_cbowpos_nce(self):
        config = build_config({
            'input_unit': InputUnit.BPE,
            'max_len': 5,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.NCE,
            'batch_size': 10,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [10, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_cnn_skipgram_sm(self):
        config = build_config({
            'input_unit': InputUnit.CNN,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.SOFTMAX,
            'batch_size': 6
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_cnn_cbow_ss(self):
        config = build_config({
            'input_unit': InputUnit.CNN,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.SAMPLED,
            'batch_size': 6,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [6, None])
            self.assertEqual(labels.dtype, tf.string)

    def test_cnn_cbowpos_nce(self):
        config = build_config({
            'input_unit': InputUnit.CNN,
            'max_len': 5,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.NCE,
            'batch_size': 10,
            'bucket_cbow': False
        })
        dataset = vocab_dataset(self.data_dir, config)

        for labels in dataset.take(1):
            self.assertTrue(tf.is_tensor(labels))
            self.assertListEqual(labels.shape.as_list(), [10, None])
            self.assertEqual(labels.dtype, tf.string)


if __name__ == "__main__":
    tf.test.main()
