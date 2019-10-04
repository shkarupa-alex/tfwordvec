# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
from nlpvocab import Vocabulary
from ..hparam import build_hparams
from ..input import train_input_fn


class TestTrainInput(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.wildcard = os.path.join(self.data_dir, '*.txg.gz')
        self.params = build_hparams()

    def testCharCbow(self):
        char_vocab = Vocabulary.load(
            os.path.join(self.data_dir, 'char_vocab.tsv'),
            format=Vocabulary.FORMAT_TSV_WITH_HEADERS
        )
        dataset = train_input_fn(self.wildcard, char_vocab, self.params)

        for features, labels in dataset.take(1):
            self.assertEqual(dict, type(labels))
            self.assertEqual(['sentence', 'token'], sorted(labels.keys()))

            self.assertEqual(dict, type(features))
            self.assertEqual([
                'document',
                'length',
                'token_weights',
                'word_length',
                'word_lower',
                'word_mixed',
                'word_ngrams',
                'word_nocase',
                'word_title',
                'word_upper',
                'words',
            ], sorted(features.keys()))

            del features['word_ngrams']  # breaks self.evaluate
            features, labels = self.evaluate([features, labels])
            # self.assertEqual(10, len(features['document']))

            self.assertEqual(3, len(labels['token'].shape))
            # self.assertEqual(10, labels['sentences'].shape[0])

            # self.assertAllEqual(labels['tokens'].shape, features['words'].shape)
            # self.assertAllEqual(labels['sentences'].shape, features['word_length'].shape)
