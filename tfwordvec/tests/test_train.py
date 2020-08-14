from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
from ..train import train_model


class TestTrainModel(tf.test.TestCase):
    def setUp(self):
        np.random.seed(1)
        tf.random.set_seed(2)
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.params_dir = os.path.join(os.path.dirname(__file__), 'config')
        self.model_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.model_dir, ignore_errors=True)

    def test_char_skipgram(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'skipgram_char.json'), self.model_dir)

    def test_word_skipgram(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'skipgram_word.json'), self.model_dir)

    def test_ngram_skipgram(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'skipgram_ngram.json'), self.model_dir)

    def test_char_cbow(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbow_char.json'), self.model_dir)

    def test_word_cbow(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbow_word.json'), self.model_dir)

    def test_ngram_cbow(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbow_ngram.json'), self.model_dir)


if __name__ == "__main__":
    tf.test.main()
