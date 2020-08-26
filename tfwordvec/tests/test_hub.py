from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
from tensorflow_hub import KerasLayer
from ..train import train_model
from ..hub import export_encoder


class TestExportEncoders(tf.test.TestCase):
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
        export_encoder(os.path.join(self.params_dir, 'skipgram_char.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        embed(['Abc012'])

    def test_word_skipgram(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'skipgram_word.json'), self.model_dir)
        export_encoder(os.path.join(self.params_dir, 'skipgram_word.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        infered = embed(['Керри'])[0]

        model = tf.keras.models.load_model(os.path.join(self.model_dir, 'train'))
        embedding = model.get_layer('context_encoder').get_layer('unit_encoding').layer.get_layer('unit_embedding')
        actual = embedding(tf.constant([3], dtype=tf.int32))[0]

        self.assertAllEqual(actual, infered)

    def test_ngram_skipgram(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'skipgram_ngram.json'), self.model_dir)
        export_encoder(os.path.join(self.params_dir, 'skipgram_ngram.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        embed(['Abc012'])

    def test_char_cbow(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbow_char.json'), self.model_dir)
        export_encoder(os.path.join(self.params_dir, 'cbow_char.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        # self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        embed(['Abc012'])

    def test_word_cbow(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbow_word.json'), self.model_dir)
        export_encoder(os.path.join(self.params_dir, 'cbow_word.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        # self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        embed(['Abc012'])

    def test_ngram_cbow(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbow_ngram.json'), self.model_dir)
        export_encoder(os.path.join(self.params_dir, 'cbow_ngram.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        # self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        embed(['Abc012'])

    def test_char_cbowpos(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbowpos_char.json'), self.model_dir)
        export_encoder(os.path.join(self.params_dir, 'cbowpos_char.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        # self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        embed(['Abc012'])

    def test_word_cbowpos(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbowpos_word.json'), self.model_dir)
        export_encoder(os.path.join(self.params_dir, 'cbowpos_word.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        # self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        embed(['Abc012'])

    def test_ngram_cbowpos(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbowpos_ngram.json'), self.model_dir)
        export_encoder(os.path.join(self.params_dir, 'cbowpos_ngram.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        # self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        embed(['Abc012'])


if __name__ == "__main__":
    tf.test.main()
