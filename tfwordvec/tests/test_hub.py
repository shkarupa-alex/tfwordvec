import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
from tensorflow_hub import KerasLayer
from ..train import train_model
from ..hparam import build_hparams
from ..hub import export_encoder


class TestExportEncoders(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(1)
        tf.random.set_seed(2)
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.params_dir = os.path.join(os.path.dirname(__file__), 'config')
        self.model_dir = tempfile.mkdtemp()
        self.default_policy = tf.keras.mixed_precision.global_policy()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.model_dir, ignore_errors=True)
        tf.keras.mixed_precision.set_global_policy(self.default_policy)

    def test_char_skipgram(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'skipgram_char.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        vectors = embed(['A', 'b']).numpy()
        self.assertTupleEqual((2, 256), vectors.shape)

    def test_word_skipgram(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'skipgram_word.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        infered = embed(['Керри'])[0]

        model = tf.keras.models.load_model(os.path.join(self.model_dir, 'train'))
        embedding = model.get_layer('context_encoder').get_layer('unit_encoding').layer.get_layer('unit_embedding')
        actual = embedding(tf.constant([3], dtype=tf.int32))[0]

        self.assertAllEqual(actual, infered)

    def test_ngram_skipgram(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'skipgram_ngram.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        vectors = embed(['Abc012', 'def']).numpy()
        self.assertTupleEqual((2, 256), vectors.shape)

    def test_char_cbow(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'cbow_char.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        vectors = embed(['A', 'b']).numpy()
        self.assertTupleEqual((2, 256), vectors.shape)

        embed = KerasLayer(os.path.join(self.model_dir, 'context_encoder'))
        vectors = embed(tf.ragged.constant([['A', 'b'], ['c']])).to_tensor().numpy()
        self.assertTupleEqual((2, 2, 256), vectors.shape)

    def test_word_cbow(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'cbow_word.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        vectors = embed(['Abc012', 'def']).numpy()
        self.assertTupleEqual((2, 256), vectors.shape)

        embed = KerasLayer(os.path.join(self.model_dir, 'context_encoder'))
        vectors = embed(tf.ragged.constant([['A', 'b'], ['c']])).to_tensor().numpy()
        self.assertTupleEqual((2, 2, 256), vectors.shape)

    def test_ngram_cbow(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'cbow_ngram.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        vectors = embed(['Abc012', 'def']).numpy()
        self.assertTupleEqual((2, 256), vectors.shape)

        embed = KerasLayer(os.path.join(self.model_dir, 'context_encoder'))
        vectors = embed(tf.ragged.constant([['A', 'b'], ['c']])).to_tensor().numpy()
        self.assertTupleEqual((2, 2, 256), vectors.shape)

    def test_char_cbowpos(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'cbowpos_char.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        vectors = embed(['A', 'b']).numpy()
        self.assertTupleEqual((2, 256), vectors.shape)

        embed = KerasLayer(os.path.join(self.model_dir, 'context_encoder'))
        vectors = embed(tf.ragged.constant([['A', 'b'], ['c']])).to_tensor().numpy()
        self.assertTupleEqual((2, 2, 256), vectors.shape)

    def test_word_cbowpos(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'cbowpos_word.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        vectors = embed(['Abc012', 'def']).numpy()
        self.assertTupleEqual((2, 256), vectors.shape)

        embed = KerasLayer(os.path.join(self.model_dir, 'context_encoder'))
        vectors = embed(tf.ragged.constant([['A', 'b'], ['c']])).to_tensor().numpy()
        self.assertTupleEqual((2, 2, 256), vectors.shape)

    def test_ngram_cbowpos(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'cbowpos_ngram.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder', 'saved_model.pb')))
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'context_encoder', 'saved_model.pb')))

        embed = KerasLayer(os.path.join(self.model_dir, 'unit_encoder'))
        vectors = embed(['Abc012', 'def']).numpy()
        self.assertTupleEqual((2, 256), vectors.shape)

        embed = KerasLayer(os.path.join(self.model_dir, 'context_encoder'))
        vectors = embed(tf.ragged.constant([['A', 'b'], ['c']])).to_tensor().numpy()
        self.assertTupleEqual((2, 2, 256), vectors.shape)


if __name__ == "__main__":
    tf.test.main()
