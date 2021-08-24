import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from ..train import train_model
from ..hub import export_encoder
from ..export import export_vectors


class TestExportEncoders(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(1)
        tf.random.set_seed(2)
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.params_dir = os.path.join(os.path.dirname(__file__), 'config')
        self.model_dir = tempfile.mkdtemp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.model_dir, ignore_errors=True)
        mixed_precision.set_policy(self.default_policy)

    def test_char_skipgram(self):
        params_path = os.path.join(self.params_dir, 'skipgram_char.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_skipgram_char_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_word_skipgram(self):
        params_path = os.path.join(self.params_dir, 'skipgram_word.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_skipgram_word_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_ngram_skipgram(self):
        params_path = os.path.join(self.params_dir, 'skipgram_ngram.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_skipgram_ngram_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_bpe_skipgram(self):
        params_path = os.path.join(self.params_dir, 'skipgram_bpe.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_skipgram_bpe_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_cnn_skipgram(self):
        params_path = os.path.join(self.params_dir, 'skipgram_cnn.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_skipgram_cnn_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_char_cbow(self):
        params_path = os.path.join(self.params_dir, 'cbow_char.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_char_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_word_cbow(self):
        params_path = os.path.join(self.params_dir, 'cbow_word.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_word_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_ngram_cbow(self):
        params_path = os.path.join(self.params_dir, 'cbow_ngram.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_ngram_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_bpe_cbow(self):
        params_path = os.path.join(self.params_dir, 'cbow_bpe.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_bpe_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_cnn_cbow(self):
        params_path = os.path.join(self.params_dir, 'cbow_cnn.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_cnn_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_char_cbowpos(self):
        params_path = os.path.join(self.params_dir, 'cbowpos_char.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_char_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_word_cbowpos(self):
        params_path = os.path.join(self.params_dir, 'cbowpos_word.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_word_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_ngram_cbowpos(self):
        params_path = os.path.join(self.params_dir, 'cbowpos_ngram.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_ngram_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_bpe_cbowpos(self):
        params_path = os.path.join(self.params_dir, 'cbowpos_bpe.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_bpe_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_cnn_cbowpos(self):
        params_path = os.path.join(self.params_dir, 'cbowpos_cnn.json')
        train_model(self.data_dir, params_path, self.model_dir)
        export_encoder(params_path, self.model_dir)
        vocab_path = os.path.join(self.data_dir, 'vocab_cbow_cnn_label.pkl')
        export_vectors(vocab_path, params_path, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))


if __name__ == "__main__":
    tf.test.main()
