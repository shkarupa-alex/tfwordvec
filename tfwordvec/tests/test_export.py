import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
from ..train import train_model
from ..hparam import build_hparams
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
        self.default_policy = tf.keras.mixed_precision.global_policy()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.model_dir, ignore_errors=True)
        tf.keras.mixed_precision.set_global_policy(self.default_policy)

    def test_char_skipgram(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'skipgram_char.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        export_vectors(self.data_dir, h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_char_cbowpos(self):
        h_params = build_hparams(os.path.join(self.params_dir, 'cbowpos_char.json'))
        train_model(self.data_dir, h_params, self.model_dir)
        export_encoder(h_params, self.model_dir)
        export_vectors(self.data_dir, h_params, self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))


if __name__ == "__main__":
    tf.test.main()
