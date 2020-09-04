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
from ..export import export_vectors


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
        export_vectors(self.data_dir, os.path.join(self.params_dir, 'skipgram_char.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))

    def test_char_cbowpos(self):
        train_model(self.data_dir, os.path.join(self.params_dir, 'cbowpos_char.json'), self.model_dir)
        export_encoder(os.path.join(self.params_dir, 'cbowpos_char.json'), self.model_dir)
        export_vectors(self.data_dir, os.path.join(self.params_dir, 'cbowpos_char.json'), self.model_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.model_dir, 'unit_encoder.bin')))


if __name__ == "__main__":
    tf.test.main()
