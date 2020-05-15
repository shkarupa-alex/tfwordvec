# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ..hparam import build_hparams


class TestHParam(tf.test.TestCase):
    def test_empty_overrides(self):
        build_hparams({})

    def test_addons_optimizer(self):
        build_hparams({'train_optim': 'Addons>RectifiedAdam'})


if __name__ == "__main__":
    tf.test.main()
