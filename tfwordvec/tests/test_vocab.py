# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from ..hparam import build_hparams
from ..vocab import extract_vocab


class TestExtractVocab(tf.test.TestCase):
    def setUp(self):
        np.random.seed(1)
        tf.random.set_seed(2)
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test_char_skipgram(self):
        h_params = build_hparams({
            'input_unit': 'char',
            'vect_model': 'skipgram',
            'model_head': 'sm',
        })
        unit_vocab, label_vocab = extract_vocab(self.data_dir, h_params)

        expected_set = {
            '\n', ' ', '-', '.', '[BOS]', '[EOS]', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м',
            'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё'}
        self.assertSetEqual(expected_set, set(unit_vocab.tokens()))
        self.assertSetEqual(expected_set, set(label_vocab.tokens()))

        expected_units_top = [
            (' ', 422), ('а', 324), ('о', 318), ('и', 272), ('е', 260), ('р', 224), ('т', 210), ('с', 202), ('н', 182),
            ('в', 162), ('л', 158), ('к', 146), ('м', 114), ('д', 112), ('п', 104)]
        self.assertListEqual(expected_units_top, unit_vocab.most_common(15))

        expected_labels_top = [
            (' ', 422), ('а', 324), ('о', 318), ('и', 272), ('е', 260), ('р', 224), ('т', 210), ('с', 202), ('н', 182),
            ('в', 162), ('л', 158), ('к', 146), ('м', 114), ('д', 112), ('п', 104)]
        self.assertListEqual(expected_labels_top, label_vocab.most_common(15))

    def test_word_cbow(self):
        h_params = build_hparams({
            'input_unit': 'word',
            'vect_model': 'cbow',
            'model_head': 'ss',
        })
        unit_vocab, label_vocab = extract_vocab(self.data_dir, h_params)

        expected_units_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('керри', 16), ('жириновского', 8), ('пайдер', 6), ('жириновский', 6),
            ('лавруша', 4), ('позы', 4), ('становясь', 4), ('сам', 4), ('можно', 4), ('начинал', 4), ('ням', 4),
            ('глубже', 4), ('лет', 4)]
        self.assertListEqual(expected_units_top, unit_vocab.most_common(15))

        expected_labels_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('керри', 8), ('жириновского', 4), ('пайдер', 3), ('жириновский', 3),
            ('лавруша', 2), ('позы', 2), ('становясь', 2), ('сам', 2), ('можно', 2), ('начинал', 2), ('ням', 2),
            ('глубже', 2), ('лет', 2)]
        self.assertListEqual(expected_labels_top, label_vocab.most_common(15))

    def test_ngram_cbowpos_nobuck(self):
        h_params = build_hparams({
            'input_unit': 'ngram',
            'vect_model': 'cbowpos',
            'model_head': 'asm',
            'bucket_cbow': False,
        })

        unit_vocab, label_vocab = extract_vocab(self.data_dir, h_params)

        expected_units_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('<по', 32), ('ть>', 30), ('<на', 24), ('нов', 22), ('<пр', 22), ('<жи', 18),
            ('ерр', 18), ('ал>', 18), ('ири', 18), ('ста', 16), ('ет>', 16), ('<ке', 16), ('кер', 16)]
        self.assertListEqual(expected_units_top, unit_vocab.most_common(15))

        expected_labels_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('керри', 8), ('жириновского', 4), ('пайдер', 3), ('жириновский', 3),
            ('лавруша', 2), ('позы', 2), ('становясь', 2), ('сам', 2), ('можно', 2), ('начинал', 2), ('ням', 2),
            ('глубже', 2), ('лет', 2)]
        self.assertListEqual(expected_labels_top, label_vocab.most_common(15))


if __name__ == "__main__":
    tf.test.main()
