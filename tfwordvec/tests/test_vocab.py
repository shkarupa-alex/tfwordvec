import numpy as np
import os
import tensorflow as tf
from ..config import InputUnit, VectModel, ModelHead, build_config
from ..vocab import extract_vocab


class TestExtractVocab(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(1)
        tf.random.set_seed(2)
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test_char_skipgram(self):
        config = build_config({
            'input_unit': InputUnit.CHAR,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.SOFTMAX,
        })
        unit_vocab, label_vocab = extract_vocab(self.data_dir, config)

        expected_set = {
            '\n', ' ', '-', '.', '[BOS]', '[EOS]', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м',
            'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё'}
        self.assertSetEqual(expected_set, set(unit_vocab.tokens()))
        self.assertSetEqual(expected_set, set(label_vocab.tokens()))

        expected_units_top = [
            (' ', 211), ('а', 162), ('о', 159), ('и', 136), ('е', 130), ('р', 112), ('т', 105), ('с', 101), ('н', 91),
            ('в', 81), ('л', 79), ('к', 73), ('м', 57), ('д', 56), ('п', 52)]
        self.assertListEqual(expected_units_top, unit_vocab.most_common(15))

        expected_labels_top = [
            (' ', 211), ('а', 162), ('о', 159), ('и', 136), ('е', 130), ('р', 112), ('т', 105), ('с', 101), ('н', 91),
            ('в', 81), ('л', 79), ('к', 73), ('м', 57), ('д', 56), ('п', 52)]
        self.assertListEqual(expected_labels_top, label_vocab.most_common(15))

    def test_word_cbow(self):
        config = build_config({
            'input_unit': InputUnit.WORD,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.SAMPLED,
            'bucket_cbow': False
        })
        unit_vocab, label_vocab = extract_vocab(self.data_dir, config)

        expected_units_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('керри', 8), ('жириновского', 4), ('жириновский', 3), ('пайдер', 3),
            ('владиракл', 2), ('глубже', 2), ('дмитриус', 2), ('есть', 2), ('лавруша', 2), ('лет', 2), ('можно', 2),
            ('нах', 2), ('начинал', 2)]
        self.assertListEqual(expected_units_top, unit_vocab.most_common(15))

        expected_labels_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('керри', 8), ('жириновского', 4), ('пайдер', 3), ('жириновский', 3),
            ('лавруша', 2), ('позы', 2), ('становясь', 2), ('сам', 2), ('можно', 2), ('начинал', 2), ('ням', 2),
            ('глубже', 2), ('лет', 2)]
        self.assertListEqual(expected_labels_top, label_vocab.most_common(15))

    def test_ngram_cbowpos(self):
        config = build_config({
            'input_unit': InputUnit.NGRAM,
            'vect_model': VectModel.CBOWPOS,
            'model_head': ModelHead.ADAPTIVE,
            'bucket_cbow': False
        })

        unit_vocab, label_vocab = extract_vocab(self.data_dir, config)

        expected_units_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('<по', 16), ('ть>', 15), ('<на', 12), ('нов', 11), ('<пр', 11), ('ерр', 9),
            ('<жи', 9), ('ири', 9), ('ал>', 9), ('<ке', 8), ('кер', 8), ('рри', 8), ('ри>', 8)]
        self.assertListEqual(expected_units_top, unit_vocab.most_common(15))

        expected_labels_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('керри', 8), ('жириновского', 4), ('пайдер', 3), ('жириновский', 3),
            ('лавруша', 2), ('позы', 2), ('становясь', 2), ('сам', 2), ('можно', 2), ('начинал', 2), ('ням', 2),
            ('глубже', 2), ('лет', 2)]
        self.assertListEqual(expected_labels_top, label_vocab.most_common(15))

    def test_bpe_skipgram(self):
        config = build_config({
            'input_unit': InputUnit.BPE,
            'bpe_size': 50,
            'bpe_chars': 10,
            'vect_model': VectModel.SKIPGRAM,
            'model_head': ModelHead.SOFTMAX
        })
        unit_vocab, label_vocab = extract_vocab(self.data_dir, config)

        expected_units_top = [
            ('##[UNK]', 451), ('##а', 153), ('##о', 135), ('[UNK]', 133), ('##и', 124), ('##е', 118), ('##р', 103),
            ('##т', 79), ('##н', 70), ('##с', 67), ('##в', 64), ('##л', 64), ('[BOS]', 51), ('[EOS]', 51), ('с', 27),
            ('о', 19), ('н', 18), ('т', 13), ('в', 12), ('л', 9), ('е', 5), ('р', 5), ('и', 4), ('лет', 2)]
        self.assertListEqual(expected_units_top, unit_vocab.most_common(24))

        expected_labels_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('керри', 8), ('жириновского', 4), ('пайдер', 3), ('жириновский', 3),
            ('лавруша', 2), ('позы', 2), ('становясь', 2), ('сам', 2), ('можно', 2), ('начинал', 2), ('ням', 2),
            ('глубже', 2), ('лет', 2)]
        self.assertListEqual(expected_labels_top, label_vocab.most_common(15))

    def test_cnn_cbow(self):
        config = build_config({
            'input_unit': InputUnit.CNN,
            'vect_model': VectModel.CBOW,
            'model_head': ModelHead.SOFTMAX,
            'bucket_cbow': False
        })
        unit_vocab, label_vocab = extract_vocab(self.data_dir, config)

        expected_units_top = [
            ('[BOW]', 364), ('[EOW]', 364), ('а', 162), ('о', 159), ('и', 136), ('е', 130), ('р', 112), ('т', 105),
            ('с', 101), ('н', 91), ('в', 81), ('л', 79), ('к', 73), ('м', 57), ('д', 56)]
        self.assertListEqual(expected_units_top, unit_vocab.most_common(15))

        expected_labels_top = [
            ('[BOS]', 51), ('[EOS]', 51), ('керри', 8), ('жириновского', 4), ('пайдер', 3), ('жириновский', 3),
            ('лавруша', 2), ('позы', 2), ('становясь', 2), ('сам', 2), ('можно', 2), ('начинал', 2), ('ням', 2),
            ('глубже', 2), ('лет', 2)]
        self.assertListEqual(expected_labels_top, label_vocab.most_common(15))


if __name__ == "__main__":
    tf.test.main()
