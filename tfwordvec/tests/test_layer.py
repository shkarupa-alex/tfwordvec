from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..layer import MapFlat, NormalizeUnits, CbowContext


@keras_parameterized.run_all_keras_modes
class MapFlatTest(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            MapFlat,
            kwargs={'layer': tf.keras.layers.Lambda(lambda x: tf.stack([x, x], axis=-1))},
            input_shape=(3, 10),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 10, 2)
        )


@keras_parameterized.run_all_keras_modes
class NormalizeUnitsTest(keras_parameterized.TestCase):
    def test_layer(self):
        # Fails with compute_output_signature
        # testing_utils.layer_test(
        #     NormalizeUnits,
        #     kwargs={'layer': tf.keras.layers.Lambda(tf.no_op), 'lower': True, 'zero': True},
        #     input_data=np.array([
        #         ['Abc', 'dEfg', 'hI'],
        #         ['1', '2345', '6789'],
        #     ]).astype('str'),
        #     expected_output_dtype='string',
        #     expected_output_shape=(None, 3)
        # )
        pass

    def test_output(self):
        inputs = tf.constant([
            ['Abc', 'dEfg', 'hI'],
            ['1', '2345', '6789'],
        ], dtype=tf.string)
        layer = NormalizeUnits(lower=True, zero=True)
        outputs = layer(inputs)
        self.assertListEqual([2, 3], outputs.shape.as_list())
        self.assertEqual(tf.string, outputs.dtype)

        outputs = self.evaluate(outputs)
        self.assertTupleEqual((2, 3), outputs.shape)

        self.assertListEqual([
            [b'abc', b'defg', b'hi'],
            [b'0', b'0000', b'0000'],
        ], outputs.tolist())


@keras_parameterized.run_all_keras_modes
class CbowContextTest(keras_parameterized.TestCase):
    def test_layer(self):
        # output_data = testing_utils.layer_test(
        #     CbowContext,
        #     kwargs={
        #         'layer': tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.strings.reduce_join(
        #             x['units'], separator='_', axis=-1), axis=-1), output_shape=(None, 1)),
        #         'window': 1,
        #         'position': True
        #     },
        #     input_data=np.array([
        #         ['abc', 'defg', 'hi'],
        #         ['1', '2345', '6789'],
        #     ]).astype('str'),
        #     expected_output_dtype='string',
        #     expected_output_shape=(None, None, 1)
        # )
        pass

    def test_ouptut(self):
        inputs = tf.constant([
            ['abc', 'defg', 'hi'],
            ['1', '2345', '6789'],
        ], dtype=tf.string)
        layer = CbowContext(
            layer=tf.keras.layers.Lambda(
                lambda x: tf.expand_dims(
                    tf.strings.reduce_join(x['units'], separator='_', axis=-1),
                    axis=-1)
            ),
            window=1,
            position=True)
        outputs = layer(inputs)
        self.assertListEqual([2, None, 1], outputs.shape.as_list())
        self.assertEqual(tf.string, outputs.dtype)

        outputs = self.evaluate(outputs.to_tensor(''))
        self.assertTupleEqual((2, 3, 1), outputs.shape)

        self.assertListEqual([
            [[b'[BOS]_defg'], [b'abc_hi'], [b'defg_[EOS]']],
            [[b'[BOS]_2345'], [b'1_6789'], [b'2345_[EOS]']],
        ], outputs.tolist())


if __name__ == "__main__":
    tf.test.main()
