import tensorflow as tf
from keras import layers, keras_parameterized, testing_utils
from ..layer import CbowContext


@keras_parameterized.run_all_keras_modes
class CbowContextTest(keras_parameterized.TestCase):
    # def test_layer(self):
    #     output_data = testing_utils.layer_test(
    #         CbowContext,
    #         kwargs={
    #             'layer': layers.Lambda(lambda x: tf.expand_dims(tf.strings.reduce_join(
    #                 x['units'], separator='_', axis=-1), axis=-1), output_shape=(None, 1)),
    #             'window': 1,
    #             'position': True
    #         },
    #         input_data=np.array([
    #             ['abc', 'defg', 'hi'],
    #             ['1', '2345', '6789'],
    #         ]).astype('str'),
    #         expected_output_dtype='string',
    #         expected_output_shape=(None, None, 1)
    #     )

    def test_ouptut(self):
        inputs = tf.constant([
            ['abc', 'defg', 'hi'],
            ['1', '2345', '6789'],
        ], dtype=tf.string)
        layer = CbowContext(
            layer=layers.Lambda(
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
