import tensorflow as tf
from tensorflow.keras.layers import InputSpec, Layer, Wrapper
from tensorflow.python.keras.utils import tf_utils
from tfmiss.text import normalize_unicode, lower_case, zero_digits
from tfmiss.preprocessing import cbow_context
from .input import BOS_MARK, EOS_MARK, UNK_MARK


@tf.keras.utils.register_keras_serializable(package='WordVec')
class MapFlat(Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self._supports_ragged_inputs = True

    def call(self, inputs, **kwargs):
        return tf.ragged.map_flat_values(self.layer, inputs)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + self.layer.compute_output_shape([None])[1:]


@tf.keras.utils.register_keras_serializable(package='WordVec')
class CbowContext(Wrapper):
    def __init__(self, layer, window, position, **kwargs):
        super().__init__(layer, **kwargs)
        self.input_spec = InputSpec(dtype='string', ndim=2)
        self._supports_ragged_inputs = True

        self.window = window
        self.position = position

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, tf.RaggedTensor):
            inputs = tf.RaggedTensor.from_tensor(inputs, ragged_rank=1)  # inputs.shape.rank - 1

        bos = tf.fill([inputs.nrows(), 1], BOS_MARK)
        eos = tf.fill([inputs.nrows(), 1], EOS_MARK)
        sources = tf.concat([bos, inputs, eos], axis=1)

        features = {}
        outputs, positions = cbow_context(sources, self.window, UNK_MARK)
        if self.position:
            positions = tf.ragged.map_flat_values(
                lambda flat: tf.where(
                    tf.greater(flat, 0),
                    flat - 1,
                    flat
                ) + self.window,
                positions)
            features['positions'] = positions

        features['units'] = outputs
        outputs = self.layer.call(features)
        outputs = sources.with_flat_values(outputs)
        outputs = outputs[:, 1:-1, :]

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + self.layer.compute_output_shape(input_shape)[1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'window': self.window,
            'position': self.position,
        })

        return config
